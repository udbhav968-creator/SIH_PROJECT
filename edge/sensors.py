"""
Sensor sources for the bus agent: the real hardware, and replay from recorded files so the whole agent can
be run and tested without a bus.

Hardware (Raspberry Pi):
    camera   any V4L2 / USB dashcam OpenCV can open (cv2.VideoCapture)
    IMU      MPU-6050 on I2C bus 1, address 0x68, read at 100 Hz (smbus2)
    GPS      Neo-6M on a serial port, NMEA 0183 at 9600 baud (pyserial); $GPRMC/$GNRMC and $GPGGA/$GNGGA

Replay:
    frames   a folder of JPEG/PNG photographs, one per tick
    imu      CSV with ax, ay, az columns in m/s^2 at 100 Hz (any case, other columns ignored), e.g. the drive
             logs in datasets/04_mobile_imu_telemetry_100hz/raw_logs
    gps      CSV with columns t,lat,lon[,speed_kmh]; positions between rows are interpolated
"""
import csv
import glob
import math
import os
import threading
import time
from collections import deque

import numpy as np

G = 9.80665


# --------------------------------------------------------------------------- NMEA
def _nmea_checksum_ok(sentence):
    if "*" not in sentence:
        return False
    body, _, given = sentence.strip().lstrip("$").partition("*")
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(given[:2], 16)
    except ValueError:
        return False


def _nmea_coord(value, hemi, deg_digits):
    if not value:
        return None
    deg = float(value[:deg_digits])
    minutes = float(value[deg_digits:])
    out = deg + minutes / 60.0
    return -out if hemi in ("S", "W") else out


def parse_nmea(sentence):
    """A fix dict from an RMC or GGA sentence, or None (bad checksum, no fix, other sentence)."""
    s = sentence.strip()
    if not s.startswith("$") or not _nmea_checksum_ok(s):
        return None
    fields = s[1:].split("*")[0].split(",")
    kind = fields[0][-3:]
    try:
        if kind == "RMC":
            if fields[2] != "A":
                return None
            lat = _nmea_coord(fields[3], fields[4], 2)
            lon = _nmea_coord(fields[5], fields[6], 3)
            speed = float(fields[7]) * 1.852 if fields[7] else None
            heading = float(fields[8]) if fields[8] else None
            return {"lat": lat, "lon": lon, "speed_kmh": speed, "heading_deg": heading, "source": "RMC"}
        if kind == "GGA":
            if not fields[6] or fields[6] == "0":
                return None
            lat = _nmea_coord(fields[2], fields[3], 2)
            lon = _nmea_coord(fields[4], fields[5], 3)
            return {"lat": lat, "lon": lon, "satellites": int(fields[7] or 0),
                    "hdop": float(fields[8]) if fields[8] else None, "source": "GGA"}
    except (ValueError, IndexError):
        return None
    return None


class SerialGps:
    def __init__(self, port="/dev/serial0", baud=9600):
        import serial
        self._ser = serial.Serial(port, baud, timeout=1)
        self._fix = None
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            line = self._ser.readline().decode("ascii", "replace")
            fix = parse_nmea(line)
            if fix:
                with self._lock:
                    merged = dict(self._fix or {})
                    merged.update({k: v for k, v in fix.items() if v is not None})
                    merged["t"] = time.time()
                    self._fix = merged

    def fix(self, now=None):
        with self._lock:
            if self._fix and time.time() - self._fix["t"] < 5.0:
                return dict(self._fix)
        return None


# --------------------------------------------------------------------------- IMU
class Mpu6050:
    """MPU-6050 accelerometer at 100 Hz into a ring buffer; window() returns the last second."""
    ADDR = 0x68
    PWR_MGMT_1, ACCEL_CONFIG, ACCEL_XOUT_H = 0x6B, 0x1C, 0x3B
    LSB_PER_G = 16384.0

    def __init__(self, bus=1, rate_hz=100):
        import smbus2
        self._bus = smbus2.SMBus(bus)
        self._bus.write_byte_data(self.ADDR, self.PWR_MGMT_1, 0)
        self._bus.write_byte_data(self.ADDR, self.ACCEL_CONFIG, 0)
        self.rate_hz = rate_hz
        self._buf = deque(maxlen=rate_hz * 2)
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _read(self):
        raw = self._bus.read_i2c_block_data(self.ADDR, self.ACCEL_XOUT_H, 6)
        vals = []
        for i in range(0, 6, 2):
            v = (raw[i] << 8) | raw[i + 1]
            vals.append((v - 65536 if v > 32767 else v) / self.LSB_PER_G * G)
        return vals

    def _loop(self):
        period = 1.0 / self.rate_hz
        nxt = time.perf_counter()
        while True:
            try:
                sample = self._read()
                with self._lock:
                    self._buf.append(sample)
            except OSError:
                pass
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def window(self, n=100):
        with self._lock:
            if len(self._buf) < n:
                return None
            return np.asarray(list(self._buf)[-n:], dtype=np.float32)


class ReplayImu:
    """Plays back a recorded accelerometer CSV (t, ax, ay, az) against the agent's clock."""

    def __init__(self, path, loop=True, rate_hz=100):
        rows = []
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                low = {k.strip().lower(): v for k, v in r.items() if k}
                try:
                    rows.append([float(low["ax"]), float(low["ay"]), float(low["az"])])
                except (KeyError, ValueError):
                    continue
        if len(rows) < 100:
            raise ValueError(f"{path}: need at least 100 samples")
        self.data = np.asarray(rows, dtype=np.float32)
        self.rate_hz = rate_hz
        self.loop = loop
        self.t0 = None

    def window(self, n=100, now=None):
        now = time.time() if now is None else now
        if self.t0 is None:
            self.t0 = now
        end = int((now - self.t0) * self.rate_hz) + n
        if self.loop:
            end = n + (end - n) % (len(self.data) - n)
        elif end > len(self.data):
            return None
        return self.data[end - n:end]


class ReplayGps:
    """Positions from a CSV route (t, lat, lon[, speed_kmh]), interpolated against the agent's clock."""

    def __init__(self, path, loop=True):
        self.rows = []
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                self.rows.append((float(r["t"]), float(r["lat"]), float(r["lon"]),
                                  float(r["speed_kmh"]) if r.get("speed_kmh") else None))
        if len(self.rows) < 2:
            raise ValueError(f"{path}: need at least two positions")
        self.rows.sort()
        self.loop = loop
        self.t0 = None

    def fix(self, now=None):
        now = time.time() if now is None else now
        if self.t0 is None:
            self.t0 = now
        span = self.rows[-1][0] - self.rows[0][0]
        t = self.rows[0][0] + (now - self.t0)
        if t > self.rows[-1][0]:
            if not self.loop or span <= 0:
                return None
            t = self.rows[0][0] + (t - self.rows[0][0]) % span
        for (ta, la, lo, sa), (tb, lb, lob, sb) in zip(self.rows, self.rows[1:]):
            if ta <= t <= tb:
                k = 0.0 if tb == ta else (t - ta) / (tb - ta)
                lat, lon = la + k * (lb - la), lo + k * (lob - lo)
                heading = math.degrees(math.atan2(lob - lo, lb - la)) % 360.0
                speed = sa if sa is not None else None
                return {"lat": lat, "lon": lon, "speed_kmh": speed, "heading_deg": round(heading, 1),
                        "source": "replay"}
        return None


# --------------------------------------------------------------------------- camera
class Camera:
    def __init__(self, index=0, width=1280, height=720):
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {index} did not open")

    def frame(self):
        ok, bgr = self._cap.read()
        if not ok:
            return None
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)


class ReplayFrames:
    def __init__(self, folder, loop=True):
        self.files = sorted(f for ext in ("*.jpg", "*.jpeg", "*.png")
                            for f in glob.glob(os.path.join(folder, "**", ext), recursive=True))
        if not self.files:
            raise ValueError(f"no images under {folder}")
        self.loop = loop
        self.i = 0

    def frame(self):
        from PIL import Image
        if self.i >= len(self.files):
            if not self.loop:
                return None
            self.i = 0
        path = self.files[self.i]
        self.i += 1
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
