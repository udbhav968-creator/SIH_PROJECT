"""
A phone as the dashcam: the bus agent's logic, fed by a phone's camera, GPS and accelerometer over HTTPS.

The page /drive (web/drive.html) runs on a phone fixed to the windscreen, rear camera facing the road. About
every two seconds it sends one 640-pixel JPEG, its latest GPS fix and the last few seconds of accelerometer
readings to POST /api/v1/fleet/phone-tick. Here each phone gets its own edge.bus_agent.BusAgent - the same
code a Raspberry Pi on a bus runs - with sources that hold what the phone sent:

    camera   the frame of this tick, read once
    gps      the phone's fix; refused when its accuracy is worse than MAX_GPS_ACCURACY_M, because a defect
             reported 80 m from where it is cannot be found by a repair crew
    imu      the phone's readings turned into the vehicle frame the shock model was trained on
             (lateral, longitudinal, vertical including gravity, 100 samples per second), kept for a few
             seconds so a defect the camera saw ahead is fused with the second in which the wheels reached it.
             The axes are set by a running average of the gravity reading (about 8 s), so braking or
             accelerating for a few seconds does not tilt them; turning the phone by more than 25 degrees
             restarts the average

What the agent emits is applied directly to the ledger and the live map (pipeline/edge_ingest.apply_direct):
the request is already authenticated by the API (X-API-Key when the server sets one), so there is no sealed
packet. Camera-only sightings are held until a second vehicle sees the same thing, as for buses. The
photograph is analysed and dropped: nothing is stored.

Phone axes (W3C DeviceMotion, accelerationIncludingGravity): x to the right of the screen, y to its top,
z out of the screen towards the driver; the rear camera looks along -z. Vertical is the direction of the
mean reading over the samples sent (gravity, about 9.8 m/s^2), forward is -z with its vertical part removed,
lateral is forward x up. The vertical axis does not depend on the sign convention a browser uses; the
longitudinal axis flips with it (some iOS versions report the readings negated), which the model tolerates
because each window's mean is removed before the band energies are taken.

Identity: a phone's id is chosen by the page (kept in the browser), and the request is authenticated by the
API key, not by the phone. Revoking PHONE-<id> stops that id; it does not stop a key holder from using a new
one, so the API key is what controls who may send. Phone positions are shown on the map but are not
counted as fleet passes for repair verification (pipeline/works.py), which stays with the buses.

Not tested on a real drive yet. Known limits: a phone in a holder vibrates differently from an IMU bolted to
a chassis, and phones sample at 50-200 Hz with jitter, which is resampled here by linear interpolation.
"""
import base64
import io
import re
import threading
import time
from collections import deque

import numpy as np

from edge import crypto
from edge.bus_agent import BusAgent

ID_RE = re.compile(r"[A-Za-z0-9_-]{4,24}")      # used with fullmatch
MAX_DEVICES = 50
IDLE_DROP_S = 1800
MIN_TICK_S = 0.8
MAX_GPS_ACCURACY_M = 30.0
MAX_JPEG_BYTES = 1_500_000
MAX_SAMPLES = 2000
HISTORY_S = 12.0
RATE = 100
GRAVITY_WEIGHT = 0.25        # per frame (about 2 s): the orientation follows the mount over ~8 s, not braking
REMOUNT_DEG = 25.0
MAX_GAP_S = 0.25


def clean_readings(samples):
    """[[t_ms, x, y, z], ...] -> (sorted finite array without repeated times, readings per second); ValueError if unusable."""
    a = np.asarray(samples, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError("imu must be a list of [t_ms, x, y, z]")
    a = a[np.all(np.isfinite(a), axis=1)]
    a = a[np.argsort(a[:, 0], kind="stable")]
    if len(a):
        a = a[np.concatenate([[True], np.diff(a[:, 0]) > 0])]
    if len(a) < 20:
        raise ValueError("need at least 20 accelerometer readings")
    span_s = (a[-1, 0] - a[0, 0]) / 1000.0
    rate = (len(a) - 1) / span_s if span_s > 0 else 0.0
    if rate < 20:
        raise ValueError(f"accelerometer at {rate:.0f} readings/s; at least 20 are needed")
    return a, rate


def to_vehicle_frame(samples, gravity=None):
    """
    samples: [[t_ms, x, y, z], ...] accelerationIncludingGravity in the phone's frame.
    gravity: the phone-frame gravity reading to orient by (a running average over earlier readings, so a few
    seconds of braking do not tilt the axes); defaults to the mean of these readings.
    Returns (t_ms array, (N, 3) array of [lateral, longitudinal, vertical], info); ValueError if unusable.
    info["mean_reading"] is this batch's mean, for the caller's running average.
    """
    a, rate = clean_readings(samples)
    batch = a[:, 1:].mean(axis=0)
    g = np.asarray(gravity, dtype=np.float64) if gravity is not None else batch
    gn = float(np.linalg.norm(g))
    if not 7.0 <= gn <= 12.5:
        raise ValueError(f"the readings do not contain gravity (mean {gn:.1f} m/s^2): send accelerationIncludingGravity")
    up = g / gn
    back = np.array([0.0, 0.0, 1.0])
    fwd = -back - np.dot(-back, up) * up
    if np.linalg.norm(fwd) < 0.25:
        # lying nearly flat: the camera looks at the sky or the floor; use the top of the phone as forward
        top = np.array([0.0, 1.0, 0.0])
        fwd = top - np.dot(top, up) * up
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    xyz = a[:, 1:]
    out = np.stack([xyz @ right, xyz @ fwd, xyz @ up], axis=1)
    pitch = float(np.degrees(np.arcsin(np.clip(np.dot(-back, up), -1.0, 1.0))))
    info = {"readings": int(len(a)), "rate_hz": round(rate, 1), "gravity_ms2": round(gn, 2),
            "camera_pitch_deg": round(pitch, 1),       # 0 = level with the road, negative = looking down
            "mean_reading": [round(float(x), 4) for x in batch]}
    return a[:, 0], out, info


def running_gravity(previous, batch_mean, weight=GRAVITY_WEIGHT):
    """Running average of the phone-frame gravity reading; restarts when the phone was turned or re-mounted."""
    b = np.asarray(batch_mean, dtype=np.float64)
    if previous is None:
        return b
    p = np.asarray(previous, dtype=np.float64)
    cosang = float(np.dot(p, b) / (np.linalg.norm(p) * np.linalg.norm(b) + 1e-9))
    if cosang < np.cos(np.radians(REMOUNT_DEG)):
        return b
    return (1 - weight) * p + weight * b


class PhoneImu:
    """Vehicle-frame readings on the server's clock, resampled to 100 Hz on request."""

    def __init__(self):
        self.t = deque()
        self.v = deque()

    def add(self, t_s, rows):
        last = self.t[-1] if self.t else -np.inf
        for ti, r in zip(t_s, rows):
            if ti > last:
                self.t.append(float(ti))
                self.v.append(r)
                last = ti
        cut = last - HISTORY_S
        while self.t and self.t[0] < cut:
            self.t.popleft()
            self.v.popleft()

    def covers(self, t_center):
        """True once the readings reach half a second past t_center (the window around it is complete)."""
        return bool(self.t) and self.t[-1] >= float(t_center) + 0.5 - 0.02

    def window_at(self, t_center, n=100):
        if len(self.t) < 20:
            return None
        t = np.fromiter(self.t, float)
        end = float(t_center) + 0.5
        if end > t[-1] + 0.02:
            return None          # the readings do not reach that moment yet: no window, never an earlier second
        start = end - n / RATE
        if start < t[0] - 0.05:
            return None
        inside = (t >= start - 0.05) & (t <= end + 0.05)
        ti = t[inside]
        if len(ti) < 20 or np.max(np.diff(np.concatenate([[start], ti, [end]]))) > MAX_GAP_S:
            return None          # readings missing for part of that second: no window rather than a made-up one
        v = np.asarray(self.v)
        q = np.linspace(start, end, n)
        return np.stack([np.interp(q, t, v[:, k]) for k in range(3)], axis=1).astype(np.float32)

    def window(self, n=100):
        return self.window_at(self.t[-1] - 0.5 - 1e-9, n) if self.t else None


class _Frame:
    def __init__(self):
        self.img = None

    def frame(self):
        img, self.img = self.img, None
        return img


class _Gps:
    def __init__(self):
        self.last = None

    def fix(self, now=None):
        return self.last


class _Direct:
    """The agent's queue: events go straight to the ledger, and their results are kept for the reply."""

    def __init__(self, ingest, bus):
        self.ingest, self.bus = ingest, bus
        self.results = []

    def put(self, event):
        try:
            self.results.append(self.ingest.apply_direct(self.bus, event))
        except crypto.PacketError as e:
            self.results.append({"status": "rejected", "error": str(e)})

    def stats(self):
        return {"queued": 0}


class _Recorder:
    """Passes every call to the served pipeline and keeps the last audit, so the phone sees what was found."""

    def __init__(self, pipe):
        self._p = pipe
        self.last = None

    def audit_image(self, *a, **k):
        self.last = self._p.audit_image(*a, **k)
        return self.last

    def __getattr__(self, name):
        return getattr(self._p, name)


def _haversine_m(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * 6371000 * np.arcsin(np.sqrt(h)))


def _decode_jpeg(b64):
    from PIL import Image
    if not isinstance(b64, str) or not b64:
        return None
    if "," in b64[:80]:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64, validate=False)
    if len(raw) > MAX_JPEG_BYTES:
        raise ValueError(f"frame is {len(raw)} bytes; send at most {MAX_JPEG_BYTES} (640 px JPEG)")
    img = Image.open(io.BytesIO(raw))
    if img.width * img.height > 4_000_000:
        raise ValueError("frame larger than 4 megapixels; send a 640 px frame")
    img = img.convert("RGB")
    if max(img.size) > 960:
        img.thumbnail((960, 960))
    return np.asarray(img)


def summarise(audit, quality):
    """What the phone shows: the class seen, whether it was a defect, PCI, the input guard, vehicles."""
    if quality is not None and not quality.get("usable", True):
        return {"frame": "unusable", "reasons": quality.get("reasons", [])}
    if not audit:
        return {"frame": "not analysed"}
    d = audit.get("primary_distress") or {}
    guard = audit.get("input_check") or {}
    out = {"frame": "analysed", "class": d.get("class_name"), "confidence": d.get("confidence"),
           "is_distress": bool(audit.get("is_distress")),
           "pci": (audit.get("astm_d6433_pci") or {}).get("pci_score"),
           "area_m2": d.get("surface_area_m2"), "guard": guard.get("verdict")}
    try:
        from pipeline.traffic import counts_from_audit
        out["vehicles"] = counts_from_audit(audit)
    except Exception:
        pass
    return out


class BusyError(RuntimeError):
    """The phone's previous frame is still being analysed (HTTP 409)."""


class PhoneFleet:
    def __init__(self, ingest, pipeline_getter, corridor="PHONE"):
        self.ingest = ingest
        self.pipeline_getter = pipeline_getter
        self.corridor = corridor
        self._devices = {}
        self._lock = threading.Lock()

    @staticmethod
    def bus_id(device_id):
        if not ID_RE.fullmatch(str(device_id or "")):
            raise ValueError("device_id must be 4-24 letters, digits, '_' or '-'")
        return "PHONE-" + device_id

    def _device(self, device_id, now):
        bus = self.bus_id(device_id)
        with self._lock:
            for k in [k for k, d in self._devices.items() if now - d["seen"] > IDLE_DROP_S]:
                del self._devices[k]
            d = self._devices.get(bus)
            if d is None:
                if len(self._devices) >= MAX_DEVICES:
                    raise OverflowError(f"{MAX_DEVICES} phones are already driving; try again later")
                q = _Direct(self.ingest, bus)
                agent = BusAgent(bus, q, _Frame(), PhoneImu(), _Gps(), pipeline=None, corridor=self.corridor,
                                 position_every=5.0, device_id=bus)
                d = {"bus": bus, "agent": agent, "queue": q, "lock": threading.Lock(), "seen": now, "last_tick": -1e9,
                     "started": now, "imu": None, "g": None, "offsets": deque(maxlen=30)}
                self._devices[bus] = d
            d["seen"] = now
            return d

    def tick(self, body, now=None):
        """One request from the phone. Returns the reply dict; raises ValueError (400) or OverflowError (503)."""
        now = time.time() if now is None else now
        d = self._device(body.get("device_id"), now)
        if not d["lock"].acquire(timeout=10):
            raise BusyError("the previous frame from this phone is still being analysed")
        try:
            agent, q = d["agent"], d["queue"]
            if self.ingest.is_revoked(d["bus"]):
                raise PermissionError(f"{d['bus']} has been revoked by an operator")
            if body.get("stop"):
                q.results = []
                agent.pipeline = _Recorder(self.pipeline_getter())
                agent._resolve_pending(now, agent.gps.last, [], flush=True)
                return {"bus_id": d["bus"], "stopped": True, "applied": q.results, "counters": agent.counters}
            if now - d["last_tick"] < MIN_TICK_S:
                raise ValueError("too fast: send at most one frame per second")
            d["last_tick"] = now
            # clock: the phone's timestamps mapped onto the server's. receive time - send time is the clock
            # offset plus the upload time, so the smallest value over recent frames is the best estimate; a
            # per-frame value would move the readings by the upload time's jitter
            try:
                sent_ms = float(body.get("sent_ms") or 0)
            except (TypeError, ValueError):
                sent_ms = 0.0
            offset = None
            if sent_ms > 0 and np.isfinite(sent_ms):
                d["offsets"].append(now - sent_ms / 1000.0)
                offset = min(d["offsets"])
            imu_info = None
            samples = body.get("imu")
            if isinstance(samples, list) and samples and offset is not None:
                if len(samples) > MAX_SAMPLES:
                    raise ValueError(f"at most {MAX_SAMPLES} accelerometer readings per frame")
                try:
                    a, _ = clean_readings(samples)
                    g = running_gravity(d.get("g"), a[:, 1:].mean(axis=0))
                    t_ms, rows, imu_info = to_vehicle_frame(samples, gravity=g)
                    d["g"] = g                  # only after the readings proved usable
                    imu_info.pop("mean_reading", None)
                    agent.imu.add(t_ms / 1000.0 + offset, rows)
                except ValueError as e:
                    imu_info = {"error": str(e)}   # the frame is still analysed, camera only
                d["imu"] = imu_info
            gps = body.get("gps") or {}
            fix, gps_note = self._fix(gps, agent.gps.last, now)
            agent.gps.last = fix
            try:
                agent.camera.img = _decode_jpeg(body.get("frame_base64"))
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"frame is not a readable JPEG: {e}")
            q.results = []
            rec = _Recorder(self.pipeline_getter())
            agent.pipeline = rec
            events = agent.tick(now)
            return {"bus_id": d["bus"], "events": events, "seen": summarise(rec.last, agent.last_quality),
                    "applied": q.results, "gps": gps_note, "imu": imu_info,
                    "waiting_for_wheels": len(agent._pending), "counters": agent.counters}
        finally:
            d["lock"].release()

    @staticmethod
    def _fix(gps, previous, now):
        try:
            lat, lon = float(gps["lat"]), float(gps["lon"])
            acc = float(gps.get("accuracy_m") if gps.get("accuracy_m") is not None else 999)
        except (KeyError, TypeError, ValueError):
            # the page leaves the fix out when it is too old for a moving vehicle; an older one is not reused
            return None, "no GPS fix in this frame"
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or not np.isfinite(acc):
            return None, "GPS fix out of range"
        if acc > MAX_GPS_ACCURACY_M:
            return None, f"GPS accuracy {acc:.0f} m; defects are reported only within {MAX_GPS_ACCURACY_M:.0f} m"
        spd = gps.get("speed_mps")
        speed = float(spd) * 3.6 if isinstance(spd, (int, float)) and np.isfinite(spd) and spd >= 0 else None
        if speed is None and previous and now - previous["at"] > 0.5:
            speed = _haversine_m((previous["lat"], previous["lon"]), (lat, lon)) / (now - previous["at"]) * 3.6
            speed = speed if speed < 150 else None
        hdg = gps.get("heading_deg")
        hdg = float(hdg) if isinstance(hdg, (int, float)) and np.isfinite(hdg) else None
        return {"lat": lat, "lon": lon, "speed_kmh": speed, "heading_deg": hdg, "accuracy_m": acc, "at": now}, "ok"

    def devices(self, now=None):
        now = time.time() if now is None else now
        with self._lock:
            return [{"bus_id": d["bus"], "last_seen_s_ago": round(now - d["seen"], 1),
                     "driving_for_s": round(d["seen"] - d["started"], 1), "imu": d["imu"],
                     "counters": dict(d["agent"].counters)} for d in self._devices.values()]
