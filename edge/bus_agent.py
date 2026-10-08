"""
ROAD-SHIELD bus agent: what runs on the Raspberry Pi in each bus.

    python -m edge.bus_agent --bus BMTC-KA01-1234 --server https://<engine>        # real camera, MPU-6050, GPS
    python -m edge.bus_agent --bus DEMO-1 --server http://127.0.0.1:8001 \\
        --replay-frames datasets/02_kaggle_pothole_600 --replay-imu data/drive.csv --replay-gps data/route.csv

Every tick (default every 2 s, about what a Pi 4 can analyse):

    1. read a frame, the last second of accelerometer data (100 samples) and the GPS fix
    2. frame quality check (edge/frame_quality.py)
         usable    -> the vision pipeline on the frame. The camera sees the road about LOOKAHEAD_M ahead, so a
                      candidate defect waits until the bus has driven that far (distance / speed, 0.5-4 s); the
                      accelerometer second from THAT moment is fused with it (models/bayesian_fusion_gate.py), and
                      the position at that moment, when the bus is over the defect, is the one reported
                      felt or not contradicted -> "defect" event
                      seen but not felt (another lane, the kerb side, or a shadow) -> "cam" event: the server
                      keeps it apart and adds it to the ledger only when a second bus sees the same thing
         unusable  -> IMU only; a shock the classifier calls a pothole becomes an IMU-only sighting
    3. a confirmed defect becomes a compact event (well under 1 KB), sealed with AES-256-GCM and queued
    4. a position heartbeat is queued every --position-every seconds, for the live map
    5. a background thread sends the queue to the server whenever there is a connection

What never leaves the bus: the photograph. Events carry the class, measurements and location only. With
--evidence-dir, the frame behind each defect is kept on the device after people and number plates are
blurred (models/privacy_redactor.py), for a supervisor to inspect at the depot.

No GPS fix, no defect event: a defect without a location cannot be repaired and would only be guessed at
later. The tick is counted under "skipped_no_gps".
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request

from edge import crypto, frame_quality
from edge.store_forward import StoreAndForward

AREA_CLASS_IDS = {1, 2, 3}
LOOKAHEAD_M = 8.0            # where a dashcam's lower frame meets the road, roughly; calibrate per mount
LOOKAHEAD_MIN_S, LOOKAHEAD_MAX_S = 0.5, 4.0
# An IMU-only sighting needs the trained shock classifier to say pothole (>= 0.85: 18% of 1-second windows on the
# recorded pothole drives, 0% on plain road), or a jolt beyond anything plain road produced (models/
# bayesian_fusion_gate.SEVERE_JOLT_MS2). A raw 6 m/s^2 jolt, the first rule here, fired on 78-96% of plain road.
from models.bayesian_fusion_gate import SEVERE_JOLT_MS2
SHOCK_MIN_PROBABILITY = 0.85
SHOCK_MIN_DELTA_Z = SEVERE_JOLT_MS2


def _r(x, nd):
    return None if x is None else round(float(x), nd)


def defect_event(audit, fix, now):
    """Compact event for a confirmed area defect in a pipeline result, or None."""
    if not audit or not audit.get("is_distress"):
        return None
    d = audit.get("primary_distress") or {}
    if d.get("class_id") not in AREA_CLASS_IDS or not d.get("is_distress"):
        return None
    fusion = audit.get("bayesian_sensor_fusion") or {}
    pci = (audit.get("astm_d6433_pci") or {}).get("pci_score")
    area = d.get("surface_area_m2")
    if pci is None or area is None:
        return None
    felt = fusion.get("imu_evidence_source", "").startswith("real_sensor_window")
    kind = "cam" if fusion.get("verdict") == "REJECTED_OPTICAL_FALSE_ALARM" else "defect"
    return {
        "t": kind, "ts": int(now),
        "lat": _r(fix["lat"], 6), "lon": _r(fix["lon"], 6),
        "cls": d.get("class_name"), "pci": _r(pci, 1), "area": _r(area, 3), "depth": _r(d.get("depth_cm"), 1),
        "conf": _r(d.get("confidence"), 3),
        "fusion": fusion.get("verdict"), "imu": felt,
    }


def lookahead_s(speed_kmh):
    v = max(1.0, float(speed_kmh or 0.0) / 3.6)
    return min(LOOKAHEAD_MAX_S, max(LOOKAHEAD_MIN_S, LOOKAHEAD_M / v))


def shock_event(imu_report, fix, now, reasons):
    """IMU-only sighting when the camera could not be used, or None if the shock is not strong enough."""
    if not imu_report or not imu_report.get("available"):
        return None
    p = float(imu_report.get("pothole_shock_probability") or 0.0)
    dz = float(imu_report.get("peak_delta_z_ms2") or 0.0)
    if p < SHOCK_MIN_PROBABILITY and dz < SHOCK_MIN_DELTA_Z:
        return None
    return {"t": "shock", "ts": int(now), "lat": _r(fix["lat"], 6), "lon": _r(fix["lon"], 6),
            "p": _r(p, 3), "dz": _r(dz, 2), "cls": imu_report.get("shock_classification"),
            "why": ",".join(reasons)[:60]}


def position_event(fix, now, queued):
    return {"t": "pos", "ts": int(now), "lat": _r(fix["lat"], 6), "lon": _r(fix["lon"], 6),
            "spd": _r(fix.get("speed_kmh"), 1), "hdg": _r(fix.get("heading_deg"), 1), "q": int(queued)}


def http_sender(server, api_key=None, timeout=20):
    url = server.rstrip("/") + "/api/v1/fleet/ingest-sealed"

    def send(envelopes):
        req = urllib.request.Request(url, data=json.dumps({"packets": envelopes}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        if api_key:
            req.add_header("X-API-Key", api_key)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return int(body.get("accepted_in_order", 0))
    return send


class BusAgent:
    def __init__(self, bus_id, queue, camera, imu, gps, pipeline=None, corridor="UNSPECIFIED",
                 position_every=10.0, evidence_dir=None, device_id=None, quality_thresholds=None):
        self.bus_id = bus_id
        self.queue = queue
        self.camera, self.imu, self.gps = camera, imu, gps
        self.pipeline = pipeline
        self.corridor = corridor
        self.position_every = float(position_every)
        self.evidence_dir = evidence_dir
        self.device_id = device_id or bus_id
        self.quality_thresholds = quality_thresholds
        self._last_pos = float("-inf")
        self._pending = []           # camera candidates waiting for the bus to reach them
        self.counters = {"ticks": 0, "frames_usable": 0, "frames_unusable": 0, "defects": 0, "shocks": 0,
                         "camera_only": 0, "positions": 0, "skipped_no_gps": 0, "skipped_no_frame": 0, "errors": 0}

    def _pipeline(self):
        if self.pipeline is None:
            from pipeline.deep_inference_pipeline import DeepInferencePipeline
            self.pipeline = DeepInferencePipeline()
        return self.pipeline

    def _keep_evidence(self, frame, now):
        if not self.evidence_dir:
            return
        from PIL import Image
        from models.privacy_redactor import redact
        os.makedirs(self.evidence_dir, exist_ok=True)
        detector = getattr(self._pipeline(), "object_detector", None)
        red, _ = redact(frame, detector=detector)
        Image.fromarray(red).save(os.path.join(self.evidence_dir, f"{self.bus_id}_{int(now)}.jpg"), quality=85)

    def _emit(self, audit, fix, now, frame, out):
        ev = defect_event(audit, fix, now)
        if not ev:
            return
        self.queue.put(ev)
        if ev["t"] == "defect":
            self.counters["defects"] += 1
            self._keep_evidence(frame, now)
            out.append("defect")
        else:
            self.counters["camera_only"] += 1
            out.append("cam")

    def _resolve_pending(self, now, fix, out, flush=False):
        """Fuse each waiting camera candidate with the accelerometer second from when the bus reached it."""
        keep = []
        for c in self._pending:
            if not flush and now - c["t"] < c["wait"]:
                keep.append(c)
                continue
            audit = c["audit"]
            window = self.imu.window(100) if self.imu is not None else None
            pipe = self._pipeline()
            gate = getattr(pipe, "bayesian_gate", None)
            if window is not None and gate is not None:
                rep = pipe._run_imu_stage(window)[0]
                pv = ((audit.get("primary_distress") or {}).get("probabilities") or {}).get("Pothole Cavity", 0.05)
                fused = gate.fuse(p_visual=pv, p_imu_shock=rep.get("pothole_shock_probability", 0.05),
                                  delta_z_ms2=rep.get("peak_delta_z_ms2", 0.0))
                fused["imu_evidence_source"] = "real_sensor_window_at_defect"
                audit = dict(audit, bayesian_sensor_fusion=fused, imu_shock_telemetry=rep)
            self._emit(audit, fix or c["fix"], c["t"], c["frame"], out)
        self._pending = keep

    def tick(self, now=None):
        now = time.time() if now is None else now
        self.counters["ticks"] += 1
        fix = self.gps.fix(now) if self.gps is not None else None
        out = []
        if fix and now - self._last_pos >= self.position_every:
            self.queue.put(position_event(fix, now, self.queue.stats()["queued"]))
            self._last_pos = now
            self.counters["positions"] += 1
            out.append("pos")
        if self._pending:
            try:
                self._resolve_pending(now, fix, out)
            except crypto.PacketError:
                self.counters["errors"] += 1
        frame = self.camera.frame() if self.camera is not None else None
        if frame is None:
            self.counters["skipped_no_frame"] += 1
            return out
        if not fix:
            self.counters["skipped_no_gps"] += 1
            return out
        window = self.imu.window(100) if self.imu is not None else None
        q = frame_quality.assess(frame, self.quality_thresholds)
        try:
            if q["usable"]:
                self.counters["frames_usable"] += 1
                # vision only here: the wheel has not reached what the camera sees yet
                audit = self._pipeline().audit_image(frame, corridor_id=self.corridor, latitude=fix["lat"],
                                                     longitude=fix["lon"], imu_series=None,
                                                     device_id=self.device_id,
                                                     vehicle_speed_kmh=fix.get("speed_kmh") or 30.0)
                if defect_event(audit, fix, now):
                    if self.imu is None:
                        self._emit(audit, fix, now, frame, out)
                    else:
                        self._pending.append({"audit": audit, "fix": fix, "t": now, "frame": frame,
                                              "wait": lookahead_s(fix.get("speed_kmh"))})
            else:
                self.counters["frames_unusable"] += 1
                if window is not None:
                    imu_report = self._pipeline()._run_imu_stage(window)[0]
                    ev = shock_event(imu_report, fix, now, q["reasons"])
                    if ev:
                        self.queue.put(ev)
                        self.counters["shocks"] += 1
                        out.append("shock")
        except crypto.PacketError:
            self.counters["errors"] += 1
        return out

    def run(self, interval=2.0, duration=None, sender=None, sync_every=5.0):
        stop = threading.Event()
        if sender is not None:
            def sync():
                while not stop.wait(sync_every):
                    try:
                        self.queue.flush(sender)
                    except Exception:
                        pass
            threading.Thread(target=sync, daemon=True).start()
        t_end = None if duration is None else time.time() + duration
        try:
            while t_end is None or time.time() < t_end:
                t0 = time.time()
                try:
                    self.tick(t0)
                except Exception as e:
                    self.counters["errors"] += 1
                    print(f"[agent] tick failed: {e}", flush=True)
                if self.counters["ticks"] % 30 == 0:
                    print(f"[agent] {self.counters} queue {self.queue.stats()}", flush=True)
                time.sleep(max(0.0, interval - (time.time() - t0)))
        finally:
            stop.set()
            try:
                self._resolve_pending(time.time(), None, [], flush=True)
            except Exception:
                pass
            if sender is not None:
                self.queue.flush(sender)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", required=True)
    ap.add_argument("--server", default=None, help="engine URL; without it events only queue on the device")
    ap.add_argument("--api-key", default=os.environ.get("ROAD_SHIELD_API_KEY"))
    ap.add_argument("--queue", default="edge_queue.db")
    ap.add_argument("--corridor", default="UNSPECIFIED")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--position-every", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=None, help="seconds, for tests; default runs forever")
    ap.add_argument("--evidence-dir", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--gps-port", default="/dev/serial0")
    ap.add_argument("--replay-frames", default=None)
    ap.add_argument("--replay-imu", default=None)
    ap.add_argument("--replay-gps", default=None)
    a = ap.parse_args(argv)

    try:
        crypto.check_ids(a.bus, "0")
    except crypto.PacketError as e:
        raise SystemExit(f"--bus {a.bus!r}: {e}")
    key = crypto.load_key()
    if key is None:
        raise SystemExit("set ROAD_SHIELD_FLEET_KEY (the same value on the server). "
                         f"A new random key: {crypto.new_key_hex()}")
    from edge import sensors
    camera = sensors.ReplayFrames(a.replay_frames) if a.replay_frames else sensors.Camera(a.camera)
    imu = sensors.ReplayImu(a.replay_imu) if a.replay_imu else (None if a.replay_frames else sensors.Mpu6050())
    gps = sensors.ReplayGps(a.replay_gps) if a.replay_gps else (None if a.replay_frames else sensors.SerialGps(a.gps_port))
    queue = StoreAndForward(a.queue, a.bus, key)
    agent = BusAgent(a.bus, queue, camera, imu, gps, corridor=a.corridor, position_every=a.position_every,
                     evidence_dir=a.evidence_dir)
    sender = http_sender(a.server, a.api_key) if a.server else None
    print(f"[agent] bus {a.bus}: {'replay' if a.replay_frames else 'live sensors'}, "
          f"server {a.server or 'none (queue only)'}", flush=True)
    agent.run(interval=a.interval, duration=a.duration, sender=sender)
    print(f"[agent] stopped. {agent.counters} queue {queue.stats()}")


if __name__ == "__main__":
    main()
