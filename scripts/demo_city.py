"""
A whole city loop in one command, for a demonstration: several buses replay road photographs and recorded
accelerometer data along a road, report sealed packets to the engine, and one repair is taken from order
to fleet verification while you watch the Road map and Works pages.

    # window 1: the engine, with settings that let a repair be verified in minutes instead of a day
    $env:ROAD_SHIELD_FLEET_KEY = "demo-fleet-key"; $env:ROAD_SHIELD_VERIFY_MIN_HOURS = "0"; $env:ROAD_SHIELD_VERIFY_PASSES = "2"
    python -m api.server 8001

    # window 2: three buses for five minutes, and the repair story
    $env:ROAD_SHIELD_FLEET_KEY = "demo-fleet-key"
    python -m scripts.demo_city --server http://127.0.0.1:8001 --buses 3 --minutes 8 --repair-demo

Then open http://127.0.0.1:8001/corridor and http://127.0.0.1:8001/works.

What is real and what is replayed: the photographs are the project's own road photographs (a mix of
damaged and sound road), the
accelerometer data is a real recorded drive (datasets/04_mobile_imu_telemetry_100hz), and every model,
packet, ledger entry and order is produced by the same code a real bus would use. What is not real: the
buses, and the pairing of a photograph with a place - the route is a demonstration route along MG Road,
Bengaluru, and the photographs were not taken there. Say so when you show it.

--repair-demo, after the first defects arrive: issues a work order for the highest-priority defect that has
a depth on record, assigns it to "Demo Roads Pvt Ltd", starts and finishes the repair, and then leaves it to
the buses: their passes verify it, or a new sighting of the same defect reopens it. With three buses on the
2 km demo route, two passes over one spot take about 3 to 6 minutes after the repair.
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

ROUTE = os.path.join(ENGINE_ROOT, "edge", "samples", "demo_route.csv")
# a mix of damaged and sound road: RDD2022 India, pothole photographs, and clean "hard negatives" (markings,
# patches, shadows that look like damage), so most frames are not potholes - as on a real road
PHOTOS = [os.path.join(ENGINE_ROOT, "datasets", d) for d in
          ("01_rdd2022_india", "02_kaggle_pothole_600", "05_morth_civil_hard_negatives")]
IMU_LOGS = [os.path.join(ENGINE_ROOT, "datasets", "04_mobile_imu_telemetry_100hz", "raw_logs", f)
            for f in ("potholes_1.csv", "potholes_2.csv", "plain_road_1.csv", "unmarked_sb_1.csv")]


class LockedPipeline:
    """One copy of the models shared by every replayed bus (the pipeline is not thread-safe)."""

    def __init__(self):
        from pipeline.deep_inference_pipeline import DeepInferencePipeline
        self._p = DeepInferencePipeline()
        self._lock = threading.Lock()
        self.object_detector = getattr(self._p, "object_detector", None)
        self.bayesian_gate = self._p.bayesian_gate          # pure arithmetic, safe to share

    def audit_image(self, *a, **kw):
        with self._lock:
            return self._p.audit_image(*a, **kw)

    def _run_imu_stage(self, window):
        with self._lock:
            return self._p._run_imu_stage(window)


def call(server, method, path, body=None, api_key=None):
    req = urllib.request.Request(server.rstrip("/") + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("X-API-Key", api_key)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def repair_story(server, api_key, stop, log):
    """Order -> assign -> in progress -> repaired for the top defect; the buses do the rest."""
    oid = None
    while not stop.is_set() and oid is None:
        stop.wait(20)
        code, r = call(server, "GET", "/api/v1/priority/ranking")
        if code != 200:
            continue
        for d in r.get("defects", []):
            if d.get("depth_cm") is None or d.get("repair"):
                continue
            code, o = call(server, "POST", "/api/v1/works/orders", {"defect_id": d["defect_id"], "actor": "demo",
                                                                    "note": "demo: highest-priority defect"}, api_key)
            if code == 200:
                oid = o["work_order_id"]
                log(f"order {oid} issued for {d['defect_id']} ({d['defect_class']}, {d['priority']['band']})")
            elif code == 401:
                log("the engine wants an operator key: pass --api-key")
                return
            break
    for status, extra, pause in (("ASSIGNED", {"contractor": "Demo Roads Pvt Ltd"}, 10), ("IN_PROGRESS", {}, 15),
                                 ("REPAIRED", {"note": "demo: patched with DBM"}, 20)):
        if oid is None or stop.wait(pause):
            return
        code, o = call(server, "POST", "/api/v1/works/status", {"work_order_id": oid, "status": status,
                                                                "actor": "demo", **extra}, api_key)
        log(f"order {oid} -> {status}" + ("" if code == 200 else f" refused: {o.get('error')}"))
    while not stop.wait(15):
        code, o = call(server, "GET", f"/api/v1/works/order?id={oid}")
        if code == 200:
            log(f"order {oid}: {o['status']}, fleet passes {o['fleet_passes_since_repair']}/{o['passes_needed']}")
            if o["status"] in ("VERIFIED", "REOPENED"):
                log("verified by the fleet" if o["status"] == "VERIFIED"
                    else "a bus saw the defect again, so the order reopened")
                return


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8001")
    ap.add_argument("--buses", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--api-key", default=os.environ.get("ROAD_SHIELD_API_KEY"))
    ap.add_argument("--repair-demo", action="store_true")
    a = ap.parse_args(argv)

    from edge import crypto, sensors
    from edge.bus_agent import BusAgent, http_sender
    from edge.store_forward import StoreAndForward
    key = crypto.load_key()
    if key is None:
        raise SystemExit("set ROAD_SHIELD_FLEET_KEY to the same value as the engine")
    code, _ = call(a.server, "GET", "/api/v1/health")
    if code != 200:
        raise SystemExit(f"no engine at {a.server}: start it first (python -m api.server 8001)")

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)

    log("loading the models once for every bus...")
    pipe = LockedPipeline()
    tmp = tempfile.mkdtemp(prefix="road_shield_demo_")
    route_len = sensors.ReplayGps(ROUTE).rows[-1][0]
    agents, threads = [], []
    for i in range(a.buses):
        bus = f"DEMO-BMTC-{201 + i}"
        # spread the buses around the loop: a westbound bus's offset counts from the other end of the road
        frac = (i / max(1, a.buses) + (0.5 if i % 2 else 0.0)) % 1.0
        gps = sensors.ReplayGps(ROUTE, offset_s=route_len * frac, reverse=bool(i % 2))
        cam = sensors.ReplayFrames(PHOTOS, start=i * 211, seed=7)
        imu = sensors.ReplayImu(IMU_LOGS[i % len(IMU_LOGS)])
        q = StoreAndForward(os.path.join(tmp, f"{bus}.db"), bus, key)
        agent = BusAgent(bus, q, cam, imu, gps, pipeline=pipe, corridor="MG Road (demo)", position_every=6)
        agents.append(agent)
        t = threading.Thread(target=agent.run, kwargs=dict(interval=a.interval, duration=a.minutes * 60,
                                                          sender=http_sender(a.server), sync_every=3), daemon=True)
        threads.append(t)
        log(f"{bus} {'eastbound' if i % 2 == 0 else 'westbound'} on the demo route")
    for t in threads:
        t.start()
    stop = threading.Event()
    story = None
    if a.repair_demo:
        story = threading.Thread(target=repair_story, args=(a.server, a.api_key, stop, log), daemon=True)
        story.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(30)
            total = {k: sum(ag.counters[k] for ag in agents) for k in ("defects", "shocks", "positions", "errors")}
            log(f"buses so far: {total}")
    except KeyboardInterrupt:
        log("stopping")
    stop.set()
    if story:
        story.join(timeout=5)
    code, s = call(a.server, "GET", "/api/v1/works/orders")
    if code == 200:
        log(f"works: {s['summary']['by_status']}")
    log(f"done. Packets queued on the buses are in {tmp}")


if __name__ == "__main__":
    main()
