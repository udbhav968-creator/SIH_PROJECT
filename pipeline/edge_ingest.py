"""Edge device telemetry and survey packet ingestion."""
import math
import sqlite3
import threading
import time

from edge import crypto

SCHEMA = """
CREATE TABLE IF NOT EXISTS edge_streams (
    bus_id     TEXT    NOT NULL,
    epoch      TEXT    NOT NULL,
    last_seq   INTEGER NOT NULL,
    last_seen  REAL    NOT NULL,
    packets    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (bus_id, epoch)
);
"""


def _num(v, lo=None, hi=None, nd=6):
    """float within [lo, hi], rounded; None when absent or not a usable number."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or (lo is not None and f < lo) or (hi is not None and f > hi):
        return None
    return round(f, nd)


def _text(v, n=80):
    return None if v is None else str(v)[:n]
POSITION_TTL_S = 600
SHOCKS_KEPT = 500
CAMERA_ONLY_KEPT = 2000
CAMERA_MATCH_M = 8.0
CAMERA_MATCH_S = 7 * 24 * 3600


class EdgeIngest:
    def __init__(self, db_path, dedup_engine, events=None, key=None):
        self.key = crypto.load_key(key) if key is not None else crypto.load_key()
        self.dedup = dedup_engine
        self.events = events
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._db.executescript(SCHEMA)
        self.positions = {}
        self.shocks = []
        self.camera_only = []
        self.position_listeners = []
        self.event_listeners = []

    @property
    def configured(self):
        return self.key is not None

    def _last_seq(self, bus, epoch):
        row = self._db.execute("SELECT last_seq FROM edge_streams WHERE bus_id=? AND epoch=?",
                               (bus, epoch)).fetchone()
        return row[0] if row else 0

    def _accept_seq(self, bus, epoch, seq):
        self._db.execute("""INSERT INTO edge_streams(bus_id, epoch, last_seq, last_seen, packets) VALUES(?,?,?,?,1)
                            ON CONFLICT(bus_id, epoch) DO UPDATE SET last_seq=excluded.last_seq,
                            last_seen=excluded.last_seen, packets=packets+1""", (bus, epoch, seq, time.time()))

    def _apply(self, bus, event):
        out = self._apply_event(bus, event)
        ts = _num(event.get("ts"), 0, None, 0)
        at = min(ts, time.time()) if ts else time.time()
        for fn in list(self.event_listeners):
            try:
                fn(bus, event.get("t"), event, at)
            except Exception as e:
                print(f"[edge] event listener failed: {e}")
        return out

    def _apply_event(self, bus, event):
        kind = event.get("t")
        lat, lon = _num(event.get("lat"), -90, 90), _num(event.get("lon"), -180, 180)
        if lat is None or lon is None:
            raise crypto.PacketError("event has no usable location")
        ts = _num(event.get("ts"), 0, None, 0)
        if kind == "defect":
            cls = _text(event.get("cls"))
            if not cls:
                raise crypto.PacketError("defect has no class")
            res = self.dedup.ingest_fleet_detection(bus, lat, lon, cls, event.get("pci"), event.get("area"),
                                                    image_timestamp=ts, depth_cm=_num(event.get("depth"), 0, 200),
                                                    enrich_location=False)
            if self.events:
                rec = next((d for d in self.dedup.get_all_deduplicated_defects()
                            if d["defect_id"] == res["defect_id"]), None)
                self.events.publish("defect", {"bus_id": bus, "result": res, "defect": rec, "via": "edge"})
            return {"type": "defect", **res}
        if kind == "cam":
            return self._camera_only(bus, lat, lon, ts, event)
        if kind == "shock":
            s = {"bus_id": bus, "lat": lat, "lon": lon, "ts": ts, "p": _num(event.get("p"), 0, 1, 3),
                 "dz": _num(event.get("dz"), 0, 1000, 2), "class": _text(event.get("cls")),
                 "camera_unusable_because": _text(event.get("why"), 60)}
            self.shocks.append(s)
            del self.shocks[:-SHOCKS_KEPT]
            if self.events:
                self.events.publish("shock", s)
            return {"type": "shock"}
        if kind == "pos":
            q = _num(event.get("q"), 0, None, 0)
            p = {"bus_id": bus, "lat": lat, "lon": lon, "speed_kmh": _num(event.get("spd"), 0, 300, 1),
                 "heading_deg": _num(event.get("hdg"), 0, 360, 1), "ts": ts, "received_unix": time.time(),
                 "queued_on_bus": int(q) if q is not None else None}
            self.positions[bus] = p
            if self.events:
                self.events.publish("bus_position", p)

            at = min(ts, time.time()) if ts else time.time()
            for fn in list(self.position_listeners):
                try:
                    fn(bus, lat, lon, at)
                except Exception as e:
                    print(f"[edge] position listener failed: {e}")
            return {"type": "pos"}
        if kind == "trf":
            veh = event.get("veh")
            if not isinstance(veh, dict) or len(veh) > 8:
                raise crypto.PacketError("traffic event needs vehicle counts")
            for k, v in veh.items():
                if _num(v, 0, 100) is None:
                    raise crypto.PacketError("bad vehicle count")
            return {"type": "trf"}
        raise crypto.PacketError(f"unknown event type {kind!r}")

    def _camera_only(self, bus, lat, lon, ts, event):
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine as F
        cls = _text(event.get("cls"))
        if not cls:
            raise crypto.PacketError("sighting has no class")
        sighting = {"bus_id": bus, "lat": lat, "lon": lon, "ts": ts or time.time(), "cls": cls,
                    "pci": event.get("pci"), "area": event.get("area"), "depth": _num(event.get("depth"), 0, 200)}
        known = any(d.get("defect_class") == cls and F.haversine_distance(lat, lon, d["lat"], d["lon"]) <= CAMERA_MATCH_M
                    for d in self.dedup.get_all_deduplicated_defects())
        now = time.time()
        match = None if known else next(
            (c for c in self.camera_only if c["cls"] == cls and c["bus_id"] != bus
             and now - c["ts"] <= CAMERA_MATCH_S
             and F.haversine_distance(lat, lon, c["lat"], c["lon"]) <= CAMERA_MATCH_M), None)
        if not known and match is None:
            self.camera_only.append(sighting)
            del self.camera_only[:-CAMERA_ONLY_KEPT]
            if self.events:
                self.events.publish("camera_only", {k: sighting[k] for k in ("bus_id", "lat", "lon", "cls", "ts")})
            return {"type": "cam", "decision": "held until a second bus confirms it"}
        promoted = []
        for s_ in ([match] if match else []) + [sighting]:
            res = self.dedup.ingest_fleet_detection(s_["bus_id"], s_["lat"], s_["lon"], s_["cls"], s_["pci"], s_["area"],
                                                    image_timestamp=s_["ts"], depth_cm=s_["depth"], enrich_location=False)
            promoted.append(res["defect_id"])
            if self.events:
                rec = next((d for d in self.dedup.get_all_deduplicated_defects() if d["defect_id"] == res["defect_id"]), None)
                self.events.publish("defect", {"bus_id": s_["bus_id"], "result": res, "defect": rec, "via": "edge-camera"})
        if match:
            self.camera_only.remove(match)
        return {"type": "cam", "decision": "added to the ledger", "defect_id": promoted[-1],
                "confirmed_by": "existing defect" if known else f"second bus {match['bus_id']}"}

    def ingest(self, envelopes):
        if not self.configured:
            raise RuntimeError("ROAD_SHIELD_FLEET_KEY is not set on the server")
        results, accepted = [], 0
        with self._lock:
            for env in envelopes:
                try:
                    bus, epoch, seq, event = crypto.unpack(env, self.key)
                except crypto.PacketError as e:
                    results.append({"status": "rejected", "error": str(e)})
                    break
                if seq <= self._last_seq(bus, epoch):
                    results.append({"status": "duplicate", "bus": bus, "seq": seq})
                    accepted += 1
                    continue
                try:
                    applied = self._apply(bus, event)
                    status = "applied"
                except (crypto.PacketError, KeyError, TypeError, ValueError) as e:
                    applied, status = {"error": str(e)}, "invalid_event"
                self._accept_seq(bus, epoch, seq)
                results.append({"status": status, "bus": bus, "seq": seq, **applied})
                accepted += 1
        return {"accepted_in_order": accepted, "received": len(envelopes), "results": results}

    def live_positions(self, now=None):
        now = time.time() if now is None else now
        with self._lock:
            return [dict(p) for p in self.positions.values() if now - p["received_unix"] <= POSITION_TTL_S]

    def recent_shocks(self, n=100):
        with self._lock:
            return [dict(s) for s in self.shocks[-n:]]

    def buses(self):
        with self._lock:
            rows = self._db.execute("SELECT bus_id, epoch, last_seq, last_seen, packets FROM edge_streams "
                                    "ORDER BY bus_id, last_seen").fetchall()
        return [{"bus_id": r[0], "epoch": r[1], "last_seq": r[2], "last_seen": r[3], "packets": r[4]} for r in rows]
