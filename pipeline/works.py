"""
Repair lifecycle: from a defect in the ledger to a repair the fleet has confirmed.

    ISSUED ──> ASSIGNED ──> IN_PROGRESS ──> REPAIRED ──> VERIFIED
       │           │                           │            │
       └───────────┴──> CANCELLED              └─> REOPENED <┘   (the defect is seen again)
                                                      │
                                                      └──> ASSIGNED / IN_PROGRESS / CANCELLED

Orders are issued from a ledger defect, so every quantity on them (class, area, depth, PCI, location) is
the measured one; the order itself is sealed by models/morth_dispatch_agent.py.

History is append-only and hash-chained: each event stores the SHA-256 of the previous event plus its own
fields, so deleting or editing a step (say, backdating "REPAIRED" to meet an SLA) breaks every hash after
it, and get() reports the chain as broken.

Verification by the fleet, which is what makes a contractor's "repaired" claim checkable without sending an
inspector:

  * After REPAIRED, every bus whose track passes within PASS_RADIUS_M of the defect is recorded (once per
    bus per 10 minutes, and only for positions recorded after the repair: a bus that was offline uploads
    its backlog late, and its old positions are not evidence). The track is the straight line between two consecutive positions of the bus (at
    most 2 minutes and 600 m apart), because positions arrive every few seconds, tens of metres apart. When VERIFY_PASSES different buses have passed and at least VERIFY_MIN_HOURS have gone
    by with no new sighting of that defect, the order becomes VERIFIED by "fleet", with the passes as
    evidence.
  * A new sighting of the same defect (same class, within the deduplication radius) after REPAIRED or
    VERIFIED reopens the order automatically, with the sighting as evidence.

Limit, stated rather than hidden: a pass is a GPS position near the defect, not proof that the camera
looked at that patch of road (the bus may have been in the far lane). Several passes by different buses
over a day make a miss by all of them unlikely. A pass also only counts once SETTLE_S (60 s) have gone by
without a sighting, so a bus that drives over a pothole that is still there reopens the order instead of
verifying it. Both numbers are configurable, which is why both numbers are configurable
(ROAD_SHIELD_VERIFY_PASSES, ROAD_SHIELD_VERIFY_MIN_HOURS) and why an inspector can still verify or reopen
by hand.
"""
import contextlib
import hashlib
import json
import math
import os
import sqlite3
import threading
import time

STATES = ("ISSUED", "ASSIGNED", "IN_PROGRESS", "REPAIRED", "VERIFIED", "REOPENED", "CANCELLED")
ACTIVE = {"ISSUED", "ASSIGNED", "IN_PROGRESS", "REPAIRED", "REOPENED"}
CLOSED = {"VERIFIED", "CANCELLED"}
TRANSITIONS = {
    "ISSUED": {"ASSIGNED", "CANCELLED"},
    "ASSIGNED": {"IN_PROGRESS", "CANCELLED"},
    "IN_PROGRESS": {"REPAIRED", "CANCELLED"},
    "REPAIRED": {"VERIFIED", "REOPENED"},
    "VERIFIED": {"REOPENED"},
    "REOPENED": {"ASSIGNED", "IN_PROGRESS", "CANCELLED"},
    "CANCELLED": set(),
}
SLA_STOPS = {"REPAIRED", "VERIFIED", "CANCELLED"}
PASS_RADIUS_M = 15.0
PASS_COOLDOWN_S = 600

SCHEMA = """
CREATE TABLE IF NOT EXISTS work_orders (
    order_id    TEXT PRIMARY KEY,
    defect_id   TEXT REFERENCES defects(defect_id),
    payload     TEXT NOT NULL,
    seal_sha256 TEXT NOT NULL,
    issued_at   REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ISSUED'
);
CREATE TABLE IF NOT EXISTS order_events (
    event_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id  TEXT    NOT NULL,
    at        REAL    NOT NULL,
    status    TEXT    NOT NULL,
    actor     TEXT    NOT NULL,
    note      TEXT,
    evidence  TEXT    NOT NULL DEFAULT '{}',
    prev_hash TEXT    NOT NULL,
    hash      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_order ON order_events(order_id, event_id);
CREATE TABLE IF NOT EXISTS order_details (
    order_id    TEXT PRIMARY KEY,
    contractor  TEXT,
    repaired_at REAL,
    lat         REAL,
    lon         REAL,
    defect_class TEXT
);
CREATE TABLE IF NOT EXISTS repair_passes (
    order_id TEXT NOT NULL,
    bus_id   TEXT NOT NULL,
    at       REAL NOT NULL,
    distance_m REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_passes_order ON repair_passes(order_id);
"""
GENESIS = "0" * 64


class WorkflowError(ValueError):
    pass


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _event_hash(prev_hash, order_id, at, status, actor, note, evidence):
    body = json.dumps({"order_id": order_id, "at": round(float(at), 3), "status": status, "actor": actor,
                       "note": note, "evidence": evidence}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()


def _segment_distance_m(lat, lon, a, b):
    """Shortest distance (m) from point (lat, lon) to the straight track a -> b, each (lat, lon). A bus reports
    a position every few seconds, tens of metres apart, so 'did it pass the defect' is a question about the
    track between two fixes, not about either fix."""
    k = 111_320.0
    c = math.cos(math.radians(lat))
    ax, ay = (a[1] - lon) * k * c, (a[0] - lat) * k
    bx, by = (b[1] - lon) * k * c, (b[0] - lat) * k
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / L2))
    return math.hypot(ax + t * dx, ay + t * dy)


SETTLE_S = 60                  # a pass only counts as clean once its bus's own report for that spot could have arrived
MAX_TRACK_GAP_S = 120          # fixes further apart than this are not joined into a track
MAX_TRACK_LEN_M = 600


def _env_num(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


class WorkOrders:
    def __init__(self, db_path, dedup_engine, dispatch_agent, events=None, verify_passes=None,
                 verify_min_hours=None, pass_radius_m=PASS_RADIUS_M, clock=time.time):
        self.dedup = dedup_engine
        self.dispatch = dispatch_agent
        self.events = events
        self.verify_passes = int(verify_passes if verify_passes is not None else _env_num("ROAD_SHIELD_VERIFY_PASSES", 3))
        self.verify_min_hours = float(verify_min_hours if verify_min_hours is not None
                                      else _env_num("ROAD_SHIELD_VERIFY_MIN_HOURS", 24))
        self.pass_radius_m = float(pass_radius_m)
        self.clock = clock
        self._lock = threading.RLock()
        self._last_fix = {}            # bus_id -> (lat, lon, at): the previous position, to form a track
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None, timeout=10)
        self._db.row_factory = sqlite3.Row
        if db_path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)

    def close(self):
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ helpers
    @contextlib.contextmanager
    def _tx(self):
        """One SQLite transaction for a multi-statement change, so a crash cannot leave half an order."""
        if self._db.in_transaction:
            yield
            return
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def _facts(self, order_id):
        """Status, contractor and repair time as recorded in the hash-chained history, which is the only place
        they are trusted from (a cached column could be edited without breaking the chain)."""
        rows = self._db.execute("SELECT status, at, evidence FROM order_events WHERE order_id=? ORDER BY event_id",
                                (order_id,)).fetchall()
        status, contractor, repaired_at = None, None, None
        for r in rows:
            ev = json.loads(r["evidence"] or "{}")
            status = r["status"]
            if ev.get("contractor"):
                contractor = ev["contractor"]
            if status == "REPAIRED":
                repaired_at = r["at"]
            elif status == "REOPENED":
                repaired_at = None
        if status not in ("REPAIRED", "VERIFIED"):
            repaired_at = None
        return {"status": status, "contractor": contractor, "repaired_at": repaired_at}

    def _publish(self, kind, data):
        if self.events is not None:
            try:
                self.events.publish(kind, data)
            except Exception:
                pass

    def _defect(self, defect_id):
        return next((d for d in self.dedup.get_all_deduplicated_defects() if d["defect_id"] == defect_id), None)

    def _append(self, order_id, status, actor, note=None, evidence=None, at=None):
        at = self.clock() if at is None else at
        evidence = evidence or {}
        row = self._db.execute("SELECT hash FROM order_events WHERE order_id=? ORDER BY event_id DESC LIMIT 1",
                               (order_id,)).fetchone()
        prev = row["hash"] if row else GENESIS
        h = _event_hash(prev, order_id, at, status, actor, note, evidence)
        self._db.execute("""INSERT INTO order_events(order_id, at, status, actor, note, evidence, prev_hash, hash)
                            VALUES(?,?,?,?,?,?,?,?)""",
                         (order_id, at, status, actor, note, json.dumps(evidence, default=str), prev, h))
        self._db.execute("UPDATE work_orders SET status=? WHERE order_id=?", (status, order_id))
        return at

    def active_order_for(self, defect_id):
        r = self._db.execute(f"""SELECT order_id FROM work_orders WHERE defect_id=? AND status IN
                                 ({','.join('?' * len(ACTIVE | {'VERIFIED'}))}) ORDER BY issued_at DESC LIMIT 1""",
                             (defect_id, *sorted(ACTIVE | {"VERIFIED"}))).fetchone()
        return r["order_id"] if r else None

    # ------------------------------------------------------------------ actions
    def issue(self, defect_id, actor="operator", depth_cm=None, corridor=None, note=None):
        with self._lock:
            d = self._defect(defect_id)
            if d is None:
                raise WorkflowError(f"no defect {defect_id} in the ledger")
            existing = self.active_order_for(defect_id)
            if existing and self._status(existing) in ACTIVE:
                raise WorkflowError(f"defect {defect_id} already has an open order {existing}")
            depth = depth_cm if depth_cm is not None else d.get("depth_cm")
            if depth is None:
                raise WorkflowError("this defect has no depth on record; give depth_cm measured on site")
            try:
                depth = float(depth)
            except (TypeError, ValueError):
                raise WorkflowError("depth_cm must be a number")
            if not (math.isfinite(depth) and 0 < depth <= 100):
                raise WorkflowError("depth_cm must be more than 0 and at most 100")
            wo = self.dispatch.generate_work_order(
                corridor_id=corridor or (d.get("address") or "UNSPECIFIED")[:60], latitude=d["lat"],
                longitude=d["lon"], distress_class=d["defect_class"], area_sqm=d.get("area_m2"),
                depth_cm=float(depth), pci_score=d.get("severity_pci"))
            oid = wo["work_order_id"]
            now = self.clock()
            with self._tx():
                self._db.execute("""INSERT INTO work_orders(order_id, defect_id, payload, seal_sha256, issued_at, status)
                                    VALUES(?,?,?,?,?, 'ISSUED')""",
                                 (oid, defect_id, json.dumps(wo, default=str), wo["sha256_cryptographic_seal"], now))
                self._db.execute("""INSERT INTO order_details(order_id, lat, lon, defect_class) VALUES(?,?,?,?)""",
                                 (oid, d["lat"], d["lon"], d["defect_class"]))
                self._append(oid, "ISSUED", actor, note, {"defect_id": defect_id}, at=now)
        self._publish("work_order", {"work_order_id": oid, "defect_id": defect_id, "status": "ISSUED",
                                     "priority": wo.get("priority"), "allocated_budget_inr": wo.get("allocated_budget_inr"),
                                     "seal_algorithm": wo.get("seal_algorithm"), "coordinates": wo.get("coordinates")})
        return self.get(oid)

    def _status(self, order_id):
        r = self._db.execute("SELECT status FROM work_orders WHERE order_id=?", (order_id,)).fetchone()
        return r["status"] if r else None

    def transition(self, order_id, status, actor="operator", note=None, contractor=None, evidence=None):
        status = str(status or "").upper()
        if status not in STATES:
            raise WorkflowError(f"unknown status {status!r}; one of {', '.join(STATES)}")
        with self._lock:
            cur = self._status(order_id)
            if cur is None:
                raise WorkflowError(f"no work order {order_id}")
            if status not in TRANSITIONS[cur]:
                allowed = ", ".join(sorted(TRANSITIONS[cur])) or "none (the order is closed)"
                raise WorkflowError(f"{order_id} is {cur}; it can move to: {allowed}")
            if status == "ASSIGNED" and not (contractor or self._contractor(order_id)):
                raise WorkflowError("ASSIGNED needs a contractor")
            evidence = dict(evidence or {})
            if contractor:
                evidence["contractor"] = str(contractor)[:80]
            with self._tx():
                self._append(order_id, status, actor, note, evidence)
                if status in ("REPAIRED", "REOPENED", "CANCELLED"):
                    self._db.execute("DELETE FROM repair_passes WHERE order_id=?", (order_id,))
        self._publish("work_order", {"work_order_id": order_id, "status": status, "previous": cur, "actor": actor,
                                     "note": note})
        return self.get(order_id)

    def _contractor(self, order_id):
        return self._facts(order_id)["contractor"]

    # --------------------------------------------------------- fleet evidence
    def on_sighting(self, defect_id, bus_id, at=None):
        """A bus (or a photo) reported this defect. After a repair, that reopens the order."""
        at = self.clock() if at is None else min(at, self.clock())
        reopened = None
        with self._lock:
            oid = self.active_order_for(defect_id)
            if oid and self._status(oid) in ("REPAIRED", "VERIFIED"):
                repaired_at = self._facts(oid)["repaired_at"]
                if repaired_at is not None and at < repaired_at:
                    return None                 # recorded before the repair, uploaded late: not evidence
                with self._tx():
                    self._append(oid, "REOPENED", "fleet", f"seen again by {bus_id} after the repair",
                                 {"bus_id": bus_id, "seen_at": at})
                    self._db.execute("DELETE FROM repair_passes WHERE order_id=?", (oid,))
                reopened = oid
        if reopened:
            self._publish("work_order", {"work_order_id": reopened, "status": "REOPENED", "actor": "fleet",
                                         "note": f"seen again by {bus_id}"})
        return reopened

    def on_bus_position(self, bus_id, lat, lon, at=None):
        """Record passes over repaired defects; verify the ones with enough clean passes."""
        at = self.clock() if at is None else min(at, self.clock())
        verified = []
        with self._lock, self._tx():
            prev = self._last_fix.get(bus_id)
            self._last_fix[bus_id] = (lat, lon, at)
            track = None
            if prev and 0 <= at - prev[2] <= MAX_TRACK_GAP_S and _haversine(prev[0], prev[1], lat, lon) <= MAX_TRACK_LEN_M:
                track = ((prev[0], prev[1]), (lat, lon))
            rows = self._db.execute("""SELECT w.order_id, d.lat, d.lon FROM work_orders w
                                       JOIN order_details d ON d.order_id = w.order_id
                                       WHERE w.status = 'REPAIRED'""").fetchall()
            for r in rows:
                dist = (_segment_distance_m(r["lat"], r["lon"], *track) if track
                        else _haversine(lat, lon, r["lat"], r["lon"]))
                if dist > self.pass_radius_m:
                    continue
                repaired_at = self._facts(r["order_id"])["repaired_at"]
                if repaired_at is None or at < repaired_at:
                    continue                    # a position from before the repair, uploaded late
                last = self._db.execute("SELECT MAX(at) m FROM repair_passes WHERE order_id=? AND bus_id=?",
                                        (r["order_id"], bus_id)).fetchone()["m"]
                if last is not None and at - last < PASS_COOLDOWN_S:
                    continue
                self._db.execute("INSERT INTO repair_passes(order_id, bus_id, at, distance_m) VALUES(?,?,?,?)",
                                 (r["order_id"], bus_id, at, round(dist, 1)))
            verified = self._settle(at)
        for oid in verified:
            self._publish("work_order", {"work_order_id": oid, "status": "VERIFIED", "actor": "fleet"})
        return verified

    def _settle(self, now):
        """VERIFIED for every repaired order with enough passes, the waiting time served, and no sighting in the
        SETTLE_S seconds since its last pass (a sighting would already have reopened it). Caller holds the lock."""
        done = []
        rows = self._db.execute("SELECT order_id FROM work_orders WHERE status = 'REPAIRED'").fetchall()
        for r in rows:
            passes = self._db.execute("SELECT bus_id, at, distance_m FROM repair_passes WHERE order_id=? ORDER BY at",
                                      (r["order_id"],)).fetchall()
            if len({p["bus_id"] for p in passes}) < self.verify_passes or now - passes[-1]["at"] < SETTLE_S:
                continue
            repaired_at = self._facts(r["order_id"])["repaired_at"] or now
            waited_h = (now - repaired_at) / 3600.0
            if waited_h < self.verify_min_hours:
                continue
            self._append(r["order_id"], "VERIFIED", "fleet",
                         f"{len({p['bus_id'] for p in passes})} different buses passed over {waited_h:.1f} h "
                         f"with no new sighting",
                         {"passes": [dict(p) for p in passes],
                          "rule": {"passes": self.verify_passes, "min_hours": self.verify_min_hours,
                                   "settle_s": SETTLE_S}}, at=now)
            done.append(r["order_id"])
        return done

    def tick(self, now=None):
        """Settle pending verifications; called by readers and the server's background loop."""
        now = self.clock() if now is None else now
        with self._lock, self._tx():
            done = self._settle(now)
        for oid in done:
            self._publish("work_order", {"work_order_id": oid, "status": "VERIFIED", "actor": "fleet"})
        return done

    # ---------------------------------------------------------------- reading
    def history(self, order_id):
        rows = self._db.execute("SELECT * FROM order_events WHERE order_id=? ORDER BY event_id", (order_id,)).fetchall()
        return [{"at": r["at"], "status": r["status"], "actor": r["actor"], "note": r["note"],
                 "evidence": json.loads(r["evidence"] or "{}"), "prev_hash": r["prev_hash"], "hash": r["hash"]}
                for r in rows]

    def verify_chain(self, order_id):
        """True when every event hashes onto the one before it and the order's current status is the last
        event's. An empty history (an order with no ISSUED event) is not intact."""
        prev, last = GENESIS, None
        for e in self.history(order_id):
            if e["prev_hash"] != prev:
                return False
            if _event_hash(prev, order_id, e["at"], e["status"], e["actor"], e["note"], e["evidence"]) != e["hash"]:
                return False
            prev, last = e["hash"], e["status"]
        return last is not None and last == self._status(order_id)

    def _row(self, r, now):
        payload = json.loads(r["payload"])
        sla_h = payload.get("sla_resolution_hours")
        due = r["issued_at"] + sla_h * 3600 if sla_h else None
        passes = self._db.execute("SELECT COUNT(DISTINCT bus_id) c FROM repair_passes WHERE order_id=?",
                                  (r["order_id"],)).fetchone()["c"]
        det = self._db.execute("SELECT * FROM order_details WHERE order_id=?", (r["order_id"],)).fetchone()
        facts = self._facts(r["order_id"])
        repaired_at = facts["repaired_at"]
        stop = repaired_at if repaired_at and r["status"] in ("REPAIRED", "VERIFIED") else None
        return {
            "work_order_id": r["order_id"], "defect_id": r["defect_id"], "status": r["status"],
            "issued_at": r["issued_at"], "sla_hours": sla_h, "due_at": due,
            "overdue": bool(due and r["status"] not in SLA_STOPS and now > due),
            "met_sla": (bool(stop <= due) if (due and stop) else None),
            "contractor": facts["contractor"], "repaired_at": repaired_at,
            "fleet_passes_since_repair": passes, "passes_needed": self.verify_passes,
            "lat": det["lat"] if det else None, "lon": det["lon"] if det else None,
            "defect_class": payload.get("distress_type"), "priority": payload.get("priority"),
            "budget_inr": payload.get("allocated_budget_inr"), "tonnes": payload.get("required_mass_tonnes"),
            "area_m2": payload.get("surface_area_sqm"), "depth_cm": payload.get("depth_cm"),
            "pci": payload.get("pavement_pci"), "seal_algorithm": payload.get("seal_algorithm", "SHA-256"),
        }

    def list(self, status=None):
        self.tick()
        now = self.clock()
        with self._lock:
            if status:
                rows = self._db.execute("SELECT * FROM work_orders WHERE status=? ORDER BY issued_at DESC",
                                        (str(status).upper(),)).fetchall()
            else:
                rows = self._db.execute("SELECT * FROM work_orders ORDER BY issued_at DESC").fetchall()
            return [self._row(r, now) for r in rows]

    def get(self, order_id):
        self.tick()
        with self._lock:
            r = self._db.execute("SELECT * FROM work_orders WHERE order_id=?", (order_id,)).fetchone()
            if r is None:
                return None
            out = self._row(r, self.clock())
            payload = json.loads(r["payload"])
            out["work_order"] = payload
            out["history"] = self.history(order_id)
            out["history_chain_intact"] = self.verify_chain(order_id)
            out["seal_status"] = self.dispatch.check_work_order_seal(payload)
            out["allowed_next"] = sorted(TRANSITIONS[r["status"]])
            return out

    def status_by_defect(self):
        with self._lock:
            rows = self._db.execute("""SELECT defect_id, order_id, status FROM work_orders w
                                       WHERE issued_at = (SELECT MAX(issued_at) FROM work_orders x
                                                          WHERE x.defect_id = w.defect_id)""").fetchall()
        return {r["defect_id"]: {"work_order_id": r["order_id"], "status": r["status"]} for r in rows}

    def summary(self):
        orders = self.list()
        by = {s: 0 for s in STATES}
        for o in orders:
            by[o["status"]] += 1
        closed = [o for o in orders if o["met_sla"] is not None]
        return {"orders": len(orders), "by_status": by, "overdue": sum(o["overdue"] for o in orders),
                "met_sla_pct": round(100.0 * sum(o["met_sla"] for o in closed) / len(closed), 1) if closed else None,
                "budget_open_inr": round(sum(o["budget_inr"] or 0 for o in orders if o["status"] in ACTIVE), 2),
                "verify_rule": {"passes": self.verify_passes, "min_hours": self.verify_min_hours,
                                "pass_radius_m": self.pass_radius_m}}
