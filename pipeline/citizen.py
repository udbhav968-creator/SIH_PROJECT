"""
Citizen reports: anyone can send a photograph of a damaged road from the Report page.

A public form is also a spam channel, so a citizen report never goes straight into the repair ledger:

    PENDING     the engine found a pothole or crack in the photograph; it is pinned on the map for the
                operators, with the measurements, and waits
    CONFIRMED   a bus later reported the same class of defect within MATCH_RADIUS_M (automatic), or the
                ledger already had it, or an operator added it after looking ("promote")
    DISMISSED   an operator decided it was not a defect, or a duplicate

What is kept: location, class, PCI, area, depth estimate, model confidence, time, and a salted hash of the
sender's network address (for the per-sender limit only; the address itself is never stored). The
photograph is analysed in memory and not stored, so a citizen's picture of a street with people in it is
never kept by the city.

Limit: MAX_PER_HOUR reports per sender, on top of the server's general rate limit.
"""
import hashlib
import math
import os
import secrets
import sqlite3
import threading
import time
import uuid

MATCH_RADIUS_M = 15.0
MAX_PER_HOUR = 10
AREA_CLASS_IDS = {1, 2, 3}

SCHEMA = """
CREATE TABLE IF NOT EXISTS citizen_reports (
    report_id   TEXT PRIMARY KEY,
    at          REAL NOT NULL,
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    defect_class TEXT NOT NULL,
    pci         REAL,
    area_m2     REAL,
    depth_cm    REAL,
    confidence  REAL,
    status      TEXT NOT NULL DEFAULT 'PENDING',
    defect_id   TEXT,
    sender      TEXT NOT NULL,
    location_source TEXT,
    reviewed_by TEXT,
    reviewed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_citizen_status ON citizen_reports(status);
"""


class CitizenError(ValueError):
    pass


def _dist(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


class CitizenReports:
    def __init__(self, db_path, dedup_engine, events=None, clock=time.time):
        self.dedup = dedup_engine
        self.events = events
        self.clock = clock
        self._salt = secrets.token_bytes(16)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def close(self):
        with self._lock:
            self._db.close()

    def _publish(self, kind, data):
        if self.events is not None:
            try:
                self.events.publish(kind, data)
            except Exception:
                pass

    def _sender(self, client):
        return hashlib.sha256(self._salt + str(client).encode()).hexdigest()[:16]

    def _row(self, r):
        return {k: r[k] for k in r.keys() if k != "sender"}

    def _ledger_match(self, cls, lat, lon):
        best = None
        for d in self.dedup.get_all_deduplicated_defects():
            if d.get("defect_class") != cls:
                continue
            dist = _dist(lat, lon, d["lat"], d["lon"])
            if dist <= MATCH_RADIUS_M and (best is None or dist < best[1]):
                best = (d["defect_id"], dist)
        return best[0] if best else None

    def submit(self, audit, lat, lon, client="anonymous", location_source="unknown"):
        """Record a citizen photograph's result. Returns (report or None, message)."""
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            raise CitizenError("a location is needed: allow location access, or use a photograph with GPS")
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise CitizenError("location out of range")
        d = (audit or {}).get("primary_distress") or {}
        if not audit.get("is_distress") or d.get("class_id") not in AREA_CLASS_IDS:
            return None, "No pothole or crack was found in this photograph, so nothing was reported."
        sender = self._sender(client)
        now = self.clock()
        with self._lock:
            recent = self._db.execute("SELECT COUNT(*) c FROM citizen_reports WHERE sender=? AND at > ?",
                                      (sender, now - 3600)).fetchone()["c"]
            if recent >= MAX_PER_HOUR:
                raise CitizenError(f"at most {MAX_PER_HOUR} reports per hour from one connection")
            cls = str(d.get("class_name"))[:80]
            match = self._ledger_match(cls, lat, lon)
            rid = "CR-" + uuid.uuid4().hex[:10].upper()
            pci = (audit.get("astm_d6433_pci") or {}).get("pci_score")
            de = d.get("depth_estimate") or {}
            self._db.execute("""INSERT INTO citizen_reports(report_id, at, lat, lon, defect_class, pci, area_m2, depth_cm,
                                confidence, status, defect_id, sender, location_source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                             (rid, now, round(lat, 6), round(lon, 6), cls, pci, d.get("surface_area_m2"),
                              de.get("depth_cm", d.get("depth_cm")), d.get("confidence"),
                              "CONFIRMED" if match else "PENDING", match, sender, str(location_source)[:40]))
            report = self.get(rid)
        self._publish("citizen_report", report)
        msg = (f"Thank you. This matches defect {match}, which is already in the repair ledger."
               if match else "Thank you. The pothole is pinned on the map and will be added to the repair ledger "
                             "when a bus confirms it or an operator reviews it.")
        return report, msg

    def on_sighting(self, bus_id, result, at=None):
        """A bus reported a defect: confirm pending citizen reports of the same class nearby."""
        rec = next((d for d in self.dedup.get_all_deduplicated_defects() if d["defect_id"] == result["defect_id"]), None)
        if rec is None:
            return []
        done = []
        with self._lock:
            rows = self._db.execute("SELECT * FROM citizen_reports WHERE status='PENDING' AND defect_class=?",
                                    (rec["defect_class"],)).fetchall()
            for r in rows:
                if _dist(r["lat"], r["lon"], rec["lat"], rec["lon"]) <= MATCH_RADIUS_M:
                    self._db.execute("""UPDATE citizen_reports SET status='CONFIRMED', defect_id=?, reviewed_by=?,
                                        reviewed_at=? WHERE report_id=?""",
                                     (rec["defect_id"], f"fleet:{bus_id}", self.clock(), r["report_id"]))
                    done.append(r["report_id"])
        for rid in done:
            self._publish("citizen_report", self.get(rid))
        return done

    def review(self, report_id, action, actor="operator"):
        """promote: add to the ledger now; dismiss: not a defect."""
        action = str(action or "").lower()
        if action not in ("promote", "dismiss"):
            raise CitizenError("action must be promote or dismiss")
        with self._lock:
            r = self._db.execute("SELECT * FROM citizen_reports WHERE report_id=?", (report_id,)).fetchone()
            if r is None:
                raise CitizenError(f"no report {report_id}")
            if r["status"] != "PENDING":
                raise CitizenError(f"{report_id} is already {r['status'].lower()}")
            defect_id = None
            if action == "promote":
                if r["pci"] is None or r["area_m2"] is None:
                    raise CitizenError("this report has no PCI or area to put in the ledger")
                res = self.dedup.ingest_fleet_detection("citizen-report", r["lat"], r["lon"], r["defect_class"],
                                                        r["pci"], r["area_m2"], image_timestamp=r["at"],
                                                        depth_cm=r["depth_cm"], enrich_location=False)
                defect_id = res["defect_id"]
            self._db.execute("""UPDATE citizen_reports SET status=?, defect_id=?, reviewed_by=?, reviewed_at=?
                                WHERE report_id=?""",
                             ("CONFIRMED" if action == "promote" else "DISMISSED", defect_id, str(actor)[:40],
                              self.clock(), report_id))
            report = self.get(report_id)
        self._publish("citizen_report", report)
        return report

    def get(self, report_id):
        with self._lock:
            r = self._db.execute("SELECT * FROM citizen_reports WHERE report_id=?", (report_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, status=None, limit=500):
        with self._lock:
            if status:
                rows = self._db.execute("SELECT * FROM citizen_reports WHERE status=? ORDER BY at DESC LIMIT ?",
                                        (str(status).upper(), int(limit))).fetchall()
            else:
                rows = self._db.execute("SELECT * FROM citizen_reports ORDER BY at DESC LIMIT ?", (int(limit),)).fetchall()
        return [self._row(r) for r in rows]

    def counts(self):
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) c FROM citizen_reports GROUP BY status").fetchall()
        out = {"PENDING": 0, "CONFIRMED": 0, "DISMISSED": 0}
        out.update({r["status"]: r["c"] for r in rows})
        return out
