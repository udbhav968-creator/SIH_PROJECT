"""
Durable storage for the fleet defect ledger.

The deduplication engine held its registry in a Python dict. Restart the
server and every confirmed defect, every merge, every work order was gone.
That is acceptable for a demo and disqualifying for a pilot: a municipal
corporation's repair backlog cannot live in a process's memory.

This is a SQLite store behind the same interface. SQLite because it needs no
server, ships with Python, survives power loss, and comfortably handles the
write rate of a bus fleet - a few thousand reports a day is nothing for it.

Schema
------
    defects          one row per physical defect, after deduplication
    reports          one row per raw sighting, with the defect it merged into
    work_orders      issued orders and their SHA-256 seals

The `reports` table is the part worth arguing for. Storing only deduplicated
defects loses the evidence that deduplication happened - and that evidence is
exactly what a contractor disputes ("you billed this pothole twice"). Keeping
every raw sighting with a foreign key to the defect it merged into means the
merge can always be re-examined, and the dedup rate is a real query rather
than a counter.

Concurrency: WAL mode, one connection per thread. The HTTP server is
single-threaded today, but the store should not be the thing that breaks when
it stops being.
"""

import json
import os
import sqlite3
import threading
import time

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "road_shield.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS defects (
    defect_id           TEXT    PRIMARY KEY,
    lat                 REAL    NOT NULL,
    lon                 REAL    NOT NULL,
    defect_class        TEXT    NOT NULL,
    severity_pci        REAL,
    area_m2             REAL,
    confirmation_count  INTEGER NOT NULL DEFAULT 1,
    reporting_buses     TEXT    NOT NULL DEFAULT '[]',
    first_seen          REAL    NOT NULL,
    last_seen           REAL    NOT NULL,
    address             TEXT,
    elevation_m         REAL,
    extra               TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_defects_latlon ON defects(lat, lon);
CREATE INDEX IF NOT EXISTS idx_defects_class  ON defects(defect_class);

CREATE TABLE IF NOT EXISTS reports (
    report_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    defect_id    TEXT    REFERENCES defects(defect_id),
    bus_id       TEXT    NOT NULL,
    lat          REAL    NOT NULL,
    lon          REAL    NOT NULL,
    defect_class TEXT,
    area_m2      REAL,
    severity_pci REAL,
    reported_at  REAL    NOT NULL,
    merged       INTEGER NOT NULL DEFAULT 0,
    distance_m   REAL
);
CREATE INDEX IF NOT EXISTS idx_reports_defect ON reports(defect_id);
CREATE INDEX IF NOT EXISTS idx_reports_bus    ON reports(bus_id);

CREATE TABLE IF NOT EXISTS work_orders (
    order_id    TEXT PRIMARY KEY,
    defect_id   TEXT REFERENCES defects(defect_id),
    payload     TEXT NOT NULL,
    seal_sha256 TEXT NOT NULL,
    issued_at   REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ISSUED'
);
"""


class DefectStore:
    """SQLite-backed persistence. Safe to construct many times against one file."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DEFAULT_DB
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._local = threading.local()
        self._memory_conn = None
        if self.db_path == ":memory:":
            # One shared connection, or each thread would get its own empty DB.
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        with self._cursor() as cur:
            cur.executescript(SCHEMA)

    # ------------------------------------------------------------------
    def _conn(self):
        if self._memory_conn is not None:
            return self._memory_conn
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    class _Cursor:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            self.cur = self.conn.cursor()
            return self.cur

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.cur.close()
            return False

    def _cursor(self):
        return DefectStore._Cursor(self._conn())

    # ------------------------------------------------------------------
    def upsert_defect(self, record):
        """Insert or update one deduplicated defect. `record` uses engine field names."""
        known = {"defect_id", "lat", "lon", "defect_class", "severity_pci", "area_m2",
                 "confirmation_count", "reporting_buses", "first_seen_timestamp",
                 "last_seen_timestamp", "address", "elevation_m"}
        extra = {k: v for k, v in record.items() if k not in known}
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO defects (defect_id, lat, lon, defect_class, severity_pci, area_m2,
                                     confirmation_count, reporting_buses, first_seen, last_seen,
                                     address, elevation_m, extra)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(defect_id) DO UPDATE SET
                    lat=excluded.lat, lon=excluded.lon, defect_class=excluded.defect_class,
                    severity_pci=excluded.severity_pci, area_m2=excluded.area_m2,
                    confirmation_count=excluded.confirmation_count,
                    reporting_buses=excluded.reporting_buses, last_seen=excluded.last_seen,
                    address=excluded.address, elevation_m=excluded.elevation_m,
                    extra=excluded.extra
            """, (
                str(record["defect_id"]), float(record["lat"]), float(record["lon"]),
                str(record.get("defect_class", "")),
                record.get("severity_pci"), record.get("area_m2"),
                int(record.get("confirmation_count", 1)),
                json.dumps(record.get("reporting_buses", [])),
                float(record.get("first_seen_timestamp", time.time())),
                float(record.get("last_seen_timestamp", time.time())),
                record.get("address"), record.get("elevation_m"),
                json.dumps(extra, default=str),
            ))

    def record_report(self, bus_id, lat, lon, defect_class, area_m2, severity_pci,
                      defect_id, merged, distance_m=None, reported_at=None):
        """Every raw sighting, merged or not. This is the dedup audit trail."""
        with self._cursor() as cur:
            cur.execute("""INSERT INTO reports (defect_id, bus_id, lat, lon, defect_class,
                                                area_m2, severity_pci, reported_at, merged,
                                                distance_m)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (str(defect_id) if defect_id is not None else None, str(bus_id),
                         float(lat), float(lon), defect_class, area_m2, severity_pci,
                         float(reported_at or time.time()), 1 if merged else 0,
                         float(distance_m) if distance_m is not None else None))

    def load_defects(self):
        """Every stored defect, in the engine's own record shape."""
        with self._cursor() as cur:
            rows = cur.execute("SELECT * FROM defects ORDER BY defect_id").fetchall()
        out = {}
        for r in rows:
            rec = {
                "defect_id": r["defect_id"], "lat": r["lat"], "lon": r["lon"],
                "defect_class": r["defect_class"], "severity_pci": r["severity_pci"],
                "area_m2": r["area_m2"], "confirmation_count": r["confirmation_count"],
                "reporting_buses": json.loads(r["reporting_buses"] or "[]"),
                "first_seen_timestamp": r["first_seen"], "last_seen_timestamp": r["last_seen"],
                "address": r["address"], "elevation_m": r["elevation_m"],
            }
            rec.update(json.loads(r["extra"] or "{}"))
            out[r["defect_id"]] = rec
        return out

    def save_work_order(self, order_id, defect_id, payload, seal_sha256, status="ISSUED"):
        with self._cursor() as cur:
            cur.execute("""INSERT INTO work_orders (order_id, defect_id, payload, seal_sha256,
                                                    issued_at, status)
                           VALUES (?,?,?,?,?,?)
                           ON CONFLICT(order_id) DO UPDATE SET
                               payload=excluded.payload, seal_sha256=excluded.seal_sha256,
                               status=excluded.status""",
                        (str(order_id), str(defect_id) if defect_id is not None else None,
                         json.dumps(payload, default=str), str(seal_sha256),
                         time.time(), status))

    def get_work_order(self, order_id):
        with self._cursor() as cur:
            r = cur.execute("SELECT * FROM work_orders WHERE order_id=?", (str(order_id),)).fetchone()
        if not r:
            return None
        return {"order_id": r["order_id"], "defect_id": r["defect_id"],
                "payload": json.loads(r["payload"]), "seal_sha256": r["seal_sha256"],
                "issued_at": r["issued_at"], "status": r["status"]}

    def stats(self):
        """Real counts from the tables, not counters that can drift from them."""
        with self._cursor() as cur:
            defects = cur.execute("SELECT COUNT(*) c FROM defects").fetchone()["c"]
            reports = cur.execute("SELECT COUNT(*) c FROM reports").fetchone()["c"]
            merged = cur.execute("SELECT COUNT(*) c FROM reports WHERE merged=1").fetchone()["c"]
            orders = cur.execute("SELECT COUNT(*) c FROM work_orders").fetchone()["c"]
            buses = cur.execute("SELECT COUNT(DISTINCT bus_id) c FROM reports").fetchone()["c"]
        return {
            "unique_defects": defects,
            "total_reports": reports,
            "merged_reports": merged,
            "deduplication_rate_pct": round(100.0 * merged / reports, 1) if reports else 0.0,
            "work_orders": orders,
            "distinct_buses": buses,
            "database": self.db_path,
            "durable": self.db_path != ":memory:",
        }

    def next_defect_id(self, start=1001):
        """
        The next free numeric suffix.

        Ids look like DEF-BLR-1001, so the maximum is taken over the numeric
        tail rather than the string - otherwise DEF-BLR-999 would sort above
        DEF-BLR-1001 and the next id would collide with an existing row.
        """
        with self._cursor() as cur:
            rows = cur.execute("SELECT defect_id FROM defects").fetchall()
        highest = start - 1
        for r in rows:
            tail = str(r["defect_id"]).rsplit("-", 1)[-1]
            if tail.isdigit():
                highest = max(highest, int(tail))
        return highest + 1

    def close(self):
        conn = self._memory_conn or getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
