"""
Traffic on each road, estimated from the buses' own cameras.

The Priority Index weighs a defect by the traffic over it. Until now the only basis was how many
buses had reported the defect, a count of routes rather than of vehicles. This module turns the
COCO detector's vehicle counts in bus frames into a flow estimate per 100 m road cell:

    density   k = PCU visible in the frame / length of road in view   (PCU per km of carriageway)
    speed     v = the bus's own speed, as the speed of the stream it is moving in (km/h)
    flow      q = k * v                                                (PCU per hour)
    daily     Q = q / share of daily traffic in that hour             (PCU per day)

k * v is the fundamental relation of traffic flow (q = kv). The bus's speed stands in for the
stream's: a bus moves with general traffic except at stops, so frames with the bus below
MIN_SPEED_KMH (a stop, or a red light) are counted for density but not for flow. The hour-of-day
shares are a typical Indian urban arterial profile, not a count on these roads, and every estimate
says so. A cell's value is the median over its observations of the last 30 days, and only cells with
MIN_OBSERVATIONS in at least MIN_HOURS different hours are used by the Priority Index; the
rest fall back to the bus-count basis, as before.

What a city should do before relying on it: run classified volume counts (IRC:SP:41) on a few
corridors, compare, and set VIEW_LENGTH_M and the hour profile from the result.
"""
import math
import sqlite3
import statistics
import threading
import time

PCU = {"Car": 1.0, "Two-Wheeler": 0.5, "City Bus": 2.0, "Heavy Truck": 2.5}   # models/urban_traffic_net.py
VIEW_LENGTH_M = 40.0          # road length a dashcam frame covers well enough to count vehicles, one direction
MIN_SPEED_KMH = 8.0
CELL_DEG = 0.001              # about 110 m north-south
WINDOW_S = 30 * 24 * 3600
MIN_OBSERVATIONS = 6
MIN_HOURS = 3
# Share of a day's traffic in each hour (0-23). A generic Indian urban arterial profile with morning and evening
# peaks; replace with the city's own counts.
_PROFILE = [0.010, 0.007, 0.006, 0.006, 0.009, 0.018, 0.035, 0.058, 0.073, 0.072, 0.062, 0.057,
            0.055, 0.054, 0.053, 0.055, 0.061, 0.070, 0.074, 0.066, 0.049, 0.033, 0.022, 0.015]
HOURLY_SHARE = [x / sum(_PROFILE) for x in _PROFILE]          # shares of one day: sums to 1
# Night hours carry under 1% of the day each, so one frame's count would be multiplied about 150 times: flow is not
# expanded from them (density is still recorded).
MIN_HOUR_SHARE = 0.02

SCHEMA = """
CREATE TABLE IF NOT EXISTS traffic_obs (
    cell        TEXT NOT NULL,
    at          REAL NOT NULL,
    hour        INTEGER NOT NULL,
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    pcu_in_view REAL NOT NULL,
    density     REAL NOT NULL,
    speed_kmh   REAL,
    flow_pcu_h  REAL,
    daily_pcu   REAL,
    frames      INTEGER NOT NULL,
    bus_id      TEXT,
    source      TEXT
);
CREATE INDEX IF NOT EXISTS idx_traffic_cell ON traffic_obs(cell, at);
"""


def cell_of(lat, lon):
    return f"{math.floor(lat / CELL_DEG)}:{math.floor(lon / CELL_DEG)}"


def cell_centre(cell):
    a, b = (int(x) for x in cell.split(":"))
    return round((a + 0.5) * CELL_DEG, 6), round((b + 0.5) * CELL_DEG, 6)


def pcu_of(counts):
    return sum(PCU.get(k, 1.0) * float(v) for k, v in (counts or {}).items() if v)


def local_hour(at, utc_offset_h=5.5):
    return int(((at / 3600.0) + utc_offset_h) % 24)


class TrafficEstimator:
    def __init__(self, db_path, events=None, clock=time.time, utc_offset_h=5.5):
        self.events = events
        self.clock = clock
        self.utc_offset_h = utc_offset_h
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._cache = {}

    def observe(self, lat, lon, vehicle_counts, speed_kmh=None, at=None, frames=1, bus_id=None, source="fleet"):
        """One observation: mean vehicles per frame by type over `frames` frames at this place."""
        lat, lon = float(lat), float(lon)
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("bad location")
        at = float(at) if at else self.clock()
        pcu = pcu_of(vehicle_counts)
        if not math.isfinite(pcu) or pcu < 0 or pcu > 200:
            raise ValueError("implausible vehicle count")
        density = pcu / (VIEW_LENGTH_M / 1000.0)
        spd = None if speed_kmh is None else float(speed_kmh)
        if spd is not None and not (math.isfinite(spd) and 0 <= spd <= 150):
            spd = None
        hour = local_hour(at, self.utc_offset_h)
        flow = density * spd if spd is not None and spd >= MIN_SPEED_KMH else None
        daily = flow / HOURLY_SHARE[hour] if flow is not None and HOURLY_SHARE[hour] >= MIN_HOUR_SHARE else None
        with self._lock:
            self._db.execute("""INSERT INTO traffic_obs(cell, at, hour, lat, lon, pcu_in_view, density, speed_kmh,
                                flow_pcu_h, daily_pcu, frames, bus_id, source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                             (cell_of(lat, lon), at, hour, lat, lon, round(pcu, 3), round(density, 2), spd,
                              None if flow is None else round(flow, 1), None if daily is None else round(daily, 0),
                              int(frames), bus_id, source))
            self._cache.clear()
        return {"cell": cell_of(lat, lon), "pcu_in_view": round(pcu, 2), "density_pcu_per_km": round(density, 1),
                "flow_pcu_per_h": None if flow is None else round(flow, 1),
                "daily_pcu_estimate": None if daily is None else round(daily),
                "counted_for_flow": daily is not None}

    def _cell_estimate(self, cells, now):
        rows = []
        with self._lock:
            for c in cells:
                rows += self._db.execute("SELECT * FROM traffic_obs WHERE cell=? AND at>=? AND daily_pcu IS NOT NULL",
                                         (c, now - WINDOW_S)).fetchall()
        if not rows:
            return None
        daily = [r["daily_pcu"] for r in rows]
        hours = sorted({r["hour"] for r in rows})
        usable = len(rows) >= MIN_OBSERVATIONS and len(hours) >= MIN_HOURS
        return {
            "daily_pcu_estimate": round(statistics.median(daily)),
            "observations": len(rows),
            "hours_observed": hours,
            "buses": len({r["bus_id"] for r in rows if r["bus_id"]}),
            "density_pcu_per_km_median": round(statistics.median(r["density"] for r in rows), 1),
            "usable_for_priority": usable,
            "basis": "camera density x bus speed, expanded by an assumed hour-of-day profile",
        }

    def estimate(self, lat, lon, now=None):
        """The cell's estimate, widened to the 8 neighbouring cells when the cell alone has too little."""
        now = now or self.clock()
        key = (cell_of(lat, lon), int(now // 300))
        if key in self._cache:
            return self._cache[key]
        c = cell_of(lat, lon)
        est = self._cell_estimate([c], now)
        if est is None or not est["usable_for_priority"]:
            a, b = (int(x) for x in c.split(":"))
            wide = self._cell_estimate([f"{a + i}:{b + j}" for i in (-1, 0, 1) for j in (-1, 0, 1)], now)
            if wide is not None and (est is None or wide["usable_for_priority"]):
                est = dict(wide, widened_to_neighbours=True)
        if len(self._cache) > 5000:
            self._cache.clear()
        self._cache[key] = est
        return est

    def cells(self, now=None, limit=2000):
        now = now or self.clock()
        with self._lock:
            rows = self._db.execute("""SELECT cell, COUNT(*) n, COUNT(DISTINCT hour) h FROM traffic_obs
                                       WHERE at>=? AND daily_pcu IS NOT NULL GROUP BY cell ORDER BY n DESC LIMIT ?""",
                                    (now - WINDOW_S, int(limit))).fetchall()
        out = []
        for r in rows:
            est = self._cell_estimate([r["cell"]], now)
            lat, lon = cell_centre(r["cell"])
            out.append({"cell": r["cell"], "lat": lat, "lon": lon, **est})
        return out

    def stats(self, now=None):
        now = now or self.clock()
        with self._lock:
            r = self._db.execute("""SELECT COUNT(*) n, COUNT(DISTINCT cell) cells, SUM(daily_pcu IS NOT NULL) flow_n,
                                    MAX(at) last FROM traffic_obs WHERE at>=?""", (now - WINDOW_S,)).fetchone()
        return {"observations_30d": r["n"] or 0, "cells_30d": r["cells"] or 0,
                "observations_with_flow": r["flow_n"] or 0, "last_observation": r["last"],
                "method": __doc__.strip().splitlines()[0], "view_length_m": VIEW_LENGTH_M,
                "min_speed_kmh": MIN_SPEED_KMH, "min_observations": MIN_OBSERVATIONS, "min_hours": MIN_HOURS}

    def enrich(self, defect):
        """A copy of a ledger defect with traffic_pcu_per_day set when its cell has a usable estimate."""
        if defect.get("lat") is None or defect.get("lon") is None or defect.get("traffic_pcu_per_day") is not None:
            return defect
        est = self.estimate(defect["lat"], defect["lon"])
        if est and est["usable_for_priority"]:
            return dict(defect, traffic_pcu_per_day=est["daily_pcu_estimate"], traffic_estimate=est)
        return defect


def counts_from_audit(audit):
    """Vehicle counts by type from a pipeline result, or None when the detector did not run
    (no detector is not the same as no vehicles)."""
    if not audit or not (audit.get("scene_summary") or {}).get("available"):
        return None
    counts = {}
    for v in audit.get("all_vehicles") or []:
        d = v.get("distance_meters")
        if d is not None and d > VIEW_LENGTH_M:
            continue            # beyond the stretch the density is divided by
        counts[v.get("vehicle_type", "Car")] = counts.get(v.get("vehicle_type", "Car"), 0) + 1
    return counts
