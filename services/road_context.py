"""
Facts about a defect's location from public APIs: the rain it will get, and the road it is on.

    rainfall(lat, lon)      Open-Meteo historical weather API (ERA5 reanalysis), no key needed
                            - observed rain over the last 30 days
                            - expected rain over the next 180 days: the mean of the same 180 calendar days in
                              each of the last three years (a climatology, not a weather forecast)
    road(lat, lon)          OpenStreetMap through the Overpass API, no key needed
                            - the nearest mapped road within 30 m (by distance to the road's line; within 3 m the bigger road
                              class wins): class (highway=*), name, ref, surface,
                              lanes, speed limit; data (c) OpenStreetMap contributors, ODbL

What uses them
  The deterioration forecast (models/pavement_deterioration_forecaster.py) needs "seasonal rain (mm)". Until
  now it was always the typical value 650 mm, listed under modelling_assumptions. With the context APIs on,
  the 180-day climatology at the photograph's location is used instead and the response says where it came
  from. The forecaster was fitted on 0-650 mm; above that the input is held at 650 and flagged, because a
  tree model does not extrapolate and pretending otherwise would be inventing a number.
  The road record is shown on the printable work order (fetched live, it is not part of the sealed order) and
  by GET /api/v1/context. It is not an input to the Priority Index: road class says how important a road was planned to be, not how much it is used, and
  the index deliberately has no "important road" term (models/priority_index.py).

Switch
  ROAD_SHIELD_CONTEXT_APIS=1 turns them on (`python -m api.server` sets it unless it is set to 0). Off by
  default in the library, so tests and offline runs never wait on the network. Answers are cached per ~1 km
  cell (rain, one day) and per ~11 m cell (road, a week); a failed call is cached for ten minutes and
  reported as unavailable, never filled in. After three failures in a row the lookups stop for ten minutes
  (circuit breaker), so an offline machine does not wait on a timeout for every new place.
"""
import datetime
import json
import os
import threading
import time
import urllib.parse
import urllib.request

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TRAINED_RAIN_MAX_MM = 650.0
RAIN_TTL_S, ROAD_TTL_S, FAIL_TTL_S = 86400, 7 * 86400, 600
BREAKER_FAILURES, BREAKER_OPEN_S = 3, 600
TIMEOUT_S = 4.0
MAJOR = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
         "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link", "living_street",
         "service", "track"]


def enabled():
    return os.environ.get("ROAD_SHIELD_CONTEXT_APIS", "0") == "1"


class RoadContext:
    def __init__(self, fetch=None, today=None):
        contact = os.environ.get("ROAD_SHIELD_CONTACT", "SIH 2026 student project")
        self.user_agent = f"ROAD-SHIELD/3.0 ({contact})"
        self._fetch = fetch or self._http_json
        self._today = today
        self._cache = {}
        self._lock = threading.Lock()
        self._fails, self._open_until = 0, 0.0

    # -- plumbing ------------------------------------------------------------------------------
    def _http_json(self, url, data=None, timeout=TIMEOUT_S):
        req = urllib.request.Request(url, data=data, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _cached(self, key, ttl, compute):
        now = time.time()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] < (ttl if hit[1].get("available") else FAIL_TTL_S):
                return hit[1]
            if now < self._open_until:
                return {"available": False, "reason": "public APIs unreachable recently; lookups paused for a few minutes"}
        try:
            val = compute()
        except Exception as e:                                     # a malformed answer is a failure, not a crash
            val = {"available": False, "reason": f"unexpected answer: {str(e)[:120]}"}
        with self._lock:
            if val.get("available"):
                self._fails = 0
            else:
                self._fails += 1
                if self._fails >= BREAKER_FAILURES:
                    self._open_until, self._fails = now + BREAKER_OPEN_S, 0
            self._cache[key] = (now, val)
            if len(self._cache) > 5000:
                for k in sorted(self._cache, key=lambda k: self._cache[k][0])[:1000]:
                    del self._cache[k]
        return val

    def today(self):
        return self._today or datetime.date.today()

    # -- rain ----------------------------------------------------------------------------------
    def rainfall(self, lat, lon):
        lat, lon = float(lat), float(lon)
        key = ("rain", round(lat, 2), round(lon, 2), self.today().isoformat())
        return self._cached(key, RAIN_TTL_S, lambda: self._rainfall(lat, lon))

    def _rainfall(self, lat, lon):
        today = self.today()
        end = today - datetime.timedelta(days=6)                    # the archive lags by a few days
        start = datetime.date(today.year - 3, today.month, 1)
        q = urllib.parse.urlencode({"latitude": round(lat, 3), "longitude": round(lon, 3),
                                    "start_date": start.isoformat(), "end_date": end.isoformat(),
                                    "daily": "precipitation_sum", "timezone": "auto"})
        try:
            data = self._fetch(f"{ARCHIVE_URL}?{q}")
            days = data["daily"]["time"]
            rain = data["daily"]["precipitation_sum"]
        except Exception as e:
            return {"available": False, "reason": f"Open-Meteo not reachable: {str(e)[:120]}"}
        series = {datetime.date.fromisoformat(d): float(r) for d, r in zip(days, rain) if r is not None}
        if len(series) < 365:
            return {"available": False, "reason": "Open-Meteo returned too few days"}
        last30 = [series.get(end - datetime.timedelta(days=i)) for i in range(30)]
        missing30 = sum(v is None for v in last30)
        last30 = [v for v in last30 if v is not None]
        windows = []
        for back in (1, 2, 3):
            try:
                s = today.replace(year=today.year - back)
            except ValueError:                                     # 29 February
                s = today.replace(year=today.year - back, day=28)
            vals = [series.get(s + datetime.timedelta(days=i)) for i in range(180)]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 170:
                windows.append(sum(vals) * 180.0 / len(vals))
        if not windows:
            return {"available": False, "reason": "not enough history for the next-180-day climatology"}
        expected = sum(windows) / len(windows)
        return {"available": True, "source": "Open-Meteo historical weather API (ERA5)",
                "observed_last_30_days_mm": round(sum(last30), 1),
                "observed_window": f"{(end - datetime.timedelta(days=29)).isoformat()} to {end.isoformat()} "
                                   f"(the archive lags a few days)" + (f"; {missing30} days missing" if missing30 else ""),
                "expected_next_180_days_mm": round(expected, 1),
                "years_averaged": len(windows),
                "method": "mean rainfall over the same 180 calendar days in each of the last three years",
                "lat": round(lat, 3), "lon": round(lon, 3)}

    def forecast_rain_input(self, lat, lon):
        """(rain_mm for the deterioration forecaster, provenance dict) or (None, reason dict)."""
        r = self.rainfall(lat, lon)
        if not r.get("available"):
            return None, r
        mm = r["expected_next_180_days_mm"]
        prov = {"seasonal_rain_mm": round(min(mm, TRAINED_RAIN_MAX_MM), 1), "from": r["source"],
                "expected_next_180_days_mm": mm, "method": r["method"]}
        if mm > TRAINED_RAIN_MAX_MM:
            prov["clipped"] = (f"{mm:.0f} mm is above the {TRAINED_RAIN_MAX_MM:.0f} mm the forecaster was fitted on; "
                               f"held at {TRAINED_RAIN_MAX_MM:.0f} mm, so the forecast is a lower bound")
        return min(mm, TRAINED_RAIN_MAX_MM), prov

    # -- road ----------------------------------------------------------------------------------
    def road(self, lat, lon, radius_m=30):
        lat, lon = float(lat), float(lon)
        key = ("road", round(lat, 4), round(lon, 4), int(radius_m))
        return self._cached(key, ROAD_TTL_S, lambda: self._road(lat, lon, radius_m))

    def _road(self, lat, lon, radius_m):
        qlat, qlon = round(lat, 5), round(lon, 5)                  # ~1 m: enough to find the road, no more
        query = f'[out:json][timeout:8];way(around:{int(radius_m)},{qlat},{qlon})["highway"];out tags geom 20;'
        try:
            data = self._fetch(OVERPASS_URL, data=urllib.parse.urlencode({"data": query}).encode())
            ways = data.get("elements", [])
        except Exception as e:
            return {"available": False, "reason": f"Overpass not reachable: {str(e)[:120]}"}
        if data.get("remark"):                                     # overloaded server: partial or empty answer
            return {"available": False, "reason": f"Overpass: {str(data['remark'])[:120]}"}
        if not ways:
            return {"available": True, "found": False, "reason": f"no mapped road within {radius_m} m"}

        def rank(w):
            # distance to the way's own line, not to its centre (a long road's centre can be kilometres away);
            # a road within 3 m of the nearest one counts as tied, and then the bigger road class wins
            hw = (w.get("tags") or {}).get("highway", "")
            d = distance_to_line_m(lat, lon, w.get("geometry") or [])
            return (round(d / 3.0), MAJOR.index(hw) if hw in MAJOR else len(MAJOR))

        best = min(ways, key=rank)
        t = best.get("tags") or {}
        lanes = t.get("lanes")
        return {"available": True, "found": True, "source": "OpenStreetMap (Overpass API), (c) OpenStreetMap "
                                                             "contributors, ODbL",
                "highway": t.get("highway"), "name": t.get("name") or t.get("name:en"), "ref": t.get("ref"),
                "surface": t.get("surface"), "lanes": int(lanes) if str(lanes or "").isdigit() else None,
                "maxspeed": t.get("maxspeed"), "oneway": t.get("oneway") == "yes",
                "distance_m": round(distance_to_line_m(lat, lon, best.get("geometry") or []), 1),
                "ways_within_radius": len(ways)}


def distance_to_line_m(lat, lon, geometry):
    """Metres from a point to a polyline [{lat, lon}, ...] (equirectangular, fine at road scale)."""
    import math
    pts = [(g["lat"], g["lon"]) for g in geometry if "lat" in g and "lon" in g]
    if not pts:
        return float("inf")
    k = math.cos(math.radians(lat))
    xy = [((p[1] - lon) * 111320.0 * k, (p[0] - lat) * 110540.0) for p in pts]
    if len(xy) == 1:
        return math.hypot(*xy[0])
    best = float("inf")
    for (x1, y1), (x2, y2) in zip(xy, xy[1:]):
        dx, dy = x2 - x1, y2 - y1
        L = dx * dx + dy * dy
        t = 0.0 if L == 0 else max(0.0, min(1.0, -(x1 * dx + y1 * dy) / L))
        best = min(best, math.hypot(x1 + t * dx, y1 + t * dy))
    return best


_DEFAULT = None


def default():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RoadContext()
    return _DEFAULT
