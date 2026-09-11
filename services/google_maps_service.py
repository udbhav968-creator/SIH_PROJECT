"""
Geocoding, routing, elevation and nearby-facility lookups.

Every lookup tries real data sources in order and says which one answered:

    geocode / reverse geocode : Google Geocoding API (if GOOGLE_MAPS_API_KEY is set)
                                -> OpenStreetMap Nominatim
                                -> a small offline table of city bounding boxes and
                                   well-known Bengaluru landmarks (approximate)
    directions                : Google Directions API -> OSRM public router
                                -> straight-line distance estimate (labelled as such)
    elevation / drainage      : Google Elevation API -> Open-Meteo elevation API
                                (Copernicus 90 m DEM) -> unavailable
    nearby facilities         : Google Places API -> OpenStreetMap Overpass API
                                -> unavailable

When every source fails the response says "UNAVAILABLE" with empty values -
nothing is filled in with made-up addresses, elevations or facilities.

Set ROAD_SHIELD_CONTACT to an email/URL you control; OpenStreetMap's usage
policies ask for a real contact in the User-Agent.
"""

import os
import json
import math
import time
import urllib.request
import urllib.parse
from typing import Dict, List, Any, Optional, Tuple


def _maps_links(lat: float, lon: float) -> Dict[str, str]:
    return {
        "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={lat},{lon}",
        "street_view_url": f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}",
    }


class GoogleMapsService:
    """Maps/GIS lookups with a Google -> OpenStreetMap -> offline fallback chain."""

    GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
    GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
    GOOGLE_ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"
    GOOGLE_PLACES_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    NOMINATIM_URL = "https://nominatim.openstreetmap.org"
    OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
    OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    # Offline fallback 1: coarse city bounding boxes. Only ever reported as an
    # approximate, city-level match - never as a street address.
    OFFLINE_CITY_BOXES = [
        {"city": "Bengaluru", "state": "Karnataka", "lat": (12.80, 13.15), "lon": (77.45, 77.80)},
        {"city": "Delhi NCR", "state": "Delhi / Uttar Pradesh / Haryana", "lat": (28.30, 28.90), "lon": (76.85, 77.60)},
        {"city": "Mumbai", "state": "Maharashtra", "lat": (18.85, 19.30), "lon": (72.75, 73.10)},
        {"city": "Pune", "state": "Maharashtra", "lat": (18.40, 18.65), "lon": (73.70, 74.00)},
        {"city": "Chennai", "state": "Tamil Nadu", "lat": (12.85, 13.25), "lon": (80.10, 80.35)},
        {"city": "Hyderabad", "state": "Telangana", "lat": (17.25, 17.60), "lon": (78.30, 78.65)},
        {"city": "Goa (Margao-Panaji)", "state": "Goa", "lat": (15.20, 15.60), "lon": (73.75, 74.10)},
    ]

    # Offline fallback 2: a few well-known Bengaluru landmarks for forward geocoding.
    OFFLINE_LANDMARKS = [
        {"name": "Silk Board Junction", "lat": 12.9176, "lng": 77.6238, "addr": "Central Silk Board Junction, Hosur Road, Bengaluru, Karnataka"},
        {"name": "MG Road", "lat": 12.9750, "lng": 77.6080, "addr": "Mahatma Gandhi Road, Bengaluru, Karnataka"},
        {"name": "Tin Factory", "lat": 12.9940, "lng": 77.6620, "addr": "Tin Factory, Old Madras Road, Bengaluru, Karnataka"},
        {"name": "Electronic City", "lat": 12.8450, "lng": 77.6630, "addr": "Electronic City, Hosur Road (NH-44), Bengaluru, Karnataka"},
        {"name": "Whitefield ITPL", "lat": 12.9850, "lng": 77.7310, "addr": "ITPL, Whitefield Main Road, Bengaluru, Karnataka"},
        {"name": "Majestic Bus Station", "lat": 12.9770, "lng": 77.5720, "addr": "Kempegowda Bus Station, Majestic, Bengaluru, Karnataka"},
        {"name": "Hebbal Flyover", "lat": 13.0350, "lng": 77.5970, "addr": "Hebbal Flyover, Bellary Road (NH-44), Bengaluru, Karnataka"},
    ]
    _GENERIC_QUERY_WORDS = {"road", "rd", "junction", "bengaluru", "bangalore", "nh", "the", "near", "karnataka", "india", "main", "cross"}

    FAILURE_CACHE_SECONDS = 300  # don't hammer a source that just failed

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or os.environ.get("GOOGLE_MAPS_API_KEY", "")).strip()
        contact = os.environ.get("ROAD_SHIELD_CONTACT", "SIH 2026 student project")
        self.user_agent = f"ROAD-SHIELD/3.0 ({contact})"
        self._cache: Dict[str, Tuple[float, Any]] = {}

    # ------------------------------------------------------------------ helpers
    def _get_json(self, url: str, timeout: float, data: Optional[bytes] = None) -> Any:
        req = urllib.request.Request(url, data=data, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _cached(self, key: str, compute):
        hit = self._cache.get(key)
        if hit is not None:
            stored_at, value = hit
            if value.get("status") not in ("UNAVAILABLE", "ZERO_RESULTS") or time.time() - stored_at < self.FAILURE_CACHE_SECONDS:
                return value
        value = compute()
        self._cache[key] = (time.time(), value)
        return value

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in meters."""
        r = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    # ------------------------------------------------------------ key / status
    def set_api_key(self, key: str) -> Dict[str, Any]:
        """Sets (or clears) the Google Maps API key at runtime and checks it."""
        self.api_key = key.strip() if key else ""
        self._cache.clear()
        valid, msg = self.validate_key()
        return {
            "api_key_configured": bool(self.api_key),
            "key_masked": f"{self.api_key[:6]}...{self.api_key[-4:]}" if len(self.api_key) > 10 else ("Configured" if self.api_key else "None"),
            "is_valid": valid,
            "status_message": msg,
            "active_engine": "Google Maps Platform" if self.api_key else "OpenStreetMap services (Nominatim / OSRM / Overpass) + Open-Meteo",
        }

    def validate_key(self) -> Tuple[bool, str]:
        if not self.api_key:
            return False, "No Google Maps API key set - using OpenStreetMap and Open-Meteo instead."
        try:
            params = urllib.parse.urlencode({"address": "Bengaluru", "key": self.api_key})
            data = self._get_json(f"{self.GOOGLE_GEOCODE_URL}?{params}", timeout=4.0)
            status = data.get("status")
            if status in ("OK", "ZERO_RESULTS"):
                return True, "Google Maps API key is active."
            if status == "REQUEST_DENIED":
                return False, f"Google Maps API key rejected: {data.get('error_message', 'Request denied')}"
            return False, f"Google Maps API returned status: {status}"
        except Exception as e:
            return False, f"Could not reach Google to verify the key: {e}"

    def get_service_status(self) -> Dict[str, Any]:
        has_key = bool(self.api_key and len(self.api_key) > 8)
        google_or = lambda fallback: "GOOGLE_MAPS_API" if has_key else fallback
        return {
            "api_key_configured": has_key,
            "active_provider": "Google Maps Platform" if has_key else "OpenStreetMap services + Open-Meteo (no Google key set)",
            "services": {
                "geocoding": google_or("OSM_NOMINATIM (offline landmark table if unreachable)"),
                "reverse_geocoding": google_or("OSM_NOMINATIM (offline city-level match if unreachable)"),
                "directions_routing": google_or("OSRM (straight-line estimate if unreachable)"),
                "pothole_avoidance": "Picks, among the router's alternative routes, the one passing fewest known defects",
                "elevation_drainage": google_or("OPEN_METEO_DEM (unavailable if unreachable)"),
                "nearby_facilities": google_or("OSM_OVERPASS (unavailable if unreachable)"),
                "street_view": "Google Maps links (thumbnail needs an API key)",
            },
            "tile_endpoints": self.get_tile_layers(),
            "timestamp_utc": int(time.time()),
        }

    @staticmethod
    def get_tile_layers() -> Dict[str, Dict[str, Any]]:
        """Map tile URL templates used by the dashboard's Leaflet map."""
        return {
            "google_roadmap": {"name": "Google Maps (Roadmap)", "url": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
                               "attribution": "&copy; Google Maps", "max_zoom": 20, "type": "vector_road"},
            "google_satellite": {"name": "Google Maps (Satellite)", "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                                 "attribution": "&copy; Google", "max_zoom": 20, "type": "highres_satellite"},
            "google_hybrid": {"name": "Google Maps (Hybrid)", "url": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                              "attribution": "&copy; Google", "max_zoom": 20, "type": "satellite_with_roads"},
            "google_terrain": {"name": "Google Maps (Terrain)", "url": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
                               "attribution": "&copy; Google", "max_zoom": 20, "type": "topographic_elevation"},
            "google_traffic": {"name": "Google Maps (Traffic)", "url": "https://mt1.google.com/vt/lyrs=m,traffic&x={x}&y={y}&z={z}",
                               "attribution": "&copy; Google", "max_zoom": 20, "type": "live_traffic"},
            "carto_dark": {"name": "CartoDB Dark", "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                           "attribution": "&copy; OpenStreetMap contributors &copy; CARTO", "max_zoom": 19, "type": "tactical_dark"},
        }

    # --------------------------------------------------------- reverse geocode
    def reverse_geocode(self, lat: float, lon: float) -> Dict[str, Any]:
        """Coordinates -> address, with the source that produced it."""
        return self._cached(f"rev_{round(lat, 5)}_{round(lon, 5)}", lambda: self._reverse_geocode(lat, lon))

    def _reverse_geocode(self, lat: float, lon: float) -> Dict[str, Any]:
        base = {"latitude": lat, "longitude": lon, **_maps_links(lat, lon)}

        if self.api_key:
            try:
                params = urllib.parse.urlencode({"latlng": f"{lat},{lon}", "key": self.api_key, "language": "en"})
                data = self._get_json(f"{self.GOOGLE_GEOCODE_URL}?{params}", timeout=4.0)
                if data.get("status") == "OK" and data.get("results"):
                    top = data["results"][0]
                    return {**base, "status": "OK", "provider": "Google Geocoding API",
                            "formatted_address": top.get("formatted_address", ""), "place_id": top.get("place_id", ""),
                            "address_components": top.get("address_components", [])}
            except Exception:
                pass

        try:
            params = urllib.parse.urlencode({"format": "json", "lat": lat, "lon": lon, "zoom": 18, "addressdetails": 1})
            data = self._get_json(f"{self.NOMINATIM_URL}/reverse?{params}", timeout=3.0)
            if data.get("display_name"):
                addr = data.get("address", {})
                return {**base, "status": "OK", "provider": "OpenStreetMap Nominatim",
                        "formatted_address": data["display_name"],
                        "road": addr.get("road"),
                        "locality": addr.get("suburb") or addr.get("neighbourhood"),
                        "city": addr.get("city") or addr.get("town") or addr.get("village"),
                        "state": addr.get("state"),
                        "postcode": addr.get("postcode")}
        except Exception:
            pass

        for box in self.OFFLINE_CITY_BOXES:
            if box["lat"][0] <= lat <= box["lat"][1] and box["lon"][0] <= lon <= box["lon"][1]:
                return {**base, "status": "APPROXIMATE", "provider": "Offline city bounding box (no geocoding service reachable)",
                        "formatted_address": f"Within {box['city']}, {box['state']} (approximate - street address unavailable offline)",
                        "city": box["city"], "state": box["state"]}

        return {**base, "status": "UNAVAILABLE", "provider": None, "formatted_address": None,
                "reason": "No geocoding service was reachable and the point is outside the offline city table."}

    # --------------------------------------------------------- forward geocode
    def geocode(self, query: str) -> Dict[str, Any]:
        """Search text -> candidate coordinates."""
        query_clean = (query or "").strip()
        if not query_clean:
            return {"status": "ZERO_RESULTS", "results": []}
        return self._cached(f"fwd_{query_clean.lower()}", lambda: self._geocode(query_clean))

    def _geocode(self, query: str) -> Dict[str, Any]:
        if self.api_key:
            try:
                params = urllib.parse.urlencode({"address": query, "key": self.api_key})
                data = self._get_json(f"{self.GOOGLE_GEOCODE_URL}?{params}", timeout=4.0)
                if data.get("status") == "OK" and data.get("results"):
                    return {"status": "OK", "provider": "Google Geocoding API", "results": [
                        {"formatted_address": r["formatted_address"],
                         "lat": r["geometry"]["location"]["lat"], "lng": r["geometry"]["location"]["lng"],
                         "place_id": r.get("place_id", ""), "viewport": r["geometry"].get("viewport", {}),
                         "google_maps_url": _maps_links(r["geometry"]["location"]["lat"], r["geometry"]["location"]["lng"])["google_maps_url"]}
                        for r in data["results"][:5]]}
            except Exception:
                pass

        try:
            params = urllib.parse.urlencode({"format": "json", "q": query, "limit": 5})
            data = self._get_json(f"{self.NOMINATIM_URL}/search?{params}", timeout=3.0)
            if data:
                return {"status": "OK", "provider": "OpenStreetMap Nominatim", "results": [
                    {"formatted_address": item["display_name"], "lat": float(item["lat"]), "lng": float(item["lon"]),
                     "place_id": str(item.get("place_id", "")),
                     "google_maps_url": _maps_links(item["lat"], item["lon"])["google_maps_url"]}
                    for item in data]}
            return {"status": "ZERO_RESULTS", "provider": "OpenStreetMap Nominatim", "results": []}
        except Exception:
            pass

        words = [w for w in query.lower().replace(",", " ").split() if len(w) >= 3 and w not in self._GENERIC_QUERY_WORDS]
        matches = [p for p in self.OFFLINE_LANDMARKS
                   if any(w in p["name"].lower() or w in p["addr"].lower() for w in words)]
        if not matches:
            return {"status": "ZERO_RESULTS", "provider": "Offline landmark table (no geocoding service reachable)", "results": []}
        return {"status": "OK", "approximate": True, "provider": "Offline landmark table (no geocoding service reachable)", "results": [
            {"formatted_address": m["addr"], "lat": m["lat"], "lng": m["lng"], "place_id": "",
             "google_maps_url": _maps_links(m["lat"], m["lng"])["google_maps_url"]} for m in matches]}

    # -------------------------------------------------------------- directions
    def get_directions(self, origin_lat: float, origin_lng: float,
                       dest_lat: float, dest_lng: float,
                       avoid_defects: bool = False,
                       known_defects: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Driving route between two points. With avoid_defects, the router is
        asked for alternative routes and the one passing the fewest known
        defects (within 40 m) is chosen - the geometry itself is never edited.
        """
        defects = known_defects or []
        key = f"dir_{round(origin_lat, 4)}_{round(origin_lng, 4)}_{round(dest_lat, 4)}_{round(dest_lng, 4)}_{avoid_defects}_{len(defects)}"
        return self._cached(key, lambda: self._directions(origin_lat, origin_lng, dest_lat, dest_lng, avoid_defects, defects))

    def _defects_near_route(self, polyline: List[List[float]], defects: List[Dict[str, Any]], radius_m: float = 40.0) -> int:
        hits = 0
        for d in defects:
            d_lat, d_lon = d.get("lat"), d.get("lon", d.get("lng"))
            if d_lat is None or d_lon is None:
                continue
            if any(self._haversine(p[0], p[1], d_lat, d_lon) <= radius_m for p in polyline):
                hits += 1
        return hits

    def _pick_route(self, candidates: List[Dict[str, Any]], defects: List[Dict[str, Any]], avoid: bool) -> Dict[str, Any]:
        for c in candidates:
            c["defects_within_40m"] = self._defects_near_route(c["polyline_coords"], defects) if defects else 0
        if avoid and defects:
            return min(candidates, key=lambda c: (c["defects_within_40m"], c["duration_minutes"]))
        return candidates[0]

    def _directions(self, o_lat, o_lng, d_lat, d_lng, avoid, defects) -> Dict[str, Any]:
        nav_url = f"https://www.google.com/maps/dir/?api=1&origin={o_lat},{o_lng}&destination={d_lat},{d_lng}&travelmode=driving"

        if self.api_key:
            try:
                params = urllib.parse.urlencode({"origin": f"{o_lat},{o_lng}", "destination": f"{d_lat},{d_lng}",
                                                 "mode": "driving", "alternatives": "true", "key": self.api_key})
                data = self._get_json(f"{self.GOOGLE_DIRECTIONS_URL}?{params}", timeout=4.0)
                if data.get("status") == "OK" and data.get("routes"):
                    candidates = []
                    for route in data["routes"]:
                        leg = route["legs"][0]
                        candidates.append({
                            "summary": route.get("summary", ""),
                            "distance_km": round(leg["distance"]["value"] / 1000.0, 2),
                            "duration_minutes": round(leg["duration"]["value"] / 60.0, 1),
                            "duration_text": leg["duration"]["text"],
                            "distance_text": leg["distance"]["text"],
                            "polyline_coords": self._decode_polyline(route["overview_polyline"]["points"]),
                            "turn_by_turn_steps": [
                                {"instruction": s.get("html_instructions", "").replace("<b>", "").replace("</b>", ""),
                                 "distance_text": s.get("distance", {}).get("text", ""),
                                 "duration_text": s.get("duration", {}).get("text", "")}
                                for s in leg.get("steps", [])],
                        })
                    best = self._pick_route(candidates, defects, avoid)
                    return {"status": "OK", "provider": "Google Directions API", "route_type": "road_network",
                            "alternatives_considered": len(candidates), "pothole_avoidance_mode": avoid,
                            "google_maps_nav_url": nav_url, **best}
            except Exception:
                pass

        try:
            url = (f"{self.OSRM_URL}/{o_lng},{o_lat};{d_lng},{d_lat}"
                   f"?overview=full&geometries=geojson&steps=true&alternatives=true")
            data = self._get_json(url, timeout=4.0)
            if data.get("code") == "Ok" and data.get("routes"):
                candidates = []
                for route in data["routes"]:
                    steps = []
                    for leg in route.get("legs", []):
                        for st in leg.get("steps", []):
                            m = st.get("maneuver", {})
                            name = st.get("name") or "unnamed road"
                            steps.append({"instruction": f"{m.get('type', 'continue').title()} {m.get('modifier', '')} onto {name}".replace("  ", " "),
                                          "distance_text": f"{round(st.get('distance', 0))} m",
                                          "duration_text": f"{round(st.get('duration', 0) / 60.0, 1)} min"})
                    dist_km = round(route["distance"] / 1000.0, 2)
                    dur_min = round(route["duration"] / 60.0, 1)
                    candidates.append({
                        "summary": "OSRM driving route",
                        "distance_km": dist_km, "duration_minutes": dur_min,
                        "duration_text": f"{dur_min} mins", "distance_text": f"{dist_km} km",
                        "polyline_coords": [[pt[1], pt[0]] for pt in route["geometry"]["coordinates"]],
                        "turn_by_turn_steps": steps,
                    })
                best = self._pick_route(candidates, defects, avoid)
                return {"status": "OK", "provider": "OSRM public router (OpenStreetMap road network)", "route_type": "road_network",
                        "alternatives_considered": len(candidates), "pothole_avoidance_mode": avoid,
                        "google_maps_nav_url": nav_url, **best}
        except Exception:
            pass

        straight_km = round(self._haversine(o_lat, o_lng, d_lat, d_lng) / 1000.0, 2)
        return {
            "status": "OK",
            "provider": "Straight-line estimate (no routing service reachable)",
            "route_type": "straight_line_estimate",
            "summary": "Straight line between the two points - not a drivable route",
            "distance_km": straight_km,
            "distance_text": f"{straight_km} km (straight line)",
            "duration_minutes": None,
            "duration_text": "unknown (no route)",
            "polyline_coords": [[o_lat, o_lng], [d_lat, d_lng]],
            "turn_by_turn_steps": [],
            "pothole_avoidance_mode": avoid,
            "defects_within_40m": None,
            "google_maps_nav_url": nav_url,
        }

    # --------------------------------------------------- elevation & drainage
    # Drainage risk uses the topographic position index: the point's elevation
    # minus the mean of 8 points on a ring ~250 m away. A point clearly lower
    # than its surroundings collects runoff. It's a screening heuristic, not a
    # hydrological model - it knows nothing about drains or kerbs.
    RING_RADIUS_M = 250.0

    def _ring_points(self, lat: float, lon: float) -> List[Tuple[float, float]]:
        pts = [(lat, lon)]
        dlat = self.RING_RADIUS_M / 111320.0
        dlon = self.RING_RADIUS_M / (111320.0 * max(0.1, math.cos(math.radians(lat))))
        for k in range(8):
            a = 2.0 * math.pi * k / 8.0
            pts.append((round(lat + dlat * math.sin(a), 6), round(lon + dlon * math.cos(a), 6)))
        return pts

    def _elevations_google(self, pts) -> Optional[List[float]]:
        locs = "|".join(f"{a},{b}" for a, b in pts)
        data = self._get_json(f"{self.GOOGLE_ELEVATION_URL}?{urllib.parse.urlencode({'locations': locs, 'key': self.api_key})}", timeout=4.0)
        if data.get("status") == "OK" and len(data.get("results", [])) == len(pts):
            return [r["elevation"] for r in data["results"]]
        return None

    def _elevations_open_meteo(self, pts) -> Optional[List[float]]:
        params = urllib.parse.urlencode({"latitude": ",".join(str(a) for a, _ in pts),
                                         "longitude": ",".join(str(b) for _, b in pts)})
        data = self._get_json(f"{self.OPEN_METEO_ELEVATION_URL}?{params}", timeout=3.0)
        elev = data.get("elevation")
        if isinstance(elev, list) and len(elev) == len(pts) and all(e is not None for e in elev):
            return [float(e) for e in elev]
        return None

    def get_elevation(self, lat: float, lon: float) -> Dict[str, Any]:
        """Elevation at a point plus a local-relief drainage screening."""
        return self._cached(f"elev_{round(lat, 5)}_{round(lon, 5)}", lambda: self._elevation(lat, lon))

    def _elevation(self, lat: float, lon: float) -> Dict[str, Any]:
        pts = self._ring_points(lat, lon)
        sources = []
        if self.api_key:
            sources.append(("Google Elevation API", self._elevations_google))
        sources.append(("Open-Meteo elevation API (Copernicus 90 m DEM)", self._elevations_open_meteo))

        for provider, fetch in sources:
            try:
                elevs = fetch(pts)
            except Exception:
                elevs = None
            if elevs:
                return self._drainage_from_relief(lat, lon, elevs, provider)

        return {"status": "UNAVAILABLE", "provider": None, "latitude": lat, "longitude": lon,
                "elevation_meters": None, "relative_relief_m": None,
                "waterlogging_vulnerability_pct": None, "drainage_risk_category": None,
                "civil_recommendation": None,
                "reason": "No elevation service was reachable."}

    def _drainage_from_relief(self, lat, lon, elevs: List[float], provider: str) -> Dict[str, Any]:
        center = elevs[0]
        ring_mean = sum(elevs[1:]) / len(elevs[1:])
        tpi = center - ring_mean  # negative = lower than surroundings
        if tpi <= -3.0:
            category = "HIGH_MONSOON_WATERLOGGING_RISK"
            rec = "Point sits in a local depression - check cross-drainage / culvert capacity (IRC:SP:42)."
        elif tpi <= -1.0:
            category = "MODERATE_WATER_ACCUMULATION"
            rec = "Slightly lower than surroundings - keep side drains clear and verify camber."
        else:
            category = "OPTIMAL_DRAINAGE"
            rec = "Not in a local depression - standard camber should shed water."
        # Simple, documented mapping from relief to a 0-100 screening score.
        vulnerability = round(max(0.0, min(100.0, 30.0 - tpi * 12.0)), 1)
        return {
            "status": "OK", "provider": provider, "latitude": lat, "longitude": lon,
            "elevation_meters": round(center, 1),
            "surrounding_mean_elevation_m": round(ring_mean, 1),
            "relative_relief_m": round(tpi, 2),
            "waterlogging_vulnerability_pct": vulnerability,
            "drainage_risk_category": category,
            "civil_recommendation": rec,
            "method": f"Topographic position index: point elevation minus mean of 8 points {int(self.RING_RADIUS_M)} m away (screening heuristic).",
        }

    # ------------------------------------------------------ nearby facilities
    FACILITY_TYPES = {"hospital": "hospital", "police": "police", "fire_station": "fire_station"}

    def find_nearby_civil_facilities(self, lat: float, lon: float, facility_type: str = "all", radius_m: int = 5000) -> Dict[str, Any]:
        """Real hospitals, police and fire stations near a point (Google Places or OpenStreetMap)."""
        res = self._cached(f"fac_{round(lat, 3)}_{round(lon, 3)}_{radius_m}", lambda: self._facilities(lat, lon, radius_m))
        facilities = res["facilities"]
        if facility_type != "all":
            facilities = [f for f in facilities if f["type"] == facility_type]
        return {**res, "facilities": facilities, "facility_count": len(facilities)}

    def _facility_entry(self, name, ftype, f_lat, f_lon, lat, lon):
        return {"name": name or f"Unnamed {ftype.replace('_', ' ')}", "type": ftype,
                "lat": f_lat, "lng": f_lon,
                "distance_km": round(self._haversine(lat, lon, f_lat, f_lon) / 1000.0, 2),
                "google_maps_url": _maps_links(f_lat, f_lon)["google_maps_url"]}

    def _facilities(self, lat: float, lon: float, radius_m: int) -> Dict[str, Any]:
        center = {"lat": lat, "lng": lon}
        if self.api_key:
            try:
                found = []
                for ftype in self.FACILITY_TYPES:
                    params = urllib.parse.urlencode({"location": f"{lat},{lon}", "radius": radius_m, "type": ftype, "key": self.api_key})
                    data = self._get_json(f"{self.GOOGLE_PLACES_URL}?{params}", timeout=4.0)
                    for p in data.get("results", [])[:5]:
                        loc = p["geometry"]["location"]
                        found.append(self._facility_entry(p.get("name"), ftype, loc["lat"], loc["lng"], lat, lon))
                found.sort(key=lambda f: f["distance_km"])
                return {"status": "OK", "provider": "Google Places API", "query_center": center, "facilities": found}
            except Exception:
                pass

        try:
            query = (f"[out:json][timeout:8];"
                     f"nwr[\"amenity\"~\"^(hospital|police|fire_station)$\"](around:{radius_m},{lat},{lon});"
                     f"out center 40;")
            data = self._get_json(self.OVERPASS_URL, timeout=8.0, data=urllib.parse.urlencode({"data": query}).encode())
            found = []
            for el in data.get("elements", []):
                f_lat = el.get("lat", el.get("center", {}).get("lat"))
                f_lon = el.get("lon", el.get("center", {}).get("lon"))
                if f_lat is None or f_lon is None:
                    continue
                tags = el.get("tags", {})
                found.append(self._facility_entry(tags.get("name"), tags.get("amenity"), f_lat, f_lon, lat, lon))
            found.sort(key=lambda f: f["distance_km"])
            return {"status": "OK", "provider": "OpenStreetMap Overpass API", "query_center": center, "facilities": found[:15]}
        except Exception:
            pass

        return {"status": "UNAVAILABLE", "provider": None, "query_center": center, "facilities": [],
                "reason": "Neither Google Places nor the OpenStreetMap Overpass API was reachable."}

    # ------------------------------------------------------------ street view
    def get_streetview_metadata(self, lat: float, lon: float) -> Dict[str, Any]:
        """Google Maps / Street View links for a point (thumbnail only with an API key)."""
        thumb = ""
        if self.api_key:
            thumb = (f"https://maps.googleapis.com/maps/api/streetview?size=600x300&location={lat},{lon}"
                     f"&heading=150&pitch=-10&key={self.api_key}")
        return {
            "status": "OK",
            "provider": "Google Maps links",
            "latitude": lat,
            "longitude": lon,
            "google_street_view_url": f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}&heading=-45&pitch=10&fov=80",
            "google_maps_search_url": f"https://www.google.com/maps/search/?api=1&query={lat},{lon}",
            "google_maps_embed_url": f"https://maps.google.com/maps?q={lat},{lon}&z=17&output=embed",
            "streetview_thumbnail_url": thumb,
        }

    # ---------------------------------------------------------------- utility
    @staticmethod
    def _decode_polyline(encoded: str) -> List[List[float]]:
        """Decodes Google's encoded polyline format into [lat, lng] pairs."""
        points, index, lat, lng = [], 0, 0, 0
        while index < len(encoded):
            for is_lng in (False, True):
                shift = result = 0
                while True:
                    b = ord(encoded[index]) - 63
                    index += 1
                    result |= (b & 0x1F) << shift
                    shift += 5
                    if b < 0x20:
                        break
                delta = ~(result >> 1) if (result & 1) else (result >> 1)
                if is_lng:
                    lng += delta
                else:
                    lat += delta
            points.append([round(lat * 1e-5, 6), round(lng * 1e-5, 6)])
        return points


google_maps_service = GoogleMapsService()
