"""
Fleet deduplication: spatial clustering of defect reports from multiple buses.

When several buses (or the same bus on repeat trips) photograph the same
pothole, this merges those reports into one persistent defect record instead
of double-counting it, using real Haversine great-circle distance against a
proximity threshold (default 8m, roughly GPS accuracy under tree cover).
Only reports of the same defect class are merged.
"""
import math
import threading
import time


def _finite(value, name, lo, hi):
    """A float in [lo, hi] (hi None = no upper bound); ValueError for NaN, infinity or out of range, so one bad
    report can never be stored and break every later read of the ledger."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(v) or v < lo or (hi is not None and v > hi):
        raise ValueError(f"{name} must be between {lo} and {hi if hi is not None else 'any positive value'}")
    return v

class FleetDeduplicationEngine:
    """
    Spatial deduplication of fleet defect reports.

    Pass a `store` (pipeline.defect_store.DefectStore) to make the ledger
    durable. Without one the behaviour is exactly as before - an in-memory
    registry that dies with the process - which is fine for tests and wrong
    for anything a municipality would rely on.

    With a store, the registry is loaded from disk on construction and every
    sighting is written back, merged or not. The raw sightings matter as much
    as the merged defects: they are the evidence that a pothole was reported
    five times and billed once.
    """

    def __init__(self, proximity_threshold_meters=8.0, store=None):
        self.proximity_threshold_m = proximity_threshold_meters
        self.store = store
        # Several request threads (fleet reports, sealed bus packets, video ingest) write here at once.
        self._lock = threading.RLock()
        # Called as fn(bus_id, result, at) after every ingested sighting, outside the lock (at = when the
        # sighting was recorded, which for a bus uploading a backlog is earlier than now): the repair workflow
        # reopens an order when a repaired defect is seen again, alerts check for new P1 defects.
        self.listeners = []
        # Registry of persistent ground-truth defects: {defect_id: defect_record}
        self.defect_registry = {}
        self.next_defect_id = 1001
        self.total_reports_ingested = 0  # every ingest_fleet_detection call, matched or not
        if store is not None:
            self.defect_registry = store.load_defects()
            self.next_defect_id = store.next_defect_id()
            self.total_reports_ingested = store.stats()["total_reports"]

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculates great-circle distance between two GPS points in meters.
        """
        R = 6371000.0  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def ingest_fleet_detection(self, bus_id, lat, lon, defect_class, severity_pci, area_m2, image_timestamp=None,
                               enrich_location=True, depth_cm=None, traffic_pcu_per_day=None):
        """
        Ingests a detection from any bus in the fleet.
        If near an existing defect (<= threshold), merges it, updates confirmation count and timestamp.
        Otherwise, registers a new unique defect. depth_cm and traffic_pcu_per_day
        are optional measurements kept with the defect for the Priority Index
        (models/priority_index.py); nothing is assumed when they are absent.
        enrich_location=False skips
        the (network) address/elevation lookup - used for startup fixtures so
        a cold start never waits on external services.
        """
        severity_pci, area_m2 = _finite(severity_pci, "severity_pci", 0, 100), _finite(area_m2, "area_m2", 0, None)
        depth_cm = None if depth_cm is None else _finite(depth_cm, "depth_cm", 0, 200)
        traffic_pcu_per_day = (None if traffic_pcu_per_day is None
                               else _finite(traffic_pcu_per_day, "traffic_pcu_per_day", 0, None))
        with self._lock:
            result = self._ingest(bus_id, lat, lon, defect_class, severity_pci, area_m2, image_timestamp,
                                  enrich_location, depth_cm, traffic_pcu_per_day)
        at = float(image_timestamp) if image_timestamp else time.time()
        for fn in list(self.listeners):
            try:
                fn(bus_id, result, at)
            except Exception as e:
                print(f"[dedup] listener failed: {e}")
        return result

    def _ingest(self, bus_id, lat, lon, defect_class, severity_pci, area_m2, image_timestamp,
                enrich_location, depth_cm, traffic_pcu_per_day):
        now = image_timestamp or time.time()
        self.total_reports_ingested += 1
        matched_id = None
        min_dist = float('inf')
        
        lat, lon = float(lat), float(lon)
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError(f"GPS out of range: lat={lat}, lon={lon}")
        for d_id, record in self.defect_registry.items():
            # Only the same kind of defect can be the same defect. A damaged
            # sign 5 m from a pothole is two problems for two crews; merging
            # them (as this used to, on distance alone) silently dropped one.
            if record.get("defect_class") != defect_class:
                continue
            dist = self.haversine_distance(lat, lon, record["lat"], record["lon"])
            if dist <= self.proximity_threshold_m and dist < min_dist:
                min_dist = dist
                matched_id = d_id
                
        if matched_id is not None:
            # Duplicate / Verification pass by another bus!
            rec = self.defect_registry[matched_id]
            rec["confirmation_count"] += 1
            if bus_id not in rec["reporting_buses"]:
                rec["reporting_buses"].append(bus_id)
            rec["last_seen_timestamp"] = now
            # Weighted moving average of severity & area
            rec["severity_pci"] = round((rec["severity_pci"] * 0.7) + (severity_pci * 0.3), 2)
            rec["area_m2"] = round(max(rec["area_m2"], area_m2), 2)
            if depth_cm is not None:
                rec["depth_cm"] = round(max(float(rec.get("depth_cm") or 0.0), float(depth_cm)), 1)
            if traffic_pcu_per_day is not None:
                rec["traffic_pcu_per_day"] = float(traffic_pcu_per_day)
            # Verified means seen by two different sources; the same bus (or the same photo uploaded twice)
            # repeating itself is not independent confirmation.
            rec["is_verified_hotspot"] = len(set(rec["reporting_buses"])) >= 2
            if self.store is not None:
                self.store.upsert_defect(rec)
                self.store.record_report(bus_id, lat, lon, defect_class, area_m2, severity_pci,
                                         matched_id, merged=True, distance_m=min_dist,
                                         reported_at=now)
            return {
                "action": "DEDUPLICATED_AND_UPDATED",
                "defect_id": matched_id,
                "confirmations": rec["confirmation_count"],
                "is_hotspot": rec["is_verified_hotspot"],
                "distance_to_centroid_m": round(min_dist, 2)
            }
        else:
            # Brand new unique physical defect
            new_id = f"DEF-BLR-{self.next_defect_id}"
            self.next_defect_id += 1

            # Map links are just coordinate-based URLs, always real. Address/
            # elevation/drainage come from a real geocoding lookup when one
            # succeeds; when it doesn't, we say so rather than filling in a
            # plausible-looking but made-up address or elevation.
            google_maps_url = f"https://www.google.com/maps/search/?api=1&query={round(lat, 6)},{round(lon, 6)}"
            street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={round(lat, 6)},{round(lon, 6)}"
            formatted_address = None
            elevation_m = None
            drainage_risk = None
            geocode_source = "unavailable"

            try:
                if not enrich_location:
                    raise LookupError("location enrichment skipped")
                from services.google_maps_service import google_maps_service
                geo = google_maps_service.reverse_geocode(lat, lon)
                formatted_address = geo.get("formatted_address")
                geocode_source = geo.get("provider") or "unavailable"
                elev = google_maps_service.get_elevation(lat, lon)
                elevation_m = elev.get("elevation_meters")
                drainage_risk = elev.get("drainage_risk_category")
            except Exception:
                pass

            self.defect_registry[new_id] = {
                "defect_id": new_id,
                "defect_class": defect_class,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "severity_pci": severity_pci,
                "area_m2": area_m2,
                "first_detected_by": bus_id,
                "reporting_buses": [bus_id],
                "confirmation_count": 1,
                "first_seen_timestamp": now,
                "last_seen_timestamp": now,
                "is_verified_hotspot": False,
                "address": formatted_address,
                "geocode_source": geocode_source,
                "google_maps_url": google_maps_url,
                "street_view_url": street_view_url,
                "elevation_m": elevation_m,
                "drainage_risk": drainage_risk
            }
            if depth_cm is not None:
                self.defect_registry[new_id]["depth_cm"] = round(float(depth_cm), 1)
            if traffic_pcu_per_day is not None:
                self.defect_registry[new_id]["traffic_pcu_per_day"] = float(traffic_pcu_per_day)
            if self.store is not None:
                self.store.upsert_defect(self.defect_registry[new_id])
                self.store.record_report(bus_id, lat, lon, defect_class, area_m2, severity_pci,
                                         new_id, merged=False, distance_m=0.0, reported_at=now)
            return {
                "action": "REGISTERED_NEW_DEFECT",
                "defect_id": new_id,
                "confirmations": 1,
                "is_hotspot": False,
                "distance_to_centroid_m": 0.0,
                "address": formatted_address,
                "google_maps_url": google_maps_url,
                "street_view_url": street_view_url
            }

    def get_all_deduplicated_defects(self):
        """Returns the deduplicated list for GIS map visualization (copies, safe to read while buses report)."""
        with self._lock:
            return [dict(r, reporting_buses=list(r.get("reporting_buses", []))) for r in self.defect_registry.values()]

    def get_deduplication_stats(self):
        """Real dedup efficiency from actual ingested/registered counts - not a fixed placeholder."""
        unique_defects = len(self.defect_registry)
        dedup_pct = 0.0
        if self.total_reports_ingested > 0:
            dedup_pct = round(100.0 * (1.0 - unique_defects / float(self.total_reports_ingested)), 1)
        out = {
            "total_reports_ingested": self.total_reports_ingested,
            "unique_defects_registered": unique_defects,
            "deduplication_efficiency_pct": dedup_pct,
            "storage": "sqlite" if self.store is not None else "in-memory (lost on restart)",
        }
        if self.store is not None:
            # Counts queried from the tables, which cannot drift from the rows
            # the way an incremented counter can.
            out["persisted"] = self.store.stats()
        return out

