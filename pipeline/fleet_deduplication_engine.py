"""
Fleet deduplication: spatial clustering of defect reports from multiple buses.

When several buses (or the same bus on repeat trips) photograph the same
pothole, this merges those reports into one persistent defect record instead
of double-counting it, using real Haversine great-circle distance against a
proximity threshold (default 8m, roughly GPS accuracy under tree cover).
"""
import math
import time

class FleetDeduplicationEngine:
    def __init__(self, proximity_threshold_meters=8.0):
        self.proximity_threshold_m = proximity_threshold_meters
        # Registry of persistent ground-truth defects: {defect_id: defect_record}
        self.defect_registry = {}
        self.next_defect_id = 1001
        self.total_reports_ingested = 0  # every ingest_fleet_detection call, matched or not

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

    def ingest_fleet_detection(self, bus_id, lat, lon, defect_class, severity_pci, area_m2, image_timestamp=None):
        """
        Ingests a detection from any bus in the fleet.
        If near an existing defect (<= threshold), merges it, updates confirmation count and timestamp.
        Otherwise, registers a new unique defect.
        """
        now = image_timestamp or time.time()
        self.total_reports_ingested += 1
        matched_id = None
        min_dist = float('inf')
        
        for d_id, record in self.defect_registry.items():
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
            rec["is_verified_hotspot"] = (rec["confirmation_count"] >= 2)
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
                from services.google_maps_service import google_maps_service
                geo = google_maps_service.reverse_geocode(lat, lon)
                formatted_address = geo.get("formatted_address")
                geocode_source = geo.get("provider", "geocoding_service")
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
        """Returns the deduplicated list for GIS map visualization."""
        return list(self.defect_registry.values())

    def get_deduplication_stats(self):
        """Real dedup efficiency from actual ingested/registered counts - not a fixed placeholder."""
        unique_defects = len(self.defect_registry)
        dedup_pct = 0.0
        if self.total_reports_ingested > 0:
            dedup_pct = round(100.0 * (1.0 - unique_defects / float(self.total_reports_ingested)), 1)
        return {
            "total_reports_ingested": self.total_reports_ingested,
            "unique_defects_registered": unique_defects,
            "deduplication_efficiency_pct": dedup_pct,
        }

