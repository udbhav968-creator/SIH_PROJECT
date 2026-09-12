"""
Test Suite: Google Maps Platform & GIS Intelligence APIs (SIH2026 / MoRTH)
Verifies:
  1. Service Status & Tile Layer Configuration
  2. Reverse Geocoding API (GPS -> Physical Address)
  3. Forward Geocoding API (Search Query -> Coordinates)
  4. Google Maps Directions API & Turn-by-Turn Maneuvers
  5. Pothole-Avoidance Detour Planner
  6. Google Elevation API & Waterlogging Risk Analysis
  7. Google Places API (Asphalt Plants & Emergency Depots)
  8. Google Street View 360° Panorama Deep-Linking
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.google_maps_service import google_maps_service

def test_google_maps_apis():
    print("=" * 70)
    print("🗺️ TEST SUITE: GOOGLE MAPS PLATFORM & REAL GIS APIS")
    print("=" * 70)
    
    t0 = time.time()
    
    # 1. Status & Tile Layers
    status = google_maps_service.get_service_status()
    print(f"  [1/8] Service Status: {status['google_maps_platform']} | Provider: {status['active_provider']}")
    assert status["google_maps_platform"] == "ONLINE"
    tiles = google_maps_service.get_tile_layers()
    assert "google_roadmap" in tiles
    assert "google_satellite" in tiles
    assert "google_hybrid" in tiles
    assert "google_terrain" in tiles
    assert "google_traffic" in tiles
    print("        ✓ Google Tile Layers (Roadmap, Satellite, Hybrid, Terrain, Traffic) Verified")
    
    # 2. Reverse Geocoding API
    geo = google_maps_service.reverse_geocode(12.9716, 77.5946)
    print(f"  [2/8] Reverse Geocoding (12.9716°N, 77.5946°E):")
    print(f"        Address: {geo.get('formatted_address', '')[:65]}...")
    print(f"        Maps URL: {geo.get('google_maps_url', '')}")
    assert geo["status"] == "OK"
    assert len(geo["formatted_address"]) > 5
    assert "google.com/maps" in geo["google_maps_url"]
    
    # 3. Forward Geocoding API
    forward = google_maps_service.geocode("Silk Board Junction")
    print(f"  [3/8] Forward Geocoding ('Silk Board Junction'):")
    assert forward["status"] == "OK"
    assert len(forward["results"]) > 0
    top_hit = forward["results"][0]
    print(f"        Resolved Lat/Lng: {top_hit['lat']:.4f}°N, {top_hit['lng']:.4f}°E")
    print(f"        Address: {top_hit['formatted_address'][:60]}...")
    assert abs(top_hit["lat"] - 12.9176) < 0.1
    
    # 4. Google Maps Directions API
    dirs = google_maps_service.get_directions(12.9725, 77.5955, 12.9780, 77.6020, avoid_defects=False)
    print(f"  [4/8] Google Maps Directions:")
    print(f"        Distance: {dirs['distance_km']} km | Duration: {dirs['duration_text']}")
    print(f"        Polyline Waypoints: {len(dirs['polyline_coords'])} coordinates")
    assert dirs["status"] == "OK"
    assert dirs["distance_km"] > 0
    assert len(dirs["polyline_coords"]) >= 2
    
    # 5. Pothole-Avoidance Detour Planner
    sample_defects = [{"lat": 12.9750, "lon": 77.5980, "defect_class": "D40 Severe Pothole"}]
    avoid_dirs = google_maps_service.get_directions(12.9725, 77.5955, 12.9780, 77.6020, avoid_defects=True, known_defects=sample_defects)
    print(f"  [5/8] Pothole-Avoidance Planner:")
    print(f"        Avoidance Mode: {avoid_dirs['pothole_avoidance_mode']}")
    print(f"        Alternative Distance: {avoid_dirs['distance_km']} km")
    assert avoid_dirs["pothole_avoidance_mode"] is True
    
    # 6. Google Maps Elevation & Waterlogging Vulnerability
    elev = google_maps_service.get_elevation(12.9716, 77.5946)
    print(f"  [6/8] Google Elevation & Drainage Solver:")
    print(f"        Elevation: {elev['elevation_meters']} m ASL")
    print(f"        Waterlogging Vulnerability: {elev['waterlogging_vulnerability_pct']}%")
    print(f"        Risk Category: {elev['drainage_risk_category']}")
    assert elev["status"] == "OK"
    assert elev["elevation_meters"] > 0
    assert 0 <= elev["waterlogging_vulnerability_pct"] <= 100
    
    # 7. Nearby Civil Facilities (Asphalt Plants & Emergency Depots)
    places = google_maps_service.find_nearby_civil_facilities(12.9716, 77.5946)
    print(f"  [7/8] Civil Facilities Nearby:")
    assert places["status"] == "OK"
    assert places["facility_count"] >= 3
    for fac in places["facilities"][:2]:
        print(f"        • {fac['name']} ({fac['type']}) - {fac['distance_km']} km away")
        
    # 8. Google Street View 360° Panorama Deep-Linking
    sv = google_maps_service.get_streetview_metadata(12.9716, 77.5946)
    print(f"  [8/8] Google Street View 360° Metadata:")
    print(f"        Panorama URL: {sv['google_street_view_url']}")
    print(f"        Embed URL: {sv['google_maps_embed_url']}")
    assert sv["status"] == "OK"
    assert "pano" in sv["google_street_view_url"]
    
    walltime = round(time.time() - t0, 3)
    print("-" * 70)
    print(f"🏆 ALL 8/8 GOOGLE MAPS PLATFORM TEST SUITES PASSED ({walltime}s)!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_google_maps_apis()
    sys.exit(0 if success else 1)
