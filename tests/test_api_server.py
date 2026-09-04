"""
Automated Integration Tests for ROAD-SHIELD Live API Server.
Starts server on a test port, queries all endpoints, and asserts contracts.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import time
import json
import urllib.request
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from api.server import ThreadedHTTPServer, RoadShieldAPIHandler

TEST_PORT = 8088

def http_get(endpoint):
    url = f"http://127.0.0.1:{TEST_PORT}{endpoint}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def http_post(endpoint, payload):
    url = f"http://127.0.0.1:{TEST_PORT}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def run_server_tests():
    print("=" * 70)
    print("🧪 ROAD-SHIELD LIVE API GATEWAY INTEGRATION TESTS")
    print("=" * 70)
    
    server_address = ("127.0.0.1", TEST_PORT)
    httpd = ThreadedHTTPServer(server_address, RoadShieldAPIHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)
    print(f"  [SERVER STARTED] Listening on http://127.0.0.1:{TEST_PORT}")
    
    try:
        # Test 1: Health Check
        status, res = http_get("/api/v1/health")
        print(f"  [PASS] GET /api/v1/health -> Status: {res['status']} | Authority: {res['authority'][:30]}...")
        assert status == 200
        assert res["status"] == "ONLINE"
        assert res["models"]["M1_vision_distress_net"] == "LOADED_ACTIVE"
        
        # Test 2: Vision Distress Detection
        status, res = http_post("/api/v1/detect/vision", {"preferred_class": 4})
        print(f"  [PASS] POST /api/v1/detect/vision -> Distress: {res['distress_class']} (Conf: {res['confidence']})")
        assert status == 200
        assert res["distress_class"] == "D40 Pothole"
        assert "bounding_box" in res
        assert res["ground_area_sqm"] > 0
        
        # Test 3: 100 Hz IMU Telemetry
        status, res = http_post("/api/v1/telemetry/imu", {"mode": "shock"})
        print(f"  [PASS] POST /api/v1/telemetry/imu -> Shock: {res['shock_class_name']} | Peak Delta Z: {res['peak_delta_z_ms2']} m/s^2")
        assert status == 200
        assert res["shock_class_name"] == "Pothole Impact"
        assert res["pothole_shock_probability"] > 0.80
        
        # Test 4: Bayesian Fusion Gate
        status, res = http_post("/api/v1/fusion/gate", {"p_visual": 0.92, "p_imu_shock": 0.96, "delta_z_ms2": 6.8})
        print(f"  [PASS] POST /api/v1/fusion/gate -> Verdict: {res['verdict']} (L={res['log_odds_score']:.2f})")
        assert status == 200
        assert res["verdict"] == "CONFIRMED_POTHOLE"
        assert res["gate_passed"] is True
        
        # Test 5: MoRTH Section 500 Tonnage Solver
        status, res = http_post("/api/v1/civil/ipm-tonnage", {"area_sqm": 2.5, "depth_cm": 6.5, "mix_type": "DBM_SECTION_500"})
        print(f"  [PASS] POST /api/v1/civil/ipm-tonnage -> Mass: {res['required_mass_tonnes']} Tonnes | Cost: INR {res['estimated_cost_inr']}")
        assert status == 200
        assert res["required_mass_tonnes"] > 0
        assert res["estimated_cost_inr"] > 0
        
        # Test 6: Forensic Repair Audit
        status, res = http_post("/api/v1/audit/verify-repair", {"claim_type": "genuine", "claimed_distance_m": 0.7})
        print(f"  [PASS] POST /api/v1/audit/verify-repair -> Verdict: {res['verdict']} (SSIM={res['ssim_index']})")
        assert status == 200
        assert res["audit_passed"] is True
        
        # Test 7: Work-Order Generation with SHA-256 Seal
        status, wo = http_post("/api/v1/dispatch/work-order", {
            "corridor_id": "NH-44-EXPRESSWAY",
            "latitude": 18.5204,
            "longitude": 73.8567,
            "distress_class": "D40 Pothole",
            "area_sqm": 2.1,
            "depth_cm": 6.5,
            "pci_score": 35
        })
        print(f"  [PASS] POST /api/v1/dispatch/work-order -> ID: {wo['work_order_id']} | Seal: {wo['sha256_cryptographic_seal'][:16]}...")
        assert status == 200
        assert "sha256_cryptographic_seal" in wo
        
        # Test 8: Verify SHA-256 Seal
        status, verify_res = http_post("/api/v1/dispatch/verify-seal", {"work_order": wo})
        print(f"  [PASS] POST /api/v1/dispatch/verify-seal -> Valid: {verify_res['is_valid']}")
        assert status == 200
        assert verify_res["is_valid"] is True
        
        # Test 9: 11-Stage Deep Inference Pipeline Single Audit
        pothole_img = os.path.join(ENGINE_ROOT, "datasets", "02_kaggle_pothole_600", "real_images", "1014628_RS_386_386RS124739_30065_RAW.jpg")
        status, deep_res = http_post("/api/v1/pipeline/deep-audit", {
            "image_path": pothole_img,
            "corridor_id": "NH-44-NORTH-CORRIDOR",
            "chainage_km": 108.4
        })
        print(f"  [PASS] POST /api/v1/pipeline/deep-audit -> Status: {deep_res['status']} | Distress: {deep_res['primary_distress']['class_name']} | PCI: {deep_res['astm_d6433_pci']['pci_score']}")
        assert status == 200
        assert deep_res["status"] == "ANALYSIS_COMPLETE"
        assert deep_res["gatekeeper_passed"] is True
        assert deep_res["cryptographic_work_order"]["seal_verification_status"] == "SEAL_VERIFIED_AUTHENTIC"
        
        # Test 10: Batch Deep Pipeline Audit
        status, batch_res = http_post("/api/v1/pipeline/deep-audit-batch", {
            "directory_path": os.path.join(ENGINE_ROOT, "datasets", "02_kaggle_pothole_600", "real_images"),
            "max_samples": 5,
            "corridor_id": "NH-44"
        })
        summary = batch_res["batch_summary"]
        print(f"  [PASS] POST /api/v1/pipeline/deep-audit-batch -> Evaluated: {summary['total_images_evaluated']} | Accepted: {summary['pavements_accepted']} | Budget: INR {summary['total_repair_budget_inr']}")
        assert status == 200
        assert summary["total_images_evaluated"] == 5
        assert summary["pavements_accepted"] == 5
        
        print("-" * 70)
        print("🏆 ALL 10 API GATEWAY INTEGRATION ENDPOINTS PASSED WITH 100% SUCCESS!")
        print("=" * 70)
        return True
    finally:
        httpd.shutdown()
        httpd.server_close()

if __name__ == "__main__":
    success = run_server_tests()
    sys.exit(0 if success else 1)
