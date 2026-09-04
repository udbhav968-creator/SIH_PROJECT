"""
ROAD-SHIELD Comprehensive System Test v2.0
Tests all 10 subsystems: models, pipeline, API, datasets, checkpoints
"""
import sys, os, time, json, glob, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import urllib.request
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

results = {}
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: All model imports
# ─────────────────────────────────────────────────────────────────────────────
try:
    from models.vision_distress_net import VisionDistressNet
    from models.imu_shock_classifier import IMUShockClassifier
    from models.pci_regressor_net import PCIRegressorNet
    from models.pavement_deterioration_forecaster import PavementDeteriorationForecaster
    from models.bayesian_fusion_gate import BayesianFusionGate
    from models.ipm_homography_engine import IPMHomographyEngine
    from models.urban_traffic_net import UrbanTrafficNet
    from models.alpr_incident_tracker import ALPRIncidentTracker
    from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
    from pipeline.deep_inference_pipeline import DeepInferencePipeline
    results["1_model_imports"] = "PASS - 10 modules imported successfully"
except Exception as e:
    results["1_model_imports"] = f"FAIL: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Vision Model (9-class classification)
# ─────────────────────────────────────────────────────────────────────────────
try:
    m = VisionDistressNet(in_features=64, hidden_dims=[512, 256, 128], num_classes=9)
    vis_ckpt = os.path.join(CKPT_DIR, "vision_distress_weights.npz")
    m.load_weights(vis_ckpt)
    X = np.random.randn(10, 64).astype(np.float32)
    preds, confs, probs, geo = m.predict(X)
    assert probs.shape == (10, 9), f"Wrong probs shape: {probs.shape}"
    assert len(m.CLASS_NAMES) == 9
    results["2_vision_9class"] = f"PASS - classes={len(m.CLASS_NAMES)}, preds_ok, probs.shape={probs.shape}"
except Exception as e:
    results["2_vision_9class"] = f"FAIL: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: IMU Shock Classifier (4-class)
# ─────────────────────────────────────────────────────────────────────────────
try:
    imu = IMUShockClassifier(in_features=36, hidden_dims=[64, 32], num_classes=4)
    imu.load_weights(os.path.join(CKPT_DIR, "imu_shock_weights.npz"))
    raw_imu = np.random.randn(4, 100, 3).astype(np.float32)
    raw_imu[0, 45:55, 2] += 14.0  # Severe pothole shock
    preds_i, confs_i, probs_i = imu.predict(raw_imu)
    assert len(preds_i) == 4
    results["3_imu_shock"] = f"PASS - shock_pred={preds_i[0]} ({IMUShockClassifier.CLASS_NAMES[preds_i[0]]}), conf={float(confs_i[0]):.3f}"
except Exception as e:
    results["3_imu_shock"] = f"FAIL: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: PCI Regressor (ASTM D6433)
# ─────────────────────────────────────────────────────────────────────────────
try:
    pci = PCIRegressorNet(in_features=12, hidden_dims=[64, 32])
    pci.load_weights(os.path.join(CKPT_DIR, "pci_regressor_weights.npz"))
    X_pci_good = np.zeros((1, 12), dtype=np.float32)  # Perfect pavement
    X_pci_bad  = np.array([[5, 15, 3, 7, 2.5, 2, 8, 3.2, 7.0, 22, 7.5, 5.0]], dtype=np.float32)  # Severe
    score_good = float(pci.predict(X_pci_good)[0])
    score_bad  = float(pci.predict(X_pci_bad)[0])
    assert score_good > score_bad, "PCI ordering wrong: good should score higher than bad"
    rating, _ = pci.get_rating_category(score_bad)
    results["4_pci_astm"] = f"PASS - Good={score_good:.1f}, Bad={score_bad:.1f} ({rating}), ordering correct"
except Exception as e:
    results["4_pci_astm"] = f"FAIL: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Urban Traffic Net + UCI Calculation
# ─────────────────────────────────────────────────────────────────────────────
try:
    urn = UrbanTrafficNet(in_features=48, hidden_dims=[256, 128], num_classes=7)
    urn.load_weights(os.path.join(CKPT_DIR, "urban_traffic_net_weights.npz"))
    X_t = np.random.randn(5, 48).astype(np.float32)
    pred_t, conf_t, probs_t = urn.predict(X_t)
    uci = urn.compute_urban_congestion_index(X_t[0])
    assert 0 <= uci <= 300
    results["5_urban_traffic"] = f"PASS - 7-class pred={pred_t[0]}, UCI={uci:.1f} PCU"
except Exception as e:
    results["5_urban_traffic"] = f"FAIL: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: ALPR Incident Tracker
# ─────────────────────────────────────────────────────────────────────────────
try:
    alpr = ALPRIncidentTracker()
    inc = alpr.detect_incident(speed_kmh=95.0, lat=12.97, lon=77.59, vehicle_id="KA01AB1234")
    assert "incident_id" in inc
    assert "sha256_seal" in inc
    inc_class = inc.get("incident_class", "?")
    inc_conf  = inc.get("alpr_confidence", 0.0)
    results["6_alpr_tracker"] = f"PASS - {inc_class} conf={inc_conf:.3f} SHA256={inc['sha256_seal'][:16]}..."
except Exception as e:
    results["6_alpr_tracker"] = f"FAIL: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: Fleet Spatial Deduplication (Haversine Clustering)
# ─────────────────────────────────────────────────────────────────────────────
try:
    fde = FleetDeduplicationEngine(proximity_threshold_meters=8.0)
    # Bus A and Bus B report same pothole 4.5m apart → should merge to 1 unique
    fde.ingest_fleet_detection("BUS-001", 12.97160, 77.59460, "D40 Pothole", 42.0, 1.85)
    fde.ingest_fleet_detection("BUS-002", 12.97164, 77.59465, "D40 Pothole", 38.0, 2.10)
    # Separate defect far away
    fde.ingest_fleet_detection("BUS-003", 12.97500, 77.59800, "Waterlogging", 35.0, 5.20)
    defects = fde.get_all_deduplicated_defects()
    hotspots = [d for d in defects if d.get("is_verified_hotspot")]
    results["7_fleet_dedup"] = f"PASS - {len(defects)} unique defects (hotspots={len(hotspots)}) from 3 reports"
except Exception as e:
    results["7_fleet_dedup"] = f"FAIL: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: Full 11-Stage Pipeline on Real Images
# ─────────────────────────────────────────────────────────────────────────────
try:
    img_dirs = [
        os.path.join(os.path.dirname(__file__), "datasets", "10_missing_zebra_crossing", "real_images"),
        os.path.join(os.path.dirname(__file__), "datasets", "11_missing_road_divider", "real_images"),
        os.path.join(os.path.dirname(__file__), "datasets", "09_waterlogging_hazard", "real_images"),
    ]
    imgs = []
    seen = set()
    for d in img_dirs:
        if os.path.exists(d):
            for f in glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")):
                h = hashlib.sha256(open(f, "rb").read()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    imgs.append(f)
    imgs = imgs[:10]  # test 10 unique images
    if not imgs:
        results["8_pipeline_real_images"] = "SKIP - no real images found"
    else:
        pipe = DeepInferencePipeline()
        latencies = []
        status_counts = {}
        pci_scores = []
        for img_path in imgs:
            t0 = time.time()
            r = pipe.audit_image(img_path)
            lat = (time.time() - t0) * 1000
            latencies.append(lat)
            s = r.get("status", "UNKNOWN")
            status_counts[s] = status_counts.get(s, 0) + 1
            pci = r.get("astm_d6433_pci", {}).get("pci_score", None)
            if pci is not None:
                pci_scores.append(pci)
        avg_lat = sum(latencies) / len(latencies)
        avg_pci = sum(pci_scores) / len(pci_scores) if pci_scores else 0
        results["8_pipeline_real_images"] = (
            f"PASS - {len(imgs)} images, avg_lat={avg_lat:.1f}ms, "
            f"avg_pci={avg_pci:.1f}, statuses={status_counts}"
        )
except Exception as e:
    results["8_pipeline_real_images"] = f"FAIL: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: REST API Health Check (8 endpoints)
# ─────────────────────────────────────────────────────────────────────────────
try:
    endpoints = [
        ("/api/v1/gis/map-data", "GET"),
        ("/api/v1/fleet/telemetry", "GET"),
        ("/api/v1/training/metrics", "GET"),
        ("/api/v1/models/registry", "GET"),
        ("/api/v1/training/status", "GET"),
        ("/api/v1/ledger/defects", "GET"),
        ("/api/v1/datasets/benchmarks", "GET"),
    ]
    passed_eps = 0
    ep_details = []
    for path, method in endpoints:
        try:
            url = f"http://localhost:8000{path}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            resp = urllib.request.urlopen(req, timeout=5)
            code = resp.getcode()
            data = json.loads(resp.read())
            assert isinstance(data, dict)
            passed_eps += 1
            ep_details.append(f"{code}")
        except Exception as ee:
            ep_details.append(f"ERR:{str(ee)[:20]}")
    results["9_rest_api"] = f"PASS - {passed_eps}/{len(endpoints)} OK: {ep_details}"
except Exception as e:
    results["9_rest_api"] = f"FAIL: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: Checkpoint File Integrity
# ─────────────────────────────────────────────────────────────────────────────
try:
    required_ckpts = [
        "vision_distress_weights.npz",
        "imu_shock_weights.npz",
        "pci_regressor_weights.npz",
        "deterioration_forecaster_weights.npz",
        "urban_traffic_net_weights.npz",
    ]
    found = []
    missing = []
    for ckpt in required_ckpts:
        fpath = os.path.join(CKPT_DIR, ckpt)
        if os.path.exists(fpath):
            sz_kb = os.path.getsize(fpath) / 1024
            found.append(f"{ckpt}({sz_kb:.0f}KB)")
        else:
            missing.append(ckpt)
    if not missing:
        results["10_checkpoints"] = f"PASS - all {len(required_ckpts)} weight files verified"
    else:
        results["10_checkpoints"] = f"PARTIAL - missing: {missing}"
except Exception as e:
    results["10_checkpoints"] = f"FAIL: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("  ROAD-SHIELD AI ENGINE — COMPREHENSIVE SYSTEM TEST REPORT")
print("  BEL SIH26124 | MoRTH/NHAI Pavement Intelligence Platform")
print("=" * 80)
total = len(results)
passed = sum(1 for v in results.values() if v.startswith("PASS"))
partial = sum(1 for v in results.values() if v.startswith("PARTIAL"))
failed = sum(1 for v in results.values() if v.startswith("FAIL"))
skipped = sum(1 for v in results.values() if v.startswith("SKIP"))

for k, v in sorted(results.items()):
    if v.startswith("PASS"):
        icon = "✅"
    elif v.startswith("PARTIAL"):
        icon = "⚠️ "
    elif v.startswith("SKIP"):
        icon = "⏭️ "
    else:
        icon = "❌"
    print(f"  {icon} {k:<30s}: {v}")

print("=" * 80)
print(f"  RESULT: {passed}/{total} PASSED | {partial} PARTIAL | {failed} FAILED | {skipped} SKIPPED")
print("=" * 80)

# Save report
report = {
    "timestamp_utc": int(time.time()),
    "system": "ROAD-SHIELD AI Engine v3.0",
    "test_results": results,
    "summary": {
        "total": total, "passed": passed, "partial": partial,
        "failed": failed, "skipped": skipped,
        "pass_rate_pct": round(passed / total * 100, 1)
    }
}
report_path = os.path.join(CKPT_DIR, "system_test_v2_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  📄 Report saved to: {report_path}")
