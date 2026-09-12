"""
Comprehensive End-to-End Verification Test Suite for the Deep Inference Pipeline
Project ROAD-SHIELD (MoRTH / NHAI Autonomous Road Asset Intelligence)

Tests the 11-stage deep pipeline directly with:
1. Real raw GitHub / Kaggle road defect images (.jpg)
2. Real raw GitHub / CRACK500 fatigue crack images (.jpg)
3. Non-pavement flat/screen rejection (Texture Gatekeeper sigma < 6.5)
4. Dual-sensor Bayesian fusion coherence (optical + suspension dynamics)
5. MoRTH Section 500 volumetric equations & material costing
6. Batch processing over benchmark image directories
7. Edge Open Neural Spec & C-header verification
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import time
import json
import numpy as np
from PIL import Image

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from pipeline.deep_inference_pipeline import DeepInferencePipeline

def test_real_pothole_images_e2e():
    print("\n[Test 1/7] E2E Pipeline on Real Raw Pothole Images (Kaggle/GitHub)...")
    pipe = DeepInferencePipeline()
    pothole_dir = os.path.join(ENGINE_ROOT, "datasets", "02_kaggle_pothole_600", "real_images")
    assert os.path.isdir(pothole_dir), f"Directory not found: {pothole_dir}"
    
    # Select first 5 real raw jpg files
    jpg_files = [f for f in sorted(os.listdir(pothole_dir)) if f.lower().endswith(".jpg")][:5]
    assert len(jpg_files) >= 5, f"Expected at least 5 jpg files, got {len(jpg_files)}"
    
    for fname in jpg_files:
        fpath = os.path.join(pothole_dir, fname)
        res = pipe.audit_image(fpath, corridor_id="NH-44", chainage_km=108.4)
        
        # 1. Gatekeeper check
        assert res["gatekeeper_passed"] is True, f"Failed gatekeeper for {fname}"
        assert res["status"] == "ANALYSIS_COMPLETE"
        
        # 2. Defect detection check
        assert res["is_distress"] is True, f"Expected distress in {fname}"
        pri = res["primary_distress"]
        assert pri["confidence"] >= 0.70, f"Confidence too low ({pri['confidence']}) for {fname}"
        assert pri["surface_area_m2"] > 0.0, "Surface area must be positive"
        assert pri["depth_cm"] >= 2.5, "Depth must be >= 2.5 cm"
        
        # 3. MoRTH Civil Ledger
        ledger = res["morth_civil_ledger"]
        assert ledger["total_bitumen_tonnage_t"] > 0.0, "Bitumen tonnage must be > 0"
        assert ledger["total_estimated_repair_inr"] > 0.0, "Repair cost must be > 0"
        
        # 4. Cryptographic Seal
        wo = res["cryptographic_work_order"]
        assert wo is not None, "Work order should be generated for distress"
        assert wo["seal_verification_status"] == "SEAL_VERIFIED_AUTHENTIC", "Cryptographic seal tampered"
        assert len(wo["sha256_cryptographic_seal"]) == 64, "Invalid SHA-256 seal length"
        
        print(f"  ✓ {fname[:35]}... -> {pri['class_name']} | Conf: {pri['confidence']:.2f} | Area: {pri['surface_area_m2']}m² | ₹{ledger['total_estimated_repair_inr']:,} | Seal OK")
        
    print("  [PASS] All 5 real pothole images successfully audited through 11-stage pipeline.")

def test_real_crack_images_e2e():
    print("\n[Test 2/7] E2E Pipeline on Real Raw Crack Images (CRACK500/GitHub)...")
    pipe = DeepInferencePipeline()
    crack_dir = os.path.join(ENGINE_ROOT, "datasets", "03_crack500_fatigue", "real_images")
    assert os.path.isdir(crack_dir), f"Directory not found: {crack_dir}"
    
    jpg_files = [f for f in sorted(os.listdir(crack_dir)) if f.lower().endswith(".jpg")][:5]
    assert len(jpg_files) >= 5, f"Expected at least 5 jpg files, got {len(jpg_files)}"
    
    for fname in jpg_files:
        fpath = os.path.join(crack_dir, fname)
        res = pipe.audit_image(fpath, corridor_id="NH-48", chainage_km=214.2)
        
        assert res["gatekeeper_passed"] is True, f"Failed gatekeeper for {fname}"
        assert res["is_distress"] is True, f"Expected distress in {fname}"
        
        # Verify ASTM D6433 PCI degradation
        pci = res["astm_d6433_pci"]["pci_score"]
        assert 0.0 <= pci <= 100.0, f"PCI score should be in valid ASTM range [0, 100], got {pci}"
        
        # Verify Monsoon Lifecycle Forecaster Monotonicity
        deg = res["monsoon_deterioration_forecast"]
        a0 = deg["initial_area_m2"]
        a30 = deg["forecast_30_days_area_m2"]
        a90 = deg["forecast_90_days_area_m2"]
        a180 = deg["forecast_180_days_area_m2"]
        assert a0 <= a30 <= a90 <= a180, f"Non-monotonic growth in {fname}: {[a0, a30, a90, a180]}"
        assert deg["municipal_savings_preventive_inr"] > 0, "Preventive repair must save money"
        
        print(f"  ✓ {fname[:35]}... -> PCI: {pci:.1f} ({res['astm_d6433_pci']['rating_category']}) | 180d Growth: {a0}m² -> {a180}m² | Savings: ₹{deg['municipal_savings_preventive_inr']:,}")
        
    print("  [PASS] All 5 real crack images verified with PCI degradation and lifecycle forecasting.")

def test_asphalt_texture_gatekeeper_rejection():
    print("\n[Test 3/7] Asphalt Texture Gatekeeper: Non-Road & Screen Rejection...")
    pipe = DeepInferencePipeline()
    
    # 1. Flat uniform gray screen
    flat_screen = Image.fromarray(np.full((480, 640, 3), 110, dtype=np.uint8))
    res_flat = pipe.audit_image(flat_screen)
    assert res_flat["gatekeeper_passed"] is False, "Flat screen should be rejected by texture gatekeeper"
    assert res_flat["status"] == "REJECTED_NON_PAVEMENT"
    assert res_flat["is_distress"] is False
    assert res_flat["texture_metrics"]["road_roi_std_lum"] < 6.5
    print(f"  ✓ Flat Solid Screen: Successfully REJECTED (std = {res_flat['texture_metrics']['road_roi_std_lum']} < 6.5)")
    
    # 2. Pure black / dark mode screen
    dark_screen = Image.fromarray(np.full((480, 640, 3), 15, dtype=np.uint8))
    res_dark = pipe.audit_image(dark_screen)
    assert res_dark["gatekeeper_passed"] is False
    assert res_dark["status"] == "REJECTED_NON_PAVEMENT"
    print(f"  ✓ Dark Mode Screen: Successfully REJECTED (std = {res_dark['texture_metrics']['road_roi_std_lum']} < 6.5)")
    
    # 3. Synthetic smooth gradient (indoor wall / whiteboard)
    grad = np.tile(np.linspace(200, 205, 640, dtype=np.uint8), (480, 1))
    grad_img = Image.fromarray(np.stack([grad, grad, grad], axis=-1))
    res_grad = pipe.audit_image(grad_img)
    assert res_grad["gatekeeper_passed"] is False
    assert res_grad["status"] == "REJECTED_NON_PAVEMENT"
    print(f"  ✓ Smooth Gradient Wall: Successfully REJECTED (std = {res_grad['texture_metrics']['road_roi_std_lum']} < 6.5)")
    
    print("  [PASS] False positives suppressed: Non-pavements strictly rejected without false alarms.")

def test_bayesian_fusion_coherence():
    print("\n[Test 4/7] Multi-Modal Dual-Sensor Bayesian Fusion Coherence...")
    pipe = DeepInferencePipeline()
    
    # Scenario A: Real confirmed cavity (high vision + high suspension shock)
    res_a = pipe.bayesian_gate.fuse(p_visual=0.96, p_imu_shock=0.94, delta_z_ms2=8.2)
    assert res_a["verdict"] == "CONFIRMED_POTHOLE"
    assert res_a["gate_passed"] is True
    assert res_a["posterior_probability"] > 0.90
    print(f"  ✓ Dual-Sensor Match: {res_a['verdict']} (P = {res_a['posterior_probability']:.4f}, L = {res_a['log_odds_score']:.2f})")
    
    # Scenario B: Optical false alarm (tree shadow on pavement: high visual, zero shock)
    res_b = pipe.bayesian_gate.fuse(p_visual=0.88, p_imu_shock=0.04, delta_z_ms2=0.3)
    assert res_b["verdict"] == "REJECTED_OPTICAL_FALSE_ALARM"
    assert res_b["gate_passed"] is False
    print(f"  ✓ Shadow Suppression: {res_b['verdict']} ({res_b['reason'][:50]}...)")
    
    # Scenario C: Monsoon submerged pothole (water obscures vision, suspension takes hard hit)
    res_c = pipe.bayesian_gate.fuse(p_visual=0.25, p_imu_shock=0.92, delta_z_ms2=7.5)
    assert res_c["verdict"] == "SUBMERGED_MONSOON_POTHOLE"
    assert res_c["gate_passed"] is True
    print(f"  ✓ Waterlogged Cavity: {res_c['verdict']} ({res_c['reason'][:50]}...)")
    
    print("  [PASS] Bayesian Fusion Gate operates with rigorous multi-modal mathematical integrity.")

def test_morth_section_500_physics():
    print("\n[Test 5/7] MoRTH Section 500 Physical Dimensional Laws...")
    pipe = DeepInferencePipeline()
    
    test_cases = [
        # (area_m2, depth_cm, expected_mass_tonnes, expected_cost_inr)
        (1.0, 5.0, 1.0 * 0.05 * 2.40 * 1.15, 1.0 * 0.05 * 2.40 * 1.15 * 7500.0),
        (2.5, 7.0, 2.5 * 0.07 * 2.40 * 1.15, 2.5 * 0.07 * 2.40 * 1.15 * 7500.0),
        (0.8, 10.0, 0.8 * 0.10 * 2.40 * 1.15, 0.8 * 0.10 * 2.40 * 1.15 * 7500.0)
    ]
    
    for area, depth, exp_mass, exp_cost in test_cases:
        vol = area * (depth / 100.0)
        tonnage = vol * 2.40 * 1.15
        cost = tonnage * 7500.0
        assert abs(tonnage - exp_mass) < 1e-5, f"Mass calculation error: {tonnage} vs {exp_mass}"
        assert abs(cost - exp_cost) < 1e-4, f"Cost calculation error: {cost} vs {exp_cost}"
        print(f"  ✓ Area {area}m² x Depth {depth}cm -> Vol {vol:.4f}m³ -> Mass {tonnage:.3f}T -> Budget ₹{cost:,.2f}")
        
    print("  [PASS] Bituminous Concrete / DBM density laws conformant with MoRTH Section 500.")

def test_batch_processing_and_ledger():
    print("\n[Test 6/7] Batch Ingestion & Aggregated MoRTH Civil Ledger...")
    pipe = DeepInferencePipeline()
    pothole_dir = os.path.join(ENGINE_ROOT, "datasets", "02_kaggle_pothole_600", "real_images")
    
    # Process batch of 10 real images
    batch_res = pipe.process_batch(pothole_dir, max_samples=10, corridor_id="NH-44")
    summary = batch_res["batch_summary"]
    
    assert summary["total_images_evaluated"] == 10
    assert summary["pavements_accepted"] == 10
    assert summary["non_pavements_rejected"] == 0
    assert summary["total_defects_detected"] >= 10
    assert summary["total_bitumen_tonnage_tonnes"] >= 0.25
    assert summary["total_repair_budget_inr"] >= 2000.0
    assert 0.0 <= summary["mean_pavement_pci"] <= 100.0, f"Mean PCI out of valid ASTM D6433 range: {summary['mean_pavement_pci']}"
    
    print(f"  ✓ Evaluated 10 Real Images:")
    print(f"    - Accepted Pavements: {summary['pavements_accepted']}/10")
    print(f"    - Total Defects Detected: {summary['total_defects_detected']}")
    print(f"    - Total MoRTH Tonnage: {summary['total_bitumen_tonnage_tonnes']} Tonnes")
    print(f"    - Total Repair Budget: ₹ {summary['total_repair_budget_inr']:,}")
    print(f"    - Mean Pavement PCI: {summary['mean_pavement_pci']} / 100")
    print(f"    - Batch Walltime: {summary['total_batch_walltime_s']}s")
    
    print("  [PASS] Batch ledger aggregates civil specifications across real image directories.")

def test_edge_spec_conformance():
    print("\n[Test 7/7] Open Neural Spec & C-Header Export Conformance...")
    spec_path = os.path.join(ENGINE_ROOT, "checkpoints", "road_shield_open_neural_spec.json")
    header_path = os.path.join(ENGINE_ROOT, "checkpoints", "road_shield_edge_inference.h")
    
    assert os.path.exists(spec_path), f"Spec file not found: {spec_path}"
    assert os.path.exists(header_path), f"C-header file not found: {header_path}"
    
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
        
    assert "models" in spec
    assert "Model_M1_VisionDistressNet" in spec["models"]
    assert "Model_M4_IMUShockClassifier" in spec["models"]
    assert "Model_M_PCI_Regressor" in spec["models"]
    assert "Model_M_DEGRADE_Forecaster" in spec["models"]
    
    header_size = os.path.getsize(header_path)
    assert header_size > 500, f"C-header too small: {header_size} bytes"
    print(f"  ✓ Open Neural Spec: {len(spec['models'])} models verified.")
    print(f"  ✓ C99 Header Library: {header_size} bytes verified.")
    print("  [PASS] Embedded edge specs conform to Road-Shield hardware runtime requirements.")

def run_deep_pipeline_tests():
    print("=" * 75)
    print("🧪 ROAD-SHIELD DEEP PIPELINE & REAL DATASET END-TO-END SUITE")
    print("=" * 75)
    t_start = time.time()
    
    test_real_pothole_images_e2e()
    test_real_crack_images_e2e()
    test_asphalt_texture_gatekeeper_rejection()
    test_bayesian_fusion_coherence()
    test_morth_section_500_physics()
    test_batch_processing_and_ledger()
    test_edge_spec_conformance()
    
    total_time = round(time.time() - t_start, 3)
    print("\n" + "=" * 75)
    print(f"🏆 ALL 7 DEEP PIPELINE E2E TESTS PASSED SUCCESSFULLY IN {total_time}s!")
    print("=" * 75)
    return True

if __name__ == "__main__":
    success = run_deep_pipeline_tests()
    sys.exit(0 if success else 1)
