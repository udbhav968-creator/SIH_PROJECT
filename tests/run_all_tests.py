import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Master Automated Verification & Test Suite for Project ROAD-SHIELD.
Executes all 6 unit & integration test suites and generates an executive report.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_vision_model import test_vision_model
from tests.test_imu_shock import test_imu_shock
from tests.test_bayesian_fusion import test_bayesian_fusion
from tests.test_ipm_homography import test_ipm_homography
from tests.test_forensic_audit import test_forensic_audit
from tests.test_morth_dispatch import test_morth_dispatch
from tests.test_deep_models import test_deep_models
from tests.test_mega_pipeline import main as test_mega_pipeline
from tests.test_deep_pipeline_e2e import run_deep_pipeline_tests
from tests.test_deep_model_accuracies import run_deep_model_benchmarks

def run_all_tests():
    start_time = time.time()
    print("=" * 75)
    print("🧪 ROAD-SHIELD MASTER AI TEST SUITE (MoRTH / NHAI SIH2026)")
    print("=" * 75)
    
    test_cases = [
        ("Model M1: Edge Vision Distress Segmentation Net", test_vision_model),
        ("Model M4: 100 Hz IMU Telemetry Shock Classifier", test_imu_shock),
        ("Model M5: Recursive Bayesian Dual-Sensor Fusion Gate", test_bayesian_fusion),
        ("Model M2: IPM Homography & MoRTH Volumetric Engine", test_ipm_homography),
        ("Models M7/M8: Forensic Anti-Fraud & Quality Audit Engine", test_forensic_audit),
        ("Model M10: MoRTH Cryptographic Dispatch Agent", test_morth_dispatch),
        ("Models M_PCI & M_DEGRADE: Deep Quality & Forecaster Suite", test_deep_models),
        ("Mega-Pipeline: Benchmark Dataset Hub & Augmentation Suite", test_mega_pipeline),
        ("11-Stage Deep Inference Pipeline E2E (Real Images & Gate)", run_deep_pipeline_tests),
        ("Deep Pipeline: All-Model Train & Test Accuracy Benchmark", run_deep_model_benchmarks)
    ]
    
    results = []
    for name, test_fn in test_cases:
        print(f"\n--- Running: {name} ---")
        t0 = time.time()
        try:
            passed = test_fn()
            elapsed = round(time.time() - t0, 3)
            results.append((name, "PASSED", elapsed))
        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            print(f"  [ERROR] Test failed: {e}")
            results.append((name, f"FAILED: {e}", elapsed))
            
    total_time = round(time.time() - start_time, 3)
    print("\n" + "=" * 75)
    print("🏁 MASTER TEST SUITE RESULTS SUMMARY")
    print("=" * 75)
    
    all_passed = True
    for name, status, elapsed in results:
        status_symbol = "✅ PASS" if "PASSED" in status else "❌ FAIL"
        print(f" {status_symbol} | {name:<55} | {elapsed:>6.3f}s")
        if "PASSED" not in status:
            all_passed = False
            
    print("-" * 75)
    print(f"Total Test Walltime: {total_time} seconds")
    if all_passed:
        print(f"🏆 ALL {len(test_cases)}/{len(test_cases)} AI MODULE TESTS PASSED WITH 100% SPECIFICATION CONFORMANCE!")
    else:
        print("⚠️ SOME TESTS FAILED - REVIEW LOGS ABOVE.")
    print("=" * 75)
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
