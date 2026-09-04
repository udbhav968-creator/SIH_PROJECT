import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Unit tests for Model M5 (Recursive Bayesian Sensor Fusion Gate).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.bayesian_fusion_gate import BayesianFusionGate

def test_bayesian_fusion():
    print("Testing Model M5 (Recursive Bayesian Sensor Fusion Gate)...")
    gate = BayesianFusionGate(prior_pothole_prob=0.05, decision_threshold_log_odds=1.8)
    
    # Scenario 1: Tree shadow false alarm (High visual, Zero physical shock)
    res_shadow = gate.fuse(p_visual=0.92, p_imu_shock=0.04, delta_z_ms2=0.2)
    print(f"  [PASS] Shadow Rejection Test: Verdict = {res_shadow['verdict']}")
    assert res_shadow["verdict"] == "REJECTED_OPTICAL_FALSE_ALARM"
    assert res_shadow["gate_passed"] is False
    
    # Scenario 2: Confirmed Structural Pothole (High visual, High physical shock)
    res_pothole = gate.fuse(p_visual=0.92, p_imu_shock=0.96, delta_z_ms2=6.4)
    print(f"  [PASS] Confirmed Pothole Test: Verdict = {res_pothole['verdict']} (L={res_pothole['log_odds_score']:.2f})")
    assert res_pothole["verdict"] == "CONFIRMED_POTHOLE"
    assert res_pothole["gate_passed"] is True
    assert res_pothole["posterior_probability"] > 0.90
    
    # Scenario 3: Monsoon Submerged Pothole (Obscured visual, Violent chassis drop)
    res_submerged = gate.fuse(p_visual=0.35, p_imu_shock=0.98, delta_z_ms2=7.8)
    print(f"  [PASS] Submerged Pothole Test: Verdict = {res_submerged['verdict']}")
    assert res_submerged["verdict"] == "SUBMERGED_MONSOON_POTHOLE"
    assert res_submerged["gate_passed"] is True
    
    # Scenario 4: Normal smooth road (Low visual, Low shock)
    res_smooth = gate.fuse(p_visual=0.05, p_imu_shock=0.02, delta_z_ms2=0.1)
    print(f"  [PASS] Smooth Road Test: Verdict = {res_smooth['verdict']}")
    assert res_smooth["gate_passed"] is False
    
    print("  -> Model M5 Bayesian Fusion Gate: ALL TESTS PASSED.")
    return True

if __name__ == "__main__":
    test_bayesian_fusion()
