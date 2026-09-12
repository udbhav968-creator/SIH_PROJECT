import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Unit tests for Model M2 (IPM Homography & MoRTH Volumetric Engine).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ipm_homography_engine import IPMHomographyEngine

def test_ipm_homography():
    print("Testing Model M2 (IPM Homography & MoRTH Volumetric Engine)...")
    engine = IPMHomographyEngine(camera_height_m=1.45, pitch_deg=18.4)
    
    # Test 1: Ground projection logic
    # Center pixel should map to lateral X = 0
    x, y = engine.pixel_to_ground(320.0, 300.0)
    print(f"  [PASS] Center Pixel Projection: Ground X = {x:.4f}m, Y = {y:.2f}m")
    assert abs(x) < 1e-4, f"Center pixel should have X=0, got {x}"
    assert y > 0.0, f"Forward distance Y must be positive, got {y}"
    
    # Test 2: Surface area calculation
    area_sqm = engine.calculate_surface_area_sqm(u_min=280, v_min=280, width_px=80, height_px=60)
    print(f"  [PASS] Pothole Ground Area: {area_sqm:.3f} m^2")
    assert 0.10 <= area_sqm <= 5.0, f"Area {area_sqm} m^2 outside expected physical bounds"
    
    # Test 3: MoRTH Section 500 Material Calculation Verification
    # Expected: Mass = 2.0 m^2 * 0.065 m * 2.40 T/m^3 * 1.15 = 0.3588 Tonnes
    # Cost = 0.3588 * 7500 = 2691.00 INR
    res_calc = engine.compute_asphalt_procurement(
        area_sqm=2.0, depth_cm=6.5, mix_type="DBM_SECTION_500", compaction_margin=1.15
    )
    print(f"  [PASS] MoRTH Tonnage Calculation: {res_calc['required_mass_tonnes']} Tonnes | Cost: INR {res_calc['estimated_cost_inr']}")
    assert abs(res_calc["required_mass_tonnes"] - 0.3588) < 1e-3, f"Expected 0.3588, got {res_calc['required_mass_tonnes']}"
    assert abs(res_calc["estimated_cost_inr"] - 2691.00) < 1.0, f"Expected 2691.00, got {res_calc['estimated_cost_inr']}"
    
    print("  -> Model M2 IPM Homography Engine: ALL TESTS PASSED.")
    return True

if __name__ == "__main__":
    test_ipm_homography()
