import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Unit tests for Model M10 (MoRTH Autonomous Dispatch & Cryptographic Work-Order Agent).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.morth_dispatch_agent import MoRTHDispatchAgent

def test_morth_dispatch():
    print("Testing Model M10 (MoRTH Cryptographic Dispatch Agent)...")
    agent = MoRTHDispatchAgent()
    
    # Generate Work Order
    wo = agent.generate_work_order(
        corridor_id="NH-44-SECTOR-8",
        latitude=18.52043,
        longitude=73.85674,
        distress_class="D40 Pothole",
        area_sqm=2.45,
        depth_cm=7.0,
        pci_score=38
    )
    
    print(f"  [PASS] Generated Work Order: {wo['work_order_id']} | SLA: {wo['sla_resolution_hours']}h | Tonnage: {wo['required_mass_tonnes']} T")
    assert wo["sla_resolution_hours"] == 24, "Pothole on PCI 38 must have 24h SLA"
    assert wo["asphalt_mix"] == "DBM_SECTION_500"
    assert "sha256_cryptographic_seal" in wo
    assert len(wo["sha256_cryptographic_seal"]) == 64
    
    # Test Cryptographic Verification
    is_valid = agent.verify_work_order_seal(wo)
    print(f"  [PASS] Cryptographic Integrity Verification: {is_valid}")
    assert is_valid is True
    
    # Test Anti-Tamper Protection (Alter allocated budget)
    tampered_wo = dict(wo)
    tampered_wo["allocated_budget_inr"] += 50000.0  # Contractor attempts inflation
    is_tampered_valid = agent.verify_work_order_seal(tampered_wo)
    print(f"  [PASS] Tamper Detection Check: Valid = {is_tampered_valid} (Correctly Rejected)")
    assert is_tampered_valid is False, "Tampered work order must fail verification"
    
    print("  -> Model M10 MoRTH Dispatch Agent: ALL TESTS PASSED.")
    return True

if __name__ == "__main__":
    test_morth_dispatch()
