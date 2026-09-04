import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Unit tests for Models M7 & M8 (Anti-Fraud Metric Embedder & Texture Forensics).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.forensic_audit_engine import ForensicMetricEmbedder, ForensicTextureAuditor

def test_forensic_audit():
    print("Testing Models M7 & M8 (Forensic Anti-Fraud & Repair Quality Engine)...")
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "forensic_embedder_weights.npz"))
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    
    embedder = ForensicMetricEmbedder(in_dim=48, hidden_dim=64, embed_dim=32)
    embedder.load_weights(ckpt_path)
    
    # Test 1: Duplicate Photo Detection
    v_orig = np.random.randn(1, 48).astype(np.float32)
    v_dup = v_orig + np.random.normal(0, 0.02, (1, 48)).astype(np.float32)  # Same photo with noise
    v_diff = np.random.randn(1, 48).astype(np.float32)  # Different photo
    
    e1 = embedder.forward(v_orig)
    e_dup = embedder.forward(v_dup)
    e_diff = embedder.forward(v_diff)
    
    sim_dup = embedder.cosine_similarity(e1, e_dup)
    sim_diff = embedder.cosine_similarity(e1, e_diff)
    print(f"  [PASS] Duplicate Claim Similarity: {sim_dup:.3f} | Different Claim Similarity: {sim_diff:.3f}")
    assert sim_dup >= 0.90, f"Expected duplicate sim >= 0.90, got {sim_dup}"
    assert sim_diff < 0.65, f"Expected different sim < 0.65, got {sim_diff}"
    
    # Test 2: SSIM & Laplacian Texture Verification
    auditor = ForensicTextureAuditor()
    
    # Generate synthetic before (jagged pothole) and after (smooth asphalt)
    np.random.seed(42)
    img_before = np.random.uniform(20, 240, (128, 128)).astype(np.float32)  # Rough texture
    img_after = np.ones((128, 128), dtype=np.float32) * 80.0 + np.random.normal(0, 3.0, (128, 128)).astype(np.float32)  # Smooth rolled asphalt
    
    res_audit_pass = auditor.evaluate_repair(img_before, img_after, claimed_dist_m=0.8)
    print(f"  [PASS] Genuine Repair Audit: Verdict = {res_audit_pass['verdict']} (SSIM={res_audit_pass['ssim_index']}, Laplacian={res_audit_pass['laplacian_variance_after']})")
    assert res_audit_pass["audit_passed"] is True
    
    # Test 3: Ghost Claim (Uploaded same photo as before, no repair done)
    res_audit_ghost = auditor.evaluate_repair(img_before, img_before, claimed_dist_m=0.8)
    print(f"  [PASS] Ghost Claim Rejection: Verdict = {res_audit_ghost['verdict']}")
    assert res_audit_ghost["audit_passed"] is False
    assert res_audit_ghost["verdict"] == "REJECTED_GHOST_CLAIM"
    
    print("  -> Models M7 & M8 Forensic Audit Engine: ALL TESTS PASSED.")
    return True

if __name__ == "__main__":
    test_forensic_audit()
