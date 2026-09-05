import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Unit tests & benchmark for Model M1 (Vision Distress Net).
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vision_distress_net import VisionDistressNet
from data.dataset_generator import generate_vision_dataset

def test_vision_model():
    print("Testing Model M1 (Vision Distress Net)...")
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "vision_distress_weights.npz"))
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    
    model = VisionDistressNet(in_features=64, hidden_dims=[512, 256, 128], num_classes=10)
    model.load_weights(ckpt_path)
    
    # Test 1: Inference shape & probabilities
    X_dummy = np.random.randn(10, 64).astype(np.float32)
    preds, conf, probs, geo_preds = model.predict(X_dummy)
    assert preds.shape == (10,), f"Expected shape (10,), got {preds.shape}"
    assert np.allclose(np.sum(probs, axis=-1), 1.0, atol=1e-5), "Probabilities must sum to 1"
    assert geo_preds.shape == (10, 4), f"Expected geo shape (10, 4), got {geo_preds.shape}"
    
    # Test 2: Latency benchmark
    t0 = time.time()
    for _ in range(100):
        _ = model.predict(X_dummy[:1])
    avg_latency_ms = ((time.time() - t0) / 100) * 1000.0
    print(f"  [PASS] Single-frame inference latency: {avg_latency_ms:.3f} ms (Target < 25ms)")
    assert avg_latency_ms < 25.0, "Latency exceeds 25ms threshold"
    
    # Test 3: Holdout test accuracy
    X_test, y_test, _, _ = generate_vision_dataset(num_samples=500, seed=999)
    preds_test, _, _, _ = model.predict(X_test)
    acc = np.mean(preds_test == y_test) * 100.0
    print(f"  [PASS] Holdout Test Accuracy: {acc:.2f}% (Target > 80%)")
    assert acc >= 80.0, f"Accuracy {acc:.2f}% below required 80%"
    
    print("  -> Model M1 Vision Distress Net: ALL TESTS PASSED.")
    return True

if __name__ == "__main__":
    test_vision_model()
