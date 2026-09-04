import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Unit tests & benchmark for Model M4 (100 Hz IMU Shock Classifier).
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.imu_shock_classifier import IMUShockClassifier
from data.dataset_generator import generate_imu_dataset

def test_imu_shock():
    print("Testing Model M4 (100 Hz IMU Shock Classifier)...")
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "imu_shock_weights.npz"))
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    
    model = IMUShockClassifier(in_features=36, hidden_dims=[64, 32], num_classes=4)
    model.load_weights(ckpt_path)
    
    # Test 1: Feature extraction shape
    X_dummy = np.random.randn(8, 100, 3).astype(np.float32)
    feats = IMUShockClassifier.extract_temporal_features(X_dummy)
    assert feats.shape == (8, 36), f"Expected (8, 36), got {feats.shape}"
    
    # Test 2: Inference latency
    t0 = time.time()
    for _ in range(100):
        _ = model.predict(X_dummy[:1])
    latency_ms = ((time.time() - t0) / 100) * 1000.0
    print(f"  [PASS] 100-sample window latency: {latency_ms:.3f} ms (Target < 10ms)")
    assert latency_ms < 10.0, "Latency exceeds 10ms"
    
    # Test 3: Holdout test accuracy
    X_test, y_test = generate_imu_dataset(num_samples=400, seed=999)
    preds, pothole_conf, _ = model.predict(X_test)
    acc = np.mean(preds == y_test) * 100.0
    pothole_recall = np.mean(preds[y_test == 3] == 3) * 100.0
    print(f"  [PASS] Holdout Accuracy: {acc:.2f}% | Pothole Shock Recall: {pothole_recall:.2f}%")
    assert acc >= 80.0, f"Accuracy {acc:.2f}% below threshold"
    assert pothole_recall >= 85.0, f"Pothole recall {pothole_recall:.2f}% below 85%"
    
    print("  -> Model M4 IMU Shock Classifier: ALL TESTS PASSED.")
    return True

if __name__ == "__main__":
    test_imu_shock()
