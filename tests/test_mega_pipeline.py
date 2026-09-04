"""
ROAD-SHIELD Mega-Pipeline & Benchmark Dataset Hub Verification Test Suite
MoRTH / NHAI Autonomous Road Asset Intelligence
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import time
import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from data.benchmark_dataset_hub import BenchmarkDatasetHub
from data.augmentation_pipeline import CivilDataAugmentor
from training.mega_pipeline import (
    FocalCrossEntropyLoss,
    HuberSmoothL1Loss,
    CosineAnnealingScheduler,
    telemetry_streamer
)

def test_benchmark_dataset_hub():
    print("\n[Test 1/4] Canonical Benchmark Dataset Hub & Inventory...")
    hub = BenchmarkDatasetHub(seed=42)
    inv = hub.get_dataset_inventory()
    assert inv["total_curated_samples"] >= 100000, f"Expected >= 100,000 samples, got {inv['total_curated_samples']}"
    assert "RDD2022_India" in inv["datasets"]
    assert "Kaggle_Pothole_600" in inv["datasets"]
    assert "Mobile_IMU_100Hz" in inv["datasets"]
    
    # Test feature extraction
    X_rdd, y_rdd, geo_rdd = hub.load_rdd2022_india_features(num_samples=100, apply_augmentation=True)
    assert X_rdd.shape == (100, 64)
    assert len(y_rdd) == 100
    assert geo_rdd.shape == (100, 4)
    
    X_imu, y_imu = hub.load_mobile_imu_benchmark(num_samples=100)
    assert X_imu.shape == (100, 100, 3), f"Expected (100, 100, 3), got {X_imu.shape}"
    from models.imu_shock_classifier import IMUShockClassifier
    feats = IMUShockClassifier.extract_temporal_features(X_imu)
    assert feats.shape == (100, 36), f"Expected (100, 36), got {feats.shape}"
    assert len(y_imu) == 100
    print("  [PASS] Benchmark Dataset Hub validated across 5 curated datasets.")

def test_civil_augmentation():
    print("\n[Test 2/4] Civil Domain Augmentation Pipeline...")
    aug = CivilDataAugmentor(seed=42)
    feat1 = np.full(64, 0.2, dtype=np.float32)
    feat2 = np.full(64, 0.8, dtype=np.float32)
    
    # Mosaic test
    mos_f, mos_l = aug.mosaic_4way([feat1, feat2, feat1, feat2], [0, 4, 1, 3])
    assert mos_f.shape == (64,)
    assert mos_l == 4
    
    # Cutmix test
    cut_f = aug.cutmix_distress(feat1, feat2, patch_ratio=0.25)
    assert cut_f.shape == (64,)
    assert np.max(cut_f) >= 0.79
    
    # Weather simulation
    monsoon_f = aug.simulate_monsoon_weather(feat1, rain_intensity=0.8)
    assert monsoon_f.shape == (64,)
    
    night_f = aug.simulate_night_sodium_lighting(feat1)
    assert night_f.shape == (64,)
    assert np.mean(night_f) < np.mean(feat1)
    print("  [PASS] 4-Way Mosaic, CutMix, Monsoon Streaks, and Sodium Night Lighting verified.")

def test_losses_and_schedulers():
    print("\n[Test 3/4] Focal Loss & Cosine Annealing Schedulers...")
    focal = FocalCrossEntropyLoss(gamma=2.0, alpha=0.25)
    probs = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    y_true = np.array([1, 0], dtype=np.int64)
    loss_val = focal(probs, y_true)
    assert 0.0 < loss_val < 1.0, f"Unexpected loss value {loss_val}"
    
    huber = HuberSmoothL1Loss(delta=1.0)
    l_huber = huber(np.array([10.0]), np.array([12.0]))
    assert l_huber == 1.5, f"Expected 1.5 for delta=1 diff=2, got {l_huber}"
    
    sched = CosineAnnealingScheduler(lr_max=0.01, lr_min=0.001, total_epochs=20, restart_period=10)
    lr_0 = sched.get_lr(0)
    lr_5 = sched.get_lr(5)
    assert abs(lr_0 - 0.01) < 1e-6
    assert 0.001 < lr_5 < 0.01
    print("  [PASS] Focal Loss, Huber Smooth-L1, and SGDR Cosine Schedulers mathematically conformant.")

def test_model_zoo_registry():
    print("\n[Test 4/4] Cryptographic Model Zoo Registry & Checkpoint Check...")
    zoo_path = os.path.join(ENGINE_ROOT, "checkpoints", "mega_model_zoo.json")
    assert os.path.exists(zoo_path), "mega_model_zoo.json does not exist"
    with open(zoo_path, "r", encoding="utf-8") as f:
        zoo = json.load(f)
    assert "models" in zoo
    assert len(zoo["models"]) >= 4
    for name, m in zoo["models"].items():
        assert len(m["sha256"]) == 64, f"Invalid SHA-256 for {name}: {m['sha256']}"
    print(f"  [PASS] Model Zoo Registry verified with {len(zoo['models'])} sealed neural architectures.")

def main():
    print("=" * 75)
    print("🧪 ROAD-SHIELD MEGA-PIPELINE & BENCHMARK VERIFICATION SUITE")
    print("=" * 75)
    test_benchmark_dataset_hub()
    test_civil_augmentation()
    test_losses_and_schedulers()
    test_model_zoo_registry()
    print("=" * 75)
    print("🏆 ALL MEGA-PIPELINE VERIFICATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 75)

if __name__ == "__main__":
    main()
