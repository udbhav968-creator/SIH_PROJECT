import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Master training orchestrator for ROAD-SHIELD.
Trains all core neural models and writes a JSON summary report.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train_vision import run_training as train_vision
from training.train_imu import run_training as train_imu
from training.train_forensic_embedder import run_training as train_forensics

def train_all_models():
    start_time = time.time()
    print("=" * 70)
    print("🚀 ROAD-SHIELD MASTER AI MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    results = {}
    
    # 1. Train Vision Net
    t0 = time.time()
    res_vis = train_vision(epochs=15, batch_size=64)
    res_vis["training_time_seconds"] = round(time.time() - t0, 2)
    results["vision_distress_net"] = res_vis
    print("-" * 70)
    
    # 2. Train IMU Shock Net
    t0 = time.time()
    res_imu = train_imu(epochs=20, batch_size=64)
    res_imu["training_time_seconds"] = round(time.time() - t0, 2)
    results["imu_shock_classifier"] = res_imu
    print("-" * 70)
    
    # 3. Train Forensic Embedder
    t0 = time.time()
    res_for = train_forensics(epochs=15, batch_size=64)
    res_for["training_time_seconds"] = round(time.time() - t0, 2)
    results["forensic_embedder"] = res_for
    print("-" * 70)
    
    total_time = round(time.time() - start_time, 2)
    results["master_training_summary"] = {
        "status": "ALL_MODELS_TRAINED_SUCCESSFULLY",
        "total_training_walltime_seconds": total_time,
        "timestamp_utc": int(time.time()),
        "verified_framework": "Pure Vectorized NumPy Engine (Python 3.14 Compatible)"
    }
    
    summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "training_summary.json"))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"🎉 MASTER TRAINING COMPLETE in {total_time}s!")
    print(f"Summary saved to: {summary_path}")
    print("=" * 70)
    return results

if __name__ == "__main__":
    train_all_models()
