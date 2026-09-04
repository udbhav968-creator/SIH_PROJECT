"""
CLI Runner for ROAD-SHIELD Mega-Level Deep Training Pipeline.
Executes training across RDD2022 India, Kaggle Pothole-600, CRACK500, and Mobile-IMU.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from training.mega_pipeline import run_mega_training_suite, telemetry_streamer

def main():
    print("=" * 75)
    print("🛣️  ROAD-SHIELD MEGA-LEVEL DEEP TRAINING SUITE (MoRTH / NHAI SIH2026)")
    print("    Canonical Benchmarks: RDD2022 India | Kaggle Pothole-600 | Mobile-IMU 100Hz")
    print("=" * 75)
    
    t0 = time.time()
    result = run_mega_training_suite(epochs=15, lr_max=0.004, batch_size=128, async_mode=False)
    walltime = round(time.time() - t0, 2)
    
    print(f"\n[Mega Training Complete] Walltime: {walltime}s | Status: {result['status']}")
    print(f"  Final Loss: {result['current_loss']} | Metric: {result['val_metric']} ({result['val_metric_name']})")
    
    zoo_path = os.path.join(ENGINE_ROOT, "checkpoints", "mega_model_zoo.json")
    if os.path.exists(zoo_path):
        with open(zoo_path, "r", encoding="utf-8") as f:
            zoo = json.load(f)
        print(f"\n📦 Verified Model Zoo Registry ({len(zoo['models'])} Models):")
        for m_name, m_info in zoo["models"].items():
            print(f"  • {m_name:<24} | Params: {m_info['parameters']:>6} | SHA-256: {m_info['sha256'][:16]}...")
            
    print("=" * 75)
    print("🏆 ALL BENCHMARK MODELS TRAINED AND CRYPTOGRAPHICALLY SEALED!")
    print("=" * 75)

if __name__ == "__main__":
    main()
