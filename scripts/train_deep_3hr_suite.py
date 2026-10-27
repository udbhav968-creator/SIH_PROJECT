"""
ROAD-SHIELD: 3-Hour Deep Training Pipeline (Local CPU / PowerShell)
===================================================================
SIH 2026 Problem Statement SIH26124 | Bharat Electronics Limited (BEL)

Executes an exhaustive, real 2.5-to-3 hour deep training job locally:
  1. Rare-Class Dataset Expansion & Deduplication (~10 min)
  2. Real Photometric Civil Models (ASTM D6433 PCI, HDM-4, IRC Depth) (~15 min)
  3. 100 Hz IMU Chassis Vibration Classifier (~5 min)
  4. Defect Pixel Segmenter on 2,000 images / 4.5M pixels (~35 min)
  5. Deep Mirror-Averaged CNN Embeddings & Cross-Validation (~45 min)
  6. Flagship Dual-Attention Network (PAM/CAM) + 21-Node Decision DAG (80 Epochs PyTorch) (~40 min)
  7. Automated Verification & System Claims Audit (~10 min)

Outputs:
  checkpoints/dan_dag_model.joblib
  checkpoints/cnn_head_mobilenetv2.joblib
  checkpoints/defect_segmenter.joblib
  checkpoints/depth_estimator_model.joblib
  checkpoints/deterioration_model.joblib
  checkpoints/pci_model.joblib
  checkpoints/imu_shock_model.joblib
"""

import os
import sys
import time
import subprocess
import argparse
import json

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")


def log(msg, symbol=">>>"):
    elapsed = time.strftime("%H:%M:%S")
    print(f"\n[{elapsed}] {symbol} {msg}", flush=True)


def run_stage(name, args, required=True):
    log(f"STAGE: {name.upper()}", symbol="🚀")
    t0 = time.time()
    cmd = [sys.executable] + args
    print(f"Command: {' '.join(cmd)}\n" + "-" * 72, flush=True)
    res = subprocess.run(cmd, cwd=ENGINE_ROOT)
    dt = round(time.time() - t0, 1)
    if res.returncode != 0:
        log(f"FAILED: Stage '{name}' exited with code {res.returncode} in {dt/60:.1f} min", symbol="❌")
        if required:
            sys.exit(f"Master training pipeline aborted on critical failure in '{name}'.")
        return False, dt
    log(f"COMPLETED: Stage '{name}' in {dt/60:.1f} min", symbol="✅")
    return True, dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dan-epochs", type=int, default=80, help="PyTorch epochs for DAN-DAG training")
    parser.add_argument("--seg-images", type=int, default=2000, help="Photographs for defect segmentation")
    parser.add_argument("--target-rare", type=int, default=80, help="Target real images per scarce class")
    parser.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "resnet50"])
    args = parser.parse_args()

    overall_start = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    timings = {}

    print("=" * 78, flush=True)
    print("  ROAD-SHIELD: FULL 3-HOUR DEEP MULTI-MODAL TRAINING SYSTEM", flush=True)
    print("  SIH 2026 Problem Statement SIH26124 | Team 42", flush=True)
    print("  Device: Laptop CPU | Optimization: PyTorch + scikit-learn + ONNX", flush=True)
    print("=" * 78, flush=True)

    # 1. Expand rare classes (download & verify real road hazard photographs)
    ok, dt = run_stage(
        "Rare-Class Dataset Expansion & 64-bit pHash Audit",
        ["-m", "scripts.expand_rare_classes", "--target", str(args.target_rare)],
        required=False
    )
    timings["Rare Class Expansion"] = dt

    # 2. Civil Engineering Models (ASTM D6433 PCI, HDM-4 Deterioration, IRC Depth)
    ok, dt = run_stage(
        "Civil Engineering Models (Real Image Distresses)",
        ["-m", "training.train_civil_real"],
        required=True
    )
    timings["Civil Models"] = dt

    # 3. 100 Hz IMU Shock Vibration Classifier
    ok, dt = run_stage(
        "100 Hz IMU Chassis Vibration Classifier (36 features)",
        ["-m", "training.train_imu"],
        required=True
    )
    timings["IMU Shock Model"] = dt

    # 4. Pixel Defect Segmenter on 2,000 photographs
    ok, dt = run_stage(
        "11-Feature Pixel Defect Segmenter (4.5M pixels)",
        ["-m", "training.train_segmenter", "--images", str(args.seg_images)],
        required=True
    )
    timings["Pixel Defect Segmenter"] = dt

    # 5. Deep CNN Feature Embedding Head
    ok, dt = run_stage(
        f"Deep CNN Embeddings ({args.backbone}) & Mirror-Averaged Grouped CV",
        ["-m", "training.train_cnn_head", "--backbone", args.backbone, "--compare"],
        required=True
    )
    timings["CNN Head"] = dt

    # 6. Flagship Dual Attention Network (DAN) + 21-Node Decision DAG
    ok, dt = run_stage(
        f"Flagship DAN-DAG Deep Network (PAM/CAM + 21-Node DAG, {args.dan_epochs} epochs)",
        ["-m", "training.train_dan_dag", "--backbone", args.backbone, "--epochs", str(args.dan_epochs), "--max-per-class", "2000"],
        required=True
    )
    timings["DAN-DAG Deep Network"] = dt

    # 7. Verification Test Suite & Claims Validation
    ok, dt = run_stage(
        "Automated System Regression & Claims Verification Suite",
        ["-m", "unittest", "tests.test_road_shield"],
        required=True
    )
    timings["Test Suite Verification"] = dt

    total_time = round(time.time() - overall_start, 1)

    print("\n" + "=" * 78, flush=True)
    print("  ROAD-SHIELD: COMPLETE 3-HOUR TRAINING FINISHED SUCCESSFULLY", flush=True)
    print("=" * 78, flush=True)
    for stage, sec in timings.items():
        print(f"  {stage:<45}: {sec/60:>5.1f} min ({sec:>6.1f} s)", flush=True)
    print("-" * 78, flush=True)
    print(f"  Total End-to-End Elapsed Time              : {total_time/3600:>5.2f} hours ({total_time/60:.1f} min)", flush=True)
    print("=" * 78, flush=True)
    print("  All checkpoints have been updated in checkpoints/.", flush=True)
    print("  You can now start the dashboard: powershell .\\start_server.ps1 -OpenBrowser", flush=True)


if __name__ == "__main__":
    main()
