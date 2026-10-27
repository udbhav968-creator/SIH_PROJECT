"""
ROAD-SHIELD: Master Unified Deep Training Suite
================================================
SIH 2026 Problem Statement SIH26124 | CSET485 AI and Society

Orchestrates the entire multi-modal training pipeline from end to end:
  Stage 1: Rare-Class Ingestion & Quality Audit (scripts/expand_rare_classes.py)
  Stage 2: Civil Engineering Physics Models (training/train_civil_models.py)
            - ASTM D6433 Pavement Condition Index (PCI 0-100) Regressor
            - HDM-4 / IRC:37 Pavement Deterioration 180-Day Forecaster
            - IRC:SP:83 / IRC:82 Photometric Cavity & Crack Depth Estimator
  Stage 3: 100 Hz IMU Chassis Vibration Classifier (training/train_imu.py)
            - Tri-axial accelerometer FFT & temporal feature shock classifier
  Stage 4: 11-Feature Pixel Defect Segmenter (training/train_segmenter.py)
            - HistGradientBoosting pixel classifier on defect polygons + clean road hard negatives
  Stage 5: Dual Attention Network & 21-Node Decision DAG (training/train_dan_dag.py)
            - MobileNetV2 / ResNet-50 Backbone
            - Position Attention Module (PAM, 16-D spatial self-attention)
            - Channel Attention Module (CAM, 1280 -> 160 -> 1280 Fisher Squeeze-Excitation)
            - 21-Node Rooted Pairwise Decision DAG Classifier
  Stage 6: CNN Embedding Head with Mirror Averaging (training/train_cnn_head.py)
            - ResNet-50 / MobileNetV2 with grouped cross-validation
  Stage 7: Full Automated Regression Verification (tests/test_road_shield.py)

Usage:
    python -m training.train_all_subsystems
    python -m training.train_all_subsystems --fast
    python -m training.train_all_subsystems --target-rare 80
"""

import argparse
import json
import os
import subprocess
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")


def run_stage(name, cmd_args, required=True):
    print("\n" + "=" * 78, flush=True)
    print(f"  [STAGE] {name.upper()}", flush=True)
    print(f"  Command: {sys.executable} {' '.join(cmd_args)}", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    res = subprocess.run([sys.executable] + cmd_args, cwd=ENGINE_ROOT)
    elapsed = round(time.time() - t0, 1)
    if res.returncode != 0:
        print(f"\n[ERROR] Stage '{name}' FAILED with exit code {res.returncode} in {elapsed}s", flush=True)
        if required:
            sys.exit(f"Terminating master training pipeline: critical stage '{name}' failed.")
        return False, elapsed
    print(f"\n[SUCCESS] Stage '{name}' completed successfully in {elapsed}s", flush=True)
    return True, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Skip heavy segmenter retrain if weights exist")
    parser.add_argument("--target-rare", type=int, default=60, help="Target real images per scarce class")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip online dataset fetching")
    parser.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "resnet50"])
    args = parser.parse_args()

    total_start = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    stage_timings = {}

    print("*" * 78, flush=True)
    print("  ROAD-SHIELD: MASTER UNIFIED DEEP TRAINING SYSTEM", flush=True)
    print("  SIH26124 / Bharat Electronics Limited (BEL) | Team 42", flush=True)
    print("*" * 78, flush=True)

    # 1. Dataset Expansion for Scarce Classes
    if not args.skip_fetch:
        ok, dt = run_stage("Rare Class Dataset Expansion",
                           ["-m", "scripts.expand_rare_classes", "--target", str(args.target_rare)],
                           required=False)
        stage_timings["Dataset Expansion"] = dt

    # 2. Civil Engineering Models (PCI, Deterioration, Depth)
    ok, dt = run_stage("Civil Engineering Models (ASTM D6433, HDM-4, IRC Depth)",
                       ["-m", "training.train_civil_models"],
                       required=True)
    stage_timings["Civil Models"] = dt

    # 3. 100 Hz IMU Shock Vibration Classifier
    ok, dt = run_stage("100 Hz IMU Shock Classifier",
                       ["-m", "training.train_imu"],
                       required=True)
    stage_timings["IMU Shock"] = dt

    # 4. Defect Pixel Segmenter
    seg_ckpt = os.path.join(CKPT_DIR, "defect_segmenter.joblib")
    if args.fast and os.path.exists(seg_ckpt):
        print("\n[INFO] --fast flag active: using existing defect segmenter checkpoint.", flush=True)
    else:
        ok, dt = run_stage("Defect Pixel Segmenter (HistGradientBoosting)",
                           ["-m", "training.train_segmenter", "--images", "1200"],
                           required=False)
        stage_timings["Defect Segmenter"] = dt

    # 5. Dual Attention Network & 21-Node Decision DAG
    ok, dt = run_stage("DAN-DAG Deep Vision Classifier (PAM/CAM Attention + 21-Node DAG)",
                       ["-m", "training.train_dan_dag", "--backbone", args.backbone, "--epochs", "65"],
                       required=True)
    stage_timings["DAN-DAG Network"] = dt

    # 6. CNN Embedding Head (ResNet-50 / MobileNetV2 Mirror Averaging)
    ok, dt = run_stage("CNN Embedder Head & Baseline Comparison",
                       ["-m", "training.train_cnn_head", "--backbone", args.backbone, "--compare"],
                       required=False)
    stage_timings["CNN Head"] = dt

    # 7. Verification Test Suite
    ok, dt = run_stage("Automated Verification Suite",
                       ["-m", "unittest", "tests.test_road_shield"],
                       required=True)
    stage_timings["Regression Tests"] = dt

    total_time = round(time.time() - total_start, 1)

    print("\n" + "=" * 78, flush=True)
    print("  ROAD-SHIELD MASTER TRAINING SUITE FINISHED", flush=True)
    print("=" * 78, flush=True)
    for st, sec in stage_timings.items():
        print(f"  {st:<42}: {sec:>6.1f} s", flush=True)
    print("-" * 78, flush=True)
    print(f"  Total End-to-End Elapsed Time            : {total_time:>6.1f} s", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
