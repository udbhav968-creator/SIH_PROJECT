"""Train and calibrate all road inspection subsystems."""

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

    if not args.skip_fetch:
        ok, dt = run_stage("Rare Class Dataset Expansion",
                           ["-m", "scripts.expand_rare_classes", "--target", str(args.target_rare)],
                           required=False)
        stage_timings["Dataset Expansion"] = dt

    ok, dt = run_stage("Civil Engineering Models (ASTM D6433, HDM-4, IRC Depth)",
                       ["-m", "training.train_civil_models"],
                       required=True)
    stage_timings["Civil Models"] = dt

    ok, dt = run_stage("100 Hz IMU Shock Classifier",
                       ["-m", "training.train_imu"],
                       required=True)
    stage_timings["IMU Shock"] = dt

    seg_ckpt = os.path.join(CKPT_DIR, "defect_segmenter.joblib")
    if args.fast and os.path.exists(seg_ckpt):
        print("\n[INFO] --fast flag active: using existing defect segmenter checkpoint.", flush=True)
    else:
        ok, dt = run_stage("Defect Pixel Segmenter (HistGradientBoosting)",
                           ["-m", "training.train_segmenter", "--images", "1200"],
                           required=False)
        stage_timings["Defect Segmenter"] = dt

    ok, dt = run_stage("DAN-DAG Deep Vision Classifier (PAM/CAM Attention + 21-Node DAG)",
                       ["-m", "training.train_dan_dag", "--backbone", args.backbone, "--epochs", "65"],
                       required=True)
    stage_timings["DAN-DAG Network"] = dt

    ok, dt = run_stage("CNN Embedder Head & Baseline Comparison",
                       ["-m", "training.train_cnn_head", "--backbone", args.backbone, "--compare"],
                       required=False)
    stage_timings["CNN Head"] = dt

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
