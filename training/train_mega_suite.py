"""
CLI entrypoint that runs this project's real training suite end-to-end:
training/train_vision.py (StandardScaler -> PCA -> SVC on the labeled photo
dataset) and training/train_imu.py (StandardScaler -> RandomForestClassifier
on the 100Hz IMU windows), via training.mega_pipeline.run_training_suite.

Usage:
    python3 -m training.train_mega_suite
"""
import os
import sys
import json
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from training.mega_pipeline import run_training_suite


def main():
    print("=" * 72)
    print("ROAD-SHIELD training suite: vision distress classifier + IMU shock classifier")
    print("=" * 72)

    t0 = time.time()
    result = run_training_suite(async_mode=False)
    walltime = round(time.time() - t0, 2)

    print(f"\nTraining complete in {walltime}s | status: {result.get('status')}")

    for key in ("vision", "imu"):
        report = result.get("results", {}).get(key)
        if not report:
            continue
        if "error" in report:
            print(f"  [{key}] FAILED: {report['error']}")
            continue
        acc = report.get("held_out_validation_accuracy")
        baseline = report.get("random_guess_baseline")
        n_val = report.get("held_out_validation_windows", report.get("held_out_validation_photos"))
        print(f"  [{key}] {report.get('classifier')}: held-out accuracy {acc} "
              f"(random-guess baseline {baseline}) on {n_val} held-out samples")

    report_path = os.path.join(ENGINE_ROOT, "checkpoints", "last_training_suite_report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull report written to {report_path}")
    except OSError as e:
        print(f"\n(Could not write report file: {e})")

    print("=" * 72)


if __name__ == "__main__":
    main()
