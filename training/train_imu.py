"""
Trains Model M4 (IMUShockClassifier) on the real logged windows in
datasets/04_mobile_imu_telemetry_100hz/imu_shock_100hz_{train,val}.npz and
reports genuine held-out validation metrics (the val.npz file is never
touched during fitting).

Run directly: python -m training.train_imu
"""

# Cap the BLAS thread pools before NumPy/scikit-learn are imported. On
# laptops with modest RAM, OpenBLAS otherwise allocates a buffer per
# thread per core and dies with "Memory allocation still failed".
import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "2")


import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from models.imu_shock_classifier import IMUShockClassifier

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "04_mobile_imu_telemetry_100hz"))
CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
MODEL_PATH = os.path.join(CKPT_DIR, "imu_shock_model.joblib")
REPORT_PATH = os.path.join(CKPT_DIR, "imu_shock_report.json")


def _load_split(name):
    path = os.path.join(DATA_DIR, f"imu_shock_100hz_{name}.npz")
    data = np.load(path)
    return data["imu_signals"], data["labels"]


def run_training(save_dir=None):
    save_dir = save_dir or CKPT_DIR
    os.makedirs(save_dir, exist_ok=True)

    print("[M4 IMU] Loading logged 100Hz accelerometer windows ...")
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    print(f"  train windows: {X_train.shape[0]} | held-out val windows: {X_val.shape[0]}")

    t0 = time.time()
    model = IMUShockClassifier()
    model.fit(X_train, y_train)
    print(f"  trained in {time.time() - t0:.1f}s")

    metrics = model.evaluate(X_val, y_val)
    print(f"  HELD-OUT validation accuracy: {metrics['accuracy'] * 100:.1f}% "
          f"(random-guess baseline for {len(IMUShockClassifier.CLASS_NAMES)} classes = "
          f"{100.0 / len(IMUShockClassifier.CLASS_NAMES):.1f}%)")

    model.save(MODEL_PATH)
    print(f"  saved trained model -> {MODEL_PATH}")

    report = {
        "model": "IMUShockClassifier",
        "classifier": "StandardScaler -> RandomForestClassifier",
        "class_names": IMUShockClassifier.CLASS_NAMES,
        "train_windows": int(X_train.shape[0]),
        "held_out_validation_windows": int(X_val.shape[0]),
        "held_out_validation_accuracy": round(metrics["accuracy"], 4),
        "random_guess_baseline": round(1.0 / len(IMUShockClassifier.CLASS_NAMES), 4),
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class_report": metrics["per_class_report"],
        "data_provenance": (
            "Simulated 100Hz tri-axial accelerometer windows shipped with this repo "
            "(datasets/04_mobile_imu_telemetry_100hz) - not field-collected MoRTH fleet "
            "logs, whatever the dataset's own metadata file claims. Treat this model as "
            "validated against the simulator, not against real vehicles, until it's "
            "retrained on genuine sensor logs."
        ),
        "trained_at_unix": int(time.time()),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  wrote training report -> {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_training()
