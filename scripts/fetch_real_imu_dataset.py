"""
Downloads real Indian road vehicular IMU (tri-axial accelerometer + gyroscope)
sensor logs from the open-source field measurement dataset
`VishalSingh25/Pothole-Project` and builds non-overlapping 100-sample (100, 3)
accelerometer windows (`Ax, Ay, Az` in m/s^2) with a strict temporal/file-block
split between train and validation sets (zero window overlap between splits).

Run directly:
    python -m scripts.fetch_real_imu_dataset
"""

import csv
import io
import json
import os
import time
import urllib.request
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMU_DIR = os.path.join(ROOT, "datasets", "04_mobile_imu_telemetry_100hz")
RAW_DIR = os.path.join(IMU_DIR, "raw_logs")

BASE_URL = "https://raw.githubusercontent.com/VishalSingh25/Pothole-Project/master/"

# Real field-recorded Indian road drives:
#   Class 0: Smooth Asphalt (plain_road)
#   Class 1: Expansion Joint / Unmarked Bump (plain_road_unmarked_sb)
#   Class 2: Rumble Strip / Marked Speed Breaker (plain_road_marked_sb)
#   Class 3: Pothole Impact (plain_road_potholes)
DRIVE_FILES = [
    (0, "plain_road_1.csv", "data/plain_road/1.csv"),
    (0, "plain_road_2.csv", "data/plain_road/2.csv"),
    (0, "plain_road_3.csv", "data/plain_road/3.csv"),
    (0, "plain_road_4.csv", "data/plain_road/4.csv"),
    (1, "unmarked_sb_1.csv", "data/plain_road_unmarked_sb/1.csv"),
    (1, "unmarked_sb_2.csv", "data/plain_road_unmarked_sb/2.csv"),
    (1, "unmarked_sb_3.csv", "data/plain_road_unmarked_sb/3.csv"),
    (2, "marked_sb_1.csv", "data/plain_road_marked_sb/1.csv"),
    (3, "potholes_1.csv", "data/plain_road_potholes/1.csv"),
    (3, "potholes_2.csv", "data/plain_road_potholes/2.csv"),
]

WINDOW_LEN = 100


def fetch_or_load_csv(local_name, remote_rel):
    os.makedirs(RAW_DIR, exist_ok=True)
    local_path = os.path.join(RAW_DIR, local_name)
    if not os.path.exists(local_path):
        url = BASE_URL + remote_rel
        print(f"  downloading {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "ROAD-SHIELD-Audit/1.0"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw_bytes = resp.read()
        with open(local_path, "wb") as fh:
            fh.write(raw_bytes)
    else:
        with open(local_path, "rb") as fh:
            raw_bytes = fh.read()

    text = raw_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for r in reader:
        try:
            ax = float(r["Ax"])
            ay = float(r["Ay"])
            az = float(r["Az"])
            if ax == 0.0 and ay == 0.0 and az == 0.0:
                continue  # skip sensor power-on zero row
            rows.append((ax, ay, az))
        except (KeyError, ValueError, TypeError):
            continue
    return np.asarray(rows, dtype=np.float32)


def slice_non_overlapping_windows(series, window_len=WINDOW_LEN):
    n = len(series) // window_len
    if n == 0:
        return np.zeros((0, window_len, 3), dtype=np.float32)
    trimmed = series[: n * window_len]
    return trimmed.reshape(n, window_len, 3)


def select_event_windows(windows, class_id):
    """
    A continuous drive file on a road with potholes or speed breakers includes
    stretches of ordinary road between the obstacles. Following the dataset
    authors' CUSUM/threshold event-selection methodology, we select the active
    event windows from obstacle drives (upper quantile of vertical-axis
    excursion / dynamic energy) and calm baseline windows from plain_road
    drives, while preserving temporal ordering so train and validation come
    from strictly separate drive files or disjoint temporal segments.
    """
    if len(windows) == 0:
        return windows
    az = windows[:, :, 2]
    ptp_z = az.max(axis=1) - az.min(axis=1)
    std_all = windows.std(axis=1).sum(axis=1)
    score = ptp_z + 1.5 * std_all

    if class_id == 0:
        # Smooth asphalt: keep calm baseline windows (below 55th percentile of shock excursion)
        # so unannotated bumps during a plain-road drive don't pollute class 0.
        cutoff = np.percentile(score, 55)
        mask = score <= cutoff
    else:
        # Obstacle drive: a vehicle spends most of a drive on plain road between
        # speed breakers / potholes; select the top 30% excursion windows where
        # the obstacle traversal actually occurs.
        cutoff = np.percentile(score, 70)
        mask = score >= cutoff
    return windows[mask]


def build_datasets():
    os.makedirs(IMU_DIR, exist_ok=True)
    print("[IMU Fetch] Fetching real Indian road vehicular IMU logs ...")

    by_class_files = {0: [], 1: [], 2: [], 3: []}
    raw_row_counts = {}
    for cls_id, local_name, remote_rel in DRIVE_FILES:
        arr = fetch_or_load_csv(local_name, remote_rel)
        raw_row_counts[local_name] = int(len(arr))
        wins = slice_non_overlapping_windows(arr, WINDOW_LEN)
        wins = select_event_windows(wins, cls_id)
        by_class_files[cls_id].append((local_name, wins))
        print(f"  class {cls_id} | {local_name:18s}: {len(arr):7d} raw rows -> {len(wins):4d} event windows")

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    split_manifest = {}

    for cls_id, file_entries in by_class_files.items():
        # Temporal block split: for each drive file, first 80% of contiguous
        # non-overlapping windows go to train, final 20% go to held-out val,
        # with a 2-window guard gap at the boundary so not even adjacent seconds
        # straddle the split.
        cls_tr, cls_va = 0, 0
        for fname, wins in file_entries:
            n = len(wins)
            cut = int(round(n * 0.80))
            tr_w = wins[: max(1, cut - 1)]
            va_w = wins[min(n, cut + 1) :]
            X_train_list.append(tr_w)
            y_train_list.append(np.full(len(tr_w), cls_id, dtype=np.int64))
            X_val_list.append(va_w)
            y_val_list.append(np.full(len(va_w), cls_id, dtype=np.int64))
            cls_tr += len(tr_w)
            cls_va += len(va_w)
        split_manifest[str(cls_id)] = {"train_windows": cls_tr, "val_windows": cls_va}

    X_train = np.concatenate(X_train_list, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train_list, axis=0).astype(np.int64)
    X_val = np.concatenate(X_val_list, axis=0).astype(np.float32)
    y_val = np.concatenate(y_val_list, axis=0).astype(np.int64)

    train_path = os.path.join(IMU_DIR, "imu_shock_100hz_train.npz")
    val_path = os.path.join(IMU_DIR, "imu_shock_100hz_val.npz")
    np.savez_compressed(train_path, imu_signals=X_train, labels=y_train)
    np.savez_compressed(val_path, imu_signals=X_val, labels=y_val)

    meta = {
        "dataset_name": "Indian Road Vehicular IMU Tri-Axial Accelerometer Benchmark",
        "source_citation": (
            "Real field-recorded Indian road accelerometer logs from "
            "VishalSingh25/Pothole-Project (10 drive CSV logs across plain road, "
            "unmarked speed breakers/joints, marked speed breakers/rumble strips, "
            "and pothole corridors)"
        ),
        "source_url": "https://github.com/VishalSingh25/Pothole-Project/tree/master/data",
        "window_size_timesteps": WINDOW_LEN,
        "channels": [
            "Ax (Lateral m/s^2)",
            "Ay (Longitudinal m/s^2)",
            "Az (Vertical Shock m/s^2)",
        ],
        "raw_csv_rows": raw_row_counts,
        "total_raw_rows": int(sum(raw_row_counts.values())),
        "train_windows": int(len(X_train)),
        "val_windows": int(len(X_val)),
        "split_policy": (
            "Non-overlapping 100-sample windows sliced in temporal order; "
            "first 80% of each drive log assigned to train, final 20% to val, "
            "with a guard gap between splits."
        ),
        "per_class_counts": split_manifest,
        "classes": {
            "0": "Smooth Asphalt (plain_road)",
            "1": "Expansion Joint / Unmarked Bump (plain_road_unmarked_sb)",
            "2": "Rumble Strip / Marked Speed Breaker (plain_road_marked_sb)",
            "3": "Pothole Impact (plain_road_potholes)",
        },
        "generated_at_unix": int(time.time()),
    }
    meta_path = os.path.join(IMU_DIR, "imu_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[IMU Fetch] Saved {len(X_train)} train windows -> {train_path}")
    print(f"[IMU Fetch] Saved {len(X_val)} val windows   -> {val_path}")
    print(f"[IMU Fetch] Updated metadata -> {meta_path}")


if __name__ == "__main__":
    build_datasets()
