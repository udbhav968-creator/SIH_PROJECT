"""Dataset synthesis and sample extraction utilities."""

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "04_mobile_imu_telemetry_100hz")


def sample_real_imu_window(split="val", pothole_only=False, seed=None):
    """
    Returns one real (100, 3) accelerometer window from the project's own
    logged dataset, optionally restricted to windows labeled as a pothole
    impact. Returns None if the dataset file isn't present.
    """
    path = os.path.join(DATA_DIR, f"imu_shock_100hz_{split}.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    signals, labels = data["imu_signals"], data["labels"]
    idx_pool = np.where(labels == 3)[0] if pothole_only else np.arange(len(labels))
    if len(idx_pool) == 0:
        idx_pool = np.arange(len(labels))
    rng = np.random.RandomState(seed) if seed is not None else np.random
    idx = int(rng.choice(idx_pool))
    return signals[idx], int(labels[idx])
