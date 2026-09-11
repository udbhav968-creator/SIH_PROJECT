"""
Real-sample helpers for demo endpoints that need one example input.

The earlier version of this module was a synthetic-data generator:
`generate_vision_dataset` produced pure Gaussian noise with a few
dimensions nudged per class (never an image, never used to train anything
real), `generate_imu_dataset` fabricated accelerometer windows from
hand-tuned impulse/oscillation shapes, and `generate_forensic_triplets`
made up random unit vectors for a metric-learning setup nothing in this
project actually trains. None of it was real sensor data.

This project now has real, trained models: VisionDistressNet (trained on
photos under datasets/*/real_images) and IMUShockClassifier (trained on
datasets/04_mobile_imu_telemetry_100hz). When a demo endpoint needs "an
example input", the honest thing is to hand it a real one from those same
sources - which is what the functions below do - rather than fabricate a
new one.
"""

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
