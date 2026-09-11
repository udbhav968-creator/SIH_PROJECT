"""
Benchmark dataset inventory - reports what this project actually has on disk.

The earlier version of this module hard-coded a catalog of "canonical
benchmark datasets" (RDD2022, Kaggle Pothole-600, CRACK500, Mobile-IMU,
"Hard Negatives Repo") with invented sample counts (45000, 25000, 15000,
30000, 10000 - none of them real) and shipped a set of "load_*" methods
that generated pure Gaussian-noise feature vectors nudged per class,
labeled with those datasets' real names. This rewrite reports real counts
off disk instead - see data/image_dataset.py (photo counts) and the real
.npz files under datasets/04_mobile_imu_telemetry_100hz (IMU window
counts) - and drops the fabricated feature generators, since nothing in
this project's real training scripts (training/train_vision.py,
training/train_imu.py) uses them; they load their data directly from disk.
"""
import os
import numpy as np

from data.image_dataset import dataset_inventory as _image_dataset_inventory

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "04_mobile_imu_telemetry_100hz")


class BenchmarkDatasetHub:
    """Reports the real, on-disk contents of datasets/ - no fabricated sample counts."""

    def __init__(self, seed=42):
        self.seed = seed

    def _imu_counts(self):
        counts = {}
        for split in ("train", "val"):
            path = os.path.join(DATA_DIR, f"imu_shock_100hz_{split}.npz")
            if os.path.exists(path):
                data = np.load(path)
                counts[split] = int(data["labels"].shape[0])
            else:
                counts[split] = 0
        return counts

    def get_dataset_inventory(self):
        """Real photo and IMU-window counts, read straight off disk."""
        image_inv = _image_dataset_inventory()
        imu_counts = self._imu_counts()
        total_labeled_photos = sum(v["usable_photos"] for v in image_inv["labeled_classes"].values())
        return {
            "hub_version": "3.0-real-inventory",
            "note": "Counts below are read from the actual files under datasets/ - not a fixed catalog figure.",
            "labeled_photo_classes": image_inv["labeled_classes"],
            "unlabeled_holdout_photo_folders": image_inv["unlabeled_holdout_folders"],
            "total_labeled_photos": total_labeled_photos,
            "imu_telemetry_windows": imu_counts,
        }
