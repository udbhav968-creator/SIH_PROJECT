"""
Dataset preparation and training plumbing for the YOLO detectors.

What matters most here is the leakage guarantee: CDSet frames come from three
videos, and a frame one step away from a training frame is effectively a
training frame. These tests pin that no kept frame sits within the purge
margin of a frame from a different split.
"""

import io
import itertools
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import fetch_rdd2022
from scripts import prepare_crosswalk_dataset as pcd
from training import train_detector


def frame(video, index, labelled=True):
    return pcd.Frame(Path(f"{video}_filename{index:04d}.jpg"), Path("x.txt") if labelled else None,
                     video, index)


class CrosswalkSplitTests(unittest.TestCase):
    def setUp(self):
        self.frames = [frame("01_4885", i) for i in range(0, 3000, 2)]
        self.frames += [frame("02_0036", i) for i in range(0, 6000, 3)]
        self.frames += [frame("02_0036", i, labelled=False) for i in range(1, 6000, 7)]

    def test_no_kept_frame_is_near_a_frame_of_another_split(self):
        purge = 30
        assignment = pcd.assign_splits(self.frames, block_size=300, purge=purge)
        by_video = {}
        for f, split in assignment.items():
            by_video.setdefault(f.video, []).append((f.index, split))
        for video, items in by_video.items():
            items.sort()
            for (i, split_a), (j, split_b) in itertools.pairwise(items):
                if split_a != split_b:
                    self.assertGreater(j - i, 2 * purge, f"{video}: frames {i} and {j} straddle splits")

    def test_all_three_splits_are_populated(self):
        assignment = pcd.assign_splits(self.frames, block_size=300, purge=30)
        self.assertEqual(set(assignment.values()), {"train", "val", "test"})

    def test_negatives_never_enter_training(self):
        assignment = pcd.assign_splits(self.frames, block_size=300, purge=30)
        for f, split in assignment.items():
            if f.label is None:
                self.assertNotEqual(split, "train")

    def test_split_is_deterministic(self):
        a = pcd.assign_splits(self.frames, block_size=300, purge=30)
        b = pcd.assign_splits(list(reversed(self.frames)), block_size=300, purge=30)
        self.assertEqual(a, b)

    def test_rejects_unexpected_file_names(self):
        with self.assertRaises(ValueError):
            pcd._parse(Path("holiday_photo.jpg"))


class RddFetchTests(unittest.TestCase):
    def test_country_parsing_handles_multiword_names(self):
        sample = fetch_rdd2022.Sample("train", "shard_000", "United_States_001234.jpg")
        self.assertEqual(sample.country, "United_States")
        self.assertEqual(fetch_rdd2022.Sample("train", "shard_000", "India_000007.jpg").country, "India")

    def test_missing_label_file_means_background_image(self):
        sample = fetch_rdd2022.Sample("train", "shard_002", "India_000001.jpg")

        def fake_get(url, **_):
            if url.endswith(".txt"):
                raise urllib.error.HTTPError(url, 404, "not found", {}, io.BytesIO())
            return b"\xff\xd8jpeg"

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(fetch_rdd2022, "_http_get", fake_get):
            out = Path(tmp)
            self.assertTrue(fetch_rdd2022._fetch_one(sample, out))
            self.assertEqual((out / "labels/train/India_000001.txt").read_text(), "")
            self.assertTrue((out / "images/train/India_000001.jpg").exists())
            # Second call is a no-op: downloads are resumable.
            self.assertFalse(fetch_rdd2022._fetch_one(sample, out))

    def test_train_list_caps_background_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "images/train").mkdir(parents=True)
            (out / "labels/train").mkdir(parents=True)
            for i in range(40):
                (out / f"images/train/India_{i:06d}.jpg").write_bytes(b"x")
                label = "3 0.5 0.5 0.1 0.1\n" if i < 10 else ""
                (out / f"labels/train/India_{i:06d}.txt").write_text(label)
            path, composition = fetch_rdd2022.write_train_list(out, background_fraction=0.25, seed=0)
            lines = path.read_text().split()
            self.assertEqual(composition, {"positive": 10, "background": 3})
            self.assertEqual(len(lines), 13)
            with self.assertRaises(ValueError):
                fetch_rdd2022.write_train_list(out, background_fraction=1.0, seed=0)

    def test_data_yaml_lists_classes_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = fetch_rdd2022.write_data_yaml(Path(tmp), "train.txt").read_text()
            self.assertIn("train: train.txt", text)
            self.assertIn("3: pothole", text)


class TrainDetectorTests(unittest.TestCase):
    def test_config_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("name: x\ndata: d.yaml\nlearning_rate: 0.1\n")
            with self.assertRaises(ValueError):
                train_detector.DetectorConfig.load(path)

    def test_config_resolves_data_relative_to_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.yaml"
            path.write_text("name: x\ndata: datasets/x/data.yaml\nhours: 0.5\n")
            config = train_detector.DetectorConfig.load(path)
            self.assertTrue(Path(config.data).is_absolute())
            self.assertEqual(config.hours, 0.5)

    def test_shipped_configs_are_valid(self):
        config_dir = Path(train_detector.PROJECT_ROOT) / "configs" / "detectors"
        configs = sorted(config_dir.glob("*.yaml"))
        self.assertTrue(configs)
        for path in configs:
            config = train_detector.DetectorConfig.load(path)
            self.assertEqual(config.name, path.stem)
            self.assertEqual(config.imgsz % 32, 0, "YOLO strides need a multiple of 32")

    def test_gpu_overrides(self):
        import argparse
        config = train_detector.DetectorConfig(name="x", data="d.yaml", hours=3.5, epochs=80)
        args = argparse.Namespace(hours=None, epochs=100, imgsz=640, device="0", batch=32)
        train_detector.apply_overrides(config, args)
        self.assertEqual((config.epochs, config.hours, config.imgsz, config.device, config.batch),
                         (100, None, 640, "0", 32))
        with self.assertRaises(ValueError):
            train_detector.apply_overrides(config, argparse.Namespace(imgsz=500))

    def test_serving_threshold_is_the_f1_peak(self):
        px = np.linspace(0, 1, 1000)
        peaked = np.exp(-((px - 0.42) ** 2) / 0.01)      # F1 peaks at confidence 0.42
        flat_low = np.exp(-((px - 0.01) ** 2) / 0.001)   # would pick ~0.01 without the floor
        thresholds = train_detector.serving_thresholds(
            np.stack([peaked, flat_low]), px, [0, 2], {0: "a", 1: "b", 2: "c"})
        self.assertAlmostEqual(thresholds["a"], 0.42, delta=0.01)
        self.assertEqual(thresholds["b"], 0.25)  # class absent from val keeps the default
        self.assertEqual(thresholds["c"], 0.05)  # clamped to the floor

    def test_split_images_pairs_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images/val/shard").mkdir(parents=True)
            (root / "images/val/shard/a.jpg").write_bytes(b"x")
            (root / "images/val/notes.md").write_text("ignored")
            (root / "data.yaml").write_text(f"path: {root.as_posix()}\nval: images/val\n")
            pairs = train_detector.split_images(root / "data.yaml", "val")
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][1], root / "labels/val/shard/a.txt")


if __name__ == "__main__":
    unittest.main()
