"""
Head-to-head: learned zebra-crossing detector vs the geometric bar finder.

Both answer the same question on the same held-out dashcam frames: is there a
zebra crossing in this image? The learned model is scored through the ONNX
file the server actually loads, at the thresholds in its model card; the
geometric detector (models/marking_detector.py) is scored as the pipeline
calls it.

    python -m scripts.benchmark_crosswalk                     # test split
    python -m scripts.benchmark_crosswalk --split val --limit 200

Writes checkpoints/detectors/crosswalk_benchmark.json.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import marking_detector
from models.road_scene_perception import DETECTOR_DIR, RoadScenePerception
from training.train_detector import split_images

CROSSWALK_CLASS = 0


def presence_scores(truth, predicted):
    tp = sum(t and p for t, p in zip(truth, predicted, strict=True))
    fp = sum(p and not t for t, p in zip(truth, predicted, strict=True))
    fn = sum(t and not p for t, p in zip(truth, predicted, strict=True))
    negatives = sum(not t for t in truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "false_alarm_rate_on_crossing_free_frames": round(fp / negatives, 4) if negatives else None,
        "tp": tp, "fp": fp, "fn": fn,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="datasets/crosswalk/data.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="evaluate a deterministic subset")
    args = parser.parse_args(argv)

    data = Path(args.data)
    if not data.exists():
        print(f"{data} not found; build it with python -m scripts.prepare_crosswalk_dataset")
        return 1
    perception = RoadScenePerception()
    head = perception.heads["markings"]
    if not head.is_ready:
        print("crosswalk detector not trained; run python -m training.train_detector configs/detectors/crosswalk.yaml")
        return 1

    pairs = split_images(data, args.split)
    if args.limit:
        pairs = pairs[:: max(1, len(pairs) // args.limit)][: args.limit]

    truth, learned, geometric = [], [], []
    learned_ms, geometric_ms = [], []
    for image_path, label_path in pairs:
        labels = label_path.read_text().split("\n") if label_path.exists() else []
        truth.append(any(line.split() and int(line.split()[0]) == CROSSWALK_CLASS for line in labels))
        with Image.open(image_path) as img:
            frame = np.asarray(img.convert("RGB"))

        t0 = time.perf_counter()
        result = perception.analyze(frame, groups={"markings"})
        learned_ms.append((time.perf_counter() - t0) * 1000)
        learned.append(result["scene"]["zebra_crossing_visible"])

        t0 = time.perf_counter()
        zebra = marking_detector.detect_zebra(frame)
        geometric_ms.append((time.perf_counter() - t0) * 1000)
        geometric.append(bool(zebra.get("found") and zebra.get("is_crossing")))

    report = {
        "question": "does this frame contain a zebra crossing?",
        "data": f"{data.name}:{args.split}",
        "frames": len(pairs),
        "frames_with_crossing": sum(truth),
        "frames_without_crossing": len(truth) - sum(truth),
        "learned_yolo_onnx": {**presence_scores(truth, learned),
                              "median_ms": round(statistics.median(learned_ms), 1),
                              "thresholds": head.thresholds},
        "geometric_bar_finder": {**presence_scores(truth, geometric),
                                 "median_ms": round(statistics.median(geometric_ms), 1)},
    }
    out = Path(DETECTOR_DIR) / "crosswalk_benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
