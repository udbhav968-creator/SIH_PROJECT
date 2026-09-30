"""
Measure the frame-quality gate on real road frames and degraded copies of them.

    python -m scripts.calibrate_frame_gate
    python -m scripts.calibrate_frame_gate --images "datasets/my_bus_camera/*.jpg"

Reports, for the shipped thresholds in models/frame_gate.py, the share of
real frames kept (should be ~100%) and the share of each degradation caught.
Run it on footage from a new camera model before trusting the gate on it.
Writes checkpoints/frame_gate_report.json.
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import frame_gate

DEFAULT_SOURCES = {
    "rdd2022_india": "datasets/rdd2022_india/images/valid/*.jpg",
    "cdset_dashcam": "datasets/crosswalk/images/val/*.jpg",
    "project_photos": "datasets/0[1-3]_*/real_images/*.jpg",
}


def degradations(rng):
    import cv2
    return {
        "lens_covered": lambda f: np.clip(rng.normal(6, 2, f.shape), 0, 255).astype(np.uint8),
        "severe_defocus": lambda f: cv2.GaussianBlur(f, (0, 0), 12),
        "moderate_defocus": lambda f: cv2.GaussianBlur(f, (0, 0), 5),
        "horizontal_motion_blur": lambda f: cv2.filter2D(f, -1, np.ones((1, 41)) / 41),
        "whiteout": lambda f: np.clip(f.astype(int) * 4 + 150, 0, 255).astype(np.uint8),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", action="append", help="glob of real road frames (repeatable)")
    parser.add_argument("--per-source", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from PIL import Image

    sources = {f"custom_{i}": g for i, g in enumerate(args.images)} if args.images else DEFAULT_SOURCES
    rng_py, rng = random.Random(args.seed), np.random.default_rng(args.seed)
    frames, per_source = [], {}
    for name, pattern in sources.items():
        files = sorted(glob.glob(pattern))
        picked = rng_py.sample(files, min(args.per_source, len(files)))
        loaded = [np.asarray(Image.open(f).convert("RGB")) for f in picked]
        if loaded:
            kept = np.mean([frame_gate.frame_quality(f)["analysable"] for f in loaded])
            per_source[name] = {"frames": len(loaded), "kept_pct": round(100 * kept, 1)}
            frames.extend(loaded)
    if not frames:
        print("no frames found; pass --images or fetch the datasets first")
        return 1

    kept_all = np.mean([frame_gate.frame_quality(f)["analysable"] for f in frames])
    caught = {}
    subset = frames[: min(100, len(frames))]
    for name, degrade in degradations(rng).items():
        caught[name] = round(100 * np.mean([not frame_gate.frame_quality(degrade(f))["analysable"]
                                            for f in subset]), 1)

    report = {
        "thresholds": frame_gate.frame_quality(frames[0])["thresholds"],
        "real_frames": len(frames), "real_frames_kept_pct": round(100 * kept_all, 1),
        "per_source": per_source, "degraded_frames_per_type": len(subset), "caught_pct": caught,
    }
    out = os.path.join(os.path.dirname(frame_gate.__file__), "..", "checkpoints", "frame_gate_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
