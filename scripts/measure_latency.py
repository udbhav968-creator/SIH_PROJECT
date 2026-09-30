"""
Measure road-scene perception latency per detector and for the full stack.

    python -m scripts.measure_latency                       # a held-out CDSet frame
    python -m scripts.measure_latency --image frame.jpg --runs 30

Times RoadScenePerception.analyze (quality gate included) on one frame,
per detector group and all together, and writes the medians to
checkpoints/perception_latency.json. Numbers depend on the machine; the file
records which one.
"""

import argparse
import glob
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.road_scene_perception import CKPT_DIR, RoadScenePerception


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image", help="frame to time (default: a held-out CDSet test frame)")
    parser.add_argument("--runs", type=int, default=15)
    args = parser.parse_args(argv)

    from PIL import Image

    path = args.image or next(iter(sorted(glob.glob("datasets/crosswalk/images/test/*.jpg"))[50:51]), None)
    if not path:
        print("no frame given and datasets/crosswalk is not prepared; pass --image")
        return 1
    frame = np.asarray(Image.open(path).convert("RGB"))
    perception = RoadScenePerception()
    ready = [key for key, head in perception.heads.items() if head.is_ready]

    def median_ms(groups):
        perception.analyze(frame, groups=groups)  # warm-up
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            perception.analyze(frame, groups=groups)
            times.append((time.perf_counter() - t0) * 1000)
        return round(statistics.median(times), 1)

    result = {key: median_ms({key}) for key in ready}
    if len(ready) > 1:
        result["all " + ("three" if len(ready) == 3 else str(len(ready)))] = median_ms(set(ready))
    report = {"frame": f"{frame.shape[1]}x{frame.shape[0]}", "runs": args.runs, "median_ms": result,
              "machine": platform.processor() or platform.machine()}
    out = os.path.join(CKPT_DIR, "perception_latency.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
