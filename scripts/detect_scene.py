"""
Run the full road-scene perception stack on images, folders or dashcam video.

    python -m scripts.detect_scene photo.jpg
    python -m scripts.detect_scene datasets/08_dashcam_video_streams/real_frames --out out/frames
    python -m scripts.detect_scene drive.mp4 --every 5 --out out/drive
    python -m scripts.detect_scene photo.jpg --only damage          # maintenance survey mode

For every input frame it writes an annotated JPEG (or, for video, one
annotated MP4) and appends one JSON line per frame to detections.jsonl, with
every box, the derived scene facts and the alerts. People (and plates, when the
plate model is trained) are blurred in written images unless --no-redact. A summary is printed at
the end.

Video frames are sampled every Nth frame; the output video keeps the source
frame rate divided by N so it plays back at real speed.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def iter_images(path):
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(path, name)
    else:
        yield path


def load_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def _record(source, frame_index, result):
    return {"source": source, "frame": frame_index, **{k: result[k] for k in
            ("image_size", "counts", "scene", "alerts", "detections", "latency_ms")}}


def render(perception, frame, result, redact):
    """Annotated copy of a frame, with people and plates blurred unless redaction is off."""
    from models.road_scene_perception import annotate
    return annotate(perception.redact(frame, result)[0] if redact else frame, result)


def process_images(perception, paths, out_dir, groups, log, redact=True):
    for path in paths:
        frame = load_rgb(path)
        result = perception.analyze(frame, groups=groups)
        stem = os.path.splitext(os.path.basename(path))[0]
        render(perception, frame, result, redact).save(os.path.join(out_dir, f"{stem}_annotated.jpg"), quality=90)
        log(_record(path, 0, result))


def process_video(perception, path, out_dir, groups, every, log, max_frames=None, redact=True):
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_annotated.mp4")
    index = processed = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if index % every == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = perception.analyze(rgb, groups=groups)
                drawn = cv2.cvtColor(np.asarray(render(perception, rgb, result, redact)), cv2.COLOR_RGB2BGR)
                if writer is None:
                    h, w = drawn.shape[:2]
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(1.0, fps / every), (w, h))
                writer.write(drawn)
                log(_record(path, index, result))
                processed += 1
                if max_frames and processed >= max_frames:
                    break
            index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect potholes, cracks, zebra crossings, "
                                                 "people and vehicles in road imagery.")
    parser.add_argument("inputs", nargs="+", help="image files, folders of images, or video files")
    parser.add_argument("--out", default="out/scene", help="output folder")
    parser.add_argument("--only", nargs="+", choices=["traffic", "damage", "markings"],
                        help="run only these detector groups")
    parser.add_argument("--every", type=int, default=5, help="video: analyse every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="video: stop after N analysed frames")
    parser.add_argument("--no-redact", action="store_true",
                        help="do not blur people and plates in the written images (blurred by default)")
    args = parser.parse_args(argv)

    from models.road_scene_perception import RoadScenePerception

    perception = RoadScenePerception()
    status = perception.describe()
    for key, info in status.items():
        print(f"  {key:9s} {'ready' if info['ready'] else 'NOT AVAILABLE'}  {info.get('backend') or ''}")
    if not perception.is_ready:
        print("No detector is available. Fetch or train them first:\n"
              "  python -m scripts.fetch_detector\n"
              "  python -m training.train_detector configs/detectors/road_damage.yaml\n"
              "  python -m training.train_detector configs/detectors/crosswalk.yaml")
        return 1

    os.makedirs(args.out, exist_ok=True)
    groups = set(args.only) if args.only else None
    totals, alert_totals, frames, latencies = Counter(), Counter(), 0, []
    started = time.perf_counter()

    with open(os.path.join(args.out, "detections.jsonl"), "w", encoding="utf-8") as jsonl:
        def log(record):
            nonlocal frames
            frames += 1
            totals.update(record["counts"])
            alert_totals.update(a["code"] for a in record["alerts"])
            latencies.append(record["latency_ms"]["total"])
            jsonl.write(json.dumps(record) + "\n")

        for item in args.inputs:
            if not os.path.exists(item):
                print(f"skip {item}: not found")
                continue
            if os.path.splitext(item)[1].lower() in VIDEO_EXTS:
                written = process_video(perception, item, args.out, groups, max(1, args.every), log,
                                        args.max_frames, redact=not args.no_redact)
                print(f"wrote {written}")
            else:
                process_images(perception, list(iter_images(item)), args.out, groups, log,
                               redact=not args.no_redact)

    elapsed = time.perf_counter() - started
    print(f"\n{frames} frame(s) in {elapsed:.1f}s"
          + (f", median {sorted(latencies)[len(latencies) // 2]:.0f} ms/frame" if latencies else ""))
    for name, count in totals.most_common():
        print(f"  {name:20s} {count}")
    for code, count in alert_totals.most_common():
        print(f"  alert {code:26s} {count} frame(s)")
    print(f"output: {os.path.abspath(args.out)}")
    return 0 if frames else 1


if __name__ == "__main__":
    sys.exit(main())
