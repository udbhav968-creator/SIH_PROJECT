"""Build a leakage-free zebra-crossing detection dataset from CDSet-3434.

CDSet-3434 (Zhang et al., "CDNet", Neural Computing and Applications 2022,
Apache-2.0) contains 3,434 dashcam frames labelled with two classes,
``crosswalk`` and ``guide_arrows``, plus 1,416 frames verified to contain no
crosswalk.

Why the official split is not used
----------------------------------
All frames come from three videos, and every frame in the official test split
sits one or two frames away from a training frame. Adjacent video frames are
near-identical, so a score on that split measures memorisation, not
generalisation. This script re-splits by **contiguous time blocks** within
each video and discards a purge margin of frames at every boundary between
blocks that land in different splits.

The crosswalk-free frames are kept out of training and placed in the val/test
splits of their own time block, so the evaluation can report how often the
model hallucinates a crossing on a road that has none.

Usage::

    python -m scripts.prepare_crosswalk_dataset --src datasets/CDSet --out datasets/crosswalk
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("prepare_crosswalk_dataset")

CLASS_NAMES = ("crosswalk", "guide_arrows")
DEFAULT_OUT = Path("datasets") / "crosswalk"  # what configs/detectors/crosswalk.yaml reads
FRAME_RE = re.compile(r"^(?P<video>\d+_\d+)_filename(?P<frame>\d+)$")
# Per-video block assignment cycle: 7 train, 1 val, 2 test blocks out of 10.
# Interleaving (rather than "last 30% is test") keeps every split covering
# the full range of lighting and road conditions within each drive.
SPLIT_CYCLE = ("train", "train", "val", "train", "train", "test",
               "train", "train", "test", "train")


@dataclass(frozen=True)
class Frame:
    image: Path
    label: Path | None  # None for verified crosswalk-free frames
    video: str
    index: int


def _parse(path: Path) -> tuple[str, int]:
    match = FRAME_RE.match(path.stem)
    if not match:
        raise ValueError(f"unexpected CDSet file name: {path.name}")
    return match["video"], int(match["frame"])


def collect_frames(src: Path) -> list[Frame]:
    yolo_root = src / "dataset_YOLO_format_3434"
    frames: list[Frame] = []
    for split in ("train", "test"):
        for image in sorted((yolo_root / "images" / split).glob("*.jpg")):
            video, index = _parse(image)
            label = yolo_root / "labels" / split / f"{image.stem}.txt"
            frames.append(Frame(image, label if label.exists() else None, video, index))

    labelled = {f.image.name for f in frames}
    negatives_list = src / "binary_testset_1770" / "negative.txt"
    image_dir = src / "binary_testset_1770" / "Images"
    for name in negatives_list.read_text().split():
        if name in labelled:
            continue
        image = image_dir / name
        if image.exists():
            video, index = _parse(image)
            frames.append(Frame(image, None, video, index))
    return frames


def block_split(index: int, block_size: int) -> str:
    """The split a frame index falls in, before any boundary purge."""
    return SPLIT_CYCLE[(index // block_size) % len(SPLIT_CYCLE)]


def near_boundary(index: int, block_size: int, purge: int) -> bool:
    split = block_split(index, block_size)
    return block_split(index - purge, block_size) != split or block_split(index + purge, block_size) != split


def assign_splits(frames: list[Frame], block_size: int, purge: int) -> dict[Frame, str]:
    """Map each frame to train/val/test by time block, dropping boundary frames."""
    by_video: dict[str, list[Frame]] = defaultdict(list)
    for frame in frames:
        by_video[frame.video].append(frame)

    assignment: dict[Frame, str] = {}
    for _video, video_frames in sorted(by_video.items()):
        def split_of(index: int) -> str:
            return block_split(index, block_size)

        for frame in video_frames:
            split = split_of(frame.index)
            # Purge frames whose neighbourhood reaches into a block of another split.
            if split_of(frame.index - purge) != split or split_of(frame.index + purge) != split:
                continue
            if frame.label is None and split == "train":
                continue  # negatives are evaluation-only
            assignment[frame] = split
    return assignment


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hard link: no extra disk space
    except OSError:
        shutil.copy2(src, dst)


def build(src: Path, out: Path, block_size: int, purge: int) -> dict:
    frames = collect_frames(src)
    assignment = assign_splits(frames, block_size, purge)

    stats: dict[str, Counter] = defaultdict(Counter)
    negatives: dict[str, list[str]] = defaultdict(list)
    for frame, split in sorted(assignment.items(), key=lambda kv: kv[0].image.name):
        _place(frame.image, out / "images" / split / frame.image.name)
        label_dst = out / "labels" / split / f"{frame.image.stem}.txt"
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        if frame.label is not None:
            text = frame.label.read_text()
            label_dst.write_text(text)
            stats[split]["positive_images"] += 1
            for line in text.splitlines():
                if line.strip():
                    stats[split][CLASS_NAMES[int(line.split()[0])]] += 1
        else:
            label_dst.write_text("")
            negatives[split].append(frame.image.name)
            stats[split]["negative_images"] += 1

    (out / "data.yaml").write_text(
        "\n".join([
            f"path: {out.absolute().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(CLASS_NAMES)}",
            "names:",
            *(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)),
            "",
        ]),
        encoding="utf-8",
    )
    manifest = {
        "source": "CDSet-3434 (Zhang et al. 2022), Apache-2.0",
        "split_method": f"time blocks of {block_size} frames per video, purge margin {purge} frames",
        "total_frames_seen": len(frames),
        "frames_kept": len(assignment),
        "frames_purged_at_boundaries": sum(1 for f in frames if f not in assignment
                                           and near_boundary(f.index, block_size, purge)),
        "negatives_kept_out_of_training": sum(1 for f in frames if f not in assignment
                                              and not near_boundary(f.index, block_size, purge)),
        "splits": {k: dict(v) for k, v in sorted(stats.items())},
        "negative_images": {k: sorted(v) for k, v in negatives.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leakage-free re-split of CDSet-3434")
    parser.add_argument("--src", type=Path, default=Path("datasets/CDSet"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--block-size", type=int, default=300, help="frames per time block")
    parser.add_argument("--purge", type=int, default=30, help="frames dropped at split boundaries")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not (args.src / "dataset_YOLO_format_3434").is_dir():
        LOG.error("CDSet not found under %s. Download CDSet.zip from "
                  "https://huggingface.co/datasets/zzd0225/crosswalk-detection-dataset", args.src)
        return 1
    manifest = build(args.src, args.out, args.block_size, args.purge)
    LOG.info("kept %d of %d frames: %d purged at split boundaries, %d crossing-free frames "
             "kept out of training by design", manifest["frames_kept"], manifest["total_frames_seen"],
             manifest["frames_purged_at_boundaries"], manifest["negatives_kept_out_of_training"])
    for split, counts in manifest["splits"].items():
        LOG.info("%-5s %s", split, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
