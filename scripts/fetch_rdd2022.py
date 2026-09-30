"""Download a country-filtered subset of RDD2022 in Ultralytics YOLO layout.

RDD2022 (Arya et al., arXiv:2209.08538, CC BY-SA 4.0) is the standard
benchmark for street-level road damage: 38,385 labelled images from six
countries with four CRDDC classes (D00 longitudinal crack, D10 transverse
crack, D20 alligator crack, D40 pothole). India is the largest single
contributor, which is why it is the default filter here.

The source is the Hugging Face mirror ``dronefreak/RDD2022``, which is
already converted to YOLO labels and sharded per image, so a subset can be
fetched without pulling the whole 9 GB archive.

Usage::

    python -m scripts.fetch_rdd2022                         # India, all splits
    python -m scripts.fetch_rdd2022 --countries India Japan --max-per-split 2000
    python -m scripts.fetch_rdd2022 --dry-run               # counts only

Downloads are resumable: files already on disk are skipped, and every file is
written to a temporary name and renamed only once complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("fetch_rdd2022")

REPO_URL = "https://huggingface.co/datasets/dronefreak/RDD2022/resolve/main/data"
CLASS_NAMES = ("longitudinal_crack", "transverse_crack", "alligator_crack", "pothole")
# Shard layout of the mirror; each shard holds ~3,000 images.
SHARDS = {
    "train": [f"shard_{i:03d}" for i in range(9)],
    "valid": [f"shard_{i:03d}" for i in range(2)],
    "test": [f"shard_{i:03d}" for i in range(2)],
}
DEFAULT_OUT = Path("datasets") / "rdd2022"


@dataclass(frozen=True)
class Sample:
    split: str
    shard: str
    file_name: str

    @property
    def stem(self) -> str:
        return Path(self.file_name).stem

    @property
    def country(self) -> str:
        # "United_States_000123.jpg" -> "United_States"
        return self.stem.rsplit("_", 1)[0]


def _http_get(url: str, *, retries: int = 6, timeout: float = 60.0) -> bytes:
    headers = {"User-Agent": "road-shield-dataset-fetcher/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            # 404 is permanent; 429 and 5xx are worth retrying.
            if exc.code == 404 or attempt == retries:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if exc.code == 429 and retry_after and retry_after.isdigit():
                delay = max(delay, float(retry_after))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(delay + random.random())
        delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def list_samples(splits: list[str], countries: set[str] | None) -> list[Sample]:
    samples: list[Sample] = []
    for split in splits:
        for shard in SHARDS[split]:
            raw = _http_get(f"{REPO_URL}/images/{split}/{shard}/metadata.jsonl")
            for line in raw.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                sample = Sample(split, shard, json.loads(line)["file_name"])
                if countries is None or sample.country in countries:
                    samples.append(sample)
    return samples


def _fetch_one(sample: Sample, out_dir: Path) -> bool:
    """Fetch image and label for one sample. Returns True if anything was downloaded."""
    image_path = out_dir / "images" / sample.split / sample.file_name
    label_path = out_dir / "labels" / sample.split / f"{sample.stem}.txt"
    fetched = False
    if not label_path.exists():
        url = f"{REPO_URL}/labels/{sample.split}/{sample.shard}/{sample.stem}.txt"
        try:
            payload = _http_get(url)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            payload = b""  # background image: no in-taxonomy damage
        _write_atomic(label_path, payload)
        fetched = True
    if not image_path.exists():
        url = f"{REPO_URL}/images/{sample.split}/{sample.shard}/{sample.file_name}"
        _write_atomic(image_path, _http_get(url))
        fetched = True
    return fetched


def write_train_list(out_dir: Path, background_fraction: float, seed: int) -> tuple[Path, dict[str, int]]:
    """Write train.txt with every damaged image plus a capped share of clean-road images.

    Roughly 60% of RDD2022 India frames contain no in-taxonomy damage. Some
    background is essential (it is how the model learns that a shadow or a
    tar patch is not a pothole); beyond that it mostly costs compute.
    ``background_fraction`` is the share of background images in the result.
    """
    positives: list[Path] = []
    backgrounds: list[Path] = []
    for image in sorted((out_dir / "images" / "train").glob("*.jpg")):
        label = out_dir / "labels" / "train" / f"{image.stem}.txt"
        has_boxes = label.exists() and label.read_text().strip() != ""
        (positives if has_boxes else backgrounds).append(image)

    if not 0.0 <= background_fraction < 1.0:
        raise ValueError("background_fraction must be in [0, 1)")
    wanted = round(len(positives) * background_fraction / (1.0 - background_fraction))
    rng = random.Random(seed)
    rng.shuffle(backgrounds)
    chosen = sorted(positives + backgrounds[:wanted])

    path = out_dir / "train.txt"
    path.write_text("\n".join(p.absolute().as_posix() for p in chosen) + "\n", encoding="utf-8")
    return path, {"positive": len(positives), "background": min(wanted, len(backgrounds))}


def write_data_yaml(out_dir: Path, train_entry: str = "images/train") -> Path:
    lines = [
        f"path: {out_dir.absolute().as_posix()}",
        f"train: {train_entry}",
        "val: images/valid",
        "test: images/test",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
        *(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)),
        "",
    ]
    path = out_dir / "data.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--countries", nargs="+", default=["India"],
                        help="country prefixes to keep, or 'all'")
    parser.add_argument("--splits", nargs="+", default=list(SHARDS), choices=list(SHARDS))
    parser.add_argument("--max-per-split", type=int, default=None,
                        help="random subset size per split (seeded, reproducible)")
    parser.add_argument("--train-background-fraction", type=float, default=None,
                        help="cap clean-road images to this share of the training list, e.g. 0.33")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    countries = None if args.countries == ["all"] else set(args.countries)
    samples = list_samples(args.splits, countries)
    if args.max_per_split:
        rng = random.Random(args.seed)
        subset: list[Sample] = []
        for split in args.splits:
            in_split = [s for s in samples if s.split == split]
            rng.shuffle(in_split)
            subset.extend(in_split[: args.max_per_split])
        samples = subset

    for split, count in sorted(Counter(s.split for s in samples).items()):
        LOG.info("%-5s %6d images", split, count)
    if args.dry_run:
        return 0
    if not samples:
        LOG.error("no images matched countries=%s", args.countries)
        return 1

    failures = 0
    fetched = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_one, s, args.out): s for s in samples}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                fetched += future.result()
            except Exception as exc:
                failures += 1
                LOG.warning("failed %s: %s", futures[future].file_name, exc)
            if done % 500 == 0 or done == len(samples):
                LOG.info("%d/%d done (%d downloaded, %d failed)", done, len(samples), fetched, failures)

    train_entry = "images/train"
    if args.train_background_fraction is not None and "train" in args.splits:
        train_list, composition = write_train_list(args.out, args.train_background_fraction, args.seed)
        train_entry = train_list.name
        LOG.info("training list %s: %s", train_list, composition)
    yaml_path = write_data_yaml(args.out, train_entry)
    LOG.info("wrote %s", yaml_path)
    # A few failures on a flaky link are tolerable; rerunning resumes them.
    return 0 if failures <= len(samples) * 0.01 else 2


if __name__ == "__main__":
    sys.exit(main())
