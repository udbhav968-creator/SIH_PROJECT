"""Download and prepare the licence-plate detection data used for privacy redaction.

Source: "Vehicle Registration Plates" (Augmented Startups, Roboflow Universe,
CC BY 4.0), mirrored on Hugging Face as keremberke/license-plate-object-detection:
8,823 images, one class, COCO boxes.

The plate detector exists to BLUR plates before an image leaves a bus (DPDP
Act 2023). It never reads them. Finding the plate-shaped region is what
matters, which is why an international dataset is acceptable here even
though the fleet is Indian.

Two things the Roboflow export gets wrong and this script fixes:

* Split leakage. Roboflow names each exported copy <source>_<ext>.rf.<hash>,
  and the same source photograph can appear in more than one split. The
  images are regrouped by source name and re-split 80/10/10 by a hash of
  that name, so no photograph is on both sides.
* Hidden synthetic data. 3,904 of the 8,823 images (44%) are rendered
  plates ("...PlateGen..."). They are kept for training (they help
  localisation) but never enter val or test, so every reported score is on
  real photographs.

    python -m scripts.fetch_license_plates            # -> datasets/license_plates
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

LOG = logging.getLogger("fetch_license_plates")
BASE_URL = "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data"
ARCHIVES = ("train.zip", "valid.zip", "test.zip")
DEFAULT_OUT = Path("datasets") / "license_plates"  # what configs/detectors/license_plate.yaml reads
SOURCE_RE = re.compile(r"^(?P<source>.+?)(?:_(?:jpg|jpeg|png|JPG|PNG))?\.rf\.[0-9a-f]+\.\w+$")


def source_name(file_name):
    """'Cars224_png_jpg.rf.35..84.jpg' -> 'Cars224_png' (the photograph it came from)."""
    match = SOURCE_RE.match(file_name)
    return match["source"] if match else Path(file_name).stem


def split_for(source, fractions=(0.8, 0.1)):
    bucket = int(hashlib.sha256(source.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < fractions[0]:
        return "train"
    return "val" if bucket < fractions[0] + fractions[1] else "test"


def coco_to_yolo(annotations, width, height):
    lines = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue  # degenerate boxes exist in the export
        cx, cy = (x + w / 2) / width, (y + h / 2) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
    return lines


def _download(url, cache):
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        LOG.info("downloading %s", url)
        tmp = cache.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        tmp.replace(cache)
    return cache


def build(out, cache_dir):
    stats = Counter()
    per_split = Counter()
    for name in ARCHIVES:
        with zipfile.ZipFile(_download(f"{BASE_URL}/{name}", cache_dir / name)) as archive:
            coco = json.loads(archive.read("_annotations.coco.json"))
            by_image = {}
            for ann in coco["annotations"]:
                by_image.setdefault(ann["image_id"], []).append(ann)
            for image in coco["images"]:
                source = source_name(image["file_name"])
                synthetic = "PlateGen" in image["file_name"]
                # Rendered plates help localisation but are easy; scoring on them
                # would flatter the model, so they only ever train.
                split = "train" if synthetic else split_for(source)
                lines = coco_to_yolo(by_image.get(image["id"], []), image["width"], image["height"])
                image_path = out / "images" / split / image["file_name"]
                label_path = out / "labels" / split / (Path(image["file_name"]).stem + ".txt")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                if not image_path.exists():
                    image_path.write_bytes(archive.read(image["file_name"]))
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
                per_split[split] += 1
                stats["boxes"] += len(lines)
                stats["synthetic"] += synthetic
                stats[f"source::{source}"] = 1

    (out / "data.yaml").write_text(
        f"path: {out.absolute().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: license_plate\n", encoding="utf-8")
    manifest = {
        "source": "Vehicle Registration Plates, Augmented Startups (Roboflow Universe), CC BY 4.0",
        "split_method": "real photographs by source name (sha256), 80/10/10; synthetic renders train only",
        "images_per_split": dict(per_split),
        "boxes": stats["boxes"],
        "distinct_source_photographs": sum(1 for k in stats if k.startswith("source::")),
        "synthetic_rendered_plate_images": stats["synthetic"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=Path("datasets") / "_downloads" / "license_plates")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = build(args.out, args.cache)
    LOG.info("%s", json.dumps(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
