"""
Download the "Cracks and Potholes in Road Images" dataset and turn its real
annotations into training patches.

Source: https://github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset
COCO conversion used here: https://github.com/andrijdavid/Cracks-and-Potholes-in-Road-Images-Dataset
Collected by DNIT, the Brazilian federal highway department: 2,235 road
photographs with 1,921 crack, 564 pothole and 2,235 lane annotations
(bounding boxes and polygon segmentation).

    python -m scripts.fetch_cracks_potholes_dataset --limit 800
    python -m scripts.fetch_cracks_potholes_dataset --limit 2235 --workers 12

What it does, per image: downloads it, crops each annotated defect with a
margin of surrounding road, drops near-duplicate crops by perceptual hash, and
writes the result into datasets/<class folder>/real_images/ ready for
training. Images that carry only a lane annotation and no defect become
sound-pavement examples, which is exactly the hard-negative class this
project was short of.

Crops are named `cpr_<image id>_<annotation id>.jpg` so they are traceable
back to the source annotation, and `--undo` removes everything this script
added.
"""

import argparse
import concurrent.futures
import io
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import numpy as np
from PIL import Image

from data.image_dataset import CLASS_FOLDERS, DATASETS_ROOT
from models.forensic_audit_engine import ForensicDuplicateHasher

REPO_RAW = "https://raw.githubusercontent.com/andrijdavid/Cracks-and-Potholes-in-Road-Images-Dataset/main"
COCO_URL = f"{REPO_RAW}/coco.json"
IMAGE_BASE = f"{REPO_RAW}/images"
CACHE_DIR = os.path.join(DATASETS_ROOT, "incoming", "cracks_potholes_dnit")
PREFIX = "cpr_"

CRACK_CLASS, POTHOLE_CLASS, NORMAL_CLASS = 1, 2, 0
CATEGORY_TO_CLASS = {"crack": CRACK_CLASS, "pothole": POTHOLE_CLASS}
MIN_BOX_PX = 28
CONTEXT = 0.15


def class_dir(cls):
    return os.path.join(DATASETS_ROOT, CLASS_FOLDERS[cls][0], "real_images")


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "ROAD-SHIELD/3.0 (SIH 2026 student project)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_coco(refresh=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "coco.json")
    if refresh or not os.path.exists(path):
        print(f"[1/4] downloading annotations ({COCO_URL})")
        data = fetch(COCO_URL, timeout=180)
        with open(path, "wb") as fh:
            fh.write(data)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=800, help="how many source photographs to use (max 2235)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-per-class", type=int, default=1200)
    ap.add_argument("--refresh", action="store_true", help="re-download the annotation file")
    ap.add_argument("--undo", action="store_true", help="remove everything this script added")
    args = ap.parse_args()

    if args.undo:
        removed = 0
        for cls in (CRACK_CLASS, POTHOLE_CLASS, NORMAL_CLASS):
            d = class_dir(cls)
            if not os.path.isdir(d):
                continue
            for f in list(os.listdir(d)):
                if f.startswith(PREFIX):
                    os.remove(os.path.join(d, f))
                    removed += 1
        print(f"removed {removed} crops added by this script")
        return

    coco = load_coco(refresh=args.refresh)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    images = coco["images"][: args.limit]
    print(f"[2/4] {len(coco['images'])} photographs available, using {len(images)}; "
          f"{len(coco['annotations'])} annotations in total")

    hasher = ForensicDuplicateHasher(hash_size=8, duplicate_hamming_threshold=4)
    seen, written, skipped = [], Counter(), Counter()
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()

    def get_image(meta):
        cached = os.path.join(CACHE_DIR, meta["file_name"])
        if os.path.exists(cached):
            return meta, open(cached, "rb").read()
        try:
            blob = fetch(f"{IMAGE_BASE}/{meta['file_name']}")
            with open(cached, "wb") as fh:
                fh.write(blob)
            return meta, blob
        except Exception as e:
            return meta, e

    print(f"[3/4] downloading and cropping with {args.workers} workers ...")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for meta, blob in pool.map(get_image, images):
            done += 1
            if done % 100 == 0:
                print(f"      {done}/{len(images)} photographs, {sum(written.values())} crops "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if isinstance(blob, Exception):
                skipped["download_failed"] += 1
                continue
            try:
                img = Image.open(io.BytesIO(blob)).convert("RGB")
            except Exception:
                skipped["unreadable"] += 1
                continue
            W, H = img.size
            anns = by_image.get(meta["id"], [])
            defects = [a for a in anns if cats.get(a["category_id"]) in CATEGORY_TO_CLASS]

            for ann in defects:
                cls = CATEGORY_TO_CLASS[cats[ann["category_id"]]]
                if written[cls] >= args.max_per_class:
                    continue
                x, y, w, h = ann["bbox"]
                if w < MIN_BOX_PX or h < MIN_BOX_PX:
                    skipped["box_too_small"] += 1
                    continue
                px, py = w * CONTEXT, h * CONTEXT
                crop = img.crop((max(0, int(x - px)), max(0, int(y - py)),
                                 min(W, int(x + w + px)), min(H, int(y + h + py))))
                digest = hasher.compute_hash(np.asarray(crop.resize((64, 64)), dtype=np.uint8))
                if any(hasher.hamming_distance(digest, prev) <= 4 for prev in seen):
                    skipped["near_duplicate"] += 1
                    continue
                seen.append(digest)
                out = os.path.join(class_dir(cls), f"{PREFIX}{meta['id']}_{ann['id']}.jpg")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                crop.save(out, quality=92)
                written[cls] += 1

            # A photograph with a lane marking but no crack or pothole is a
            # genuine sound-pavement example - the class this project lacked.
            if not defects and anns and written[NORMAL_CLASS] < args.max_per_class:
                band = img.crop((int(W * 0.15), int(H * 0.45), int(W * 0.85), int(H * 0.95)))
                digest = hasher.compute_hash(np.asarray(band.resize((64, 64)), dtype=np.uint8))
                if any(hasher.hamming_distance(digest, prev) <= 4 for prev in seen):
                    skipped["near_duplicate"] += 1
                else:
                    seen.append(digest)
                    band.save(os.path.join(class_dir(NORMAL_CLASS), f"{PREFIX}{meta['id']}_road.jpg"), quality=92)
                    written[NORMAL_CLASS] += 1

    print(f"\n[4/4] done in {time.time() - t0:.0f}s")
    for cls, n in sorted(written.items()):
        print(f"      {CLASS_FOLDERS[cls][1][:44]:46s} +{n} images")
    print(f"      skipped: {dict(skipped)}")
    print("\nRetrain:  python -m training.train_mega_suite")


if __name__ == "__main__":
    main()
