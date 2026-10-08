"""
Training crops for the image classifier from the OTHER countries of the multi-country RDD2022 set
(scripts/prepare_rdd2022_world.py): Japan, Czech, United States, China.

    python -m scripts.ingest_rdd2022_world [--max-per-class 2500] [--normal 1200]

Crops are written into the training folders with the prefix "rddw_" (one group per photograph, so crops
of one photograph never straddle train and test). India is never read here - its training crops come from
scripts/ingest_rdd2022_india.py (prefix rddin_), and its test photographs stay in
datasets/_eval_rdd2022_india, which no training run reads. Only train/valid photographs of the other
countries are used; no country but India has a test split.

    python -m scripts.ingest_rdd2022_world --undo      # remove every rddw_ crop
"""
import argparse
import glob
import json
import os
import random
import sys

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from scripts.ingest_rdd2022_india import (CLASS_FOLDERS, CRACK, EVAL_NAMES, NORMAL, POTHOLE,  # noqa: E402
                                          class_for, crops_from, read_names, train_dir)

PREFIX = "rddw_"
WORLD = os.path.join(ENGINE_ROOT, "datasets", "rdd2022_world")


def clean_previous():
    n = 0
    for cls in (NORMAL, CRACK, POTHOLE):
        for f in glob.glob(os.path.join(train_dir(cls), PREFIX + "*.jpg")):
            os.remove(f)
            n += 1
    return n


def country_of(stem):
    """'Japan_000123' -> 'Japan'; 'China_MotorBike_000001' -> 'China_MotorBike'; 'United_States_..' likewise."""
    parts = stem.split("_")
    return "_".join(parts[:2]) if parts[0] in ("China", "United") and len(parts) > 2 else parts[0]


def interleave_countries(photos, seed=42):
    """Each country's photographs shuffled with a fixed seed, then taken in turn, one country at a time, so
    the crop caps fill from every country instead of whichever sorts first alphabetically."""
    groups = {}
    for row in photos:
        groups.setdefault(country_of(row[0]), []).append(row)
    order = sorted(groups)
    for c in order:
        groups[c].sort()
        random.Random(f"{seed}:{c}").shuffle(groups[c])
    longest = max((len(g) for g in groups.values()), default=0)
    return [groups[c][k] for k in range(longest) for c in order if k < len(groups[c])]


def other_country_photos(root):
    """(image, label) pairs of every non-India photograph in train/valid, in a fixed order."""
    out = []
    for split in ("train", "valid"):
        for ip in sorted(glob.glob(os.path.join(root, "images", split, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            if stem.lower().startswith("india"):
                continue
            out.append((stem, ip, os.path.join(root, "labels", split, stem + ".txt")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=WORLD)
    ap.add_argument("--max-per-class", type=int, default=2500, help="crack and pothole crops")
    ap.add_argument("--normal", type=int, default=1200, help="normal-road crops")
    ap.add_argument("--undo", action="store_true")
    a = ap.parse_args(argv)
    removed = clean_previous()
    if a.undo:
        print(f"removed {removed} rddw_ crops")
        return {"removed": removed}
    names = read_names(a.root)
    if not names:
        sys.exit(f"No class names in a .yaml under {a.root} - run scripts/prepare_rdd2022_world.py first")
    mapping = {i: class_for(n) for i, n in enumerate(names)}
    caps = {CRACK: a.max_per_class, POTHOLE: a.max_per_class, NORMAL: a.normal}
    counts = {k: 0 for k in caps}
    photos = 0
    by_country = {}
    for stem, ip, lp in interleave_countries(other_country_photos(a.root)):
        if all(counts[k] >= caps[k] for k in caps):
            break
        used = False
        for k, (cls, crop) in enumerate(crops_from(ip, lp, mapping)):
            if counts[cls] >= caps[cls]:
                continue
            crop.save(os.path.join(train_dir(cls), f"{PREFIX}{stem}_{k}.jpg"), quality=92)
            counts[cls] += 1
            used = True
        if used:
            photos += 1
            country = country_of(stem)
            by_country[country] = by_country.get(country, 0) + 1
    pretty = {EVAL_NAMES[k]: v for k, v in counts.items()}
    manifest = {"source": a.root, "prefix": PREFIX, "training_crops": pretty, "photographs": photos,
                "photographs_by_country": by_country,
                "order": "each country shuffled (seed 42), countries taken in turn until the caps are reached",
                "rule": "other countries' train/valid photographs only; India and every test photograph excluded"}
    with open(os.path.join(ENGINE_ROOT, "datasets", "rdd2022_world_crops_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"[RDD world] training crops {pretty} from {photos} photographs {by_country}")
    return manifest


if __name__ == "__main__":
    main()
