"""
Build a multi-country road-damage detection set from the full RDD2022 release, for a YOLO detector that
learns from about four times more photographs than RDD2022 India alone.

    python -m scripts.prepare_rdd2022_world                       # downloads the full release (~13 GB) if needed
    python -m scripts.prepare_rdd2022_world --zip /content/RDD2022_all.zip
    python -m scripts.prepare_rdd2022_world --countries Japan Czech   # a subset

Countries
---------
Japan, Czech, United_States and China_MotorBike by default: smartphone or vehicle cameras looking along the
road, like a bus camera. China_Drone is excluded (aerial, a different viewpoint). Norway is optional
(--with-norway): its photographs are very wide 3,650 px frames from a different rig.

What stays fixed - so the new detector is compared with the India-only one like for like
------------------------------------------------------------------------------------------
India's train / valid / test split is copied unchanged from datasets/rdd2022_india (written by
scripts/prepare_rdd2022_voc.py, seed 42). Layout:

    images/train        India train + every other country's 90% share
    images/valid        India valid + every other country's 10% share   (early stopping)
    images/valid_india  India valid only                                (model selection, confidence)
    images/test         India test only                                 (scored once, never used to choose)

Other countries are split 90/10 by photograph (seed 42); none of their photographs is in any test set.
Classes kept: D00 longitudinal, D10 transverse, D20 alligator crack, D40 pothole - the same as India. A
photograph whose only objects are other classes is skipped (it is not clean road); one with no objects at
all keeps an empty label file. Photographs wider than 1,280 px are resized (labels are normalised, so
unchanged).
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys
import zipfile

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from scripts.prepare_rdd2022_voc import KEEP, find_split_root, parse_xml  # noqa: E402

DEFAULT_COUNTRIES = ["Japan", "Czech", "United_States", "China_MotorBike"]
DOWNLOADS = os.path.join(ENGINE_ROOT, "datasets", "_downloads", "rdd2022")
RAW = os.path.join(ENGINE_ROOT, "datasets", "_downloads", "rdd2022_world")
INDIA = os.path.join(ENGINE_ROOT, "datasets", "rdd2022_india")
OUT = os.path.join(ENGINE_ROOT, "datasets", "rdd2022_world")
MAX_SIDE = 1280


def extract_countries(zpath, countries, dest=RAW):
    """Extract the wanted countries from the full release: nested per-country zips or plain folders."""
    os.makedirs(dest, exist_ok=True)
    done = []
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        for c in countries:
            target = os.path.join(dest, c)
            if os.path.isdir(target) and glob.glob(os.path.join(target, "**", "*.xml"), recursive=True):
                done.append(c)
                continue
            nested = [n for n in names if n.lower().endswith(".zip") and c.lower() in os.path.basename(n).lower()]
            if nested:
                print(f"  {c}: nested archive {nested[0]}", flush=True)
                inner = os.path.join(dest, os.path.basename(nested[0]))
                with z.open(nested[0]) as src, open(inner, "wb") as dst:
                    shutil.copyfileobj(src, dst, 16 << 20)
                with zipfile.ZipFile(inner) as zi:
                    zi.extractall(target)
                os.remove(inner)
                done.append(c)
                continue
            members = [n for n in names if f"/{c.lower()}/" in "/" + n.lower()]
            if members:
                print(f"  {c}: extracting {len(members)} files", flush=True)
                z.extractall(target, members=members)
                done.append(c)
            else:
                print(f"  {c}: not found in {os.path.basename(zpath)} - skipped", flush=True)
    return done


def country_root(c, base=RAW):
    """The folder holding train/images and train/annotations/xmls for a country."""
    for d, subdirs, _ in os.walk(os.path.join(base, c)):
        try:
            return find_split_root(d)            # raises SystemExit when d is not a split root
        except (SystemExit, Exception):
            continue
    return None


def split_others(stems, seed=42, valid_frac=0.10):
    """Deterministic 90/10 split by photograph."""
    stems = sorted(stems)
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_valid = int(round(len(stems) * valid_frac))
    return set(stems[:n_valid]), set(stems[n_valid:])


def yolo_lines(W, H, kept):
    lines = []
    for name, x0, y0, x1, y1 in kept:
        x0, x1 = max(0.0, min(x0, x1)), min(W, max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(H, max(y0, y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
        lines.append(f"{KEEP.index(name)} {cx:.6f} {cy:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return lines


def copy_image(src, dst, max_side=MAX_SIDE):
    from PIL import Image
    with Image.open(src) as im:
        if max(im.size) <= max_side:
            shutil.copy2(src, dst)
            return
        k = max_side / float(max(im.size))
        im.convert("RGB").resize((int(im.size[0] * k), int(im.size[1] * k))).save(dst, quality=90)


def add_country(c, root, out, counts):
    xmls = sorted(glob.glob(os.path.join(root, "annotations", "xmls", "*.xml")))
    rows = []
    for xp in xmls:
        stem = os.path.splitext(os.path.basename(xp))[0]
        img = os.path.join(root, "images", stem + ".jpg")
        if not os.path.exists(img):
            continue
        W, H, objs = parse_xml(xp)
        if W <= 0 or H <= 0:
            from PIL import Image
            with Image.open(img) as im:
                W, H = im.size
        kept = [o for o in objs if o[0] in KEEP]
        if objs and not kept:
            continue
        rows.append((stem, img, W, H, kept))
    valid, _train = split_others([r[0] for r in rows])
    c_counts = {"train": 0, "valid": 0, "boxes": {k: 0 for k in KEEP}}
    for stem, img, W, H, kept in rows:
        split = "valid" if stem in valid else "train"
        name = stem if stem.lower().startswith(c.lower()) else f"{c}_{stem}"
        copy_image(img, os.path.join(out, "images", split, name + ".jpg"))
        lines = yolo_lines(W, H, kept)
        with open(os.path.join(out, "labels", split, name + ".txt"), "w") as fh:
            fh.write("\n".join(lines))
        c_counts[split] += 1
        for ln in lines:
            c_counts["boxes"][KEEP[int(ln.split()[0])]] += 1
    counts[c] = c_counts


def add_india(india, out, counts):
    """India's own split, unchanged; valid also goes to valid_india, test only to test."""
    c = {"train": 0, "valid": 0, "valid_india": 0, "test": 0}
    for split, targets in (("train", ["train"]), ("valid", ["valid", "valid_india"]), ("test", ["test"])):
        for ip in sorted(glob.glob(os.path.join(india, "images", split, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(india, "labels", split, stem + ".txt")
            for t in targets:
                shutil.copy2(ip, os.path.join(out, "images", t, stem + ".jpg"))
                dst = os.path.join(out, "labels", t, stem + ".txt")
                if os.path.exists(lp):
                    shutil.copy2(lp, dst)
                else:
                    open(dst, "w").close()
                c[t] += 1
    counts["India"] = c


def build(countries, india=INDIA, out=OUT, raw=RAW):
    if not os.path.isdir(os.path.join(india, "images", "test")):
        sys.exit(f"RDD2022 India split not found at {india}: run scripts/prepare_rdd2022_voc.py first, so the "
                 f"India test photographs are the same ones the India-only detector was scored on")
    if os.path.isdir(out):
        shutil.rmtree(out)
    for split in ("train", "valid", "valid_india", "test"):
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)
    counts = {}
    add_india(india, out, counts)
    for c in countries:
        root = country_root(c, raw)
        if not root:
            print(f"  {c}: no labelled train folder found - skipped")
            continue
        add_country(c, root, out, counts)
        print(f"  {c}: {counts[c]['train']} train, {counts[c]['valid']} valid photographs", flush=True)
    with open(os.path.join(out, "data.yaml"), "w") as fh:
        fh.write("train: images/train\nval: images/valid\ntest: images/test\n"
                 f"nc: {len(KEEP)}\nnames: [{', '.join(KEEP)}]\n")
    total = {s: len(os.listdir(os.path.join(out, "images", s))) for s in ("train", "valid", "valid_india", "test")}
    manifest = {
        "countries": counts, "photographs": total,
        "india_split": "copied unchanged from datasets/rdd2022_india (scripts/prepare_rdd2022_voc.py, seed 42)",
        "others_split": "90/10 train/valid by photograph, seed 42; no other country is in any test set",
        "selection_split": "images/valid_india (India valid only)", "test_split": "images/test (India test only)",
        "classes": KEEP, "max_side_px": MAX_SIDE,
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"[rdd2022-world] photographs {total}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=None, help="the full RDD2022 release zip, if you already have it")
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--with-norway", action="store_true")
    ap.add_argument("--keep-archive", action="store_true")
    ap.add_argument("--india", default=INDIA)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    countries = list(a.countries or DEFAULT_COUNTRIES) + (["Norway"] if a.with_norway else [])

    missing = [c for c in countries if not country_root(c)]
    if missing:
        zpath = a.zip
        if not zpath:
            from scripts.fetch_rdd2022_india import FIGSHARE_ALL, download
            os.makedirs(DOWNLOADS, exist_ok=True)
            zpath = os.path.join(DOWNLOADS, "RDD2022_all.zip")
            if not (os.path.exists(zpath) and zipfile.is_zipfile(zpath)):
                zpath = download(FIGSHARE_ALL, zpath)
            if not zpath:
                sys.exit("the full RDD2022 release could not be downloaded; pass --zip with a copy you have")
        extract_countries(zpath, missing)
        if not a.keep_archive and not a.zip and os.path.exists(zpath):
            os.remove(zpath)
    return build(countries, a.india, a.out)


if __name__ == "__main__":
    main()
