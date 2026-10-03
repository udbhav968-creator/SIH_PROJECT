"""
Turn the official RDD2022 country archive (Pascal-VOC XML) into the YOLO layout
that scripts/ingest_rdd2022_india.py reads.

The official RDD2022_India.zip (CRDDC 2022) ships

    India/train/images/India_XXXXXX.jpg
    India/train/annotations/xmls/India_XXXXXX.xml
    India/test/images/...                      <- no public labels

so only the labelled train folder is usable. It is split here, by photograph,
into train / valid / test (70 / 15 / 15, seeded), and written as

    <out>/images/{train,valid,test}/India_XXXXXX.jpg
    <out>/labels/{train,valid,test}/India_XXXXXX.txt     (YOLO, normalised)
    <out>/data.yaml                                       names: D00 D10 D20 D40

Classes kept: D00 longitudinal crack, D10 transverse crack, D20 alligator crack,
D40 pothole. A photograph whose XML has no objects at all gets an empty label
file (the ingest uses those for normal-road crops). A photograph whose only
objects are other classes (D43/D44 blurred markings, D50 manhole, Repair, ...)
is skipped entirely: it is not "clean road", and an empty label would make the
ingest treat it as such.

    python -m scripts.prepare_rdd2022_voc --src /path/to/India --out datasets/rdd2022_india
"""
import argparse
import glob
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET

KEEP = ["D00", "D10", "D20", "D40"]


def parse_xml(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    W = float(size.findtext("width") or 0)
    H = float(size.findtext("height") or 0)
    objs = []
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip()
        bb = o.find("bndbox")
        if bb is None:
            continue
        x0, y0 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
        x1, y1 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        objs.append((name, x0, y0, x1, y1))
    return W, H, objs


def find_split_root(src):
    """Accept either .../India or .../India/train."""
    for cand in (os.path.join(src, "train"), src):
        if os.path.isdir(os.path.join(cand, "images")) and glob.glob(
                os.path.join(cand, "annotations", "xmls", "*.xml")):
            return cand
    raise SystemExit(f"no train/images + annotations/xmls under {src}")


def convert(src, out, seed=42, val_frac=0.15, test_frac=0.15):
    root = find_split_root(src)
    xmls = sorted(glob.glob(os.path.join(root, "annotations", "xmls", "*.xml")))
    usable = []
    skipped_other_only = 0
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
            skipped_other_only += 1
            continue
        usable.append((stem, img, W, H, kept))

    rng = random.Random(seed)
    rng.shuffle(usable)
    n = len(usable)
    n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
    splits = {"test": usable[:n_test], "valid": usable[n_test:n_test + n_val],
              "train": usable[n_test + n_val:]}

    if os.path.isdir(out):
        shutil.rmtree(out)
    counts = {}
    for split, rows in splits.items():
        idir = os.path.join(out, "images", split)
        ldir = os.path.join(out, "labels", split)
        os.makedirs(idir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        c = {k: 0 for k in KEEP}
        c["empty_photos"] = 0
        for stem, img, W, H, kept in rows:
            shutil.copy2(img, os.path.join(idir, stem + ".jpg"))
            lines = []
            for name, x0, y0, x1, y1 in kept:
                x0, x1 = max(0.0, min(x0, x1)), min(W, max(x0, x1))
                y0, y1 = max(0.0, min(y0, y1)), min(H, max(y0, y1))
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
                lines.append(f"{KEEP.index(name)} {cx:.6f} {cy:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
                c[name] += 1
            if not lines:
                c["empty_photos"] += 1
            with open(os.path.join(ldir, stem + ".txt"), "w") as fh:
                fh.write("\n".join(lines))
        c["photos"] = len(rows)
        counts[split] = c

    with open(os.path.join(out, "data.yaml"), "w") as fh:
        fh.write("train: images/train\nval: images/valid\ntest: images/test\n"
                 f"nc: {len(KEEP)}\nnames: [{', '.join(KEEP)}]\n")
    return {"source": os.path.abspath(root), "photos_with_xml": len(xmls),
            "skipped_other_classes_only": skipped_other_only, "splits": counts,
            "split_rule": f"by photograph, seed {seed}, {1 - val_frac - test_frac:.0%}/{val_frac:.0%}/{test_frac:.0%}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped RDD2022 India folder (contains train/)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "datasets", "rdd2022_india"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    import json
    rep = convert(a.src, os.path.abspath(a.out), seed=a.seed)
    print(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
