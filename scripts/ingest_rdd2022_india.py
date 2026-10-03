"""
Bring RDD2022 India (YOLO layout) into ROAD-SHIELD.

    datasets/rdd2022_india/{images,labels}/{train,valid,test}/India_XXXXXX.{jpg,txt}
    datasets/rdd2022_india/*.yaml        class names (D00/D10/D20 cracks, D40 pothole, ...)

train + valid  -> crops written into the training folders, prefixed "rddin_"
test           -> crops written to datasets/_eval_rdd2022_india/<class>/, a folder
                  training never reads, so Indian accuracy is measured on Indian
                  photographs the model has not seen.

Crack and pothole crops come from the labelled boxes (15% padding). Photographs
with an empty label file supply "normal road" crops from the lower carriageway
band, capped, because RDD only labels some damage types and an unlabelled photo
is clean only for those.

    python -m scripts.ingest_rdd2022_india [--max-per-class 1200] [--normal 600]
"""
import argparse, glob, json, os, re, shutil, sys

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE_ROOT)
from data.image_dataset import CLASS_FOLDERS                          # noqa: E402

DATASETS = os.path.join(ENGINE_ROOT, "datasets")
EVAL_DIR = os.path.join(DATASETS, "_eval_rdd2022_india")
PREFIX = "rddin_"
NORMAL, CRACK, POTHOLE = 0, 1, 2
EVAL_NAMES = {NORMAL: "normal", CRACK: "crack", POTHOLE: "pothole"}


def class_for(name):
    """Map an RDD class name to ROAD-SHIELD's classes; None = not used."""
    n = str(name).strip().lower()
    if re.search(r"\bd40\b|pothole", n):
        return POTHOLE
    if re.search(r"\bd0\d\b|\bd1\d\b|\bd2\d\b|crack|alligator|longitudinal|transverse", n):
        return CRACK
    return None


def read_names(root):
    for y in sorted(glob.glob(os.path.join(root, "*.yaml")) + glob.glob(os.path.join(root, "*.yml"))):
        txt = open(y, encoding="utf-8", errors="replace").read()
        m = re.search(r"^names\s*:\s*\[(.*?)\]", txt, re.M | re.S)
        if m:
            return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
        m = re.search(r"^names\s*:\s*\n((?:\s+.*\n?)+)", txt, re.M)
        if m:
            out = {}
            for line in m.group(1).splitlines():
                k = re.match(r"\s*(\d+)\s*:\s*(.+)", line)
                if k:
                    out[int(k.group(1))] = k.group(2).strip().strip("'\"")
                elif line.strip().startswith("-"):
                    out[len(out)] = line.strip()[1:].strip().strip("'\"")
            if out:
                return [out[i] for i in sorted(out)]
    return None


def train_dir(cls):
    return os.path.join(DATASETS, CLASS_FOLDERS[cls][0], "real_images")


def clean_previous():
    n = 0
    for cls in (NORMAL, CRACK, POTHOLE):
        for f in glob.glob(os.path.join(train_dir(cls), PREFIX + "*.jpg")):
            os.remove(f); n += 1
    if os.path.isdir(EVAL_DIR):
        shutil.rmtree(EVAL_DIR)
    return n


def crops_from(img_path, lbl_path, mapping, pad=0.15, min_px=24):
    """Yield (class, PIL crop). Empty label file -> one NORMAL band crop."""
    from PIL import Image
    img = Image.open(img_path).convert("RGB"); W, H = img.size
    lines = [l.split() for l in open(lbl_path).read().splitlines() if l.strip()] if os.path.exists(lbl_path) else []
    if not lines:
        yield NORMAL, img.crop((int(W * 0.15), int(H * 0.50), int(W * 0.85), int(H * 0.95)))
        return
    for parts in lines:
        try:
            cid = int(float(parts[0])); cx, cy, bw, bh = map(float, parts[1:5])
        except (ValueError, IndexError):
            continue
        cls = mapping.get(cid)
        if cls is None:
            continue
        w, h = bw * W, bh * H
        if w < min_px or h < min_px:
            continue
        x0, y0 = cx * W - w / 2, cy * H - h / 2
        px, py = w * pad, h * pad
        yield cls, img.crop((max(0, int(x0 - px)), max(0, int(y0 - py)),
                             min(W, int(x0 + w + px)), min(H, int(y0 + h + py))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(DATASETS, "rdd2022_india"))
    ap.add_argument("--max-per-class", type=int, default=1200, help="training crops per class")
    ap.add_argument("--normal", type=int, default=600, help="training normal-road crops")
    ap.add_argument("--max-eval-per-class", type=int, default=400)
    a = ap.parse_args(argv)

    names = read_names(a.root)
    if not names:
        sys.exit(f"No class names found in a .yaml under {a.root}")
    mapping = {i: class_for(n) for i, n in enumerate(names)}
    print("[RDD India] class mapping:")
    for i, n in enumerate(names):
        print(f"    {i}: {n:28s} -> {CLASS_FOLDERS[mapping[i]][1] if mapping[i] is not None else 'not used'}")
    removed = clean_previous()
    if removed:
        print(f"[RDD India] removed {removed} crops from a previous run")

    caps = {CRACK: a.max_per_class, POTHOLE: a.max_per_class, NORMAL: a.normal}
    counts = {"train": {k: 0 for k in caps}, "eval": {k: 0 for k in caps}}
    photos = {"train": 0, "eval": 0}
    for split in ("train", "valid", "test"):
        kind = "eval" if split == "test" else "train"
        imgs = sorted(glob.glob(os.path.join(a.root, "images", split, "*.jpg")))
        for ip in imgs:
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(a.root, "labels", split, stem + ".txt")
            used = False
            for k, (cls, crop) in enumerate(crops_from(ip, lp, mapping)):
                cap = caps[cls] if kind == "train" else a.max_eval_per_class
                if counts[kind][cls] >= cap:
                    continue
                if kind == "train":
                    out = os.path.join(train_dir(cls), f"{PREFIX}{stem}_{k}.jpg")
                else:
                    d = os.path.join(EVAL_DIR, EVAL_NAMES[cls]); os.makedirs(d, exist_ok=True)
                    out = os.path.join(d, f"{stem}_{k}.jpg")
                crop.save(out, quality=92); counts[kind][cls] += 1; used = True
            photos[kind] += int(used)
    pretty = lambda c: {EVAL_NAMES[k]: v for k, v in c.items()}
    manifest = {"source": a.root, "class_names": names,
                "mapping": {n: (CLASS_FOLDERS[m][1] if m is not None else None) for n, m in zip(names, mapping.values())},
                "training_crops": pretty(counts["train"]), "training_photographs": photos["train"],
                "eval_crops": pretty(counts["eval"]), "eval_photographs": photos["eval"],
                "eval_split": ("held-out 15% of the labelled RDD2022 India photographs, split by photograph "
                               "(the official test split has no public labels) - never written to a training folder")}
    os.makedirs(EVAL_DIR, exist_ok=True)
    json.dump(manifest, open(os.path.join(EVAL_DIR, "manifest.json"), "w"), indent=1)
    print(f"[RDD India] training crops {pretty(counts['train'])} from {photos['train']} photographs")
    print(f"[RDD India] Indian test crops {pretty(counts['eval'])} from {photos['eval']} photographs -> {EVAL_DIR}")
    return manifest


if __name__ == "__main__":
    main()
