"""
Collect more PIXEL-LABELLED road-damage data for the U-Net, from public sources.

    python -m scripts.fetch_seg_datasets                         # every source that is reachable
    python -m scripts.fetch_seg_datasets --pothole-mix-zip /content/pothole_mix.zip

Why these and not "a million images from the web"
-------------------------------------------------
A segmenter learns an outline only from an outline. Photographs without pixel
labels (YouTube frames, image search, box-only datasets) cannot teach where a
pothole's edge is, and scraping them breaks the sites' terms and drags in faces
and number plates. The U-Net failed because it saw ONE dataset (DNIT, 2,235
photographs); what it needs is the same kind of label from DIFFERENT roads,
cameras and weather. These public datasets have exactly that:

  crackseg9k     9,159 crack masks, a benchmark collection of 10 crack datasets
                 (Hugging Face mirror of the Harvard Dataverse release)
  kaggle_pothole ~780 pothole polygons, road photographs (Kaggle; needs your
                 Kaggle credentials in this session, skipped otherwise)
  pothole_mix    4,340 masks of potholes AND cracks from several sources
                 (Mendeley Data; download the zip in a browser and pass it with
                 --pothole-mix-zip, because Mendeley blocks scripted downloads)

Leakage guard
-------------
CrackSeg9k contains CRACK500, and the pothole sets overlap Kaggle's pothole
photographs - both of which this project uses to MEASURE the pipeline
(datasets/02_kaggle_pothole_600, 03_crack500_fatigue, and the clean-road
folders). Every incoming image is perceptually hashed and dropped if it is a
near-duplicate of any measurement image, so a new model can never be trained on
the photographs that will judge it. The count dropped is recorded.

Output
------
datasets/seg_multi/<source>/img/<id>.jpg   resized to 512x320
datasets/seg_multi/<source>/lab/<id>.png   0 sound, 1 crack, 2 pothole, 255 ignore
datasets/seg_multi/manifest.json           per-image split (75/10/15 by image id,
                                           seeded) and the counts above
"""

import argparse
import base64
import glob
import hashlib
import io
import json
import os
import sys
import zipfile

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import numpy as np

OUT = os.path.join(ENGINE_ROOT, "datasets", "seg_multi")
IN_W, IN_H = 512, 320
EVAL_FOLDERS = ["02_kaggle_pothole_600", "03_crack500_fatigue", "05_morth_civil_hard_negatives",
                "10_missing_zebra_crossing", "11_missing_road_divider"]


# ---------------------------------------------------------------------------
# leakage guard: 64-bit difference hash, vectorised Hamming distance
# ---------------------------------------------------------------------------
def dhash(img_rgb):
    import cv2
    g = cv2.cvtColor(np.asarray(img_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return np.packbits(bits).view(">u8")[0]


class LeakGuard:
    def __init__(self, threshold=6):
        import cv2
        self.threshold = threshold
        hashes = []
        for f in EVAL_FOLDERS:
            for p in glob.glob(os.path.join(ENGINE_ROOT, "datasets", f, "**", "*.jpg"), recursive=True):
                im = cv2.imread(p)
                if im is not None:
                    hashes.append(dhash(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        self.ref = np.array(hashes, dtype=np.uint64)
        print(f"[leak guard] {len(self.ref)} measurement photographs hashed from {', '.join(EVAL_FOLDERS)}")

    def is_leak(self, img_rgb):
        if not len(self.ref):
            return False
        x = np.bitwise_xor(self.ref, np.uint64(dhash(img_rgb)))
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        return bool(d.min() <= self.threshold)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def split_of(source, ident, seed=42):
    h = int(hashlib.sha1(f"{seed}:{source}:{ident}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < 0.75 else ("cal" if h < 0.85 else "test")


class Writer:
    def __init__(self, source, guard):
        self.source, self.guard = source, guard
        self.dir_img = os.path.join(OUT, source, "img")
        self.dir_lab = os.path.join(OUT, source, "lab")
        os.makedirs(self.dir_img, exist_ok=True)
        os.makedirs(self.dir_lab, exist_ok=True)
        self.items, self.dropped_leak, self.dropped_empty, self.dropped_bad = [], 0, 0, 0

    def add(self, ident, img_rgb, label):
        import cv2
        if img_rgb is None or label is None:
            self.dropped_bad += 1
            return
        img_rgb = np.asarray(img_rgb, dtype=np.uint8)
        if img_rgb.ndim != 3 or label.shape[:2] != img_rgb.shape[:2]:
            self.dropped_bad += 1
            return
        if self.guard.is_leak(img_rgb):
            self.dropped_leak += 1
            return
        img = cv2.resize(img_rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        lab = cv2.resize(label.astype(np.uint8), (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
        if not ((lab == 1) | (lab == 2)).any():
            self.dropped_empty += 1          # a mask with no defect teaches nothing about outlines
            return
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(ident))[:80]
        cv2.imwrite(os.path.join(self.dir_img, safe + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(self.dir_lab, safe + ".png"), lab)
        self.items.append({"id": safe, "split": split_of(self.source, safe),
                           "crack_px": int((lab == 1).sum()), "pothole_px": int((lab == 2).sum())})

    def summary(self):
        sp = {s: sum(1 for i in self.items if i["split"] == s) for s in ("train", "cal", "test")}
        return {"kept": len(self.items), "splits": sp, "dropped_near_duplicate_of_measurement_set": self.dropped_leak,
                "dropped_no_defect_pixels": self.dropped_empty, "dropped_unreadable": self.dropped_bad}


# ---------------------------------------------------------------------------
# helpers to decode whatever a dataset stores
# ---------------------------------------------------------------------------
def to_array(v):
    """PIL image, {'bytes':...}, raw bytes or a base64 string -> numpy array, else None."""
    from PIL import Image
    try:
        if hasattr(v, "convert"):
            return np.asarray(v)
        if isinstance(v, dict) and v.get("bytes"):
            v = v["bytes"]
        if isinstance(v, str):
            v = base64.b64decode(v.split(",")[-1])
        if isinstance(v, (bytes, bytearray)):
            return np.asarray(Image.open(io.BytesIO(v)))
    except Exception:
        return None
    return None


def colour_mask_to_label(m):
    """RGB or grey mask -> 0/1/2. Red-dominant pixels are pothole, other coloured pixels crack."""
    m = np.asarray(m)
    if m.ndim == 2:
        return None                                   # binary: class unknown, caller decides
    r, g, b = (m[..., i].astype(int) for i in range(3))
    on = (r + g + b) > 60
    lab = np.zeros(m.shape[:2], np.uint8)
    lab[on & (r > g + 40) & (r > b + 40)] = 2
    lab[on & (lab == 0)] = 1
    return lab


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def fetch_crackseg9k(guard, limit=None):
    try:
        from datasets import load_dataset
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q datasets")
        from datasets import load_dataset
    w = Writer("crackseg9k", guard)
    for split in ("train", "test"):
        ds = load_dataset("rimvydasrub/crackseg9k", split=split)
        cols = ds.column_names
        print(f"[crackseg9k] {split}: {len(ds)} rows, columns {cols}")
        for i, row in enumerate(ds):
            if limit and len(w.items) >= limit:
                break
            arrays = {c: to_array(row[c]) for c in cols}
            arrays = {c: a for c, a in arrays.items() if a is not None}
            if len(arrays) < 2:
                w.dropped_bad += 1
                continue
            # the mask is the array with the fewest distinct values
            mask_col = min(arrays, key=lambda c: len(np.unique(arrays[c][::4, ::4])))
            img_col = next(c for c in arrays if c != mask_col)
            img, m = arrays[img_col], arrays[mask_col]
            if img.ndim == 2:
                img = np.stack([img] * 3, -1)
            img = img[..., :3]
            if m.ndim == 3:
                m = m[..., 0]
            w.add(f"{split}_{i}", img, (m > 127).astype(np.uint8))          # crack = 1
            if i % 1000 == 0:
                print(f"  {i} rows, {len(w.items)} kept", flush=True)
    return w.summary()


def fetch_kaggle_pothole(guard):
    try:
        import kagglehub
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q kagglehub")
        import kagglehub
    import cv2
    try:
        root = kagglehub.dataset_download("farzadnekouei/pothole-image-segmentation-dataset")
    except Exception as e:
        return {"skipped": f"Kaggle download failed ({str(e)[:120]}); add your Kaggle credentials to this "
                           f"session to include it"}
    w = Writer("kaggle_pothole", guard)
    for img_path in glob.glob(os.path.join(root, "**", "images", "**", "*.*"), recursive=True):
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lab_path = os.path.splitext(img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))[0] + ".txt"
        im = cv2.imread(img_path)
        if im is None or not os.path.exists(lab_path):
            w.dropped_bad += 1
            continue
        h, wd = im.shape[:2]
        lab = np.zeros((h, wd), np.uint8)
        for line in open(lab_path).read().splitlines():
            v = line.split()
            if len(v) < 7:
                continue
            pts = (np.array(v[1:], float).reshape(-1, 2) * [wd, h]).astype(np.int32)
            cv2.fillPoly(lab, [pts], 2)                                     # pothole = 2
        w.add(os.path.splitext(os.path.basename(img_path))[0], cv2.cvtColor(im, cv2.COLOR_BGR2RGB), lab)
    return w.summary()


def fetch_pothole_mix(guard, zip_path):
    import cv2
    if not zip_path or not os.path.exists(zip_path):
        return {"skipped": "no --pothole-mix-zip given (download it from https://data.mendeley.com/datasets/kfth5g2xk3/2)"}
    root = os.path.join(OUT, "_pothole_mix_raw")
    if not os.path.isdir(root):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
        for inner in glob.glob(os.path.join(root, "**", "*.zip"), recursive=True):
            with zipfile.ZipFile(inner) as z:
                z.extractall(os.path.dirname(inner))
    files = [p for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True)
             if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    is_mask = lambda p: any(k in p.lower() for k in ("mask", "label", "annot", "gt"))
    masks = {}
    for p in files:
        if is_mask(p):
            masks.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    w = Writer("pothole_mix", guard)
    colours = {}
    for p in files:
        if is_mask(p):
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        mp = masks.get(stem)
        if not mp:
            continue
        im, m = cv2.imread(p), cv2.imread(mp)
        if im is None or m is None:
            w.dropped_bad += 1
            continue
        m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
        lab = colour_mask_to_label(m)
        if lab is not None and len(colours) < 4000:
            for c in np.unique(m.reshape(-1, 3)[::50], axis=0)[:6]:
                colours[tuple(int(x) for x in c)] = colours.get(tuple(int(x) for x in c), 0) + 1
        w.add(stem, cv2.cvtColor(im, cv2.COLOR_BGR2RGB), lab)
    s = w.summary()
    s["mask_colours_seen"] = {str(k): v for k, v in sorted(colours.items(), key=lambda kv: -kv[1])[:8]}
    s["colour_rule"] = "red-dominant pixels -> pothole, other coloured pixels -> crack"
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pothole-mix-zip", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="crackseg9k kaggle_pothole pothole_mix")
    ap.add_argument("--limit", type=int, default=None, help="max images per source (smoke runs)")
    a = ap.parse_args(argv)

    guard = LeakGuard()
    want = set(a.only or ["crackseg9k", "kaggle_pothole", "pothole_mix"])
    report = {"leak_guard": {"measurement_photographs_hashed": int(len(guard.ref)), "hamming_threshold": guard.threshold,
                             "folders": EVAL_FOLDERS}, "sources": {}}
    if "crackseg9k" in want:
        try:
            report["sources"]["crackseg9k"] = fetch_crackseg9k(guard, a.limit)
        except Exception as e:
            report["sources"]["crackseg9k"] = {"skipped": f"failed: {str(e)[:200]}"}
    if "kaggle_pothole" in want:
        report["sources"]["kaggle_pothole"] = fetch_kaggle_pothole(guard)
    if "pothole_mix" in want:
        report["sources"]["pothole_mix"] = fetch_pothole_mix(guard, a.pothole_mix_zip)

    # Merge with an earlier run, so fetching one more source never forgets the ones already on disk.
    man_path = os.path.join(OUT, "manifest.json")
    if os.path.exists(man_path):
        try:
            with open(man_path, "r", encoding="utf-8") as fh:
                earlier = json.load(fh).get("sources") or {}
            for src, summ in earlier.items():
                report["sources"].setdefault(src, summ)
        except Exception:
            pass
    items = {}
    for src in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
        d = os.path.join(OUT, src)
        if not src.startswith("_") and os.path.isdir(os.path.join(d, "lab")):
            items[src] = [{"id": os.path.splitext(os.path.basename(p))[0],
                           "split": split_of(src, os.path.splitext(os.path.basename(p))[0])}
                          for p in sorted(glob.glob(os.path.join(d, "lab", "*.png")))]
    report["items"] = items
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh)
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2))
    print(f"total images with pixel labels: {sum(len(v) for v in items.values())}")


if __name__ == "__main__":
    main()
