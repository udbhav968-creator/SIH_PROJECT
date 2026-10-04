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
  own_india      YOUR photographs of Indian roads, outlined in Roboflow, CVAT
                 or LabelMe (--own-photos <zip or folder>). Accepts a COCO
                 segmentation export, LabelMe JSON files, or images/ + masks/
                 folders (mask values 0 sound, 1 crack, 2 pothole, or red =
                 pothole and any other colour = crack). Class names containing
                 "pothole" become pothole, names containing "crack" become crack.
                 Several photographs of the SAME spot must share a name prefix
                 before a double underscore (spot12__a.jpg, spot12__b.jpg): the
                 split is made by that prefix, so one spot is never in both
                 training and test. Without "__" each photograph is its own spot.

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
import re
import shutil
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
        self.groups = {}

    def add(self, ident, img_rgb, label, group=None):
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
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(ident))[-120:]
        base, n = safe, 1
        while os.path.exists(os.path.join(self.dir_lab, safe + ".png")):   # never overwrite another pair
            safe, n = f"{base}_{n}", n + 1
        cv2.imwrite(os.path.join(self.dir_img, safe + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(self.dir_lab, safe + ".png"), lab)
        if group:
            self.groups[safe] = str(group)
        self.items.append({"id": safe, "split": split_of(self.source, group or safe),
                           "crack_px": int((lab == 1).sum()), "pothole_px": int((lab == 2).sum())})

    def save_groups(self):
        """Split keys for photographs that share a spot; main() reads them when it rebuilds the manifest."""
        if self.groups:
            with open(os.path.join(OUT, self.source, "groups.json"), "w", encoding="utf-8") as fh:
                json.dump(self.groups, fh)

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
    mask_words = ("mask", "label", "annot", "gt", "ground")
    image_words = ("image", "img", "rgb", "photo")

    def parts_of(p):
        return [c.lower() for c in os.path.relpath(os.path.dirname(p), root).split(os.sep)]

    def is_mask(p):
        return any(any(wd in c for wd in mask_words) for c in parts_of(p))

    def key_of(p):
        # the folder path with the images/masks level removed, plus the file stem: unique per pair
        keep = [c for c in parts_of(p) if not any(wd in c for wd in mask_words + image_words)]
        return tuple(keep), os.path.splitext(os.path.basename(p))[0]

    masks = {}
    for p in files:
        if is_mask(p):
            masks[key_of(p)] = p
    out_dir = os.path.join(OUT, "pothole_mix")
    if os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)                       # rebuilt from scratch: no stale or overwritten pairs
    w = Writer("pothole_mix", guard)
    colours, unpaired = {}, 0
    for p in files:
        if is_mask(p):
            continue
        k = key_of(p)
        mp = masks.get(k)
        if not mp:
            unpaired += 1
            continue
        ident = "_".join(list(k[0]) + [k[1]])
        im, m = cv2.imread(p), cv2.imread(mp)
        if im is None or m is None:
            w.dropped_bad += 1
            continue
        m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
        lab = colour_mask_to_label(m)
        if lab is not None and len(colours) < 4000:
            for c in np.unique(m.reshape(-1, 3)[::50], axis=0)[:6]:
                colours[tuple(int(x) for x in c)] = colours.get(tuple(int(x) for x in c), 0) + 1
        w.add(ident, cv2.cvtColor(im, cv2.COLOR_BGR2RGB), lab)
    s = w.summary()
    s["images_without_a_matching_mask"] = unpaired
    s["mask_colours_seen"] = {str(k): v for k, v in sorted(colours.items(), key=lambda kv: -kv[1])[:8]}
    s["colour_rule"] = "red-dominant pixels -> pothole, other coloured pixels -> crack"
    return s


# ---------------------------------------------------------------------------
# own photographs (Roboflow / CVAT COCO export, LabelMe, or mask folders)
# ---------------------------------------------------------------------------
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def class_of(name):
    """Annotation class name -> 1 crack, 2 pothole, None (ignored)."""
    n = str(name).lower()
    if "pothole" in n or "pot hole" in n or "pot-hole" in n:
        return 2
    if "crack" in n:
        return 1
    return None


def spot_of(stem):
    """Split key: the file stem without Roboflow's '_jpg.rf.<hash>' suffix, cut at the first '__'."""
    stem = re.sub(r"[._-](jpe?g|png|bmp|webp)\.rf\.[0-9a-zA-Z]+$", "", stem, flags=re.I)
    stem = re.sub(r"\.rf\.[0-9a-zA-Z]+$", "", stem)
    return stem.split("__")[0] if "__" in stem else stem


def draw_shapes(shape_hw, shapes):
    """shapes: [(cls, kind, points Nx2 in pixels)] -> label image. Cracks first, potholes painted over them."""
    import cv2
    h, w = shape_hw
    lab = np.zeros((h, w), np.uint8)
    thick = max(3, int(round(0.006 * w)))
    for want in (1, 2):
        for cls, kind, pts in shapes:
            if cls != want or len(pts) < 2:
                continue
            p = np.round(np.asarray(pts, float)).astype(np.int32).reshape(-1, 1, 2)
            if kind in ("line", "linestrip", "polyline") or len(pts) < 3:
                cv2.polylines(lab, [p], False, int(cls), thickness=thick)
            elif kind == "rectangle" and len(pts) == 2:
                (x0, y0), (x1, y1) = p.reshape(-1, 2)
                cv2.rectangle(lab, (int(x0), int(y0)), (int(x1), int(y1)), int(cls), thickness=-1)
            else:
                cv2.fillPoly(lab, [p], int(cls))
    return lab


def _coco_rle_to_mask(seg, h, w):
    try:
        from pycocotools import mask as mask_utils
        rle = mask_utils.frPyObjects(seg, h, w) if isinstance(seg.get("counts"), list) else seg
        return mask_utils.decode(rle).astype(bool)
    except Exception:
        return None


def own_from_coco(json_path, w, root=None):
    import cv2
    with open(json_path, "r", encoding="utf-8") as fh:
        coco = json.load(fh)
    cats = {c["id"]: class_of(c.get("name", "")) for c in coco.get("categories", [])}
    anns = {}
    for a in coco.get("annotations", []):
        anns.setdefault(a["image_id"], []).append(a)
    base = os.path.dirname(json_path)
    stats = {"images": 0, "unknown_classes": sorted({c.get("name", "") for c in coco.get("categories", [])
                                                     if class_of(c.get("name", "")) is None})}
    for im in coco.get("images", []):
        path = os.path.join(base, im["file_name"])
        if not os.path.exists(path):
            # CVAT puts annotations/ and images/ side by side: look anywhere in the export
            name = os.path.basename(im["file_name"])
            path = None
            for top in (base, root or base):
                hits = glob.glob(os.path.join(top, "**", name), recursive=True)
                if hits:
                    path = hits[0]
                    break
        img = cv2.imread(path) if path else None
        if img is None:
            w.dropped_bad += 1
            continue
        h, wd = img.shape[:2]
        shapes, rle_masks = [], []
        for a in anns.get(im["id"], []):
            cls = cats.get(a.get("category_id"))
            if cls is None:
                continue
            seg = a.get("segmentation")
            if isinstance(seg, list) and seg:
                for poly in seg:
                    if len(poly) >= 4:
                        shapes.append((cls, "polygon", np.asarray(poly, float).reshape(-1, 2)))
            elif isinstance(seg, dict):
                m = _coco_rle_to_mask(seg, h, wd)
                if m is not None:
                    rle_masks.append((cls, m))
            elif a.get("bbox"):          # a box with no outline: it cannot teach an edge, so it is skipped
                continue
        lab = draw_shapes((h, wd), shapes)
        for want in (1, 2):
            for cls, m in rle_masks:
                if cls == want:
                    lab[m] = cls
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        w.add(stem, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lab, group=spot_of(stem))
        stats["images"] += 1
    return stats


def own_from_labelme(json_paths, w):
    import cv2
    n = 0
    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        cand = [os.path.join(os.path.dirname(jp), d.get("imagePath") or "")] + \
               [os.path.splitext(jp)[0] + e for e in IMG_EXT]
        path = next((c for c in cand if os.path.isfile(c)), None)
        img = cv2.imread(path) if path else None
        if img is None:
            w.dropped_bad += 1
            continue
        shapes = [(class_of(sh.get("label", "")), sh.get("shape_type") or "polygon", sh.get("points") or [])
                  for sh in d.get("shapes", [])]
        lab = draw_shapes(img.shape[:2], [s_ for s_ in shapes if s_[0]])
        stem = os.path.splitext(os.path.basename(path))[0]
        w.add(stem, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lab, group=spot_of(stem))
        n += 1
    return {"images": n}


def mask_to_label(m):
    """Mask image -> 0/1/2 labels, or None if its classes cannot be told apart."""
    m = np.asarray(m)
    if m.ndim == 3:
        if np.ptp(m.astype(int), axis=2).max() < 10:      # grey stored as RGB
            m = m[..., 0]
        else:
            return colour_mask_to_label(m)
    vals = set(np.unique(m).tolist())
    if vals <= {0, 1, 2, 255}:
        lab = m.astype(np.uint8).copy()
        lab[lab == 255] = 255 if vals & {1, 2} else 0        # 255 with 1/2 present = ignore; alone = unknown
        return lab if ((lab == 1) | (lab == 2)).any() else None
    return None                                              # binary 0/255: crack or pothole is unknown


def own_from_mask_folders(root, w):
    import cv2
    masks = {}
    for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        parts = [c.lower() for c in os.path.relpath(p, root).split(os.sep)[:-1]]
        if p.lower().endswith((".png", ".bmp")) and any(c in ("masks", "mask", "labels", "label", "gt") for c in parts):
            masks[spot_key(p)] = p
    n, unpaired, unknown = 0, 0, 0
    for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        parts = [c.lower() for c in os.path.relpath(p, root).split(os.sep)[:-1]]
        if not p.lower().endswith(IMG_EXT) or any(c in ("masks", "mask", "labels", "label", "gt") for c in parts):
            continue
        mp = masks.get(spot_key(p))
        if not mp:
            unpaired += 1
            continue
        img, m = cv2.imread(p), cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if img is None or m is None:
            w.dropped_bad += 1
            continue
        if m.ndim == 3:
            m = cv2.cvtColor(m[..., :3], cv2.COLOR_BGR2RGB)
        lab = mask_to_label(m)
        if lab is None:
            unknown += 1
            continue
        if lab.shape[:2] != img.shape[:2]:
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        stem = os.path.splitext(os.path.basename(p))[0]
        w.add(stem, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lab, group=spot_of(stem))
        n += 1
    return {"images": n, "images_without_a_mask": unpaired, "masks_with_unknown_classes": unknown}


def spot_key(p):
    s = os.path.splitext(os.path.basename(p))[0].lower()
    for suffix in ("_mask", "-mask", "_label", "-label", "_gt"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def merge_near_duplicate_spots(w, threshold=6):
    """
    Burst shots of one spot rarely carry the '__' naming. A photograph whose difference hash is within
    `threshold` bits of an earlier "leader" photograph is joined to that leader's spot. Hashes are compared with
    leaders only (a joined photograph never becomes a leader), so similar-looking roads cannot chain into one
    giant spot; spots named with '__' stay whole. A burst then never straddles train and test.
    Returns how many photographs were joined to another photograph's spot.
    """
    import cv2
    ids = [it["id"] for it in w.items]
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    leaders, merges = [], 0           # [(hash, id)]
    for i in ids:
        im = cv2.imread(os.path.join(w.dir_img, i + ".jpg"))
        if im is None:
            continue
        h = dhash(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if leaders:
            ref = np.array([lh for lh, _ in leaders], dtype=np.uint64)
            d = np.unpackbits(np.bitwise_xor(ref, np.uint64(h)).view(np.uint8).reshape(-1, 8), axis=1).sum(1)
            k = int(np.argmin(d))
            if d[k] <= threshold:
                a, b = find(w.groups.get(i, i)), find(w.groups.get(leaders[k][1], leaders[k][1]))
                if a != b:
                    parent[max(a, b)] = min(a, b)
                    merges += 1
                continue
        leaders.append((h, i))
    for i in ids:
        w.groups[i] = find(w.groups.get(i, i))
    for it in w.items:
        it["split"] = split_of(w.source, w.groups[it["id"]])
    return merges


def fetch_own_photos(guard, path):
    if not path or not os.path.exists(path):
        return {"skipped": "no --own-photos given (a zip or folder of your outlined road photographs)"}
    root = path
    if path.lower().endswith(".zip"):
        root = os.path.join(OUT, "_own_raw")
        shutil.rmtree(root, ignore_errors=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(root)
    out_dir = os.path.join(OUT, "own_india")
    shutil.rmtree(out_dir, ignore_errors=True)        # rebuilt from scratch each time you add photographs
    w = Writer("own_india", guard)
    jsons = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    coco, labelme = [], []
    for jp in jsons:
        try:
            with open(jp, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if isinstance(d, dict) and "annotations" in d and "images" in d:
            coco.append(jp)
        elif isinstance(d, dict) and "shapes" in d:
            labelme.append(jp)
    if coco:
        fmt = "coco"
        detail = {os.path.relpath(jp, root): own_from_coco(jp, w, root) for jp in coco}
    elif labelme:
        fmt, detail = "labelme", own_from_labelme(labelme, w)
    else:
        fmt, detail = "mask_folders", own_from_mask_folders(root, w)
    merged = merge_near_duplicate_spots(w)
    w.save_groups()
    s = w.summary()
    s["spots_merged_as_near_duplicates"] = merged
    sizes = {}
    for g in w.groups.values():
        sizes[g] = sizes.get(g, 0) + 1
    sp_counts = s.get("splits", {})
    if len(w.items) >= 20 and (not sp_counts.get("test") or not sp_counts.get("cal")):
        s["warning_splits"] = (f"your photos split as {sp_counts}: with no test (or calibration) photographs "
                               f"there is no Indian-road score; add photographs of more different spots")
    if sizes and max(sizes.values()) > max(5, 0.2 * len(w.items)):
        s["warning_spots"] = (f"one spot holds {max(sizes.values())} of {len(w.items)} photographs; if they are really "
                              f"different places, give them distinct names so the test split is not starved")
    s.update({"format": fmt, "detail": detail, "spots": len(set(w.groups.values())),
              "split_by": "spot (file-name prefix before '__', else the photograph)"})
    if not s["kept"]:
        s["warning"] = ("no usable outlined photographs found: check the export format and that class names contain "
                        "'pothole' or 'crack'")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pothole-mix-zip", default=None)
    ap.add_argument("--own-photos", default=None,
                    help="zip or folder of your own outlined road photographs (COCO, LabelMe or mask folders)")
    ap.add_argument("--only", nargs="*", default=None, help="crackseg9k kaggle_pothole pothole_mix own_india")
    ap.add_argument("--limit", type=int, default=None, help="max images per source (smoke runs)")
    a = ap.parse_args(argv)

    guard = LeakGuard()
    want = set(a.only or ["crackseg9k", "kaggle_pothole", "pothole_mix", "own_india"])
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
    if "own_india" in want:
        report["sources"]["own_india"] = fetch_own_photos(guard, a.own_photos)

    # Merge with an earlier run, so fetching one more source never forgets the ones already on disk.
    man_path = os.path.join(OUT, "manifest.json")
    if os.path.exists(man_path):
        try:
            with open(man_path, "r", encoding="utf-8") as fh:
                earlier = json.load(fh).get("sources") or {}
            for src, summ in earlier.items():
                now = report["sources"].get(src)
                if now is None or ("skipped" in now and "skipped" not in summ):
                    report["sources"][src] = summ     # not refetched this time: keep what is on disk
        except Exception:
            pass
    items = {}
    for src in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
        d = os.path.join(OUT, src)
        if not src.startswith("_") and os.path.isdir(os.path.join(d, "lab")):
            groups = {}
            if os.path.exists(os.path.join(d, "groups.json")):
                with open(os.path.join(d, "groups.json"), "r", encoding="utf-8") as fh:
                    groups = json.load(fh)
            ids = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(d, "lab", "*.png")))]
            items[src] = [{"id": i, "split": split_of(src, groups.get(i, i))} for i in ids]
    report["items"] = items
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh)
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2))
    print(f"total images with pixel labels: {sum(len(v) for v in items.values())}")


if __name__ == "__main__":
    main()
