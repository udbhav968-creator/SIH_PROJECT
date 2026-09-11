"""
Download real road-condition datasets and turn them into labelled training
patches for this project.

Runs on your own machine (it needs internet and a Kaggle token); nothing here
is downloaded automatically at build or serve time.

    python -m scripts.fetch_datasets search pothole
    python -m scripts.fetch_datasets download andrewmvd/pothole-detection
    python -m scripts.fetch_datasets ingest
    python -m scripts.fetch_datasets report

What each step does:

  search    Ask Kaggle which datasets match a keyword, with size and how many
            people use them, so you can pick before downloading gigabytes.
  download  Fetch and unzip a dataset into datasets/incoming/<owner>__<name>/.
            Also accepts a plain URL for non-Kaggle sources.
  ingest    Walk everything under datasets/incoming/, read whatever
            annotations it finds (Pascal VOC XML, YOLO txt, COCO json), crop
            each labelled box, map its label onto one of this project's seven
            classes via label_map.json, and write the crop into
            datasets/patches/<class folder>/. Images with no annotations are
            copied whole into datasets/patches/_unlabelled/ for you to sort.
  report    Count what is on disk now, per class and per source dataset.

Cropping labelled boxes matters: a detection dataset of 600 photos usually
carries 1500+ labelled defects, and each one becomes a training example that
is actually centred on the defect instead of on a whole road scene.

Near-duplicate crops are dropped with the perceptual hash in
models/forensic_audit_engine.py, so the same pothole photographed twice can't
end up in both the training and validation halves later.

Kaggle authentication: put your token in %USERPROFILE%\\.kaggle\\access_token
(Windows) or ~/.kaggle/access_token, or set KAGGLE_API_TOKEN. Older
username/key kaggle.json files work too.
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

DATASETS_DIR = os.path.join(ENGINE_ROOT, "datasets")
INCOMING_DIR = os.path.join(DATASETS_DIR, "incoming")
PATCH_DIR = os.path.join(DATASETS_DIR, "patches")
LABEL_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_map.json")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_CROP_PX = 24          # smaller boxes carry no usable texture
CROP_CONTEXT = 0.12       # keep 12% of the box size as surrounding road

# Suggested starting points. These are *candidates*: the download step reports
# clearly when a slug no longer exists rather than pretending it worked, and
# `search` is there to find current alternatives.
SUGGESTED_KAGGLE = [
    ("andrewmvd/pothole-detection", "665 road photos with Pascal VOC pothole boxes"),
    ("chitholian/annotated-potholes-dataset", "Annotated potholes, VOC format"),
    ("sachinpatel21/pothole-image-dataset", "Plain pothole / normal road photos"),
    ("virenbr11/pothole-and-plain-rode-images", "Pothole vs plain road, classification split"),
    ("lakshaymiddha/crack-segmentation-dataset", "Surface crack images with masks"),
    ("arunrk7/surface-crack-detection", "Concrete/asphalt crack vs no-crack patches"),
    ("balraj98/road-damage-dataset-rdd2020", "RDD2020 road damage, VOC boxes (D00/D10/D20/D40)"),
]

SUGGESTED_URLS = [
    ("https://github.com/sekilab/RoadDamageDetector", "RDD2022 official page - country zips, no login"),
]


# ----------------------------------------------------------------------------
# label mapping
# ----------------------------------------------------------------------------
DEFAULT_LABEL_MAP = {
    "_comment": "Maps a source dataset's label onto this project's class folders. "
                "Edit freely; keys are lower-cased and matched exactly, then by substring. "
                "Use null to ignore a label entirely.",
    "_classes": {
        "0": "05_morth_civil_hard_negatives",
        "1": "03_crack500_fatigue",
        "2": "02_kaggle_pothole_600",
        "3": "09_waterlogging_hazard",
        "4": "10_missing_zebra_crossing",
        "5": "11_missing_road_divider",
        "6": "12_damaged_traffic_signs",
    },
    "labels": {
        "pothole": 2, "potholes": 2, "d40": 2, "cavity": 2, "pot hole": 2,
        "crack": 1, "cracks": 1, "d00": 1, "d10": 1, "d20": 1,
        "longitudinal crack": 1, "transverse crack": 1, "alligator crack": 1,
        "fatigue": 1, "surface crack": 1,
        "water": 3, "waterlogging": 3, "puddle": 3, "flood": 3,
        "zebra": 4, "crosswalk": 4, "crossing": 4,
        "divider": 5, "median": 5,
        "sign": 6, "traffic sign": 6, "signage": 6,
        "normal": 0, "plain": 0, "good": 0, "no crack": 0, "negative": 0, "background": 0,
        "manhole": 0, "shadow": 0,
    },
}


def load_label_map():
    if not os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "w", encoding="utf-8") as fh:
            json.dump(DEFAULT_LABEL_MAP, fh, indent=2)
        print(f"[label map] wrote a starting map to {LABEL_MAP_PATH} - edit it if a dataset uses other names")
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_label(raw_label, label_map):
    """Source label -> this project's class id, or None to skip it."""
    key = str(raw_label).strip().lower().replace("_", " ")
    labels = label_map.get("labels", {})
    if key in labels:
        return labels[key]
    for name, cls in labels.items():
        if name in key:
            return cls
    return None


# ----------------------------------------------------------------------------
# Kaggle
# ----------------------------------------------------------------------------
def kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("The kaggle package is missing. Install it with:  pip install kaggle")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        sys.exit(
            "Kaggle rejected the credentials: %s\n"
            "Put your token in ~/.kaggle/access_token (or %%USERPROFILE%%\\.kaggle\\access_token),\n"
            "or set KAGGLE_API_TOKEN, then try again." % e
        )
    return api


def cmd_search(args):
    api = kaggle_api()
    print(f"Kaggle datasets matching '{args.query}':\n")
    try:
        results = api.dataset_list(search=args.query, sort_by="votes", max_size=args.max_size_bytes)
    except Exception as e:
        sys.exit(f"Search failed: {e}")
    for d in list(results)[: args.limit]:
        size = getattr(d, "size", "") or getattr(d, "totalBytes", "")
        print(f"  {str(d.ref):55s} {str(size):>10}  votes={getattr(d, 'voteCount', '?')}")
        title = getattr(d, "title", "")
        if title:
            print(f"      {title}")
    print("\nDownload one with:  python -m scripts.fetch_datasets download <ref>")


def cmd_suggest(_args):
    print("Suggested Kaggle datasets (check with `search` if a slug has moved):\n")
    for ref, note in SUGGESTED_KAGGLE:
        print(f"  {ref:55s} {note}")
    print("\nOther sources that need no login:\n")
    for url, note in SUGGESTED_URLS:
        print(f"  {url}\n      {note}")


def _download_url(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    name = url.rstrip("/").split("/")[-1] or "download.bin"
    target = os.path.join(dest_dir, name)
    print(f"[download] {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(target, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    if zipfile.is_zipfile(target):
        print(f"[unzip] {target}")
        with zipfile.ZipFile(target) as zf:
            zf.extractall(dest_dir)
        os.remove(target)
    return dest_dir


def cmd_download(args):
    os.makedirs(INCOMING_DIR, exist_ok=True)
    for ref in args.refs:
        if ref.startswith("http://") or ref.startswith("https://"):
            folder = os.path.join(INCOMING_DIR, ref.rstrip("/").split("/")[-1].split(".")[0])
            try:
                _download_url(ref, folder)
                print(f"[ok] {ref} -> {folder}")
            except Exception as e:
                print(f"[FAILED] {ref}: {e}")
            continue

        api = kaggle_api()
        folder = os.path.join(INCOMING_DIR, ref.replace("/", "__"))
        if os.path.isdir(folder) and os.listdir(folder) and not args.force:
            print(f"[skip] {ref} already downloaded ({folder}) - pass --force to redo")
            continue
        os.makedirs(folder, exist_ok=True)
        print(f"[download] kaggle: {ref}")
        try:
            api.dataset_download_files(ref, path=folder, unzip=True, quiet=False)
            print(f"[ok] {ref} -> {folder}")
        except Exception as e:
            print(f"[FAILED] {ref}: {e}")
            print("         The slug may have changed. Try: python -m scripts.fetch_datasets search <keyword>")


# ----------------------------------------------------------------------------
# annotation readers  (each returns [(label, x1, y1, x2, y2), ...])
# ----------------------------------------------------------------------------
def read_voc(xml_path):
    boxes = []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return boxes
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bnd = obj.find("bndbox")
        if name_el is None or bnd is None:
            continue
        try:
            coords = [float(bnd.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (AttributeError, TypeError, ValueError):
            continue
        boxes.append((name_el.text, *coords))
    return boxes


def read_yolo(txt_path, width, height, class_names):
    boxes = []
    try:
        with open(txt_path, "r", encoding="utf-8") as fh:
            lines = fh.read().strip().splitlines()
    except OSError:
        return boxes
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_idx = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        label = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
        boxes.append((label,
                      (cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def read_coco(json_path):
    """Returns {image_file_name: [(label, x1, y1, x2, y2), ...]}."""
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    if not isinstance(data, dict) or "annotations" not in data or "images" not in data:
        return {}
    cats = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    imgs = {i["id"]: i.get("file_name", "") for i in data.get("images", [])}
    out = {}
    for ann in data["annotations"]:
        bbox = ann.get("bbox")
        fname = imgs.get(ann.get("image_id"))
        if not bbox or not fname or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        out.setdefault(fname, []).append((cats.get(ann.get("category_id"), "?"), x, y, x + w, y + h))
    return out


def find_yolo_class_names(root):
    for name in ("classes.txt", "obj.names", "data.yaml", "classes.names"):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        try:
            text = open(path, "r", encoding="utf-8").read()
        except OSError:
            continue
        if name.endswith(".yaml"):
            if "names:" in text:
                tail = text.split("names:", 1)[1]
                inline = tail.strip().splitlines()[0].strip()
                if inline.startswith("["):
                    return [s.strip().strip("'\"") for s in inline.strip("[]").split(",") if s.strip()]
                names = []
                for line in tail.splitlines()[1:]:
                    stripped = line.strip()
                    if stripped.startswith("-"):
                        names.append(stripped.lstrip("- ").strip().strip("'\""))
                    elif stripped and not stripped.startswith("#"):
                        break
                if names:
                    return names
        else:
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
    return []


# ----------------------------------------------------------------------------
# ingest
# ----------------------------------------------------------------------------
def cmd_ingest(args):
    from PIL import Image
    from models.forensic_audit_engine import ForensicDuplicateHasher
    import numpy as np

    label_map = load_label_map()
    class_folders = {int(k): v for k, v in label_map["_classes"].items()}
    hasher = ForensicDuplicateHasher(hash_size=8, duplicate_hamming_threshold=4)
    seen_hashes = []

    if not os.path.isdir(INCOMING_DIR):
        sys.exit(f"Nothing downloaded yet - {INCOMING_DIR} does not exist. Run the download step first.")

    stats = {"images_seen": 0, "boxes_found": 0, "crops_written": 0, "dupes_skipped": 0,
             "unmapped_labels": {}, "whole_images_copied": 0, "per_class": {}}

    for source in sorted(os.listdir(INCOMING_DIR)):
        source_dir = os.path.join(INCOMING_DIR, source)
        if not os.path.isdir(source_dir):
            continue
        print(f"\n[ingest] {source}")
        yolo_names = find_yolo_class_names(source_dir)
        coco_boxes = {}
        for root, _dirs, files in os.walk(source_dir):
            for f in files:
                if f.lower().endswith(".json") and "annotation" in f.lower() or f.lower() in ("instances.json", "coco.json"):
                    coco_boxes.update(read_coco(os.path.join(root, f)))

        for root, _dirs, files in os.walk(source_dir):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                    continue
                img_path = os.path.join(root, fname)
                stats["images_seen"] += 1
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                W, H = img.size
                stem = os.path.splitext(fname)[0]

                boxes = []
                for cand in (os.path.join(root, stem + ".xml"),
                             os.path.join(os.path.dirname(root), "annotations", stem + ".xml"),
                             os.path.join(source_dir, "annotations", stem + ".xml")):
                    if os.path.exists(cand):
                        boxes = read_voc(cand)
                        break
                if not boxes:
                    for cand in (os.path.join(root, stem + ".txt"),
                                 os.path.join(os.path.dirname(root), "labels", stem + ".txt")):
                        if os.path.exists(cand):
                            boxes = read_yolo(cand, W, H, yolo_names)
                            break
                if not boxes and fname in coco_boxes:
                    boxes = coco_boxes[fname]

                if not boxes:
                    # No annotation: keep the whole image for manual sorting, or
                    # treat the folder name as the label if it maps cleanly.
                    folder_label = resolve_label(os.path.basename(root), label_map)
                    if folder_label is not None:
                        out_dir = os.path.join(PATCH_DIR, class_folders[folder_label], "real_images")
                        os.makedirs(out_dir, exist_ok=True)
                        img.save(os.path.join(out_dir, f"{source}__{stem}.jpg"), quality=92)
                        stats["per_class"][folder_label] = stats["per_class"].get(folder_label, 0) + 1
                        stats["crops_written"] += 1
                    else:
                        out_dir = os.path.join(PATCH_DIR, "_unlabelled")
                        os.makedirs(out_dir, exist_ok=True)
                        img.save(os.path.join(out_dir, f"{source}__{stem}.jpg"), quality=88)
                        stats["whole_images_copied"] += 1
                    continue

                for idx, (raw_label, x1, y1, x2, y2) in enumerate(boxes):
                    stats["boxes_found"] += 1
                    cls = resolve_label(raw_label, label_map)
                    if cls is None:
                        key = str(raw_label).lower()
                        stats["unmapped_labels"][key] = stats["unmapped_labels"].get(key, 0) + 1
                        continue
                    bw, bh = x2 - x1, y2 - y1
                    if bw < MIN_CROP_PX or bh < MIN_CROP_PX:
                        continue
                    px, py = bw * CROP_CONTEXT, bh * CROP_CONTEXT
                    crop = img.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                                     min(W, int(x2 + px)), min(H, int(y2 + py))))
                    small = crop.resize((64, 64))
                    h = hasher.compute_hash(np.array(small, dtype=np.uint8))
                    if any(hasher.hamming_distance(h, prev) <= 4 for prev in seen_hashes):
                        stats["dupes_skipped"] += 1
                        continue
                    seen_hashes.append(h)
                    out_dir = os.path.join(PATCH_DIR, class_folders[cls], "real_images")
                    os.makedirs(out_dir, exist_ok=True)
                    crop.save(os.path.join(out_dir, f"{source}__{stem}__{idx}.jpg"), quality=92)
                    stats["crops_written"] += 1
                    stats["per_class"][cls] = stats["per_class"].get(cls, 0) + 1

    print("\n" + "=" * 66)
    print(f"images read          : {stats['images_seen']}")
    print(f"annotated boxes found: {stats['boxes_found']}")
    print(f"crops written        : {stats['crops_written']}")
    print(f"near-duplicates kept out: {stats['dupes_skipped']}")
    print(f"unannotated images parked for sorting: {stats['whole_images_copied']}")
    for cls, n in sorted(stats["per_class"].items()):
        print(f"  class {cls} ({class_folders[cls]}): {n}")
    if stats["unmapped_labels"]:
        print("\nLabels no rule matched (add them to scripts/label_map.json):")
        for k, v in sorted(stats["unmapped_labels"].items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {k!r}: {v} boxes")
    print(f"\nPatches are under {PATCH_DIR}")
    print("Next: python -m scripts.fetch_datasets promote   (merges them into the training folders)")


def cmd_promote(args):
    """Copy ingested patches into the folders the training scripts read."""
    label_map = load_label_map()
    class_folders = {int(k): v for k, v in label_map["_classes"].items()}
    moved = 0
    for cls, folder in class_folders.items():
        src = os.path.join(PATCH_DIR, folder, "real_images")
        dst = os.path.join(DATASETS_DIR, folder, "real_images")
        if not os.path.isdir(src):
            continue
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            target = os.path.join(dst, f)
            if os.path.exists(target) and not args.force:
                continue
            shutil.copy2(os.path.join(src, f), target)
            moved += 1
    print(f"promoted {moved} images into datasets/*/real_images")
    print("Now retrain:  python -m training.train_mega_suite")


def cmd_report(_args):
    from data.image_dataset import dataset_inventory
    inv = dataset_inventory()
    print("Training folders (what the models actually learn from):\n")
    total = 0
    for name, info in inv["labeled_classes"].items():
        total += info["usable_photos"]
        print(f"  {name:48s} {info['distinct_source_photos']:5d} distinct / {info['usable_photos']:5d} usable")
    print(f"\n  total usable: {total}")
    if os.path.isdir(PATCH_DIR):
        print("\nIngested but not yet promoted:")
        for folder in sorted(os.listdir(PATCH_DIR)):
            p = os.path.join(PATCH_DIR, folder, "real_images")
            p = p if os.path.isdir(p) else os.path.join(PATCH_DIR, folder)
            if os.path.isdir(p):
                print(f"  {folder:48s} {len(os.listdir(p)):5d}")
    if os.path.isdir(INCOMING_DIR):
        print("\nDownloaded sources:")
        for folder in sorted(os.listdir(INCOMING_DIR)):
            path = os.path.join(INCOMING_DIR, folder)
            if os.path.isdir(path):
                n = sum(len(fs) for _r, _d, fs in os.walk(path))
                print(f"  {folder:48s} {n:5d} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("search", help="search Kaggle for datasets")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-size-bytes", type=int, default=6_000_000_000)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("suggest", help="print suggested datasets to start with")
    p.set_defaults(func=cmd_suggest)

    p = sub.add_parser("download", help="download Kaggle datasets (owner/name) or plain URLs")
    p.add_argument("refs", nargs="+")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("ingest", help="crop labelled boxes into class patches")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("promote", help="copy patches into the training folders")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("report", help="show what is on disk")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
