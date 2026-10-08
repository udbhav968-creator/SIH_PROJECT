"""
How many faces and number plates does the privacy redactor actually blur?

    pip install huggingface_hub
    python -m scripts.measure_redactor_recall                     # downloads ~410 MB once, 300 photos each
    python -m scripts.measure_redactor_recall --faces 500 --plates 500
    python -m scripts.measure_redactor_recall --wider-dir D:/wider --plates-dir D:/plates   # local copies

Data (public, annotated by people):
    faces   WIDER FACE validation (Yang et al., CVPR 2016): WIDER_val.zip + wider_face_split.zip from the
            Hugging Face dataset CUHK-CSE/wider_face. Boxes marked invalid are skipped.
    plates  keremberke/license-plate-object-detection, validation split (Roboflow export, COCO boxes).

Neither set is Indian street scenes from a bus, so the result says how the redactor does on public photos
of faces and plates, not on BMTC footage. Repeat it on your own outlined bus frames when you have them
(--plates-dir / --wider-dir accept any folder in the same formats).

Rule: a face or plate counts as redacted when at least --cover (default 50%) of its box lies inside regions
the redactor blurred. Recall is reported overall and by box size (small < 32 px, medium 32-96 px, large >
96 px on the longer side), because a 12-pixel face in a crowd and a plate filling the frame are different
problems. Faces and plates under --min-size px are listed separately and not counted in recall: at that
size nothing is recognisable to blur. Also reported: the share of each photo's area that was blurred, the
price paid for the recall.

Photos are sampled with a fixed seed, so a rerun scores the same ones. Writes
checkpoints/privacy_redaction_report.json, which /api/v1/privacy/redact then quotes.
"""
import argparse
import glob
import json
import os
import random
import sys
import time
import zipfile

import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)
CACHE = os.path.join(ENGINE_ROOT, "datasets", "_downloads", "redaction_eval")
OUT = os.path.join(ENGINE_ROOT, "checkpoints", "privacy_redaction_report.json")
SIZE_BUCKETS = (("small", 0, 32), ("medium", 32, 96), ("large", 96, 10 ** 9))


def _hf(repo, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", cache_dir=CACHE)


def _extract(zpath, dest):
    marker = os.path.join(dest, ".extracted")
    if not os.path.exists(marker):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        open(marker, "w").close()
    return dest


def wider_items(root=None):
    """[(image path, [(x, y, w, h), ...])] from WIDER FACE validation."""
    if root is None:
        root = os.path.join(CACHE, "wider")
        _extract(_hf("CUHK-CSE/wider_face", "data/WIDER_val.zip"), root)
        _extract(_hf("CUHK-CSE/wider_face", "data/wider_face_split.zip"), root)
    gt = glob.glob(os.path.join(root, "**", "wider_face_val_bbx_gt.txt"), recursive=True)
    imgs = glob.glob(os.path.join(root, "**", "WIDER_val", "images"), recursive=True)
    if not gt or not imgs:
        raise SystemExit(f"WIDER FACE validation files not found under {root}")
    items = []
    with open(gt[0], encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh]
    i = 0
    while i < len(lines):
        name = lines[i]
        if not name.endswith(".jpg"):
            i += 1
            continue
        n = int(lines[i + 1])
        boxes = []
        for ln in lines[i + 2:i + 2 + max(n, 1)]:
            parts = ln.split()
            if n and len(parts) >= 8 and parts[7] == "0":
                x, y, w, h = map(float, parts[:4])
                if w > 0 and h > 0:
                    boxes.append((x, y, w, h))
        items.append((os.path.join(imgs[0], name), boxes))
        i += 2 + max(n, 1)
    return [it for it in items if it[1]]


def coco_items(root=None):
    """[(image path, [(x, y, w, h), ...])] from a COCO export (the licence-plate set by default)."""
    if root is None:
        root = os.path.join(CACHE, "plates")
        _extract(_hf("keremberke/license-plate-object-detection", "data/valid.zip"), root)
    ann = glob.glob(os.path.join(root, "**", "_annotations.coco.json"), recursive=True) or \
        glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    if not ann:
        raise SystemExit(f"no COCO annotation file under {root}")
    with open(ann[0], encoding="utf-8") as fh:
        coco = json.load(fh)
    base = os.path.dirname(ann[0])
    boxes = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        if w > 0 and h > 0:
            boxes.setdefault(a["image_id"], []).append((x, y, w, h))
    return [(os.path.join(base, im["file_name"]), boxes[im["id"]]) for im in coco["images"] if im["id"] in boxes]


def covered_fraction(box, mask):
    x, y, w, h = box
    H, W = mask.shape
    x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x1, y1 = min(W, int(np.ceil(x + w))), min(H, int(np.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def score(items, detector, cover, min_size, limit, seed, kind):
    from PIL import Image
    from models.privacy_redactor import redact
    rng = random.Random(seed)
    items = sorted(items)
    rng.shuffle(items)
    items = items[:limit]
    buckets = {b[0]: [0, 0] for b in SIZE_BUCKETS}
    tiny, hit, total, blurred_share, t0 = 0, 0, 0, [], time.time()
    for path, boxes in items:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        _, rep = redact(rgb, detector=detector, return_regions=True)
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        for x, y, w, h, _k in rep["regions"]:
            mask[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = True
        blurred_share.append(float(mask.mean()))
        for b in boxes:
            side = max(b[2], b[3])
            if side < min_size:
                tiny += 1
                continue
            ok = covered_fraction(b, mask) >= cover
            total += 1
            hit += int(ok)
            for name, lo, hi in SIZE_BUCKETS:
                if lo <= side < hi:
                    buckets[name][0] += int(ok)
                    buckets[name][1] += 1
        print(f"\r  {kind}: {len(blurred_share)}/{len(items)} photos, recall so far "
              f"{hit / max(total, 1):.3f}", end="", flush=True)
    print()
    return {
        "photos": len(items), "boxes_scored": total, "redacted": hit,
        "recall": round(hit / total, 4) if total else None,
        "recall_by_size": {k: {"recall": round(v[0] / v[1], 4) if v[1] else None, "boxes": v[1]}
                           for k, v in buckets.items()},
        "boxes_below_min_size_not_scored": tiny,
        "mean_share_of_photo_blurred": round(float(np.mean(blurred_share)), 4) if blurred_share else None,
        "seconds": round(time.time() - t0, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces", type=int, default=300)
    ap.add_argument("--plates", type=int, default=300)
    ap.add_argument("--wider-dir", default=None)
    ap.add_argument("--plates-dir", default=None)
    ap.add_argument("--cover", type=float, default=0.5)
    ap.add_argument("--min-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    from models.onnx_object_detector import ONNXObjectDetector
    det = ONNXObjectDetector()
    if not getattr(det, "is_ready", False):
        print("WARNING: the COCO detector did not load; only the Haar face finder will run")
    report = {
        "rule": f"a face or plate is redacted when >= {a.cover:.0%} of its box was blurred; boxes under "
                f"{a.min_size} px on the longer side are not scored",
        "measured_on": time.strftime("%Y-%m-%d"),
        "detector_loaded": bool(getattr(det, "is_ready", False)),
        "seed": a.seed,
    }
    if a.faces:
        report["faces"] = dict(score(wider_items(a.wider_dir), det, a.cover, a.min_size, a.faces, a.seed, "faces"),
                               dataset="WIDER FACE validation (CUHK-CSE/wider_face)")
    if a.plates:
        report["plates"] = dict(score(coco_items(a.plates_dir), det, a.cover, a.min_size, a.plates, a.seed, "plates"),
                                dataset="keremberke/license-plate-object-detection, validation split")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    for k in ("faces", "plates"):
        if k in report:
            r = report[k]
            print(f"[redaction] {k}: recall {r['recall']} on {r['boxes_scored']} boxes "
                  f"({r['recall_by_size']}), {r['mean_share_of_photo_blurred']:.1%} of each photo blurred")
    print(f"[redaction] written {os.path.relpath(a.out, ENGINE_ROOT)}")
    return report


if __name__ == "__main__":
    main()
