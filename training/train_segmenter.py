"""
Train the pixel-level defect segmenter on the DNIT polygon annotations.

    python -m training.train_segmenter                    # default: 700 photos
    python -m training.train_segmenter --images 1200      # more data, slower
    python -m training.train_segmenter --quick            # 200 photos, for a smoke test

What it learns from
-------------------
`datasets/incoming/cracks_potholes_dnit/coco.json` carries 4,720 polygons drawn
by hand: 1,921 cracks, 564 potholes, and 2,235 lane outlines marking the
drivable surface. Those polygons are rasterised into per-pixel labels:

    inside a crack polygon      -> 1
    inside a pothole polygon    -> 2      (drawn last; potholes win overlaps)
    inside lane, outside both   -> 0      sound road
    everywhere else             -> ignored, not trained on and not scored

Excluding everything outside the lane is deliberate. Sky, verge and vegetation
are not pavement, and a model rewarded for calling them "sound road" learns
nothing useful about roads.

Clean photographs, and why they were missing
--------------------------------------------
Every DNIT photograph contains a defect. A model trained only on those has
never been shown the most common thing that LOOKS like a defect and is not:
fresh white paint. Measured on the first version of this model, a zebra
crossing produced up to 148,000 pixels called "pothole" - more than any real
pothole in the test set. The model had learned "high-contrast patch on dark
asphalt", and a painted stripe is the purest example of that on any road.

So `datasets/05_morth_civil_hard_negatives`, `10_missing_zebra_crossing` and
`11_missing_road_divider` are now trained on as well, with every road pixel
labelled sound. Then the model is asked to predict on those same photographs,
and every pixel it still calls crack or pothole is fed back as a sound example
and the model is retrained. That second pass is hard-negative mining: the
cheapest way to spend training capacity on the pixels that are actually wrong
rather than on the 97% that were already easy.

The report carries a false-positive rate measured on clean photographs the
model never saw, because a segmentation IoU says nothing about how often the
model invents a defect on a road that has none.

Protocol
--------
Split by photograph, never by pixel: pixels from one photograph appear in
exactly one of train or test. Scoring pixels from a photograph the model trained
on would report an IoU that means nothing.

Reported per class:
    IoU     intersection over union - the standard segmentation metric
    Dice    2|A∩B| / (|A|+|B|)
    and pixel precision/recall, because for a thin structure like a crack the
    two fail very differently and one number hides that.

Outputs
-------
    checkpoints/defect_segmenter.joblib
    checkpoints/defect_segmenter_report.json
"""

import argparse
import json
import os
import re
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np

from models.defect_segmenter import (CLASS_CRACK, CLASS_POTHOLE, CLASS_SOUND, FEATURE_NAMES,
                                     IGNORE, MODEL_PATH, WORK_H, WORK_W, extract_pixel_features)

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
REPORT_PATH = os.path.join(CKPT_DIR, "defect_segmenter_report.json")
DNIT_DIR = os.path.join(ENGINE_ROOT, "datasets", "incoming", "cracks_potholes_dnit")
COCO_PATH = os.path.join(DNIT_DIR, "coco.json")

COCO_CRACK, COCO_LANE, COCO_POTHOLE = 1, 2, 3

# Photographs of roads with no crack and no pothole. Their road pixels are
# sound by construction, and they contain the confusers - paint, joints,
# shadows, kerbs, manholes - that the DNIT set never shows without a defect
# somewhere else in frame.
NEGATIVE_FOLDERS = ["05_morth_civil_hard_negatives",
                    "10_missing_zebra_crossing",
                    "11_missing_road_divider"]
DATASETS_DIR = os.path.join(ENGINE_ROOT, "datasets")

_AUG_PREFIX = re.compile(r"^(?:aug_[a-z0-9]+_\d+_|real_\d+_)")


def source_group(path):
    """
    The source photograph an augmented copy came from.

    `aug_mega_11_github_dataset_1008367.jpg` and
    `aug_mega_4_github_dataset_1008367.jpg` are two crops of ONE photograph.
    Letting them fall on opposite sides of the split would put the same scene
    in train and test and inflate every number reported here. The split is by
    this key, never by file.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return _AUG_PREFIX.sub("", stem).strip().lower()


def load_negatives(limit=None):
    """Clean-road photographs, grouped by source so augmentations stay together."""
    import glob as _glob
    recs = []
    for folder in NEGATIVE_FOLDERS:
        root = os.path.join(DATASETS_DIR, folder)
        if not os.path.isdir(root):
            continue
        for path in sorted(_glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)):
            if "_label_conflicts" in path:
                continue
            recs.append({"path": path, "negative": True, "folder": folder,
                         "group": folder + "/" + source_group(path)})
    if limit:
        recs = recs[:limit]
    return recs


def _sklearn_version():
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return None


def _joblib_version():
    try:
        import joblib
        return joblib.__version__
    except Exception:
        return None


def load_annotations():
    if not os.path.exists(COCO_PATH):
        sys.exit(f"No annotations at {COCO_PATH}.\n"
                 f"Run:  python -m scripts.fetch_cracks_potholes_dataset --limit 2235")
    with open(COCO_PATH, "r", encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = {}
    for img in coco["images"]:
        by_image[img["id"]] = {"file_name": img["file_name"], "width": img["width"],
                               "height": img["height"], "polys": []}
    for ann in coco["annotations"]:
        rec = by_image.get(ann["image_id"])
        if rec is None:
            continue
        rec["polys"].append((ann["category_id"], ann.get("segmentation") or []))
    # only photographs that are actually on disk and carry a defect polygon
    usable = []
    for rec in by_image.values():
        path = os.path.join(DNIT_DIR, rec["file_name"])
        if not os.path.exists(path):
            continue
        cats = {c for c, _ in rec["polys"]}
        if COCO_CRACK in cats or COCO_POTHOLE in cats:
            rec["path"] = path
            usable.append(rec)
    return usable


def rasterise(rec, work_w=WORK_W, work_h=WORK_H):
    """Polygons -> (work_h, work_w) label image. IGNORE outside the lane."""
    import cv2
    if rec.get("negative"):
        # A clean photograph carries no polygons, so there is no lane outline to
        # trust. The lower half of a road photograph is road; above it may be
        # sky, buildings or traffic, and asserting "sound road" there would be
        # a label this photograph does not support.
        label = np.full((work_h, work_w), IGNORE, dtype=np.uint8)
        label[work_h // 2:, :] = CLASS_SOUND
        return label
    sx, sy = work_w / float(rec["width"]), work_h / float(rec["height"])

    def polys_for(cat):
        out = []
        for c, segs in rec["polys"]:
            if c != cat:
                continue
            for seg in segs:
                if not isinstance(seg, list) or len(seg) < 6:
                    continue
                pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= sx
                pts[:, 1] *= sy
                out.append(np.round(pts).astype(np.int32))
        return out

    label = np.full((work_h, work_w), IGNORE, dtype=np.uint8)
    lane = polys_for(COCO_LANE)
    if lane:
        cv2.fillPoly(label, lane, CLASS_SOUND)
    else:
        # No lane outline: treat the lower half as road rather than discard the
        # photograph. Conservative, and it only affects the negative class.
        label[work_h // 2:, :] = CLASS_SOUND
    cv2.fillPoly(label, polys_for(COCO_CRACK), CLASS_CRACK)
    # potholes drawn after cracks so an overlap resolves to the more severe class
    cv2.fillPoly(label, polys_for(COCO_POTHOLE), CLASS_POTHOLE)
    return label


def sample_pixels(feats, label, per_class, rng, sound_multiplier=3):
    """
    Pixel sample from one photograph. Ignored pixels are never sampled.

    Sound road is sampled `sound_multiplier` times more heavily than each defect
    class. Perfectly balanced sampling teaches the model that a defect is as
    likely as intact road, which it is not - the consequence is a model with
    good recall and dreadful precision that paints half the carriageway as
    cracked. Some imbalance is kept rather than matching the true prior exactly,
    because the defect classes would otherwise be too rare to learn from.
    """
    flat = label.reshape(-1)
    xs, ys = [], []
    for cls in (CLASS_SOUND, CLASS_CRACK, CLASS_POTHOLE):
        want = per_class * sound_multiplier if cls == CLASS_SOUND else per_class
        idx = np.flatnonzero(flat == cls)
        if idx.size == 0:
            continue
        take = min(want, idx.size)
        pick = rng.choice(idx, take, replace=False) if idx.size > take else idx
        xs.append(feats[pick])
        ys.append(np.full(pick.size, cls, dtype=np.uint8))
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


def available_memory_mb():
    """
    Free memory right now, or None if it cannot be determined.

    Used to size the training set to the machine instead of to a constant. A
    fixed cap that is comfortable on a workstation is fatal on a laptop with a
    browser open: the field failure was "Unable to allocate 34.3 MiB", which is
    not a large-array problem, it is a machine with almost nothing left.

    No new dependency - ctypes on Windows, /proc/meminfo on Linux, sysctl on
    macOS. A training script should not need psutil to avoid crashing.
    """
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            # The binding constraint is whichever is smaller: free RAM, or what
            # the process can still commit.
            return min(st.ullAvailPhys, st.ullAvailPageFile) / 1e6
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) / 1000.0
        if sys.platform == "darwin":
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return float(out.strip()) / 1e6 * 0.5      # rough; macOS compresses
    except Exception:
        pass
    return None


def affordable_pixels(requested, headroom_fraction=0.12):
    """
    How many training pixels this machine can actually hold.

    Each pixel costs 11 float32 features (44 B) plus a label, and scikit-learn
    then needs the binned copy, per-class gradients and hessians, and working
    space inside the tree grower - call it 3x the matrix. Taking a small
    fraction of free memory keeps all of that inside what is available, and
    leaves the machine usable while it trains.

    Returns (pixels, explanation).
    """
    avail = available_memory_mb()
    if avail is None:
        return requested, "could not read free memory; using the requested cap"
    bytes_per_pixel = 11 * 4 + 1
    affordable = int((avail * 1e6 * headroom_fraction) / bytes_per_pixel)
    affordable = max(400_000, affordable)
    if affordable >= requested:
        return requested, f"{avail:,.0f} MB free - the requested cap fits"
    return affordable, (f"only {avail:,.0f} MB free, so the cap drops "
                        f"{requested:,} -> {affordable:,} pixels")


def cap_pixels(X, y, max_pixels, rng):
    """
    Uniform subsample down to `max_pixels`, preserving class proportions.

    Not a nicety. scikit-learn bins the matrix, then holds gradients and
    hessians per class, and the refit after hard-negative mining briefly holds
    the old matrix and the new one at once. On a laptop with the full dataset
    that reached about 1.3 GB and died with:

        numpy._core._exceptions._ArrayMemoryError: Unable to allocate 24.7 MiB

    Uniform sampling keeps every class in the same proportion it already had,
    so the cap costs a little data and changes nothing about the balance.
    """
    n = X.shape[0]
    if max_pixels <= 0 or n <= max_pixels:
        return X, y
    # A boolean mask of float32 randoms, NOT rng.choice. Measured on 5.5M rows:
    # rng.choice(n, k, replace=False) peaks at 80 MB building the index before
    # the copy even starts. Asking for 80 MB to recover from an out-of-memory
    # error is how the recovery path crashed in the field. This costs 4n bytes
    # and lands within a fraction of a percent of the target.
    import gc
    frac = max_pixels / float(n)
    keep = rng.random(n, dtype=np.float32) < frac
    Xc, yc = X[keep], y[keep]
    print(f"  capping {n:,} -> {Xc.shape[0]:,} pixels "
          f"({X.nbytes / 1e6:.0f} MB -> {Xc.nbytes / 1e6:.0f} MB)")
    del X, y, keep
    gc.collect()
    return Xc, yc


def fit_with_headroom(make_clf, X, y, rng, label="fit"):
    """
    Fit, and if memory runs out, halve the sample and say so rather than dying.

    A training run that fails at minute nine of twelve has wasted the whole nine
    minutes. Halving the data costs accuracy, and the run reports that it did so
    rather than quietly producing a model trained on less than was asked for.
    """
    import gc
    while True:
        try:
            clf = make_clf()
            clf.fit(X, y)
            return clf, X.shape[0]
        except MemoryError:
            if X.shape[0] <= 300_000:
                raise
            # Every other row. A strided slice needs no index array at all, and
            # because sample_pixels appends one block per class per photograph,
            # taking every second row keeps both the class balance and the
            # spread across photographs. Recovering from an out-of-memory error
            # must not itself ask for memory.
            print(f"  [{label}] out of memory at {X.shape[0]:,} pixels - "
                  f"retrying with {X.shape[0] // 2:,}", flush=True)
            Xh, yh = np.ascontiguousarray(X[::2]), np.ascontiguousarray(y[::2])
            del X, y
            gc.collect()
            X, y = Xh, yh
            del Xh, yh
            gc.collect()


def mine_hard_negatives(clf, recs, thresholds, rng, per_image=1500, max_images=220):
    """
    The pixels the model still gets wrong on clean roads, fed back as sound.

    After the first fit, run the model over the clean photographs it just
    trained on and collect every pixel it calls crack or pothole. On a road
    with neither, all of them are wrong by construction. Adding them back as
    sound examples and refitting spends the model's next 250 boosting rounds on
    the decision boundary that is actually failing, instead of on the bulk of
    obviously-sound tarmac it already classifies correctly.

    This is the standard hard-negative mining loop, and the reason it is worth
    a second fit here is that the errors are not spread evenly: they concentrate
    on painted markings, which are a small fraction of pixels and would almost
    never be drawn by uniform sampling.
    """
    import cv2
    xs, ys = [], []
    mined_per_image = []
    for rec in recs[:max_images]:
        img = cv2.imread(rec["path"])
        if img is None:
            continue
        feats, shape = extract_pixel_features(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        proba = clf.predict_proba(feats)
        pred = apply_thresholds(proba, thresholds, shape).reshape(-1)
        truth = rasterise(rec).reshape(-1)
        wrong = np.flatnonzero((truth == CLASS_SOUND) &
                               ((pred == CLASS_CRACK) | (pred == CLASS_POTHOLE)))
        mined_per_image.append(int(wrong.size))
        if wrong.size == 0:
            continue
        take = min(per_image, wrong.size)
        pick = rng.choice(wrong, take, replace=False) if wrong.size > take else wrong
        xs.append(feats[pick].astype(np.float32, copy=False))
        ys.append(np.full(pick.size, CLASS_SOUND, dtype=np.uint8))
    if not xs:
        return None, None, {"images_mined": len(mined_per_image), "false_pixels_found": 0}
    X, y = np.concatenate(xs), np.concatenate(ys)
    stats = {
        "images_mined": len(mined_per_image),
        "false_pixels_found": int(sum(mined_per_image)),
        "false_pixels_per_image_mean": round(float(np.mean(mined_per_image)), 1),
        "pixels_added_to_training": int(X.shape[0]),
    }
    return X, y, stats


def clean_false_positive_rate(seg, recs, max_images=120):
    """
    How often the model invents a defect on a road that has none.

    Reported two ways, because they answer different questions:

      pixel_rate    of the road pixels in these photographs, the fraction
                    called crack or pothole. This is what the pipeline's
                    area-and-cost maths consumes.

      photo_rate    the fraction of photographs with any blob above
                    `min_blob_px`. This is what a user sees: one 600-pixel
                    false blob is a wrong box drawn on their screen, however
                    small it is as a share of the frame.

    A segmentation IoU cannot answer either, because it is computed only on
    photographs that contain a defect.
    """
    import cv2
    min_blob_px = 250
    tot_road = tot_crack = tot_pothole = 0
    flagged = 0
    seen = 0
    per_folder = {}
    for rec in recs[:max_images]:
        img = cv2.imread(rec["path"])
        if img is None:
            continue
        seen += 1
        mask = seg.segment(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["mask"]
        truth = rasterise(rec)
        if mask.shape != truth.shape:
            mask = cv2.resize(mask, (truth.shape[1], truth.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        road = truth == CLASS_SOUND
        tot_road += int(road.sum())
        c = int(((mask == CLASS_CRACK) & road).sum())
        p = int(((mask == CLASS_POTHOLE) & road).sum())
        tot_crack += c
        tot_pothole += p
        biggest = 0
        for cls in (CLASS_CRACK, CLASS_POTHOLE):
            b = ((mask == cls) & road).astype(np.uint8)
            if not b.any():
                continue
            n, _lab, st, _cen = cv2.connectedComponentsWithStats(b, 8)
            if n > 1:
                biggest = max(biggest, int(st[1:, cv2.CC_STAT_AREA].max()))
        hit = biggest >= min_blob_px
        flagged += 1 if hit else 0
        f = per_folder.setdefault(rec.get("folder", "?"), {"photos": 0, "flagged": 0})
        f["photos"] += 1
        f["flagged"] += 1 if hit else 0
    return {
        "clean_photographs_scored": seen,
        "min_blob_px": min_blob_px,
        "pixel_rate_crack": round(tot_crack / tot_road, 5) if tot_road else None,
        "pixel_rate_pothole": round(tot_pothole / tot_road, 5) if tot_road else None,
        "photo_rate_any_blob": round(flagged / seen, 4) if seen else None,
        "per_folder": {k: {**v, "rate": round(v["flagged"] / v["photos"], 3)}
                       for k, v in per_folder.items()},
    }


def apply_thresholds(proba, thresholds, shape):
    """
    Probability maps -> labels, using a per-class decision threshold.

    argmax is the wrong rule here. It implicitly assumes the operating point
    where the classes are equally costly, and for a problem where 97% of pixels
    are sound road that floods the mask with false positives. A threshold per
    defect class, chosen to maximise IoU on data the model did not train on, is
    the operating point that matters.
    """
    crack_p = proba[:, CLASS_CRACK]
    pothole_p = proba[:, CLASS_POTHOLE]
    labels = np.full(proba.shape[0], CLASS_SOUND, dtype=np.uint8)
    is_crack = crack_p >= thresholds["crack"]
    is_pothole = pothole_p >= thresholds["pothole"]
    # where both fire, the more severe class wins if it is also more probable
    labels[is_crack] = CLASS_CRACK
    labels[is_pothole & (pothole_p >= crack_p)] = CLASS_POTHOLE
    labels[is_pothole & ~is_crack] = CLASS_POTHOLE
    return labels.reshape(shape)


def calibrate_thresholds(clf, recs, rng, grid=None, max_images=60):
    """
    Choose the per-class threshold that maximises held-out IoU.

    Scored on photographs held out of training but separate from the test set,
    so the reported test IoU is not the number the threshold was tuned on.
    """
    import cv2
    grid = grid if grid is not None else np.round(np.arange(0.10, 0.91, 0.05), 2)
    sample = recs[:max_images]
    cached = []
    for rec in sample:
        img = cv2.imread(rec["path"])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feats, shape = extract_pixel_features(img)
        cached.append((clf.predict_proba(feats), rasterise(rec), shape))

    best = {}
    for cls_name, cls_id in (("crack", CLASS_CRACK), ("pothole", CLASS_POTHOLE)):
        best_t, best_iou = 0.5, -1.0
        for t in grid:
            inter = union = 0
            for proba, truth, shape in cached:
                valid = truth != IGNORE
                pred = (proba[:, cls_id] >= t).reshape(shape)
                t_mask = (truth == cls_id) & valid
                p_mask = pred & valid
                inter += int(np.logical_and(t_mask, p_mask).sum())
                union += int(np.logical_or(t_mask, p_mask).sum())
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_t, best_iou = float(t), iou
        best[cls_name] = best_t
        print(f"    {cls_name:8s} threshold {best_t:.2f}  (validation IoU {best_iou:.3f})")
    return best


def score_masks(true_label, pred_label):
    """IoU, Dice, precision and recall per defect class, over valid pixels only."""
    valid = true_label != IGNORE
    out = {}
    for cls, name in ((CLASS_CRACK, "crack"), (CLASS_POTHOLE, "pothole")):
        t = (true_label == cls) & valid
        p = (pred_label == cls) & valid
        inter = int(np.logical_and(t, p).sum())
        union = int(np.logical_or(t, p).sum())
        out[name] = {
            "intersection": inter, "union": union,
            "true_px": int(t.sum()), "pred_px": int(p.sum()),
        }
    return out


def accumulate(total, part):
    for name, d in part.items():
        acc = total.setdefault(name, {"intersection": 0, "union": 0, "true_px": 0, "pred_px": 0})
        for k, v in d.items():
            acc[k] += v


def finalise(total):
    out = {}
    for name, d in total.items():
        inter, union = d["intersection"], d["union"]
        t, p = d["true_px"], d["pred_px"]
        out[name] = {
            "iou": round(inter / union, 4) if union else None,
            "dice": round(2 * inter / (t + p), 4) if (t + p) else None,
            "pixel_precision": round(inter / p, 4) if p else None,
            "pixel_recall": round(inter / t, 4) if t else None,
            "true_pixels": t, "predicted_pixels": p,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=int, default=700, help="photographs to use")
    ap.add_argument("--per-class-pixels", type=int, default=900,
                    help="pixels sampled per class per photograph")
    ap.add_argument("--test-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="200 photographs, for a smoke test")
    ap.add_argument("--no-negatives", action="store_true",
                    help="train the old way, on defect photographs only "
                         "(kept so the negatives can be shown to matter)")
    ap.add_argument("--no-mining", action="store_true",
                    help="skip the hard-negative mining pass")
    ap.add_argument("--max-pixels", type=int, default=4_500_000,
                    help="hard cap on training pixels. 7.2M float64 needs ~620 MB for the "
                         "feature matrix alone, and the refit briefly holds two copies; "
                         "that ran a laptop out of memory mid-fit.")
    ap.add_argument("--clean-pixel-share", type=float, default=0.35,
                    help="clean-road pixels as a share of the sound class. Left uncapped, a "
                         "large negatives folder drowns the defect classes: on one machine "
                         "clean photographs were 39%% of all training pixels and pothole "
                         "validation IoU fell from 0.146 to 0.098.")
    args = ap.parse_args()
    if args.quick:
        args.images = 200

    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib

    records = load_annotations()
    if not records:
        sys.exit("No usable photographs with defect polygons on disk.")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(records)
    records = records[:args.images]

    n_test = max(20, int(len(records) * args.test_fraction))
    n_cal = max(15, int(len(records) * 0.12))
    test_recs = records[:n_test]
    cal_recs = records[n_test:n_test + n_cal]
    train_recs = records[n_test + n_cal:]
    print(f"[segmenter] {len(records)} photographs with defect polygons")
    print(f"  train {len(train_recs)} | threshold-calibration {len(cal_recs)} | "
          f"held-out test {len(test_recs)}")
    print("  split by photograph; the thresholds are tuned on the calibration set,")
    print("  so the reported test IoU is not the number they were fitted to")

    # ---------------- clean photographs, split by SOURCE photograph ----------
    neg_train = neg_cal = neg_test = []
    negatives = [] if args.no_negatives else load_negatives()
    if negatives:
        groups = sorted({r["group"] for r in negatives})
        rng.shuffle(groups)
        n_g = len(groups)
        g_test = set(groups[: max(1, int(n_g * 0.30))])
        g_cal = set(groups[max(1, int(n_g * 0.30)): max(2, int(n_g * 0.45))])
        neg_test = [r for r in negatives if r["group"] in g_test]
        neg_cal = [r for r in negatives if r["group"] in g_cal]
        neg_train = [r for r in negatives if r["group"] not in g_test and r["group"] not in g_cal]
        print(f"[segmenter] {len(negatives)} clean photographs from {n_g} source scenes "
              f"({', '.join(NEGATIVE_FOLDERS)})")
        print(f"  train {len(neg_train)} | calibration {len(neg_cal)} | "
              f"held-out clean test {len(neg_test)}  (split by source scene, so no "
              f"augmented copy of a training scene is scored)")
    else:
        print("[segmenter] NO clean photographs in training - this is the configuration "
              "that called a zebra crossing a pothole")

    # ---------------- training pixels ----------------
    import cv2
    X, y, t0 = [], [], time.time()
    all_train = [(r, False) for r in train_recs] + [(r, True) for r in neg_train]
    n_neg_px = 0

    # How many sound pixels each clean photograph may contribute.
    #
    # Every clean photograph is 100% sound, so giving each the same budget a
    # defect photograph spends on sound makes the negatives scale with however
    # many happen to be on disk. That is fine at 279 of them and wrong at 1,846:
    # measured on a machine with the full dataset, clean photographs became 39%
    # of all training pixels and pothole validation IoU dropped from 0.146 to
    # 0.098. The model was being taught mostly that roads are fine.
    #
    # So the negatives get a fixed SHARE of the sound class instead of a fixed
    # budget per photograph, and the share is a flag so it can be argued with.
    # ---- size the sample to the machine, BEFORE collecting anything --------
    #
    # The previous version collected 5.5M pixels and then capped to 4.5M. That
    # means the oversized matrix has to exist first - 242 MB - and the cap then
    # allocates the copy beside it. On a laptop with little free memory the run
    # died before the cap could help, and the recovery path died too.
    #
    # So the budgets are scaled down here instead, and the matrix is never built
    # bigger than the machine can hold.
    pixel_cap, why_cap = affordable_pixels(args.max_pixels)
    print(f"  memory: {why_cap}")

    per_class = args.per_class_pixels
    defect_sound_px = len(train_recs) * per_class * 3
    target_clean_px = int(args.clean_pixel_share * defect_sound_px)
    neg_budget = max(150, target_clean_px // max(1, len(neg_train))) if neg_train else 0

    # Crack and pothole pixels are scarcer than the budget asks for, so the
    # projection below is an upper bound; scaling on it is deliberately
    # conservative.
    projected = (len(train_recs) * per_class * 5) + (len(neg_train) * neg_budget)
    if projected > pixel_cap:
        scale = pixel_cap / float(projected)
        per_class = max(120, int(per_class * scale))
        neg_budget = max(80, int(neg_budget * scale)) if neg_train else 0
        print(f"  scaling the per-photograph budget by {scale:.2f} "
              f"({projected:,} projected > {pixel_cap:,} cap): "
              f"{args.per_class_pixels} -> {per_class} px per class per photograph")

    if neg_train:
        print(f"  clean-road budget: {neg_budget:,} px from each of {len(neg_train)} "
              f"photographs ({args.clean_pixel_share:.0%} share of the sound class)")

    for i, (rec, is_neg) in enumerate(all_train):
        raw = cv2.imread(rec["path"])
        if raw is None:
            continue
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        feats, _ = extract_pixel_features(img)
        label = rasterise(rec)
        if is_neg:
            # Only the sound class exists here; the budget is the share computed
            # above, divided evenly across the clean photographs.
            xs, ys = sample_pixels(feats, label, neg_budget, rng, sound_multiplier=1)
        else:
            xs, ys = sample_pixels(feats, label, per_class, rng)
        if xs is not None:
            X.append(xs)
            y.append(ys)
            if is_neg:
                n_neg_px += int(xs.shape[0])
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(all_train)} photographs featurised "
                  f"({time.time() - t0:.0f}s)", flush=True)
    # float32 halves the feature matrix. The extra precision buys nothing here -
    # the features are colour channels, gradients and local statistics - and the
    # refit briefly holds two copies, which is where a laptop ran out of memory.
    X = np.concatenate(X).astype(np.float32, copy=False)
    y = np.concatenate(y)
    n_sampled = int(X.shape[0])
    X, y = cap_pixels(X, y, pixel_cap, rng)          # belt and braces
    counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    print(f"  {X.shape[0]:,} training pixels, {X.shape[1]} features, per class {counts}")
    if n_neg_px:
        share = n_neg_px / float(n_sampled)
        print(f"  of which {share:.1%} came from clean photographs "
              f"({n_neg_px:,} of {n_sampled:,} sampled, before any cap)")
    print(f"  feature matrix {X.nbytes / 1e6:.0f} MB as {X.dtype}")

    def make_clf():
        return HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.1, max_depth=None, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=args.seed)

    t1 = time.time()
    clf, n_used = fit_with_headroom(make_clf, X, y, rng, label="fit 1")
    print(f"  trained in {time.time() - t1:.0f}s ({clf.n_iter_} boosting iterations)")

    print("  calibrating decision thresholds (pass 1) ...")
    thresholds = calibrate_thresholds(clf, cal_recs + neg_cal, rng)

    # ---------------- hard-negative mining -------------------------
    mining = {"ran": False}
    if neg_train and not args.no_mining:
        print("  mining hard negatives from clean photographs ...", flush=True)
        t_m = time.time()
        hx, hy, mining = mine_hard_negatives(clf, neg_train, thresholds, rng)
        mining["ran"] = True
        print(f"    {mining['false_pixels_found']:,} wrongly-flagged pixels over "
              f"{mining['images_mined']} clean photographs "
              f"({mining['false_pixels_per_image_mean']:.0f} per photograph)")
        if hx is not None:
            import gc
            # Drop the first model BEFORE building the bigger matrix. Holding the
            # old classifier, the old X and the new X at once is what pushed a
            # laptop over the edge; the old model is not needed again.
            del clf
            gc.collect()
            X = np.concatenate([X, hx.astype(np.float32, copy=False)])
            y = np.concatenate([y, hy])
            del hx, hy
            gc.collect()
            X, y = cap_pixels(X, y, pixel_cap, rng)
            print(f"    refitting on {X.shape[0]:,} pixels "
                  f"({X.nbytes / 1e6:.0f} MB)", flush=True)
            clf, _n = fit_with_headroom(make_clf, X, y, rng, label="refit")
            counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
            print(f"    refit in {time.time() - t_m:.0f}s ({clf.n_iter_} iterations)")
            print("  calibrating decision thresholds (pass 2) ...")
            thresholds = calibrate_thresholds(clf, cal_recs + neg_cal, rng)

    # ---------------- held-out scoring, whole masks ----------------

    from models.defect_segmenter import DefectSegmenter
    seg = DefectSegmenter.__new__(DefectSegmenter)
    seg.model_path, seg.clf, seg.report = MODEL_PATH, clf, None
    seg.thresholds = thresholds

    total, t2 = {}, time.time()
    for rec in test_recs:
        img = cv2.cvtColor(cv2.imread(rec["path"]), cv2.COLOR_BGR2RGB)
        if img is None:
            continue
        truth = rasterise(rec)
        out = seg.segment(img)
        pred = cv2.resize(out["mask"], (truth.shape[1], truth.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        accumulate(total, score_masks(truth, pred))
    iou = finalise(total)
    per_image_ms = 1000 * (time.time() - t2) / max(1, len(test_recs))

    clean_fp = None
    if neg_test:
        print("  scoring false positives on held-out CLEAN photographs ...", flush=True)
        clean_fp = clean_false_positive_rate(seg, neg_test)
        print(f"    {clean_fp['photo_rate_any_blob'] * 100:.1f}% of "
              f"{clean_fp['clean_photographs_scored']} clean photographs still produce "
              f"a blob of >= {clean_fp['min_blob_px']} px")
        for folder, d in clean_fp["per_folder"].items():
            print(f"      {folder:34} {d['flagged']:3}/{d['photos']:<3} ({d['rate'] * 100:.0f}%)")

    report = {
        "model": "HistGradientBoosting pixel classifier on 11 hand-designed features",
        "features": FEATURE_NAMES,
        "working_resolution": [WORK_W, WORK_H],
        "thresholds": thresholds,
        "decision_rule": "per-class probability threshold tuned for IoU on a "
                         "calibration split, not argmax",
        "trained_on": {
            "photographs_total": len(records),
            "train_photographs": len(train_recs),
            "calibration_photographs": len(cal_recs),
            "test_photographs": len(test_recs),
            "training_pixels": int(X.shape[0]),
            "max_pixels_cap": args.max_pixels,
            "pixel_cap_applied": pixel_cap,
            "per_class_pixels_used": per_class,
            "memory_note": why_cap,
            "clean_pixel_share": args.clean_pixel_share,
            "pixels_per_class": counts,
            "source": "DNIT Cracks and Potholes in Road Images - hand-drawn polygons",
            "clean_photographs": {
                "folders": NEGATIVE_FOLDERS,
                "train": len(neg_train), "calibration": len(neg_cal), "test": len(neg_test),
                "pixels_contributed": n_neg_px,
                "why": ("Every DNIT photograph contains a defect, so the first version "
                        "of this model had never seen fresh white paint on dark asphalt "
                        "without a defect nearby. It called zebra crossings potholes."),
            },
            "hard_negative_mining": mining,
        },
        "split_strategy": ("defect photographs split by photograph; clean photographs "
                           "split by SOURCE scene, so augmented crops of one scene never "
                           "cross the split"),
        "scored_on": "pixels inside the lane polygon only; sky and verge excluded",
        "iou": iou,
        "false_positives_on_clean_roads": clean_fp,
        "why_two_metrics": ("IoU is measured only on photographs that contain a defect, "
                            "so it cannot detect a model that invents defects on clean "
                            "roads. That failure is what users actually saw, so it is "
                            "measured separately on photographs held out of training."),
        "inference_ms_per_image": round(per_image_ms, 1),
        "trained_at_unix": int(time.time()),
        # Recorded so a load failure on another machine can NAME the cause. A
        # pickled scikit-learn estimator is not portable across versions, and
        # the error it raises ("No module named '_loss'") says nothing useful.
        "sklearn_version": _sklearn_version(),
        "joblib_version": _joblib_version(),
        "portability_note": ("This file is a pickled scikit-learn estimator. It loads "
                             "only under a compatible scikit-learn. If it fails, retrain "
                             "in place: python -m training.train_segmenter --images 2000 "
                             "(about 12 minutes on a laptop CPU). Nothing else is needed - "
                             "the training data is in the repository."),
    }

    os.makedirs(CKPT_DIR, exist_ok=True)
    joblib.dump({"classifier": clf, **{k: v for k, v in report.items() if k != "classifier"}},
                MODEL_PATH)  # report already carries `thresholds`
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"HELD-OUT SEGMENTATION  ({len(test_recs)} unseen photographs)")
    for name, m in iou.items():
        print(f"  {name:9s} IoU {m['iou']}   Dice {m['dice']}   "
              f"precision {m['pixel_precision']}   recall {m['pixel_recall']}")
    print(f"  {per_image_ms:.0f} ms per image")
    if clean_fp:
        print(f"\nFALSE POSITIVES ON CLEAN ROADS  ({clean_fp['clean_photographs_scored']} "
              f"unseen photographs with no crack and no pothole)")
        print(f"  photographs with a blob >= {clean_fp['min_blob_px']} px : "
              f"{clean_fp['photo_rate_any_blob'] * 100:.1f}%")
        print(f"  road pixels called crack   : {clean_fp['pixel_rate_crack'] * 100:.2f}%")
        print(f"  road pixels called pothole : {clean_fp['pixel_rate_pothole'] * 100:.2f}%")
    print(f"\nmodel  -> {MODEL_PATH}")
    print(f"report -> {REPORT_PATH}")
    return report


if __name__ == "__main__":
    main()
