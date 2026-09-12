"""
Sweep the two proposal knobs end-to-end and print BOTH error rates.

    python -m scripts.tune_proposals
    python -m scripts.tune_proposals --per-folder 16

What is being chosen
--------------------
`DeepInferencePipeline.MIN_COMPONENT_FRACTION` and `MIN_COMPONENT_CONFIDENCE`
decide which blobs of the segmentation mask become candidate regions. Too
permissive and a clean road produces ten confident "Pothole Cavity 100%" boxes,
because the classifier has seven defect classes and no way to answer "none of
them". Too strict and real potholes go unreported.

How it is measured
------------------
Through `DeepInferencePipeline.audit_image` - the same call the API makes - and
never through a reimplementation of the region loop. An earlier harness did
reimplement it, reported 27.8% false positives for a build that was really
producing 44.4%, and that wrong number was published. Measuring anything but
the real entry point measures the harness.

The one shortcut taken here is a cache over `segmenter.segment`, which is a
pure function of the image and accounts for most of the runtime. It changes
nothing about which code path runs; it only stops the same photograph being
segmented twelve times.

Output
------
A grid with false-positive rate on clean roads and detection rate on annotated
defects at every setting, plus the report at
checkpoints/proposal_tuning_report.json. Any setting that reaches 0% false
positives by reaching 0% detection is visible in the same table rather than
quoted on its own.
"""

import argparse
import glob
import json
import os
import random
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

CKPT = os.path.join(ENGINE_ROOT, "checkpoints")
REPORT = os.path.join(CKPT, "proposal_tuning_report.json")
CLEAN_FOLDERS = ["10_missing_zebra_crossing", "05_morth_civil_hard_negatives",
                 "11_missing_road_divider"]
DEFECT_FOLDERS = ["02_kaggle_pothole_600", "03_crack500_fatigue"]


def sample(folders, n, seed):
    rng = random.Random(seed)
    out = []
    for f in folders:
        files = [p for p in sorted(glob.glob(os.path.join(ENGINE_ROOT, "datasets", f,
                                                          "**", "*.jpg"), recursive=True))
                 if "_label_conflicts" not in p]
        rng.shuffle(files)
        out += files[:n]
    return out


def install_segmentation_cache(pipe):
    """Memoise segmenter.segment on image identity. Pure function; no path change."""
    seg = pipe.segmenter
    real = seg.segment
    cache = {}

    def cached(image_rgb, *a, **kw):
        key = getattr(cached, "_key", None)
        if key is None or key not in cache:
            out = real(image_rgb, *a, **kw)
            if key is not None:
                cache[key] = out
            return out
        return cache[key]

    seg.segment = cached
    return cached


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-folder", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fractions", type=float, nargs="*",
                    default=[0.001, 0.002, 0.004, 0.008, 0.015, 0.030])
    ap.add_argument("--confidences", type=float, nargs="*",
                    default=[0.40, 0.55, 0.70, 0.85])
    args = ap.parse_args()

    from pipeline.deep_inference_pipeline import DeepInferencePipeline
    pipe = DeepInferencePipeline(CKPT)
    if not (getattr(pipe, "segmenter", None) and pipe.segmenter.is_ready):
        sys.exit("No segmenter on disk. Run: python -m training.train_segmenter")
    cached = install_segmentation_cache(pipe)

    clean = sample(CLEAN_FOLDERS, args.per_folder, args.seed)
    defect = sample(DEFECT_FOLDERS, args.per_folder, args.seed)
    print(f"[proposals] {len(clean)} clean, {len(defect)} defect photographs")
    print(f"[proposals] {len(args.fractions) * len(args.confidences)} settings, "
          f"each scored through audit_image()\n", flush=True)

    def reports_defect(path):
        cached._key = path
        res = pipe.audit_image(path)
        return any(("Pothole" in d.get("class_name", "") or "Crack" in d.get("class_name", ""))
                   for d in (res.get("all_detections") or []))

    t0 = time.time()
    rows = []
    print(f"{'size':>7} {'conf':>6}  {'false positives':>17}  {'defects found':>16}")
    print("-" * 56)
    for frac in args.fractions:
        for conf in args.confidences:
            DeepInferencePipeline.MIN_COMPONENT_FRACTION = frac
            DeepInferencePipeline.MIN_COMPONENT_CONFIDENCE = conf
            fp = sum(1 for p in clean if reports_defect(p))
            hit = sum(1 for p in defect if reports_defect(p))
            row = {"min_component_fraction": frac, "min_component_confidence": conf,
                   "false_positive_photos": fp, "false_positive_rate": round(fp / len(clean), 4),
                   "detected_photos": hit, "detection_rate": round(hit / len(defect), 4)}
            rows.append(row)
            print(f"{frac:>7.3f} {conf:>6.2f}  {fp:>3}/{len(clean)} "
                  f"({row['false_positive_rate'] * 100:>5.1f}%)      "
                  f"{hit:>3}/{len(defect)} ({row['detection_rate'] * 100:>5.1f}%)", flush=True)

    # A setting is only interesting if it still finds defects. Rank by detection
    # minus false positives, then prefer the higher detection rate on ties.
    best = max(rows, key=lambda r: (r["detection_rate"] - r["false_positive_rate"],
                                    r["detection_rate"]))
    print(f"\nbest balance: size {best['min_component_fraction']} / "
          f"confidence {best['min_component_confidence']} -> "
          f"false positives {best['false_positive_rate'] * 100:.1f}%, "
          f"defects found {best['detection_rate'] * 100:.1f}%")

    report = {
        "generated_unix": int(time.time()),
        "measured_through": "DeepInferencePipeline.audit_image",
        "clean_photographs": len(clean), "defect_photographs": len(defect),
        "clean_folders": CLEAN_FOLDERS, "defect_folders": DEFECT_FOLDERS,
        "grid": rows,
        "recommended": best,
        "seconds": round(time.time() - t0, 1),
        "honesty_note": ("Both error rates appear at every setting. The row with 0% "
                         "false positives usually also has 0% detection; printing only "
                         "the first column would make a broken build look solved."),
    }
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nreport -> {REPORT}")


if __name__ == "__main__":
    main()
