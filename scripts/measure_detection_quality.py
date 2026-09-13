"""
End-to-end precision and recall, measured through audit_image().

    python -m scripts.measure_detection_quality
    python -m scripts.measure_detection_quality --per-folder 20

Why this replaced scripts/tune_detection_gate.py
------------------------------------------------
The previous script measured at the region level: it called
`extract_salient_regions`, classified each box, and applied the gate by hand.
It reported 27.8% false positives and 95.8% detection, and those numbers were
wrong - not mis-stated, but measuring something the product does not do.

The pipeline differs from that shortcut in ways that decide the answer: it
excludes pedestrian boxes from the proposal stage, it runs a texture
gatekeeper, and since the proposal change it takes crack and pothole
candidates from the segmentation mask rather than from brightness. Measured
properly through audit_image(), the "fixed" build was producing 44.4% false
positives - worse than the number that had been called a bug.

The lesson is cheap to state and was expensive to learn: measure the thing the
user runs. A harness that reimplements the pipeline measures the harness.

What is scored
--------------
clean     zebra crossings, sound pavement, road dividers. Any Crack or
          Pothole Cavity in `all_detections` is a false positive.
defect    annotated pothole and crack photographs. No Crack or Pothole Cavity
          reported is a miss.

Both are printed. A change that removes every false positive by also removing
every detection is not an improvement, and reporting only the first half is
the failure this file exists to prevent.
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
REPORT = os.path.join(CKPT, "detection_quality_report.json")

CLEAN_FOLDERS = ["10_missing_zebra_crossing", "05_morth_civil_hard_negatives",
                 "11_missing_road_divider"]
DEFECT_FOLDERS = ["02_kaggle_pothole_600", "03_crack500_fatigue"]
DEFECT_NAMES = ("Pothole", "Crack")


def sample(folders, n, seed):
    rng = random.Random(seed)
    out = []
    for f in folders:
        files = [p for p in sorted(glob.glob(os.path.join(ENGINE_ROOT, "datasets", f,
                                                          "**", "*.jpg"), recursive=True))
                 if "_label_conflicts" not in p]
        if not files:
            continue
        rng.shuffle(files)
        out += [(p, f) for p in files[:n]]
    return out


def defects_in(result):
    """Crack/pothole entries the product would put on screen."""
    found = []
    for d in result.get("all_detections") or []:
        name = d.get("class_name", "")
        if any(k in name for k in DEFECT_NAMES):
            found.append({"class_name": name,
                          "confidence": round(float(d.get("confidence") or 0), 3),
                          "bbox_pixels": d.get("bbox_pixels"),
                          "area_m2": d.get("surface_area_m2"),
                          "area_method": d.get("area_method")})
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-folder", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from pipeline.corpus_fingerprint import fingerprint
    from pipeline.deep_inference_pipeline import DeepInferencePipeline
    pipe = DeepInferencePipeline(CKPT)
    seg_ready = bool(getattr(pipe, "segmenter", None) and pipe.segmenter.is_ready)
    print(f"[measure] segmenter loaded: {seg_ready}")
    if seg_ready:
        print(f"[measure] thresholds: {pipe.segmenter.thresholds}")

    clean = sample(CLEAN_FOLDERS, args.per_folder, args.seed)
    defect = sample(DEFECT_FOLDERS, args.per_folder, args.seed)
    print(f"[measure] {len(clean)} clean photographs, {len(defect)} defect photographs\n")

    t0 = time.time()
    rows = []
    fp = 0
    for path, folder in clean:
        found = defects_in(pipe.audit_image(path))
        if found:
            fp += 1
        rows.append({"path": os.path.relpath(path, ENGINE_ROOT), "folder": folder,
                     "expected": "clean", "reported": found})
        if args.verbose or found:
            tag = "FALSE POSITIVE" if found else "clean"
            names = ", ".join(f"{d['class_name'].split('(')[0].strip()} "
                              f"{d['confidence'] * 100:.0f}%" for d in found)
            print(f"  clean   {os.path.basename(path)[:34]:36} {tag} {names}")

    hit = 0
    for path, folder in defect:
        found = defects_in(pipe.audit_image(path))
        if found:
            hit += 1
        rows.append({"path": os.path.relpath(path, ENGINE_ROOT), "folder": folder,
                     "expected": "defect", "reported": found})
        if args.verbose or not found:
            names = ", ".join(f"{d['class_name'].split('(')[0].strip()} "
                              f"{d['confidence'] * 100:.0f}%" for d in found)
            print(f"  defect  {os.path.basename(path)[:34]:36} "
                  f"{names if found else 'MISS'}")

    fp_rate = fp / max(1, len(clean))
    det_rate = hit / max(1, len(defect))
    print(f"\n{'=' * 62}")
    print(f"  false positives   {fp:3}/{len(clean):<3} ({fp_rate * 100:5.1f}%)   "
          f"clean roads reported as cracked or holed")
    print(f"  defects found     {hit:3}/{len(defect):<3} ({det_rate * 100:5.1f}%)   "
          f"annotated defects the product reports")
    print(f"{'=' * 62}")
    print("  Both matter. Either number alone can be made perfect by breaking "
          "the other.")

    report = {
        "generated_unix": int(time.time()),
        "measured_through": "pipeline.deep_inference_pipeline.DeepInferencePipeline.audit_image",
        "why_not_region_level": ("An earlier harness reimplemented the region loop and "
                                 "reported 27.8% false positives for a build that was "
                                 "actually producing 44.4%. Measuring anything but the "
                                 "real entry point measures the harness."),
        "clean_folders": CLEAN_FOLDERS, "defect_folders": DEFECT_FOLDERS,
        "clean_photographs": len(clean), "defect_photographs": len(defect),
        "false_positive_photos": fp, "false_positive_rate": round(fp_rate, 4),
        "detected_photos": hit, "detection_rate": round(det_rate, 4),
        "segmenter_loaded": seg_ready,
        # What this number is a measurement OF. Without it the figure travels
        # to a machine with a different corpus and a different model and is
        # read there as a property of the code. See
        # pipeline/corpus_fingerprint.py for the failure that taught this.
        "fingerprint": fingerprint(pipe.segmenter, CLEAN_FOLDERS + DEFECT_FOLDERS),
        "seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    os.makedirs(CKPT, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nreport -> {REPORT}")


if __name__ == "__main__":
    main()
