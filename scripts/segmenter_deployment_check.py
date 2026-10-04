"""
End-to-end deployment check between the two segmenters.

    python -m scripts.segmenter_deployment_check

Why a second check
------------------
training/train_unet_segmenter.py picks a segmenter by MASK IoU on DNIT
photographs - the dataset both segmenters were trained on. That is the right
test of a mask, and the U-Net wins it clearly. It is not the question the
product asks. The product asks, through audit_image(), "does this photograph
contain a crack or a pothole?" on road photographs from other sources, where
the mask only proposes regions and the classifier and gates decide.

Measured on the U-Net build, that question came out differently: it found
16 of 24 annotated defects (all 8 misses were Kaggle pothole photographs), and
a pipeline test that requires mask proposals on pothole photographs failed.
A better mask on its own dataset had not become a better product.

Rule (fixed before the pixel classifier's current numbers were measured)
------------------------------------------------------------------------
Both segmenters are run through audit_image() on the same photographs that
scripts/measure_detection_quality.py uses (clean roads and annotated defects
from other datasets; never the DNIT test split). The U-Net stays served only
if it finds AT LEAST as many defects AND reports NO MORE false positives than
the pixel classifier. Otherwise the pixel classifier is served and the U-Net
remains on disk, measured and reported.

The decision is written into checkpoints/segmenter_selection.json under
"deployment_check", next to the IoU selection it overrides or confirms.
"""

import argparse
import json
import os
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

CKPT = os.path.join(ENGINE_ROOT, "checkpoints")
SEL = os.path.join(CKPT, "segmenter_selection.json")
RULE = ("U-Net served only if, through audit_image() on the detection-quality photographs (clean roads and "
        "annotated defects from other datasets, never the DNIT test split), it finds at least as many "
        "defects and reports no more false positives than the pixel classifier. Fixed before the pixel "
        "classifier's current end-to-end numbers were measured.")


def run(pipe, segmenter, clean, defect, label):
    from scripts.measure_detection_quality import defects_in
    pipe.segmenter = segmenter
    t0 = time.time()
    fp = sum(1 for p, _f in clean if defects_in(pipe.audit_image(p)))
    misses = [os.path.basename(p) for p, _f in defect if not defects_in(pipe.audit_image(p))]
    hit = len(defect) - len(misses)
    out = {"false_positive_photos": fp, "clean_photographs": len(clean),
           "detected_photos": hit, "defect_photographs": len(defect),
           "false_positive_rate": round(fp / max(1, len(clean)), 4),
           "detection_rate": round(hit / max(1, len(defect)), 4),
           "missed": misses, "seconds": round(time.time() - t0, 1)}
    print(f"  {label:17s} false positives {fp:2d}/{len(clean)}   defects found {hit:2d}/{len(defect)}"
          f"   ({out['seconds']:.0f}s)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-folder", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    from models.defect_segmenter import DefectSegmenter
    from models.unet_segmenter import UNetSegmenter
    from pipeline.deep_inference_pipeline import DeepInferencePipeline
    from scripts.measure_detection_quality import CLEAN_FOLDERS, DEFECT_FOLDERS, sample

    if not os.path.exists(SEL):
        sys.exit("No checkpoints/segmenter_selection.json - train the U-Net first.")
    with open(SEL, "r", encoding="utf-8") as fh:
        sel = json.load(fh)
    unet = UNetSegmenter(CKPT)
    pixel = DefectSegmenter(os.path.join(CKPT, "defect_segmenter.joblib"))
    if not unet.is_ready:
        sys.exit(f"U-Net did not load: {unet.load_error or 'file missing'}")
    if not pixel.is_ready:
        sys.exit("The pixel classifier did not load (scikit-learn must be 1.8.x): "
                 "pip install \"scikit-learn>=1.8,<1.9\"")

    pipe = DeepInferencePipeline(CKPT)
    clean = sample(CLEAN_FOLDERS, a.per_folder, a.seed)
    defect = sample(DEFECT_FOLDERS, a.per_folder, a.seed)
    print(f"[deployment check] {len(clean)} clean, {len(defect)} defect photographs, through audit_image()")
    res = {"unet": run(pipe, unet, clean, defect, "U-Net"),
           "pixel_classifier": run(pipe, pixel, clean, defect, "pixel classifier")}
    u, p = res["unet"], res["pixel_classifier"]
    passed = u["detected_photos"] >= p["detected_photos"] and u["false_positive_photos"] <= p["false_positive_photos"]
    why = (f"end to end, U-Net found {u['detected_photos']}/{u['defect_photographs']} defects with "
           f"{u['false_positive_photos']}/{u['clean_photographs']} false positives; the pixel classifier found "
           f"{p['detected_photos']}/{p['defect_photographs']} with {p['false_positive_photos']}/{p['clean_photographs']}")

    # Keep the IoU selection's own decision and reason, so re-running is idempotent.
    sel.setdefault("iou_selection_served", sel.get("served"))
    sel.setdefault("why_iou", sel.get("why", ""))
    sel["deployment_check"] = {"rule": RULE, "passed": passed, "why": why, "results": res,
                               "checked_unix": int(time.time())}
    if sel["iou_selection_served"] == "unet":
        sel["served"] = "unet" if passed else "pixel_classifier"
        sel["why"] = sel["why_iou"] + ("; confirmed by the end-to-end check: " if passed
                                       else "; NOT served after the end-to-end check: ") + why
    with open(SEL, "w", encoding="utf-8") as fh:
        json.dump(sel, fh, indent=2, default=float)
    print(f"\n  {'PASSED' if passed else 'FAILED'}: {why}")
    print(f"  served segmenter -> {sel['served']}   (written to checkpoints/segmenter_selection.json)")
    return sel


if __name__ == "__main__":
    main()
