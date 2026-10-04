"""
Two checks before the YOLOv8 road-damage detector is shown to anyone.

    python -m scripts.verify_rdd_detector --artefact      # on the training machine, after training
    python -m scripts.verify_rdd_detector --clean-roads   # on the serving machine, with the datasets

1. --artefact: does the file we serve behave like the network we trained?
   The trainer's test mAP comes from Ultralytics running best.pt. What the
   product runs is the exported ONNX file through OUR decoder
   (models/road_damage_detector.py: letterbox, decode, NMS). So:
     * the ONNX file is scored on the same held-out test photographs, and
     * on 100 test photographs our serving code's boxes are matched against
       Ultralytics' boxes from best.pt (same class, IoU >= 0.5).
   Rule: served only if ONNX test mAP@0.5 is within 0.02 of best.pt's AND our
   decoder recovers at least 90% of best.pt's boxes with at least 90% of ours
   matching one. (The same lesson as the U-Net: check the artefact, not the
   network.)

2. --clean-roads: does it draw damage on roads that have none?
   The boxes never decide whether a defect is reported (area and cost come from
   the segmentation mask), but a "Pothole" box on a zebra crossing is still a
   wrong thing to show. Run on the 36 clean-road photographs that
   scripts/measure_detection_quality.py uses.
   Rule (fixed before measuring): boxes are shown only if they appear on no
   more clean photographs than the pipeline itself already flags in the
   recorded end-to-end check (3 of 36 when this was written; read from
   checkpoints/segmenter_selection.json).

Both write their result into checkpoints/damage_rdd2022_india.json and
checkpoints/road_damage_detector_report.json. models/road_damage_detector.py
refuses to load a model whose recorded check failed.
"""

import argparse
import json
import os
import sys

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

CKPT = os.path.join(ENGINE_ROOT, "checkpoints")
DATA = os.path.join(ENGINE_ROOT, "datasets", "rdd2022_india")
META = os.path.join(CKPT, "damage_rdd2022_india.json")
REPORT = os.path.join(CKPT, "road_damage_detector_report.json")
ONNX = os.path.join(CKPT, "damage_rdd2022_india.onnx")
BEST_PT = os.path.join(ENGINE_ROOT, "runs", "rdd_india", "weights", "best.pt")


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=float)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _match(ref, ours, thr=0.5):
    """Greedy one-to-one matching of (cls, xyxy) lists; returns matched count."""
    used, n = set(), 0
    for rc, rb in ref:
        best, bj = 0.0, None
        for j, (oc, ob) in enumerate(ours):
            if j in used or oc != rc:
                continue
            v = _iou(rb, ob)
            if v > best:
                best, bj = v, j
        if bj is not None and best >= thr:
            used.add(bj)
            n += 1
    return n


def artefact_check(n_images=100):
    import glob
    import cv2
    from ultralytics import YOLO
    from models.road_damage_detector import RoadDamageDetector

    meta, report = _load(META), _load(REPORT)
    conf, imgsz = float(meta.get("conf_threshold", 0.25)), int(meta.get("input_size", 640))
    names = meta.get("class_names") or ["D00", "D10", "D20", "D40"]
    yaml_path = os.path.join(DATA, "data_abs.yaml")

    print("[artefact] scoring the exported ONNX file on the held-out test photographs (CPU)")
    onnx_val = YOLO(ONNX, task="detect").val(data=yaml_path, split="test", imgsz=imgsz, batch=1,
                                             device="cpu", plots=False, verbose=False)
    onnx_map50 = round(float(onnx_val.box.map50), 4)
    pt_map50 = float((report.get("test") or {}).get("map50") or 0.0)

    det = RoadDamageDetector(CKPT)
    det.verifying = True                     # the check has to run the model it is checking
    ref_model = YOLO(BEST_PT)
    paths = sorted(glob.glob(os.path.join(DATA, "images", "test", "*.jpg")))[:n_images]
    n_ref = n_ours = m_ref = m_ours = 0
    for p in paths:
        r = ref_model.predict(p, conf=conf, iou=0.45, imgsz=imgsz, verbose=False)[0]
        ref = [(names[int(c)], [float(v) for v in b]) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())]
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        ours = []
        for d in det.detect(rgb):
            x, y, w, h = d["bbox_pixels"]
            ours.append((d["damage_code"], [x, y, x + w, y + h]))
        n_ref, n_ours = n_ref + len(ref), n_ours + len(ours)
        m_ref += _match(ref, ours)
        m_ours += _match(ours, ref)
    recall = round(m_ref / n_ref, 4) if n_ref else 1.0
    precision = round(m_ours / n_ours, 4) if n_ours else 1.0
    passed = abs(onnx_map50 - pt_map50) <= 0.02 and recall >= 0.9 and precision >= 0.9
    res = {"onnx_test_map50": onnx_map50, "best_pt_test_map50": pt_map50,
           "serving_decoder_vs_best_pt": {"photographs": len(paths), "best_pt_boxes": n_ref,
                                          "serving_boxes": n_ours, "recall": recall, "precision": precision},
           "rule": "ONNX test mAP@0.5 within 0.02 of best.pt, and the serving decoder recovers >= 90% of "
                   "best.pt's boxes with >= 90% of its own matched (same class, IoU >= 0.5)",
           "passed": passed}
    meta["artefact_check"] = report["artefact_check"] = res
    _save(META, meta)
    _save(REPORT, report)
    print(f"  ONNX test mAP50 {onnx_map50} vs best.pt {pt_map50}; serving decoder recall {recall}, "
          f"precision {precision} over {len(paths)} photographs")
    print(f"  ARTEFACT CHECK {'PASSED' if passed else 'FAILED'}")
    return passed


def clean_roads_check(per_folder=12, seed=7):
    import cv2
    from models.road_damage_detector import RoadDamageDetector
    from scripts.measure_detection_quality import CLEAN_FOLDERS, sample

    meta, report = _load(META), _load(REPORT)
    if (meta.get("artefact_check") or {}).get("passed") is False:
        sys.exit("The artefact check failed; the detector is not served.")
    sel_path = os.path.join(CKPT, "segmenter_selection.json")
    limit = None
    if os.path.exists(sel_path):
        dc = (_load(sel_path).get("deployment_check") or {}).get("results") or {}
        served = "unet" if _load(sel_path).get("served") == "unet" else "pixel_classifier"
        limit = (dc.get(served) or {}).get("false_positive_photos")
    if limit is None:
        sys.exit("No recorded end-to-end check to compare with; run scripts.segmenter_deployment_check first.")

    det = RoadDamageDetector(CKPT)
    det.verifying = True
    if not det.is_ready:
        sys.exit("The detector did not load.")
    clean = sample(CLEAN_FOLDERS, per_folder, seed)
    flagged = []
    for p, _f in clean:
        raw = cv2.imread(p)
        if raw is None:
            continue
        boxes = det.detect(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        if boxes:
            flagged.append({"photo": os.path.basename(p),
                            "boxes": [f"{b['class_name']} {b['confidence']:.2f}" for b in boxes[:3]]})
    passed = len(flagged) <= int(limit)
    res = {"rule": "damage boxes shown only if they appear on no more clean-road photographs than the "
                   "pipeline itself already flags in the recorded end-to-end check",
           "clean_photographs": len(clean), "photographs_with_a_box": len(flagged),
           "pipeline_false_positive_photos": int(limit), "flagged": flagged, "passed": passed}
    meta["deployment_check"] = report["deployment_check"] = res
    _save(META, meta)
    _save(REPORT, report)
    print(f"  boxes on {len(flagged)}/{len(clean)} clean-road photographs (limit {limit})")
    print(f"  CLEAN-ROAD CHECK {'PASSED - boxes will be shown' if passed else 'FAILED - boxes stay hidden'}")
    return passed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artefact", action="store_true")
    ap.add_argument("--clean-roads", action="store_true")
    a = ap.parse_args(argv)
    if not (a.artefact or a.clean_roads):
        ap.error("choose --artefact and/or --clean-roads")
    ok = True
    if a.artefact:
        ok = artefact_check() and ok
    if a.clean_roads:
        ok = clean_roads_check() and ok
    return ok


if __name__ == "__main__":
    main()
