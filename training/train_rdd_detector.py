"""
Train a YOLOv8 road-damage detector on RDD2022 India, score it once on the
held-out photographs, and export it to ONNX for CPU serving.

    pip install ultralytics
    python -m training.train_rdd_detector                    # T4: ~45-70 min
    python -m training.train_rdd_detector --epochs 2 --fraction 0.05 --model yolov8n.pt   # smoke

Data
----
datasets/rdd2022_india, written by scripts/prepare_rdd2022_voc.py from the
official CRDDC 2022 release: labelled photographs split 70/15/15 BY PHOTOGRAPH
(seed 42) into train / valid / test, boxes for D00 longitudinal crack, D10
transverse crack, D20 alligator crack and D40 pothole. The official RDD2022
test split has no public labels, so "test" here is the held-out 15%.

Protocol
--------
* Model weights are chosen on `valid` only (Ultralytics keeps best.pt by
  validation fitness); `test` is scored exactly once, after training.
* The confidence threshold used when serving is chosen on `valid` too: the one
  that maximises F1 there. It is not tuned on test.
* Reported: mAP@0.5, mAP@0.5:0.95, precision, recall, and AP@0.5 per class.

Outputs
-------
    checkpoints/damage_rdd2022_india.onnx      (named so the COCO detector's
                                                 *detector*.onnx glob never picks it up)
    checkpoints/damage_rdd2022_india.json      class names, input size, threshold, test metrics
    checkpoints/road_damage_detector_report.json
"""

import argparse
import json
import os
import shutil
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
DATA_DIR = os.path.join(ENGINE_ROOT, "datasets", "rdd2022_india")
ONNX_NAME = "damage_rdd2022_india.onnx"
META_NAME = "damage_rdd2022_india.json"
REPORT_NAME = "road_damage_detector_report.json"
CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_LABELS = {"D00": "Longitudinal crack", "D10": "Transverse crack",
                "D20": "Alligator crack", "D40": "Pothole"}


def _metrics(m):
    """Ultralytics DetMetrics -> plain dict."""
    box = m.box
    per = {}
    try:
        for i, ci in enumerate(box.ap_class_index):
            name = CLASS_NAMES[int(ci)] if int(ci) < len(CLASS_NAMES) else str(int(ci))
            per[name] = {"ap50": round(float(box.ap50[i]), 4), "ap50_95": round(float(box.ap[i]), 4)}
    except Exception:
        pass
    return {"map50": round(float(box.map50), 4), "map50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
            "per_class": per}


def _best_conf_on_valid(m):
    """Confidence that maximises mean F1 over classes on the validation split."""
    try:
        import numpy as np
        f1 = np.asarray(m.box.f1_curve)          # (n_classes, 1000) over confidence 0..1
        if f1.ndim == 2 and f1.shape[1] > 1:
            xs = np.linspace(0, 1, f1.shape[1])
            return round(float(xs[int(np.argmax(f1.mean(0)))]), 3)
    except Exception:
        pass
    return 0.25


def count_split(split):
    d = os.path.join(DATA_DIR, "images", split)
    return len([f for f in os.listdir(d) if f.lower().endswith(".jpg")]) if os.path.isdir(d) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the training set (smoke runs)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=CKPT_DIR)
    a = ap.parse_args(argv)

    if not os.path.isdir(os.path.join(DATA_DIR, "images", "train")):
        sys.exit(f"No YOLO data at {DATA_DIR}. It is written by step 2 of scripts/colab_train_all.sh "
                 f"(scripts/prepare_rdd2022_voc.py); run this in the same session.")
    from ultralytics import YOLO

    # Absolute-path dataset file, so the result does not depend on the working directory.
    yaml_path = os.path.join(DATA_DIR, "data_abs.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {DATA_DIR}\ntrain: images/train\nval: images/valid\ntest: images/test\n"
                 f"nc: {len(CLASS_NAMES)}\nnames: [{', '.join(CLASS_NAMES)}]\n")
    n = {s: count_split(s) for s in ("train", "valid", "test")}
    print(f"[rdd-detector] photographs train {n['train']} | valid {n['valid']} | test {n['test']} "
          f"(split by photograph; test scored once)")

    t0 = time.time()
    model = YOLO(a.model)
    run_dir = os.path.join(ENGINE_ROOT, "runs", "rdd_india")
    model.train(data=yaml_path, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
                seed=42, deterministic=False, workers=a.workers, cos_lr=True, close_mosaic=5,
                fraction=a.fraction, project=os.path.dirname(run_dir), name=os.path.basename(run_dir),
                exist_ok=True, plots=False, verbose=False)
    train_min = round((time.time() - t0) / 60, 1)
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    best = YOLO(best_pt)

    val = best.val(data=yaml_path, split="val", imgsz=a.imgsz, batch=a.batch, plots=False, verbose=False)
    conf = _best_conf_on_valid(val)
    test = best.val(data=yaml_path, split="test", imgsz=a.imgsz, batch=a.batch, plots=False, verbose=False)
    val_m, test_m = _metrics(val), _metrics(test)
    print(f"  VAL  mAP50 {val_m['map50']}  mAP50-95 {val_m['map50_95']}  (serving confidence {conf}, chosen here)")
    print(f"  TEST mAP50 {test_m['map50']}  mAP50-95 {test_m['map50_95']}  P {test_m['precision']}  "
          f"R {test_m['recall']}  per class {test_m['per_class']}")

    exported = best.export(format="onnx", imgsz=a.imgsz, opset=12, dynamic=False, simplify=False)
    os.makedirs(a.out, exist_ok=True)
    onnx_path = os.path.join(a.out, ONNX_NAME)
    shutil.copy(str(exported), onnx_path)
    size_mb = round(os.path.getsize(onnx_path) / 1e6, 1)

    report = {
        "model": f"YOLOv8 ({a.model}, COCO-pretrained) fine-tuned on RDD2022 India, all layers trained",
        "classes": {c: CLASS_LABELS[c] for c in CLASS_NAMES},
        "data": {"source": "RDD2022 India (CRDDC 2022 official release, smartphone images)",
                 "split": "labelled photographs 70/15/15 by photograph, seed 42 (official test split has "
                          "no public labels)", "photographs": n, "fraction_of_train_used": a.fraction},
        "training": {"epochs_requested": a.epochs, "imgsz": a.imgsz, "batch": a.batch,
                     "patience": a.patience, "minutes": train_min,
                     "selection": "best.pt by validation fitness; test scored once"},
        "serving_confidence": conf, "serving_confidence_chosen_on": "validation split (max mean F1)",
        "validation": val_m, "test": test_m,
        "onnx": {"file": ONNX_NAME, "size_mb": size_mb, "opset": 12},
        "not_claimed": ("Smartphone photographs from RDD2022 India, not bus-camera frames. Boxes locate damage; "
                        "area and cost still come from the segmentation mask."),
        "trained_at_unix": int(time.time()),
    }
    meta = {"class_names": CLASS_NAMES, "class_labels": CLASS_LABELS, "input_size": a.imgsz,
            "conf_threshold": conf, "iou_threshold": 0.45, "test": test_m}
    with open(os.path.join(a.out, META_NAME), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(os.path.join(a.out, REPORT_NAME), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  exported {onnx_path} ({size_mb} MB); wrote {META_NAME} and {REPORT_NAME}")
    return report


if __name__ == "__main__":
    main()
