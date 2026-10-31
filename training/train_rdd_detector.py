"""Train road damage bounding-box detector on RDD2022."""

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
        f1 = np.asarray(m.box.f1_curve)
        if f1.ndim == 2 and f1.shape[1] > 1:
            xs = np.linspace(0, 1, f1.shape[1])
            return round(float(xs[int(np.argmax(f1.mean(0)))]), 3)
    except Exception:
        pass
    return 0.25


def run_state(last_pt):
    """'unfinished', 'finished' (Ultralytics stores epoch -1 once a run ends) or 'unreadable'."""
    try:
        import torch
        ck = torch.load(last_pt, map_location="cpu", weights_only=False)
        return "unfinished" if int(ck.get("epoch", -1)) >= 0 else "finished"
    except Exception:
        return "unreadable"


def unfinished(last_pt):
    return run_state(last_pt) == "unfinished"


def names_for(tag):
    """(onnx, meta, report) file names: the India-only model keeps the names the server loads."""
    if tag == "india":
        return ONNX_NAME, META_NAME, REPORT_NAME
    return f"damage_rdd2022_{tag}.onnx", f"damage_rdd2022_{tag}.json", f"road_damage_detector_{tag}_report.json"


def count_split(split, data_dir=DATA_DIR):
    d = os.path.join(data_dir, "images", split)
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
    ap.add_argument("--fresh", action="store_true", help="start again even if an unfinished run can be resumed")
    ap.add_argument("--run-name", default=None,
                    help="folder under runs/ (default rdd_<tag>; smoke runs use their own, so they never touch the real run)")
    ap.add_argument("--data-dir", default=DATA_DIR, help="YOLO dataset folder (default: RDD2022 India)")
    ap.add_argument("--tag", default="india", help="'india' writes the served file names; anything else writes candidates")
    a = ap.parse_args(argv)
    a.run_name = a.run_name or f"rdd_{a.tag}"
    data_dir = os.path.abspath(a.data_dir)
    onnx_name, meta_name, report_name = names_for(a.tag)

    if not os.path.isdir(os.path.join(data_dir, "images", "train")):
        sys.exit(f"No YOLO data at {data_dir}. It is written by scripts/prepare_rdd2022_voc.py "
                 f"(or prepare_rdd2022_world.py); run this in the same session.")
    from ultralytics import YOLO

    yaml_path = os.path.join(data_dir, "data_abs.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {data_dir}\ntrain: images/train\nval: images/valid\ntest: images/test\n"
                 f"nc: {len(CLASS_NAMES)}\nnames: [{', '.join(CLASS_NAMES)}]\n")

    eval_yaml = yaml_path
    if os.path.isdir(os.path.join(data_dir, "images", "valid_india")):
        eval_yaml = os.path.join(data_dir, "india_eval_abs.yaml")
        with open(eval_yaml, "w") as fh:
            fh.write(f"path: {data_dir}\ntrain: images/train\nval: images/valid_india\ntest: images/test\n"
                     f"nc: {len(CLASS_NAMES)}\nnames: [{', '.join(CLASS_NAMES)}]\n")
    n = {s: count_split(s, data_dir) for s in ("train", "valid", "valid_india", "test") if count_split(s, data_dir)}
    print(f"[rdd-detector] photographs train {n['train']} | valid {n['valid']} | test {n['test']} "
          f"(split by photograph; test scored once)")

    t0 = time.time()
    run_dir = os.path.join(ENGINE_ROOT, "runs", a.run_name)
    last_pt = os.path.join(run_dir, "weights", "last.pt")
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    resumed = False
    state = run_state(last_pt) if os.path.exists(last_pt) and not a.fresh else None
    if state == "unreadable":
        sys.exit(f"[rdd-detector] {last_pt} cannot be read (interrupted copy, or a different Ultralytics version). "
                 f"Nothing was overwritten. Fix it, or pass --fresh to train again from the start.")
    if state == "unfinished":
        print(f"  RESUMING the unfinished run from {last_pt}")
        try:
            YOLO(last_pt).train(resume=True)
        except Exception:
            if unfinished(last_pt):
                raise
            print("  the run had already finished; evaluating its best.pt")
        resumed = True
    elif state == "finished" and os.path.exists(best_pt):
        print(f"  training already finished in {run_dir}; evaluating and exporting its best.pt (no retraining)")
        resumed = True
    if not resumed:
        model = YOLO(a.model)
        model.train(data=yaml_path, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
                    seed=42, deterministic=False, workers=a.workers, cos_lr=True, close_mosaic=10,
                    fraction=a.fraction, project=os.path.dirname(run_dir), name=os.path.basename(run_dir),
                    exist_ok=True, plots=False, verbose=False)
    train_min = round((time.time() - t0) / 60, 1)
    if not os.path.exists(best_pt):
        sys.exit(f"[rdd-detector] no best.pt in {run_dir}: training did not finish an epoch")
    best = YOLO(best_pt)

    val = best.val(data=eval_yaml, split="val", imgsz=a.imgsz, batch=a.batch, plots=False, verbose=False)
    conf = _best_conf_on_valid(val)
    test = best.val(data=eval_yaml, split="test", imgsz=a.imgsz, batch=a.batch, plots=False, verbose=False)
    val_m, test_m = _metrics(val), _metrics(test)
    val_all = None
    if eval_yaml != yaml_path:
        val_all = _metrics(best.val(data=yaml_path, split="val", imgsz=a.imgsz, batch=a.batch, plots=False,
                                    verbose=False))
    print(f"  VAL  mAP50 {val_m['map50']}  mAP50-95 {val_m['map50_95']}  (serving confidence {conf}, chosen here)")
    print(f"  TEST mAP50 {test_m['map50']}  mAP50-95 {test_m['map50_95']}  P {test_m['precision']}  "
          f"R {test_m['recall']}  per class {test_m['per_class']}")

    exported = best.export(format="onnx", imgsz=a.imgsz, opset=12, dynamic=False, simplify=False)
    os.makedirs(a.out, exist_ok=True)
    onnx_path = os.path.join(a.out, onnx_name)
    shutil.copy(str(exported), onnx_path)
    size_mb = round(os.path.getsize(onnx_path) / 1e6, 1)

    world = eval_yaml != yaml_path
    countries = None
    if world and os.path.exists(os.path.join(data_dir, "manifest.json")):
        with open(os.path.join(data_dir, "manifest.json"), encoding="utf-8") as fh:
            countries = sorted((json.load(fh).get("countries") or {}).keys())
    report = {
        "model": (f"YOLOv8 ({a.model}, COCO-pretrained) fine-tuned on RDD2022 "
                  f"{'+'.join(countries) if countries else ('several countries' if world else 'India')}, all layers trained"),
        "tag": a.tag,
        "classes": {c: CLASS_LABELS[c] for c in CLASS_NAMES},
        "data": {"source": ("RDD2022 " + (", ".join(countries) if countries else "India")
                            + " (CRDDC 2022 official release, smartphone/vehicle cameras)"),
                 "split": ("India: labelled photographs 70/15/15 by photograph, seed 42 (official test split has no "
                           "public labels)" + ("; other countries 90/10 train/valid by photograph, none in any test set"
                                               if world else "")),
                 "photographs": n, "fraction_of_train_used": a.fraction},
        "training": {"epochs_requested": a.epochs, "imgsz": a.imgsz, "batch": a.batch,
                     "patience": a.patience, "minutes_this_session": train_min, "resumed": resumed,
                     "selection": "best.pt by validation fitness; test scored once"},
        "serving_confidence": conf,
        "serving_confidence_chosen_on": ("India validation photographs (max mean F1)" if world
                                         else "validation split (max mean F1)"),
        "validation": val_m, "test": test_m,
        "validation_all_countries": val_all,
        "test_photographs": "RDD2022 India held-out 15% - identical for every detector trained here",
        "onnx": {"file": onnx_name, "size_mb": size_mb, "opset": 12},
        "not_claimed": ("Smartphone photographs from RDD2022 India, not bus-camera frames. Boxes locate damage; "
                        "area and cost still come from the segmentation mask."),
        "trained_at_unix": int(time.time()),
    }
    meta = {"class_names": CLASS_NAMES, "class_labels": CLASS_LABELS, "input_size": a.imgsz,
            "conf_threshold": conf, "iou_threshold": 0.45, "test": test_m}
    with open(os.path.join(a.out, meta_name), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(os.path.join(a.out, report_name), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  exported {onnx_path} ({size_mb} MB); wrote {meta_name} and {report_name}")
    return report


if __name__ == "__main__":
    main()
