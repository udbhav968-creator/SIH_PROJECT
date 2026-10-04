"""
One description of "the image classifier that is serving", read from the
reports on disk. Standard library only, because the Vercel handler (which may
not import NumPy or scikit-learn) uses it as well as the full server.

The site used to read the frozen-head report directly. Once a fine-tuned
network can be selected instead, that would have shown the numbers of a model
that is not running - so every page now reads this summary.
"""
import glob
import json
import os


def _read(ckpt, name):
    p = os.path.join(ckpt, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _sklearn_style(per_class):
    """finetune reports store {"f1": ..}; the site's table expects {"f1-score": ..}."""
    out = {}
    for name, v in (per_class or {}).items():
        out[name] = {"precision": v.get("precision"), "recall": v.get("recall"),
                     "f1-score": v.get("f1", v.get("f1-score")), "support": v.get("support")}
    return out


def served_classifier_summary(ckpt, require_files=True):
    """require_files=False for deployments that ship reports but not weights
    (Vercel excludes *.onnx); the summary then follows the recorded selection."""
    sel = _read(ckpt, "vision_model_selection.json") or {}
    ft = _read(ckpt, "finetune_summary.json")
    has_deep = (not require_files) or bool(glob.glob(os.path.join(ckpt, "deep_vision_*.onnx")))
    if sel.get("served") == "deep_cnn" and ft and has_deep:
        rep = _read(ckpt, f"finetune_{ft['served']}_report.json") or {}
        test = rep.get("test") or {}
        return {
            "kind": "fine_tuned_cnn",
            "label": f"{ft['chosen_arch']} fine-tuned end to end",
            "held_out_test_accuracy": test.get("accuracy"),
            "held_out_test_macro_f1": test.get("macro_f1"),
            "held_out_test_images": test.get("images"),
            "held_out_test_photographs": (rep.get("split") or {}).get("test_photographs"),
            "per_class_report": _sklearn_style(test.get("per_class")),
            "confusion_matrix": test.get("confusion_matrix"),
            "class_names": rep.get("class_names"),
            "indian_roads": rep.get("indian_roads_rdd2022_test"),
            "onnx": rep.get("onnx"),
            "selection": sel,
            "evidence": f"checkpoints/finetune_{ft['served']}_report.json",
        }
    heads = sorted(glob.glob(os.path.join(ckpt, "cnn_head_*_report.json")))
    for p in heads:
        r = _read(ckpt, os.path.basename(p)) or {}
        bb = r.get("backbone")
        if bb and ((not require_files) or os.path.exists(os.path.join(ckpt, f"cnn_backbone_{bb}.onnx"))):
            return {
                "kind": "frozen_embeddings_head",
                "label": f"{bb} embeddings -> {r.get('head')}",
                "held_out_test_accuracy": r.get("held_out_test_accuracy"),
                "held_out_test_macro_f1": r.get("held_out_test_macro_f1"),
                "held_out_test_images": r.get("held_out_test_images"),
                "held_out_test_photographs": r.get("held_out_test_photographs"),
                "per_class_report": r.get("per_class_report"),
                "selection": sel or None,
                "evidence": f"checkpoints/{os.path.basename(p)}",
            }
    return None


def training_extras(ckpt, require_files=True):
    """Fine-tuning and IMU comparison reports, for the metrics endpoint."""
    return {
        "served_classifier": served_classifier_summary(ckpt, require_files),
        "finetune_summary": _read(ckpt, "finetune_summary.json"),
        "vision_model_selection": _read(ckpt, "vision_model_selection.json"),
        "imu_model_selection": _read(ckpt, "imu_model_selection.json"),
        "indian_roads_eval": _read(ckpt, "indian_roads_eval_report.json"),
        "segmenter": served_segmenter_summary(ckpt, require_files),
        "road_damage_detector": _read(ckpt, "road_damage_detector_report.json"),
    }


def served_segmenter_summary(ckpt, require_files=True):
    """Which segmenter serves, with its held-out numbers and the comparison that chose it."""
    sel = _read(ckpt, "segmenter_selection.json")
    unet_meta = _read(ckpt, "defect_segmenter_unet.json")
    pixel = _read(ckpt, "defect_segmenter_report.json")
    unet_ok = bool(sel and sel.get("served") == "unet" and unet_meta and
                   (not require_files or os.path.exists(os.path.join(ckpt, "defect_segmenter_unet.onnx"))))
    if unet_ok:
        src = unet_meta
        kind, label = "unet", "U-Net (ResNet-18 encoder, ImageNet-pretrained), ONNX Runtime"
    else:
        src = pixel or {}
        kind, label = "pixel_classifier", "HistGradientBoosting pixel classifier on 11 features"
    fp = src.get("false_positives_on_clean_roads") or {}
    return {
        "kind": kind, "label": label,
        "iou": src.get("iou"),
        "test_photographs": (src.get("trained_on") or {}).get("test_photographs"),
        "clean_false_blob_rate": fp.get("photo_rate_any_blob"),
        "selection": ({"served": sel.get("served"), "rule": sel.get("rule"), "why": sel.get("why"),
                       "test": sel.get("test"),
                       "iou_selection_served": sel.get("iou_selection_served", sel.get("served")),
                       "deployment_check": sel.get("deployment_check")} if sel else None),
    }


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------
_HASH_CACHE = {}


def _file_info(ckpt, name, hash_files):
    p = os.path.join(ckpt, name)
    if not os.path.exists(p):
        return {"file": name, "present": False}
    st = os.stat(p)
    info = {"file": name, "present": True, "size_mb": round(st.st_size / 1e6, 2), "modified_unix": int(st.st_mtime)}
    if hash_files:
        key = (p, st.st_mtime, st.st_size)
        if key not in _HASH_CACHE:
            import hashlib
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            _HASH_CACHE[key] = h.hexdigest()
        info["sha256"] = _HASH_CACHE[key]
    return info


def model_registry(ckpt, hash_files=True):
    """
    Every trained model this deployment knows about: its file (with a SHA-256,
    so a deployed artefact can be matched to the run that produced it), when it
    was trained, its held-out numbers, whether it is serving, and the rule that
    decided. Built from the same reports as everything else - nothing typed in.
    """
    sc = served_classifier_summary(ckpt, require_files=hash_files) or {}
    seg = served_segmenter_summary(ckpt, require_files=hash_files) or {}
    ft = _read(ckpt, "finetune_summary.json") or {}
    vsel = _read(ckpt, "vision_model_selection.json") or {}
    out = []

    def add(name, task, kind, files, metrics, serving, trained_unix=None, decided_by=None, evidence=None):
        out.append({"name": name, "task": task, "kind": kind,
                    "artefacts": [_file_info(ckpt, f, hash_files) for f in files],
                    "metrics": {k: v for k, v in metrics.items() if v is not None},
                    "serving": serving, "trained_unix": trained_unix,
                    "decided_by": decided_by, "evidence": evidence})

    deep = sc.get("kind") == "fine_tuned_cnn"
    for arch, r in (ft.get("archs") or {}).items():
        rep = _read(ckpt, f"finetune_{arch}_report.json") or {}
        add(f"{arch} (fine-tuned)", "road-condition classification, 7 classes", "deep CNN, ImageNet-pretrained, all layers trained",
            [f"deep_vision_{arch}.onnx"] if deep and arch == ft.get("served") else [],
            {"test_accuracy": (r.get("test") or {}).get("accuracy"), "test_macro_f1": (r.get("test") or {}).get("macro_f1"),
             "india_accuracy": (r.get("indian_roads") or {}).get("accuracy"),
             "cpu_ms_per_image": r.get("onnx_cpu_ms_per_image")},
            bool(deep and arch == ft.get("served")), rep.get("trained_unix") or rep.get("trained_at_unix"),
            ft.get("selection_rule"), f"checkpoints/finetune_{arch}_report.json")
    head = _read(ckpt, "cnn_head_mobilenetv2_report.json") or {}
    if head:
        add("MobileNetV2 embeddings + " + str(head.get("head")), "road-condition classification, 7 classes",
            "frozen ImageNet CNN + trained head", ["cnn_backbone_mobilenetv2.onnx", "cnn_head_mobilenetv2.joblib"],
            {"test_accuracy": head.get("held_out_test_accuracy"), "test_macro_f1": head.get("held_out_test_macro_f1")},
            not deep and sc.get("kind") == "frozen_embeddings_head", head.get("trained_at_unix"),
            head.get("selection"), "checkpoints/cnn_head_mobilenetv2_report.json")
    vb = _read(ckpt, "vision_distress_report.json") or {}
    if vb:
        add("HOG + LBP + SVM", "road-condition classification (fallback)", "classical features + SVM",
            ["vision_distress_model.joblib"], {"validation_accuracy": vb.get("held_out_validation_accuracy")},
            False, vb.get("trained_at_unix"), "serves only when no CNN backbone is on disk",
            "checkpoints/vision_distress_report.json")
    px = _read(ckpt, "defect_segmenter_report.json") or {}
    un = _read(ckpt, "defect_segmenter_unet.json") or {}
    if un:
        add("U-Net (ResNet-18 encoder)", "defect segmentation", "deep segmenter, ImageNet encoder, all layers trained",
            ["defect_segmenter_unet.onnx"],
            {"test_crack_iou": ((un.get("iou") or {}).get("crack") or {}).get("iou"),
             "test_pothole_iou": ((un.get("iou") or {}).get("pothole") or {}).get("iou")},
            seg.get("kind") == "unet", un.get("trained_at_unix"),
            (seg.get("selection") or {}).get("rule"), "checkpoints/defect_segmenter_unet.json, checkpoints/segmenter_selection.json")
    if px:
        add("Pixel segmenter (11 features)", "defect segmentation", "gradient boosting per pixel",
            ["defect_segmenter.joblib"],
            {"test_crack_iou": ((px.get("iou") or {}).get("crack") or {}).get("iou"),
             "test_pothole_iou": ((px.get("iou") or {}).get("pothole") or {}).get("iou"),
             "clean_false_blob_rate": (px.get("false_positives_on_clean_roads") or {}).get("photo_rate_any_blob")},
            seg.get("kind") == "pixel_classifier", px.get("trained_at_unix"),
            "thresholds tuned on a calibration split", "checkpoints/defect_segmenter_report.json")
    det = _read(ckpt, "road_damage_detector_report.json") or {}
    if det:
        t = det.get("test") or {}
        add("YOLOv8 road-damage detector", "road-damage boxes (D00/D10/D20/D40)", "COCO-pretrained detector fine-tuned on RDD2022 India",
            ["damage_rdd2022_india.onnx"], {"test_map50": t.get("map50"), "test_map50_95": t.get("map50_95")},
            os.path.exists(os.path.join(ckpt, "damage_rdd2022_india.onnx")) or not hash_files,
            det.get("trained_at_unix"), det.get("serving_confidence_chosen_on"),
            "checkpoints/road_damage_detector_report.json")
    add("YOLOv8n (COCO)", "people, vehicles, signs", "pretrained detector, not retrained here",
        ["road_shield_detector.onnx"], {}, os.path.exists(os.path.join(ckpt, "road_shield_detector.onnx")) or not hash_files,
        None, "used as published", None)
    imu_sel = _read(ckpt, "imu_model_selection.json") or {}
    imu = _read(ckpt, "imu_shock_report.json") or {}
    hfr = imu_sel.get("held_out_for_reporting") or {}
    if hfr.get("cnn"):
        add("IMU 1-D CNN", "vibration shock classification, 4 classes", "deep 1-D CNN ensemble, from scratch",
            ["imu_shock_cnn.onnx"], {"held_out_accuracy": hfr["cnn"].get("accuracy"), "held_out_macro_f1": hfr["cnn"].get("macro_f1")},
            imu_sel.get("served") == "cnn", None, imu_sel.get("rule"), "checkpoints/imu_deep_report.json")
    if imu:
        add("IMU RandomForest", "vibration shock classification, 4 classes", "RandomForest on 36 features",
            ["imu_shock_model.joblib"], {"held_out_accuracy": imu.get("held_out_validation_accuracy")},
            imu_sel.get("served") != "cnn", imu.get("trained_at_unix"), imu_sel.get("rule"),
            "checkpoints/imu_shock_report.json")
    return {"models": out, "serving": [m["name"] for m in out if m["serving"]],
            "classifier_selection": vsel or None,
            "note": "sha256 identifies the exact artefact; reports name the run that produced it"}
