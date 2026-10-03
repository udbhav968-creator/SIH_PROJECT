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
    }
