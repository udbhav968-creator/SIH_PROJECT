"""
Decide which image classifier the server runs, by a rule fixed in advance.

Candidates
    cnn_embeddings  frozen MobileNetV2 + scikit-learn head   (training/train_cnn_head.py)
    deep_cnn        network fine-tuned end to end           (training/train_finetune_cnn.py)

Rule (written before either model's test numbers were read):
    serve the fine-tuned network only if, on data that is NOT the test set, it
    beats the head on BOTH accuracy and macro-F1. The head's estimate is its
    5-fold grouped cross-validation on the fit split; the fine-tuned network's
    is its validation split (its train-only model, before the train+val refit).

The test numbers of both are copied into the selection record for reporting.
They do not enter the decision.

    python -m scripts.select_vision_model
"""
import json
import os
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT = os.path.join(ENGINE_ROOT, "checkpoints")


def _load(name):
    p = os.path.join(CKPT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def decide(head, ft):
    """Pure function of the two reports - unit-tested."""
    if not ft:
        return "cnn_embeddings", "no fine-tuned network has been trained"
    if not head:
        return "deep_cnn", "no frozen-embedding head report on disk"
    hc = head["heads_compared"][head["head"] + "|" + head.get("features", "plain")] \
        if head.get("head") and head.get("heads_compared") else None
    if hc is None:
        # fall back to the best CV entry recorded
        hc = max(head.get("heads_compared", {}).values(), key=lambda r: r.get("score", 0), default=None)
    if hc is None:
        return "deep_cnn", "head report has no cross-validation record"
    arch = ft["chosen_arch"]
    v = ft["archs"][arch]["val"]
    better = v["accuracy"] > hc["cv_accuracy"] and v["macro_f1"] > hc["cv_macro_f1"]
    why = (f"fine-tuned {arch}: validation accuracy {v['accuracy']:.4f}, macro-F1 {v['macro_f1']:.4f}; "
           f"frozen head: grouped-CV accuracy {hc['cv_accuracy']:.4f}, macro-F1 {hc['cv_macro_f1']:.4f}")
    return ("deep_cnn" if better else "cnn_embeddings"), why


def main():
    head = _load("cnn_head_mobilenetv2_report.json")
    ft = _load("finetune_summary.json")
    served, why = decide(head, ft)
    rec = {
        "served": served,
        "rule": "serve the fine-tuned network only if it beats the frozen head on BOTH accuracy and "
                "macro-F1, measured on non-test data (head: grouped 5-fold CV on the fit split; "
                "fine-tuned: validation split). Test numbers are recorded, not used.",
        "evidence": why,
        "test_for_reporting": {
            "cnn_embeddings": {"accuracy": (head or {}).get("held_out_test_accuracy"),
                               "macro_f1": (head or {}).get("held_out_test_macro_f1"),
                               "images": (head or {}).get("held_out_test_images")},
            "deep_cnn": (ft or {}).get("served_test"),
        },
        "decided_unix": int(time.time()),
    }
    with open(os.path.join(CKPT, "vision_model_selection.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
