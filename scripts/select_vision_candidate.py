"""
Decide whether a newly trained image classifier (more data) replaces the served one.

    python -m training.train_finetune_cnn --out checkpoints/_candidate_vision --max-per-class 3000 ...
    python -m scripts.select_vision_candidate --candidate checkpoints/_candidate_vision

Why a new rule is needed
------------------------
Training on a bigger corpus changes the classifier's own train/validation/test split, so its validation
and test numbers are on different photographs from the served model's and cannot be compared. The one
set both models have never trained on is RDD2022 India's held-out crops (datasets/_eval_rdd2022_india).
It is split here, BY PHOTOGRAPH and by a fixed hash, into two halves:

    selection half   decides which model is served
    report half      is reported for the winner; it never takes part in the decision

Rule (fixed before either model is scored on the selection half)
---------------------------------------------------------------
Serve the candidate only if, on the selection half, its accuracy AND its macro-F1 (normal / crack /
pothole) are both higher than the served model's. Otherwise nothing changes.

When the candidate wins, its files (deep_vision_*.onnx/.json, finetune_*_report.json,
finetune_summary.json) replace the served ones, and "served_indian_roads" in finetune_summary.json becomes
its REPORT-half score, with a note, because the selection half helped choose it.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

CKPT = os.path.join(ENGINE_ROOT, "checkpoints")
EVAL_DIR = os.path.join(ENGINE_ROOT, "datasets", "_eval_rdd2022_india")
LABELS = {"normal": 0, "crack": 1, "pothole": 2}
RULE = ("serve the candidate only if, on the selection half of RDD2022 India's held-out crops (split by "
        "photograph, sha1 hash), its accuracy AND macro-F1 over normal/crack/pothole are both higher than the "
        "served model's; the report half never enters the decision")
SELECTION = "vision_candidate_selection.json"


def half_of(crop_name):
    """'select' or 'report' for a crop file, by its source photograph (India_XXXXXX_<k>.jpg)."""
    photo = os.path.splitext(os.path.basename(crop_name))[0].rsplit("_", 1)[0]
    return "select" if int(hashlib.sha1(f"road-shield-select:{photo}".encode()).hexdigest(), 16) % 2 == 0 else "report"


def eval_items(eval_dir=EVAL_DIR):
    """{'select': [(path, label)], 'report': [...]}."""
    out = {"select": [], "report": []}
    for name, label in LABELS.items():
        for p in sorted(glob.glob(os.path.join(eval_dir, name, "*.jpg"))):
            out[half_of(p)].append((p, label))
    return out


def metrics(y_true, y_pred):
    """Accuracy and macro-F1 over the three classes (predictions of any other class count as wrong)."""
    n = len(y_true)
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / n if n else 0.0
    f1s = []
    for c in LABELS.values():
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return {"images": n, "accuracy": round(acc, 4), "macro_f1": round(sum(f1s) / len(f1s), 4)}


def score(model, items):
    import cv2
    y_true, y_pred = [], []
    for p, label in items:
        im = cv2.imread(p)
        if im is None:
            continue
        r = model.predict_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        y_true.append(label)
        y_pred.append(int(r["class_id"]))
    return metrics(y_true, y_pred)


def decide(cand_sel, served_sel):
    """('candidate' | 'served', why). Pure function - unit-tested."""
    if not cand_sel or not cand_sel.get("images"):
        return "served", "candidate could not be scored"
    if not served_sel or not served_sel.get("images"):
        return "candidate", "the served classifier could not be scored"
    better = cand_sel["accuracy"] > served_sel["accuracy"] and cand_sel["macro_f1"] > served_sel["macro_f1"]
    why = (f"selection half: candidate accuracy {cand_sel['accuracy']:.4f} / macro-F1 {cand_sel['macro_f1']:.4f}, "
           f"served {served_sel['accuracy']:.4f} / {served_sel['macro_f1']:.4f}")
    return ("candidate" if better else "served"), why


def install(cand_dir, ckpt, report_half):
    """Replace the served classifier files with the candidate's."""
    for p in glob.glob(os.path.join(ckpt, "deep_vision_*.onnx")) + glob.glob(os.path.join(ckpt, "deep_vision_*.json")):
        os.remove(p)
    for p in (glob.glob(os.path.join(cand_dir, "deep_vision_*")) + glob.glob(os.path.join(cand_dir, "finetune_*.json"))):
        shutil.copy(p, os.path.join(ckpt, os.path.basename(p)))
    summ_path = os.path.join(ckpt, "finetune_summary.json")
    with open(summ_path, encoding="utf-8") as fh:
        summ = json.load(fh)
    summ["served_indian_roads_full_set"] = summ.get("served_indian_roads")
    summ["served_indian_roads"] = dict(report_half, note=(
        "RDD2022 India held-out crops, REPORT half only: the other half chose this model over the previous one "
        "(scripts/select_vision_candidate.py)"))
    with open(summ_path, "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    sel_path = os.path.join(ckpt, "vision_model_selection.json")
    sel = {}
    if os.path.exists(sel_path):
        with open(sel_path, encoding="utf-8") as fh:
            sel = json.load(fh)
    sel.update({"served": "deep_cnn", "replaced_by_candidate_unix": int(time.time()),
                "candidate_rule": RULE})
    with open(sel_path, "w", encoding="utf-8") as fh:
        json.dump(sel, fh, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="folder the candidate was trained into (--out)")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--eval-dir", default=EVAL_DIR)
    a = ap.parse_args(argv)
    from models.deep_vision_net import DeepVisionNet, load_best_vision_model

    items = eval_items(a.eval_dir)
    if not items["select"]:
        sys.exit(f"no Indian held-out crops in {a.eval_dir}: run scripts/ingest_rdd2022_india.py first")
    served, served_backend = load_best_vision_model(a.ckpt, verbose=False)
    candidate = DeepVisionNet(checkpoints_dir=a.candidate)
    if not candidate.is_ready:
        sys.exit(f"no candidate classifier in {a.candidate}")
    res = {"served": {"backend": served_backend, "select": score(served, items["select"]),
                      "report": score(served, items["report"])},
           "candidate": {"backend": candidate.backend, "select": score(candidate, items["select"]),
                         "report": score(candidate, items["report"])}}
    winner, why = decide(res["candidate"]["select"], res["served"]["select"])
    record = {"rule": RULE, "decided_before_report_half": True, "winner": winner, "why": why, "scores": res,
              "halves": {k: len(v) for k, v in items.items()}, "decided_unix": int(time.time())}
    print(f"[select-vision] served    select {res['served']['select']}  report {res['served']['report']}")
    print(f"[select-vision] candidate select {res['candidate']['select']}  report {res['candidate']['report']}")
    if winner == "candidate":
        install(a.candidate, a.ckpt, res["candidate"]["report"])
        print(f"[select-vision] SERVING the candidate: {why}")
    else:
        print(f"[select-vision] keeping the served classifier: {why}")
    with open(os.path.join(a.ckpt, SELECTION), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return record


if __name__ == "__main__":
    main()
