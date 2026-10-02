"""
Accuracy on Indian roads: score the served classifier on the RDD2022 India test
crops written by scripts/ingest_rdd2022_india.py (photographs no training run reads).

    python -m scripts.eval_indian_roads --tag before    # current model
    python -m scripts.eval_indian_roads --tag after     # after retraining
"""
import argparse, glob, json, os, sys, time
import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE_ROOT)
EVAL_DIR = os.path.join(ENGINE_ROOT, "datasets", "_eval_rdd2022_india")
REPORT = os.path.join(ENGINE_ROOT, "checkpoints", "indian_roads_eval_report.json")
CLASSES = {"normal": 0, "crack": 1, "pothole": 2}


def evaluate(classifier):
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    y, p = [], []
    for name, cid in CLASSES.items():
        for f in sorted(glob.glob(os.path.join(EVAL_DIR, name, "*.jpg"))):
            probs = np.asarray(classifier.predict_probabilities(np.asarray(Image.open(f).convert("RGB")))).reshape(-1)
            y.append(cid); p.append(int(np.argmax(probs)))
    if not y:
        return None
    labels = sorted(set(y))
    per = f1_score(y, p, labels=labels, average=None, zero_division=0)
    return {"images": len(y),
            "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, labels=labels, average="macro", zero_division=0)), 4),
            "per_class_f1": {n: round(float(per[labels.index(c)]), 4) for n, c in CLASSES.items() if c in labels},
            "per_class_images": {n: int(sum(1 for v in y if v == c)) for n, c in CLASSES.items()},
            "confusion_rows_true_cols_pred": confusion_matrix(y, p, labels=[0, 1, 2, 3, 4, 5, 6]).tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="current"); a = ap.parse_args(argv)
    if not os.path.isdir(EVAL_DIR):
        sys.exit("Run python -m scripts.ingest_rdd2022_india first")
    from models.deep_vision_net import load_best_vision_model
    clf, backend = load_best_vision_model(os.path.join(ENGINE_ROOT, "checkpoints"), verbose=False)
    t = time.time(); r = evaluate(clf)
    if r is None:
        sys.exit("No Indian test crops found")
    r.update(backend=backend, seconds=round(time.time() - t, 1), measured_unix=int(time.time()))
    rep = json.load(open(REPORT)) if os.path.exists(REPORT) else {}
    rep[a.tag] = r
    man = os.path.join(EVAL_DIR, "manifest.json")
    if os.path.exists(man):
        rep["dataset"] = json.load(open(man))
    json.dump(rep, open(REPORT, "w"), indent=1)
    print(f"[Indian roads | {a.tag}] accuracy {r['accuracy'] * 100:.1f}%  macro-F1 {r['macro_f1']:.3f}  "
          f"on {r['images']} RDD2022 India test crops  per-class F1 {r['per_class_f1']}")
    return r


if __name__ == "__main__":
    main()
