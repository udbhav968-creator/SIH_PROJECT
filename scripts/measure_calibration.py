"""
Is the classifier's confidence honest? When it says 90%, is it right about 90% of the time?

    python -m scripts.measure_calibration              # measure only
    python -m scripts.measure_calibration --apply      # also serve the fitted temperature, if it helps

Why it matters here: the Bayesian fusion gate multiplies the classifier's pothole probability with the
IMU's. An over-confident camera (95% when it is right 80% of the time) overrules the accelerometer far
more often than it should, which is exactly the false-alarm case the gate exists to stop.

Data: RDD2022 India's held-out crops (datasets/_eval_rdd2022_india, written by
scripts/ingest_rdd2022_india.py), which no training run reads. They are split by photograph with the same
fixed hash as scripts/select_vision_candidate.py:

    selection half   fits one temperature T (grid search, minimum negative log-likelihood)
    report half      every number reported, before and after T

Reported on the report half: accuracy, top-label expected calibration error (ECE, 15 equal-width bins),
maximum calibration error, negative log-likelihood and Brier score, plus the reliability table (bin
confidence vs accuracy) for a diagram.

Temperature scaling divides the logits by T before the softmax. It changes how confident the model is,
never which class it picks, so accuracy is identical before and after. --apply writes T into the served
model's sidecar only when the report half's NLL and ECE both improve; otherwise nothing is changed.
Writes checkpoints/vision_calibration_report.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)
CKPT = os.path.join(ENGINE_ROOT, "checkpoints")
OUT = os.path.join(CKPT, "vision_calibration_report.json")
BINS = 15
T_GRID = np.round(np.concatenate([np.arange(0.5, 1.0, 0.025), np.arange(1.0, 5.01, 0.05)]), 3)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, t=1.0):
    p = softmax(logits / t)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))


def brier(probs, y):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability(probs, y, bins=BINS):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, ece, mce = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        acc, c = float(correct[m].mean()), float(conf[m].mean())
        gap = abs(acc - c)
        ece += m.mean() * gap
        mce = max(mce, gap)
        rows.append({"from": round(float(lo), 3), "to": round(float(hi), 3), "count": int(m.sum()),
                     "mean_confidence": round(c, 4), "accuracy": round(acc, 4)})
    return round(float(ece), 4), round(float(mce), 4), rows


def summarise(logits, y, t):
    p = softmax(logits / t)
    ece, mce, table = reliability(p, y)
    return {"temperature": t, "images": int(len(y)), "accuracy": round(float((p.argmax(1) == y).mean()), 4),
            "ece": ece, "mce": mce, "nll": round(nll(logits, y, t), 4), "brier": round(brier(p, y), 4),
            "reliability": table}


def fit_temperature(logits, y):
    scores = [nll(logits, y, t) for t in T_GRID]
    return float(T_GRID[int(np.argmin(scores))])


def collect(model, items):
    import cv2
    logits, labels = [], []
    for path, label in items:
        im = cv2.imread(path)
        if im is None:
            continue
        logits.append(model.predict_logits(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))[0])
        labels.append(label)
    return np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=int)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--eval-dir", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    from models.deep_vision_net import DeepVisionNet
    from scripts.select_vision_candidate import EVAL_DIR, eval_items
    model = DeepVisionNet(checkpoints_dir=a.ckpt)
    if not model.is_ready:
        sys.exit("no fine-tuned classifier is served; nothing to calibrate")
    items = eval_items(a.eval_dir or EVAL_DIR)
    if not items["select"] or not items["report"]:
        sys.exit("no Indian held-out crops: run scripts/ingest_rdd2022_india.py first")
    ls, ys = collect(model, items["select"])
    lr, yr = collect(model, items["report"])
    t = fit_temperature(ls, ys)
    before, after = summarise(lr, yr, 1.0), summarise(lr, yr, t)
    helps = after["nll"] < before["nll"] and after["ece"] < before["ece"]
    report = {
        "model": model.backend, "measured_on": time.strftime("%Y-%m-%d"),
        "data": "RDD2022 India held-out crops; temperature fitted on the selection half, numbers from the report half",
        "label_space": "normal / crack / pothole crops scored against the 7-class output (top-label calibration)",
        "fitted_temperature": t, "report_half_before": before, "report_half_after": after,
        "temperature_improves_report_half": helps,
        "applied": False,
    }
    if a.apply and helps and model.weights_path and model.weights_path.endswith(".onnx"):
        side = os.path.splitext(model.weights_path)[0] + ".json"
        meta = {}
        if os.path.exists(side):
            with open(side, encoding="utf-8") as fh:
                meta = json.load(fh)
        meta["temperature"] = t
        meta["temperature_source"] = "scripts/measure_calibration.py (selection half of the Indian held-out crops)"
        with open(side, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        report["applied"] = True
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"[calibration] T = {t}: ECE {before['ece']} -> {after['ece']}, NLL {before['nll']} -> {after['nll']}, "
          f"accuracy {before['accuracy']} (unchanged); {'applied' if report['applied'] else 'not applied'}")
    return report


if __name__ == "__main__":
    main()
