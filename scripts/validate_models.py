"""
Independent audit of the models in this project.

    python -m scripts.validate_models              # everything
    python -m scripts.validate_models --only leakage,cv
    python -m scripts.validate_models --folds 5

This is deliberately separate from training. Training reports its own numbers;
this re-derives them from scratch and looks for the specific ways a vision
project fools itself:

  leakage      Does the same photograph (or a near-duplicate of it) appear in
               more than one class folder, or in both halves of a split? Uses
               the perceptual hash from models/forensic_audit_engine.py, so it
               catches re-encoded and lightly-edited copies, not just identical
               files.

  cv           Grouped k-fold cross-validation of the classifier. One held-out
               split can be lucky; k folds with a standard deviation cannot.
               Groups are whole photographs, so augmented copies never straddle
               a fold.

  calibration  When the model says 80%, is it right 80% of the time? Reports
               expected calibration error, maximum calibration error, Brier
               score and a reliability table. An overconfident model is worse
               than an uncertain one in a system that dispatches repair crews.

  latency      p50/p95/p99 timings for feature extraction, classification and
               the whole pipeline, measured on real images.

  robustness   Accuracy under brightness, blur and JPEG-compression shifts -
               the conditions a dashcam actually meets.

Writes checkpoints/validation_report.json and prints a summary. Everything is
measured; nothing here has a hard-coded expected value.
"""

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import numpy as np
from PIL import Image

from data.image_dataset import CLASS_FOLDERS, CLASS_NAMES, DATASETS_ROOT, load_image
from models.forensic_audit_engine import ForensicDuplicateHasher
from training.train_deep_vision import collect_files, grouped_split

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")


def _print_header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 1. leakage
# ---------------------------------------------------------------------------
def audit_leakage(items, hamming_threshold=5, max_images=4000):
    _print_header("LEAKAGE AUDIT - near-duplicate photographs across classes and splits")
    hasher = ForensicDuplicateHasher(hash_size=8, duplicate_hamming_threshold=hamming_threshold)
    hashes, meta = [], []
    for path, cls, grp in items[:max_images]:
        try:
            img = Image.open(path).convert("RGB").resize((64, 64))
        except Exception:
            continue
        hashes.append(hasher.compute_hash(np.asarray(img, dtype=np.uint8)))
        meta.append((path, cls, grp))

    cross_class, within_class = [], 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hasher.hamming_distance(hashes[i], hashes[j]) <= hamming_threshold:
                if meta[i][1] != meta[j][1]:
                    cross_class.append({
                        "image_a": os.path.relpath(meta[i][0], ENGINE_ROOT), "class_a": CLASS_NAMES[meta[i][1]],
                        "image_b": os.path.relpath(meta[j][0], ENGINE_ROOT), "class_b": CLASS_NAMES[meta[j][1]],
                    })
                else:
                    within_class += 1

    train, val, test = grouped_split(items)
    gtr, gva, gte = ({i[2] for i in s} for s in (train, val, test))
    split_overlap = {"train_val": sorted(gtr & gva), "train_test": sorted(gtr & gte), "val_test": sorted(gva & gte)}

    print(f"images hashed                     : {len(hashes)}")
    print(f"near-duplicate pairs, same class  : {within_class}   (expected - these are the augmented copies)")
    print(f"near-duplicate pairs, DIFFERENT class: {len(cross_class)}   {'<- contradictory labels' if cross_class else '(none - good)'}")
    for c in cross_class[:10]:
        print(f"    {c['class_a']}  vs  {c['class_b']}")
        print(f"      {c['image_a']}\n      {c['image_b']}")
    leaked = sum(len(v) for v in split_overlap.values())
    print(f"groups appearing in two splits    : {leaked}   {'<- LEAKAGE' if leaked else '(none - splits are clean)'}")
    return {
        "images_hashed": len(hashes),
        "near_duplicate_pairs_same_class": within_class,
        "near_duplicate_pairs_cross_class": len(cross_class),
        "cross_class_examples": cross_class[:25],
        "group_overlap_between_splits": {k: len(v) for k, v in split_overlap.items()},
        "verdict": "CLEAN" if (not cross_class and not leaked) else "REVIEW_NEEDED",
    }


# ---------------------------------------------------------------------------
# 2. grouped cross-validation
# ---------------------------------------------------------------------------
def _features_for(items, cache={}):
    from data.feature_extraction import extract_image_features
    X, y, groups = [], [], []
    for path, cls, grp in items:
        if path not in cache:
            cache[path] = extract_image_features(load_image(path))
        X.append(cache[path]); y.append(cls); groups.append(grp)
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def cross_validate(items, folds=5, seed=42):
    _print_header(f"GROUPED {folds}-FOLD CROSS-VALIDATION - scikit-learn baseline")
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score, accuracy_score
    from models.vision_distress_net import VisionDistressNet

    t0 = time.time()
    X, y, groups = _features_for(items)
    print(f"feature matrix: {X.shape}  ({time.time() - t0:.1f}s)")

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s, per_fold = [], [], []
    for k, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
        model = VisionDistressNet()
        model.fit(X[tr], y[tr])
        pred, _conf, _probs = model.predict(X[te])
        acc = float(accuracy_score(y[te], pred))
        f1 = float(f1_score(y[te], pred, average="macro", zero_division=0))
        accs.append(acc); f1s.append(f1)
        per_fold.append({"fold": k, "train": int(len(tr)), "test": int(len(te)),
                         "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})
        print(f"  fold {k}: train {len(tr):4d} test {len(te):4d}  acc {acc * 100:5.1f}%  macro-F1 {f1:.3f}")

    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s))
    print(f"\n  accuracy : {mean_acc * 100:.1f}% ± {std_acc * 100:.1f}")
    print(f"  macro-F1 : {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"  random guess baseline: {100.0 / len(CLASS_NAMES):.1f}%")
    return {"folds": folds, "per_fold": per_fold,
            "accuracy_mean": round(mean_acc, 4), "accuracy_std": round(std_acc, 4),
            "macro_f1_mean": round(mean_f1, 4), "macro_f1_std": round(std_f1, 4),
            "random_guess_baseline": round(1.0 / len(CLASS_NAMES), 4)}


# ---------------------------------------------------------------------------
# 3. calibration
# ---------------------------------------------------------------------------
def calibration(items, seed=42, bins=10):
    _print_header("CALIBRATION - does a stated confidence mean what it says?")
    from models.vision_distress_net import VisionDistressNet

    train, val, test = grouped_split(items, seed=seed)
    X_tr, y_tr, _ = _features_for(train)
    X_te, y_te, _ = _features_for(test + val)
    model = VisionDistressNet()
    model.fit(X_tr, y_tr)
    pred, conf, probs = model.predict(X_te)
    correct = (pred == y_te).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    table, ece, mce = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= 1.0)
        n = int(mask.sum())
        if n == 0:
            continue
        avg_conf, acc = float(conf[mask].mean()), float(correct[mask].mean())
        gap = abs(avg_conf - acc)
        ece += (n / len(conf)) * gap
        mce = max(mce, gap)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": n, "mean_confidence": round(avg_conf, 3),
                      "actual_accuracy": round(acc, 3), "gap": round(avg_conf - acc, 3)})

    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_te)), y_te] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    print(f"  {'bin':>10}  {'n':>5}  {'says':>6}  {'actually':>9}  {'gap':>7}")
    for row in table:
        flag = "  <- overconfident" if row["gap"] > 0.15 else ("  <- underconfident" if row["gap"] < -0.15 else "")
        print(f"  {row['bin']:>10}  {row['n']:5d}  {row['mean_confidence']:6.2f}  {row['actual_accuracy']:9.2f}  {row['gap']:+7.2f}{flag}")
    print(f"\n  expected calibration error : {ece:.3f}   (0 is perfect; below 0.10 is good)")
    print(f"  maximum calibration error  : {mce:.3f}")
    print(f"  Brier score                : {brier:.3f}   (lower is better)")
    return {"expected_calibration_error": round(ece, 4), "maximum_calibration_error": round(mce, 4),
            "brier_score": round(brier, 4), "reliability_table": table, "n_evaluated": int(len(conf))}


# ---------------------------------------------------------------------------
# 4. latency
# ---------------------------------------------------------------------------
def latency(items, n=40):
    _print_header("LATENCY - measured on real images, not estimated")
    from data.feature_extraction import extract_image_features
    from pipeline.deep_inference_pipeline import DeepInferencePipeline

    sample = [p for p, _c, _g in items[:: max(1, len(items) // n)]][:n]
    pipe = DeepInferencePipeline(CKPT_DIR)

    def timed(fn, paths):
        out = []
        for p in paths:
            t = time.perf_counter()
            fn(p)
            out.append((time.perf_counter() - t) * 1000.0)
        return np.array(out)

    feat_ms = timed(lambda p: extract_image_features(load_image(p)), sample)
    full_ms = timed(lambda p: pipe.audit_image(image_input=p), sample)

    def stats(a, label):
        row = {"p50_ms": round(float(np.percentile(a, 50)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
               "p99_ms": round(float(np.percentile(a, 99)), 1), "mean_ms": round(float(a.mean()), 1), "n": len(a)}
        print(f"  {label:28s} p50 {row['p50_ms']:7.1f} ms | p95 {row['p95_ms']:7.1f} ms | p99 {row['p99_ms']:7.1f} ms")
        return row

    result = {"feature_extraction": stats(feat_ms, "feature extraction"),
              "full_pipeline": stats(full_ms, "full pipeline per image")}
    fps = 1000.0 / max(1e-6, float(np.percentile(full_ms, 50)))
    print(f"\n  sustained throughput at p50: {fps:.1f} images/second on this machine")
    result["throughput_images_per_second_p50"] = round(fps, 2)
    return result


# ---------------------------------------------------------------------------
# 5. robustness
# ---------------------------------------------------------------------------
def robustness(items, seed=42, per_class=12):
    _print_header("ROBUSTNESS - accuracy under dashcam-like image degradation")
    from models.vision_distress_net import VisionDistressNet
    from data.feature_extraction import extract_image_features

    train, val, test = grouped_split(items, seed=seed)
    X_tr, y_tr, _ = _features_for(train)
    model = VisionDistressNet()
    model.fit(X_tr, y_tr)

    by_class = defaultdict(list)
    for path, cls, _g in test + val:
        if len(by_class[cls]) < per_class:
            by_class[cls].append(path)
    eval_set = [(p, c) for c, ps in by_class.items() for p in ps]

    def bright(img, factor):
        return np.clip(np.asarray(img, np.float32) * factor, 0, 255).astype(np.uint8)

    def blur(img, radius):
        from PIL import ImageFilter
        return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8)

    def jpeg(img, quality):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)

    conditions = [
        ("original", lambda im: im),
        ("dim (x0.6)", lambda im: bright(im, 0.6)),
        ("bright (x1.5)", lambda im: bright(im, 1.5)),
        ("blur r=2", lambda im: blur(im, 2)),
        ("blur r=4", lambda im: blur(im, 4)),
        ("JPEG q=30", lambda im: jpeg(im, 30)),
        ("JPEG q=10", lambda im: jpeg(im, 10)),
    ]

    results = {}
    base_acc = None
    for name, fn in conditions:
        correct = 0
        for path, cls in eval_set:
            img = load_image(path)
            feat = extract_image_features(fn(img)).reshape(1, -1)
            pred, _c, _p = model.predict(feat)
            correct += int(pred[0] == cls)
        acc = correct / len(eval_set)
        if base_acc is None:
            base_acc = acc
        delta = acc - base_acc
        results[name] = {"accuracy": round(acc, 4), "delta_vs_original": round(delta, 4)}
        print(f"  {name:16s} accuracy {acc * 100:5.1f}%   {delta * 100:+5.1f} points")
    results["_n_images"] = len(eval_set)
    return results


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="leakage,cv,calibration,latency,robustness",
                    help="comma-separated subset of checks to run")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    items = collect_files()
    if not items:
        sys.exit("No images found under datasets/*/real_images.")
    print(f"auditing {len(items)} images across {len(CLASS_NAMES)} classes "
          f"({len({i[2] for i in items})} distinct photographs)")

    report = {"generated_unix": int(time.time()), "n_images": len(items),
              "n_distinct_photographs": len({i[2] for i in items}), "class_names": CLASS_NAMES}
    if "leakage" in wanted:
        report["leakage"] = audit_leakage(items)
    if "cv" in wanted:
        report["cross_validation"] = cross_validate(items, folds=args.folds, seed=args.seed)
    if "calibration" in wanted:
        report["calibration"] = calibration(items, seed=args.seed)
    if "latency" in wanted:
        report["latency"] = latency(items)
    if "robustness" in wanted:
        report["robustness"] = robustness(items, seed=args.seed)

    os.makedirs(CKPT_DIR, exist_ok=True)
    out = os.path.join(CKPT_DIR, "validation_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
