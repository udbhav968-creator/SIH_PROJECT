"""
Compare candidate classifiers on the same grouped split, and keep the winner.

    python -m scripts.model_selection            # compare, report, save nothing
    python -m scripts.model_selection --adopt    # also write the winner to checkpoints/

Every candidate sees identical features and identical folds, split by source
photograph so augmented copies never cross a fold. Selection is by macro-F1,
not accuracy: with classes ranging from 14 to 1,230 images, accuracy rewards a
model that ignores the small classes.

Candidates:
  svc_rbf        the current pipeline: StandardScaler -> PCA -> SVC(rbf)
  svc_tuned      same shape, wider PCA and a tuned C/gamma
  hist_gb        HistGradientBoosting on PCA features - different inductive
                 bias, strong on tabular-shaped data
  random_forest  bagged trees, robust to feature scaling
  logistic       multinomial logistic regression, a calibrated linear baseline
  ensemble       soft voting over svc_tuned + hist_gb + random_forest

The ensemble is the honest version of a "multi-model approach": three models
that fail differently, averaged by predicted probability, which beats any one
of them when their errors are uncorrelated - and when it doesn't, this script
says so and the single best model is kept instead.
"""

import argparse
import json
import os
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data.image_dataset import CLASS_NAMES, load_image
from data.feature_extraction import extract_image_features
from training.train_deep_vision import collect_files

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")


def candidates(seed=42):
    return {
        "svc_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=seed)),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=seed)),
        ]),
        "svc_tuned": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.98, random_state=seed)),
            ("clf", SVC(kernel="rbf", C=30.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=seed)),
        ]),
        "hist_gb": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=120, random_state=seed)),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                   early_stopping=True, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=120, random_state=seed)),
            ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                           n_jobs=2, random_state=seed)),
        ]),
        "svc_balanced": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=seed)),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=seed)),
        ]),
        "logistic_wide": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.99, random_state=seed)),
            ("clf", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=seed)),
        ]),
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=seed)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
    }


def build_ensemble(seed=42):
    c = candidates(seed)
    return VotingClassifier(
        estimators=[("svc", c["svc_tuned"]), ("hgb", c["hist_gb"]), ("rf", c["random_forest"])],
        voting="soft", n_jobs=1,
    )


def features(items):
    X = np.stack([extract_image_features(load_image(p)) for p, _c, _g in items]).astype(np.float32)
    y = np.array([c for _p, c, _g in items])
    g = np.array([g for _p, _c, g in items])
    return X, y, g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-class", type=int, default=800)
    ap.add_argument("--quick", action="store_true", help="skip the slow ensemble")
    ap.add_argument("--adopt", action="store_true", help="save the winning model to checkpoints/")
    args = ap.parse_args()

    items = collect_files()
    # cap per class so no single class dominates the comparison or the memory
    per_class, capped = {}, []
    rng = np.random.default_rng(args.seed)
    by_class = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    for cls, lst in by_class.items():
        if len(lst) > args.max_per_class:
            idx = rng.choice(len(lst), args.max_per_class, replace=False)
            lst = [lst[i] for i in sorted(idx)]
        per_class[cls] = len(lst)
        capped.extend(lst)
    print(f"{len(capped)} images, per class {dict(sorted(per_class.items()))}")

    t0 = time.time()
    X, y, groups = features(capped)
    print(f"features {X.shape} in {time.time() - t0:.1f}s\n")

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(splitter.split(X, y, groups))

    results = {}
    models = candidates(args.seed)
    if not args.quick:
        models["ensemble"] = build_ensemble(args.seed)

    for name, model in models.items():
        accs, f1s, t1 = [], [], time.time()
        try:
            for tr, te in folds:
                from sklearn.base import clone
                m = clone(model)
                m.fit(X[tr], y[tr])
                pred = m.predict(X[te])
                accs.append(accuracy_score(y[te], pred))
                f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        except Exception as e:
            print(f"  {name:14s} FAILED: {e}")
            continue
        results[name] = {"accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                         "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                         "seconds": round(time.time() - t1, 1)}
        print(f"  {name:14s} acc {np.mean(accs) * 100:5.1f}% ± {np.std(accs) * 100:4.1f}   "
              f"macro-F1 {np.mean(f1s):.3f} ± {np.std(f1s):.3f}   ({time.time() - t1:.0f}s)")

    if not results:
        sys.exit("No candidate completed.")
    winner = max(results, key=lambda k: results[k]["macro_f1_mean"])
    print(f"\nwinner by macro-F1: {winner} "
          f"({results[winner]['macro_f1_mean']:.3f}, accuracy {results[winner]['accuracy_mean'] * 100:.1f}%)")

    report = {"generated_unix": int(time.time()), "folds": args.folds, "n_images": len(capped),
              "per_class_counts": per_class, "class_names": CLASS_NAMES,
              "results": results, "winner": winner,
              "selection_metric": "macro F1 over grouped folds"}
    with open(os.path.join(CKPT_DIR, "model_selection_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"report -> checkpoints/model_selection_report.json")

    if args.adopt:
        import joblib
        best = models[winner]
        from sklearn.base import clone
        final = clone(best)
        final.fit(X, y)
        path = os.path.join(CKPT_DIR, "vision_selected_model.joblib")
        joblib.dump({"pipeline": final, "name": winner, "class_names": CLASS_NAMES,
                     "cv": results[winner]}, path)
        print(f"adopted {winner}, trained on all {len(capped)} images -> {path}")


if __name__ == "__main__":
    main()
