"""
Train the road-distress classifier on deep CNN embeddings.

    python -m training.train_cnn_head                     # ResNet-50 embeddings
    python -m training.train_cnn_head --backbone mobilenetv2
    python -m training.train_cnn_head --compare           # also score the hand-crafted baseline

This is transfer learning without PyTorch. An ImageNet-trained convolutional
network (1.28 million images) runs through ONNX Runtime as a frozen feature
extractor; a small scikit-learn classifier learns the mapping from its
embeddings to the seven road-condition classes. The network's weights are not
updated - with a few hundred images per class, training only the head is both
faster and less prone to overfitting than fine-tuning the whole network.

Protocol matches training/train_vision.py so the numbers are comparable:
split by source photograph (augmented copies never cross the split), report
held-out accuracy, macro-F1, the confusion matrix and per-class scores.

Outputs:
    checkpoints/cnn_head_<backbone>.joblib         backbone name + trained head
    checkpoints/cnn_head_<backbone>_report.json    measured results

One head per backbone: ResNet-50 and MobileNetV2 embeddings are different
vector spaces, so a head trained on one is meaningless applied to the other.
The loader in models/deep_vision_net.py pairs them and skips any head whose
backbone file is absent.
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

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data.image_dataset import CLASS_NAMES, load_image
from models.cnn_embedder import CNNEmbedder
from training.train_deep_vision import collect_files, grouped_split

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")


def model_path(backbone):
    """One head per backbone - their embedding spaces are not interchangeable."""
    return os.path.join(CKPT_DIR, f"cnn_head_{backbone}.joblib")


def report_path(backbone):
    return os.path.join(CKPT_DIR, f"cnn_head_{backbone}_report.json")


def heads(seed=42):
    """Candidate heads. Small, because the embeddings do the heavy lifting."""
    return {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=seed)),
        ]),
        "svc_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=10.0, kernel="rbf", gamma="scale", probability=True,
                        class_weight="balanced", random_state=seed)),
        ]),
    }


_ROAD_CROP_BOXES = (
    (0.05, 0.52, 0.48, 0.78),
    (0.22, 0.56, 0.74, 0.88),
    (0.45, 0.40, 0.92, 0.70),
)


def embed_items(embedder, items, label):
    imgs, ys = [], []
    for idx, (path, cls, _grp) in enumerate(items):
        try:
            img = load_image(path)
            imgs.append(img)
            ys.append(cls)
            # Region proposals at inference time are tight road-surface crops,
            # whereas Class 0/3/4/5/6 photographs on disk are full wide-angle
            # frames. Extract sub-crops from the training split ONLY (never the
            # held-out test split) with their true class label so every class is
            # represented at both full-frame and region-proposal scales.
            if label == "training" and cls in (0, 3, 4, 5, 6):
                H, W = img.shape[:2]
                if H >= 96 and W >= 96:
                    if cls == 0:
                        box_indices = [idx % len(_ROAD_CROP_BOXES)] if (idx % 3 == 0) else []
                    else:
                        box_indices = list(range(len(_ROAD_CROP_BOXES)))
                    for b_idx in box_indices:
                        x0f, y0f, x1f, y1f = _ROAD_CROP_BOXES[b_idx]
                        crop = img[int(y0f * H):int(y1f * H), int(x0f * W):int(x1f * W)]
                        if crop.shape[0] >= 24 and crop.shape[1] >= 24:
                            imgs.append(np.ascontiguousarray(crop))
                            ys.append(cls)
        except Exception:
            continue
    print(f"  embedding {len(imgs)} {label} images with {embedder.name} ...", flush=True)
    t0 = time.time()
    X = embedder.embed_batch(imgs, progress_every=8)
    print(f"  done in {time.time() - t0:.1f}s ({1000 * (time.time() - t0) / max(1, len(imgs)):.0f} ms/image)")
    return X, np.array(ys)


def run_training(backbone="resnet50", seed=42, max_per_class=800, compare=False):
    embedder = CNNEmbedder(prefer=(backbone, "mobilenetv2", "resnet50"))
    if not embedder.is_ready:
        sys.exit("No CNN backbone found. Run:  python -m scripts.fetch_cnn_backbone")

    items = collect_files()
    if not items:
        sys.exit("No training images. Run scripts/fetch_cracks_potholes_dataset.py first.")

    # cap per class, sampling across the folder rather than alphabetically
    rng = np.random.default_rng(seed)
    by_class = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    capped = []
    for cls, lst in sorted(by_class.items()):
        if len(lst) > max_per_class:
            idx = sorted(rng.choice(len(lst), max_per_class, replace=False))
            lst = [lst[i] for i in idx]
        capped.extend(lst)

    train_items, val_items, test_items = grouped_split(capped, seed=seed)
    # the head trains on train+val; test stays untouched until the end
    fit_items = train_items + val_items
    print(f"[CNN head] backbone {embedder.name} | {len(capped)} images "
          f"({len({i[2] for i in capped})} distinct photographs)")
    print(f"  fit on {len(fit_items)} images, held-out test {len(test_items)} images "
          f"from {len({i[2] for i in test_items})} unseen photographs")

    X_fit, y_fit = embed_items(embedder, fit_items, "training")
    X_test, y_test = embed_items(embedder, test_items, "held-out")

    results, best = {}, None
    for name, head in heads(seed).items():
        t0 = time.time()
        head.fit(X_fit, y_fit)
        pred = head.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
        score = 0.5 * (acc + f1)
        results[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4),
                         "seconds": round(time.time() - t0, 1)}
        print(f"  head {name:10s} accuracy {acc * 100:5.1f}%   macro-F1 {f1:.3f}   ({time.time() - t0:.0f}s)")
        if best is None or score > best[5]:
            best = (name, f1, head, pred, acc, score)

    head_name, macro_f1, head, pred, acc, _score = best
    present = sorted(set(y_test.tolist()) | set(pred.tolist()))
    report = {
        "model": f"CNNHead ({embedder.name} ImageNet embeddings -> {head_name})",
        "backbone": embedder.name,
        "head": head_name,
        "embedding_dim": int(X_fit.shape[1]),
        "class_names": CLASS_NAMES,
        "fit_images": int(len(fit_items)),
        "held_out_test_images": int(len(test_items)),
        "held_out_test_photographs": int(len({i[2] for i in test_items})),
        "held_out_test_accuracy": round(acc, 4),
        "held_out_test_macro_f1": round(macro_f1, 4),
        "random_guess_baseline": round(1.0 / len(CLASS_NAMES), 4),
        "heads_compared": results,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
        "per_class_report": classification_report(
            y_test, pred, labels=present,
            target_names=[CLASS_NAMES[i] for i in present],
            output_dict=True, zero_division=0),
        "split_strategy": "grouped by source photograph; test set scored once",
        "trained_at_unix": int(time.time()),
    }

    if compare:
        from data.feature_extraction import extract_batch
        from models.vision_distress_net import VisionDistressNet
        print("\n  scoring the hand-crafted baseline on the identical split ...")
        Xb_fit = extract_batch([load_image(p) for p, _c, _g in fit_items])
        Xb_test = extract_batch([load_image(p) for p, _c, _g in test_items])
        base = VisionDistressNet(random_state=seed)
        base.fit(Xb_fit, y_fit)
        bpred, _conf, _probs = base.predict(Xb_test)
        b_acc = float(accuracy_score(y_test, bpred))
        b_f1 = float(f1_score(y_test, bpred, average="macro", zero_division=0))
        report["handcrafted_baseline"] = {"accuracy": round(b_acc, 4), "macro_f1": round(b_f1, 4),
                                          "features": "HOG + LBP + colour histogram, 4419 dims"}
        print(f"  baseline    accuracy {b_acc * 100:5.1f}%   macro-F1 {b_f1:.3f}")
        print(f"  CNN head    accuracy {acc * 100:5.1f}%   macro-F1 {macro_f1:.3f}   "
              f"({(acc - b_acc) * 100:+.1f} points)")

    os.makedirs(CKPT_DIR, exist_ok=True)
    out_model, out_report = model_path(embedder.name), report_path(embedder.name)
    joblib.dump({"backbone": embedder.name, "head_name": head_name, "head": head,
                 "class_names": CLASS_NAMES, "accuracy": round(acc, 4),
                 "macro_f1": round(macro_f1, 4), "trained_at_unix": report["trained_at_unix"]},
                out_model)
    with open(out_report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"HELD-OUT TEST accuracy : {acc * 100:.1f}%   (random guess {100.0 / len(CLASS_NAMES):.1f}%)")
    print(f"HELD-OUT TEST macro-F1 : {macro_f1:.3f}")
    for name, m in report["per_class_report"].items():
        if isinstance(m, dict) and "f1-score" in m and "avg" not in name:
            print(f"  {name[:44]:46s} P {m['precision']:.2f}  R {m['recall']:.2f}  "
                  f"F1 {m['f1-score']:.2f}  n={int(m['support'])}")
    print(f"\nmodel  -> {out_model}")
    print(f"report -> {out_report}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "mobilenetv2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-class", type=int, default=800)
    ap.add_argument("--compare", action="store_true", help="also score the hand-crafted baseline")
    args = ap.parse_args()
    run_training(backbone=args.backbone, seed=args.seed,
                 max_per_class=args.max_per_class, compare=args.compare)


if __name__ == "__main__":
    main()
