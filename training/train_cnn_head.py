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
    """Candidate heads evaluated on top of the deep CNN embeddings."""
    from sklearn.neural_network import MLPClassifier

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
        "mlp_deep": Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(512, 256), activation="relu",
                                  alpha=1e-3, max_iter=400, early_stopping=True,
                                  random_state=seed)),
        ]),
    }


_ROAD_CROP_BOXES = (
    (0.05, 0.52, 0.48, 0.78),
    (0.22, 0.56, 0.74, 0.88),
    (0.45, 0.40, 0.92, 0.70),
)


def _load_aligned(items):
    """Load images and their labels together, dropping unreadable files from
    both, so features and labels can never drift out of step."""
    imgs, ys = [], []
    for path, cls, _grp in items:
        try:
            imgs.append(load_image(path))
            ys.append(cls)
        except Exception:
            continue
    return imgs, np.array(ys)


def embed_items(embedder, items, label):
    from models.dan_dag_network import DualAttentionModule
    imgs, ys, pams, domains, groups = [], [], [], [], []
    for idx, (path, cls, _grp) in enumerate(items):
        try:
            img = load_image(path)
            n_before = len(imgs)
            imgs.append(img)
            ys.append(cls)
            pam_vec, _, pam_tel = DualAttentionModule.extract_spatial_attention(img)
            pams.append(pam_vec)
            domains.append(pam_tel["domain_regime"])
            # Region proposals at inference time are tight road-surface crops,
            # whereas full photographs on disk are wide-angle frames. Extract
            # multi-scale sub-crops from the training split ONLY (never the
            # held-out test split) with their true class label so every class is
            # represented at both full-frame and region-proposal scales.
            if label == "training":
                H, W = img.shape[:2]
                if H >= 96 and W >= 96:
                    if cls in (0, 3, 4, 5, 6):
                        if cls == 0:
                            box_indices = [idx % len(_ROAD_CROP_BOXES)] if (idx % 3 == 0) else []
                        else:
                            box_indices = list(range(len(_ROAD_CROP_BOXES)))
                        for b_idx in box_indices:
                            x0f, y0f, x1f, y1f = _ROAD_CROP_BOXES[b_idx]
                            crop = img[int(y0f * H):int(y1f * H), int(x0f * W):int(x1f * W)]
                            if crop.shape[0] >= 24 and crop.shape[1] >= 24:
                                c_contig = np.ascontiguousarray(crop)
                                imgs.append(c_contig)
                                ys.append(cls)
                                cpam, _, ctel = DualAttentionModule.extract_spatial_attention(c_contig)
                                pams.append(cpam)
                                domains.append(ctel["domain_regime"])
                    elif cls in (1, 2) and (idx % 6 == 0):
                        crop = img[int(0.35 * H):int(0.92 * H), int(0.10 * W):int(0.90 * W)]
                        if crop.shape[0] >= 24 and crop.shape[1] >= 24:
                            c_contig = np.ascontiguousarray(crop)
                            imgs.append(c_contig)
                            ys.append(cls)
                            cpam, _, ctel = DualAttentionModule.extract_spatial_attention(c_contig)
                            pams.append(cpam)
                            domains.append(ctel["domain_regime"])
            # every row - the photograph and any sub-crop of it - carries the
            # photograph's group, so cross-validation can never split them
            groups.extend([_grp] * (len(imgs) - n_before))
        except Exception:
            del imgs[n_before:], ys[n_before:], pams[n_before:], domains[n_before:]
            continue
    print(f"  embedding {len(imgs)} {label} images with {embedder.name} ...", flush=True)
    t0 = time.time()
    X = embedder.embed_batch(imgs, progress_every=12)
    # Mirror-image embeddings, for flip test-time augmentation. A road photo
    # and its mirror show the same defect; averaging the two vectors removes
    # some of the backbone's left/right bias.
    Xf = embedder.embed_batch([np.ascontiguousarray(im[:, ::-1]) for im in imgs], progress_every=0)
    print(f"  done in {time.time() - t0:.1f}s ({1000 * (time.time() - t0) / max(1, 2 * len(imgs)):.0f} ms/embedding)", flush=True)
    return (X, np.array(ys), np.asarray(pams, dtype=np.float32), np.array(domains),
            np.asarray(Xf), np.array(groups))


CANDIDATES = {
    # name: (family, params)
    "logistic_C0.3": ("logistic", {"C": 0.3}),
    "logistic_C1": ("logistic", {"C": 1.0}),
    "svc_C3": ("svc", {"C": 3.0}),
    "svc_C10": ("svc", {"C": 10.0}),
    "svc_C30": ("svc", {"C": 30.0}),
    "mlp_512_256": ("mlp", {}),
    "ensemble_soft": ("ensemble", {}),
}


def build_candidate(name, seed=42, final=False, svc_C=10.0, logistic_C=1.0):
    """A fresh, unfitted head. SVC probabilities are only computed for the final
    model or inside the ensemble - Platt scaling costs five extra fits and plain
    predict() is all cross-validation needs."""
    from sklearn.ensemble import VotingClassifier
    from sklearn.neural_network import MLPClassifier
    family, p = CANDIDATES[name]

    def logistic(C):
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(
            C=C, max_iter=3000, class_weight="balanced", random_state=seed))])

    def svc(C, proba):
        return Pipeline([("scale", StandardScaler()), ("clf", SVC(
            C=C, kernel="rbf", gamma="scale", probability=proba,
            class_weight="balanced", random_state=seed))])

    def mlp():
        return Pipeline([("scale", StandardScaler()), ("clf", MLPClassifier(
            hidden_layer_sizes=(512, 256), activation="relu", alpha=1e-3, max_iter=400,
            early_stopping=True, random_state=seed))])

    if family == "logistic":
        return logistic(p["C"])
    if family == "svc":
        return svc(p["C"], final)
    if family == "mlp":
        return mlp()
    # soft vote of the three families at their CV-best settings
    return VotingClassifier([("svc", svc(svc_C, True)), ("lr", logistic(logistic_C)), ("mlp", mlp())],
                            voting="soft")


def cross_validate_candidates(features, y, groups, seed=42, folds=5):
    from sklearn.model_selection import StratifiedGroupKFold
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(features["plain"][0], y, groups))
    print(f"\n  cross-validating {len(CANDIDATES)} heads x {len(features)} feature sets, "
          f"{folds} folds grouped by photograph ({len(set(groups.tolist()))} groups)", flush=True)
    results, ranked = {}, []
    best_params = {}
    for feat_name, (X, _Xte) in features.items():
        for name in CANDIDATES:
            family = CANDIDATES[name][0]
            if family == "ensemble":
                continue
            accs, f1s, t0 = [], [], time.time()
            for tr, va in splits:
                m = build_candidate(name, seed)
                m.fit(X[tr], y[tr])
                pv = m.predict(X[va])
                accs.append(accuracy_score(y[va], pv))
                f1s.append(f1_score(y[va], pv, average="macro", zero_division=0))
            r = {"features": feat_name, "cv_accuracy": round(float(np.mean(accs)), 4),
                 "cv_accuracy_std": round(float(np.std(accs)), 4),
                 "cv_macro_f1": round(float(np.mean(f1s)), 4),
                 "cv_macro_f1_std": round(float(np.std(f1s)), 4),
                 "seconds": round(time.time() - t0, 1)}
            r["score"] = round(0.5 * (r["cv_accuracy"] + r["cv_macro_f1"]), 4)
            results[f"{name}|{feat_name}"] = r
            ranked.append((r["score"], name, feat_name))
            prev = best_params.get((family, feat_name))
            if prev is None or r["score"] > prev[0]:
                best_params[(family, feat_name)] = (r["score"], CANDIDATES[name][1])
            print(f"    {name:14s} {feat_name:9s} acc {r['cv_accuracy'] * 100:5.1f}% "
                  f"\u00b1{r['cv_accuracy_std'] * 100:4.1f}  macro-F1 {r['cv_macro_f1']:.3f} "
                  f"\u00b1{r['cv_macro_f1_std']:.3f}  ({r['seconds']:.0f}s)", flush=True)

    # ensemble on the better feature set, members at their CV-best settings
    top_feat = max(ranked)[2]
    svc_C = best_params[("svc", top_feat)][1]["C"]
    lr_C = best_params[("logistic", top_feat)][1]["C"]
    X = features[top_feat][0]
    accs, f1s, t0 = [], [], time.time()
    for tr, va in splits:
        m = build_candidate("ensemble_soft", seed, svc_C=svc_C, logistic_C=lr_C)
        m.fit(X[tr], y[tr])
        pv = m.predict(X[va])
        accs.append(accuracy_score(y[va], pv))
        f1s.append(f1_score(y[va], pv, average="macro", zero_division=0))
    r = {"features": top_feat, "members": {"svc_C": svc_C, "logistic_C": lr_C, "mlp": "512-256"},
         "cv_accuracy": round(float(np.mean(accs)), 4), "cv_accuracy_std": round(float(np.std(accs)), 4),
         "cv_macro_f1": round(float(np.mean(f1s)), 4), "cv_macro_f1_std": round(float(np.std(f1s)), 4),
         "seconds": round(time.time() - t0, 1)}
    r["score"] = round(0.5 * (r["cv_accuracy"] + r["cv_macro_f1"]), 4)
    results[f"ensemble_soft|{top_feat}"] = r
    ranked.append((r["score"], "ensemble_soft", top_feat))
    print(f"    {'ensemble_soft':14s} {top_feat:9s} acc {r['cv_accuracy'] * 100:5.1f}% "
          f"\u00b1{r['cv_accuracy_std'] * 100:4.1f}  macro-F1 {r['cv_macro_f1']:.3f} "
          f"\u00b1{r['cv_macro_f1_std']:.3f}  ({r['seconds']:.0f}s)", flush=True)

    score, name, feat = max(ranked)
    best = dict(results[f"{name}|{feat}"], name=name)
    if name == "ensemble_soft":
        CANDIDATES["ensemble_soft"] = ("ensemble", {})
        best["members"] = r["members"]
        # bind the member settings for the final refit
        global _ENSEMBLE_MEMBERS
        _ENSEMBLE_MEMBERS = (svc_C, lr_C)
    return results, best


_ENSEMBLE_MEMBERS = (10.0, 1.0)


def run_training(backbone="resnet50", seed=42, max_per_class=1500, compare=False):
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
          f"({len({i[2] for i in capped})} distinct photographs)", flush=True)
    print(f"  fit on {len(fit_items)} images, held-out test {len(test_items)} images "
          f"from {len({i[2] for i in test_items})} unseen photographs", flush=True)

    scratch_dir = os.path.join(ENGINE_ROOT, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    cache_file = os.path.join(scratch_dir, f"embed_cache_v2_{embedder.name}_{max_per_class}_{len(fit_items)}.npz")
    if os.path.exists(cache_file):
        print(f"  loading cached embeddings from {cache_file} ...", flush=True)
        c = np.load(cache_file, allow_pickle=False)
        X_fit, y_fit, Xf_fit, G_fit = c["X_fit"], c["y_fit"], c["Xf_fit"], c["G_fit"]
        X_test, y_test, Xf_test = c["X_test"], c["y_test"], c["Xf_test"]
    else:
        X_fit, y_fit, _P, _D, Xf_fit, G_fit = embed_items(embedder, fit_items, "training")
        X_test, y_test, _P, _D, Xf_test, _G = embed_items(embedder, test_items, "held-out")
        np.savez_compressed(cache_file, X_fit=X_fit.astype(np.float32), y_fit=y_fit,
                            Xf_fit=Xf_fit.astype(np.float32), G_fit=G_fit,
                            X_test=X_test.astype(np.float32), y_test=y_test,
                            Xf_test=Xf_test.astype(np.float32))

    features = {
        "plain": (X_fit, X_test),
        "flip_tta": (0.5 * (X_fit + Xf_fit), 0.5 * (X_test + Xf_test)),
    }

    # ---- model selection: grouped CV on the FIT split only -------------------
    # The previous version fitted three heads and kept whichever scored best on
    # the test set, which makes the reported test score an optimistic one. Here
    # every candidate is scored by 5-fold cross-validation grouped by source
    # photograph inside the fit split; the test set is touched exactly once, by
    # the single model that selection chose.
    cv_results, best = cross_validate_candidates(features, y_fit, G_fit, seed)
    chosen = best["name"]
    feat_name = best["features"]
    Xtr, Xte = features[feat_name]
    print(f"\n  selected by cross-validation: {chosen} on {feat_name} features "
          f"(CV accuracy {best['cv_accuracy'] * 100:.1f}%, macro-F1 {best['cv_macro_f1']:.3f})", flush=True)
    t0 = time.time()
    head = build_candidate(chosen, seed, final=True,
                           svc_C=_ENSEMBLE_MEMBERS[0], logistic_C=_ENSEMBLE_MEMBERS[1])
    head.fit(Xtr, y_fit)
    pred = head.predict(Xte)
    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    print(f"  refit on the full fit split in {time.time() - t0:.0f}s; scoring the test set once", flush=True)
    results = cv_results
    best = (chosen, f1, head, pred, acc, 0.5 * (acc + f1))
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
        "selection": "5-fold StratifiedGroupKFold on the fit split, grouped by source "
                     "photograph; score = mean(accuracy, macro-F1); test set scored once",
        "features": feat_name,
        "tta_flip": feat_name == "flip_tta",
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
        # embed_items() augments the TRAINING split with region-scale sub-crops,
        # so y_fit is longer than fit_items. The baseline sees full photographs
        # only, so its labels must come from the items themselves - reusing
        # y_fit here misaligned features and labels and crashed every --compare
        # run before the model was saved.
        imgs_fit, yb_fit = _load_aligned(fit_items)
        imgs_test, yb_test = _load_aligned(test_items)
        Xb_fit = extract_batch(imgs_fit)
        Xb_test = extract_batch(imgs_test)
        base = VisionDistressNet(random_state=seed)
        base.fit(Xb_fit, yb_fit)
        bpred, _conf, _probs = base.predict(Xb_test)
        b_acc = float(accuracy_score(yb_test, bpred))
        b_f1 = float(f1_score(yb_test, bpred, average="macro", zero_division=0))
        report["handcrafted_baseline"] = {"accuracy": round(b_acc, 4), "macro_f1": round(b_f1, 4),
                                          "features": "HOG + LBP + colour histogram, 4419 dims"}
        print(f"  baseline    accuracy {b_acc * 100:5.1f}%   macro-F1 {b_f1:.3f}")
        print(f"  CNN head    accuracy {acc * 100:5.1f}%   macro-F1 {macro_f1:.3f}   "
              f"({(acc - b_acc) * 100:+.1f} points)")

    os.makedirs(CKPT_DIR, exist_ok=True)
    out_model, out_report = model_path(embedder.name), report_path(embedder.name)
    joblib.dump({"backbone": embedder.name, "head_name": head_name, "head": head,
                 "tta_flip": feat_name == "flip_tta",
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
    ap.add_argument("--max-per-class", type=int, default=1500)
    ap.add_argument("--compare", action="store_true", help="also score the hand-crafted baseline")
    args = ap.parse_args()
    run_training(backbone=args.backbone, seed=args.seed,
                 max_per_class=args.max_per_class, compare=args.compare)


if __name__ == "__main__":
    main()
