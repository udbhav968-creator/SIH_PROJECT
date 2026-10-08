"""
Train and measure the input guard (models/ood_guard.py), and write the monitoring reference.

    git clone --depth 1 https://github.com/EliSchwartz/imagenet-sample-images ../imagenet-sample-images
    python -m training.train_ood_guard --ood-dir ../imagenet-sample-images

Data
  road photographs  every datasets/*/real_images photograph that the corpus policy counts as a road
                    scene, one per source photograph (augmented copies share a source and are
                    dropped so they cannot sit on both sides of a split); split 60/20/20 by a hash
                    of the source name into train / calibration / test
  everyday photos   the ImageNet sample set, one photograph per class, minus about 50 classes whose
                    sample shows a road, a vehicle or street furniture (those are valid inputs);
                    split 50/50 by hash into train (for the not-road classifier) / test
  close-ups         the surface-crack texture patches the corpus policy excludes: reported
                    separately, as what happens when someone photographs a crack from 20 cm

Thresholds are set on the calibration road photographs only (novelty at the 99th percentile, so
about 1 in 100 genuine road photographs is called unusual), then everything is scored once on the
test sets. Synthetic darkening, blur and overexposure of the test road photographs check the
quality limits.

Outputs: checkpoints/ood_guard.npz, checkpoints/ood_guard_report.json,
         checkpoints/monitoring_reference.json (the distributions production is compared against)
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.ood_guard import (BACKBONE, MODEL_FILE, REPORT_FILE, QUALITY_KEYS, OODGuard, image_quality,  # noqa: E402
                              _sha16)
from pipeline.corpus_policy import non_road_source, policy_record  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints")
EXTS = (".jpg", ".jpeg", ".png")
ROAD_LIKE_IMAGENET = {
    "ambulance", "amphibian", "beach_wagon", "bicycle-built-for-two", "cab", "car_wheel", "car_mirror",
    "convertible", "disk_brake", "fire_engine", "forklift", "garbage_truck", "go-kart", "golfcart", "grille",
    "half_track", "horse_cart", "jeep", "jinrikisha", "limousine", "manhole_cover", "minibus", "minivan",
    "Model_T", "moped", "motor_scooter", "mountain_bike", "moving_van", "oxcart", "parking_meter", "pickup",
    "police_van", "racer", "school_bus", "snowplow", "plow", "sports_car", "streetcar", "tank", "tow_truck",
    "tractor", "trailer_truck", "tricycle", "trolleybus", "unicycle", "street_sign", "traffic_light",
    "barrow", "steel_arch_bridge", "suspension_bridge", "viaduct", "crash_helmet", "recreational_vehicle",
    "motor_home", "worm_fence", "chain_mail", "shopping_cart", "thresher", "harvester",
}


def _bucket(key, salt="road-shield-ood"):
    return int(hashlib.sha256((salt + key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def source_key(path):
    name = os.path.splitext(os.path.basename(path))[0].lower()
    name = re.sub(r"^aug_mega_\d+_", "", name)
    return re.sub(r"^(wm_\d+_)", "", name)


def road_photographs():
    seen, roads, closeups = {}, [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "datasets", "*", "real_images", "*"))):
        if not p.lower().endswith(EXTS) or "_label_conflicts" in p:
            continue
        if non_road_source(p):
            closeups.append(p)
            continue
        k = source_key(p)
        if k in seen:
            continue
        seen[k] = p
        roads.append(p)
    return roads, closeups


def split_roads(paths):
    tr, ca, te = [], [], []
    for p in paths:
        b = _bucket(source_key(p))
        (tr if b < 0.6 else ca if b < 0.8 else te).append(p)
    return tr, ca, te


def everyday_photographs(ood_dir):
    out, dropped = [], 0
    for p in sorted(glob.glob(os.path.join(ood_dir, "*"))):
        if not p.lower().endswith(EXTS):
            continue
        cls = os.path.splitext(os.path.basename(p))[0].split("_", 1)[-1]
        if cls in ROAD_LIKE_IMAGENET:
            dropped += 1
            continue
        out.append(p)
    tr = [p for p in out if _bucket(os.path.basename(p), "ood-split") < 0.5]
    te = [p for p in out if p not in set(tr)]
    return tr, te, dropped


def load_rgb(path):
    """Decoded exactly as the server decodes an upload (models/cv_cavity_detector.decode_image: 640x480),
    so the thresholds and the monitoring reference describe what production actually sees."""
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((640, 480), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def auroc(pos, neg):
    """P(score of a positive > score of a negative); ties count half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fpr_at_tpr(pos, neg, tpr=0.95):
    thr = np.quantile(np.asarray(pos, float), 1 - tpr)
    return float((np.asarray(neg, float) >= thr).mean())


def histogram_reference(values, bins=10):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1))[1:-1])
    counts = np.histogram(v, bins=np.concatenate([[-np.inf], edges, [np.inf]]))[0]
    return {"edges": [round(float(e), 6) for e in edges], "probs": [round(float(c) / len(v), 6) for c in counts],
            "n": int(len(v)), "mean": round(float(v.mean()), 4), "p05": round(float(np.quantile(v, .05)), 4),
            "p95": round(float(np.quantile(v, .95)), 4)}


def corrupt(img, kind):
    import cv2
    x = img.astype(np.float32)
    if kind == "dark":
        return np.clip(x * 0.12, 0, 255).astype(np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=max(img.shape[:2]) / 120)
    if kind == "overexposed":
        return np.clip(x * 2.8 + 60, 0, 255).astype(np.uint8)
    raise ValueError(kind)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ood-dir", required=True, help="folder of everyday photographs (ImageNet sample set)")
    ap.add_argument("--pca-dims", type=int, default=64)
    ap.add_argument("--novelty-quantile", type=float, default=0.99)
    ap.add_argument("--max-road", type=int, default=4000)
    ap.add_argument("--out", default=CKPT)
    a = ap.parse_args(argv)

    from sklearn.covariance import LedoitWolf
    from sklearn.linear_model import LogisticRegression
    from models.cnn_embedder import CNNEmbedder
    from mlops.tracking import start_run

    t0 = time.time()
    emb = CNNEmbedder(checkpoints_dir=CKPT, prefer=(BACKBONE,))
    if not emb.is_ready or emb.name != BACKBONE:
        sys.exit(f"needs checkpoints/cnn_backbone_{BACKBONE}.onnx (scripts/fetch_cnn_backbone.py)")

    roads, closeups = road_photographs()
    roads = roads[: a.max_road]
    r_tr, r_ca, r_te = split_roads(roads)
    o_tr, o_te, o_dropped = everyday_photographs(a.ood_dir)
    if len(o_tr) < 50:
        sys.exit(f"only {len(o_tr)} everyday photographs found in {a.ood_dir}")
    print(f"road photographs: train {len(r_tr)} / calibration {len(r_ca)} / test {len(r_te)}; "
          f"everyday: train {len(o_tr)} / test {len(o_te)} ({o_dropped} road-like classes left out); "
          f"close-ups {len(closeups)}", flush=True)

    params = {"backbone": BACKBONE, "pca_dims": a.pca_dims, "novelty_quantile": a.novelty_quantile,
              "road_train": len(r_tr), "road_calibration": len(r_ca), "road_test": len(r_te),
              "everyday_train": len(o_tr), "everyday_test": len(o_te), "ood_dir": os.path.basename(a.ood_dir.rstrip("/\\"))}
    with start_run("ood_guard", params=params, tags={"script": "training/train_ood_guard.py"}) as run:
        def embed_and_quality(paths, label):
            E, Q = [], []
            for i, p in enumerate(paths):
                img = load_rgb(p)
                E.append(emb.embed(img))
                Q.append(image_quality(img))
                if (i + 1) % 400 == 0:
                    print(f"  {label}: {i + 1}/{len(paths)}", flush=True)
            return np.asarray(E, np.float64), Q

        E_tr, Q_tr = embed_and_quality(r_tr, "road_train")
        E_ca, Q_ca = embed_and_quality(r_ca, "road_calibration")
        E_te, Q_te = embed_and_quality(r_te, "road_test")
        O_tr, _ = embed_and_quality(o_tr, "everyday_train")
        O_te, _ = embed_and_quality(o_te, "everyday_test")
        C_x, _ = embed_and_quality(closeups[:300], "close-ups") if closeups else (np.zeros((0, 1000)), [])

        def l2(x):
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

        X = l2(E_tr)
        mean = X.mean(axis=0)
        _u, _s, vt = np.linalg.svd(X - mean, full_matrices=False)
        comps = vt[: a.pca_dims]
        proj = lambda E: (l2(E) - mean) @ comps.T  # noqa: E731
        Z_tr = proj(E_tr)
        lw = LedoitWolf().fit(Z_tr)

        Zlr = np.concatenate([Z_tr, proj(O_tr)])
        ylr = np.concatenate([np.zeros(len(Z_tr)), np.ones(len(O_tr))])
        lr_mean, lr_scale = Zlr.mean(axis=0), Zlr.std(axis=0) + 1e-9
        lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000).fit((Zlr - lr_mean) / lr_scale, ylr)

        meta = {"version": time.strftime("%Y%m%d-%H%M"), "backbone": BACKBONE,
                "backbone_sha16": _sha16(os.path.join(CKPT, f"cnn_backbone_{BACKBONE}.onnx")),
                "pca_dims": a.pca_dims, "thresholds": {}}
        arrays = {"pca_mean": mean, "pca_components": comps, "maha_mean": lw.location_,
                  "maha_precision": lw.precision_, "lr_mean": lr_mean, "lr_scale": lr_scale,
                  "lr_coef": lr.coef_.ravel(), "lr_intercept": np.asarray(lr.intercept_[0])}

        class _E:  # scores from the arrays exactly as served
            is_ready, name = True, BACKBONE
        g = OODGuard.__new__(OODGuard)
        g.params, g.embedder = arrays, _E()

        maha_ca, p_ca = g.scores(E_ca)
        q = lambda key, fn, qq: float(np.quantile([fn(x[key]) for x in Q_ca], qq))  # noqa: E731
        thresholds = {
            "maha": float(np.quantile(maha_ca, a.novelty_quantile)),
            "p_not_road": float(max(0.5, np.quantile(p_ca, 0.99))),
            "brightness_min": min(q("brightness", float, 0.005) * 0.8, 40.0),
            "contrast_min": min(q("contrast", float, 0.005) * 0.8, 18.0),
            "sharpness_min": min(q("sharpness", float, 0.005) * 0.6, 40.0),
            "clipped_max": max(q("clipped_fraction", float, 0.995) * 1.2, 0.25),
        }
        meta["thresholds"] = {k: round(v, 6) for k, v in thresholds.items()}
        g.thresholds, g.meta = meta["thresholds"], meta

        def flagged(E, Q=None):
            m, p = g.scores(E)
            sem = (m > thresholds["maha"]) | (p > thresholds["p_not_road"])
            if Q is None:
                return m, p, sem
            qual = np.asarray([bool(g.quality_flags(x)) for x in Q])
            return m, p, sem | qual

        m_te, p_te, f_te = flagged(E_te, Q_te)
        v_te = [OODGuard.verdict_of(g.flags_for(Q_te[i], m_te[i], p_te[i])) for i in range(len(Q_te))]
        m_ot, p_ot, f_ot = flagged(O_te)
        v_ot = ["not_road" if p > thresholds["p_not_road"] else "unusual" if m > thresholds["maha"] else "ok"
                for m, p in zip(m_ot, p_ot)]
        m_cl, p_cl, f_cl = flagged(C_x) if len(C_x) else (np.zeros(0), np.zeros(0), np.zeros(0, bool))
        comb_te = np.maximum(m_te / thresholds["maha"], p_te / thresholds["p_not_road"])
        comb_ot = np.maximum(m_ot / thresholds["maha"], p_ot / thresholds["p_not_road"])

        corruptions = {}
        sample = r_te[:150]
        for kind in ("dark", "blur", "overexposed"):
            caught = 0
            for pth in sample:
                img = corrupt(load_rgb(pth), kind)
                qq = image_quality(img)
                m1, p1 = g.scores(emb.embed(img)[None, :])
                if g.quality_flags(qq) or m1[0] > thresholds["maha"] or p1[0] > thresholds["p_not_road"]:
                    caught += 1
            corruptions[kind] = {"photos": len(sample), "flagged_rate": round(caught / max(1, len(sample)), 4)}

        report = {
            "model": "OOD guard: Mahalanobis novelty + not-road logistic regression on MobileNetV2 embeddings (PCA 64)",
            "version": meta["version"],
            "trained_unix": int(time.time()),
            "thresholds": meta["thresholds"],
            "threshold_rule": f"novelty at the {a.novelty_quantile:.0%} quantile of calibration road photographs; "
                              "not-road at max(0.5, 99th percentile of calibration roads); quality limits from the "
                              "0.5% tails of calibration roads, loosened, with fixed floors",
            "data": {**params, "everyday_classes_left_out_as_road_like": o_dropped,
                     "corpus_policy": policy_record()},
            "test": {
                "novelty_only": {"auroc": round(auroc(m_ot, m_te), 4), "fpr_at_95_tpr": round(fpr_at_tpr(m_ot, m_te), 4),
                                 "note": "never saw an everyday photograph in training"},
                "not_road_classifier": {"auroc": round(auroc(p_ot, p_te), 4)},
                "combined": {"auroc": round(auroc(comb_ot, comb_te), 4),
                             "ood_caught_rate": round(float(f_ot.mean()), 4),
                             "in_distribution_flagged_rate": round(float(f_te.mean()), 4),
                             "road_test_photos": len(r_te), "everyday_test_photos": len(o_te)},
                "refusals": {
                    "note": "what the citizen endpoint turns away (verdict not_road or poor_quality); "
                            "'unusual' is only a warning",
                    "road_photos_refused_rate": round(sum(v in ("not_road", "poor_quality") for v in v_te) / len(v_te), 4),
                    "road_photos_warned_rate": round(sum(v == "unusual" for v in v_te) / len(v_te), 4),
                    "everyday_photos_refused_rate": round(sum(v == "not_road" for v in v_ot) / len(v_ot), 4),
                    "road_verdicts": {v: v_te.count(v) for v in sorted(set(v_te))},
                },
                "closeup_textures_flagged_rate": round(float(f_cl.mean()), 4) if len(f_cl) else None,
                "synthetic_corruptions": corruptions,
            },
            "seconds": round(time.time() - t0, 1),
        }

        os.makedirs(a.out, exist_ok=True)
        np.savez(os.path.join(a.out, MODEL_FILE), meta_json=np.asarray(json.dumps(meta)), **arrays)
        with open(os.path.join(a.out, REPORT_FILE), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        ref = build_monitoring_reference(r_ca + r_te, E_ca, E_te, Q_ca + Q_te, g)
        with open(os.path.join(a.out, "monitoring_reference.json"), "w", encoding="utf-8") as fh:
            json.dump(ref, fh, indent=2)

        run.log_metrics({"auroc_combined": report["test"]["combined"]["auroc"],
                         "auroc_novelty": report["test"]["novelty_only"]["auroc"],
                         "fpr95_novelty": report["test"]["novelty_only"]["fpr_at_95_tpr"],
                         "ood_caught_rate": report["test"]["combined"]["ood_caught_rate"],
                         "road_flagged_rate": report["test"]["combined"]["in_distribution_flagged_rate"],
                         "road_refused_rate": report["test"]["refusals"]["road_photos_refused_rate"],
                         "everyday_refused_rate": report["test"]["refusals"]["everyday_photos_refused_rate"],
                         **{f"corruption_{k}_flagged": v["flagged_rate"] for k, v in corruptions.items()}})
        for f in (MODEL_FILE, REPORT_FILE, "monitoring_reference.json"):
            run.log_artifact(os.path.join(a.out, f))
        print(json.dumps(report["test"], indent=2))
        print(f"run {run.run_id}")
    return report


def build_monitoring_reference(paths, E_ca, E_te, Q, guard):
    """What production inputs and predictions looked like at training time, as histograms."""
    from models.deep_vision_net import load_best_vision_model
    model, backend = load_best_vision_model(CKPT, verbose=False)
    E = np.concatenate([E_ca, E_te])
    maha, _ = guard.scores(E)
    conf, mix = [], {}
    names = list(getattr(model, "class_names", None) or [])
    if backend in ("deep_cnn", "cnn_embeddings"):
        for p in paths:
            pr = np.asarray(model.predict_probabilities(load_rgb(p))).ravel()
            k = int(pr.argmax())
            conf.append(float(pr[k]))
            nm = names[k] if k < len(names) else str(k)
            mix[nm] = mix.get(nm, 0) + 1
    feats = {k: histogram_reference([x[k] for x in Q]) for k in QUALITY_KEYS if k != "sharpness"}
    feats["sharpness_log10"] = histogram_reference([np.log10(1 + x["sharpness"]) for x in Q])
    feats["novelty_score"] = histogram_reference(maha)
    if conf:
        feats["top_confidence"] = histogram_reference(conf)
    total = sum(mix.values()) or 1
    return {
        "built_unix": int(time.time()),
        "built_from": f"{len(paths)} held-out road photographs (calibration + test splits of train_ood_guard)",
        "classifier_backend": backend,
        "features": feats,
        "class_mix": {k: round(v / total, 4) for k, v in sorted(mix.items())},
        "class_mix_n": int(sum(mix.values())),
        "note": "Production windows are compared with these by PSI (mlops/monitor.py). The reference is "
                "public road datasets, so a city's own fleet will differ somewhat from it on day one; "
                "rebuild it from the first weeks of fleet data once that exists.",
    }


if __name__ == "__main__":
    main()
