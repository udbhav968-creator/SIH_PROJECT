"""
Train and measure the crack verifier (models/crack_verifier.py). CPU, a few minutes.

    python -m scripts.fetch_seg_datasets --only deepcrack crackforest     # the crack images, by git clone
    python -m training.train_crack_verifier

Windows (crops) and where they come from
  crack        DeepCrack + CrackForest images (datasets/seg_multi/<source>, 512x320), windows of 80-220 px
               centred on mask pixels, kept when at least 1.5% of the window is crack; and the same images
               turned into a dashcam's view (tilted, shrunk, blurred, JPEG-compressed), because both crack
               sets are photographed straight down. Added after a first version, trained on the top-down
               windows only, rejected real dashcam cracks in a 10-photograph smoke run of the gate check
  pavement     windows of the same images with no crack pixel at all
  road scene   windows from the lower 65% of clean-road photographs (sound pavement, zebra crossings,
               dividers): paint edges and seams are what the gate has to learn to let go
Splits
  crack images: the train / cal / test split already recorded in datasets/seg_multi/manifest.json
  road photographs: 60/20/20 by a hash of the source photograph (augmented copies share it)
  Excluded from every split: the photographs scripts/measure_detection_quality.py samples to decide whether
  the gate is served (40 per folder, seed 7), and anything perceptually near one of them.
Model
  logistic regression on the L2-normalised MobileNetV2 embedding, class-balanced, L2 penalty chosen on the
  calibration windows (log-loss). Threshold: the highest one that still keeps 97% of the calibration crack
  windows, so the gate is built to give up false alarms, not cracks.
Output
  checkpoints/crack_verifier.npz, checkpoints/crack_verifier_report.json; not served until
  scripts/select_crack_gate.py has measured the whole pipeline with it (checkpoints/crack_verifier_selection.json)
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.crack_verifier import BACKBONE, MODEL_FILE, REPORT_FILE  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints")
SEG = os.path.join(ROOT, "datasets", "seg_multi")
# DeepCrack and CrackForest arrive by git; CrackSeg9k (which contains both, so fetch_seg_datasets keeps only one
# copy of each image) where Hugging Face is reachable. At most MAX_PER_SOURCE images each, so no source dominates.
CRACK_SOURCES = ("deepcrack", "crackforest", "crackseg9k")
MAX_PER_SOURCE = 700
ROAD_FOLDERS = ["05_morth_civil_hard_negatives", "10_missing_zebra_crossing", "11_missing_road_divider"]
TARGET_RECALL = 0.97
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)


def _bucket(key, salt="road-shield-crack-verifier"):
    return int(hashlib.sha256((salt + key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def source_key(path):
    name = os.path.splitext(os.path.basename(path))[0].lower()
    name = re.sub(r"^aug_mega_\d+_", "", name)
    return re.sub(r"^(wm_\d+_)", "", name)


def measurement_photographs(per_folder=40, seed=7):
    """Exactly the photographs the gate will be judged on (scripts/measure_detection_quality.sample)."""
    from scripts import measure_detection_quality as mdq
    clean = mdq.sample(mdq.CLEAN_FOLDERS, per_folder, seed)
    defect = mdq.sample(mdq.DEFECT_FOLDERS, per_folder, seed)
    return [p for p, _ in clean + defect]


def load_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def crack_windows(rng, img, lab, n_pos=4, n_neg=2, min_frac=0.015, size=(80, 220)):
    H, W = lab.shape
    pos, neg = [], []
    ys, xs = np.nonzero(lab == 1)
    for _ in range(n_pos * 4):
        if len(pos) >= n_pos or not len(ys):
            break
        i = rng.randrange(len(ys))
        s = min(rng.randint(*size), H, W)
        cy, cx = ys[i] + rng.randint(-s // 4, s // 4), xs[i] + rng.randint(-s // 4, s // 4)
        y0, x0 = int(np.clip(cy - s // 2, 0, max(0, H - s))), int(np.clip(cx - s // 2, 0, max(0, W - s)))
        win = lab[y0:y0 + s, x0:x0 + s]
        if win.size and (win == 1).mean() >= min_frac:
            pos.append(img[y0:y0 + s, x0:x0 + s])
    for _ in range(n_neg * 8):
        if len(neg) >= n_neg:
            break
        s = min(rng.randint(64, 160), H, W)
        y0, x0 = rng.randint(0, max(0, H - s)), rng.randint(0, max(0, W - s))
        if not (lab[y0:y0 + s, x0:x0 + s] == 1).any():
            neg.append(img[y0:y0 + s, x0:x0 + s])
    return pos, neg


def dashcam_view(rng, img, lab):
    """The same pavement as a dashcam sees it: tilted away (far side narrower and shorter), smaller, softer,
    JPEG-compressed. The crack sets are photographed straight down; a vehicle camera never is."""
    import cv2
    H, W = lab.shape
    top = rng.uniform(0.35, 0.7)                       # far edge width / near edge width
    keep_h = rng.uniform(0.5, 0.85)                    # foreshortening
    dx = (1 - top) * W / 2
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[dx, H * (1 - keep_h)], [W - dx, H * (1 - keep_h)], [W, H], [0, H]])
    M = cv2.getPerspectiveTransform(src, dst)
    im = cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    lb = cv2.warpPerspective(lab.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST, borderValue=255)
    s = rng.uniform(0.45, 0.8)
    im = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    lb = cv2.resize(lb, (int(W * s), int(H * s)), interpolation=cv2.INTER_NEAREST)
    sig = rng.uniform(0.0, 1.2)
    if sig > 0.3:
        im = cv2.GaussianBlur(im, (0, 0), sig)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 85)])
    im = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    lb = lb.copy()
    lb[lb == 255] = 0
    return im, lb


def road_windows(rng, img, n=4):
    H, W = img.shape[:2]
    out = []
    top = int(0.35 * H)
    for _ in range(n):
        s = rng.randint(max(48, int(0.08 * W)), max(64, int(0.30 * W)))
        s = min(s, H - top, W)
        y0, x0 = rng.randint(top, max(top, H - s)), rng.randint(0, max(0, W - s))
        out.append(img[y0:y0 + s, x0:x0 + s])
    return out


def logistic_fit(X, y, lam, w_pos):
    """Class-weighted, L2-penalised logistic regression (L-BFGS). Returns (coef, intercept)."""
    from scipy.optimize import minimize
    sw = np.where(y == 1, w_pos, 1.0)
    sw = sw / sw.mean()

    def f(theta):
        w, b = theta[:-1], theta[-1]
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        loss = np.mean(sw * (np.logaddexp(0, z) - y * z)) + lam * (w @ w)
        g = sw * (p - y) / len(y)
        return loss, np.concatenate([X.T @ g + 2 * lam * w, [g.sum()]])

    res = minimize(f, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x[:-1], float(res.x[-1])


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--road-windows", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=CKPT)
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = random.Random(a.seed)

    man = json.load(open(os.path.join(SEG, "manifest.json"), encoding="utf-8"))
    crops = {s: {"crack": [], "pavement": [], "road": []} for s in ("train", "cal", "test")}
    used = {}
    for src in CRACK_SOURCES:
        items = list(man.get("items", {}).get(src) or [])
        random.Random(a.seed).shuffle(items)
        n = 0
        for it in items:
            if n >= MAX_PER_SOURCE:
                break
            ip, lp = os.path.join(SEG, src, "img", it["id"] + ".jpg"), os.path.join(SEG, src, "lab", it["id"] + ".png")
            if not (os.path.exists(ip) and os.path.exists(lp)):
                continue
            from PIL import Image
            img, lab = load_rgb(ip), np.asarray(Image.open(lp))
            pos, neg = crack_windows(rng, img, lab)
            vi, vl = dashcam_view(rng, img, lab)       # and the same pavement as a vehicle camera sees it
            p2, n2 = crack_windows(rng, vi, vl, n_pos=3, n_neg=1, min_frac=0.008, size=(56, 160))
            crops[it["split"]]["crack"] += pos + p2
            crops[it["split"]]["pavement"] += neg + n2
            n += 1
        used[src] = n
    if not sum(used.values()):
        raise SystemExit("no DeepCrack / CrackForest images under datasets/seg_multi: run "
                         "python -m scripts.fetch_seg_datasets --only deepcrack crackforest")

    # road scenes, never the measurement photographs or their near-copies
    from pipeline.corpus_policy import filter_paths
    from scripts.fetch_seg_datasets import dhash
    held = measurement_photographs()
    held_keys = {source_key(p) for p in held}
    held_hashes = np.array([dhash(load_rgb(p)) for p in held], dtype=np.uint64)
    road_used, road_dropped = {"train": 0, "cal": 0, "test": 0}, 0
    for f in ROAD_FOLDERS:
        files = sorted(glob.glob(os.path.join(ROOT, "datasets", f, "real_images", "*.jpg")))
        files, _ = filter_paths(files)
        for p in files:
            if os.path.basename(p).startswith(("rddin_", "rddw_")) or source_key(p) in held_keys:
                road_dropped += 1
                continue
            img = load_rgb(p)
            x = np.bitwise_xor(held_hashes, np.uint64(dhash(img)))
            if len(held_hashes) and np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1).min() <= 6:
                road_dropped += 1
                continue
            b = _bucket(source_key(p))
            sp = "train" if b < 0.6 else ("cal" if b < 0.8 else "test")
            crops[sp]["road"] += road_windows(rng, img, a.road_windows)
            road_used[sp] += 1

    counts = {s: {k: len(v) for k, v in d.items()} for s, d in crops.items()}
    print("[crack verifier] windows:", json.dumps(counts))

    from models.cnn_embedder import CNNEmbedder
    emb = CNNEmbedder(prefer=(BACKBONE,))
    if not emb.is_ready or emb.name != BACKBONE:
        raise SystemExit("the MobileNetV2 backbone is not on disk: python -m scripts.fetch_cnn_backbone")
    E = {s: {k: (emb.embed_batch(v) if v else np.zeros((0, 1000))) for k, v in d.items()} for s, d in crops.items()}

    def xy(split):
        d = E[split]
        X = np.vstack([d["crack"], d["pavement"], d["road"]])
        y = np.concatenate([np.ones(len(d["crack"])), np.zeros(len(d["pavement"]) + len(d["road"]))])
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9), y

    Xtr, ytr = xy("train")
    mean, scale = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = lambda X: (X - mean) / scale  # noqa: E731
    w_pos = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    Xca, yca = xy("cal")
    best = None
    for lam in LAMBDAS:
        coef, b = logistic_fit(Z(Xtr), ytr, lam, w_pos)
        p = 1 / (1 + np.exp(-np.clip(Z(Xca) @ coef + b, -40, 40)))
        ll = float(-np.mean(yca * np.log(p + 1e-9) + (1 - yca) * np.log(1 - p + 1e-9)))
        print(f"  lambda {lam:g}: calibration log-loss {ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, lam, coef, b)
    _, lam, coef, b = best

    def proba(X):
        return 1 / (1 + np.exp(-np.clip(Z(X) @ coef + b, -40, 40)))

    p_ca_crack = proba(xy_part("cal", "crack", E))
    thr = float(np.quantile(p_ca_crack, 1 - TARGET_RECALL)) if len(p_ca_crack) else 0.5

    def scored(split):
        pc, pp, pr = (proba(xy_part(split, k, E)) for k in ("crack", "pavement", "road"))
        return {"crack_windows": len(pc), "crack_kept": round(float((pc >= thr).mean()), 4) if len(pc) else None,
                "pavement_windows": len(pp), "pavement_rejected": round(float((pp < thr).mean()), 4) if len(pp) else None,
                "road_windows": len(pr), "road_scene_rejected": round(float((pr < thr).mean()), 4) if len(pr) else None,
                "auroc_crack_vs_pavement": _r(auroc(pc, pp)), "auroc_crack_vs_road_scene": _r(auroc(pc, pr))}

    report = {
        "model": "logistic regression on L2-normalised MobileNetV2 ImageNet embeddings",
        "task": "is the crop around a segmenter crack component a pavement crack",
        "trained_on": {"crack_images": used, "road_photographs": road_used,
                       "road_photographs_held_out_for_the_gate_decision_or_near_copies": road_dropped,
                       "windows": counts, "sources": {s: (man.get("sources", {}).get(s) or {}).get("repository")
                                                      for s in CRACK_SOURCES}},
        "lambda_chosen_on_calibration": lam,
        "threshold": round(thr, 4),
        "threshold_rule": f"highest threshold keeping {TARGET_RECALL:.0%} of calibration crack windows",
        "calibration": scored("cal"),
        "test": {"summary": scored("test"),
                 "note": "windows from images and road photographs never used for training or for the threshold"},
        "serving_decision": "checkpoints/crack_verifier_selection.json, written by python -m scripts.select_crack_gate "
                            "for this model file only (a retrained file needs a new decision)",
        "caveats": ["crack windows are top-down close-ups (DeepCrack, CrackForest); dashcam crack crops are seen "
                    "from an angle - the end-to-end check decides whether that matters",
                    "the licences of both crack datasets allow non-commercial research and education only"],
        "trained_unix": int(time.time()), "seconds": round(time.time() - t0, 1),
    }
    meta = {"threshold": round(thr, 4), "backbone": BACKBONE, "lambda": lam, "trained_on": report["trained_on"]["crack_images"],
            "version": time.strftime("%Y-%m-%d")}
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, MODEL_FILE), mean=mean, scale=scale, coef=coef, intercept=np.float64(b),
             meta_json=np.array(json.dumps(meta)))
    with open(os.path.join(a.out, REPORT_FILE), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({k: report[k] for k in ("threshold", "calibration", "test")}, indent=1))
    return report


def xy_part(split, kind, E):
    X = E[split][kind]
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9) if len(X) else X


def _r(v):
    return None if v is None else round(v, 4)


if __name__ == "__main__":
    main()
