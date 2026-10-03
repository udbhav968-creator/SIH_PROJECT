"""
Deep 1-D CNN for the 100 Hz IMU shock classifier, compared honestly with the
served RandomForest on hand-crafted features.

    python -m training.train_imu_deep            # GPU optional - the data is tiny

Data: datasets/04_mobile_imu_telemetry_100hz/imu_shock_100hz_{train,val}.npz -
688 training and 164 held-out windows (100 samples x 3 axes, m/s^2) cut from
10 real drive logs, split by time block upstream.

Protocol
    1. 5-fold stratified CV on the TRAIN windows only, for both the RandomForest
       (same pipeline as models/imu_shock_classifier.py) and the CNN, with
       several seeds for the CNN because 688 windows make single runs noisy.
    2. Decision rule fixed before the held-out windows are scored: serve the
       CNN only if its mean CV macro-F1 AND CV accuracy beat the forest's.
    3. Fit both on all training windows, score the 164 held-out windows once,
       report both. Export the CNN to ONNX (checked against PyTorch).

Outputs (checkpoints/):
    imu_shock_cnn.onnx + imu_shock_cnn.json   network + normalisation sidecar
    imu_deep_report.json                      CV table, held-out metrics, decision
    imu_model_selection.json                  which IMU model the pipeline serves
"""
import json
import os
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import numpy as np

from models.imu_shock_classifier import IMUShockClassifier

DATA = os.path.join(ENGINE_ROOT, "datasets", "04_mobile_imu_telemetry_100hz")
CKPT = os.path.join(ENGINE_ROOT, "checkpoints")
N_CLASSES = len(IMUShockClassifier.CLASS_NAMES)


def load(split):
    d = np.load(os.path.join(DATA, f"imu_shock_100hz_{split}.npz"))
    return d["imu_signals"].astype(np.float32), d["labels"].astype(np.int64)


def normalise(X, scale):
    """Remove each window's per-axis mean (gravity / mounting tilt), divide by
    the training set's per-axis std. Shape (N, 100, 3) -> (N, 3, 100)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    return np.transpose(Xc / scale[None, None, :], (0, 2, 1)).astype(np.float32)


def scores(y, p):
    from sklearn.metrics import accuracy_score, f1_score
    return float(accuracy_score(y, p)), float(f1_score(y, p, labels=list(range(N_CLASSES)),
                                                       average="macro", zero_division=0))


def build_net(width=32):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, cin, cout, k, stride):
            super().__init__()
            self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
            self.b1 = nn.BatchNorm1d(cout)
            self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
            self.b2 = nn.BatchNorm1d(cout)
            self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))
                         if (cin != cout or stride != 1) else nn.Identity())
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.b2(self.c2(self.act(self.b1(self.c1(x))))) + self.skip(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.stem = nn.Sequential(nn.Conv1d(3, w, 7, padding=3, bias=False), nn.BatchNorm1d(w), nn.ReLU())
            self.blocks = nn.Sequential(Block(w, w, 5, 1), Block(w, 2 * w, 5, 2), Block(2 * w, 4 * w, 3, 2))
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(8 * w, N_CLASSES)

        def forward(self, x):
            import torch
            h = self.blocks(self.stem(x))
            h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
            return self.fc(self.drop(h))

    return Net()


def fit_cnn(Xtr, ytr, seed, epochs=120, width=32, device=None):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_net(width).to(device)
    counts = np.bincount(ytr, minlength=N_CLASSES).astype(float)
    w = torch.tensor(np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** 0.5, 0.0),
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w / w.mean(), label_smoothing=0.05)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * max(1, int(np.ceil(len(ytr) / 64))))
    X = torch.tensor(Xtr, device=device)
    Y = torch.tensor(ytr, device=device)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.tensor(rng.permutation(len(Y)), device=device)
        for i in range(0, len(Y), 64):
            idx = perm[i:i + 64]
            xb = X[idx]
            # augmentation: circular time shift, amplitude scaling, sensor noise
            shift = int(rng.integers(-12, 13))
            xb = torch.roll(xb, shifts=shift, dims=2)
            xb = xb * torch.empty(len(idx), 1, 1, device=device).uniform_(0.8, 1.25)
            xb = xb + 0.03 * torch.randn_like(xb)
            opt.zero_grad()
            loss = crit(net(xb), Y[idx])
            loss.backward()
            opt.step(); sched.step()
    net.eval()
    return net


def cnn_predict(net, X):
    import torch
    dev = next(net.parameters()).device
    with torch.no_grad():
        lo = net(torch.tensor(X, device=dev)).float()
    return lo.softmax(1).cpu().numpy()


def main():
    from sklearn.model_selection import StratifiedKFold
    import torch

    Xtr_raw, ytr = load("train")
    Xva_raw, yva = load("val")
    scale = (Xtr_raw - Xtr_raw.mean(axis=1, keepdims=True)).reshape(-1, 3).std(axis=0) + 1e-6
    Xtr, Xva = normalise(Xtr_raw, scale), normalise(Xva_raw, scale)
    print(f"[IMU deep] train {len(ytr)} | held-out {len(yva)} | class counts train {np.bincount(ytr).tolist()}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(Xtr_raw, ytr))
    cv = {"random_forest": [], "cnn_w32": [], "cnn_w48": []}
    t0 = time.time()
    for k, (a, b) in enumerate(folds):
        rf = IMUShockClassifier().fit(Xtr_raw[a], ytr[a])
        cv["random_forest"].append(scores(ytr[b], rf.predict(Xtr_raw[b])[0]))
        for width in (32, 48):
            probs = np.mean([cnn_predict(fit_cnn(Xtr[a], ytr[a], seed=s, width=width), Xtr[b]) for s in (0, 1, 2)], axis=0)
            cv[f"cnn_w{width}"].append(scores(ytr[b], probs.argmax(1)))
        print(f"  fold {k + 1}/5 done ({time.time() - t0:.0f}s): " +
              ", ".join(f"{m} F1 {v[-1][1]:.3f}" for m, v in cv.items()), flush=True)
    cv_table = {m: {"cv_accuracy": round(float(np.mean([v[0] for v in r])), 4),
                    "cv_accuracy_std": round(float(np.std([v[0] for v in r])), 4),
                    "cv_macro_f1": round(float(np.mean([v[1] for v in r])), 4),
                    "cv_macro_f1_std": round(float(np.std([v[1] for v in r])), 4)} for m, r in cv.items()}
    best_cnn = max(("cnn_w32", "cnn_w48"), key=lambda m: cv_table[m]["cv_macro_f1"])
    rf_cv, cnn_cv = cv_table["random_forest"], cv_table[best_cnn]
    serve_cnn = cnn_cv["cv_macro_f1"] > rf_cv["cv_macro_f1"] and cnn_cv["cv_accuracy"] > rf_cv["cv_accuracy"]

    # final fits on all training windows; held-out scored once each
    rf = IMUShockClassifier().fit(Xtr_raw, ytr)
    rf_pred = rf.predict(Xva_raw)[0]
    width = int(best_cnn.split("w")[1])
    nets = [fit_cnn(Xtr, ytr, seed=s, width=width) for s in (0, 1, 2)]
    cnn_prob = np.mean([cnn_predict(n, Xva) for n in nets], axis=0)
    cnn_pred = cnn_prob.argmax(1)

    from sklearn.metrics import classification_report, confusion_matrix

    def held(y, p):
        acc, f1 = scores(y, p)
        return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4),
                "confusion_matrix": confusion_matrix(y, p, labels=list(range(N_CLASSES))).tolist(),
                "per_class": classification_report(y, p, labels=list(range(N_CLASSES)),
                                                   target_names=IMUShockClassifier.CLASS_NAMES,
                                                   output_dict=True, zero_division=0)}

    # Export the 3-seed ensemble as one ONNX graph (mean of softmaxes).
    class Ens(torch.nn.Module):
        def __init__(self, ms):
            super().__init__()
            self.ms = torch.nn.ModuleList(ms)

        def forward(self, x):
            return torch.stack([m(x).softmax(1) for m in self.ms]).mean(0)

    ens = Ens([n.cpu().eval() for n in nets]).eval()
    onnx_path = os.path.join(CKPT, "imu_shock_cnn.onnx")
    kw = dict(input_names=["imu"], output_names=["probs"],
              dynamic_axes={"imu": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(ens, torch.tensor(Xva[:2]), onnx_path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(ens, torch.tensor(Xva[:2]), onnx_path, **kw)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"imu": Xva})[0]
    parity = float(np.max(np.abs(got - cnn_prob)))
    with open(os.path.join(CKPT, "imu_shock_cnn.json"), "w") as fh:
        json.dump({"axis_scale": scale.tolist(), "input": "(N, 3, 100): per-window mean removed, divided by axis_scale",
                   "class_names": IMUShockClassifier.CLASS_NAMES, "ensemble_seeds": [0, 1, 2], "width": width}, fh, indent=1)

    report = {
        "data": {"train_windows": int(len(ytr)), "held_out_windows": int(len(yva)),
                 "source": "real 100 Hz drive logs (datasets/04_mobile_imu_telemetry_100hz), time-block split"},
        "cv_on_train_only": cv_table,
        "best_cnn_config": best_cnn,
        "decision_rule": "serve the CNN only if its mean 5-fold CV accuracy AND macro-F1 on the training windows "
                         "beat the RandomForest's; fixed before the held-out windows were scored",
        "served": "cnn" if serve_cnn else "random_forest",
        "held_out": {"random_forest": held(yva, rf_pred), "cnn": held(yva, cnn_pred)},
        "onnx_parity_max_abs_prob_diff": round(parity, 7),
        "generated_unix": int(time.time()),
    }
    with open(os.path.join(CKPT, "imu_deep_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(CKPT, "imu_model_selection.json"), "w") as fh:
        json.dump({"served": report["served"], "rule": report["decision_rule"],
                   "cv": {"random_forest": rf_cv, "cnn": cnn_cv},
                   "held_out_for_reporting": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
                                              for k, v in report["held_out"].items()}}, fh, indent=1)
    print(json.dumps({k: report[k] for k in ("cv_on_train_only", "served")}, indent=1))
    print("held-out:", {k: (v["accuracy"], v["macro_f1"]) for k, v in report["held_out"].items()})
    return report


if __name__ == "__main__":
    main()
