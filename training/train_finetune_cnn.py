"""
End-to-end fine-tuning of ImageNet CNNs on the road corpus (GPU recommended).

The served classifier used to be a FROZEN MobileNetV2 with a scikit-learn head
on top - transfer learning that never updates a single convolution. This script
trains the whole network, end to end, and compares it to that head on the
IDENTICAL split (training.train_cnn_head.capped_grouped_split), so any change
in the headline number is a like-for-like comparison and not a new test set.

    python -m training.train_finetune_cnn --archs efficientnet_b0,efficientnet_b2,mobilenet_v3_large,resnet50 --epochs 40
    python -m training.train_finetune_cnn --smoke            # 2-minute sanity run

Protocol (what makes the numbers trustworthy)
---------------------------------------------
* Split: grouped by source photograph (crops of one photo never straddle a
  split), the same capped corpus and seed as the frozen head.
* Train on `train`, early-stop and select on `val`, score `test` exactly once
  per architecture at the end. Architecture choice uses validation macro-F1
  only; test numbers are reported, never used to choose.
* Imbalance: square-root inverse-frequency sampling plus label smoothing - the
  rare municipal classes (7-8 test images each) would otherwise be ignored.
* Every architecture is exported to ONNX and its ONNX outputs are checked
  against PyTorch on real images (max |logit diff| reported), then timed on
  CPU with ONNX Runtime, because the edge box has no GPU.
* Indian roads: if datasets/_eval_rdd2022_india exists (written by
  scripts/ingest_rdd2022_india.py from RDD2022 India photographs no training
  run reads) each model is also scored there.

Outputs (checkpoints/):
    deep_vision_<arch>.onnx + .json    weights + preprocessing sidecar (selected arch only)
    finetune_<arch>_report.json        per-architecture curves, val/test/Indian metrics
    finetune_summary.json              all architectures side by side + the selection
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import numpy as np

from data.image_dataset import CLASS_NAMES

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
EVAL_INDIA = os.path.join(ENGINE_ROOT, "datasets", "_eval_rdd2022_india")
INDIA_CLASSES = {"normal": 0, "crack": 1, "pothole": 2}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

ARCHS = {
    # name: (torchvision builder, weights enum, train/eval size)
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", 224),
    "efficientnet_b2": ("efficientnet_b2", "EfficientNet_B2_Weights", 260),
    "resnet50": ("resnet50", "ResNet50_Weights", 224),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", 224),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", 224),
}


# ----------------------------------------------------------------------------
def macro_report(y, p, n_classes):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    labels = list(range(n_classes))
    present = sorted(set(int(v) for v in y))
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {
        "images": int(len(y)),
        "accuracy": round(float(accuracy_score(y, p)), 4),
        # macro over the classes that actually occur in this split
        "macro_f1": round(float(f1_score(y, p, labels=present, average="macro", zero_division=0)), 4),
        "per_class": {CLASS_NAMES[i]: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                                       "f1": round(float(f1[i]), 4), "support": int(sup[i])}
                      for i in labels if sup[i] > 0},
        "confusion_matrix": confusion_matrix(y, p, labels=labels).tolist(),
    }


def build(arch, n_classes):
    import torch.nn as nn
    from torchvision import models
    fn_name, w_name, size = ARCHS[arch]
    weights = getattr(models, w_name).DEFAULT
    m = getattr(models, fn_name)(weights=weights)
    if arch.startswith("efficientnet") or arch.startswith("mobilenet"):
        idx = len(m.classifier) - 1
        m.classifier[idx] = nn.Linear(m.classifier[idx].in_features, n_classes)
        head = m.classifier
    elif arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        head = m.fc
    elif arch.startswith("convnext"):
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
        head = m.classifier
    else:
        raise ValueError(arch)
    return m, head, size


class CachedImages:
    """Decode every image once, resized so the short side is `short`, and keep
    the uint8 arrays in RAM. Colab has two CPU cores; decoding 1024x640 JPEGs
    every epoch would starve the GPU."""

    def __init__(self, paths, short):
        # Kept as re-encoded JPEG bytes (~40 KB each), not decoded arrays: raw
        # arrays for the RDD-enlarged corpus, copied into each DataLoader worker,
        # would not fit a 12 GB Colab runtime. Decoding a 300-px JPEG is cheap.
        import io
        from PIL import Image
        self.arrays = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            s = short / float(min(w, h))
            if s < 1.0:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=95)
            self.arrays.append(buf.getvalue())


def make_loaders(train_items, val_items, test_items, size, batch, workers, seed):
    import torch
    from PIL import Image
    from torchvision import transforms as T

    short = int(size * 1.15) + 32
    cache = {}
    for name, items in (("train", train_items), ("val", val_items), ("test", test_items)):
        t = time.time()
        cache[name] = CachedImages([i[0] for i in items], short)
        print(f"  cached {len(items)} {name} images in {time.time() - t:.0f}s", flush=True)

    train_tf = T.Compose([
        T.RandomResizedCrop(size, scale=(0.45, 1.0), ratio=(0.7, 1.45)),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.12)),
    ])
    eval_tf = T.Compose([T.Resize(int(size * 1.15)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])

    class DS(torch.utils.data.Dataset):
        def __init__(self, arrays, labels, tf):
            self.arrays, self.labels, self.tf = arrays, labels, tf

        def __len__(self):
            return len(self.arrays)

        def __getitem__(self, i):
            import io
            return self.tf(Image.open(io.BytesIO(self.arrays[i])).convert("RGB")), int(self.labels[i])

    y_tr = np.array([i[1] for i in train_items])
    counts = np.bincount(y_tr, minlength=len(CLASS_NAMES)).astype(float)
    w_cls = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    g = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(w_cls[y_tr].tolist(), num_samples=len(y_tr),
                                                     replacement=True, generator=g)
    mk = lambda arrs, items, tf, **kw: torch.utils.data.DataLoader(
        DS(arrs, [i[1] for i in items], tf), batch_size=batch, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, **kw)
    return (mk(cache["train"].arrays, train_items, train_tf, sampler=sampler, drop_last=True),
            mk(cache["val"].arrays, val_items, eval_tf, shuffle=False),
            mk(cache["test"].arrays, test_items, eval_tf, shuffle=False),
            eval_tf)


def predict(model, loader, device, tta=True):
    import torch
    model.eval()
    ys, ps, logits_all = [], [], []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            lo = model(x).float()
            if tta:
                lo = 0.5 * (lo + model(torch.flip(x, dims=[3])).float())
            logits_all.append(lo.cpu().numpy())
            ps.append(lo.argmax(1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(logits_all)


def india_eval(model, eval_tf, device, tta=True):
    import torch
    from PIL import Image
    if not os.path.isdir(EVAL_INDIA):
        return None
    y, p = [], []
    model.eval()
    batch, labels = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            lo = model(x).float()
            if tta:
                lo = 0.5 * (lo + model(torch.flip(x, dims=[3])).float())
        p.extend(lo.argmax(1).cpu().numpy().tolist())
        y.extend(labels)
        batch.clear(); labels.clear()

    for name, cid in INDIA_CLASSES.items():
        for f in sorted(glob.glob(os.path.join(EVAL_INDIA, name, "*.jpg"))):
            batch.append(eval_tf(Image.open(f).convert("RGB")))
            labels.append(cid)
            if len(batch) == 64:
                flush()
    flush()
    if not y:
        return None
    from sklearn.metrics import accuracy_score, f1_score
    present = sorted(set(y))
    per = f1_score(y, p, labels=present, average=None, zero_division=0)
    return {"images": len(y), "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, labels=present, average="macro", zero_division=0)), 4),
            "per_class_f1": {n: round(float(per[present.index(c)]), 4) for n, c in INDIA_CLASSES.items() if c in present}}


def export_onnx(model, size, path, sample_batch, device):
    """Export, then prove the ONNX graph computes what PyTorch computes."""
    import torch
    model = model.float().eval().to("cpu")
    dummy = torch.randn(1, 3, size, size)
    kw = dict(input_names=["image"], output_names=["logits"],
              dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kw)
    except TypeError:  # torch < 2.5 has no dynamo switch
        torch.onnx.export(model, dummy, path, **kw)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(sample_batch.cpu().float()).numpy()
    got = sess.run(None, {"image": sample_batch.cpu().numpy().astype(np.float32)})[0]
    max_diff = float(np.max(np.abs(ref - got)))
    agree = float(np.mean(ref.argmax(1) == got.argmax(1)))
    # CPU latency, one image at a time, as the edge box would run it
    x1 = sample_batch[:1].cpu().numpy().astype(np.float32)
    for _ in range(3):
        sess.run(None, {"image": x1})
    t = time.time()
    n = 20
    for _ in range(n):
        sess.run(None, {"image": x1})
    lat = (time.time() - t) / n * 1000.0
    model.to(device)
    return {"onnx_max_abs_logit_diff": round(max_diff, 6), "onnx_argmax_agreement": round(agree, 4),
            "onnx_cpu_ms_per_image": round(lat, 1), "onnx_size_mb": round(os.path.getsize(path) / 1e6, 1),
            "cpu_threads": os.cpu_count()}


def train_arch(arch, splits, epochs, batch, lr, seed, workers, patience, out_dir, smoke=False, refit=False):
    """refit=True: train on train+val for exactly `epochs` epochs (the best epoch
    found with validation), no early stopping - the same data the frozen head
    is fitted on. Validation numbers are then not reported (val is in training)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_items, val_items, test_items = splits
    if refit:
        train_items = list(train_items) + list(val_items)
    model, head, size = build(arch, len(CLASS_NAMES))
    model.to(device).to(memory_format=torch.channels_last)
    tl, vl, testl, eval_tf = make_loaders(train_items, val_items, test_items, size, batch, workers, seed)

    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": lr}, {"params": list(head.parameters()), "lr": lr * 10}],
                            weight_decay=0.05)
    steps = epochs * len(tl)
    warm = max(1, len(tl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = {"score": -1.0, "epoch": 0, "state": None}
    history, bad, t0 = [], 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run, seen, te = 0.0, 0, time.time()
        for bi, (x, y) in enumerate(tl):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss.item()) * len(y); seen += len(y)
            if smoke and bi >= 3:
                break
        if refit:
            history.append({"epoch": ep, "train_loss": round(run / max(1, seen), 4),
                            "seconds": round(time.time() - te, 1)})
            print(f"  [{arch} refit] epoch {ep:2d}/{epochs} loss {history[-1]['train_loss']:.3f} | "
                  f"{history[-1]['seconds']}s", flush=True)
            best = {"score": float("nan"), "epoch": ep, "state": None}
            continue
        yv, pv, _ = predict(model, vl, device)
        rv = macro_report(yv, pv, len(CLASS_NAMES))
        score = 0.5 * (rv["accuracy"] + rv["macro_f1"])  # same selection score as the frozen head
        history.append({"epoch": ep, "train_loss": round(run / max(1, seen), 4), "val_accuracy": rv["accuracy"],
                        "val_macro_f1": rv["macro_f1"], "seconds": round(time.time() - te, 1)})
        print(f"  [{arch}] epoch {ep:2d}/{epochs} loss {history[-1]['train_loss']:.3f} | val acc "
              f"{rv['accuracy']:.3f} macro-F1 {rv['macro_f1']:.3f} | {history[-1]['seconds']}s", flush=True)
        if score > best["score"]:
            best = {"score": score, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  [{arch}] early stop at epoch {ep} (best {best['epoch']})", flush=True)
                break
        if smoke and ep >= 1:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    yv, pv, _ = predict(model, vl, device)
    yt, pt_, _ = predict(model, testl, device)  # the single test-set scoring for this arch
    val_rep = macro_report(yv, pv, len(CLASS_NAMES))
    test_rep = macro_report(yt, pt_, len(CLASS_NAMES))
    india = india_eval(model, eval_tf, device)

    if refit:
        val_rep = {"note": "validation photographs were part of this refit's training data"}
    tag = f"{arch}_refit" if refit else arch
    onnx_path = os.path.join(out_dir, f"deep_vision_{tag}.onnx")
    sample = next(iter(testl))[0][:8]
    parity = export_onnx(model, size, onnx_path, sample, device)
    report = {
        "model": f"{arch} (ImageNet-pretrained, fine-tuned end to end" + (", refit on train+val)" if refit else ")"),
        "refit_on_train_plus_val": refit,
        "arch": arch, "img_size": size, "tta_flip": True,
        "class_names": CLASS_NAMES,
        "split": {"rule": "training.train_cnn_head.capped_grouped_split - identical to the frozen head",
                  "train_images": len(train_items), "val_images": len(val_items), "test_images": len(test_items),
                  "test_photographs": len({i[2] for i in test_items})},
        "training": {"epochs_run": len(history), "best_epoch": best["epoch"], "batch": batch, "lr_backbone": lr,
                     "lr_head": lr * 10, "optimizer": "AdamW wd 0.05, warmup + cosine", "loss": "CE, label smoothing 0.1",
                     "sampler": "sqrt inverse-frequency", "augmentation": "RandomResizedCrop, flip, TrivialAugmentWide, RandomErasing",
                     "device": str(device), "seconds": round(time.time() - t0, 1), "history": history},
        "selection_score_val": None if refit else round(best["score"], 4),
        "validation": val_rep,
        "test": test_rep,
        "indian_roads_rdd2022_test": india,
        "onnx": parity,
        "trained_at_unix": int(time.time()),
    }
    with open(os.path.join(out_dir, f"finetune_{tag}_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"  [{tag}] VAL acc {val_rep.get('accuracy')} F1 {val_rep.get('macro_f1')} | TEST acc "
          f"{test_rep['accuracy']:.3f} F1 {test_rep['macro_f1']:.3f} | India "
          f"{(india or {}).get('macro_f1')} | ONNX {parity['onnx_cpu_ms_per_image']} ms/img", flush=True)
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="efficientnet_b0,efficientnet_b2,mobilenet_v3_large,resnet50")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-per-class", type=int, default=1500)
    ap.add_argument("--out", default=CKPT_DIR)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-refit", action="store_true", help="serve the train-only model of the chosen arch")
    ap.add_argument("--resume", action="store_true", help="skip architectures whose report already exists")
    ap.add_argument("--train-only", action="store_true",
                    help="train/score the listed architectures and stop (no selection, refit or cleanup); "
                         "used to train one architecture per call so a disconnect loses at most one")
    a = ap.parse_args(argv)

    from training.train_cnn_head import capped_grouped_split
    capped, splits = capped_grouped_split(a.seed, a.max_per_class)
    if a.smoke:
        splits = tuple(s[: n] for s, n in zip(splits, (96, 48, 48)))
        a.epochs = 1
    print(f"[finetune] train {len(splits[0])} | val {len(splits[1])} | test {len(splits[2])} images "
          f"({len({i[2] for i in splits[2]})} test photographs)", flush=True)
    os.makedirs(a.out, exist_ok=True)

    reports = {}
    for arch in [s.strip() for s in a.archs.split(",") if s.strip()]:
        done = os.path.join(a.out, f"finetune_{arch}_report.json")
        if a.resume and os.path.exists(done):
            # A Colab runtime can disconnect mid-run; an architecture that already
            # finished (report written, test scored once) is not trained again.
            reports[arch] = json.load(open(done))
            print(f"  [{arch}] already trained - reusing {os.path.basename(done)}", flush=True)
            continue
        reports[arch] = train_arch(arch, splits, a.epochs, a.batch, a.lr, a.seed, a.workers, a.patience,
                                   a.out, smoke=a.smoke)

    if a.train_only:
        return {"trained": list(reports)}

    # Select by validation only, among networks small enough to commit (GitHub
    # refuses files over 100 MB) - a deployment constraint fixed before training.
    eligible = [k for k in reports if reports[k]["onnx"]["onnx_size_mb"] <= 95.0] or list(reports)
    chosen = max(eligible, key=lambda k: reports[k]["selection_score_val"])
    served_tag = chosen
    refit_rep = None
    if not a.no_refit:
        # Pre-declared: the served network is the chosen architecture refit on
        # train+val for its best validation epoch count - the same data the
        # frozen head is fitted on. Its test score is reported as the served one.
        refit_done = os.path.join(a.out, f"finetune_{chosen}_refit_report.json")
        refit_onnx = os.path.join(a.out, f"deep_vision_{chosen}_refit.onnx")
        if a.resume and os.path.exists(refit_done) and os.path.exists(refit_onnx):
            refit_rep = json.load(open(refit_done))
            print(f"  [{chosen} refit] already trained - reusing", flush=True)
        else:
            refit_rep = train_arch(chosen, splits, max(1, reports[chosen]["training"]["best_epoch"]), a.batch,
                                   a.lr, a.seed, a.workers, a.patience, a.out, smoke=a.smoke, refit=True)
        served_tag = f"{chosen}_refit"
    served = refit_rep or reports[chosen]
    head_rep_path = os.path.join(a.out, "cnn_head_mobilenetv2_report.json")
    head = json.load(open(head_rep_path)) if os.path.exists(head_rep_path) else {}
    summary = {
        "selection_rule": "architecture with the highest validation score = mean(accuracy, macro-F1), among those whose ONNX file is <= 95 MB; "
                          "test scored once per arch and never used to choose; the served model is the "
                          "chosen arch refit on train+val for its best epoch count (declared before training)",
        "chosen_arch": chosen,
        "served": served_tag,
        "served_test": {"accuracy": served["test"]["accuracy"], "macro_f1": served["test"]["macro_f1"],
                        "images": served["test"]["images"]},
        "served_indian_roads": served["indian_roads_rdd2022_test"],
        "served_onnx": served["onnx"],
        "archs": {k: {"val_score": r["selection_score_val"], "val": {"accuracy": r["validation"]["accuracy"], "macro_f1": r["validation"]["macro_f1"]},
                      "test": {"accuracy": r["test"]["accuracy"], "macro_f1": r["test"]["macro_f1"]},
                      "indian_roads": r["indian_roads_rdd2022_test"], "onnx_cpu_ms_per_image": r["onnx"]["onnx_cpu_ms_per_image"],
                      "onnx_size_mb": r["onnx"]["onnx_size_mb"], "best_epoch": r["training"]["best_epoch"]}
                  for k, r in reports.items()},
        "frozen_head_same_split": {"model": head.get("model"), "test_accuracy": head.get("held_out_test_accuracy"),
                                   "test_macro_f1": head.get("held_out_test_macro_f1"),
                                   "test_images": head.get("held_out_test_images")},
        "generated_unix": int(time.time()),
    }
    with open(os.path.join(a.out, "finetune_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    # keep only the served network's weights; the sidecar tells the loader how to preprocess
    for p in glob.glob(os.path.join(a.out, "deep_vision_*.onnx")):
        if os.path.basename(p) != f"deep_vision_{served_tag}.onnx":
            os.remove(p)
    for p in glob.glob(os.path.join(a.out, "deep_vision_*.json")):
        os.remove(p)
    with open(os.path.join(a.out, f"deep_vision_{served_tag}.json"), "w") as fh:
        json.dump({"arch": chosen, "img_size": served["img_size"], "tta_flip": True,
                   "mean": MEAN, "std": STD, "resize_ratio": 1.15, "class_names": CLASS_NAMES,
                   "report": f"checkpoints/finetune_{served_tag}_report.json"}, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
