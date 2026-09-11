"""
Real deep learning for the road-condition classifier: fine-tunes an
ImageNet-pretrained CNN on the photographs under datasets/.

This replaces the hand-crafted HOG/LBP + SVM baseline in training/train_vision.py.
That baseline stays in the repo so the two can be compared honestly on the
same split.

    python -m training.train_deep_vision --epochs 12
    python -m training.train_deep_vision --smoke        # 2 min sanity run
    python -m training.train_deep_vision --arch efficientnet_b0 --epochs 20

Design decisions that matter for the numbers being trustworthy:

Grouped splitting.  The dataset contains augmented copies named
`aug_mega_<n>_<original>`. Splitting at file level would put copies of the
same photograph in both training and test, which inflates accuracy. Every
file is therefore assigned to a *group* (its original photograph) and whole
groups go to train, validation or test.

Three-way split.  Validation drives early stopping and model selection; the
test set is touched exactly once, at the end. A score you tuned against is
not a held-out score.

Class imbalance.  Classes here range from ~14 to ~270 images, so batches are
drawn with a weighted sampler and the loss is class-weighted. Accuracy alone
would be misleading, so macro-F1 is the selection metric and per-class
precision/recall are always reported.

Two-phase fine-tune.  Phase 1 trains only the new classifier head with the
backbone frozen (fast, stable on CPU). Phase 2 unfreezes the last backbone
block at a 10x lower learning rate. On a laptop CPU this is minutes per
epoch, not hours.

Outputs (checkpoints/):
    deep_vision_<arch>.pt          weights + class names + normalisation
    deep_vision_<arch>.onnx        portable graph for ONNX Runtime inference
    deep_vision_report.json        split sizes, per-class metrics, confusion
                                   matrices for validation and test, timings
"""

import argparse
import json
import os
import random
import re
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import numpy as np

from data.image_dataset import CLASS_FOLDERS, CLASS_NAMES, DATASETS_ROOT, _IMG_EXT

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
_AUG_RE = re.compile(r"^aug_mega_\d+_")
_MASK_SUFFIXES = ("_CRACK.png", "_POTHOLE.png", "_LANE.png")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        sys.exit(
            "PyTorch is not installed in this environment.\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
            "(CPU build: about 250 MB, no GPU required.)"
        )


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def group_key(filename):
    """The original photograph a file belongs to, so copies stay together."""
    stem = _AUG_RE.sub("", os.path.basename(filename))
    stem = os.path.splitext(stem)[0]
    for suffix in ("_RAW", "_CRACK", "_POTHOLE", "_LANE"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def collect_files():
    """[(path, class_id, group_key)] for every usable training photo."""
    items = []
    for cls, (folder, _name) in sorted(CLASS_FOLDERS.items()):
        d = os.path.join(DATASETS_ROOT, folder, "real_images")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(_IMG_EXT) or f.endswith(_MASK_SUFFIXES):
                continue
            items.append((os.path.join(d, f), cls, f"{cls}:{group_key(f)}"))
    return items


def grouped_split(items, seed=42, val_frac=0.15, test_frac=0.15):
    """Split by group so copies of one photograph never cross the split."""
    by_class = {}
    for path, cls, grp in items:
        by_class.setdefault(cls, {}).setdefault(grp, []).append((path, cls, grp))

    rng = random.Random(seed)
    train, val, test = [], [], []
    for cls, groups in sorted(by_class.items()):
        keys = sorted(groups)
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(1, int(round(n * test_frac))) if n >= 4 else (1 if n >= 2 else 0)
        n_val = max(1, int(round(n * val_frac))) if n >= 4 else (1 if n >= 3 else 0)
        for i, k in enumerate(keys):
            bucket = test if i < n_test else (val if i < n_test + n_val else train)
            bucket.extend(groups[k])
    return train, val, test


def build_datasets(train_items, val_items, test_items, img_size):
    import torch
    from PIL import Image
    from torchvision import transforms

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    class PhotoSet(torch.utils.data.Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, cls, _grp = self.items[i]
            img = Image.open(path).convert("RGB")
            return self.tf(img), cls

    return PhotoSet(train_items, train_tf), PhotoSet(val_items, eval_tf), PhotoSet(test_items, eval_tf)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_model(arch, n_classes):
    import torch.nn as nn
    from torchvision import models

    def pretrained(fn, weights_enum):
        try:
            return fn(weights=weights_enum.DEFAULT)
        except Exception:
            return fn(pretrained=True)  # older torchvision

    if arch == "resnet18":
        m = pretrained(models.resnet18, models.ResNet18_Weights)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        head_params = list(m.fc.parameters())
        last_block = m.layer4
    elif arch == "resnet34":
        m = pretrained(models.resnet34, models.ResNet34_Weights)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        head_params = list(m.fc.parameters())
        last_block = m.layer4
    elif arch == "efficientnet_b0":
        m = pretrained(models.efficientnet_b0, models.EfficientNet_B0_Weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
        head_params = list(m.classifier.parameters())
        last_block = m.features[-3:]
    elif arch == "mobilenet_v3_small":
        m = pretrained(models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
        head_params = list(m.classifier.parameters())
        last_block = m.features[-3:]
    else:
        sys.exit(f"Unknown --arch {arch}. Options: resnet18, resnet34, efficientnet_b0, mobilenet_v3_small")
    return m, head_params, last_block


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, n_classes):
    import torch
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    correct = total = 0
    losses = []
    import torch.nn.functional as F
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(float(F.cross_entropy(logits, y).item()))
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                cm[t, p] += 1
    return {
        "accuracy": (correct / total) if total else 0.0,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "confusion_matrix": cm.tolist(),
        "n": total,
    }


def per_class_report(cm, class_names):
    cm = np.asarray(cm)
    out, f1s = {}, []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = int(cm[i, :].sum())
        if support:
            f1s.append(f1)
        out[name] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": support}
    return out, float(np.mean(f1s)) if f1s else 0.0


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def run_training(arch="resnet18", epochs=12, batch_size=24, img_size=224, lr=3e-4,
                 seed=42, smoke=False, workers=0, save_dir=None):
    require_torch()
    import torch
    import torch.nn as nn

    save_dir = save_dir or CKPT_DIR
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4)))

    items = collect_files()
    if not items:
        sys.exit("No training photos found under datasets/*/real_images. "
                 "Run scripts/fetch_datasets.py first.")
    train_items, val_items, test_items = grouped_split(items, seed=seed)
    if smoke:
        epochs = 1
        train_items = train_items[:60]; val_items = val_items[:20]; test_items = test_items[:20]

    n_classes = len(CLASS_NAMES)
    print(f"[deep vision] arch={arch} device={device} classes={n_classes}")
    print(f"  train {len(train_items)} | val {len(val_items)} | test {len(test_items)} images")
    print(f"  groups: train {len({i[2] for i in train_items})}, "
          f"val {len({i[2] for i in val_items})}, test {len({i[2] for i in test_items})} "
          f"(no photograph appears in more than one split)")

    train_ds, val_ds, test_ds = build_datasets(train_items, val_items, test_items, img_size)

    counts = np.bincount([i[1] for i in train_items], minlength=n_classes).astype(float)
    inv = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    sample_w = [inv[i[1]] for i in train_items]
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, num_samples=len(train_items), replacement=True)
    class_w = torch.tensor((inv / inv.sum() * n_classes) if inv.sum() else np.ones(n_classes), dtype=torch.float32, device=device)

    dl = lambda ds, **kw: torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=workers, **kw)
    train_loader = dl(train_ds, sampler=sampler)
    val_loader = dl(val_ds, shuffle=False)
    test_loader = dl(test_ds, shuffle=False)

    model, head_params, last_block = build_model(arch, n_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)

    set_requires_grad(model, False)
    for p in head_params:
        p.requires_grad = True
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    phase2_at = max(1, epochs // 3)
    history, best = [], {"macro_f1": -1.0, "epoch": -1, "state": None}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        if epoch == phase2_at + 1 and not smoke:
            print(f"  [phase 2] unfreezing the last backbone block at lr/10")
            set_requires_grad(last_block, True)
            optimiser = torch.optim.AdamW([
                {"params": [p for p in last_block.parameters()], "lr": lr / 10.0},
                {"params": head_params, "lr": lr},
            ], weight_decay=1e-4)

        model.train()
        running, seen, t_ep = 0.0, 0, time.time()
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            running += float(loss.item()) * y.numel(); seen += int(y.numel())
            if bi % 10 == 0:
                print(f"    epoch {epoch} batch {bi + 1}/{len(train_loader)} loss {running / max(seen, 1):.4f}", flush=True)

        val = evaluate(model, val_loader, device, n_classes)
        _rep, macro_f1 = per_class_report(val["confusion_matrix"], CLASS_NAMES)
        history.append({"epoch": epoch, "train_loss": round(running / max(seen, 1), 4),
                        "val_accuracy": round(val["accuracy"], 4), "val_macro_f1": round(macro_f1, 4),
                        "seconds": round(time.time() - t_ep, 1)})
        print(f"  epoch {epoch}/{epochs}: train loss {history[-1]['train_loss']} | "
              f"val acc {val['accuracy'] * 100:.1f}% | val macro-F1 {macro_f1:.3f} | {history[-1]['seconds']}s")

        if macro_f1 > best["macro_f1"]:
            best = {"macro_f1": macro_f1, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"  best epoch: {best['epoch']} (val macro-F1 {best['macro_f1']:.3f})")

    val_final = evaluate(model, val_loader, device, n_classes)
    test_final = evaluate(model, test_loader, device, n_classes)
    val_rep, val_f1 = per_class_report(val_final["confusion_matrix"], CLASS_NAMES)
    test_rep, test_f1 = per_class_report(test_final["confusion_matrix"], CLASS_NAMES)

    weights_path = os.path.join(save_dir, f"deep_vision_{arch}.pt")
    torch.save({"arch": arch, "state_dict": model.state_dict(), "class_names": CLASS_NAMES,
                "img_size": img_size, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}, weights_path)

    onnx_path = os.path.join(save_dir, f"deep_vision_{arch}.onnx")
    try:
        model.eval()
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        torch.onnx.export(model, dummy, onnx_path, input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=13)
        print(f"  exported ONNX -> {onnx_path}")
    except Exception as e:
        onnx_path = None
        print(f"  ONNX export skipped: {e}")

    report = {
        "model": f"DeepVisionNet ({arch}, ImageNet-pretrained, fine-tuned)",
        "class_names": CLASS_NAMES,
        "split": {
            "strategy": "grouped by source photograph, stratified per class; test set scored once",
            "train_images": len(train_items), "val_images": len(val_items), "test_images": len(test_items),
            "train_groups": len({i[2] for i in train_items}),
            "val_groups": len({i[2] for i in val_items}),
            "test_groups": len({i[2] for i in test_items}),
        },
        "training": {"epochs": epochs, "batch_size": batch_size, "img_size": img_size, "lr": lr,
                     "device": str(device), "seconds": round(time.time() - t0, 1), "history": history,
                     "best_epoch": best["epoch"]},
        "validation": {"accuracy": round(val_final["accuracy"], 4), "macro_f1": round(val_f1, 4),
                       "confusion_matrix": val_final["confusion_matrix"], "per_class": val_rep},
        "test": {"accuracy": round(test_final["accuracy"], 4), "macro_f1": round(test_f1, 4),
                 "confusion_matrix": test_final["confusion_matrix"], "per_class": test_rep},
        "random_guess_baseline": round(1.0 / n_classes, 4),
        "weights_path": weights_path, "onnx_path": onnx_path,
        "trained_at_unix": int(time.time()),
    }
    report_path = os.path.join(save_dir, "deep_vision_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"HELD-OUT TEST accuracy : {test_final['accuracy'] * 100:.1f}%   (random guess {100.0 / n_classes:.1f}%)")
    print(f"HELD-OUT TEST macro-F1 : {test_f1:.3f}")
    print("per class (test):")
    for name, m in test_rep.items():
        print(f"  {name[:44]:46s} P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}  n={m['support']}")
    print(f"\nweights -> {weights_path}")
    print(f"report  -> {report_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a pretrained CNN on the road photo dataset.")
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "resnet34", "efficientnet_b0", "mobilenet_v3_small"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers; keep 0 on Windows")
    ap.add_argument("--smoke", action="store_true", help="one tiny epoch, to check the setup works")
    args = ap.parse_args()
    run_training(arch=args.arch, epochs=args.epochs, batch_size=args.batch_size, img_size=args.img_size,
                 lr=args.lr, seed=args.seed, smoke=args.smoke, workers=args.workers)


if __name__ == "__main__":
    main()
