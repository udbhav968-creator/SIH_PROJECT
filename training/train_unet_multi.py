"""
Train the U-Net on DNIT plus the extra pixel-labelled datasets
(scripts/fetch_seg_datasets.py), so it learns outlines from many roads and
cameras instead of one.

    python -m training.train_unet_multi                    # T4: ~45-60 min
    python -m training.train_unet_multi --smoke            # 1 epoch, a few hundred images

Why
---
The first U-Net (training/train_unet_segmenter.py) learned from DNIT only. It won
mask IoU on DNIT's own test photographs (pothole 0.641 vs 0.144) and then found
only 9 of 24 defects through the full pipeline on other datasets. A model that
has seen one camera has learned that camera. This run keeps the same network,
loss and DNIT split, and adds outlines from other sources.

Same protocol as before, plus more test sets
--------------------------------------------
* DNIT: the identical split (seed 42), so the 500 DNIT test photographs are the
  same ones every segmenter in this project was scored on.
* Each extra source: 75/10/15 by image id (seeded); test images never trained on.
* Thresholds are tuned on the combined calibration images; the IoU selection
  against the pixel classifier uses DNIT calibration photographs exactly as before.
* Test IoU is reported per source, for the U-Net and (on a sample) the pixel
  classifier, and the exported ONNX file itself is scored on the DNIT test split.
* The result is NOT served by this script. segmenter_selection.json is written
  with the pixel classifier still serving and the IoU decision recorded;
  scripts/segmenter_deployment_check.py (the end-to-end check through
  audit_image on other datasets) decides, as it did for the first U-Net.
"""

import argparse
import json
import os
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import numpy as np

from models.defect_segmenter import IGNORE, WORK_H, WORK_W
from models.unet_segmenter import IN_H, IN_W, META_NAME, ONNX_NAME, SELECTION_NAME, UNetSegmenter
from training.train_unet_segmenter import (SEL_RULE, TorchUNet, build_unet, decide_segmenter, load_pair,
                                           make_dataset, quick_val_iou, score, split_like_pixel_classifier,
                                           tune_thresholds, cache_work_probs)

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
MULTI = os.path.join(ENGINE_ROOT, "datasets", "seg_multi")
PT_NAME = "defect_segmenter_unet.pt"
# Close-up texture patches with no road scene, horizon or camera geometry. They teach what a crack looks
# like, but the serving threshold is calibrated on what the bus camera sees (protocol v2).
TEXTURE_PATCH_SOURCES = {"crackseg9k"}
PROTOCOL_V2 = {
    "version": 2,
    "change": "per-class thresholds tuned only on calibration photographs of road scenes (DNIT, Kaggle pothole, "
              "Pothole Mix); CrackSeg9k still trains the network but no longer sets the serving threshold",
    "why": "in v1 the combined calibration was dominated by CrackSeg9k close-ups; the crack threshold it chose (0.85) "
           "fitted texture patches, and on DNIT calibration the U-Net's crack IoU was 0.184 against the pixel "
           "classifier's 0.214, so the IoU rule kept the pixel classifier",
    "disclosure": "this change was decided AFTER seeing v1's results, including its test scores. The selection rule, "
                  "the DNIT split and the end-to-end check are unchanged, and v1's result stays on record "
                  "(previous_run). Read v2's test numbers with that in mind.",
}
# share of each epoch drawn from each source (renormalised over the sources present)
SHARES = {"dnit": 0.35, "dnit_clean": 0.10, "crackseg9k": 0.25, "pothole_mix": 0.20, "kaggle_pothole": 0.10}


class DiskPairs:
    """Lazy (img, label) pairs from datasets/seg_multi, read when indexed."""

    def __init__(self, source, ids):
        self.source, self.ids = source, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import cv2
        d = os.path.join(MULTI, self.source)
        img = cv2.cvtColor(cv2.imread(os.path.join(d, "img", self.ids[i] + ".jpg")), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(os.path.join(d, "lab", self.ids[i] + ".png"), cv2.IMREAD_GRAYSCALE)
        return img, lab


class ConcatPairs:
    def __init__(self, parts):
        self.parts = [p for p in parts if len(p)]
        self.offsets = np.cumsum([0] + [len(p) for p in self.parts])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, i):
        k = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return self.parts[k][i - int(self.offsets[k])]


def load_manifest():
    path = os.path.join(MULTI, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        man = json.load(fh)
    out = {}
    for src, items in (man.get("items") or {}).items():
        out[src] = {s: [it["id"] for it in items if it["split"] == s] for s in ("train", "cal", "test")}
    return out


def drop_dnit_eval_copies(extra, sp, threshold=6):
    """Remove (in place) extra TRAINING ids that near-duplicate a DNIT calibration/test photograph."""
    import cv2
    from scripts.fetch_seg_datasets import dhash
    ref = []
    for r in sp["cal"] + sp["test"] + sp.get("neg_cal", []) + sp.get("neg_test", []):
        im = cv2.imread(r["path"])
        if im is not None:
            ref.append(dhash(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    ref = np.array(ref, dtype=np.uint64)
    removed = {}
    for src, d in extra.items():
        keep = []
        for i in d["train"]:
            im = cv2.imread(os.path.join(MULTI, src, "img", i + ".jpg"))
            if im is not None and len(ref):
                x = np.bitwise_xor(ref, np.uint64(dhash(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))))
                if np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1).min() <= threshold:
                    continue
            keep.append(i)
        removed[src] = len(d["train"]) - len(keep)
        d["train"] = keep
    return removed


def work_pairs(seg, pairs, limit=None):
    """[(proba on the 320x200 grid, truth on the same grid)] for threshold tuning."""
    import cv2
    out = []
    for i in range(min(len(pairs), limit or len(pairs))):
        img, lab = pairs[i]
        out.append((seg.predict_work(img), cv2.resize(lab, (WORK_W, WORK_H), interpolation=cv2.INTER_NEAREST)))
    return out


def score_pairs(seg, pairs, limit=None):
    """Whole-mask IoU on the 320x200 grid, the same functions train_segmenter scores with."""
    import cv2
    from training.train_segmenter import accumulate, finalise, score_masks
    total = {}
    for i in range(min(len(pairs), limit or len(pairs))):
        img, lab = pairs[i]
        truth = cv2.resize(lab, (WORK_W, WORK_H), interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(seg.segment(img)["mask"], (WORK_W, WORK_H), interpolation=cv2.INTER_NEAREST)
        accumulate(total, score_masks(truth, pred))
    return finalise(total)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples-per-epoch", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pixel-sample", type=int, default=100,
                    help="test images per extra source to score the pixel classifier on (it is slow on CPU)")
    ap.add_argument("--threshold-sources", default="road",
                    help="'road' (protocol v2, default): tune the per-class thresholds only on calibration photographs "
                         "of road scenes (DNIT, Kaggle pothole, Pothole Mix), not on CrackSeg9k's close-up texture "
                         "patches; 'all' (protocol v1): every source")
    ap.add_argument("--out", default=CKPT_DIR)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    import cv2
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = split_like_pixel_classifier(2000, 42)
    extra = load_manifest()
    if a.smoke:
        for k in sp:
            sp[k] = sp[k][: {"train": 40, "cal": 10, "test": 10}.get(k, 6)]
        extra = {s: {k: v[:30] for k, v in d.items()} for s, d in extra.items()}
        a.epochs, a.samples_per_epoch, a.pixel_sample = 1, 120, 5

    t0 = time.time()
    dnit_train = [p for p in (load_pair(r) for r in sp["train"]) if p is not None]
    dnit_clean = [p for p in (load_pair(r) for r in sp["neg_train"]) if p is not None]
    dnit_cal = [p for p in (load_pair(r) for r in sp["cal"]) if p is not None]
    # Pothole Mix republishes DNIT photographs. Any extra training image that is a near-copy of a DNIT
    # calibration or test photograph is dropped here, so the IoU rule and the DNIT test stay unseen.
    overlap = drop_dnit_eval_copies(extra, sp)
    for src, n in overlap.items():
        if n:
            print(f"  removed {n} {src} training images that are near-copies of DNIT calibration/test photographs")
    parts = {"dnit": dnit_train, "dnit_clean": dnit_clean}
    for src, d in extra.items():
        parts[src] = DiskPairs(src, d["train"])
    print(f"[unet-multi] device {device} | training images: " +
          ", ".join(f"{k} {len(v)}" for k, v in parts.items()) +
          f" | DNIT test {len(sp['test'])} (same split as every segmenter here) | loaded in {time.time() - t0:.0f}s")
    if not extra:
        print("  [!] no extra datasets found (run scripts/fetch_seg_datasets.py); training on DNIT only")

    # class weights from DNIT plus a sample of each extra source
    counts = np.zeros(3)
    for src, pairs in parts.items():
        for i in range(0, len(pairs), max(1, len(pairs) // 300)):
            lab = pairs[i][1]
            for c in range(3):
                counts[c] += int((lab == c).sum())
    freq = counts / max(counts.sum(), 1)
    weights = np.minimum(np.sqrt(freq[0] / np.maximum(freq, 1e-9)), 10.0)
    weights[0] = 1.0
    print(f"  pixel shares sound/crack/pothole {np.round(freq, 4).tolist()} -> CE weights {np.round(weights, 2).tolist()}")

    all_pairs = ConcatPairs(list(parts.values()))
    present = {k: SHARES.get(k, 0.1) for k, v in parts.items() if len(v)}
    norm = sum(present.values())
    w = np.concatenate([np.full(len(v), present[k] / norm / len(v)) for k, v in parts.items() if len(v)])
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=a.samples_per_epoch, replacement=True)
    loader = torch.utils.data.DataLoader(make_dataset(all_pairs, True), batch_size=a.batch, sampler=sampler,
                                         num_workers=a.workers, drop_last=True, pin_memory=device.type == "cuda")

    model = build_unet(pretrained=not a.smoke).to(device)
    enc = [p for n, p in model.named_parameters() if n.split(".")[0] in ("stem", "l1", "l2", "l3", "l4")]
    dec = [p for n, p in model.named_parameters() if n.split(".")[0] not in ("stem", "l1", "l2", "l3", "l4")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": dec, "lr": a.lr * 3}], weight_decay=1e-4)
    steps = max(1, a.epochs * len(loader))
    warm = max(1, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce_w = torch.tensor(weights, dtype=torch.float32, device=device)

    def loss_fn(logits, y):
        ce = F.cross_entropy(logits, y, weight=ce_w, ignore_index=IGNORE)
        p = torch.softmax(logits.float(), 1)
        valid = (y != IGNORE).float()
        yv = torch.where(y == IGNORE, torch.zeros_like(y), y)
        dice = 0.0
        for c in (1, 2):
            t = (yv == c).float() * valid
            pc = p[:, c] * valid
            dice = dice + 1 - (2 * (pc * t).sum() + 1) / (pc.sum() + t.sum() + 1)
        return ce + 0.5 * dice

    # validation every epoch: DNIT calibration + up to 150 calibration images per extra source
    val_pairs = list(dnit_cal)
    for src, d in extra.items():
        dp = DiskPairs(src, d["cal"])
        val_pairs += [dp[i] for i in range(min(150, len(dp)))]

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t_ep, run = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss.detach())
        val, (vc, vp) = quick_val_iou(model, device, val_pairs)
        history.append({"epoch": ep, "loss": round(run / max(1, len(loader)), 4),
                        "val_iou_crack": round(vc, 4), "val_iou_pothole": round(vp, 4)})
        flag = ""
        if val > best:
            best, bad, flag = val, 0, " *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  epoch {ep:3d}/{a.epochs}  loss {run / max(1, len(loader)):.4f}  VAL IoU crack {vc:.3f} "
              f"pothole {vp:.3f}  ({time.time() - t_ep:.0f}s){flag}", flush=True)
        if bad >= a.patience:
            print(f"  early stop at epoch {ep}")
            break
    model.load_state_dict(best_state)
    model.eval()

    # thresholds on DNIT calibration + extra calibration
    unet = TorchUNet(model, device)
    thr_sources = ["dnit"] + [src for src in extra
                              if a.threshold_sources == "all" or src not in TEXTURE_PATCH_SOURCES]
    cal_cached = cache_work_probs(unet, sp["cal"] + sp["neg_cal"][: max(1, len(sp["cal"]) // 2)])
    for src, d in extra.items():
        if src in thr_sources:
            cal_cached += work_pairs(unet, DiskPairs(src, d["cal"]), limit=300)
    unet.thresholds = tune_thresholds(cal_cached)
    print(f"  thresholds (calibration photographs from {', '.join(thr_sources)}): {unet.thresholds}")

    # IoU selection on DNIT calibration, exactly as the first U-Net
    from models.defect_segmenter import DefectSegmenter
    from training.train_segmenter import clean_false_positive_rate
    pixel = DefectSegmenter(os.path.join(a.out, "defect_segmenter.joblib"))
    val = {"unet": score(unet, sp["cal"])}
    val["unet"]["clean_false_blob_rate"] = (clean_false_positive_rate(unet, sp["neg_cal"]) or {}).get(
        "photo_rate_any_blob") if sp["neg_cal"] else None
    if pixel.is_ready:
        val["pixel_classifier"] = score(pixel, sp["cal"])
        val["pixel_classifier"]["clean_false_blob_rate"] = (clean_false_positive_rate(pixel, sp["neg_cal"]) or {}).get(
            "photo_rate_any_blob") if sp["neg_cal"] else None
    served_by_iou, why = decide_segmenter(val["unet"], val.get("pixel_classifier"), smoke=a.smoke)
    print(f"  IoU SELECTION -> {served_by_iou}: {why}")

    def iou(d, c):
        return ((d or {}).get(c) or {}).get("iou") or 0.0

    # test, once: DNIT + every extra source
    test = {"dnit": {"unet": score(unet, sp["test"])}}
    test["dnit"]["unet"]["clean"] = clean_false_positive_rate(unet, sp["neg_test"]) if sp["neg_test"] else None
    rp = os.path.join(a.out, "defect_segmenter_report.json")
    if os.path.exists(rp):
        with open(rp, "r", encoding="utf-8") as fh:
            prep = json.load(fh)
        test["dnit"]["pixel_classifier"] = {**(prep.get("iou") or {}), "source": "defect_segmenter_report.json"}
    for src, d in extra.items():
        dp = DiskPairs(src, d["test"])
        test[src] = {"unet": score_pairs(unet, dp), "test_images": len(dp)}
        if pixel.is_ready and a.pixel_sample:
            test[src]["pixel_classifier"] = score_pairs(pixel, dp, limit=a.pixel_sample)
            test[src]["pixel_classifier"]["scored_on_first_n"] = min(a.pixel_sample, len(dp))
    for src, t in test.items():
        u, p = t.get("unet") or {}, t.get("pixel_classifier") or {}
        print(f"  TEST {src:15s} U-Net crack {iou(u, 'crack'):.3f} pothole {iou(u, 'pothole'):.3f}   |   "
              f"pixel classifier crack {iou(p, 'crack'):.3f} pothole {iou(p, 'pothole'):.3f}")

    # export, then check the exported file reproduces the network on real photographs
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, PT_NAME))
    onnx_path = os.path.join(a.out, ONNX_NAME)

    class WithSoftmax(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.softmax(self.m(x), 1)

    cpu_model = WithSoftmax(model.float().cpu().eval())
    dummy = torch.zeros(1, 3, IN_H, IN_W)
    try:
        torch.onnx.export(cpu_model, dummy, onnx_path, input_names=["image"], output_names=["probs"],
                          opset_version=17, dynamo=False)
    except Exception as e:
        print(f"  [onnx] legacy export failed ({e}); trying the dynamo exporter")
        torch.onnx.export(cpu_model, (dummy,), onnx_path, input_names=["image"], output_names=["probs"],
                          opset_version=17, dynamo=True)
    onnx_seg = UNetSegmenter(a.out)
    onnx_seg.thresholds, onnx_seg.tta_flip = dict(unet.thresholds), True
    onnx_test = score(onnx_seg, sp["test"]) if onnx_seg.is_ready else None
    onnx_ok = bool(onnx_test) and abs(iou(onnx_test, "crack") - iou(test["dnit"]["unet"], "crack")) <= 0.03 and \
        abs(iou(onnx_test, "pothole") - iou(test["dnit"]["unet"], "pothole")) <= 0.03
    print(f"  exported {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB); ONNX on DNIT test: crack "
          f"{iou(onnx_test, 'crack'):.3f} pothole {iou(onnx_test, 'pothole'):.3f} -> "
          f"{'reproduces the network' if onnx_ok else 'DOES NOT reproduce the network'}")
    if served_by_iou == "unet" and not onnx_ok:
        served_by_iou, why = "pixel_classifier", why + "; the exported ONNX file did not reproduce the network"
    if onnx_ok:
        test["dnit"]["unet_torch_gpu"] = test["dnit"]["unet"]
        test["dnit"]["unet"] = {**onnx_test, "clean": test["dnit"]["unet"].get("clean"),
                                "source": "exported ONNX file on CPU (the served artefact)"}

    sources = {"dnit": {"train": len(dnit_train), "clean_train": len(dnit_clean), "test": len(sp["test"])}}
    for src, d in extra.items():
        sources[src] = {k: len(v) for k, v in d.items()}
    meta = {
        "model": "U-Net, ResNet-18 encoder pretrained on ImageNet, all layers trained on several datasets",
        "input_size": [IN_W, IN_H], "normalisation": "ImageNet mean/std", "tta_flip": True,
        "thresholds": unet.thresholds, "threshold_sources": thr_sources,
        "decision_rule": "per-class threshold tuned for IoU on the calibration photographs of " + ", ".join(thr_sources),
        "trained_on": {"sources": sources, "epoch_shares": present,
                       "removed_near_copies_of_dnit_eval": overlap,
                       "test_photographs": len(sp["test"]),
                       "split": "DNIT identical to training/train_segmenter.py; extra sources 75/10/15 by image id",
                       "leak_guard": "extra images near-duplicate to any measurement photograph were dropped "
                                     "(scripts/fetch_seg_datasets.py)"},
        "iou": {k: v for k, v in test["dnit"]["unet"].items() if k in ("crack", "pothole")},
        "iou_per_source": {s: {k: v for k, v in (t.get("unet") or {}).items() if k in ("crack", "pothole")}
                           for s, t in test.items()},
        "false_positives_on_clean_roads": test["dnit"]["unet"].get("clean"),
        "onnx_verified_on_test": onnx_ok, "epochs_run": len(history), "history": history,
        "smoke": a.smoke, "trained_at_unix": int(time.time()),
    }
    with open(os.path.join(a.out, META_NAME), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=float)
    previous = None
    prev_path = os.path.join(a.out, SELECTION_NAME)
    if os.path.exists(prev_path):
        try:
            with open(prev_path, "r", encoding="utf-8") as fh:
                old_sel = json.load(fh)
            previous = {k: old_sel.get(k) for k in ("served", "iou_selection_served", "why", "test",
                                                    "deployment_check", "trained_with", "protocol", "decided_unix")
                        if k in old_sel}
        except Exception:
            previous = None
    selection = {
        "protocol": PROTOCOL_V2 if a.threshold_sources == "road" else {"version": 1, "change": "thresholds on every source"},
        "previous_run": previous,
        "served": "pixel_classifier",                         # until the end-to-end check says otherwise
        "iou_selection_served": served_by_iou, "rule": SEL_RULE, "why_iou": why,
        "why": why + "; awaiting the end-to-end check (python -m scripts.segmenter_deployment_check)",
        "validation": val, "test": {"unet": test["dnit"]["unet"],
                                    "pixel_classifier": test["dnit"].get("pixel_classifier")},
        "test_per_source": test, "trained_with": "training/train_unet_multi.py",
        "decided_before_test": True, "smoke": a.smoke, "decided_unix": int(time.time()),
    }
    with open(os.path.join(a.out, SELECTION_NAME), "w", encoding="utf-8") as fh:
        json.dump(selection, fh, indent=2, default=float)
    print(f"  wrote {META_NAME} and {SELECTION_NAME} (pixel classifier serves until the end-to-end check)")
    return selection


if __name__ == "__main__":
    main()
