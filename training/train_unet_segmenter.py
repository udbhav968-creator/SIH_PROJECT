"""
Train a U-Net (ResNet-18 ImageNet encoder) on the DNIT defect polygons, and
decide - by a rule fixed before the test set is scored - whether it replaces
the pixel classifier.

    python -m training.train_unet_segmenter                 # GPU, ~20-30 min on a T4
    python -m training.train_unet_segmenter --smoke         # 2 epochs, 60 photographs

Same data, same split, same scoring as training/train_segmenter.py
------------------------------------------------------------------
* The photographs are split by the identical procedure (seed 42 shuffle of the
  same annotation list, 2,000 photographs, 25% test, 12% calibration), and the
  clean-road photographs by the identical source-scene split. So the test
  photographs here are the same 500 the pixel classifier was scored on.
* Labels come from the same rasteriser: crack / pothole / sound inside the lane
  polygon, everything outside ignored.
* Scoring uses the same functions (score_masks, finalise,
  clean_false_positive_rate) on the same 320x200 grid, and the mask is produced
  by the same post-processing (models.defect_segmenter.mask_from_proba) with
  per-class thresholds tuned for IoU on the calibration split.

Selection rule (written before any test score exists)
-----------------------------------------------------
On the CALIBRATION photographs - never the test set - both segmenters are run.
The U-Net is served only if
    crack IoU  > pixel classifier's crack IoU, and
    pothole IoU > pixel classifier's pothole IoU, and
    its clean-road false-blob rate is no more than 5 points worse.
Then both are scored once on the test photographs and the clean test
photographs, and both numbers are reported whichever one won.

Outputs
-------
    checkpoints/defect_segmenter_unet.onnx      the network (softmax output)
    checkpoints/defect_segmenter_unet.json      thresholds, input size, report
    checkpoints/segmenter_selection.json        which segmenter is served and why
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

from models.defect_segmenter import CLASS_CRACK, CLASS_POTHOLE, CLASS_SOUND, IGNORE
from models.unet_segmenter import (IN_H, IN_W, MEAN, META_NAME, ONNX_NAME, SELECTION_NAME, STD,
                                   UNetSegmenter, _UNetBase, mask_from_proba, probs_to_work)

PT_NAME = "defect_segmenter_unet.pt"
from models.defect_segmenter import WORK_H, WORK_W

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
SEL_RULE = ("U-Net served only if, on the calibration photographs, its crack IoU AND pothole IoU "
            "both exceed the pixel classifier's, and its clean-road false-blob rate is no more "
            "than 5 points worse. Decided before the test set was scored.")


# ---------------------------------------------------------------------------
# split: identical to training/train_segmenter.py
# ---------------------------------------------------------------------------
def split_like_pixel_classifier(images=2000, seed=42, test_fraction=0.25):
    from training.train_segmenter import load_annotations, load_negatives
    records = load_annotations()
    rng = np.random.default_rng(seed)
    rng.shuffle(records)
    records = records[:images]
    n_test = max(20, int(len(records) * test_fraction))
    n_cal = max(15, int(len(records) * 0.12))
    out = {"test": records[:n_test], "cal": records[n_test:n_test + n_cal],
           "train": records[n_test + n_cal:]}
    negatives = load_negatives()
    out["neg_train"] = out["neg_cal"] = out["neg_test"] = []
    if negatives:
        groups = sorted({r["group"] for r in negatives})
        rng.shuffle(groups)
        n_g = len(groups)
        g_test = set(groups[: max(1, int(n_g * 0.30))])
        g_cal = set(groups[max(1, int(n_g * 0.30)): max(2, int(n_g * 0.45))])
        out["neg_test"] = [r for r in negatives if r["group"] in g_test]
        out["neg_cal"] = [r for r in negatives if r["group"] in g_cal]
        out["neg_train"] = [r for r in negatives if r["group"] not in g_test and r["group"] not in g_cal]
    return out


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
def build_unet(n_classes=3, pretrained=True):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True))

    class UNetR18(nn.Module):
        def __init__(self):
            super().__init__()
            w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            r = torchvision.models.resnet18(weights=w)
            self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)        # /2, 64
            self.pool = r.maxpool                                    # /4
            self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
            self.d4 = block(512 + 256, 256)
            self.d3 = block(256 + 128, 128)
            self.d2 = block(128 + 64, 64)
            self.d1 = block(64 + 64, 64)
            self.d0 = block(64, 32)
            self.head = nn.Conv2d(32, n_classes, 1)

        @staticmethod
        def up(x):
            return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

        def forward(self, x):
            s0 = self.stem(x)
            e1 = self.l1(self.pool(s0))
            e2 = self.l2(e1)
            e3 = self.l3(e2)
            e4 = self.l4(e3)
            d = self.d4(torch.cat([self.up(e4), e3], 1))
            d = self.d3(torch.cat([self.up(d), e2], 1))
            d = self.d2(torch.cat([self.up(d), e1], 1))
            d = self.d1(torch.cat([self.up(d), s0], 1))
            d = self.d0(self.up(d))
            return self.head(d)

    return UNetR18()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_pair(rec):
    """(IN_H, IN_W, 3) uint8 image and (IN_H, IN_W) uint8 label, or None."""
    import cv2
    from training.train_segmenter import rasterise
    raw = cv2.imread(rec["path"])
    if raw is None:
        return None
    img = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    return img, rasterise(rec, IN_W, IN_H)


class SegData:
    """Map-style dataset (module level, so DataLoader workers can pickle it under any start method)."""

    def __init__(self, pairs, train):
        self.pairs, self.train = pairs, train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        import cv2
        import torch
        img, lab = self.pairs[i]
        if self.train:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
            s = rng.uniform(0.75, 1.0)
            if s < 0.999:
                ch, cw = int(IN_H * s), int(IN_W * s)
                y0 = int(rng.integers(0, IN_H - ch + 1))
                x0 = int(rng.integers(0, IN_W - cw + 1))
                img = cv2.resize(np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw]), (IN_W, IN_H),
                                 interpolation=cv2.INTER_LINEAR)
                lab = cv2.resize(np.ascontiguousarray(lab[y0:y0 + ch, x0:x0 + cw]), (IN_W, IN_H),
                                 interpolation=cv2.INTER_NEAREST)
            f = img.astype(np.float32)
            m = f.mean()
            img = np.clip((f - m) * rng.uniform(0.8, 1.2) + m + rng.uniform(-20, 20), 0, 255)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(lab).astype(np.int64)))


def make_dataset(pairs, train):
    return SegData(pairs, train)


# ---------------------------------------------------------------------------
# a torch model behind the serving interface, so scoring runs the served code
# ---------------------------------------------------------------------------
class TorchUNet(_UNetBase):
    def __init__(self, model, device, thresholds=None, tta_flip=True):
        self.model, self.device = model, device
        self.thresholds = dict(thresholds or {"crack": 0.5, "pothole": 0.5})
        self.tta_flip = tta_flip
        self.model_path = "(in memory)"

    @property
    def is_ready(self):
        return True

    def _forward(self, x):
        import torch
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            p = torch.softmax(self.model(torch.from_numpy(np.ascontiguousarray(x)).to(self.device)), 1)
        return p.float().cpu().numpy()[0]


def cache_work_probs(seg, recs):
    """[(proba (N,3), truth (WORK_H, WORK_W))] on the shared 320x200 grid."""
    import cv2
    from training.train_segmenter import rasterise
    out = []
    for rec in recs:
        raw = cv2.imread(rec["path"])
        if raw is None:
            continue
        out.append((seg.predict_work(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)), rasterise(rec)))
    return out


def tune_thresholds(cached, grid=None):
    """Per-class threshold maximising IoU on the calibration split; ties go to the stricter one."""
    grid = grid if grid is not None else np.round(np.arange(0.10, 0.91, 0.05), 2)
    best = {}
    for name, cid in (("crack", CLASS_CRACK), ("pothole", CLASS_POTHOLE)):
        curve = []
        for t in grid:
            inter = union = 0
            for proba, truth in cached:
                valid = truth != IGNORE
                pred = (proba[:, cid] >= t).reshape(truth.shape) & valid
                tm = (truth == cid) & valid
                inter += int((tm & pred).sum())
                union += int((tm | pred).sum())
            curve.append((float(t), inter / union if union else 0.0))
        peak = max(i for _t, i in curve)
        tol = max(0.002, 0.03 * peak)
        best[name] = max((t, i) for t, i in curve if i >= peak - tol)[0]
    return best


def score(seg, recs):
    """Whole-mask IoU on the 320x200 grid, exactly as train_segmenter scores."""
    import cv2
    from training.train_segmenter import accumulate, finalise, rasterise, score_masks
    total = {}
    for rec in recs:
        raw = cv2.imread(rec["path"])
        if raw is None:
            continue
        truth = rasterise(rec)
        pred = cv2.resize(seg.segment(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))["mask"],
                          (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_NEAREST)
        accumulate(total, score_masks(truth, pred))
    return finalise(total)


def quick_val_iou(model, device, pairs):
    """Mean of crack and pothole IoU at threshold 0.5 on IN-resolution labels (epoch-level only)."""
    import torch
    inter = {1: 0, 2: 0}
    union = {1: 0, 2: 0}
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        for img, lab in pairs:
            x = (img.astype(np.float32) / 255.0 - MEAN) / STD
            p = torch.softmax(model(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(device)), 1)
            p = p.float().cpu().numpy()[0]
            valid = lab != IGNORE
            for c in (1, 2):
                pr = (p[c] >= 0.5) & valid
                tr = (lab == c) & valid
                inter[c] += int((pr & tr).sum())
                union[c] += int((pr | tr).sum())
    ious = [inter[c] / union[c] if union[c] else 0.0 for c in (1, 2)]
    return float(np.mean(ious)), ious


def decide_segmenter(unet_val, pixel_val, smoke=False):
    """
    The selection rule, applied to CALIBRATION-split scores only.

    U-Net serves iff its crack IoU and pothole IoU both beat the pixel
    classifier's and its clean-road false-blob rate is at most 5 points worse.
    Returns (served, why).
    """
    def iou(d, c):
        return ((d or {}).get(c) or {}).get("iou") or 0.0
    if not pixel_val:
        return "pixel_classifier", "the pixel classifier did not load, so no comparison was possible"
    u_fp, p_fp = unet_val.get("clean_false_blob_rate"), pixel_val.get("clean_false_blob_rate")
    fp_ok = u_fp is None or p_fp is None or u_fp <= p_fp + 0.05
    wins = iou(unet_val, "crack") > iou(pixel_val, "crack") and iou(unet_val, "pothole") > iou(pixel_val, "pothole") and fp_ok
    why = (f"calibration IoU crack {iou(unet_val, 'crack'):.3f} vs {iou(pixel_val, 'crack'):.3f}, pothole "
           f"{iou(unet_val, 'pothole'):.3f} vs {iou(pixel_val, 'pothole'):.3f}; clean false-blob rate "
           f"{u_fp} vs {p_fp}" + (" (smoke run: never served)" if smoke else ""))
    return ("unet" if wins and not smoke else "pixel_classifier"), why


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=int, default=2000, help="must match the pixel classifier's run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--neg-share", type=float, default=0.2,
                    help="share of each epoch drawn from clean-road photographs")
    ap.add_argument("--out", default=CKPT_DIR)
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 60 photographs, CPU-friendly")
    ap.add_argument("--rescore-pixel-test", action="store_true",
                    help="re-run the pixel classifier on the 500 test photographs (slow on CPU) instead "
                         "of quoting its own report, which was scored on this same split")
    a = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = split_like_pixel_classifier(a.images, a.seed)
    if a.smoke:
        for k in sp:
            sp[k] = sp[k][: {"train": 40, "cal": 10, "test": 10}.get(k, 6)]
        a.epochs, a.patience = 2, 5
    print(f"[unet] device {device} | train {len(sp['train'])} + clean {len(sp['neg_train'])} | "
          f"calibration {len(sp['cal'])} + clean {len(sp['neg_cal'])} | "
          f"test {len(sp['test'])} + clean {len(sp['neg_test'])} (same split as the pixel classifier)")

    t0 = time.time()
    train_pairs = [p for p in (load_pair(r) for r in sp["train"]) if p is not None]
    neg_pairs = [p for p in (load_pair(r) for r in sp["neg_train"]) if p is not None]
    cal_pairs = [p for p in (load_pair(r) for r in sp["cal"]) if p is not None]
    print(f"  loaded {len(train_pairs)} + {len(neg_pairs)} training photographs in {time.time() - t0:.0f}s")

    # class weights from the label frequencies (sqrt-inverse, capped)
    counts = np.zeros(3)
    for _img, lab in train_pairs:
        for c in range(3):
            counts[c] += int((lab == c).sum())
    freq = counts / max(counts.sum(), 1)
    weights = np.minimum(np.sqrt(freq[0] / np.maximum(freq, 1e-9)), 10.0)
    weights[0] = 1.0
    print(f"  pixel shares sound/crack/pothole {np.round(freq, 4).tolist()} -> CE weights "
          f"{np.round(weights, 2).tolist()}")

    all_pairs = train_pairs + neg_pairs
    w = np.ones(len(all_pairs))
    if neg_pairs:
        w[len(train_pairs):] = (a.neg_share / (1 - a.neg_share)) * len(train_pairs) / len(neg_pairs)
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(all_pairs), replacement=True)
    loader = torch.utils.data.DataLoader(make_dataset(all_pairs, True), batch_size=a.batch, sampler=sampler,
                                         num_workers=a.workers, drop_last=True,
                                         pin_memory=device.type == "cuda")

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
            run += float(loss)
        val, (vc, vp) = quick_val_iou(model, device, cal_pairs)
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
            print(f"  early stop at epoch {ep} (best VAL mean IoU {best:.3f})")
            break
    model.load_state_dict(best_state)
    model.eval()

    # ---------------- thresholds on the calibration split ----------------
    unet = TorchUNet(model, device)
    cal_cached = cache_work_probs(unet, sp["cal"] + sp["neg_cal"][: max(1, len(sp["cal"]) // 2)])
    unet.thresholds = tune_thresholds(cal_cached)
    print(f"  thresholds (calibration split): {unet.thresholds}")

    # ---------------- selection on the calibration split ----------------
    from training.train_segmenter import clean_false_positive_rate
    from models.defect_segmenter import DefectSegmenter
    pixel = DefectSegmenter(os.path.join(a.out, "defect_segmenter.joblib"))
    val = {"unet": score(unet, sp["cal"])}
    val["unet"]["clean_false_blob_rate"] = (clean_false_positive_rate(unet, sp["neg_cal"]) or {}).get(
        "photo_rate_any_blob") if sp["neg_cal"] else None
    if pixel.is_ready:
        val["pixel_classifier"] = score(pixel, sp["cal"])
        val["pixel_classifier"]["clean_false_blob_rate"] = (clean_false_positive_rate(pixel, sp["neg_cal"]) or {}).get(
            "photo_rate_any_blob") if sp["neg_cal"] else None

    def iou(d, c):
        return (d.get(c) or {}).get("iou") or 0.0

    served, why = decide_segmenter(val["unet"], val.get("pixel_classifier"), smoke=a.smoke)
    print(f"  SELECTION -> {served}: {why}")

    # ---------------- test, scored once, both models ----------------
    t_test = time.time()
    test = {"unet": score(unet, sp["test"])}
    ms = 1000 * (time.time() - t_test) / max(1, len(sp["test"]))
    test["unet"]["clean"] = clean_false_positive_rate(unet, sp["neg_test"]) if sp["neg_test"] else None
    if pixel.is_ready and a.rescore_pixel_test:
        test["pixel_classifier"] = score(pixel, sp["test"])
        test["pixel_classifier"]["clean"] = clean_false_positive_rate(pixel, sp["neg_test"]) if sp["neg_test"] else None
    else:
        rp = os.path.join(a.out, "defect_segmenter_report.json")
        if os.path.exists(rp):
            with open(rp, "r", encoding="utf-8") as fh:
                prep = json.load(fh)
            test["pixel_classifier"] = {**(prep.get("iou") or {}),
                                        "clean": prep.get("false_positives_on_clean_roads"),
                                        "source": "defect_segmenter_report.json (same split, scored at training)"}
    for k, v in test.items():
        print(f"  TEST {k:16s} crack IoU {iou(v, 'crack'):.4f}  pothole IoU {iou(v, 'pothole'):.4f}  "
              f"clean false blobs {((v.get('clean') or {}).get('photo_rate_any_blob'))}")

    # ---------------- export ----------------
    os.makedirs(a.out, exist_ok=True)
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
    except Exception as e:                       # legacy exporter unavailable -> dynamo exporter
        print(f"  [onnx] legacy export failed ({e}); trying the dynamo exporter")
        torch.onnx.export(cpu_model, (dummy,), onnx_path, input_names=["image"], output_names=["probs"],
                          opset_version=17, dynamo=True)
    # the weights too, so the network can be re-exported without retraining
    torch.save(model.state_dict(), os.path.join(a.out, PT_NAME))

    # Parity on REAL calibration photographs, through the served preprocessing.
    # (Random noise is a poor check: a segmentation net can saturate on it.)
    parity = None
    try:
        import cv2
        onnx_seg = UNetSegmenter(a.out)              # loads the ONNX file just written
        onnx_seg.thresholds, onnx_seg.tta_flip = dict(unet.thresholds), True
        ref_seg = TorchUNet(cpu_model.m, torch.device("cpu"), unet.thresholds)
        diffs = []
        for rec in sp["cal"][:8]:
            raw = cv2.imread(rec["path"])
            if raw is not None:
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                diffs.append(float(np.abs(onnx_seg.predict_work(rgb) - ref_seg.predict_work(rgb)).max()))
        parity = max(diffs) if diffs else None
    except Exception as e:
        onnx_seg = None
        print(f"  [warn] ONNX parity check failed: {e}")
    size_mb = round(os.path.getsize(onnx_path) / 1e6, 1)
    print(f"  exported {onnx_path} ({size_mb} MB, max |onnx - torch| on 8 real photographs = {parity})")

    # Score the artefact that will actually be served (ONNX, CPU) on the test split.
    onnx_test, cpu_ms = None, None
    if onnx_seg is not None and onnx_seg.is_ready:
        t_cpu = time.time()
        onnx_test = score(onnx_seg, sp["test"])
        cpu_ms = 1000 * (time.time() - t_cpu) / max(1, len(sp["test"]))
        onnx_test["clean"] = clean_false_positive_rate(onnx_seg, sp["neg_test"]) if sp["neg_test"] else None
        print(f"  TEST unet (ONNX, served) crack IoU {iou(onnx_test, 'crack'):.4f}  pothole IoU "
              f"{iou(onnx_test, 'pothole'):.4f}  clean false blobs "
              f"{(onnx_test.get('clean') or {}).get('photo_rate_any_blob')}  ({cpu_ms:.0f} ms/photo CPU)")
    onnx_ok = (onnx_test is not None
               and abs(iou(onnx_test, "crack") - iou(test["unet"], "crack")) <= 0.03
               and abs(iou(onnx_test, "pothole") - iou(test["unet"], "pothole")) <= 0.03)
    if served == "unet" and not onnx_ok:
        served = "pixel_classifier"
        why += "; NOT served: the exported ONNX file did not reproduce the network's test IoU"
        print("  [!] ONNX file disagrees with the trained network - keeping the pixel classifier")
    if onnx_ok:
        test["unet_torch_gpu"] = test["unet"]
        test["unet"] = {**onnx_test, "source": "exported ONNX file on CPU (the served artefact)"}

    trained_on = {"train_photographs": len(train_pairs), "clean_train_photographs": len(neg_pairs),
                  "calibration_photographs": len(sp["cal"]), "test_photographs": len(sp["test"]),
                  "clean_test_photographs": len(sp["neg_test"]),
                  "split": "identical to training/train_segmenter.py (seed 42, by photograph; clean "
                           "photographs by source scene)",
                  "source": "DNIT Cracks and Potholes in Road Images - hand-drawn polygons"}
    meta = {
        "model": "U-Net, ResNet-18 encoder pretrained on ImageNet, all layers trained",
        "input_size": [IN_W, IN_H], "normalisation": "ImageNet mean/std", "tta_flip": True,
        "thresholds": unet.thresholds,
        "decision_rule": "per-class probability threshold tuned for IoU on the calibration split",
        "loss": "cross-entropy (sqrt-inverse class weights, ignore outside lane) + 0.5 x soft Dice",
        "optimizer": "AdamW, warmup + cosine, AMP", "epochs_run": len(history),
        "best_val_mean_iou_at_0.5": round(best, 4), "history": history,
        "trained_on": trained_on, "iou": {k: v for k, v in test["unet"].items() if k in ("crack", "pothole")},
        "false_positives_on_clean_roads": test["unet"].get("clean"),
        "onnx_size_mb": size_mb, "onnx_parity_max_abs_real_photos": parity,
        "onnx_verified_on_test": onnx_ok,
        "inference_ms_per_image_gpu": round(ms, 1),
        "inference_ms_per_image_cpu_onnx": round(cpu_ms, 1) if cpu_ms else None, "smoke": a.smoke,
        "trained_at_unix": int(time.time()),
    }
    with open(os.path.join(a.out, META_NAME), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    selection = {"served": served, "rule": SEL_RULE, "why": why, "validation": val, "test": test,
                 "decided_before_test": True, "smoke": a.smoke, "decided_unix": int(time.time())}
    with open(os.path.join(a.out, SELECTION_NAME), "w", encoding="utf-8") as fh:
        json.dump(selection, fh, indent=2, default=float)
    print(f"  wrote {META_NAME} and {SELECTION_NAME}")
    return selection


if __name__ == "__main__":
    main()
