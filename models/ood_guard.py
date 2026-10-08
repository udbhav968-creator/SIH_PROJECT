"""
Input guard: should the road models be trusted on this photograph at all?

Every model in the pipeline was trained on road photographs taken from a vehicle or by hand at
street level. Shown a selfie, a document, a night shot with nothing visible or a blurred frame, a
classifier still returns a class with a confidence, and the pipeline would price a repair for it.
The guard answers the question before that happens, in two parts:

  novelty     Mahalanobis distance of the photograph's CNN embedding (MobileNetV2, PCA to 64
              dimensions) from the training photographs. Needs no examples of what "not a road"
              looks like, so it also catches kinds of input nobody thought of.
  not-road    a logistic regression on the same 64 dimensions, trained with road photographs
              against a broad set of everyday photographs (outlier exposure). Sharper on the
              common cases: people, rooms, screens, animals.

  quality     brightness, contrast, sharpness and clipped highlights, with limits set from the
              training photographs, so a frame that is too dark or blurred is named as such
              rather than reported as "not a road".

The fitted model is stored as plain arrays (checkpoints/ood_guard.npz) and evaluated with NumPy
only, so it loads under any scikit-learn version. The measurement that set the thresholds is in
checkpoints/ood_guard_report.json; training/train_ood_guard.py rebuilds both.
"""
import hashlib
import json
import os

import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
MODEL_FILE = "ood_guard.npz"
REPORT_FILE = "ood_guard_report.json"
BACKBONE = "mobilenetv2"
QUALITY_KEYS = ("brightness", "contrast", "sharpness", "clipped_fraction")


def _gray_small(img_rgb, width=512):
    img = np.asarray(img_rgb)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    h, w = img.shape[:2]
    if w > width:
        import cv2
        img = cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)
    img = img[:, :, :3].astype(np.float32)
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def image_quality(img_rgb):
    """Brightness (0-255 mean), contrast (std), sharpness (variance of the Laplacian at 512 px wide)
    and the fraction of clipped-white pixels."""
    import cv2
    g = _gray_small(img_rgb)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return {
        "brightness": round(float(g.mean()), 2),
        "contrast": round(float(g.std()), 2),
        "sharpness": round(float(lap.var()), 2),
        "clipped_fraction": round(float((g >= 250).mean()), 4),
    }


def normalise_embedding(e):
    e = np.asarray(e, dtype=np.float64).ravel()
    return e / (np.linalg.norm(e) + 1e-9)


def _sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class OODGuard:
    def __init__(self, checkpoints_dir=None, embedder=None):
        self.ckpt_dir = checkpoints_dir or CKPT_DIR
        self.load_error = None
        self.params = None
        self.report = {}
        self.embedder = embedder
        path = os.path.join(self.ckpt_dir, MODEL_FILE)
        if not os.path.exists(path):
            self.load_error = f"{MODEL_FILE} not trained - run python -m training.train_ood_guard"
            return
        try:
            with np.load(path, allow_pickle=False) as z:
                self.params = {k: z[k] for k in z.files}
            meta = json.loads(str(self.params.pop("meta_json")))
            self.meta = meta
            self.thresholds = meta["thresholds"]
        except Exception as e:
            self.load_error = f"could not read {MODEL_FILE}: {e}"
            self.params = None
            return
        try:
            with open(os.path.join(self.ckpt_dir, REPORT_FILE), encoding="utf-8") as fh:
                self.report = json.load(fh)
        except Exception:
            self.report = {}
        if self.embedder is None:
            from models.cnn_embedder import CNNEmbedder
            self.embedder = CNNEmbedder(checkpoints_dir=self.ckpt_dir, prefer=(BACKBONE,))
        if not self.embedder.is_ready or self.embedder.name != BACKBONE:
            self.load_error = f"the {BACKBONE} backbone the guard was fitted on is not loaded"
            self.params = None
            return
        backbone_path = os.path.join(self.ckpt_dir, "cnn_backbone_mobilenetv2.onnx")
        want = meta.get("backbone_sha16")
        if want and os.path.exists(backbone_path) and _sha16(backbone_path) != want:
            self.load_error = "the backbone file differs from the one the guard was fitted on - retrain the guard"
            self.params = None

    @property
    def is_ready(self):
        return self.params is not None

    # -- scoring on embeddings (used by training and serving alike) ---------------------------
    def project(self, emb):
        p = self.params
        x = np.atleast_2d(np.asarray(emb, dtype=np.float64))
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        return (x - p["pca_mean"]) @ p["pca_components"].T

    def scores(self, emb):
        """(mahalanobis distance, probability the photo is not a road scene) per row."""
        p = self.params
        z = self.project(emb)
        d = z - p["maha_mean"]
        maha = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, p["maha_precision"], d), 0.0))
        zs = (z - p["lr_mean"]) / p["lr_scale"]
        logit = zs @ p["lr_coef"] + float(p["lr_intercept"])
        return maha, 1.0 / (1.0 + np.exp(-logit))

    def quality_flags(self, q):
        t = self.thresholds
        out = []
        if q["brightness"] < t["brightness_min"]:
            out.append("too_dark")
        if q["clipped_fraction"] > t["clipped_max"]:
            out.append("overexposed")
        if q["sharpness"] < t["sharpness_min"]:
            out.append("blurred")
        if q["contrast"] < t["contrast_min"] and "too_dark" not in out:
            out.append("low_contrast")
        return out

    @staticmethod
    def verdict_of(flags):
        """not_road and poor_quality are refusals (the citizen endpoint turns them away); unusual is a warning."""
        if "not_a_road_scene" in flags:
            return "not_road"
        if any(f in flags for f in ("too_dark", "overexposed", "blurred")):
            return "poor_quality"
        return "unusual" if flags else "ok"

    def flags_for(self, quality, maha, p_not):
        flags = self.quality_flags(quality)
        if maha > self.thresholds["maha"]:
            flags.append("unlike_training_photos")
        if p_not > self.thresholds["p_not_road"]:
            flags.append("not_a_road_scene")
        return flags

    def check(self, img_rgb, with_projection=False):
        if not self.is_ready:
            return {"available": False, "reason": self.load_error}
        q = image_quality(img_rgb)
        emb = self.embedder.embed(img_rgb)
        maha, p_not = self.scores(emb[None, :])
        proj = self.project(emb[None, :])[0] if with_projection else None
        maha, p_not = float(maha[0]), float(p_not[0])
        t = self.thresholds
        flags = self.flags_for(q, maha, p_not)
        verdict = self.verdict_of(flags)
        out = {
            "available": True,
            "verdict": verdict,
            "in_distribution": verdict == "ok",
            "flags": flags,
            "novelty_score": round(maha, 3),
            "novelty_threshold": round(t["maha"], 3),
            "p_not_road": round(p_not, 4),
            "p_not_road_threshold": round(t["p_not_road"], 4),
            "quality": q,
            "model": f"ood_guard {self.meta.get('version', '')}".strip(),
        }
        if proj is not None:
            out["projection"] = [round(float(x), 5) for x in proj]
        return out

    def describe(self):
        if not self.is_ready:
            return {"ready": False, "reason": self.load_error}
        test = (self.report.get("test") or {})
        return {"ready": True, "backbone": BACKBONE, "thresholds": self.thresholds,
                "version": self.meta.get("version"),
                "auroc_combined": (test.get("combined") or {}).get("auroc"),
                "road_photos_flagged": (test.get("combined") or {}).get("in_distribution_flagged_rate"),
                "non_road_photos_caught": (test.get("combined") or {}).get("ood_caught_rate"),
                "road_photos_refused": (test.get("refusals") or {}).get("road_photos_refused_rate"),
                "non_road_photos_refused": (test.get("refusals") or {}).get("everyday_photos_refused_rate")}

    MESSAGES = {
        "not_road": "This does not look like a photograph of a road.",
        "poor_quality": "The photograph is too dark, overexposed or blurred to measure.",
        "unusual": "This photograph is unlike the road photographs the models were trained on.",
    }
