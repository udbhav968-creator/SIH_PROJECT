"""Pavement defect segmentation using feature extraction and pixel classification."""

import os

import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
MODEL_PATH = os.path.join(CKPT_DIR, "defect_segmenter.joblib")

CLASS_SOUND, CLASS_CRACK, CLASS_POTHOLE = 0, 1, 2
CLASS_LABELS = {CLASS_SOUND: "sound road", CLASS_CRACK: "crack", CLASS_POTHOLE: "pothole"}
IGNORE = 255

WORK_W, WORK_H = 320, 200
FEATURE_NAMES = [
    "L", "a", "b", "grad_mag", "laplacian_abs",
    "std_5", "std_15", "L_minus_local_mean_31",
    "row_v_norm", "col_offset_norm", "L_minus_image_mean",
]


def _as_uint8_rgb(image):
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.dstack([arr] * 3)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr[:, :, :3]


def extract_pixel_features(image_rgb, work_size=(WORK_W, WORK_H)):
    """
    (H*W, 11) float32 features, plus the (H, W) they were computed at.

    Everything here is a whole-image vectorised operation, so the cost is a
    handful of convolutions rather than a Python loop over pixels.
    """
    import cv2

    img = _as_uint8_rgb(image_rgb)
    w, h = work_size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    lap = np.abs(cv2.Laplacian(L, cv2.CV_32F, ksize=3))

    def local_std(src, k):
        mean = cv2.blur(src, (k, k))
        sq = cv2.blur(src * src, (k, k))
        return np.sqrt(np.maximum(sq - mean * mean, 0.0))

    std5 = local_std(L, 5)
    std15 = local_std(L, 15)

    local_mean31 = cv2.blur(L, (31, 31))
    darkness = L - local_mean31

    rows = np.repeat(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], w, axis=1)
    cols = np.repeat(np.abs(np.linspace(-1.0, 1.0, w, dtype=np.float32))[None, :], h, axis=0)
    global_rel = L - float(L.mean())

    stack = np.stack([L, a, b, grad, lap, std5, std15, darkness, rows, cols, global_rel], axis=-1)
    return stack.reshape(-1, stack.shape[-1]).astype(np.float32), (h, w)


def mask_from_proba(proba, work_shape, orig_shape, thresholds, min_blob_px=12):
    """
    (H*W, 3) class probabilities at working resolution -> the segment() dict.

    Shared by every segmenter (the pixel classifier and the U-Net), so a mask,
    its size floor and its confidence mean the same thing whichever model
    produced the probabilities.
    """
    import cv2
    h, w = work_shape
    orig_h, orig_w = orig_shape

    crack_p, pothole_p = proba[:, CLASS_CRACK], proba[:, CLASS_POTHOLE]
    flat = np.full(proba.shape[0], CLASS_SOUND, dtype=np.uint8)
    is_crack = crack_p >= thresholds.get("crack", 0.5)
    is_pothole = pothole_p >= thresholds.get("pothole", 0.5)
    flat[is_crack] = CLASS_CRACK
    flat[is_pothole & (pothole_p >= crack_p)] = CLASS_POTHOLE
    flat[is_pothole & ~is_crack] = CLASS_POTHOLE
    labels = flat.reshape(h, w)
    conf = np.where(flat == CLASS_CRACK, crack_p,
                    np.where(flat == CLASS_POTHOLE, pothole_p,
                             proba[:, CLASS_SOUND])).reshape(h, w)

    cleaned = np.zeros_like(labels)
    for cls in (CLASS_CRACK, CLASS_POTHOLE):
        binary = (labels == cls).astype(np.uint8)
        if binary.sum() == 0:
            continue
        n, comp, stats, _cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_px:
                cleaned[comp == i] = cls

    mask_full = cv2.resize(cleaned, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    out = {
        "mask": mask_full,

        "mask_work": cleaned,
        "work_shape": (h, w),
        "scale_x": orig_w / float(w),
        "scale_y": orig_h / float(h),

        "proba_crack": crack_p.reshape(h, w).astype(np.float32),
        "proba_pothole": pothole_p.reshape(h, w).astype(np.float32),
    }
    for cls, name in ((CLASS_CRACK, "crack"), (CLASS_POTHOLE, "pothole")):
        sel = cleaned == cls
        out[f"{name}_px"] = int(sel.sum())
        out[f"{name}_px_full"] = int((mask_full == cls).sum())
        out[f"{name}_mean_confidence"] = float(conf[sel].mean()) if sel.any() else 0.0
    out["defect_fraction"] = float((cleaned != CLASS_SOUND).mean())
    return out


class DefectSegmenter:
    """Loads the trained pixel classifier and produces defect masks."""

    DEFAULT_THRESHOLDS = {"crack": 0.5, "pothole": 0.5}

    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_PATH
        self.clf = None
        self.report = None
        self.load_error = None
        self.load_error_detail = None
        self.environment_failed = False
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        self._load()

    @property
    def file_exists(self):
        """A model is on disk. Distinguishes 'never trained' from 'will not load'."""
        return os.path.exists(self.model_path)

    @property
    def is_ready(self):
        return self.clf is not None

    def _load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            import joblib
            blob = joblib.load(self.model_path)
            self.clf = blob["classifier"]
            self.report = {k: v for k, v in blob.items() if k != "classifier"}
            self.thresholds = dict(blob.get("thresholds") or self.DEFAULT_THRESHOLDS)
        except Exception as e:

            self.load_error = str(e)

            trained_with = None
            report_path = os.path.splitext(self.model_path)[0] + "_report.json"
            try:
                import json as _json
                with open(report_path, "r", encoding="utf-8") as _fh:
                    trained_with = _json.load(_fh).get("sklearn_version")
            except Exception:
                pass
            try:
                import sklearn
                here = sklearn.__version__
            except Exception:
                here = "unknown"

            environment_failed = (here == "unknown") or (not str(e).strip())
            self.environment_failed = environment_failed
            self.load_error_detail = {
                "path": self.model_path,
                "error": str(e) or "(no message - typically MemoryError)",
                "environment_failed": environment_failed,
                "sklearn_here": here,
                "sklearn_trained_with": trained_with,
                "likely_cause": (
                    "THIS MACHINE, not the model: scikit-learn could not even be "
                    "read. Almost always memory exhaustion. Do NOT retrain - "
                    "training needs far more memory than loading, and it would "
                    "overwrite a model that is probably fine. Free memory first, "
                    "then try again."
                    if environment_failed else
                    f"scikit-learn version mismatch: trained on {trained_with}, "
                    f"running {here}. A pickled estimator is not portable across versions."
                    if trained_with and trained_with != here else
                    "scikit-learn version mismatch (the error names a private module that "
                    "moved between versions); retraining resolves it"
                    if "No module named" in str(e) else
                    "unreadable model file"),
                "fix": ("free memory and retry - do not retrain"
                        if environment_failed
                        else "python -m training.train_segmenter --images 2000"),
            }
            print(f"[DefectSegmenter] could not load {self.model_path}: "
                  f"{e or '(no message - typically MemoryError)'}")
            if environment_failed:
                print("[DefectSegmenter] scikit-learn itself could not be read. This is "
                      "the MACHINE, not the model - almost always out of memory.")
                print("[DefectSegmenter] DO NOT RETRAIN: training needs far more memory "
                      "than loading, and would overwrite a model that is probably fine.")
                print("[DefectSegmenter] free memory and try again.")
            elif trained_with and trained_with != here:
                print(f"[DefectSegmenter] trained with scikit-learn {trained_with}, "
                      f"this environment has {here}")
            print("[DefectSegmenter] WITHOUT A SEGMENTER the pipeline falls back to "
                  "brightness proposals, which report zebra crossings as potholes.")
            print("[DefectSegmenter] fix: python -m training.train_segmenter --images 2000")

    def segment(self, image_rgb, min_blob_px=12):
        """
        Per-pixel defect mask at the original image resolution.

        Returns a dict with the mask, the pixel counts per class, and the mean
        predicted probability over each defect's own pixels - which is a far
        more honest confidence than a whole-image classifier score, because it
        is computed only where the model says the defect is.
        """
        if not self.is_ready:
            raise RuntimeError("No segmenter on disk. Train one with training/train_segmenter.py")
        import cv2

        img = _as_uint8_rgb(image_rgb)
        orig_h, orig_w = img.shape[:2]
        feats, (h, w) = extract_pixel_features(img)

        proba = self.clf.predict_proba(feats)
        return mask_from_proba(proba, (h, w), (orig_h, orig_w), self.thresholds, min_blob_px)

    def largest_component_box(self, mask, cls):
        """Bounding box of the biggest blob of `cls`, or None. (x, y, w, h)."""
        import cv2
        binary = (mask == cls).astype(np.uint8)
        if binary.sum() == 0:
            return None
        n, _comp, stats, _c = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 1:
            return None
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))

    def describe(self):
        return {
            "ready": self.is_ready,
            "model_path": self.model_path if self.is_ready else None,
            "classes": CLASS_LABELS,
            "features": FEATURE_NAMES,
            "trained_on": (self.report or {}).get("trained_on"),
            "iou": (self.report or {}).get("iou"),
            "thresholds": self.thresholds,
        }
