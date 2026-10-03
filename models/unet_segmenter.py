"""
Deep defect segmentation: a U-Net with an ImageNet ResNet-18 encoder.

Why it exists
-------------
The pixel classifier (models/defect_segmenter.py) sees eleven hand-designed
measurements per pixel and nothing else. It cannot see shape: a crack is a thin
connected line and a pothole is a closed dark cavity, and neither fact is in a
per-pixel feature vector. Its held-out IoU (crack 0.23, pothole 0.14) is the
weakest number in the measurement chain, and area -> tonnage -> cost all rest
on the mask.

A U-Net sees the whole frame at several scales and keeps full-resolution detail
through its skip connections, which is what thin cracks need.

What is shared with the pixel classifier
----------------------------------------
Everything after the probabilities: models.defect_segmenter.mask_from_proba
turns (H*W, 3) probabilities at 320x200 into the same segment() dictionary,
with the same per-class thresholds rule and the same blob-size floor. So area,
confidence and the semantic gate mean the same thing whichever model produced
the probabilities, and swapping the model changes nothing downstream.

Which one is served is decided by checkpoints/segmenter_selection.json, written
by training/train_unet_segmenter.py from a rule fixed before the test set is
scored. Absent that file, or if the ONNX model fails to load, the pixel
classifier is used exactly as before.

    seg = load_best_segmenter()
    out = seg.segment(image_rgb)
"""

import json
import os

import numpy as np

from models.defect_segmenter import (CLASS_CRACK, CLASS_LABELS, CLASS_POTHOLE, WORK_H, WORK_W,
                                     DefectSegmenter, _as_uint8_rgb, mask_from_proba)

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
ONNX_NAME = "defect_segmenter_unet.onnx"
META_NAME = "defect_segmenter_unet.json"
SELECTION_NAME = "segmenter_selection.json"

# Network input. 512x320 keeps the 1.6 aspect of the 320x200 working grid and
# is divisible by 32, which the five encoder stages need.
IN_W, IN_H = 512, 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_rgb, in_w=IN_W, in_h=IN_H):
    """uint8 RGB of any size -> (1, 3, in_h, in_w) float32, ImageNet-normalised."""
    import cv2
    img = cv2.resize(_as_uint8_rgb(image_rgb), (in_w, in_h), interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def probs_to_work(probs_chw):
    """(3, in_h, in_w) softmax -> (WORK_H*WORK_W, 3), the grid every segmenter reports on."""
    import cv2
    chans = [cv2.resize(np.asarray(probs_chw[c], dtype=np.float32), (WORK_W, WORK_H),
                        interpolation=cv2.INTER_LINEAR) for c in range(probs_chw.shape[0])]
    p = np.stack(chans, axis=-1).reshape(-1, probs_chw.shape[0])
    p = np.clip(p, 0.0, 1.0)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-6)


class _UNetBase:
    """Shared segment() logic; subclasses provide _forward(x) -> (3, H, W) softmax."""

    tta_flip = True
    thresholds = {"crack": 0.5, "pothole": 0.5}
    report = None
    load_error = None
    load_error_detail = None
    environment_failed = False

    def predict_work(self, image_rgb):
        """Class probabilities on the 320x200 grid, flip-averaged when enabled."""
        x = preprocess(image_rgb)
        p = self._forward(x)
        if self.tta_flip:
            p = 0.5 * (p + self._forward(np.ascontiguousarray(x[:, :, :, ::-1]))[:, :, ::-1])
        return probs_to_work(p)

    def segment(self, image_rgb, min_blob_px=12):
        if not self.is_ready:
            raise RuntimeError("No U-Net segmenter loaded")
        img = _as_uint8_rgb(image_rgb)
        proba = self.predict_work(img)
        return mask_from_proba(proba, (WORK_H, WORK_W), img.shape[:2], self.thresholds, min_blob_px)

    largest_component_box = DefectSegmenter.largest_component_box


class UNetSegmenter(_UNetBase):
    """The exported U-Net, served through ONNX Runtime on the CPU."""

    def __init__(self, ckpt_dir=None):
        self.ckpt_dir = ckpt_dir or CKPT_DIR
        self.model_path = os.path.join(self.ckpt_dir, ONNX_NAME)
        self._session = None
        self._input = None
        meta_path = os.path.join(self.ckpt_dir, META_NAME)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.report = meta
            self.thresholds = dict(meta.get("thresholds") or self.thresholds)
            self.tta_flip = bool(meta.get("tta_flip", True))
        if os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                self._input = self._session.get_inputs()[0].name
            except Exception as e:     # an unloadable model must not take the pipeline down
                self.load_error = str(e)
                self.load_error_detail = {"path": self.model_path, "error": str(e)}
                self._session = None

    @property
    def file_exists(self):
        return os.path.exists(self.model_path)

    @property
    def is_ready(self):
        return self._session is not None

    def _forward(self, x):
        return self._session.run(None, {self._input: x})[0][0]

    def describe(self):
        r = self.report or {}
        return {
            "ready": self.is_ready,
            "model": "U-Net, ResNet-18 encoder (ImageNet), ONNX Runtime",
            "model_path": self.model_path if self.is_ready else None,
            "classes": CLASS_LABELS,
            "input_size": [IN_W, IN_H],
            "tta_flip": self.tta_flip,
            "trained_on": r.get("trained_on"),
            "iou": r.get("iou"),
            "thresholds": self.thresholds,
        }


def read_selection(ckpt_dir=None):
    path = os.path.join(ckpt_dir or CKPT_DIR, SELECTION_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def load_best_segmenter(ckpt_dir=None):
    """
    The segmenter the selection rule chose, or the pixel classifier.

    The U-Net is used only when segmenter_selection.json says "unet" AND the
    ONNX file loads. Any other state falls back to the pixel classifier, so a
    missing or broken deep model can never leave the pipeline without a mask.
    """
    sel = read_selection(ckpt_dir)
    if sel and sel.get("served") == "unet":
        unet = UNetSegmenter(ckpt_dir)
        if unet.is_ready:
            unet.selection = sel
            return unet
        print(f"[segmenter] selection says U-Net but it did not load "
              f"({unet.load_error or 'file missing'}); using the pixel classifier")
    if ckpt_dir:
        return DefectSegmenter(os.path.join(ckpt_dir, "defect_segmenter.joblib"))
    return DefectSegmenter()
