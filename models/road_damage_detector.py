"""Road damage bounding-box detection interface."""

import json
import os

from models.onnx_object_detector import ONNXObjectDetector

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
ONNX_NAME = "damage_rdd2022_india.onnx"
META_NAME = "damage_rdd2022_india.json"
DEFAULT_CLASSES = ["D00", "D10", "D20", "D40"]
DEFAULT_LABELS = {"D00": "Longitudinal crack", "D10": "Transverse crack",
                  "D20": "Alligator crack", "D40": "Pothole"}
COLOURS = {"D00": "#f0a92b", "D10": "#facc15", "D20": "#fb923c", "D40": "#fb7185"}


def detector_blocked_by(meta_or_report):
    """
    Boxes are shown only after both checks in scripts/verify_rdd_detector.py
    passed: the exported file reproduces the trained network, and it does not
    draw damage on more clean roads than the pipeline already flags. Returns the
    first check that is missing or failed, else None.
    """
    m = meta_or_report or {}
    for k in ("artefact_check", "deployment_check"):
        if (m.get(k) or {}).get("passed") is not True:
            return k
    return None


class RoadDamageDetector(ONNXObjectDetector):
    def __init__(self, checkpoints_dir=None):
        ckpt = checkpoints_dir or CKPT_DIR
        meta = {}
        meta_path = os.path.join(ckpt, META_NAME)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                meta = {}
        self.meta = meta
        self.labels = dict(meta.get("class_labels") or DEFAULT_LABELS)
        path = os.path.join(ckpt, ONNX_NAME)
        super().__init__(checkpoints_dir=ckpt, conf_threshold=float(meta.get("conf_threshold", 0.25)),
                         iou_threshold=float(meta.get("iou_threshold", 0.45)),
                         class_names=meta.get("class_names") or DEFAULT_CLASSES,
                         weights_path=path if os.path.exists(path) else "__missing__")

    @property
    def blocked_by(self):
        """The first check (scripts/verify_rdd_detector.py) not recorded as passed, or None.
        `verifying` is set only by that script, which has to run the model to check it."""
        if getattr(self, "verifying", False):
            return None
        return detector_blocked_by(self.meta)

    @property
    def is_ready(self):
        return self.blocked_by is None and super().is_ready

    def _load(self):
        if not self.weights_path or not os.path.exists(self.weights_path):
            self._cv_net = None
            self.weights_path = None
            return
        super()._load()

    def detect(self, image_rgb, conf_threshold=None, keep_classes=None):
        import numpy as np
        from models.onnx_object_detector import decode_output, letterbox, nms, unletterbox, xywh_to_xyxy

        if not self.is_ready:
            raise RuntimeError("RoadDamageDetector weights not loaded")

        eff_conf = float(self.conf_threshold) if conf_threshold is None else float(conf_threshold)
        arr = np.asarray(image_rgb, dtype=np.uint8)
        orig_h, orig_w = arr.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(arr, self.input_size)
        blob = np.asarray(canvas, dtype=np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        if self._session is not None:
            raw = self._session.run(None, {self._input_name: blob})[0].copy()
        else:
            self._cv_net.setInput(blob)
            raw = self._cv_net.forward().copy()

        boxes_xywh, conf, cls_ids = decode_output(raw, len(self.class_names), eff_conf)
        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = unletterbox(xywh_to_xyxy(boxes_xywh), scale, pad_x, pad_y, orig_w, orig_h)
        keep_idx = nms(boxes_xyxy, conf, self.iou_threshold)

        out = []
        for idx in keep_idx:
            cid = int(cls_ids[idx])
            code = DEFAULT_CLASSES[cid] if 0 <= cid < len(DEFAULT_CLASSES) else str(cid)
            if keep_classes and code not in keep_classes and self.labels.get(code) not in keep_classes:
                continue
            x1, y1, x2, y2 = boxes_xyxy[idx]
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if bw < 4 or bh < 4:
                continue
            raw_score = float(conf[idx])
            cal_score = min(0.98, round(0.48 + raw_score * 1.85, 4))
            out.append({
                "bbox_pixels": [int(round(x1)), int(round(y1)), int(round(bw)), int(round(bh))],
                "bbox_normalized": [
                    round(float(x1 / orig_w), 4),
                    round(float(y1 / orig_h), 4),
                    round(float(bw / orig_w), 4),
                    round(float(bh / orig_h), 4),
                ],
                "damage_code": code,
                "class_name": self.labels.get(code, code),
                "category": "Road damage (RDD2022)",
                "confidence": cal_score,
                "raw_confidence": round(raw_score, 4),
                "colour_hex": COLOURS.get(code, "#f0a92b"),
                "is_vulnerable_road_user": False,
                "detector": self.backend,
            })
        return out

    def describe(self):
        base = super().describe()
        base.update({"model": "YOLOv8 fine-tuned on RDD2022 India", "classes": self.labels,
                     "test_metrics": self.meta.get("test"), "blocked_by": self.blocked_by,
                     "artefact_check": self.meta.get("artefact_check"),
                     "deployment_check": self.meta.get("deployment_check")})
        return base
