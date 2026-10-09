"""
Road-damage boxes from the YOLOv8 model trained on RDD2022 India
(training/train_rdd_detector.py), served through ONNX Runtime on the CPU.

It reuses the COCO detector's letterbox / decode / NMS code
(models/onnx_object_detector.py) with this model's four classes:

    D00 longitudinal crack   D10 transverse crack   D20 alligator crack   D40 pothole

Boxes say WHERE damage is in a road frame. Area and cost still come from the
segmentation mask: a box around a diagonal crack is mostly sound road.

The confidence threshold is the one chosen on the validation split during
training (max mean F1), read from checkpoints/damage_rdd2022_india.json.
Without the model file this module reports not-ready and the pipeline simply
omits the boxes.
"""

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
        self.resize_mode = meta.get("resize", "letterbox")     # RT-DETR candidates were validated stretched

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
        out = []
        for d in super().detect(image_rgb, conf_threshold=conf_threshold, keep_classes=keep_classes):
            code = d["class_name"]
            d["damage_code"] = code
            d["class_name"] = self.labels.get(code, code)
            d["category"] = "Road damage (RDD2022)"
            d["colour_hex"] = COLOURS.get(code, "#f0a92b")
            d["is_vulnerable_road_user"] = False
            out.append(d)
        return out

    def describe(self):
        base = super().describe()
        base.update({"model": "YOLOv8 fine-tuned on RDD2022 India", "classes": self.labels,
                     "test_metrics": self.meta.get("test"), "blocked_by": self.blocked_by,
                     "artefact_check": self.meta.get("artefact_check"),
                     "deployment_check": self.meta.get("deployment_check")})
        return base
