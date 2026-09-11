"""
Real object detection through ONNX Runtime: people, vehicles, bicycles,
traffic lights and signs in front of the camera.

The weights are a YOLO model trained on COCO (330k images, 80 classes) -
downloaded and exported by scripts/fetch_detector.py. Nothing here is trained
in this repo; this is inference over published weights, which is the honest
and sensible way to get 80-class detection.

Supported output layouts, detected from the tensor shape:
  * YOLOv8 / v9 / v11 : (1, 4 + n_classes, n_boxes)   - no objectness channel
  * YOLOv5 / v7       : (1, n_boxes, 5 + n_classes)   - objectness channel

Everything except the network forward pass is plain NumPy (letterbox resize,
score thresholding, class-wise non-maximum suppression, coordinate
un-letterboxing), so it can be unit-tested without the runtime installed.

If the weights or onnxruntime are missing, `is_ready` is False and callers
skip the detection stage rather than inventing detections.
"""

import glob
import os

import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# What a road-safety system cares about, and how the dashboard colours it.
ROAD_RELEVANT = {
    "person": ("Vulnerable road user", "#ef4444"),
    "bicycle": ("Vulnerable road user", "#f97316"),
    "motorcycle": ("Two-wheeler", "#fb923c"),
    "car": ("Vehicle", "#38bdf8"),
    "bus": ("Heavy vehicle", "#22d3ee"),
    "truck": ("Heavy vehicle", "#0ea5e9"),
    "train": ("Rail vehicle", "#818cf8"),
    "traffic light": ("Traffic control", "#a3e635"),
    "stop sign": ("Traffic control", "#84cc16"),
    "fire hydrant": ("Roadside asset", "#94a3b8"),
    "bench": ("Roadside asset", "#94a3b8"),
    "parking meter": ("Roadside asset", "#94a3b8"),
}

VULNERABLE = {"person", "bicycle", "motorcycle"}


# ---------------------------------------------------------------------------
# geometry helpers (pure numpy, unit-testable without onnxruntime)
# ---------------------------------------------------------------------------
def letterbox(image_rgb, target=640, pad_value=114):
    """Resize keeping aspect ratio and pad to a square. Returns (image, scale, pad_x, pad_y)."""
    from PIL import Image
    img = image_rgb
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img, dtype=np.uint8))
    img = img.convert("RGB")
    w, h = img.size
    scale = min(target / w, target / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), (pad_value, pad_value, pad_value))
    pad_x, pad_y = (target - nw) // 2, (target - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def xywh_to_xyxy(boxes):
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def nms(boxes, scores, iou_threshold=0.45):
    """Greedy non-maximum suppression. boxes are [x1, y1, x2, y2]."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou <= iou_threshold]
    return keep


def decode_output(raw, n_classes, conf_threshold):
    """
    Normalises either YOLO output layout into (boxes_xywh, scores, class_ids)
    in letterboxed pixel coordinates.
    """
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    # Orient to (n_boxes, attributes). v8-style exports are (attributes, n_boxes).
    if arr.shape[1] not in (4 + n_classes, 5 + n_classes) and arr.shape[0] in (4 + n_classes, 5 + n_classes):
        arr = arr.T

    if arr.shape[1] == 4 + n_classes:          # v8/v9/v11: no objectness
        boxes = arr[:, :4]
        cls_scores = arr[:, 4:]
        conf = cls_scores.max(axis=1)
        cls_ids = cls_scores.argmax(axis=1)
    elif arr.shape[1] == 5 + n_classes:        # v5/v7: objectness * class score
        boxes = arr[:, :4]
        obj = arr[:, 4]
        cls_scores = arr[:, 5:]
        cls_ids = cls_scores.argmax(axis=1)
        conf = obj * cls_scores[np.arange(len(cls_ids)), cls_ids]
    else:
        raise ValueError(
            f"Unrecognised detector output shape {np.asarray(raw).shape} for {n_classes} classes"
        )

    keep = conf >= conf_threshold
    return boxes[keep], conf[keep], cls_ids[keep]


def unletterbox(boxes_xyxy, scale, pad_x, pad_y, orig_w, orig_h):
    out = boxes_xyxy.copy().astype(np.float64)
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, orig_h)
    return out


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------
class ONNXObjectDetector:
    """COCO-pretrained detector served through ONNX Runtime."""

    def __init__(self, checkpoints_dir=None, conf_threshold=0.35, iou_threshold=0.45,
                 class_names=None, weights_path=None):
        self.ckpt_dir = checkpoints_dir or CKPT_DIR
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = list(class_names or COCO_CLASSES)
        self.input_size = 640
        self.weights_path = weights_path
        self.backend = None
        self._session = None
        self._input_name = None
        self._load()

    @property
    def is_ready(self):
        return self._session is not None

    def _load(self):
        path = self.weights_path
        if not path:
            matches = sorted(glob.glob(os.path.join(self.ckpt_dir, "*detector*.onnx"))) or \
                      sorted(glob.glob(os.path.join(self.ckpt_dir, "yolo*.onnx")))
            path = matches[0] if matches else None
        if not path or not os.path.exists(path):
            return
        try:
            import onnxruntime as ort
        except ImportError:
            print("[detector] weights found but onnxruntime is not installed (pip install onnxruntime)")
            return
        try:
            self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            if isinstance(inp.shape[-1], int) and inp.shape[-1] > 0:
                self.input_size = int(inp.shape[-1])
            self.weights_path = path
            self.backend = f"onnxruntime:{os.path.basename(path)}"
        except Exception as e:
            print(f"[detector] could not load {path}: {e}")

    def detect(self, image_rgb, conf_threshold=None, keep_classes=None):
        """
        Returns a list of detections in this project's shared shape:
        bbox_pixels, bbox_normalized, class_name, category, confidence.
        Raises if the model isn't loaded - callers check `is_ready` first.
        """
        if not self.is_ready:
            raise RuntimeError("No detector weights loaded. Run scripts/fetch_detector.py")
        conf_threshold = self.conf_threshold if conf_threshold is None else conf_threshold

        arr = np.asarray(image_rgb, dtype=np.uint8)
        orig_h, orig_w = arr.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(arr, self.input_size)
        blob = np.asarray(canvas, dtype=np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        raw = self._session.run(None, {self._input_name: blob})[0]
        boxes_xywh, conf, cls_ids = decode_output(raw, len(self.class_names), conf_threshold)
        if len(boxes_xywh) == 0:
            return []

        boxes = xywh_to_xyxy(boxes_xywh)
        detections = []
        for cls in np.unique(cls_ids):
            mask = cls_ids == cls
            name = self.class_names[int(cls)] if int(cls) < len(self.class_names) else str(cls)
            if keep_classes and name not in keep_classes:
                continue
            kept = nms(boxes[mask], conf[mask], self.iou_threshold)
            sel_boxes = unletterbox(boxes[mask][kept], scale, pad_x, pad_y, orig_w, orig_h)
            sel_conf = conf[mask][kept]
            for (x1, y1, x2, y2), c in zip(sel_boxes, sel_conf):
                w, h = x2 - x1, y2 - y1
                if w < 2 or h < 2:
                    continue
                category, colour = ROAD_RELEVANT.get(name, ("Other object", "#94a3b8"))
                detections.append({
                    "class_name": name,
                    "category": category,
                    "colour_hex": colour,
                    "confidence": round(float(c), 4),
                    "bbox_pixels": [int(x1), int(y1), int(w), int(h)],
                    "bbox_normalized": [round(x1 / orig_w, 4), round(y1 / orig_h, 4),
                                        round(w / orig_w, 4), round(h / orig_h, 4)],
                    "is_vulnerable_road_user": name in VULNERABLE,
                    "detector": self.backend,
                })
        detections.sort(key=lambda d: -d["confidence"])
        return detections

    def summarise(self, detections):
        """Counts per class plus a vulnerable-road-user flag, for the dashboard."""
        counts = {}
        for d in detections:
            counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
        vru = [d for d in detections if d["is_vulnerable_road_user"]]
        return {
            "objects_detected": len(detections),
            "counts_by_class": counts,
            "vulnerable_road_users": len(vru),
            "vulnerable_alert": bool(vru),
            "detector": self.backend,
        }

    def describe(self):
        return {"ready": self.is_ready, "backend": self.backend, "weights_path": self.weights_path,
                "input_size": self.input_size, "n_classes": len(self.class_names),
                "confidence_threshold": self.conf_threshold}
