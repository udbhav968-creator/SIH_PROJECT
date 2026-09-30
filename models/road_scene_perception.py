"""
Unified road-scene perception: one call, every detector, one answer.

Three independently trained detectors run over the same frame:

  * traffic   - COCO-pretrained YOLO: people, cars, buses, trucks, two-wheelers,
                bicycles, traffic lights, stop signs (scripts/fetch_detector.py)
  * damage    - trained here on RDD2022 India: longitudinal, transverse and
                alligator cracks, potholes (configs/detectors/road_damage.yaml)
  * markings  - trained here on CDSet: zebra crossings and lane guide arrows
                (configs/detectors/crosswalk.yaml)

They are kept as separate models on purpose. Each dataset labels only its own
classes, so a single merged model would be taught that every unlabelled
pothole in a crosswalk photograph is background. Separate models avoid that
and can be retrained, versioned and rolled back independently.

Each trained model ships with a model card (checkpoints/detectors/<name>.json)
carrying its class names and per-class serving thresholds, chosen for maximum
F1 on the validation split. Thresholds are never hand-tuned in code.

On top of the raw detections the scene layer derives facts an operator acts
on: pedestrians inside a zebra crossing, vehicles stopped on one, a pothole in
the vehicle's own path. These are geometric rules over detected boxes and are
labelled as such in the output.

If a detector's weights are missing it is reported as unavailable and the
others still run; nothing is invented to fill the gap.
"""

import json
import os
import time

import numpy as np

from models.frame_gate import frame_quality
from models.onnx_object_detector import CKPT_DIR, ONNXObjectDetector

DETECTOR_DIR = os.path.join(CKPT_DIR, "detectors")

# COCO classes worth reporting from a road-facing camera. The other 68
# (cups, laptops, teddy bears...) are dropped rather than shown as noise.
TRAFFIC_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "dog", "cow",  # stray animals are a real hazard on Indian roads
}

# group -> (display label, colour) for every class any detector can emit.
CLASS_STYLE = {
    "person": ("pedestrian", "vulnerable_road_user", "#ef4444"),
    "bicycle": ("bicycle", "vulnerable_road_user", "#f97316"),
    "motorcycle": ("two-wheeler", "vulnerable_road_user", "#fb923c"),
    "dog": ("animal", "vulnerable_road_user", "#fdba74"),
    "cow": ("animal", "vulnerable_road_user", "#fdba74"),
    "car": ("car", "vehicle", "#38bdf8"),
    "bus": ("bus", "vehicle", "#22d3ee"),
    "truck": ("truck", "vehicle", "#0ea5e9"),
    "traffic light": ("traffic light", "traffic_control", "#a3e635"),
    "stop sign": ("stop sign", "traffic_control", "#84cc16"),
    "longitudinal_crack": ("longitudinal crack (D00)", "road_damage", "#eab308"),
    "transverse_crack": ("transverse crack (D10)", "road_damage", "#f59e0b"),
    "alligator_crack": ("alligator crack (D20)", "road_damage", "#d97706"),
    "pothole": ("pothole (D40)", "road_damage", "#dc2626"),
    "crosswalk": ("zebra crossing", "road_marking", "#e5e7eb"),
    "guide_arrows": ("lane guide arrow", "road_marking", "#a78bfa"),
}
VULNERABLE_GROUP = "vulnerable_road_user"


def load_model_card(name, detector_dir=DETECTOR_DIR):
    """Model card for a trained detector, or None if it has not been trained."""
    path = os.path.join(detector_dir, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# scene geometry (pure functions over [x, y, w, h] pixel boxes)
# ---------------------------------------------------------------------------
def ground_contact_point(bbox):
    """Where an upright object touches the road: bottom-centre of its box."""
    x, y, w, h = bbox
    return x + w / 2.0, y + h


def point_in_box(point, bbox, margin=0.0):
    """True if (px, py) lies inside bbox grown by `margin` of its own size."""
    px, py = point
    x, y, w, h = bbox
    dx, dy = w * margin, h * margin
    return x - dx <= px <= x + w + dx and y - dy <= py <= y + h + dy


def overlap_fraction(inner, outer):
    """Fraction of `inner`'s area that lies inside `outer`."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    w = max(0.0, min(ix + iw, ox + ow) - max(ix, ox))
    h = max(0.0, min(iy + ih, oy + oh) - max(iy, oy))
    area = iw * ih
    return (w * h) / area if area > 0 else 0.0


def in_ego_path(bbox, image_w, image_h, half_width=0.2, horizon=0.55):
    """
    Rough ego-lane test for a forward camera: the box's ground contact point
    is below the horizon line and within the central corridor of the frame.
    A heuristic, not lane detection; reported as such.
    """
    cx, cy = ground_contact_point(bbox)
    return abs(cx / image_w - 0.5) <= half_width and cy / image_h >= horizon


def derive_scene_facts(detections, image_w, image_h):
    """Operational facts from box geometry. Every fact names the rule it used."""
    crossings = [d for d in detections if d["class_name"] == "crosswalk"]
    vru = [d for d in detections if d["group"] == VULNERABLE_GROUP]
    vehicles = [d for d in detections if d["group"] == "vehicle"]
    damage = [d for d in detections if d["group"] == "road_damage"]

    on_crossing = [
        p for p in vru
        if any(point_in_box(ground_contact_point(p["bbox_pixels"]), c["bbox_pixels"], margin=0.1)
               for c in crossings)
    ]
    vehicles_blocking = [
        v for v in vehicles
        if any(overlap_fraction(v["bbox_pixels"], c["bbox_pixels"]) >= 0.3 for c in crossings)
    ]
    potholes_in_path = [
        d for d in damage
        if d["class_name"] == "pothole" and in_ego_path(d["bbox_pixels"], image_w, image_h)
    ]
    damage_area = sum(d["bbox_pixels"][2] * d["bbox_pixels"][3] for d in damage)

    return {
        "zebra_crossing_visible": bool(crossings),
        "pedestrians_on_crossing": len(on_crossing),
        "vulnerable_road_users": len(vru),
        "vehicles": len(vehicles),
        "vehicles_blocking_crossing": len(vehicles_blocking),
        "road_damage_count": len(damage),
        "potholes_in_ego_path": len(potholes_in_path),
        # Box area over-states a thin diagonal crack; this is a triage signal,
        # not a measurement. Measured area comes from the segmentation stage.
        "damage_bbox_coverage_pct": round(100.0 * min(1.0, damage_area / float(image_w * image_h)), 2),
        "rules": {
            "pedestrians_on_crossing": "ground contact point inside a crossing box grown by 10%",
            "vehicles_blocking_crossing": ">=30% of the vehicle box overlaps a crossing box",
            "potholes_in_ego_path": "contact point in the central 40% of the frame, below 55% height",
        },
    }


def alerts_from_facts(facts):
    """Prioritised alerts; the first is the most urgent."""
    alerts = []
    if facts["pedestrians_on_crossing"]:
        alerts.append({"level": "CRITICAL", "code": "PEDESTRIAN_ON_CROSSING",
                       "message": f"{facts['pedestrians_on_crossing']} road user(s) on a zebra crossing ahead"})
    if facts["potholes_in_ego_path"]:
        alerts.append({"level": "HIGH", "code": "POTHOLE_IN_PATH",
                       "message": f"{facts['potholes_in_ego_path']} pothole(s) in the vehicle's path"})
    if facts["vehicles_blocking_crossing"]:
        alerts.append({"level": "MEDIUM", "code": "CROSSING_BLOCKED",
                       "message": f"{facts['vehicles_blocking_crossing']} vehicle(s) stopped on a zebra crossing"})
    if facts["road_damage_count"] and not facts["potholes_in_ego_path"]:
        alerts.append({"level": "INFO", "code": "ROAD_DAMAGE_LOGGED",
                       "message": f"{facts['road_damage_count']} road-surface defect(s) recorded for maintenance"})
    return alerts


# ---------------------------------------------------------------------------
# perception service
# ---------------------------------------------------------------------------
class _Head:
    """One detector plus the metadata needed to interpret its output."""

    def __init__(self, key, detector, thresholds, card):
        self.key = key
        self.detector = detector
        self.thresholds = thresholds
        self.card = card

    @property
    def is_ready(self):
        return self.detector is not None and self.detector.is_ready

    def describe(self):
        info = {"ready": self.is_ready}
        if self.detector is not None:
            info.update(backend=self.detector.backend, input_size=self.detector.input_size)
        if self.card:
            info.update(
                classes=self.card.get("classes"),
                serving_thresholds=self.thresholds,
                test_metrics={k: self.card["metrics"]["test"][k]
                              for k in ("mAP50", "mAP50_95", "precision", "recall")},
                trained_utc=self.card.get("created_utc"),
                dataset_license=self.card.get("dataset", {}).get("license"),
            )
        return info


class RoadScenePerception:
    """Runs every available detector on a frame and merges the results."""

    TRAFFIC_THRESHOLD = 0.35  # published COCO weights; no project validation split to tune on

    def __init__(self, checkpoints_dir=None, detector_dir=None, traffic_detector=None):
        ckpt = checkpoints_dir or CKPT_DIR
        detector_dir = detector_dir or os.path.join(ckpt, "detectors")
        traffic = traffic_detector or ONNXObjectDetector(checkpoints_dir=ckpt, conf_threshold=self.TRAFFIC_THRESHOLD)
        self.heads = {
            "traffic": _Head("traffic", traffic,
                             {name: self.TRAFFIC_THRESHOLD for name in TRAFFIC_CLASSES}, None),
            "damage": self._load_trained("road_damage", detector_dir),
            "markings": self._load_trained("crosswalk", detector_dir),
        }

    @staticmethod
    def _load_trained(name, detector_dir):
        card = load_model_card(name, detector_dir)
        onnx_path = os.path.join(detector_dir, f"{name}.onnx")
        if card is None or not os.path.exists(onnx_path):
            return _Head(name, None, {}, None)
        detector = ONNXObjectDetector(weights_path=onnx_path, class_names=card["classes"],
                                      conf_threshold=min(card["serving_thresholds"].values()))
        return _Head(name, detector, dict(card["serving_thresholds"]), card)

    @property
    def is_ready(self):
        return any(head.is_ready for head in self.heads.values())

    def describe(self):
        return {key: head.describe() for key, head in self.heads.items()}

    def _run_head(self, head, image):
        if not head.is_ready:
            return []
        keep = set(head.thresholds)
        raw = head.detector.detect(image, conf_threshold=min(head.thresholds.values()), keep_classes=keep)
        detections = []
        for det in raw:
            name = det["class_name"]
            if det["confidence"] < head.thresholds.get(name, 1.0):
                continue
            label, group, colour = CLASS_STYLE.get(name, (name, "other", "#94a3b8"))
            detections.append({
                "class_name": name,
                "label": label,
                "group": group,
                "confidence": det["confidence"],
                "bbox_pixels": det["bbox_pixels"],
                "bbox_normalized": det["bbox_normalized"],
                "colour_hex": colour,
                "model": head.key,
            })
        return detections

    def analyze(self, image_rgb, groups=None, gate=True):
        """
        Detect everything in one RGB frame.

        `groups` optionally restricts which detector heads run, e.g.
        {"damage"} for a maintenance survey where people and cars are noise.
        With `gate`, frames the quality gate rejects (covered lens, darkness,
        glare, defocus) skip the detectors; the result says why.
        """
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 RGB image, got shape {image.shape}")
        height, width = image.shape[:2]

        started = time.perf_counter()
        quality = frame_quality(image) if gate else None
        skip = quality is not None and not quality["analysable"]
        detections, timings = [], {}
        for key, head in self.heads.items():
            if skip or (groups and key not in groups):
                continue
            t0 = time.perf_counter()
            detections.extend(self._run_head(head, image))
            if head.is_ready:
                timings[key] = round((time.perf_counter() - t0) * 1000.0, 1)
        detections.sort(key=lambda d: -d["confidence"])

        counts = {}
        for det in detections:
            counts[det["class_name"]] = counts.get(det["class_name"], 0) + 1
        facts = derive_scene_facts(detections, width, height)
        return {
            "image_size": [width, height],
            "frame_quality": quality,
            "skipped_by_quality_gate": skip,
            "detections": detections,
            "counts": counts,
            "scene": facts,
            "alerts": alerts_from_facts(facts),
            "models": {key: head.is_ready for key, head in self.heads.items()},
            "unavailable_models": [key for key, head in self.heads.items() if not head.is_ready],
            "latency_ms": {"total": round((time.perf_counter() - started) * 1000.0, 1), **timings},
        }


def redact_people(image_rgb, result, pad=0.08):
    """
    Blur every detected person before an image leaves the system (DPDP Act
    2023: road imagery is collected for maintenance, not to identify people).

    Whole person boxes are blurred, padded slightly because a box is rarely
    tight around a head. Number plates are NOT redacted: that needs a plate
    detector this project does not have, and claiming otherwise would be worse
    than the gap. Returns a new uint8 array; the input is not modified.
    """
    from PIL import Image, ImageFilter

    frame = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    width, height = frame.size
    for det in result["detections"]:
        if det["class_name"] != "person":
            continue
        x, y, w, h = det["bbox_pixels"]
        x0, y0 = max(0, int(x - w * pad)), max(0, int(y - h * pad))
        x1, y1 = min(width, int(x + w * (1 + pad))), min(height, int(y + h * (1 + pad)))
        if x1 <= x0 or y1 <= y0:
            continue
        region = frame.crop((x0, y0, x1, y1))
        radius = max(6, max(x1 - x0, y1 - y0) // 6)
        frame.paste(region.filter(ImageFilter.GaussianBlur(radius)), (x0, y0))
    return np.asarray(frame)


def annotate(image_rgb, result, line_width=None):
    """Draw detections on a copy of the frame. Returns a PIL image."""
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width = line_width or max(2, round(max(canvas.size) / 400))
    # Draw big, low-priority regions first so people and potholes stay on top.
    order = {"road_marking": 0, "road_damage": 1, "vehicle": 2, "traffic_control": 2, VULNERABLE_GROUP: 3}
    for det in sorted(result["detections"], key=lambda d: order.get(d["group"], 2)):
        x, y, w, h = det["bbox_pixels"]
        colour = det["colour_hex"]
        draw.rectangle([x, y, x + w, y + h], outline=colour, width=width)
        text = f"{det['label']} {det['confidence']:.2f}"
        tx0, ty0, tx1, ty1 = draw.textbbox((x, y), text)
        ty = y - (ty1 - ty0) - 4 if y - (ty1 - ty0) - 4 >= 0 else y
        draw.rectangle([x, ty, x + (tx1 - tx0) + 6, ty + (ty1 - ty0) + 4], fill=colour)
        draw.text((x + 3, ty + 1), text, fill="#000000")
    return canvas
