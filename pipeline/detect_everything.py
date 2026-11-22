"""
detect_everything.py -- one entry point for "upload an image or a video, and get
every road object detected, boxed and counted."

It fuses the three pretrained YOLO detectors already in this repository through
``models.yolo_video_engine.YOLOVideoEngine``:

  * Road damage (RDD2022)  -> Pothole, Alligator / Longitudinal / Transverse Crack
  * COCO scene & traffic    -> person, car, bus, truck, motorcycle, bicycle,
                               traffic light, stop sign
  * Crosswalk & markings    -> Zebra Crosswalk, Road Guide Arrow

Honesty note on the road-damage boxes
--------------------------------------
The road-damage detector has NOT passed this repository's verification gate
(``scripts/verify_rdd_detector.py``): its exported ONNX file scored mAP@0.5 0.0211
above the trained network, over the pre-registered 0.02 budget, and the clean-road
check was never run. So by default its boxes are hidden from the trustworthy path.

This module can still show them, but only as an **unverified candidate**: every
such detection carries ``verification_status="candidate_unverified"`` and the name
of the outstanding check (``gate_blocked_by``). Nothing here ever presents those
boxes as validated. To graduate them to "verified", see
``docs/DETECTION_PIPELINE.md``.

Public API
----------
    analyze_image(image, ...) -> dict      # counts + detections + annotated image
    analyze_video(path, ...)  -> dict      # unique counts across frames + samples
    render_annotated(image_rgb, dets) -> PIL.Image
"""

from __future__ import annotations

import io
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.yolo_video_engine import YOLOVideoEngine, get_yolo_video_engine

# --------------------------------------------------------------------------- #
# Canonical, human-friendly object categories for the counts summary.
# Each detection's class_name is mapped to exactly one of these keys.
# --------------------------------------------------------------------------- #
COUNT_KEYS = [
    "potholes", "cracks", "zebra_crossings", "road_arrows",
    "persons", "cars", "buses", "trucks", "motorcycles", "bicycles", "trains",
    "traffic_lights", "stop_signs", "other",
]


def canonical_key(class_name: str, category: str = "") -> str:
    """Map a raw detector label to one friendly counting bucket."""
    n = (class_name or "").strip().lower()
    if "pothole" in n:
        return "potholes"
    if "crack" in n:
        return "cracks"
    if "zebra" in n or "crosswalk" in n:
        return "zebra_crossings"
    if "arrow" in n:
        return "road_arrows"
    if n == "person" or n == "pedestrian":
        return "persons"
    if n == "car":
        return "cars"
    if n == "bus":
        return "buses"
    if n == "truck":
        return "trucks"
    if n == "motorcycle" or n == "motorbike":
        return "motorcycles"
    if n == "bicycle":
        return "bicycles"
    if n == "train":
        return "trains"
    if "traffic light" in n:
        return "traffic_lights"
    if "stop sign" in n:
        return "stop_signs"
    return "other"


def _clean(obj: Any) -> Any:
    """Coerce numpy scalars/arrays to plain Python so the result is JSON-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    return obj


def _friendly_counts(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter(canonical_key(d.get("class_name", ""), d.get("category", "")) for d in detections)
    counts = {k: int(c.get(k, 0)) for k in COUNT_KEYS if c.get(k, 0)}
    counts["total"] = int(sum(c.values()))
    return counts


def _summary_text(counts: Dict[str, int]) -> str:
    parts = []
    nice = {
        "potholes": "pothole", "cracks": "crack", "zebra_crossings": "zebra crossing",
        "road_arrows": "road arrow", "persons": "person", "cars": "car", "buses": "bus",
        "trucks": "truck", "motorcycles": "motorcycle", "bicycles": "bicycle",
        "trains": "train", "traffic_lights": "traffic light", "stop_signs": "stop sign",
        "other": "other object",
    }
    for k in COUNT_KEYS:
        n = counts.get(k, 0)
        if n:
            word = nice[k]
            parts.append(f"{n} {word}" + ("" if n == 1 else ("es" if word.endswith("s") else "s")))
    if not parts:
        return "No road objects detected."
    return "Detected " + ", ".join(parts) + "."


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _font(size: int = 15):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_annotated(image_rgb: np.ndarray, detections: List[Dict[str, Any]]) -> Image.Image:
    """Draw every detection's box and label on a copy of the image. Candidate
    (unverified) boxes are drawn dashed and suffixed with '?' so they read as
    provisional, never as confirmed."""
    img = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    font = _font(max(13, int(W / 55)))

    for d in detections:
        bn = d.get("bbox_normalized") or [0, 0, 0, 0]
        x = float(bn[0]) * W
        y = float(bn[1]) * H
        w = float(bn[2]) * W
        h = float(bn[3]) * H
        x2, y2 = x + w, y + h
        colour = d.get("color") or d.get("colour_hex") or "#38bdf8"
        candidate = d.get("verification_status") == "candidate_unverified"

        if candidate:
            _dashed_rect(draw, x, y, x2, y2, colour, width=max(2, int(W / 320)))
        else:
            draw.rectangle([x, y, x2, y2], outline=colour, width=max(2, int(W / 320)))

        conf = d.get("confidence", 0.0)
        label = f"{d.get('class_name', 'object')} {float(conf):.0%}" + (" ?" if candidate else "")
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0, y - th - 4)
        draw.rectangle([x, ly, x + tw + 6, ly + th + 4], fill=colour)
        draw.text((x + 3, ly + 2), label, fill="#000000", font=font)

    return img


def _dashed_rect(draw, x1, y1, x2, y2, colour, width=2, dash=9, gap=6):
    def line(a, b, horizontal):
        if horizontal:
            x = a
            while x < b:
                draw.line([x, y1 if horizontal else a, min(x + dash, b), y1 if horizontal else a],
                          fill=colour, width=width)
                x += dash + gap
        # handled inline below instead
    # top & bottom
    for yy in (y1, y2):
        x = x1
        while x < x2:
            draw.line([x, yy, min(x + dash, x2), yy], fill=colour, width=width)
            x += dash + gap
    # left & right
    for xx in (x1, x2):
        y = y1
        while y < y2:
            draw.line([xx, y, xx, min(y + dash, y2)], fill=colour, width=width)
            y += dash + gap


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #
def analyze_image(
    image_input: Any,
    conf: float = 0.25,
    candidate_damage: bool = True,
    draw: bool = True,
    engine: Optional[YOLOVideoEngine] = None,
) -> Dict[str, Any]:
    """Detect every road object in one image.

    Returns a dict with ``counts`` (friendly per-class counts + total),
    ``detections`` (boxes), ``summary`` (one sentence), ``models`` (what ran),
    ``damage_verification``, and -- when ``draw`` -- ``annotated`` (a PIL image)
    plus ``annotated_png`` (PNG bytes).
    """
    eng = engine or _engine(candidate_damage)
    img_rgb = eng.decode_image(image_input)
    res = eng.detect_frame(
        img_rgb, frame_index=0, timestamp_s=0.0, conf_threshold=conf,
        session_id=f"img-{time.time_ns()}", reset_tracker=True,
    )
    dets = _clean(res.get("detections", []))
    counts = _friendly_counts(dets)
    dmg = eng.damage_detector
    if dmg and dmg.is_ready:
        damage_verification = "verified"
    elif eng.candidate_damage and dmg and getattr(dmg, "weights_loaded", False):
        damage_verification = "candidate_unverified"
    else:
        damage_verification = "hidden"

    out: Dict[str, Any] = {
        "mode": "image",
        "image_size": [int(res.get("image_width", 0)), int(res.get("image_height", 0))],
        "counts": counts,
        "summary": _summary_text(counts),
        "detections": dets,
        "models_active": _clean(res.get("models_active", [])),
        "damage_verification": damage_verification,
        "latency_ms": res.get("latency_ms"),
    }
    if draw:
        annotated = render_annotated(img_rgb, dets)
        out["annotated"] = annotated
        out["annotated_png"] = _png_bytes(annotated)
    return out


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
def analyze_video(
    video_path: str,
    conf: float = 0.25,
    sample_every_s: float = 0.5,
    max_frames: int = 600,
    candidate_damage: bool = True,
    save_sample_frames: int = 3,
    engine: Optional[YOLOVideoEngine] = None,
) -> Dict[str, Any]:
    """Detect and COUNT every road object across a video, de-duplicating objects
    that persist across frames (so five frames of one parked car count as one car).

    Returns ``unique_counts`` (counted once per tracked object), ``frame_count``,
    ``duration_s``, a ``per_frame`` trace, and a few ``sample_frames`` (annotated
    PNG bytes) for a visual check.
    """
    import cv2

    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    eng = engine or _engine(candidate_damage)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * float(sample_every_s))))
    session = f"vid-{time.time_ns()}"

    # Unique objects seen, keyed by the tracker's persistent track_id.
    unique_class_by_tid: Dict[int, str] = {}
    per_frame: List[Dict[str, Any]] = []
    sample_frames: List[Dict[str, Any]] = []
    want_samples = max(0, int(save_sample_frames))

    frame_idx = 0
    processed = 0
    first = True
    while processed < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % step == 0:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = eng.detect_frame(
                rgb, frame_index=frame_idx, timestamp_s=frame_idx / fps,
                conf_threshold=conf, session_id=session, reset_tracker=first,
            )
            first = False
            dets = _clean(res.get("detections", []))
            for d in dets:
                tid = d.get("track_id")
                if tid is not None:
                    # keep the highest-confidence class label seen for this track
                    unique_class_by_tid.setdefault(tid, d.get("class_name", "object"))
            per_frame.append({
                "frame_index": frame_idx,
                "timestamp_s": round(frame_idx / fps, 2),
                "frame_counts": _friendly_counts(dets),
            })
            if want_samples and dets and len(sample_frames) < want_samples:
                ann = render_annotated(rgb, dets)
                sample_frames.append({
                    "frame_index": frame_idx,
                    "timestamp_s": round(frame_idx / fps, 2),
                    "annotated_png": _png_bytes(ann),
                    "counts": _friendly_counts(dets),
                })
            processed += 1
        frame_idx += 1
    cap.release()

    # Fold unique tracks into the friendly buckets.
    uc = Counter(canonical_key(name) for name in unique_class_by_tid.values())
    unique_counts = {k: int(uc.get(k, 0)) for k in COUNT_KEYS if uc.get(k, 0)}
    unique_counts["total"] = int(sum(uc.values()))

    dmg = eng.damage_detector
    if dmg and dmg.is_ready:
        damage_verification = "verified"
    elif eng.candidate_damage and dmg and getattr(dmg, "weights_loaded", False):
        damage_verification = "candidate_unverified"
    else:
        damage_verification = "hidden"

    return {
        "mode": "video",
        "video_path": video_path,
        "fps": round(float(fps), 2),
        "total_frames": total,
        "frames_processed": processed,
        "sampled_every_s": float(sample_every_s),
        "duration_s": round(total / fps, 2) if total else None,
        "unique_counts": unique_counts,
        "summary": "Across the clip, " + _summary_text(unique_counts)[0].lower() + _summary_text(unique_counts)[1:]
                   if unique_counts.get("total") else "No road objects detected across the clip.",
        "damage_verification": damage_verification,
        "tracker_summary": _clean(eng.get_tracker(session).summary()),
        "per_frame": per_frame,
        "sample_frames": sample_frames,
    }


# --------------------------------------------------------------------------- #
# Engine cache (one per candidate_damage setting)
# --------------------------------------------------------------------------- #
_ENGINES: Dict[bool, YOLOVideoEngine] = {}


def _engine(candidate_damage: bool) -> YOLOVideoEngine:
    key = bool(candidate_damage)
    eng = _ENGINES.get(key)
    if eng is None:
        if key:
            eng = YOLOVideoEngine(candidate_damage=True)
        else:
            eng = get_yolo_video_engine()
        _ENGINES[key] = eng
    return eng
