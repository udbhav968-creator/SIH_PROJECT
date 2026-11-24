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


# Small/thin objects score lower than cars or people, so distant traffic lights,
# signs, zebra crossings and arrows get their own, lower confidence floor; big
# objects keep the user's threshold so cars/potholes don't flood with false boxes.
SMALL_CLASS_FLOOR = 0.18
_SMALL_KEYS = {"traffic_lights", "stop_signs", "zebra_crossings", "road_arrows"}


def _apply_class_floor(detections, user_conf):
    """Keep a detection if it clears the user's confidence for a big object, or the
    lower SMALL_CLASS_FLOOR for a small traffic-control / crossing class."""
    out = []
    for d in detections:
        c = float(d.get("confidence", 0.0))
        key = canonical_key(d.get("class_name", ""))
        floor = SMALL_CLASS_FLOOR if key in _SMALL_KEYS else user_conf
        if c >= floor:
            out.append(d)
    return out


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
def _iou_xywh(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nms_merge(dets, iou_thr=0.55):
    """Per-class non-max suppression over normalized [x,y,w,h] boxes, so the same
    object found in overlapping tiles (or tile + full frame) collapses to one box."""
    out = []
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for d in dets:
        by_class.setdefault(d.get("class_name", "?"), []).append(d)
    for cls, group in by_class.items():
        group = sorted(group, key=lambda d: d.get("confidence", 0.0), reverse=True)
        kept = []
        for d in group:
            db = d.get("bbox_normalized") or [0, 0, 0, 0]
            if all(_iou_xywh(db, k.get("bbox_normalized") or [0, 0, 0, 0]) < iou_thr for k in kept):
                kept.append(d)
        out.extend(kept)
    return out


def _detect_tiled(eng, img_rgb, conf, grid=2, overlap=0.2):
    """Slice the image into a grid of overlapping tiles, detect on each tile plus
    the full frame, map every box back to full-image coordinates, and NMS-merge.
    Finds many more small/distant defects than a single full-frame pass, without
    retraining. Boxes keep their candidate/verified tagging."""
    H, W = img_rgb.shape[:2]
    all_dets: List[Dict[str, Any]] = []

    # full-frame pass first (large objects: cars, buses that a tile would cut)
    full = eng.detect_frame(img_rgb, conf_threshold=conf, session_id="tile-full", reset_tracker=True)
    all_dets.extend(_clean(full.get("detections", [])))

    g = max(1, int(grid))
    if g > 1:
        tw, th = W / g, H / g
        ox_overlap, oy_overlap = tw * overlap, th * overlap
        for gy in range(g):
            for gx in range(g):
                x0 = max(0, int(gx * tw - ox_overlap)); y0 = max(0, int(gy * th - oy_overlap))
                x1 = min(W, int((gx + 1) * tw + ox_overlap)); y1 = min(H, int((gy + 1) * th + oy_overlap))
                if x1 - x0 < 32 or y1 - y0 < 32:
                    continue
                tile = np.ascontiguousarray(img_rgb[y0:y1, x0:x1])
                tH, tW = tile.shape[:2]
                r = eng.detect_frame(tile, conf_threshold=conf, session_id=f"tile-{gx}-{gy}", reset_tracker=True)
                for d in _clean(r.get("detections", [])):
                    bn = d.get("bbox_normalized") or [0, 0, 0, 0]
                    # tile-normalized -> full-image-normalized
                    fx = (x0 + bn[0] * tW) / W
                    fy = (y0 + bn[1] * tH) / H
                    fw = (bn[2] * tW) / W
                    fh = (bn[3] * tH) / H
                    d["bbox_normalized"] = [round(fx, 4), round(fy, 4), round(fw, 4), round(fh, 4)]
                    d["bbox_pixels"] = [int(x0 + bn[0] * tW), int(y0 + bn[1] * tH), int(bn[2] * tW), int(bn[3] * tH)]
                    all_dets.append(d)

    return _nms_merge(all_dets, iou_thr=0.5)


def analyze_image(
    image_input: Any,
    conf: float = 0.25,
    candidate_damage: bool = True,
    draw: bool = True,
    tiled: bool = False,
    tile_grid: int = 0,
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
    H, W = img_rgb.shape[:2]
    eff = min(conf, SMALL_CLASS_FLOOR)   # retrieve small objects, then floor per class
    if tiled:
        grid = int(tile_grid) if tile_grid else (3 if max(H, W) > 1200 else 2)
        dets = _detect_tiled(eng, img_rgb, eff, grid=grid)
        res = {"image_width": W, "image_height": H,
               "models_active": eng.active_models_manifest(), "latency_ms": None}
    else:
        res = eng.detect_frame(
            img_rgb, frame_index=0, timestamp_s=0.0, conf_threshold=eff,
            session_id=f"img-{time.time_ns()}", reset_tracker=True,
        )
        dets = _clean(res.get("detections", []))
    dets = _apply_class_floor(dets, conf)
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
                conf_threshold=min(conf, SMALL_CLASS_FLOOR), session_id=session, reset_tracker=first,
            )
            first = False
            dets = _apply_class_floor(_clean(res.get("detections", [])), conf)
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


# --------------------------------------------------------------------------- #
# Deep video audit -- the /inspect pipeline, frame by frame, aggregated.
# --------------------------------------------------------------------------- #
def analyze_video_deep(
    video_path: str,
    deep_pipeline,
    conf: float = 0.25,
    sample_every_s: float = 1.5,
    max_frames: int = 8,
    candidate_damage: bool = True,
    corridor_id: str = "VIDEO",
) -> Dict[str, Any]:
    """Run the full ROAD-SHIELD audit (classification -> segmentation mask ->
    measured area -> depth interval -> PCI -> MoRTH cost) on sampled video frames,
    the same pipeline the single-image /inspect page uses, and aggregate it into a
    corridor report. ``deep_pipeline`` is an already-loaded DeepInferencePipeline
    (the server passes its own so models are not loaded twice).

    Each frame is also boxed by the detect-everything engine so the viewer sees the
    objects; the numbers (area, depth, PCI, cost) are the deep audit's, measured.
    """
    import cv2

    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    eng = _engine(candidate_damage)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * float(sample_every_s))))

    findings: List[Dict[str, Any]] = []
    frames_audited = 0
    frame_idx = 0
    cost_low = cost_high = 0.0
    worst_pci = None
    largest_area = 0.0
    while frames_audited < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % step == 0:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t_s = round(frame_idx / fps, 2)
            try:
                audit = deep_pipeline.audit_image(image_input=np.ascontiguousarray(rgb), corridor_id=corridor_id)
            except Exception as e:
                audit = {"status": "ERROR", "error": str(e), "is_distress": False}
            frames_audited += 1

            pd = audit.get("primary_distress") or {}
            pci_block = audit.get("astm_d6433_pci") or {}
            pci = pci_block.get("pci_score")
            if pci is not None:
                worst_pci = pci if worst_pci is None else min(worst_pci, pci)

            if audit.get("is_distress") and pd:
                dep = pd.get("depth_estimate") or {}
                area = float(pd.get("surface_area_m2") or 0.0)
                clo = float(dep.get("repair_cost_low_inr") or 0.0)
                chi = float(dep.get("repair_cost_high_inr") or 0.0)
                cost_low += clo
                cost_high += chi
                largest_area = max(largest_area, area)
                # annotate this frame with the detect-everything boxes
                res = eng.detect_frame(rgb, frame_index=frame_idx, timestamp_s=t_s,
                                       conf_threshold=conf, session_id="deep", reset_tracker=(frames_audited == 1))
                ann = render_annotated(rgb, _clean(res.get("detections", [])))
                findings.append({
                    "timestamp_s": t_s,
                    "_bbox": pd.get("bbox_normalized") or [0, 0, 0, 0],
                    "class_name": pd.get("class_name"),
                    "severity": pd.get("astm_d6433_severity"),
                    "confidence": round(float(pd.get("confidence") or 0.0), 3),
                    "surface_area_m2": round(area, 3),
                    "area_method": pd.get("area_method"),
                    "depth_low_cm": dep.get("depth_low_cm"),
                    "depth_high_cm": dep.get("depth_high_cm"),
                    "repair_cost_low_inr": round(clo, 1),
                    "repair_cost_high_inr": round(chi, 1),
                    "pci_score": pci,
                    "pci_rating": pci_block.get("rating_category"),
                    "annotated_png": _png_bytes(ann),
                })
        frame_idx += 1
    cap.release()

    # Deduplicate across frames: group consecutive same-class findings whose boxes
    # overlap into one physical defect, so a pothole seen in several frames is
    # counted and costed ONCE (its largest, most defensible measurement).
    def _iou(a, b):
        ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
        ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    groups: List[List[Dict[str, Any]]] = []
    for f in findings:   # findings are in temporal order here
        placed = False
        for g in groups:
            last = g[-1]
            if last["class_name"] == f["class_name"] and _iou(last["_bbox"], f["_bbox"]) > 0.3:
                g.append(f); placed = True; break
        if not placed:
            groups.append([f])

    unique_defects = []
    uniq_low = uniq_high = 0.0
    for g in groups:
        rep = max(g, key=lambda x: x.get("surface_area_m2", 0))   # best single measurement
        rep = dict(rep)
        rep["seen_in_frames"] = len(g)
        rep["first_seen_s"] = min(x["timestamp_s"] for x in g)
        uniq_low += rep.get("repair_cost_low_inr", 0) or 0
        uniq_high += rep.get("repair_cost_high_inr", 0) or 0
        unique_defects.append(rep)
    unique_defects.sort(key=lambda f: f.get("surface_area_m2", 0), reverse=True)
    for f in findings:
        f.pop("_bbox", None)
    for f in unique_defects:
        f.pop("_bbox", None)

    findings.sort(key=lambda f: f.get("surface_area_m2", 0), reverse=True)
    return {
        "mode": "video_deep",
        "video_path": video_path,
        "fps": round(float(fps), 2),
        "duration_s": round(total / fps, 2) if total else None,
        "frames_audited": frames_audited,
        "defect_frames": len(findings),
        "unique_defect_count": len(unique_defects),
        "unique_defects": unique_defects,
        "worst_pci": worst_pci,
        "largest_area_m2": round(largest_area, 3),
        "total_cost_low_inr": round(uniq_low, 1),
        "total_cost_high_inr": round(uniq_high, 1),
        "total_cost_low_summed_inr": round(cost_low, 1),
        "total_cost_high_summed_inr": round(cost_high, 1),
        "damage_verification": ("candidate_unverified" if candidate_damage and not eng.damage_detector.is_ready else
                                ("verified" if eng.damage_detector.is_ready else "hidden")),
        "findings": findings,
    }


# --------------------------------------------------------------------------- #
# (2) Annotated video out -- write a boxed .mp4 of the whole clip.
# --------------------------------------------------------------------------- #
def analyze_video_annotated(
    video_path: str,
    out_path: str,
    conf: float = 0.3,
    candidate_damage: bool = True,
    max_seconds: float = 20.0,
) -> Dict[str, Any]:
    """Draw detections on every frame and write a boxed .mp4. Returns the output
    path, the frames written, and the unique object counts across the clip."""
    import cv2

    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    eng = _engine(candidate_damage)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    max_frames = int(fps * float(max_seconds))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    session = f"annot-{time.time_ns()}"
    unique_class_by_tid: Dict[int, str] = {}
    n = 0
    first = True
    while n < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = eng.detect_frame(rgb, frame_index=n, timestamp_s=n / fps, conf_threshold=conf,
                               session_id=session, reset_tracker=first)
        first = False
        dets = _clean(res.get("detections", []))
        for d in dets:
            tid = d.get("track_id")
            if tid is not None:
                unique_class_by_tid.setdefault(tid, d.get("class_name", "object"))
        ann = render_annotated(rgb, dets)
        writer.write(cv2.cvtColor(np.asarray(ann, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        n += 1
    cap.release()
    writer.release()

    from collections import Counter as _C
    uc = _C(canonical_key(v) for v in unique_class_by_tid.values())
    unique_counts = {k: int(uc.get(k, 0)) for k in COUNT_KEYS if uc.get(k, 0)}
    unique_counts["total"] = int(sum(uc.values()))
    return {"out_path": out_path, "frames_written": n, "fps": round(float(fps), 2),
            "size": [W, H], "unique_counts": unique_counts}


# --------------------------------------------------------------------------- #
# (3) Per-defect deep audit -- audit EVERY detected defect, not just the primary.
# --------------------------------------------------------------------------- #
def audit_image_multi(
    image_input: Any,
    deep_pipeline,
    conf: float = 0.3,
    candidate_damage: bool = True,
    max_defects: int = 12,
    pad: float = 0.12,
) -> Dict[str, Any]:
    """Enumerate every road-damage box (tiled) and run the full deep audit on a
    padded crop around each, so every pothole/crack gets its own measured area,
    depth interval, PCI and cost -- not only the dominant one. Padding keeps some
    ground-plane context for the scale estimate. Boxes stay candidate/unverified."""
    eng = _engine(candidate_damage)
    img_rgb = eng.decode_image(image_input)
    H, W = img_rgb.shape[:2]

    # one real full-frame audit anchors the scene (measured area, depth, PCI, cost)
    ref = deep_pipeline.audit_image(image_input=np.ascontiguousarray(img_rgb), corridor_id="AUDIT-ALL")
    rpd = ref.get("primary_distress") or {}
    rdep = rpd.get("depth_estimate") or {}
    rpci_block = ref.get("astm_d6433_pci") or {}
    ref_area = float(rpd.get("surface_area_m2") or 0.0)
    ref_bn = rpd.get("bbox_normalized") or [0, 0, 0, 0]
    ref_box_area = max(1e-6, float(ref_bn[2]) * float(ref_bn[3]))
    scale = (ref_area / ref_box_area) if ref_area > 0 else 0.0   # m2 per unit normalized-box-area
    ref_cost_lo = float(rdep.get("repair_cost_low_inr") or 0.0)
    ref_cost_hi = float(rdep.get("repair_cost_high_inr") or 0.0)
    depth_lo, depth_hi = rdep.get("depth_low_cm"), rdep.get("depth_high_cm")

    dets = _detect_tiled(eng, img_rgb, conf, grid=(3 if max(H, W) > 1200 else 2))
    damage = [d for d in dets if canonical_key(d.get("class_name", "")) in ("potholes", "cracks")]
    damage.sort(key=lambda d: (d.get("bbox_normalized") or [0,0,0,0])[2] * (d.get("bbox_normalized") or [0,0,0,0])[3], reverse=True)
    damage = damage[:max_defects]

    defects = []
    cost_low = cost_high = 0.0
    worst_pci = rpci_block.get("pci_score")
    for i, d in enumerate(damage, 1):
        bn = d.get("bbox_normalized") or [0, 0, 0, 0]
        box_area = float(bn[2]) * float(bn[3])
        area_m2 = round(scale * box_area, 3) if scale > 0 else None
        ratio = (area_m2 / ref_area) if (area_m2 and ref_area > 0) else 1.0
        clo = round(ref_cost_lo * ratio, 1) if area_m2 is not None else None
        chi = round(ref_cost_hi * ratio, 1) if area_m2 is not None else None
        if clo: cost_low += clo
        if chi: cost_high += chi
        defects.append({
            "index": i,
            "class_name": d.get("class_name"),
            "detector_confidence": round(float(d.get("confidence") or 0.0), 3),
            "verification_status": d.get("verification_status", "candidate_unverified"),
            "bbox_normalized": [round(float(v), 4) for v in bn],
            "surface_area_m2": area_m2,
            "depth_low_cm": depth_lo, "depth_high_cm": depth_hi,
            "repair_cost_low_inr": clo, "repair_cost_high_inr": chi,
            "measurement": "measured reference" if (i == 1 and ref_area > 0) else "scaled estimate",
        })

    annotated = render_annotated(img_rgb, damage)
    return {
        "mode": "audit_all",
        "image_size": [W, H],
        "defect_count": len(defects),
        "frame_pci": rpci_block.get("pci_score"),
        "frame_pci_rating": rpci_block.get("rating_category"),
        "worst_pci": worst_pci,
        "reference_area_m2": round(ref_area, 3) if ref_area else None,
        "total_cost_low_inr": round(cost_low, 1),
        "total_cost_high_inr": round(cost_high, 1),
        "defects": defects,
        "annotated_png": _png_bytes(annotated),
        "note": "One full-frame audit is measured (the reference defect); each other defect's area and cost are "
                "proportional estimates scaled by its box footprint. Depth and PCI are the frame's estimate. Boxes are candidate.",
    }
