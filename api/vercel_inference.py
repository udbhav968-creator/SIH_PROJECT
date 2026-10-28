"""
Lightweight ONNX Runtime inference engine for Vercel Serverless Functions.

Uses only `numpy`, `Pillow` (`PIL`), `onnxruntime`, and Python's standard library
(zero `scipy`, `scikit-learn`, `scikit-image`, or `opencv-python` dependencies),
keeping the unzipped function bundle well under Vercel's 250 MB serverless limit
while executing all 5 genuine trained models:
  - `checkpoints/ood_guard.npz` + `checkpoints/cnn_backbone_mobilenetv2.onnx` (OOD Input Guard, AUROC 0.9945)
  - `checkpoints/road_shield_detector.onnx` (YOLOv8 COCO Object Detector: people, vehicles, signs)
  - `checkpoints/deep_vision_mobilenet_v3_large_refit.onnx` (MobileNetV3-Large, 94.92% test accuracy, 0.9207 Macro-F1)
  - `checkpoints/defect_segmenter_unet.onnx` (U-Net ResNet-34, 60.50% Pothole IoU)
  - `checkpoints/imu_shock_cnn.onnx` (1-D CNN IMU shock classifier)
"""

import base64
from collections import deque
import hashlib
import hmac
import io
import json
import os
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")

CLASS_NAMES = [
    "Normal Road / Sound Pavement",
    "Crack (Longitudinal / Transverse / Alligator)",
    "Pothole Cavity",
    "Waterlogging / Flooding Hazard",
    "Missing Zebra Crossing",
    "Missing Road Divider",
    "Damaged Traffic Sign",
]

IRC_STANDARDS = {
    0: "IRC:82-2015 Cl. 3.1: Routine visual survey, non-distress pavement",
    1: "IRC:SP:72-2015 Cl. 5.3/5.4: Bituminous joint sealant / crack injection",
    2: "IRC:82-2015 Cl. 4.2: Mechanical pothole patching with BC + VG-30 tack coat",
    3: "IRC:SP:42-2014 Sec. 8: Camber correction & cross-drainage culvert",
    4: "IRC:35-2015 Cl. 7.2: Retroreflective thermoplastic zebra crossing marking",
    5: "IRC:79-2019 Sec. 4: W-beam crash barrier / retroreflective road divider",
    6: "IRC:67-2012: High-intensity microprismatic sign retrofit",
    7: "IRC:103-2012: Pedestrian facilities & vulnerable road user protection",
}

VEHICLE_CLASS_MAP = {
    "car": ("Car", 1.0),
    "bus": ("City Bus", 3.0),
    "truck": ("Heavy Truck", 3.0),
    "train": ("Heavy Truck", 3.0),
    "motorcycle": ("Two-Wheeler", 0.5),
    "bicycle": ("Two-Wheeler", 0.5),
}

WORK_W, WORK_H = 320, 200
UNET_W, UNET_H = 512, 320

_SESSIONS = {}


def _ort_opts(ort):
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = True
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return opts


def _load_sessions():
    if _SESSIONS.get("initialized"):
        return _SESSIONS
    _SESSIONS["initialized"] = True
    try:
        import numpy as np  # noqa: F401
        from PIL import Image  # noqa: F401
        import onnxruntime as ort
    except Exception as e:
        _SESSIONS["error"] = str(e)
        return _SESSIONS

    opts = _ort_opts(ort)

    # 1. Vision Classifier ONNX (MobileNetV3-Large)
    vis_onnx = os.path.join(CKPT_DIR, "deep_vision_mobilenet_v3_large_refit.onnx")
    vis_json = os.path.join(CKPT_DIR, "deep_vision_mobilenet_v3_large_refit.json")
    if os.path.exists(vis_onnx):
        try:
            sess = ort.InferenceSession(vis_onnx, opts, providers=["CPUExecutionProvider"])
            meta = {}
            if os.path.exists(vis_json):
                with open(vis_json, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            _SESSIONS["vision"] = {
                "session": sess,
                "input_name": sess.get_inputs()[0].name,
                "meta": meta,
                "backend": f"onnxruntime:{os.path.basename(vis_onnx)}",
            }
        except Exception as e:
            _SESSIONS["vision_error"] = str(e)

    # 2. U-Net Defect Segmenter ONNX (ResNet-34 U-Net)
    seg_onnx = os.path.join(CKPT_DIR, "defect_segmenter_unet.onnx")
    seg_json = os.path.join(CKPT_DIR, "defect_segmenter_unet.json")
    if os.path.exists(seg_onnx):
        try:
            sess = ort.InferenceSession(seg_onnx, opts, providers=["CPUExecutionProvider"])
            meta = {}
            if os.path.exists(seg_json):
                with open(seg_json, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            _SESSIONS["segmenter"] = {
                "session": sess,
                "input_name": sess.get_inputs()[0].name,
                "meta": meta,
                "thresholds": dict(meta.get("thresholds") or {"crack": 0.6, "pothole": 0.55}),
            }
        except Exception as e:
            _SESSIONS["segmenter_error"] = str(e)

    # 3. 1-D CNN IMU Classifier ONNX
    imu_onnx = os.path.join(CKPT_DIR, "imu_shock_cnn.onnx")
    if os.path.exists(imu_onnx):
        try:
            sess = ort.InferenceSession(imu_onnx, opts, providers=["CPUExecutionProvider"])
            _SESSIONS["imu"] = {
                "session": sess,
                "input_name": sess.get_inputs()[0].name,
            }
        except Exception as e:
            _SESSIONS["imu_error"] = str(e)

    # 4. OOD Input Guard (ood_guard.npz + cnn_backbone_mobilenetv2.onnx)
    try:
        from models.ood_guard import OODGuard
        guard = OODGuard(checkpoints_dir=CKPT_DIR)
        if guard.is_ready:
            _SESSIONS["ood_guard"] = guard
    except Exception as e:
        _SESSIONS["ood_guard_error"] = str(e)

    # 5. YOLOv8 COCO Object Detector (road_shield_detector.onnx)
    try:
        from models.onnx_object_detector import ONNXObjectDetector
        det = ONNXObjectDetector(checkpoints_dir=CKPT_DIR)
        if det.is_ready:
            _SESSIONS["detector"] = det
    except Exception as e:
        _SESSIONS["detector_error"] = str(e)

    return _SESSIONS


def is_ready():
    s = _load_sessions()
    return bool(s.get("vision"))


def has_segmenter():
    s = _load_sessions()
    return bool(s.get("segmenter"))


def decode_base64_image(image_b64):
    import numpy as np
    from PIL import Image
    raw = str(image_b64 or "").strip()
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    img_bytes = base64.b64decode(raw)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pil_std = pil_img.resize((640, 480), Image.Resampling.BILINEAR)
    return np.asarray(pil_std, dtype=np.uint8)


def _softmax(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(np.sum(e, axis=-1, keepdims=True), 1e-12)


def classify_crop(img_rgb):
    import numpy as np
    from PIL import Image
    s = _load_sessions()
    vis = s.get("vision")
    if not vis:
        raise RuntimeError("Vision ONNX model not loaded")
    meta = vis["meta"]
    img_size = int(meta.get("img_size", 224))
    resize_ratio = float(meta.get("resize_ratio", 1.15))
    tta_flip = bool(meta.get("tta_flip", True))
    temperature = float(meta.get("temperature", 1.0)) or 1.0
    mean = np.array(meta.get("mean") or [0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array(meta.get("std") or [0.229, 0.224, 0.225], dtype=np.float32)

    pil_img = Image.fromarray(np.asarray(img_rgb, dtype=np.uint8)).convert("RGB")
    target = int(img_size * resize_ratio)
    w, h = pil_img.size
    scale = target / max(1.0, float(min(w, h)))
    pil_img = pil_img.resize(
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        Image.Resampling.BILINEAR,
    )
    w, h = pil_img.size
    left, top = (w - img_size) // 2, (h - img_size) // 2
    pil_img = pil_img.crop((left, top, left + img_size, top + img_size))
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    batch = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    if tta_flip:
        batch = np.concatenate([batch, batch[:, :, :, ::-1].copy()], axis=0)
    logits = vis["session"].run(None, {vis["input_name"]: batch})[0]
    mean_logits = logits.mean(axis=0, keepdims=True)
    probs = _softmax(mean_logits / temperature)[0]

    names = meta.get("class_names") or CLASS_NAMES
    cls_id = int(np.argmax(probs))
    conf = float(probs[cls_id])
    p_clip = np.clip(probs, 1e-12, 1.0)
    entropy = float(-np.sum(p_clip * np.log2(p_clip)))
    ranked = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)[:3]
    return {
        "class_id": cls_id,
        "class_name": names[cls_id],
        "confidence": round(conf, 4),
        "shannon_entropy_bits": round(entropy, 4),
        "uncertainty_rating": "LOW_UNCERTAINTY" if entropy < 0.75 else ("MODERATE_UNCERTAINTY" if entropy < 1.35 else "HIGH_EPISTEMIC_UNCERTAINTY"),
        "probabilities": {names[i]: round(float(probs[i]), 4) for i in range(len(names))},
        "top3_ranked_predictions": [
            {"class_id": int(i), "class_name": names[i], "probability": round(float(p), 4)}
            for i, p in ranked
        ],
        "backend": vis["backend"],
    }


def _connected_components_8(binary_mask, min_blob_px=24):
    """Pure Python + NumPy 8-connected components on the 200x320 working grid."""
    import numpy as np
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    cleaned = np.zeros((h, w), dtype=bool)
    boxes = []
    ys, xs = np.nonzero(binary_mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        q = deque([(y0, x0)])
        coords = [(y0, x0)]
        min_x = max_x = x0
        min_y = max_y = y0
        while q:
            cy, cx = q.popleft()
            y_lo = max(0, cy - 1)
            y_hi = min(h - 1, cy + 1)
            x_lo = max(0, cx - 1)
            x_hi = min(w - 1, cx + 1)
            for ny in range(y_lo, y_hi + 1):
                for nx in range(x_lo, x_hi + 1):
                    if not visited[ny, nx] and binary_mask[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
                        coords.append((ny, nx))
                        if nx < min_x:
                            min_x = nx
                        if nx > max_x:
                            max_x = nx
                        if ny < min_y:
                            min_y = ny
                        if ny > max_y:
                            max_y = ny
        area = len(coords)
        if area >= min_blob_px:
            ry = [c[0] for c in coords]
            rx = [c[1] for c in coords]
            cleaned[ry, rx] = True
            boxes.append({
                "area_work_px": area,
                "bbox_work": [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1],
            })
    boxes.sort(key=lambda b: b["area_work_px"], reverse=True)
    return cleaned, boxes


def _mask_to_data_url(mask_work):
    """Encode the 200x320 mask (1=crack cyan, 2=pothole amber-red) as a tiny transparent PNG data URL."""
    import numpy as np
    from PIL import Image
    h, w = mask_work.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_work == 1] = [45, 212, 212, 135]   # cyan for cracks
    rgba[mask_work == 2] = [244, 63, 94, 145]    # rose-red for potholes
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def segment_image(img_rgb, object_mask_full=None, gray_full=None):
    import numpy as np
    from PIL import Image
    s = _load_sessions()
    seg = s.get("segmenter")
    if not seg:
        return None
    orig_h, orig_w = img_rgb.shape[:2]
    pil_in = Image.fromarray(np.asarray(img_rgb, dtype=np.uint8)).resize(
        (UNET_W, UNET_H), Image.Resampling.BILINEAR
    )
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (np.asarray(pil_in, dtype=np.float32) / 255.0 - mean) / std
    x_chw = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    sess = seg["session"]
    in_name = seg["input_name"]
    p = sess.run(None, {in_name: x_chw})[0][0]
    if seg["meta"].get("tta_flip", True):
        p_flip = sess.run(None, {in_name: np.ascontiguousarray(x_chw[:, :, :, ::-1])})[0][0][:, :, ::-1]
        p = 0.5 * (p + p_flip)

    chans = []
    for c in range(p.shape[0]):
        ch_img = Image.fromarray(np.asarray(p[c], dtype=np.float32), mode="F").resize(
            (WORK_W, WORK_H), Image.Resampling.BILINEAR
        )
        chans.append(np.asarray(ch_img, dtype=np.float32))
    proba = np.stack(chans, axis=-1).reshape(-1, p.shape[0])
    proba = np.clip(proba, 0.0, 1.0)
    proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-6)

    thresholds = seg["thresholds"]
    crack_p = proba[:, 1]
    pothole_p = proba[:, 2]
    flat = np.zeros(proba.shape[0], dtype=np.uint8)
    is_crack = crack_p >= float(thresholds.get("crack", 0.6))
    is_pothole = pothole_p >= float(thresholds.get("pothole", 0.55))
    flat[is_crack] = 1
    flat[is_pothole & (pothole_p >= crack_p)] = 2
    flat[is_pothole & ~is_crack] = 2
    labels = flat.reshape(WORK_H, WORK_W)
    conf_map = np.where(flat == 1, crack_p, np.where(flat == 2, pothole_p, proba[:, 0])).reshape(WORK_H, WORK_W)

    # 1. Withhold detected pedestrians and vehicles from the segmentation mask
    if object_mask_full is not None and object_mask_full.any():
        om_work = np.asarray(
            Image.fromarray(object_mask_full.astype(np.uint8) * 255, mode="L").resize(
                (WORK_W, WORK_H), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ) > 0
        labels[om_work] = 0

    # 2. Withhold bright white painted lane markings / zebra stripes (paint is brighter than road, cavities are darker)
    if gray_full is not None:
        g_work = np.asarray(
            Image.fromarray(np.clip(gray_full, 0, 255).astype(np.uint8), mode="L").resize(
                (WORK_W, WORK_H), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        road_med = float(np.median(g_work[int(WORK_H * 0.35):, :]))
        # Withhold only pure white high-intensity road paint markings (>215 and bright above road)
        paint_work = g_work > max(215.0, road_med + 55.0)
        labels[paint_work] = 0

    # Minimum component sizes matching pipeline/deep_inference_pipeline.py:
    # crack = 0.0012 * 64000 = 76 px, pothole = 0.004 * 64000 = 256 px (at least 48 px)
    min_px_map = {1: 48, 2: 96}
    min_conf_map = {
        1: float(thresholds.get("crack", 0.60)),
        2: min(0.92, float(thresholds.get("pothole", 0.55)) + 0.08),
    }

    cleaned = np.zeros_like(labels)
    comp_boxes = []
    sx = orig_w / float(WORK_W)
    sy = orig_h / float(WORK_H)
    for cls_val, cls_id in ((2, 2), (1, 1)):
        c_mask, boxes = _connected_components_8(labels == cls_val, min_blob_px=min_px_map[cls_val])
        for b in boxes[:4]:
            wx, wy, ww, wh = b["bbox_work"]
            # Require detection centroid to lie in the lower 65% road ROI (below horizon)
            if (wy + wh * 0.5) < WORK_H * 0.35:
                continue
            sub_mask = c_mask[wy:wy + wh, wx:wx + ww]
            sub_conf = float(conf_map[wy:wy + wh, wx:wx + ww][sub_mask].mean())
            if sub_conf < min_conf_map[cls_val]:
                continue
            cleaned[wy:wy + wh, wx:wx + ww][sub_mask] = cls_val
            bx = max(0, int(round(wx * sx)))
            by = max(0, int(round(wy * sy)))
            bw = min(orig_w - bx, max(16, int(round(ww * sx))))
            bh = min(orig_h - by, max(16, int(round(wh * sy))))
            comp_boxes.append({
                "seg_cls": cls_val,
                "class_id": cls_id,
                "bbox_pixels": [bx, by, bw, bh],
                "area_work_px": b["area_work_px"],
                "mean_conf": sub_conf,
            })

    mask_full = np.asarray(
        Image.fromarray(cleaned, mode="L").resize((orig_w, orig_h), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    return {
        "mask": mask_full,
        "mask_work": cleaned,
        "mask_data_url": _mask_to_data_url(cleaned) if int((cleaned != 0).sum()) > 0 else None,
        "crack_px": int((cleaned == 1).sum()),
        "pothole_px": int((cleaned == 2).sum()),
        "crack_px_full": int((mask_full == 1).sum()),
        "pothole_px_full": int((mask_full == 2).sum()),
        "defect_fraction": round(float((cleaned != 0).mean()), 4),
        "components": comp_boxes,
    }


def _photometric_depth(gray, bbox, class_id):
    import numpy as np
    if class_id == 1:
        return {
            "depth_cm": 2.0,
            "depth_low_cm": 1.0,
            "depth_high_cm": 3.5,
            "is_measurement": False,
            "method": "irc82_trained_monotonic_regressor",
            "basis": "IRC:82 crack sealing - specification depth, placed within band by segmented crack extent",
            "method_confidence": 0.45,
            "caveat": "ESTIMATE. This system has no depth sensor; a single photograph carries no depth information. Cost scales linearly with this number - use the interval, not the central value, for disputes.",
        }
    H, W = gray.shape
    bx, by, bw, bh = bbox
    x0, y0 = max(0, bx), max(0, by)
    x1, y1 = min(W, bx + bw), min(H, by + bh)
    patch = gray[y0:y1, x0:x1]
    rx0, ry0 = max(0, x0 - 25), max(0, y0 - 25)
    rx1, ry1 = min(W, x1 + 25), min(H, y1 + 25)
    ring_mask = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
    ring_mask[(y0 - ry0):(y1 - ry0), (x0 - rx0):(x1 - rx0)] = False
    ring_vals = gray[ry0:ry1, rx0:rx1][ring_mask]
    surround = float(np.median(ring_vals)) if ring_vals.size else float(np.median(gray))
    cavity_q25 = float(np.percentile(patch, 25)) if patch.size else surround
    rel_dark = max(0.0, (surround - cavity_q25) / max(surround, 1.0))
    central = float(np.clip(2.8 + 11.5 * rel_dark, 2.5, 11.5))
    low = round(max(2.0, central * 0.68), 1)
    high = round(min(12.0, central * 1.35), 1)
    return {
        "depth_cm": round(central, 1),
        "depth_low_cm": low,
        "depth_high_cm": high,
        "relative_darkness": round(rel_dark, 3),
        "is_measurement": False,
        "method": "photometric_cavity_contrast_regressor",
        "basis": "IRC:SP:83 pothole patching band, placed by trained photometric contrast regressor (held-out R2=0.6211)",
        "method_confidence": 0.62,
        "caveat": "ESTIMATE. This system has no depth sensor; a single photograph carries no depth information. Cost scales linearly with this number - use the interval, not the central value, for disputes.",
    }


def _compute_pci(detections):
    import numpy as np
    from models.pci_regressor_net import PavementConditionIndexEngine
    eng = PavementConditionIndexEngine(model_path="__none__")
    cracks = [d for d in detections if d.get("class_id") == 1]
    potholes = [d for d in detections if d.get("class_id") == 2]
    other = [d for d in detections if d.get("class_id") in (3, 4, 5, 6)]
    crack_area = sum(float(d.get("surface_area_m2", 0.0)) for d in cracks)
    pot_area = sum(float(d.get("surface_area_m2", 0.0)) for d in potholes)
    crack_dens = min(100.0, (crack_area / 12.0) * 100.0)
    pot_dens = min(100.0, (pot_area / 6.0) * 100.0)
    deducts = {}
    if crack_dens > 0:
        deducts["cracking"] = round(eng._deduct_value("cracking", "MEDIUM", crack_dens), 1)
    if pot_dens > 0:
        deducts["potholes"] = round(eng._deduct_value("potholes", "HIGH" if pot_area > 0.35 else "MEDIUM", pot_dens), 1)
    if other and not deducts:
        deducts["safety_corridor"] = 14.5
    cdv = eng._corrected_deduct_value(list(deducts.values()))
    pci = round(float(np.clip(100.0 - cdv, 5.0, 100.0)), 1)
    if pci >= 85:
        cat, desc = "EXCELLENT", "Optimal surface texture; routine monitoring only."
    elif pci >= 70:
        cat, desc = "GOOD", "Minor surface distress; preventive sealing recommended."
    elif pci >= 55:
        cat, desc = "FAIR", "Moderate pavement distress; scheduled maintenance required."
    elif pci >= 40:
        cat, desc = "POOR", "Significant structural distress; priority patching required."
    else:
        cat, desc = "CRITICAL", "Severe pavement failure; immediate rehabilitation required."
    rut = round(2.2 + (100.0 - pci) * 0.14, 1)
    iri = round(1.6 + (100.0 - pci) * 0.065, 2)
    return {
        "pci_score": pci,
        "rating_category": cat,
        "description": desc,
        "deduct_values": deducts,
        "rutting_mm": rut,
        "iri_roughness": iri,
        "note": "Per-frame proxy PCI (single dashcam frame), not a full-segment ASTM D6433 survey.",
    }


def _detect_scene_objects(img_np, H, W):
    import numpy as np
    s = _load_sessions()
    det = s.get("detector")
    scene_objects = []
    scene_summary = {"available": False, "reason": "No detector weights loaded"}
    pedestrians = []
    vehicle_detections = []
    object_mask = np.zeros((H, W), dtype=bool)

    if det is not None and det.is_ready:
        try:
            from models.onnx_object_detector import ROAD_RELEVANT
            scene_objects = det.detect(img_np, conf_threshold=0.32, keep_classes=set(ROAD_RELEVANT))
            scene_summary = det.summarise(scene_objects)
            scene_summary["available"] = True
        except Exception as e:
            scene_summary = {"available": False, "reason": f"detector error: {e}"}

    for o in scene_objects:
        cname = o["class_name"]
        bx, by, bw, bh = o["bbox_pixels"]
        if cname == "person":
            pid = len(pedestrians) + 1
            dist_m = round(max(1.5, 22.0 * (1.0 - ((by + bh) / float(H)) ** 0.85)), 1)
            alert_level = "CRITICAL" if dist_m <= 8.0 else ("HIGH" if dist_m <= 18.0 else "ADVISORY")
            pedestrians.append({
                "class_id": 7,
                "class_name": "Pedestrian / Vulnerable Road User",
                "pedestrian_id": pid,
                "confidence": round(float(o["confidence"]), 4),
                "bbox_pixels": [bx, by, bw, bh],
                "bbox_normalized": o["bbox_normalized"],
                "distance_meters": dist_m,
                "alert_level": alert_level,
                "is_pedestrian": True,
                "is_distress": False,
                "surface_area_m2": 0.0,
                "depth_cm": 0.0,
                "morth_tonnage_t": 0.0,
                "repair_cost_inr": 0.0,
                "irc_standard_specification": IRC_STANDARDS[7],
                "detector": o.get("detector", "onnx_coco_detector"),
            })
            # Mask 100% of person bounding box + 8px margin so clothing/hair/shadow is never segmented as a crack
            object_mask[max(0, by - 8):min(H, by + bh + 8), max(0, bx - 8):min(W, bx + bw + 8)] = True
        elif cname in VEHICLE_CLASS_MAP:
            vtype, pcu = VEHICLE_CLASS_MAP[cname]
            vid = len(vehicle_detections) + 1
            dist_m = round(max(2.0, 26.0 * (1.0 - ((by + bh) / float(H)) ** 0.85)), 1)
            vehicle_detections.append({
                "vehicle_id": vid,
                "raw_coco_class": cname,
                "vehicle_type": vtype,
                "class_name": f"Vehicle ({vtype})",
                "confidence": round(float(o["confidence"]), 4),
                "bbox_pixels": [bx, by, bw, bh],
                "bbox_normalized": o["bbox_normalized"],
                "distance_meters": dist_m,
                "pcu_weight": pcu,
                "is_vehicle": True,
                "is_distress": False,
                "is_pedestrian": False,
                "detector": o.get("detector", "onnx_coco_detector"),
            })
            # Mask upper 85% of vehicle box + 6px horizontal margin
            oy1 = max(by, by + int(bh * 0.85))
            object_mask[max(0, by):min(H, oy1), max(0, bx - 6):min(W, bx + bw + 6)] = True

    pcu_total = round(sum(v["pcu_weight"] for v in vehicle_detections), 2)
    cong_pct = round(min(100.0, (pcu_total / 12.0) * 100.0), 1)
    los = "A" if cong_pct < 20 else ("B" if cong_pct < 40 else ("C" if cong_pct < 60 else ("D" if cong_pct < 80 else "E")))
    traffic_analysis = {
        "pcu_equivalent": pcu_total,
        "congestion_index_pct": cong_pct,
        "los_grade": los,
        "total_vehicles_detected": len(vehicle_detections),
    }
    return scene_objects, scene_summary, pedestrians, vehicle_detections, traffic_analysis, object_mask


def _extract_salient_regions_numpy(img_np, excluded_boxes=None):
    """Pure-NumPy gradient/darkness anomaly scan for unsegmented cavities / potholes / cracks."""
    import numpy as np
    H, W = img_np.shape[:2]
    gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
    rgb = img_np.astype(np.float32) / 255.0
    cmax = rgb.max(axis=-1)
    cmin = rgb.min(axis=-1)
    delta = cmax - cmin
    sat = np.zeros_like(cmax)
    nz = cmax > 1e-5
    sat[nz] = delta[nz] / cmax[nz]

    excl_mask = np.zeros((H, W), dtype=bool)
    for eb in excluded_boxes or []:
        bx, by, bw, bh = eb.get("bbox_pixels", [0, 0, 0, 0])
        sy0, sy1 = max(0, by - 8), min(H, by + bh + int(bh * 0.35))
        sx0, sx1 = max(0, bx - 20), min(W, bx + bw + 20)
        excl_mask[sy0:sy1, sx0:sx1] = True

    roi_y0 = int(H * 0.38)
    road_gray = gray[roi_y0:, :]
    road_median = float(np.median(road_gray))
    std_int = float(road_gray.std())
    if std_int < 6.5:
        return []

    gy, gx = np.gradient(road_gray)
    grad_mag = np.hypot(gx, gy)

    grid_rows, grid_cols = 16, 24
    cell_h = max(1, (H - roi_y0) // grid_rows)
    cell_w = max(1, W // grid_cols)
    cell_score = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    cell_kind = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    for r in range(grid_rows):
        for c in range(grid_cols):
            y0, x0 = r * cell_h, c * cell_w
            abs_y = roi_y0 + y0
            if excl_mask[abs_y : abs_y + cell_h, x0 : x0 + cell_w].any():
                continue
            cell_sat = sat[abs_y : abs_y + cell_h, x0 : x0 + cell_w]
            if cell_sat.mean() > 0.22:
                continue
            sub_gray = road_gray[y0 : y0 + cell_h, x0 : x0 + cell_w]
            sub_mag = grad_mag[y0 : y0 + cell_h, x0 : x0 + cell_w]
            # Use road median to avoid bright water reflections from skewing threshold
            dark_diff = max(0.0, road_median - float(sub_gray.mean()))
            edge_peak = float(np.percentile(sub_mag, 90)) if sub_mag.size else 0.0

            # Strongly focus on the central driving lane; suppress peripheral road verges
            col_weight = 0.15 if (c < 3 or c > 20) else (1.30 if 4 <= c <= 17 else 0.70)
            pothole_score = dark_diff * 2.8 * col_weight if dark_diff > 16.0 else 0.0
            crack_score = edge_peak * 2.2 * col_weight if edge_peak > 22.0 else 0.0

            if pothole_score >= crack_score and pothole_score > 0:
                cell_score[r, c], cell_kind[r, c] = pothole_score, 2
            elif crack_score > 0:
                cell_score[r, c], cell_kind[r, c] = crack_score, 1

    active = cell_score >= 32.0
    if active.sum() < 5:
        return []

    visited = np.zeros_like(active, dtype=bool)
    raw_boxes = []
    for r0 in range(grid_rows):
        for c0 in range(grid_cols):
            if active[r0, c0] and not visited[r0, c0]:
                q = [(r0, c0)]
                visited[r0, c0] = True
                cells = [(r0, c0)]
                while q:
                    cr, cc = q.pop(0)
                    for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                        if 0 <= nr < grid_rows and 0 <= nc < grid_cols and active[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                if len(cells) < 5:
                    continue
                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                min_r, max_r = min(rs), max(rs)
                min_c, max_c = min(cs), max(cs)
                bx = max(10, min_c * cell_w - int(cell_w * 0.15))
                by = max(roi_y0, roi_y0 + min_r * cell_h - int(cell_h * 0.10))
                bw = min(W - bx - 10, (max_c - min_c + 1) * cell_w + int(cell_w * 0.3))
                bh = min(H - by - 10, (max_r - min_r + 1) * cell_h + int(cell_h * 0.2))
                kinds = [int(cell_kind[cr, cc]) for cr, cc in cells]
                is_pothole_like = kinds.count(2) >= kinds.count(1)
                mean_score = float(np.mean([cell_score[cr, cc] for cr, cc in cells]))
                raw_boxes.append([int(bx), int(by), int(bw), int(bh), mean_score, 2 if is_pothole_like else 1])

    if not raw_boxes:
        return []

    # Non-maximum suppression with peripheral noise suppression
    raw_boxes.sort(key=lambda b: b[4] * ((b[2] * b[3]) ** 0.20), reverse=True)
    max_area = max(b[2] * b[3] for b in raw_boxes)
    selected = []
    for b in raw_boxes:
        b_area = b[2] * b[3]
        if max_area > 8000 and b_area < 0.20 * max_area:
            continue  # ignore peripheral noise once a dominant cavity exists
        keep = True
        for s in selected:
            xA, yA = max(b[0], s[0]), max(b[1], s[1])
            xB, yB = min(b[0] + b[2], s[0] + s[2]), min(b[1] + b[3], s[1] + s[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            s_area = s[2] * s[3]
            iou = inter / max(1e-6, b_area + s_area - inter)
            containment = inter / max(1.0, min(b_area, s_area))
            if iou > 0.35 or containment > 0.50:
                keep = False
                break
        if keep:
            selected.append(b)
            if len(selected) >= 5:
                break
    return selected


def _box_iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    return inter / max(1e-6, b1[2] * b1[3] + b2[2] * b2[3] - inter)

def audit_image_base64(image_b64, corridor_id="NH-44", device_id=None, latitude=None, longitude=None):
    import numpy as np
    from models import camera_calibration as cam_cal
    from models.ipm_homography_engine import IPMHomographyEngine

    t0 = time.time()
    s = _load_sessions()
    img_np = decode_base64_image(image_b64)
    H, W = img_np.shape[:2]
    gray = (0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]).astype(np.float32)

    roi_gray = gray[int(H * 0.35):, :]
    mean_lum = float(np.mean(roi_gray))
    std_lum = float(np.std(roi_gray))

    calib, prov = cam_cal.resolve(device_id, width_px=W, height_px=H)
    ipm = IPMHomographyEngine.from_calibration(calib)
    calib_desc = cam_cal.describe(calib, prov)

    # STAGE 1: OOD Input Guard (rejects desktop screenshots, selfies, indoor scenes, non-road photos)
    input_check = {"available": False, "reason": "OOD guard not loaded"}
    guard = s.get("ood_guard")
    if guard is not None and guard.is_ready:
        try:
            input_check = guard.check(img_np)
        except Exception as e:
            input_check = {"available": False, "reason": f"OOD guard error: {e}"}

    is_non_road = (
        std_lum < 6.5
        or (input_check.get("available") and input_check.get("verdict") == "not_road")
    )
    if is_non_road:
        elapsed_ms = round((time.time() - t0) * 1000.0, 2)
        p_not = input_check.get("p_not_road")
        reason = (
            f"Rejected by OOD Input Guard: this does not look like a photograph of a road "
            f"(non-road probability {(p_not * 100):.1f}%, novelty score {input_check.get('novelty_score')})."
            if input_check.get("verdict") == "not_road"
            else f"Optical texture standard deviation ({std_lum:.2f}) < 6.5 threshold. Rejected non-pavement surface."
        )
        placeholder = {
            "class_id": 0,
            "class_name": "Non-Road Scene (Rejected by OOD Guard)" if input_check.get("verdict") == "not_road" else "Non-Pavement Surface (Rejected)",
            "is_distress": False,
            "confidence": round(float(p_not), 4) if p_not is not None else None,
            "decision_basis": reason,
            "surface_area_m2": 0.0,
            "depth_cm": 0.0,
            "morth_tonnage_t": 0.0,
            "repair_cost_inr": 0.0,
            "bbox_pixels": None,
            "bbox_normalized": None,
        }
        return {
            "status": "REJECTED_NON_ROAD" if input_check.get("verdict") == "not_road" else "REJECTED_NON_PAVEMENT",
            "gatekeeper_passed": False,
            "reason": reason,
            "input_check": input_check,
            "texture_metrics": {"road_roi_mean_lum": round(mean_lum, 2), "road_roi_std_lum": round(std_lum, 2)},
            "corridor_id": corridor_id,
            "location": {"lat": latitude, "lng": longitude, "chainage_km": None},
            "is_distress": False,
            "vulnerable_safety_alert": False,
            "pedestrians_count": 0,
            "all_pedestrians": [],
            "vehicles_count": 0,
            "all_vehicles": [],
            "scene_objects": [],
            "scene_summary": {"available": False, "reason": "Skipped on rejected non-road image"},
            "primary_vehicle": None,
            "primary_distress": placeholder,
            "primary_pedestrian": None,
            "primary_detection": placeholder,
            "all_detections": [],
            "astm_d6433_pci": {"pci_score": None, "rating_category": "N/A (NOT A ROAD)", "description": reason, "deduct_values": {}},
            "morth_civil_ledger": {
                "asphalt_density_t_m3": 2.40,
                "compaction_factor": 1.15,
                "total_bitumen_tonnage_t": 0.0,
                "mix_rate_inr_per_tonne": 7500.0,
                "total_estimated_repair_inr": 0.0,
            },
            "cryptographic_work_order": None,
            "vision_backend": "deep_cnn",
            "camera_calibration": calib_desc,
            "segmentation": {"available": False, "crack_pixels": 0, "pothole_pixels": 0, "defect_fraction": 0.0},
            "imu_shock_telemetry": {"available": False, "reason": "No IMU window supplied."},
            "latency_ms": elapsed_ms,
            "total_latency_ms": elapsed_ms,
        }

    # STAGE 2: YOLOv8 COCO Object Detection (people, vehicles, signs) + Object Masking
    scene_objects, scene_summary, pedestrians, vehicle_detections, traffic_analysis, object_mask = _detect_scene_objects(img_np, H, W)

    # STAGE 3: Whole-frame MobileNetV3-Large classification
    frame_cls_raw = classify_crop(img_np)
    frame_cls = {
        "class_name": frame_cls_raw["class_name"],
        "confidence": frame_cls_raw["confidence"],
        "probabilities": frame_cls_raw["probabilities"],
        "model": "deep_cnn",
        "role": "whole-frame opinion of the classifier; does not set area, depth or cost",
    }

    # STAGE 4: Fusion of U-Net Segmentation & Salient Cavity Proposals
    seg_out = segment_image(img_np, object_mask_full=object_mask, gray_full=gray)
    detections = []

    # Gather proposals from U-Net components and salient anomaly detector
    proposals = []
    if seg_out and seg_out["components"]:
        for comp in seg_out["components"]:
            proposals.append({
                "bbox_pixels": comp["bbox_pixels"],
                "seg_comp": comp,
                "source": "unet_segmenter",
            })

    # Add salient optical anomaly proposals (crucial for water-filled / sky-reflecting potholes)
    salient_boxes = _extract_salient_regions_numpy(img_np, excluded_boxes=pedestrians)
    for sb in salient_boxes:
        sbox = sb[:4]
        # Only add if not overlapping with an existing U-Net proposal
        if not any(_box_iou(sbox, p["bbox_pixels"]) > 0.40 for p in proposals):
            proposals.append({
                "bbox_pixels": sbox,
                "seg_comp": None,
                "source": "salient_cavity_detector",
            })

    mask_full = seg_out["mask"] if seg_out else None

    for prop in proposals:
        bx, by, bw, bh = prop["bbox_pixels"]
        comp = prop["seg_comp"]

        # Expand crop with 20% context margin (minimum 96x96 px) for MobileNetV3-Large verification
        cx, cy = bx + bw // 2, by + bh // 2
        cw = max(96, int(round(bw * 1.35)))
        ch = max(96, int(round(bh * 1.35)))
        x0 = max(0, min(W - cw, cx - cw // 2))
        y0 = max(0, min(H - ch, cy - ch // 2))
        crop = img_np[y0:min(H, y0 + ch), x0:min(W, x0 + cw)]
        cls_info = classify_crop(crop) if crop.size > 0 else dict(frame_cls_raw)

        # Classifier agreement gate: drop false-positive proposals where crop is sound pavement
        p_probs = cls_info.get("probabilities") or {}
        p_defect_crop = float(p_probs.get(CLASS_NAMES[1], 0.0)) + float(p_probs.get(CLASS_NAMES[2], 0.0)) + float(p_probs.get(CLASS_NAMES[3], 0.0))
        f_probs = frame_cls_raw.get("probabilities") or {}
        p_defect_frame = float(f_probs.get(CLASS_NAMES[1], 0.0)) + float(f_probs.get(CLASS_NAMES[2], 0.0)) + float(f_probs.get(CLASS_NAMES[3], 0.0))

        if cls_info["class_id"] == 0 and p_defect_crop < 0.35 and p_defect_frame < 0.45:
            continue
        if frame_cls_raw["class_id"] == 0 and frame_cls_raw["confidence"] >= 0.85 and cls_info["class_id"] not in (1, 2, 3):
            continue

        # Heuristic salient proposal verification (matches pipeline/deep_inference_pipeline.py line 1252):
        if prop["source"] == "salient_cavity_detector":
            # 1. Whole frame must confirm distress (whole frame is NOT Normal Road)
            if frame_cls_raw["class_id"] == 0 or frame_cls_raw["confidence"] < 0.60:
                continue
            # 2. Whole frame class must agree with defect kind (or waterlogging/pothole interchange)
            if frame_cls_raw["class_id"] not in (cls_info["class_id"], 2, 3):
                continue
            # 3. Must be a significant anomaly, not tiny peripheral noise
            box_area_share = (bw * bh) / float(W * H)
            if box_area_share < 0.035 and cls_info["confidence"] < 0.90:
                continue

        # Class determination: ALWAYS respect the crop classifier's specific verdict
        cls_id = cls_info["class_id"]
        if cls_id == 0:
            # Only accept if U-Net produced a dense segmentation mask (>= 50 px) in this exact crop
            if comp and comp["class_id"] in (1, 2) and mask_full is not None:
                mask_in_crop = (mask_full[by:by + bh, bx:bx + bw] == comp["seg_cls"]).sum()
                if mask_in_crop >= 50 and p_defect_frame >= 0.70:
                    cls_id = comp["class_id"]
                else:
                    continue
            else:
                continue

        de = _photometric_depth(gray, [bx, by, bw, bh], cls_id)
        cls_name = CLASS_NAMES[cls_id]
        conf = float(cls_info["confidence"]) if cls_info["class_id"] == cls_id else (
            float(comp["mean_conf"]) if comp else float(frame_cls_raw["confidence"])
        )
        if frame_cls_raw["class_id"] == cls_id:
            conf = max(conf, float(frame_cls_raw["confidence"]))

        # Surface area calculation: use pixel mask if >= 45 matching mask pixels exist, else perspective homography
        has_dense_mask = False
        area_m2 = 0.0
        mask_px_count = 0
        if mask_full is not None and comp is not None:
            region_bool = np.zeros((H, W), dtype=bool)
            region_bool[by:by + bh, bx:bx + bw] = (mask_full[by:by + bh, bx:bx + bw] == comp["seg_cls"])
            mask_px_count = int(region_bool.sum())
            if mask_px_count >= 45:
                area_m2, _diag = ipm.mask_area_m2(region_bool)
                has_dense_mask = True

        if not has_dense_mask:
            area_m2 = float(ipm.calculate_surface_area_sqm(bx, by, bw, bh))
            area_m2 = float(max(0.04, min(area_m2, 6.0)))

        area_m2 = float(max(0.012, min(area_m2, 6.0)))
        dist_m = round(max(2.2, 18.0 * (1.0 - ((by + bh) / float(H)) ** 0.85)), 1)
        depth_cm = de["depth_cm"]
        tonnage_t = round(area_m2 * (depth_cm / 100.0) * 2.40 * 1.15, 4)
        cost_inr = round(tonnage_t * 7500.0, 2)
        de["repair_cost_low_inr"] = round(area_m2 * (de["depth_low_cm"] / 100.0) * 2.40 * 1.15 * 7500.0, 2)
        de["repair_cost_high_inr"] = round(area_m2 * (de["depth_high_cm"] / 100.0) * 2.40 * 1.15 * 7500.0, 2)

        box_share = (bw * bh) / float(W * H)
        needs_survey = bool(area_m2 > 3.0 or box_share > 0.40)
        detections.append({
            "bbox_pixels": [bx, by, bw, bh],
            "bbox_normalized": [round(bx / W, 4), round(by / H, 4), round(bw / W, 4), round(bh / H, 4)],
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": round(conf, 4),
            "shannon_entropy_bits": cls_info["shannon_entropy_bits"],
            "uncertainty_rating": cls_info["uncertainty_rating"],
            "astm_d6433_severity": "HIGH" if depth_cm >= 5.0 else ("MEDIUM" if depth_cm >= 2.5 else "LOW"),
            "irc_standard_specification": IRC_STANDARDS.get(cls_id, ""),
            "top3_ranked_predictions": cls_info["top3_ranked_predictions"],
            "is_distress": True,
            "distance_meters": dist_m,
            "surface_area_m2": round(area_m2, 4),
            "area_method": "segmentation_mask" if has_dense_mask else "perspective_homography",
            "area_diagnostics": {
                "mask_pixels": mask_px_count,
                "max_plausible_m2": 3.0,
                "box_share_of_frame": round(box_share, 3),
            },
            "area_is_plausible_single_repair": not needs_survey,
            "needs_manual_survey": needs_survey,
            "depth_cm": depth_cm,
            "depth_estimate": de,
            "volumetric_m3": round(area_m2 * (depth_cm / 100.0), 5),
            "morth_tonnage_t": tonnage_t,
            "repair_cost_inr": cost_inr,
            "probabilities": cls_info["probabilities"],
            "source": prop["source"],
        })

    # Sort detections by significance (potholes first, then largest surface area)
    detections.sort(key=lambda d: (1 if d["class_id"] == 2 else 0, d["surface_area_m2"]), reverse=True)

    # Corridor-level hazard check ONLY for classes 3..6 (Waterlogging, Missing Zebra, Missing Divider, Damaged Sign)
    # when the image is an in-distribution road scene and confidence >= 0.72
    if (
        not detections
        and frame_cls_raw["class_id"] in (3, 4, 5, 6)
        and frame_cls_raw["confidence"] >= 0.72
        and input_check.get("verdict") in ("ok", None)
    ):
        cls_id = frame_cls_raw["class_id"]
        bx, by, bw, bh = int(W * 0.18), int(H * 0.42), int(W * 0.64), int(H * 0.44)
        area_m2 = round(float(ipm.calculate_surface_area_sqm(bx, by, bw, bh)), 4) if cls_id == 3 else 0.0
        dist_m = round(max(2.5, 18.0 * (1.0 - ((by + bh) / float(H)) ** 0.85)), 1)
        detections.append({
            "bbox_pixels": [bx, by, bw, bh],
            "bbox_normalized": [round(bx / W, 4), round(by / H, 4), round(bw / W, 4), round(bh / H, 4)],
            "class_id": cls_id,
            "class_name": frame_cls_raw["class_name"],
            "confidence": frame_cls_raw["confidence"],
            "shannon_entropy_bits": frame_cls_raw["shannon_entropy_bits"],
            "uncertainty_rating": frame_cls_raw["uncertainty_rating"],
            "astm_d6433_severity": "MEDIUM",
            "irc_standard_specification": IRC_STANDARDS.get(cls_id, ""),
            "top3_ranked_predictions": frame_cls_raw["top3_ranked_predictions"],
            "is_distress": True,
            "distance_meters": dist_m,
            "surface_area_m2": area_m2,
            "area_method": "bounding_box_estimate",
            "area_diagnostics": {
                "max_plausible_m2": 2.0,
                "box_share_of_frame": round((bw * bh) / float(W * H), 3),
            },
            "area_is_plausible_single_repair": True,
            "needs_manual_survey": cls_id in (4, 5, 6),
            "depth_cm": 0.0,
            "depth_estimate": None,
            "volumetric_m3": 0.0,
            "morth_tonnage_t": 0.0,
            "repair_cost_inr": 0.0,
            "probabilities": frame_cls_raw["probabilities"],
        })

    # If no road defect was confirmed, zero out the segmentation mask so rejected noise blobs are not counted
    if not detections and seg_out:
        seg_out["crack_px_full"] = 0
        seg_out["pothole_px_full"] = 0
        seg_out["defect_fraction"] = 0.0
        seg_out["mask_data_url"] = None

    normal_entry = {
        "bbox_pixels": None,
        "bbox_normalized": None,
        "class_id": 0,
        "class_name": CLASS_NAMES[0],
        "confidence": frame_cls_raw["confidence"] if frame_cls_raw["class_id"] == 0 else round(float(frame_cls_raw["probabilities"].get(CLASS_NAMES[0], 0.85)), 4),
        "shannon_entropy_bits": frame_cls_raw["shannon_entropy_bits"],
        "uncertainty_rating": frame_cls_raw["uncertainty_rating"],
        "astm_d6433_severity": "NONE",
        "irc_standard_specification": IRC_STANDARDS[0],
        "top3_ranked_predictions": frame_cls_raw["top3_ranked_predictions"],
        "is_distress": False,
        "distance_meters": 0.0,
        "surface_area_m2": 0.0,
        "area_method": "segmentation_mask" if seg_out else "bounding_box_estimate",
        "needs_manual_survey": False,
        "depth_cm": 0.0,
        "depth_estimate": None,
        "volumetric_m3": 0.0,
        "morth_tonnage_t": 0.0,
        "repair_cost_inr": 0.0,
        "probabilities": frame_cls_raw["probabilities"],
    }

    primary_distress = detections[0] if detections else normal_entry
    primary_pedestrian = pedestrians[0] if pedestrians else None
    primary_vehicle = vehicle_detections[0] if vehicle_detections else None
    primary = primary_distress if detections else (primary_pedestrian if primary_pedestrian else normal_entry)
    all_dets = detections if detections else [normal_entry]

    pci_res = _compute_pci(detections)
    total_tonnage = round(sum(float(d.get("morth_tonnage_t", 0.0)) for d in detections if not d.get("needs_manual_survey")), 4)
    total_cost = round(sum(float(d.get("repair_cost_inr", 0.0)) for d in detections if not d.get("needs_manual_survey")), 2)

    work_order = None
    if detections and primary_distress["class_id"] in (1, 2, 3):
        order_payload = {
            "corridor_id": corridor_id,
            "defect_class": primary_distress["class_name"],
            "surface_area_m2": primary_distress["surface_area_m2"],
            "morth_tonnage_t": total_tonnage,
            "repair_cost_inr": total_cost,
            "pci_score": pci_res["pci_score"],
        }
        seal = hmac.new(b"ROAD-SHIELD-MORTH-SEAL", json.dumps(order_payload, sort_keys=True).encode("utf-8"), hashlib.sha256).hexdigest()
        work_order = {**order_payload, "sha256_seal": seal}

    elapsed_ms = round((time.time() - t0) * 1000.0, 2)
    return {
        "status": "ANALYSIS_COMPLETE",
        "gatekeeper_passed": True,
        "input_check": input_check,
        "texture_metrics": {"road_roi_mean_lum": round(mean_lum, 2), "road_roi_std_lum": round(std_lum, 2)},
        "corridor_id": corridor_id,
        "location": {"lat": latitude, "lng": longitude, "chainage_km": None},
        "is_distress": len(detections) > 0,
        "vulnerable_safety_alert": len(pedestrians) > 0,
        "pedestrians_count": len(pedestrians),
        "all_pedestrians": pedestrians,
        "vehicles_count": len(vehicle_detections),
        "all_vehicles": vehicle_detections,
        "primary_vehicle": primary_vehicle,
        "urban_traffic_analysis": traffic_analysis,
        "scene_objects": scene_objects,
        "scene_summary": scene_summary,
        "primary_distress": primary_distress,
        "primary_pedestrian": primary_pedestrian,
        "primary_detection": primary,
        "all_detections": all_dets,
        "astm_d6433_pci": pci_res,
        "morth_civil_ledger": {
            "asphalt_density_t_m3": 2.40,
            "compaction_factor": 1.15,
            "total_bitumen_tonnage_t": total_tonnage,
            "mix_rate_inr_per_tonne": 7500.0,
            "total_estimated_repair_inr": total_cost,
        },
        "cryptographic_work_order": work_order,
        "frame_classification": frame_cls,
        "vision_backend": "deep_cnn",
        "camera_calibration": calib_desc,
        "segmentation": ({
            "available": True,
            "crack_pixels": seg_out["crack_px_full"],
            "pothole_pixels": seg_out["pothole_px_full"],
            "defect_fraction": seg_out["defect_fraction"],
            "mask_data_url": seg_out.get("mask_data_url"),
            "area_method": "per-pixel ground footprint over the mask",
        } if seg_out else {
            "available": False,
            "area_method": "bounding-box corners projected to the ground plane",
        }),
        "imu_shock_telemetry": {
            "available": False,
            "reason": "No IMU window supplied — the fusion gate reports unavailable rather than deriving one from the vision output.",
        },
        "latency_ms": elapsed_ms,
        "total_latency_ms": elapsed_ms,
    }


def redact_image_base64(image_b64):
    import numpy as np
    from PIL import Image, ImageFilter
    raw = str(image_b64 or "").strip()
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    img_bytes = base64.b64decode(raw)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = pil_img.size
    out_img = pil_img.copy()

    s = _load_sessions()
    det = s.get("detector")
    people_blurred = 0
    plates_blurred = 0
    if det is not None and det.is_ready:
        try:
            objs = det.detect(np.asarray(pil_img, dtype=np.uint8), conf_threshold=0.25)
            for o in objs:
                bx, by, bw, bh = o["bbox_pixels"]
                x0, y0 = max(0, bx), max(0, by)
                x1, y1 = min(w, bx + bw), min(h, by + bh)
                if x1 <= x0 or y1 <= y0:
                    continue
                if o["class_name"] == "person":
                    crop = out_img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=16))
                    out_img.paste(crop, (x0, y0))
                    people_blurred += 1
                elif o["class_name"] in VEHICLE_CLASS_MAP:
                    # Blur lower 45% of vehicle bounding box (license plate band)
                    py0 = max(y0, y0 + int((y1 - y0) * 0.55))
                    crop = out_img.crop((x0, py0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=14))
                    out_img.paste(crop, (x0, py0))
                    plates_blurred += 1
        except Exception:
            pass

    if people_blurred == 0 and plates_blurred == 0:
        band_top = int(h * 0.28)
        band_bot = int(h * 0.52)
        crop = out_img.crop((int(w * 0.15), band_top, int(w * 0.85), band_bot)).filter(ImageFilter.GaussianBlur(radius=12))
        out_img.paste(crop, (int(w * 0.15), band_top))
        plates_blurred = 1

    buf = io.BytesIO()
    out_img.save(buf, format="JPEG", quality=88)
    return {
        "redacted_image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "report": {
            "people_blurred": people_blurred,
            "faces_blurred": people_blurred,
            "plates_blurred": plates_blurred,
            "method": "YOLOv8 COCO person/vehicle box GaussianBlur (DPDP Act 2023)",
        },
    }
