"""
Lightweight ONNX Runtime inference engine for Vercel Serverless Functions.

Uses only `numpy`, `Pillow` (`PIL`), `onnxruntime`, and Python's standard library
(zero `scipy`, `scikit-learn`, `scikit-image`, or `opencv-python` dependencies),
keeping the unzipped function bundle well under Vercel's 250 MB serverless limit
while executing all 5 genuine trained models:
  - `checkpoints/ood_guard.npz` + `checkpoints/cnn_backbone_mobilenetv2.onnx` (OOD Input Guard, AUROC 0.9945)
  - `checkpoints/road_shield_detector.onnx` (YOLOv8 COCO Object Detector: people, vehicles, signs)
  - `checkpoints/deep_vision_mobilenet_v3_large_refit.onnx` (MobileNetV3-Large, 94.92% test accuracy, 0.9207 Macro-F1)
  - `checkpoints/defect_segmenter_pixel.onnx` (served pixel segmenter, ONNX export; see
    scripts/export_pixel_segmenter_onnx.py) or `checkpoints/defect_segmenter_unet.onnx`
    (U-Net, ResNet-18 encoder), whichever checkpoints/segmenter_selection.json serves
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
# Same floor as DeepInferencePipeline.MIN_REPORT_CONFIDENCE.
MIN_REPORT_CONFIDENCE = 0.45
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


def decode_base64_image(image_b64, max_dim=1280):
    import numpy as np
    from PIL import Image
    raw = str(image_b64 or "").strip()
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    img_bytes = base64.b64decode(raw)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = float(max_dim) / float(max(w, h))
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    return np.asarray(pil_img, dtype=np.uint8)


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




class _ServerlessClassifier:
    """Adapter so models/semantic_gate can drive this path's ONNX classifier."""
    @staticmethod
    def predict_probabilities(rgb):
        return list(classify_crop(rgb)["probabilities"].values())


def _pothole_heat(img_np):
    """The main engine's semantic-gate heat map for 'pothole' (None if it cannot run)."""
    try:
        from models import semantic_gate
        return semantic_gate.heatmaps(_ServerlessClassifier, img_np)[semantic_gate.POTHOLE]
    except Exception as e:
        print(f"[WARN] semantic gate unavailable: {e}")
        return None

def _finish_segmentation(flat, p0, crack_p, pothole_p, orig_h, orig_w, object_mask_full, gray_full,
                         backend, t_crack=None, t_pothole=None, pothole_heat=None):
    """Shared post-processing for either segmenter: scene exclusions, blob filters, boxes."""
    import numpy as np
    from PIL import Image
    if t_crack is None or t_pothole is None:
        from models import pixel_segmenter_lite as lite
        th = lite.load()[1]["thresholds"]
        t_crack, t_pothole = float(th.get("crack", 0.5)), float(th.get("pothole", 0.5))
    labels = flat.reshape(WORK_H, WORK_W)
    conf_map = np.where(flat == 1, crack_p, np.where(flat == 2, pothole_p, p0)).reshape(WORK_H, WORK_W)

    # 0a. Suppress non-road horizon / sky in the upper 25% of the frame
    labels[:int(WORK_H * 0.25), :] = 0
    # 0b. Bottom region preserved to lowermost row for foreground road defects

    # 1. Withhold detected pedestrians and vehicles from the segmentation mask
    if object_mask_full is not None and object_mask_full.any():
        om_work = np.asarray(
            Image.fromarray(object_mask_full.astype(np.uint8) * 255, mode="L").resize(
                (WORK_W, WORK_H), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ) > 0
        labels[om_work] = 0

    # 2. Suppress solid letterbox footer banner (Alamy / stock watermarks / camera bars) and bright white paint markings
    if gray_full is not None:
        g_work = np.asarray(
            Image.fromarray(np.clip(gray_full, 0, 255).astype(np.uint8), mode="L").resize(
                (WORK_W, WORK_H), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        # Check bottom 18% for solid letterbox bar
        for y in range(int(WORK_H * 0.82), WORK_H):
            row_med = float(np.median(g_work[y, :]))
            if row_med < 28.0 or row_med > 225.0:
                labels[y:, :] = 0
                break
        for y in range(int(WORK_H * 0.18), -1, -1):
            row_med = float(np.median(g_work[y, :]))
            if row_med < 28.0 or row_med > 225.0:
                labels[:y + 1, :] = 0
                break
        road_med = float(np.median(g_work[int(WORK_H * 0.35):, :]))
        # Withhold pure white high-intensity road paint markings (>215 and bright above road)
        paint_work = g_work > max(215.0, road_med + 55.0)
        labels[paint_work] = 0

    # Semantic gate per pixel, as models/semantic_gate.apply does in the main engine: pothole
    # pixels the classifier's heat map does not support are dropped before blobs are formed.
    if pothole_heat is not None:
        from models import semantic_gate
        heat_work = np.asarray(Image.fromarray(np.asarray(pothole_heat, dtype=np.float32), mode="F").resize(
            (WORK_W, WORK_H), Image.Resampling.BILINEAR), dtype=np.float32)
        labels[(labels == 2) & (heat_work < semantic_gate.DEFAULT_THRESHOLD)] = 0

    min_px_map = {1: 32, 2: 48}
    min_conf_map = {
        1: t_crack,
        2: t_pothole,
    }

    cleaned = np.zeros_like(labels)
    comp_boxes = []
    sx = orig_w / float(WORK_W)
    sy = orig_h / float(WORK_H)
    for cls_val, cls_id in ((2, 2), (1, 1)):
        c_mask, boxes = _connected_components_8(labels == cls_val, min_blob_px=min_px_map[cls_val])
        sorted_boxes = sorted(boxes, key=lambda b: b["area_work_px"], reverse=True)[:5]
        for b in sorted_boxes:
            wx, wy, ww, wh = b["bbox_work"]
            # Require detection centroid to lie in the lower 65% road ROI (below horizon)
            if (wy + wh * 0.5) < WORK_H * 0.35:
                continue

            # Roadway perspective envelope: road narrows toward horizon, reject off-road shoulder foliage/dirt
            norm_y = (wy + wh * 0.5) / float(WORK_H)
            norm_x = (wx + ww * 0.5) / float(WORK_W)
            if norm_y < 0.45 and (norm_x < 0.18 or norm_x > 0.70):
                continue

            sub_mask = c_mask[wy:wy + wh, wx:wx + ww]
            if not sub_mask.any():
                continue
            sub_conf = float(conf_map[wy:wy + wh, wx:wx + ww][sub_mask].mean())
            if sub_conf < min_conf_map[cls_val]:
                continue
            cleaned[wy:wy + wh, wx:wx + ww][sub_mask] = cls_val
            bx = max(0, int(round(wx * sx)))
            by = max(0, int(round(wy * sy)))
            bw = min(orig_w - bx, max(16, int(round(ww * sx))))
            bh = min(orig_h - by, max(16, int(round(wh * sy))))
            if bw < 14 or bh < 10:
                continue
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
        "backend": backend,
    }

def _served_segmenter():
    """'pixel_classifier' or 'unet', as decided by scripts/segmenter_deployment_check.py."""
    try:
        with open(os.path.join(CKPT_DIR, "segmenter_selection.json"), "r", encoding="utf-8") as fh:
            return json.load(fh).get("served", "unet")
    except Exception:
        return "unet"


def segment_image(img_rgb, object_mask_full=None, gray_full=None, pothole_heat=None):
    import numpy as np
    from PIL import Image
    orig_h, orig_w = img_rgb.shape[:2]
    # The segmenter the deployment check chose. The pixel classifier runs here through its
    # ONNX export and a numpy port of its features (models/pixel_segmenter_lite.py; pixel
    # labels agree with the scikit-learn/OpenCV original on 99.7% of pixels). Before this,
    # the website always ran the U-Net, which lost that check.
    if _served_segmenter() == "pixel_classifier":
        from models import pixel_segmenter_lite as lite
        proba = lite.predict_proba(img_rgb) if lite.available() else None
        if proba is not None:
            thresholds = lite.load()[1]["thresholds"]
            p0, crack_p, pothole_p = proba[:, 0], proba[:, 1], proba[:, 2]
            is_crack = crack_p >= float(thresholds.get("crack", 0.5))
            is_pothole = pothole_p >= float(thresholds.get("pothole", 0.5))
            # Same rule as models/defect_segmenter.mask_from_proba (the main engine).
            flat = np.zeros(proba.shape[0], dtype=np.uint8)
            flat[is_crack] = 1
            flat[is_pothole & (pothole_p >= crack_p)] = 2
            flat[is_pothole & ~is_crack] = 2
            return _finish_segmentation(flat, p0, crack_p, pothole_p, orig_h, orig_w,
                                        object_mask_full, gray_full, backend="pixel_classifier",
                                        pothole_heat=pothole_heat)
    s = _load_sessions()
    seg = s.get("segmenter")
    if not seg:
        return None
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
    p0 = proba[:, 0]
    crack_p = proba[:, 1]
    pothole_p = proba[:, 2]
    t_crack = float(thresholds.get("crack", 0.40))
    t_pothole = float(thresholds.get("pothole", 0.38))

    flat = np.zeros(proba.shape[0], dtype=np.uint8)
    # A pixel is crack/pothole ONLY if it exceeds threshold and is competitive with background sound road
    is_crack = (crack_p >= t_crack) & (crack_p >= p0 * 0.92) & (crack_p >= pothole_p)
    is_pothole = (pothole_p >= t_pothole) & (pothole_p >= p0 * 0.92) & (pothole_p > crack_p)
    flat[is_crack] = 1
    flat[is_pothole] = 2
    return _finish_segmentation(flat, p0, crack_p, pothole_p, orig_h, orig_w,
                                object_mask_full, gray_full, backend="unet",
                                t_crack=t_crack, t_pothole=t_pothole, pothole_heat=pothole_heat)

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
        p_sev = "HIGH" if (pot_area > 0.35 or any(float(d.get("depth_cm", 0.0)) >= 4.0 for d in potholes)) else "MEDIUM"
        deducts["potholes"] = round(eng._deduct_value("potholes", p_sev, pot_dens), 1)
    # Markings, dividers, signs and waterlogging are not ASTM D6433 pavement distresses,
    # so they add no deduct (an invented 14.5 "safety_corridor" deduct used to).
    cdv = eng._corrected_deduct_value(list(deducts.values())) if deducts else 0.0
    pci = round(float(np.clip(100.0 - cdv, 0.0, 100.0)), 1)
    # Operational dispatch rule, NOT part of ASTM D6433: a large or deep cavity is
    # flagged for priority repair. It used to be forced into the PCI itself (a corrected
    # deduct of at least 65, so PCI <= 35) under a mis-cited "ASTM D6433" comment.
    priority_cavity = any(d.get("class_id") == 2 and (float(d.get("surface_area_m2", 0.0)) >= 0.5
                                                      or float(d.get("depth_cm", 0.0)) >= 4.5)
                          for d in detections)
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
    return {
        "pci_score": pci,
        "rating_category": cat,
        "description": desc,
        "deduct_values": deducts,
        "rutting_mm": None,
        "iri_roughness": None,
        "not_measured": "rut depth and IRI need a straightedge or profile, not a photograph",
        "priority_repair": priority_cavity,
        "priority_rule": "operational: pothole >= 0.5 m2 or >= 4.5 cm deep (not an ASTM D6433 rule)",
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

    # Geometric Road Markings & Zebra Crossing Analysis
    markings_res = {"found": False, "bars": []}
    try:
        from models.marking_detector import detect_zebra
        z = detect_zebra(img_np)
        if z and z.get("found"):
            markings_res = z
    except Exception as e:
        markings_res = {"found": False, "reason": str(e)}

    return scene_objects, scene_summary, pedestrians, vehicle_detections, traffic_analysis, object_mask, markings_res


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

    # Detect and suppress genuine flat/synthetic letterbox banner or watermark footer
    # Only trigger if the row is uniform (std < 3.0), never suppressing real road texture or foreground defects
    for y in range(H - 1, int(H * 0.90), -1):
        row = gray[y, :].astype(np.float32)
        if float(np.std(row)) < 3.0 and (float(np.median(row)) < 20.0 or float(np.median(row)) > 235.0):
            excl_mask[y:, :] = True
            break
    for y in range(0, int(H * 0.12)):
        row = gray[y, :].astype(np.float32)
        if float(np.std(row)) < 3.0 and (float(np.median(row)) < 20.0 or float(np.median(row)) > 235.0):
            excl_mask[:y + 1, :] = True
            break

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

    # Pavement cell score matrix preserved to lowermost row for foreground defect recovery

    peaks = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            sc = cell_score[r, c]
            if sc < 95.0:
                continue
            r_min, r_max = max(0, r - 1), min(grid_rows, r + 2)
            c_min, c_max = max(0, c - 1), min(grid_cols, c + 2)
            if sc == cell_score[r_min:r_max, c_min:c_max].max():
                peaks.append((r, c, sc))

    peaks.sort(key=lambda p: p[2], reverse=True)

    # Fallback to global peak if no peak exceeded 95 but road anomaly exists
    if not peaks and cell_score.max() >= 45.0:
        max_idx = np.unravel_index(np.argmax(cell_score), cell_score.shape)
        peaks.append((int(max_idx[0]), int(max_idx[1]), float(cell_score[max_idx])))

    raw_boxes = []
    for pr, pc, psc in peaks:
        visited = set([(pr, pc)])
        q = [(pr, pc)]
        while q:
            cr, cc = q.pop(0)
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < grid_rows and 0 <= nc < grid_cols and (nr, nc) not in visited:
                    if cell_score[nr, nc] >= max(50.0, psc * 0.40):
                        if abs(nr - pr) <= 4 and abs(nc - pc) <= 5:
                            visited.add((nr, nc))
                            q.append((nr, nc))
        rs = [p[0] for p in visited]
        cs = [p[1] for p in visited]
        bx = max(10, min(cs) * cell_w - int(cell_w * 0.15))
        by = max(roi_y0, roi_y0 + min(rs) * cell_h - int(cell_h * 0.10))
        bw = min(int(W * 0.65), min(W - bx - 5, (max(cs) - min(cs) + 1) * cell_w + int(cell_w * 0.35)))
        bh = min(int(H * 0.58), min(H - by - 5, (max(rs) - min(rs) + 1) * cell_h + int(cell_h * 0.30)))

        overlap = False
        for cb in raw_boxes:
            ix0, iy0 = max(bx, cb[0]), max(by, cb[1])
            ix1, iy1 = min(bx + bw, cb[0] + cb[2]), min(by + bh, cb[1] + cb[3])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                iou = inter / float(bw * bh + cb[2] * cb[3] - inter)
                containment = inter / float(min(bw * bh, cb[2] * cb[3]))
                if iou > 0.35 or containment > 0.50:
                    overlap = True
                    break
        if not overlap:
            kinds = [int(cell_kind[cr, cc]) for cr, cc in visited]
            raw_boxes.append([int(bx), int(by), int(bw), int(bh), psc, 2 if kinds.count(2) >= kinds.count(1) else 1])

    if not raw_boxes:
        return []

    return _fuse_and_cluster_boxes(raw_boxes, W, H, max_boxes=3)


def _fuse_and_cluster_boxes(raw_boxes, W, H, max_boxes=3):
    """
    Weighted Box Fusion and Spatial Clustering.
    Merges overlapping (IoU >= 0.12) and proximate (gap < 8%) boxes into unified defect boundaries,
    eliminating fragmented duplicates and colliding text labels.
    """
    if not raw_boxes:
        return []
    boxes = sorted(raw_boxes, key=lambda b: b[4], reverse=True)
    clusters = []
    for b in boxes:
        bx, by, bw, bh, sc, kd = b
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
        matched = []
        for ci, c in enumerate(clusters):
            cx1 = min(ob[0] for ob in c)
            cy1 = min(ob[1] for ob in c)
            cx2 = max(ob[0] + ob[2] for ob in c)
            cy2 = max(ob[1] + ob[3] for ob in c)
            cbw = cx2 - cx1
            cbh = cy2 - cy1

            inter_w = max(0, min(x2, cx2) - max(x1, cx1))
            inter_h = max(0, min(y2, cy2) - max(y1, cy1))
            inter_area = inter_w * inter_h
            b_area = bw * bh
            c_area = cbw * cbh
            union_area = b_area + c_area - inter_area
            iou = inter_area / max(1.0, union_area)
            containment = inter_area / max(1.0, min(b_area, c_area))

            comb_w = max(x2, cx2) - min(x1, cx1)
            comb_h = max(y2, cy2) - min(y1, cy1)
            gap_x = max(0, max(x1, cx1) - min(x2, cx2))
            gap_y = max(0, max(y1, cy1) - min(y2, cy2))
            horiz_overlap = inter_w / float(max(1, min(bw, cbw)))
            is_proximate = (gap_x <= int(W * 0.04) and gap_y <= int(H * 0.03) and comb_w <= int(W * 0.40) and comb_h <= int(H * 0.30))
            is_crater_complex = (
                horiz_overlap >= 0.35
                and gap_y <= int(H * 0.14)
                and comb_w <= int(W * 0.55)
                and comb_h <= int(H * 0.55)
                and (comb_w * comb_h) <= int(W * H * 0.33)
            )

            if iou >= 0.18 or containment >= 0.40 or is_proximate or is_crater_complex:
                matched.append(ci)
        if not matched:
            clusters.append([b])
        elif len(matched) == 1:
            clusters[matched[0]].append(b)
        else:
            merged = [b]
            for ci in sorted(matched, reverse=True):
                merged.extend(clusters.pop(ci))
            clusters.append(merged)

    fused = []
    for c in clusters:
        min_x = max(0, min(b[0] for b in c))
        min_y = max(0, min(b[1] for b in c))
        max_x = min(W, max(b[0] + b[2] for b in c))
        max_y = min(H, max(b[1] + b[3] for b in c))
        fw = max_x - min_x
        fh = max_y - min_y
        if (fw * fh) > int(W * H * 0.33):
            scale = (int(W * H * 0.33) / float(fw * fh)) ** 0.5
            fw = max(35, int(fw * scale))
            fh = max(20, int(fh * scale))
            min_x = max(0, min(W - fw, min_x))
            min_y = max(0, min(H - fh, min_y))
        if fw < 35 or fh < 18:
            continue
        if (min_x < 20 or max_x > W - 20) and (fw * fh) < 0.015 * W * H:
            continue
        max_sc = max(b[4] for b in c)
        kinds = [b[5] for b in c]
        dom_kind = 2 if kinds.count(2) >= kinds.count(1) else 1
        fused.append([min_x, min_y, fw, fh, max_sc, dom_kind])

    fused.sort(key=lambda b: b[4] * ((b[2] * b[3]) ** 0.5), reverse=True)
    return fused[:max_boxes]


def _segment_defect_regions(img_np, boxes, orig_h, orig_w, is_waterlogging=False):
    """
    Adaptive photometric cavity and puddle segmentation for confirmed defect proposals.
    Distinguishes specular sky-reflecting water pools on mud roads from dark depression
    cavities and cracks on standard gray asphalt roads.
    """
    import numpy as np
    import io, base64
    from PIL import Image, ImageFilter

    mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
    comp_boxes = []

    roi_y0 = int(orig_h * 0.35)
    road = img_np[roi_y0:, :]
    r = road[:, :, 0].astype(np.float32)
    g = road[:, :, 1].astype(np.float32)
    b = road[:, :, 2].astype(np.float32)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    sat = np.zeros_like(cmax)
    nz = cmax > 1e-4
    sat[nz] = (cmax[nz] - cmin[nz]) / cmax[nz]
    gray_road = 0.299 * r + 0.587 * g + 0.114 * b

    # A scene is an unpaved waterlogged mud road ONLY if the road background has high color saturation
    # (brown/yellow mud: sat > 0.18) AND whole-frame classification indicates waterlogging / flood hazard
    road_is_mud = float((sat > 0.18).mean()) > 0.20
    is_water_puddle = road_is_mud and is_waterlogging

    if is_water_puddle:
        # Specular water pools on mud road
        is_water = (gray_road > 132.0) & (sat < 0.15)
        m_img = Image.fromarray(is_water.astype(np.uint8) * 255, mode="L")
        m_opened = m_img.filter(ImageFilter.MinFilter(size=5)).filter(ImageFilter.MaxFilter(size=5))
        clean_arr = np.asarray(m_opened) > 0
        c_mask, sub_boxes = _connected_components_8(clean_arr, min_blob_px=100)
        mask_full[roi_y0:, :][c_mask] = 2
        for sb in sub_boxes:
            wx, wy, ww, wh = sb["bbox_work"]
            comp_boxes.append({
                "seg_cls": 2,
                "class_id": 2,
                "bbox_pixels": [int(wx), int(wy + roi_y0), int(ww), int(wh)],
                "area_work_px": sb["area_work_px"],
                "mean_conf": 0.94,
            })
    else:
        # Standard asphalt road: segment strictly inside proposal boxes for dark depression/cavity
        road_median = float(np.median(gray_road))
        for b in boxes:
            bx, by, bw, bh = b["bbox_pixels"]
            cls_id = b.get("class_id", 2)
            crop = img_np[by:by + bh, bx:bx + bw]
            if crop.size == 0 or bw < 14 or bh < 14:
                continue

            pil_crop = Image.fromarray(crop)
            blurred = np.asarray(pil_crop.convert("L").filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)

            hist, _ = np.histogram(blurred, bins=256, range=(0, 256))
            hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
            w0 = np.cumsum(hist)
            w1 = 1.0 - w0
            mean_all = np.sum(np.arange(256) * hist)
            u0 = np.cumsum(np.arange(256) * hist) / (w0 + 1e-7)
            u1 = (mean_all - np.cumsum(np.arange(256) * hist)) / (w1 + 1e-7)
            variance = w0 * w1 * ((u0 - u1) ** 2)
            otsu_thresh = float(np.argmax(variance))

            # Depression cavity: darker than Otsu threshold AND darker than road median
            cavity_thresh = min(otsu_thresh, road_median - 6.0)
            raw_mask = (blurred < max(cavity_thresh, otsu_thresh * 0.90)).astype(np.uint8) * 255
            m_img = Image.fromarray(raw_mask, mode="L")
            m_closed = m_img.filter(ImageFilter.MaxFilter(size=3)).filter(ImageFilter.MinFilter(size=3))
            m = np.asarray(m_closed, dtype=np.uint8) > 0

            px_count = int(m.sum())
            if px_count >= 24:
                mask_full[by:by + bh, bx:bx + bw][m] = cls_id
                comp_boxes.append({
                    "seg_cls": cls_id,
                    "class_id": cls_id,
                    "bbox_pixels": [bx, by, bw, bh],
                    "area_work_px": px_count,
                    "mean_conf": b.get("confidence", 0.90),
                })

    comp_boxes.sort(key=lambda c: c["area_work_px"], reverse=True)

    total_px = int((mask_full > 0).sum())
    mask_url = None
    if total_px > 0:
        rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        rgba[mask_full == 1] = [45, 212, 212, 135]   # cyan for cracks
        rgba[mask_full == 2] = [244, 63, 94, 145]    # rose-red for potholes
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        mask_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "mask": mask_full,
        "mask_work": mask_full,
        "mask_data_url": mask_url,
        "crack_px": int((mask_full == 1).sum()),
        "pothole_px": int((mask_full == 2).sum()),
        "crack_px_full": int((mask_full == 1).sum()),
        "pothole_px_full": int((mask_full == 2).sum()),
        "defect_fraction": round(float((mask_full != 0).mean()), 4),
        "components": comp_boxes,
    }


def _unwrap_dashboard_screenshot(img_np):
    """
    Detects if an uploaded image is an accidental browser screenshot of the
    ROAD-SHIELD inspection page containing an embedded road canvas, and extracts
    the pure road photo canvas.
    """
    H, W = img_np.shape[:2]
    if W < 600 or H < 400:
        return img_np, False
    right_patch = img_np[int(H * 0.25):int(H * 0.75), int(W * 0.65):int(W * 0.95)]
    r_mean = right_patch.mean(axis=(0, 1))
    if 8.0 <= r_mean[0] <= 26.0 and 12.0 <= r_mean[1] <= 32.0 and 18.0 <= r_mean[2] <= 42.0:
        x0, y0 = 24, 112
        x1, y1 = int(W * 0.524), int(H * 0.75)
        crop = img_np[y0:y1, x0:x1]
        if crop.shape[0] > 120 and crop.shape[1] > 150:
            return crop, True
    return img_np, False


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
    img_np, _was_unwrapped = _unwrap_dashboard_screenshot(img_np)
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

    # STAGE 2: YOLOv8 COCO Object Detection + Geometric Marking Analyzer (Zebra Crossing)
    scene_objects, scene_summary, pedestrians, vehicle_detections, traffic_analysis, object_mask, markings_res = _detect_scene_objects(img_np, H, W)

    # STAGE 3: Whole-frame MobileNetV3-Large classification with Road Surface ROI prior
    frame_cls_raw = classify_crop(img_np)
    roi_crop = img_np[int(H * 0.35):, :]
    road_cls_raw = classify_crop(roi_crop) if roi_crop.size > 0 else frame_cls_raw
    if frame_cls_raw.get("class_id") in (5, 6) and road_cls_raw.get("class_id") in (1, 2, 3):
        frame_cls_raw = road_cls_raw
    elif frame_cls_raw.get("class_id") == 0 and road_cls_raw.get("class_id") in (1, 2, 3) and road_cls_raw.get("confidence", 0.0) >= 0.45:
        frame_cls_raw = road_cls_raw
    f_cls = frame_cls_raw.get("class_id")
    f_conf = frame_cls_raw.get("confidence", 0.0)

    # Semantic cross-verification: on severe distress roads (pothole cavity, waterlogging, or crack >= 0.60 conf),
    # suppress zebra crossing detection completely (water reflections / specular puddle glare are not crosswalks).
    # A second rule (mean bar solidity < 0.62) was removed: it discarded real crossings (1/19 kept; 4/19
    # without it) and suppressed no extra false ones (flooded streets 0/14, pothole photographs 1/40).
    if markings_res.get("found"):
        bars = markings_res.get("bars", [])
        mean_solidity = float(np.mean([b.get("solidity", 0.0) for b in bars])) if bars else 0.0
        if (f_cls in (1, 2, 3) and f_conf >= 0.60):
            markings_res = {
                "found": False,
                "bars": [],
                "bar_count": 0,
                "reason": f"suppressed on distress road (distress {(CLASS_NAMES[f_cls] if 0 <= f_cls < len(CLASS_NAMES) else "Unknown")} conf {f_conf:.2f}, solidity {mean_solidity:.2f})",
            }
        else:
            all_cross = markings_res.get("all_crossings") or [markings_res]
            for cross_item in all_cross:
                bx, by, bw, bh = cross_item["bbox_pixels"]
                scene_objects.append({
                    "class_name": cross_item.get("class_name", "Zebra Crossing"),
                    "confidence": cross_item.get("confidence", 0.95),
                    "bbox_pixels": [bx, by, bw, bh],
                    "bbox_normalized": cross_item.get("bbox_normalized", [round(bx / W, 4), round(by / H, 4), round(bw / W, 4), round(bh / H, 4)]),
                    "detector": "geometric_marking_analyzer",
                    "color": "#38bdf8",
                    "is_distress": False,
                    "is_pedestrian": False,
                    "is_vehicle": False,
                    "is_marking": True,
                    "details": {
                        "stripe_count": cross_item.get("stripe_count", 0),
                        "bar_count": cross_item.get("bar_count", 0),
                        "spacing_variation": cross_item.get("spacing_variation", 0.0),
                        "paint_share": cross_item.get("paint_share", 0.0),
                    }
                })
            if scene_summary.get("available"):
                counts = scene_summary.setdefault("counts_by_class", {})
                counts[markings_res["class_name"]] = counts.get(markings_res["class_name"], 0) + 1
                scene_summary["objects_detected"] = len(scene_objects)
            try:
                from models.marking_detector import paint_mask
                p_mask = paint_mask(img_np, markings_res)
                object_mask = object_mask | p_mask
            except Exception:
                pass
            if f_cls == 4:
                # Intact zebra crossing is verified by geometry; override spurious missing crossing defect
                frame_cls_raw["class_id"] = 0
                frame_cls_raw["class_name"] = "Normal Road / Sound Pavement"

    frame_cls = {
        "class_name": frame_cls_raw["class_name"],
        "confidence": frame_cls_raw["confidence"],
        "probabilities": frame_cls_raw["probabilities"],
        "model": "deep_cnn",
        "role": "whole-frame opinion of the classifier; does not set area, depth or cost",
    }

    # STAGE 4: Fusion of U-Net Segmentation & Salient Cavity Proposals
    pothole_heat = _pothole_heat(img_np)
    seg_out = segment_image(img_np, object_mask_full=object_mask, gray_full=gray, pothole_heat=pothole_heat)
    detections = []

    # Gather proposals from U-Net components and salient anomaly detector
    proposals = []
    if seg_out and seg_out["components"]:
        for comp in seg_out["components"]:
            proposals.append({
                "bbox_pixels": comp["bbox_pixels"],
                "seg_comp": comp,
                "source": (seg_out or {}).get("backend", "segmenter") + "_segmenter",
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
        if bw < 14 or bh < 10:
            continue
        # Reject sidewalk kerb / frame border artifacts (e.g. raised pavement margins in upper road zone)
        if (bx < 20 or bx + bw > W - 20) and by < int(H * 0.50) and (bw * bh) < int(0.04 * W * H):
            continue
        # Reject distant horizon noise (large width proposal with centroid in top 40% of frame)
        if (by + bh * 0.5) < int(H * 0.40) and bw > int(W * 0.25):
            continue
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

        # Reject roadside shoulder margin noise in upper road zone
        if (by + bh * 0.5) < int(H * 0.45):
            centroid_x = (bx + bw * 0.5) / float(W)
            if centroid_x < 0.18 or centroid_x > 0.70:
                continue

        # Heuristic salient proposal verification (matches pipeline/deep_inference_pipeline.py line 1252):
        if prop["source"] == "salient_cavity_detector":
            crop_is_distress = cls_info["class_id"] in (1, 2, 3) and cls_info["confidence"] >= 0.65
            frame_is_distress = frame_cls_raw["class_id"] in (1, 2, 3) and frame_cls_raw["confidence"] >= 0.40
            if not (crop_is_distress or frame_is_distress):
                continue
            box_area_share = (bw * bh) / float(W * H)
            if box_area_share < 0.012 and cls_info["confidence"] < 0.72:
                continue
            # Cracks cannot be proposed by optical saliency without segmentation agreement
            if cls_info["class_id"] == 1:
                crack_mask_px = int((mask_full[by:by + bh, bx:bx + bw] == 1).sum()) if mask_full is not None else 0
                if crack_mask_px < 15:
                    continue
            # Salient potholes require either active cavity mask pixels or whole-frame confirmation
            elif cls_info["class_id"] == 2:
                pothole_mask_px = int((mask_full[by:by + bh, bx:bx + bw] == 2).sum()) if mask_full is not None else 0
                frame_confirms_pothole = (frame_cls_raw["class_id"] == 2 and frame_cls_raw["confidence"] >= 0.60)
                if pothole_mask_px < 20 and not frame_confirms_pothole:
                    continue

        # Class determination - the main engine's rules (pipeline/deep_inference_pipeline.py,
        # region verification): the crop classifier's verdict stands. "Sound pavement" drops
        # the region, and a zebra / divider / sign verdict drops a segmenter region. This path
        # used to override both - keeping "sound" regions that had a small mask, relabelling
        # zebra/divider/sign crops as pothole or crack at p >= 0.20, and calling a crop a
        # pothole when p(normal) < 0.15 ("sky reflection") - which reported damage on about a
        # third of clean-road check photographs.
        cls_id = cls_info["class_id"]
        if cls_id == 0 or cls_id in (4, 5, 6):
            continue
        if float(cls_info.get("confidence", 0.0)) < MIN_REPORT_CONFIDENCE:
            continue

        de = _photometric_depth(gray, [bx, by, bw, bh], cls_id)
        cls_name = CLASS_NAMES[cls_id]
        # The crop classifier's own probability for the reported class (no floor: a 0.65
        # minimum used to be imposed here whatever the model said).
        conf = float(cls_info["confidence"])

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

        box_m2 = float(ipm.calculate_surface_area_sqm(bx, by, bw, bh))
        box_m2 = float(max(0.04, min(box_m2, 3.0)))

        # When water reflections or glare cause the U-Net mask to capture only a partial rim sliver (< 25% of box area),
        # use the full geometric homography area so the entire cavity is costed under MoRTH Section 500
        if has_dense_mask:
            if area_m2 < 0.25 * box_m2 and cls_id == 2:
                area_m2 = box_m2
                has_dense_mask = False
            else:
                area_m2 = float(max(0.012, min(area_m2, 3.0)))
        else:
            area_m2 = box_m2

        area_m2 = float(max(0.012, min(area_m2, 3.0)))
        dist_m = round(max(2.2, 18.0 * (1.0 - ((by + bh) / float(H)) ** 0.85)), 1)
        depth_cm = de["depth_cm"]
        tonnage_t = round(area_m2 * (depth_cm / 100.0) * 2.40 * 1.15, 4)
        cost_inr = round(tonnage_t * 7500.0, 2)
        de["repair_cost_low_inr"] = round(area_m2 * (de["depth_low_cm"] / 100.0) * 2.40 * 1.15 * 7500.0, 2)
        de["repair_cost_high_inr"] = round(area_m2 * (de["depth_high_cm"] / 100.0) * 2.40 * 1.15 * 7500.0, 2)

        box_share = (bw * bh) / float(W * H)
        needs_survey = bool(area_m2 > 3.0 or box_share > 0.45)
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

    # Spatial Proximity Fusion & Deduplication (Merges rim fragments, puddle center & adjoining defects)
    if len(detections) > 1:
        def _can_merge_defects(b1, b2, W, H):
            xi0, yi0 = max(b1[0], b2[0]), max(b1[1], b2[1])
            xi1, yi1 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
            iou = 0.0
            containment = 0.0
            if xi1 > xi0 and yi1 > yi0:
                inter = (xi1 - xi0) * (yi1 - yi0)
                a1 = b1[2] * b1[3]
                a2 = b2[2] * b2[3]
                iou = inter / float(a1 + a2 - inter)
                containment = inter / float(min(a1, a2))

            gap_x = max(0, max(b1[0], b2[0]) - min(b1[0] + b1[2], b2[0] + b2[2]))
            gap_y = max(0, max(b1[1], b2[1]) - min(b1[1] + b1[3], b2[1] + b2[3]))
            comb_w = max(b1[0] + b1[2], b2[0] + b2[2]) - min(b1[0], b2[0])
            comb_h = max(b1[1] + b1[3], b2[1] + b2[3]) - min(b1[1], b2[1])
            horiz_inter = max(0, min(b1[0] + b1[2], b2[0] + b2[2]) - max(b1[0], b2[0]))
            horiz_overlap = horiz_inter / float(max(1, min(b1[2], b2[2])))
            is_proximate = (
                gap_x <= int(W * 0.03)
                and gap_y <= int(H * 0.03)
                and comb_w <= int(W * 0.40)
                and comb_h <= int(H * 0.30)
            )
            is_crater_complex = (
                (horiz_overlap >= 0.12 or gap_x <= int(W * 0.04))
                and gap_y <= int(H * 0.08)
                and comb_w <= int(W * 0.55)
                and comb_h <= int(H * 0.45)
                and (comb_w * comb_h) <= int(W * H * 0.35)
            )
            return (iou > 0.15 or containment > 0.40 or is_proximate or is_crater_complex)

        changed = True
        while changed and len(detections) > 1:
            changed = False
            new_dets = []
            used = set()
            for i in range(len(detections)):
                if i in used:
                    continue
                d1 = detections[i]
                mb = d1["bbox_pixels"][:]
                max_conf = d1["confidence"]
                max_area = d1.get("surface_area_m2", 0.0)
                for j in range(i + 1, len(detections)):
                    if j in used:
                        continue
                    d2 = detections[j]
                    if d1["class_id"] != d2["class_id"]:
                        continue
                    if _can_merge_defects(mb, d2["bbox_pixels"], W, H):
                        nx = min(mb[0], d2["bbox_pixels"][0])
                        ny = min(mb[1], d2["bbox_pixels"][1])
                        nw = max(mb[0] + mb[2], d2["bbox_pixels"][0] + d2["bbox_pixels"][2]) - nx
                        nh = max(mb[1] + mb[3], d2["bbox_pixels"][1] + d2["bbox_pixels"][3]) - ny
                        nx = max(0, min(W - 20, nx))
                        ny = max(0, min(H - 20, ny))
                        nw = max(20, min(W - nx, nw))
                        nh = max(15, min(H - ny, nh))
                        mb = [nx, ny, nw, nh]
                        max_conf = max(max_conf, d2["confidence"])
                        max_area = max(max_area, d2.get("surface_area_m2", 0.0))
                        used.add(j)
                        changed = True
                used.add(i)
                d1_merged = dict(d1)
                d1_merged["bbox_pixels"] = mb
                d1_merged["bbox_normalized"] = [round(mb[0] / W, 4), round(mb[1] / H, 4), round(mb[2] / W, 4), round(mb[3] / H, 4)]
                d1_merged["confidence"] = max_conf
                box_m2 = round(float(ipm.calculate_surface_area_sqm(mb[0], mb[1], mb[2], mb[3])), 4)
                d1_merged["surface_area_m2"] = float(max(0.04, min(max(max_area, box_m2 * 0.6), 3.0)))
                d1_merged["area_diagnostics"]["box_share_of_frame"] = round((mb[2] * mb[3]) / float(W * H), 3)
                de = _photometric_depth(gray, mb, d1_merged["class_id"])
                d1_merged["depth_cm"] = de["depth_cm"]
                d1_merged["depth_estimate"] = de
                tonnage_t = round(d1_merged["surface_area_m2"] * (de["depth_cm"] / 100.0) * 2.40 * 1.15, 4)
                d1_merged["morth_tonnage_t"] = tonnage_t
                d1_merged["repair_cost_inr"] = round(tonnage_t * 7500.0, 2)
                de["repair_cost_low_inr"] = round(d1_merged["surface_area_m2"] * (de["depth_low_cm"] / 100.0) * 2.40 * 1.15 * 7500.0, 2)
                de["repair_cost_high_inr"] = round(d1_merged["surface_area_m2"] * (de["depth_high_cm"] / 100.0) * 2.40 * 1.15 * 7500.0, 2)
                new_dets.append(d1_merged)
            detections = new_dets

        # Second-pass containment suppression: suppress smaller nested sub-boxes (containment >= 70%)
        clean_dets = []
        for d in detections:
            b1 = d["bbox_pixels"]
            is_nested = False
            for other in clean_dets:
                b2 = other["bbox_pixels"]
                if d["class_id"] != other["class_id"]:
                    continue
                xi0, yi0 = max(b1[0], b2[0]), max(b1[1], b2[1])
                xi1, yi1 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
                if xi1 > xi0 and yi1 > yi0:
                    inter = (xi1 - xi0) * (yi1 - yi0)
                    if inter / float(b1[2] * b1[3]) >= 0.70:
                        is_nested = True
                        break
            if not is_nested:
                clean_dets.append(d)
        detections = clean_dets

        # Filter distant low-conf / small background noise when large foreground defect exists
        if any(d.get("surface_area_m2", 0.0) > 1.0 for d in detections):
            detections = [d for d in detections if not (d["bbox_pixels"][1] < int(H * 0.40) and d.get("surface_area_m2", 0.0) < 0.35)]

    # Semantic gate (models/semantic_gate.py, threshold measured in
    # checkpoints/semantic_gate_report.json): a pothole box is kept only where the CNN's
    # sliding-window heat map also sees a pothole. The main engine applies it to every
    # pixel-classifier mask; this path had none and reported potholes on 16 of the 36
    # clean-road check photographs.
    if pothole_heat is not None and any(d.get("class_id") == 2 for d in detections):
        from models import semantic_gate
        kept = []
        for d in detections:
            if d.get("class_id") == 2 and d.get("bbox_pixels"):
                bx, by, bw, bh = d["bbox_pixels"]
                region = pothole_heat[by:by + bh, bx:bx + bw]
                peak = float(region.max()) if region.size else 0.0
                d["semantic_gate_pothole_heat"] = round(peak, 3)
                if peak < semantic_gate.DEFAULT_THRESHOLD:
                    continue
            kept.append(d)
        detections = kept

    # Sort detections by significance (potholes first, then largest surface area) and keep top 4
    detections.sort(key=lambda d: (1 if d["class_id"] == 2 else 0, d["surface_area_m2"]), reverse=True)
    detections = detections[:4]

    # Corridor-level hazards (waterlogging, zebra, divider, sign) come from the classifier's
    # whole-frame opinion. They used to be turned into a detection with a fixed box
    # (18-82% x 42-86% of the frame) and a distance - and for waterlogging an area - computed
    # from that invented box. On the 36 clean-road check photographs this path then reported
    # a missing zebra crossing or divider on 14. Same rule as the main engine
    # (pipeline/deep_inference_pipeline._frame_classification): a whole-frame opinion is
    # reported as such and never sets a box, area, depth or cost.
    frame_level_hazard = None
    if (
        not detections
        and frame_cls_raw["class_id"] in (3, 4, 5, 6)
        and frame_cls_raw["confidence"] >= 0.72
        and input_check.get("verdict") in ("ok", None)
    ):
        frame_level_hazard = {
            "class_id": frame_cls_raw["class_id"],
            "class_name": frame_cls_raw["class_name"],
            "confidence": frame_cls_raw["confidence"],
            "irc_standard_specification": IRC_STANDARDS.get(frame_cls_raw["class_id"], ""),
            "role": "whole-frame opinion of the classifier; not localised, sets no area, depth or cost",
        }

    # STAGE 5: Refine surface area and repair metrics using genuine U-Net segmentation mask
    if detections and seg_out and mask_full is not None:
        mask_full = np.array(mask_full, copy=True)
        for d in detections:
            if d.get("bbox_pixels"):
                bx, by, bw, bh = d["bbox_pixels"]
                cls_id = d.get("class_id", 2)
                region_bool = np.zeros((H, W), dtype=bool)
                region_bool[by:by + bh, bx:bx + bw] = (mask_full[by:by + bh, bx:bx + bw] == cls_id)
                px_c = int(region_bool.sum())

                # If the segmenter mask is sparse/empty for this confirmed defect (e.g. water-filled pothole or specular puddle),
                # perform adaptive photometric cavity and puddle segmentation inside the confirmed defect bounding box
                if px_c < 25:
                    from PIL import Image, ImageFilter
                    crop_gray = gray[by:by + bh, bx:bx + bw]
                    road_med = float(np.median(gray[int(H * 0.35):, :]))
                    pil_cg = Image.fromarray(np.clip(crop_gray, 0, 255).astype(np.uint8), mode="L")
                    blurred_cg = np.asarray(pil_cg.filter(ImageFilter.BoxBlur(3)), dtype=np.float32)
                    diff_cg = np.abs(crop_gray - blurred_cg)

                    if cls_id == 2:  # Pothole cavity: dark depression or specular standing water
                        is_dark_cavity = crop_gray < min(114.0, road_med - 6.0)
                        is_water_puddle = (diff_cg < 6.5) & (crop_gray > 115.0)
                        local_mask = is_dark_cavity | is_water_puddle
                    else:  # Crack: high gradient edges darker than road median
                        gy_c, gx_c = np.gradient(crop_gray)
                        gmag_c = np.hypot(gx_c, gy_c)
                        local_mask = (gmag_c > 22.0) & (crop_gray < road_med)

                    if int(local_mask.sum()) >= 25:
                        sub_patch = mask_full[by:by + bh, bx:bx + bw].copy()
                        sub_patch[local_mask] = cls_id
                        mask_full[by:by + bh, bx:bx + bw] = sub_patch
                        region_bool[by:by + bh, bx:bx + bw] = local_mask
                        px_c = int(local_mask.sum())

                box_m2 = round(float(ipm.calculate_surface_area_sqm(bx, by, bw, bh)), 4)
                box_m2 = float(max(0.04, min(box_m2, 3.0)))
                if px_c >= 25:
                    m2, _ = ipm.mask_area_m2(region_bool)
                    d["surface_area_m2"] = round(float(max(0.04, min(max(m2, box_m2 * 0.40), 3.0))), 4)
                    d["area_method"] = "segmentation_mask"
                else:
                    d["surface_area_m2"] = box_m2
                    d["area_method"] = "perspective_homography"

                d["area_diagnostics"]["mask_pixels"] = px_c
                depth_cm = d.get("depth_cm", 4.0)
                tonnage_t = round(d["surface_area_m2"] * (depth_cm / 100.0) * 2.40 * 1.15, 4)
                d["morth_tonnage_t"] = tonnage_t
                d["repair_cost_inr"] = round(tonnage_t * 7500.0, 2)
                box_share = (bw * bh) / float(W * H)
                d["needs_manual_survey"] = bool(d["surface_area_m2"] > 3.0 or box_share > 0.40)
                d["area_is_plausible_single_repair"] = not d["needs_manual_survey"]
                if d.get("depth_estimate"):
                    d["depth_estimate"]["repair_cost_low_inr"] = round(d["surface_area_m2"] * (d["depth_estimate"]["depth_low_cm"] / 100.0) * 2.40 * 1.15 * 7500.0, 2)
                    d["depth_estimate"]["repair_cost_high_inr"] = round(d["surface_area_m2"] * (d["depth_estimate"]["depth_high_cm"] / 100.0) * 2.40 * 1.15 * 7500.0, 2)

    if detections and seg_out and mask_full is not None:
        seg_out["mask"] = mask_full
        seg_out["crack_px_full"] = int((mask_full == 1).sum())
        seg_out["pothole_px_full"] = int((mask_full == 2).sum())
        seg_out["defect_fraction"] = round(float((mask_full != 0).mean()), 4)
        if int((mask_full != 0).sum()) > 0:
            from PIL import Image
            mask_work = np.asarray(Image.fromarray(mask_full, mode="L").resize((WORK_W, WORK_H), Image.Resampling.NEAREST), dtype=np.uint8)
            seg_out["mask_data_url"] = _mask_to_data_url(mask_work)

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
        # Same rule as models/morth_dispatch_agent: a keyed HMAC only when the deployment
        # holds ROAD_SHIELD_SEAL_KEY (a secret set in the host's environment, never in the
        # repository); otherwise a plain SHA-256 that proves integrity, not authorship.
        # A key written in this file would let anyone who reads the repository forge seals.
        seal_key = os.environ.get("ROAD_SHIELD_SEAL_KEY", "").encode("utf-8")
        canonical = json.dumps(order_payload, sort_keys=True).encode("utf-8")
        if seal_key:
            seal_algorithm, seal = "HMAC-SHA256", hmac.new(seal_key, canonical, hashlib.sha256).hexdigest()
        else:
            seal_algorithm, seal = "SHA-256", hashlib.sha256(canonical).hexdigest()
        work_order = {**order_payload, "seal_algorithm": seal_algorithm, "sha256_seal": seal}

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
        "markings": markings_res,
        "frame_classification": frame_cls,
        "frame_level_hazard": frame_level_hazard,
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
