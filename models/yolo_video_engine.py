"""
Dedicated Pretrained-YOLO Video Detection & Multi-Object Tracking Engine.

Runs ONLY the pretrained YOLO ONNX detectors for high-speed video stream analysis:
  1. YOLOv11n Pretrained Road Damage Detector (`checkpoints/damage_rdd2022_india.onnx`)
     - Classes: Longitudinal Crack (D00), Transverse Crack (D10), Alligator Crack (D20), Pothole (D40)
  2. YOLOv8n Pretrained COCO Scene & Traffic Detector (`checkpoints/road_shield_detector.onnx`)
     - Classes: 80 COCO classes (car, bus, truck, motorcycle, bicycle, person, traffic light, stop sign, etc.)
  3. YOLOv8n Pretrained Crosswalk & Road Marking Detector (`checkpoints/crosswalk_rdd.onnx`)
     - Classes: Zebra Crosswalk, Road Guide Arrow
"""

import base64
import io
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from models.onnx_object_detector import ONNXObjectDetector
from models.road_damage_detector import RoadDamageDetector

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

ROAD_SCENE_COCO_KEEP = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "traffic light", "fire hydrant", "stop sign", "parking meter",
}

CLASS_STYLE_MAP = {
    "Pothole": ("road_damage", "#ef4444", True),
    "Alligator Crack": ("road_damage", "#f97316", True),
    "Alligator crack": ("road_damage", "#f97316", True),
    "Longitudinal Crack": ("road_damage", "#f59e0b", True),
    "Longitudinal crack": ("road_damage", "#f59e0b", True),
    "Transverse Crack": ("road_damage", "#eab308", True),
    "Transverse crack": ("road_damage", "#eab308", True),
    "Zebra Crosswalk": ("road_marking", "#06b6d4", False),
    "Road Guide Arrow": ("road_marking", "#22d3ee", False),
    "person": ("pedestrian", "#ec4899", False),
    "bicycle": ("vehicle", "#a855f7", False),
    "motorcycle": ("vehicle", "#8b5cf6", False),
    "car": ("vehicle", "#10b981", False),
    "bus": ("vehicle", "#3b82f6", False),
    "truck": ("vehicle", "#6366f1", False),
    "train": ("vehicle", "#64748b", False),
    "traffic light": ("traffic_control", "#84cc16", False),
    "stop sign": ("traffic_control", "#f43f5e", False),
}


def _box_iou_xywh(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two normalized [x, y, w, h] boxes."""
    ax1, ay1 = float(box_a[0]), float(box_a[1])
    ax2, ay2 = ax1 + float(box_a[2]), ay1 + float(box_a[3])
    bx1, by1 = float(box_b[0]), float(box_b[1])
    bx2, by2 = bx1 + float(box_b[2]), by1 + float(box_b[3])

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


class YOLOMultiObjectTracker:
    """
    Lightweight ByteTrack/IoU multi-object tracker for Pretrained YOLO video streams.
    Assigns persistent track_ids and prevents double-counting across consecutive frames.
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 8):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.reset()

    def reset(self):
        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.unique_potholes = set()
        self.unique_cracks = set()
        self.unique_crosswalks = set()
        self.unique_vehicles = set()
        self.unique_pedestrians = set()
        self.unique_traffic_controls = set()

    def update(self, detections: List[Dict[str, Any]], frame_index: int) -> List[Dict[str, Any]]:
        matched_track_ids = set()
        assigned_detections = []

        # Sort detections by confidence descending (ByteTrack high-confidence first)
        sorted_dets = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)

        for det in sorted_dets:
            bbox = det.get("bbox_normalized") or [0.0, 0.0, 0.0, 0.0]
            cat = det.get("category", "scene")
            best_iou = 0.0
            best_tid = None

            for tid, trk in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                if trk["category"] != cat:
                    continue
                iou = _box_iou_xywh(bbox, trk["bbox_normalized"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_threshold:
                trk = self.tracks[best_tid]
                # Exponential moving average box smoothing for stable video overlay
                old_b = trk["bbox_normalized"]
                smoothed_b = [
                    round(0.65 * float(bbox[i]) + 0.35 * float(old_b[i]), 4)
                    for i in range(4)
                ]
                trk["bbox_normalized"] = smoothed_b
                trk["confidence"] = max(float(det.get("confidence", 0.0)), round(0.9 * trk["confidence"], 4))
                trk["last_frame"] = frame_index
                trk["age"] = 0
                trk["hits"] += 1
                matched_track_ids.add(best_tid)
                tid = best_tid
                hits = trk["hits"]
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = {
                    "track_id": tid,
                    "class_name": det.get("class_name", "Object"),
                    "category": cat,
                    "bbox_normalized": list(bbox),
                    "confidence": float(det.get("confidence", 0.0)),
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                    "age": 0,
                    "hits": 1,
                }
                matched_track_ids.add(tid)
                hits = 1

            cls_name = det.get("class_name", "")
            if "pothole" in cls_name.lower():
                self.unique_potholes.add(tid)
            elif "crack" in cls_name.lower():
                self.unique_cracks.add(tid)
            elif cat == "road_marking":
                self.unique_crosswalks.add(tid)
            elif cat == "vehicle":
                self.unique_vehicles.add(tid)
            elif cat == "pedestrian":
                self.unique_pedestrians.add(tid)
            elif cat == "traffic_control":
                self.unique_traffic_controls.add(tid)

            enriched = dict(det)
            enriched["track_id"] = tid
            enriched["frames_seen"] = hits
            assigned_detections.append(enriched)

        # Age unmatched tracks and prune expired ones
        expired = []
        for tid, trk in self.tracks.items():
            if tid not in matched_track_ids:
                trk["age"] += 1
                if trk["age"] > self.max_age:
                    expired.append(tid)
        for tid in expired:
            del self.tracks[tid]

        return assigned_detections

    def summary(self) -> Dict[str, int]:
        return {
            "active_tracks": len(self.tracks),
            "total_unique_tracks": self.next_track_id - 1,
            "unique_potholes": len(self.unique_potholes),
            "unique_cracks": len(self.unique_cracks),
            "unique_crosswalks": len(self.unique_crosswalks),
            "unique_vehicles": len(self.unique_vehicles),
            "unique_pedestrians": len(self.unique_pedestrians),
            "unique_traffic_controls": len(self.unique_traffic_controls),
        }


class YOLOVideoEngine:
    """
    Pure Pretrained-YOLO Video Inference Engine.
    Executes ONLY pretrained YOLO ONNX detectors (Road Damage + COCO Scene + Crosswalk)
    without running the heavy 22-model civil audit pipeline.
    """

    def __init__(self, checkpoints_dir: Optional[str] = None):
        self.ckpt_dir = checkpoints_dir or CKPT_DIR
        self.coco_detector = ONNXObjectDetector(
            checkpoints_dir=self.ckpt_dir,
            weights_path=os.path.join(self.ckpt_dir, "road_shield_detector.onnx"),
            conf_threshold=0.30,
        )
        self.damage_detector = RoadDamageDetector(
            checkpoints_dir=self.ckpt_dir,
        )
        cw_path = os.path.join(self.ckpt_dir, "crosswalk_rdd.onnx")
        self.crosswalk_detector = ONNXObjectDetector(
            checkpoints_dir=self.ckpt_dir,
            weights_path=cw_path if os.path.exists(cw_path) else None,
            class_names=["Zebra Crosswalk", "Road Guide Arrow"],
            conf_threshold=0.25,
        ) if os.path.exists(cw_path) else None

        self.trackers: Dict[str, YOLOMultiObjectTracker] = {}

    @property
    def is_ready(self) -> bool:
        return bool(self.coco_detector.is_ready or self.damage_detector.is_ready)

    def active_models_manifest(self) -> List[Dict[str, Any]]:
        models = []
        if self.damage_detector and self.damage_detector.is_ready:
            models.append({
                "id": "yolo11n_rdd2022",
                "name": "Pretrained YOLOv11n Road Damage (RDD2022)",
                "backend": self.damage_detector.backend,
                "input_resolution": f"{self.damage_detector.input_size}x{self.damage_detector.input_size}",
                "classes": ["Pothole", "Alligator Crack", "Longitudinal Crack", "Transverse Crack"],
            })
        if self.coco_detector and self.coco_detector.is_ready:
            models.append({
                "id": "yolov8n_coco",
                "name": "Pretrained YOLOv8n COCO Traffic & Scene",
                "backend": self.coco_detector.backend,
                "input_resolution": f"{self.coco_detector.input_size}x{self.coco_detector.input_size}",
                "classes": sorted(ROAD_SCENE_COCO_KEEP),
            })
        if self.crosswalk_detector and self.crosswalk_detector.is_ready:
            models.append({
                "id": "yolov8n_crosswalk",
                "name": "Pretrained YOLOv8n Crosswalk & Markings",
                "backend": self.crosswalk_detector.backend,
                "input_resolution": f"{self.crosswalk_detector.input_size}x{self.crosswalk_detector.input_size}",
                "classes": ["Zebra Crosswalk", "Road Guide Arrow"],
            })
        return models

    @staticmethod
    def decode_image(image_input: Any, max_dim: int = 960) -> np.ndarray:
        if isinstance(image_input, np.ndarray):
            return image_input
        if isinstance(image_input, str):
            raw = image_input.strip()
            if os.path.isfile(raw):
                pil_img = Image.open(raw).convert("RGB")
            else:
                if "," in raw and raw.startswith("data:"):
                    raw = raw.split(",", 1)[1]
                img_bytes = base64.b64decode(raw)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil_img.size
            if max(w, h) > max_dim:
                scale = float(max_dim) / float(max(w, h))
                pil_img = pil_img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.BILINEAR,
                )
            return np.asarray(pil_img, dtype=np.uint8)
        raise ValueError("Unsupported image_input format for YOLOVideoEngine")

    def get_tracker(self, session_id: str = "default", reset: bool = False) -> YOLOMultiObjectTracker:
        if reset or session_id not in self.trackers:
            self.trackers[session_id] = YOLOMultiObjectTracker()
        return self.trackers[session_id]

    def _detect_road_damage(self, img_rgb: np.ndarray, conf_threshold: float) -> List[Dict[str, Any]]:
        if not self.damage_detector or not self.damage_detector.is_ready:
            return []
        from models.onnx_object_detector import decode_output, letterbox, nms, unletterbox, xywh_to_xyxy

        arr = np.asarray(img_rgb, dtype=np.uint8)
        orig_h, orig_w = arr.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(arr, self.damage_detector.input_size)
        blob = np.asarray(canvas, dtype=np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        if self.damage_detector._session is not None:
            raw = self.damage_detector._session.run(None, {self.damage_detector._input_name: blob})[0].copy()
        else:
            return self.damage_detector.detect(img_rgb, conf_threshold=conf_threshold)

        # Apply RDD2022 inverse class-frequency prior balancing [D00, D10, D20, D40]
        priors = np.array([1.0, 2.2, 1.35, 2.15], dtype=np.float32)
        if raw.ndim == 3 and raw.shape[1] == 8:
            raw[0, 4:8, :] *= priors[:, None]

        boxes_xywh, conf, cls_ids = decode_output(raw, 4, conf_threshold)
        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = unletterbox(xywh_to_xyxy(boxes_xywh), scale, pad_x, pad_y, orig_w, orig_h)
        keep_idx = nms(boxes_xyxy, conf, 0.45)

        names = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Pothole"]
        out = []
        for idx in keep_idx:
            cid = int(cls_ids[idx])
            cname = names[cid] if 0 <= cid < len(names) else "Road Damage"
            x1, y1, x2, y2 = boxes_xyxy[idx]
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if bw < 4 or bh < 4:
                continue
            raw_score = float(conf[idx])
            cal_score = min(0.98, round(0.48 + raw_score * 1.85, 4))
            cat, color, is_distress = CLASS_STYLE_MAP.get(cname, ("road_damage", "#ef4444", True))
            out.append({
                "class_name": cname,
                "raw_class": cname,
                "category": cat,
                "is_distress": is_distress,
                "confidence": cal_score,
                "raw_confidence": round(raw_score, 4),
                "bbox_pixels": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "bbox_normalized": [
                    round(float(x1 / orig_w), 4),
                    round(float(y1 / orig_h), 4),
                    round(float(bw / orig_w), 4),
                    round(float(bh / orig_h), 4),
                ],
                "color": color,
                "source_model": "YOLOv11n-RDD2022",
            })
        return out

    def detect_frame(
        self,
        image_input: Any,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
        conf_threshold: float = 0.25,
        session_id: str = "default",
        reset_tracker: bool = False,
        include_coco: bool = True,
        include_damage: bool = True,
        include_crosswalk: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        img_rgb = self.decode_image(image_input)
        h, w = img_rgb.shape[:2]

        raw_detections: List[Dict[str, Any]] = []

        # 1. Pretrained YOLOv11n Road Damage Detector (RDD2022)
        if include_damage and self.damage_detector and self.damage_detector.is_ready:
            dmg_conf = max(0.075, float(conf_threshold) * 0.35)
            raw_detections.extend(self._detect_road_damage(img_rgb, conf_threshold=dmg_conf))

        # 2. Pretrained YOLOv8n Crosswalk & Road Marking Detector
        if include_crosswalk and self.crosswalk_detector and self.crosswalk_detector.is_ready:
            cw_conf = max(0.20, float(conf_threshold))
            for d in self.crosswalk_detector.detect(img_rgb, conf_threshold=cw_conf):
                cname = d["class_name"]
                cat, color, is_distress = CLASS_STYLE_MAP.get(cname, ("road_marking", "#06b6d4", False))
                raw_detections.append({
                    "class_name": cname,
                    "raw_class": cname,
                    "category": cat,
                    "is_distress": is_distress,
                    "confidence": round(float(d["confidence"]), 4),
                    "bbox_pixels": d["bbox_pixels"],
                    "bbox_normalized": d["bbox_normalized"],
                    "color": color,
                    "source_model": "YOLOv8n-Crosswalk",
                })

        # 3. Pretrained YOLOv8n COCO Scene & Traffic Detector
        if include_coco and self.coco_detector and self.coco_detector.is_ready:
            coco_conf = max(0.22, float(conf_threshold))
            for d in self.coco_detector.detect(
                img_rgb, conf_threshold=coco_conf, keep_classes=ROAD_SCENE_COCO_KEEP
            ):
                cname = d["class_name"]
                cat, color, is_distress = CLASS_STYLE_MAP.get(cname, ("scene", "#38bdf8", False))
                display_name = cname.replace("_", " ").title()
                raw_detections.append({
                    "class_name": display_name,
                    "raw_class": cname,
                    "category": cat,
                    "is_distress": is_distress,
                    "confidence": round(float(d["confidence"]), 4),
                    "bbox_pixels": d["bbox_pixels"],
                    "bbox_normalized": d["bbox_normalized"],
                    "color": color,
                    "source_model": "YOLOv8n-COCO",
                })

        tracker = self.get_tracker(session_id=session_id, reset=reset_tracker)
        tracked = tracker.update(raw_detections, frame_index=frame_index)

        frame_counts = {
            "potholes": sum(1 for d in tracked if "pothole" in d["class_name"].lower()),
            "cracks": sum(1 for d in tracked if "crack" in d["class_name"].lower()),
            "crosswalks": sum(1 for d in tracked if d["category"] == "road_marking"),
            "vehicles": sum(1 for d in tracked if d["category"] == "vehicle"),
            "pedestrians": sum(1 for d in tracked if d["category"] == "pedestrian"),
            "traffic_controls": sum(1 for d in tracked if d["category"] == "traffic_control"),
            "total": len(tracked),
        }

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        fps_est = round(1000.0 / max(latency_ms, 1.0), 1)

        return {
            "status": "SUCCESS",
            "pipeline_mode": "PRETRAINED_YOLO_ONLY",
            "engine": "Pretrained YOLO Ensemble (YOLOv11n-RDD2022 + YOLOv8n-COCO + YOLOv8n-Crosswalk)",
            "models_active": self.active_models_manifest(),
            "frame_index": int(frame_index),
            "timestamp_s": round(float(timestamp_s), 3),
            "image_width": int(w),
            "image_height": int(h),
            "detections": tracked,
            "frame_counts": frame_counts,
            "tracker_summary": tracker.summary(),
            "latency_ms": latency_ms,
            "fps_estimate": fps_est,
        }

    def ingest_video(
        self,
        video_path: str,
        sample_every_s: float = 0.5,
        max_frames: int = 120,
        conf_threshold: float = 0.25,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file from disk using ONLY the Pretrained YOLO models.
        """
        import cv2

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = round(total_frames / fps, 2) if fps > 0 else 0.0

        step_frames = max(1, int(round(fps * max(0.05, float(sample_every_s)))))
        sid = session_id or f"video_ingest_{int(time.time() * 1000)}"
        self.get_tracker(session_id=sid, reset=True)

        timeline: List[Dict[str, Any]] = []
        analysed = 0
        frame_idx = 0
        t_start = time.perf_counter()

        while cap.isOpened() and analysed < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % step_frames != 0:
                frame_idx += 1
                continue

            t_sec = frame_idx / fps
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.detect_frame(
                image_input=rgb,
                frame_index=frame_idx,
                timestamp_s=t_sec,
                conf_threshold=conf_threshold,
                session_id=sid,
                reset_tracker=False,
            )
            analysed += 1

            top_det = res["detections"][0] if res["detections"] else None
            timeline.append({
                "frame_index": frame_idx,
                "timestamp_s": round(t_sec, 2),
                "total_detections": res["frame_counts"]["total"],
                "potholes": res["frame_counts"]["potholes"],
                "cracks": res["frame_counts"]["cracks"],
                "crosswalks": res["frame_counts"]["crosswalks"],
                "vehicles": res["frame_counts"]["vehicles"],
                "pedestrians": res["frame_counts"]["pedestrians"],
                "class_name": top_det["class_name"] if top_det else "Clear Roadway",
                "confidence": top_det["confidence"] if top_det else 0.0,
                "source_model": top_det["source_model"] if top_det else "YOLOv8n/v11n",
                "is_defect": bool(res["frame_counts"]["potholes"] + res["frame_counts"]["cracks"] > 0),
                "detections": res["detections"],
                "latency_ms": res["latency_ms"],
            })
            frame_idx += 1

        cap.release()
        elapsed_s = round(time.perf_counter() - t_start, 2)
        tracker_stats = self.get_tracker(session_id=sid).summary()
        defect_frames = sum(1 for f in timeline if f["is_defect"])

        return {
            "status": "SUCCESS",
            "pipeline_mode": "PRETRAINED_YOLO_ONLY",
            "engine": "Pretrained YOLO Ensemble (YOLOv11n-RDD2022 + YOLOv8n-COCO + YOLOv8n-Crosswalk)",
            "models_active": self.active_models_manifest(),
            "video": {
                "path": os.path.basename(video_path),
                "fps": round(fps, 2),
                "frame_count": total_frames,
                "duration_s": duration_s,
                "width": width,
                "height": height,
            },
            "sampling": {
                "mode": "by_time_yolo_only",
                "sample_every_s": sample_every_s,
                "step_frames": step_frames,
                "max_frames": max_frames,
            },
            "frames_analysed": analysed,
            "frames_skipped_as_near_duplicates": 0,
            "defect_frames": defect_frames,
            "unique_defects_after_dedup": tracker_stats["unique_potholes"] + tracker_stats["unique_cracks"],
            "tracker_summary": tracker_stats,
            "seconds_elapsed": elapsed_s,
            "ms_per_analysed_frame": round((elapsed_s * 1000.0) / max(1, analysed), 1),
            "detections": timeline,
        }


_GLOBAL_YOLO_VIDEO_ENGINE: Optional[YOLOVideoEngine] = None


def get_yolo_video_engine() -> YOLOVideoEngine:
    global _GLOBAL_YOLO_VIDEO_ENGINE
    if _GLOBAL_YOLO_VIDEO_ENGINE is None:
        _GLOBAL_YOLO_VIDEO_ENGINE = YOLOVideoEngine()
    return _GLOBAL_YOLO_VIDEO_ENGINE
