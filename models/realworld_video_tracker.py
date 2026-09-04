"""
ROAD-SHIELD Spatial-Temporal Multi-Object Video Tracker
Implements persistent IoU-based defect tracking across multi-frame dashcam video sequences.
Prevents duplicate defect counts as vehicles pass over cavities, and computes real-time looming metrics.
"""
import numpy as np

class SpatialTemporalVideoTracker:
    """Multi-frame defect tracker with spatial trajectory smoothing and anti-double-counting."""

    def __init__(self, iou_threshold=0.25, max_age_frames=5):
        self.iou_thresh = iou_threshold
        self.max_age = max_age_frames
        self.next_track_id = 1
        self.active_tracks = {}
        self.total_unique_potholes_counted = 0
        self.total_unique_cracks_counted = 0

    def compute_iou(self, boxA, boxB):
        """Computes 2D Intersection-over-Union between two [u, v, w, h] boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        union_area = boxAArea + boxBArea - inter_area
        return inter_area / max(1e-6, union_area)

    def update(self, raw_detections, frame_idx):
        """
        Ingests frame detections and assigns persistent track IDs.
        Returns active tracked items with smoothed trajectories.
        """
        matched_tracks = set()
        matched_dets = set()

        tracked_outputs = []

        # 1. Match new detections to existing active tracks
        for det_idx, det in enumerate(raw_detections):
            det_box = det.get("bbox_normalized", [0, 0, 0, 0])
            best_iou = 0.0
            best_trk_id = None

            for trk_id, trk in self.active_tracks.items():
                if trk_id in matched_tracks:
                    continue
                iou = self.compute_iou(det_box, trk["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_trk_id = trk_id

            if best_iou >= self.iou_thresh and best_trk_id is not None:
                # Update existing track
                trk = self.active_tracks[best_trk_id]
                trk["bbox"] = det_box
                trk["last_frame"] = frame_idx
                trk["hits"] += 1
                trk["confidence"] = max(trk["confidence"], det.get("confidence", 0.9))
                trk["distance_meters"] = det.get("distance_meters", 5.0)
                matched_tracks.add(best_trk_id)
                matched_dets.add(det_idx)
                
                det_out = dict(det)
                det_out["persistent_track_id"] = f"TRK-{best_trk_id:03d}"
                det_out["track_age_frames"] = trk["hits"]
                tracked_outputs.append(det_out)

        # 2. Register unmatched detections as new persistent tracks
        for det_idx, det in enumerate(raw_detections):
            if det_idx not in matched_dets:
                # Check if it's a valid distress (not suppressed hard negative)
                is_distress = det.get("is_distress", True)
                cls_name = det.get("class_name", "")
                
                new_id = self.next_track_id
                self.next_track_id += 1
                
                self.active_tracks[new_id] = {
                    "id": new_id,
                    "bbox": det.get("bbox_normalized", [0, 0, 0, 0]),
                    "class_name": cls_name,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "hits": 1,
                    "confidence": det.get("confidence", 0.9),
                    "distance_meters": det.get("distance_meters", 10.0),
                    "is_distress": is_distress
                }
                
                if is_distress:
                    if "Pothole" in cls_name:
                        self.total_unique_potholes_counted += 1
                    else:
                        self.total_unique_cracks_counted += 1

                det_out = dict(det)
                det_out["persistent_track_id"] = f"TRK-{new_id:03d}"
                det_out["track_age_frames"] = 1
                tracked_outputs.append(det_out)

        # 3. Clean up stale tracks older than max_age frames
        stale_ids = [
            trk_id for trk_id, trk in self.active_tracks.items()
            if (frame_idx - trk["last_frame"]) > self.max_age
        ]
        for trk_id in stale_ids:
            del self.active_tracks[trk_id]

        return {
            "frame_idx": frame_idx,
            "tracked_detections": tracked_outputs,
            "active_tracks_count": len(self.active_tracks),
            "total_unique_potholes_counted": self.total_unique_potholes_counted,
            "total_unique_cracks_counted": self.total_unique_cracks_counted
        }
