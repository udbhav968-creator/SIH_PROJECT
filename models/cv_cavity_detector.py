"""
Classical computer-vision front end for the pipeline: turns a raw road photo
into (a) a list of candidate "something's wrong with the pavement here"
bounding boxes and (b) a list of detected pedestrians. Neither of these steps
uses a trained model - the region proposals are gradient/brightness grid
clustering (a normal pre-deep-learning technique for this kind of anomaly
spotting), and pedestrian detection is OpenCV's bundled HOG+SVM person
detector, which is genuinely pretrained (on the INRIA person dataset) and
ships inside opencv-python, so it works completely offline.

The actual "what kind of distress is this" decision is left to
VisionDistressNet, which is called from pipeline/deep_inference_pipeline.py
on the crops this module proposes - this file only does image processing.
"""

import os
import base64
import io
import numpy as np
import cv2
from PIL import Image


class CVCavityDetector:
    """Decodes input images and proposes candidate regions / pedestrians."""

    def __init__(self, target_size=(640, 480)):
        self.target_w, self.target_h = target_size
        # OpenCV 5 dropped HOGDescriptor from the main package. Rather than
        # crash the whole server over one optional detector, run without
        # pedestrian detection and say so (see pedestrian_detector_available).
        self._hog = None
        if hasattr(cv2, "HOGDescriptor"):
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    @property
    def pedestrian_detector_available(self):
        return self._hog is not None

    def decode_image(self, image_input):
        if isinstance(image_input, str):
            if os.path.exists(image_input):
                pil_img = Image.open(image_input).convert("RGB")
            else:
                if "," in image_input:
                    image_input = image_input.split(",", 1)[1]
                image_bytes = base64.b64decode(image_input)
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Arrays (camera frames from edge/bus_agent.py, video frames) used to be returned at
            # their own resolution while files, uploads and bytes were resized below, so the
            # same photograph gave different detections by input type: on the 36 clean / 24
            # defect check photographs, 6/36 false positives as an array vs 3/36 as a file.
            # Every input is now analysed at the working resolution the engine is measured at.
            arr = np.asarray(image_input)
            if arr.ndim == 2:
                arr = np.dstack([arr] * 3)
            arr = np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)
            if arr.shape[1] == self.target_w and arr.shape[0] == self.target_h:
                return arr
            pil_img = Image.fromarray(arr)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        pil_img = pil_img.resize((self.target_w, self.target_h), Image.Resampling.BILINEAR)
        return np.array(pil_img, dtype=np.uint8)

    def detect_pedestrians(self, img_np):
        """
        Runs OpenCV's pretrained HOG+SVM person detector. Returns a list of
        detections; empty list if nobody is in frame - we don't invent one.
        """
        if self._hog is None:
            return []
        H, W = img_np.shape[:2]
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # winStride/scale tuned for dashcam-style frames; padding helps catch
        # people near the image edge.
        rects, weights = self._hog.detectMultiScale(
            bgr, winStride=(8, 8), padding=(8, 8), scale=1.05
        )

        pedestrians = []
        for i, (x, y, w, h) in enumerate(rects):
            score = float(weights[i]) if len(weights) > i else 0.5
            # The SVM decision score is unbounded; squash it to a confidence.
            confidence = float(1.0 / (1.0 + np.exp(-score)))
            if confidence < 0.55:
                continue
            x, y = max(0, int(x)), max(0, int(y))
            w, h = min(W - x, int(w)), min(H - y, int(h))
            if w <= 0 or h <= 0:
                continue
            feet_y = y + h
            dist_m = round(max(1.5, 22.0 * (1.0 - (feet_y / float(H)) ** 0.85)), 1)
            pedestrians.append(
                {
                    "bbox_pixels": [x, y, w, h],
                    "bbox_normalized": [round(x / W, 4), round(y / H, 4), round(w / W, 4), round(h / H, 4)],
                    "pedestrian_id": i + 1,
                    "confidence": round(confidence, 4),
                    "distance_meters": dist_m,
                    "detector": "opencv_hog_svm_person_detector",
                }
            )
        return pedestrians

    def extract_salient_regions(self, img_np, excluded_boxes=None):
        """
        Grid-based gradient/darkness anomaly scan of the pavement area
        (below 35% of frame height, to skip sky/foliage). Returns candidate
        boxes as [x, y, w, h, score, kind] where kind is 2 (pothole-like:
        dark, blob-shaped) or 1 (crack-like: high edge energy, thin).
        """
        H, W, _ = img_np.shape
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        hue = hsv[:, :, 0].astype(np.float32)
        is_vegetation = (hue > 35) & (hue < 85) & (sat > 0.25)

        excl_mask = np.zeros((H, W), dtype=bool)
        for eb in excluded_boxes or []:
            ex, ey, ew, eh = eb["bbox_pixels"]
            sy0, sy1 = max(0, ey - 10), min(H, ey + eh + int(eh * 0.45))
            sx0, sx1 = max(0, ex - 35), min(W, ex + ew + 35)
            excl_mask[sy0:sy1, sx0:sx1] = True

        # Suppress non-road vehicle hood / dashboard in bottom 6%
        excl_mask[int(H * 0.94):, :] = True

        # Detect and suppress solid letterbox banner / watermark footer in bottom 18% of frame
        for y in range(int(H * 0.82), H):
            row_med = float(np.median(gray[y, :]))
            if row_med < 28.0 or row_med > 225.0:
                excl_mask[y:, :] = True
                break
        # Detect and suppress solid letterbox banner in top 18% of frame
        for y in range(int(H * 0.18), -1, -1):
            row_med = float(np.median(gray[y, :]))
            if row_med < 28.0 or row_med > 225.0:
                excl_mask[:y + 1, :] = True
                break

        roi_y0 = int(H * 0.35)
        road_gray = gray[roi_y0:, :].astype(np.float32)
        mean_intensity = float(road_gray.mean())
        std_intensity = float(road_gray.std())
        if std_intensity < 6.5:
            return []  # handled upstream as "not a pavement photo"

        gx = cv2.Sobel(road_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(road_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

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
                cell_veg = is_vegetation[abs_y : abs_y + cell_h, x0 : x0 + cell_w]
                if cell_sat.mean() > 0.18 or cell_veg.mean() > 0.15:
                    continue

                sub_gray = road_gray[y0 : y0 + cell_h, x0 : x0 + cell_w]
                sub_mag = grad_mag[y0 : y0 + cell_h, x0 : x0 + cell_w]
                dark_diff = max(0.0, mean_intensity - float(sub_gray.mean()))
                edge_peak = float(np.percentile(sub_mag, 90)) if sub_mag.size else 0.0

                # de-emphasize the far left/right verges, emphasize the wheel path
                col_weight = 0.40 if (c < 2 or c > 21) else (1.30 if 5 <= c <= 18 else 1.0)
                pothole_score = dark_diff * 2.5 * col_weight if dark_diff > 14.0 else 0.0
                crack_score = edge_peak * 2.2 * col_weight if edge_peak > 18.0 else 0.0

                if pothole_score >= crack_score and pothole_score > 0:
                    cell_score[r, c], cell_kind[r, c] = pothole_score, 2
                elif crack_score > 0:
                    cell_score[r, c], cell_kind[r, c] = crack_score, 1

        # Suppress letterbox footer banner in lower 2 rows
        cell_score[max(0, grid_rows - 2):, :] = 0.0

        peaks = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                sc = float(cell_score[r, c])
                if sc < 60.0:
                    continue
                r_min, r_max = max(0, r - 1), min(grid_rows, r + 2)
                c_min, c_max = max(0, c - 1), min(grid_cols, c + 2)
                if sc == float(cell_score[r_min:r_max, c_min:c_max].max()):
                    peaks.append((r, c, sc))

        peaks.sort(key=lambda p: p[2], reverse=True)

        # Fallback to highest road anomaly if no peak exceeded 60
        if not peaks and float(cell_score.max()) >= 35.0:
            max_idx = np.unravel_index(np.argmax(cell_score), cell_score.shape)
            peaks.append((int(max_idx[0]), int(max_idx[1]), float(cell_score[max_idx])))

        boxes = []
        for pr, pc, psc in peaks:
            visited = set([(pr, pc)])
            q = [(pr, pc)]
            while q:
                cr, cc = q.pop(0)
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < grid_rows - 1 and 0 <= nc < grid_cols and (nr, nc) not in visited:
                        if cell_score[nr, nc] >= max(35.0, psc * 0.40):
                            if abs(nr - pr) <= 2 and abs(nc - pc) <= 3:
                                visited.add((nr, nc))
                                q.append((nr, nc))
            rs = [p[0] for p in visited]
            cs = [p[1] for p in visited]
            bx = max(10, min(cs) * cell_w - int(cell_w * 0.15))
            by = max(roi_y0, roi_y0 + min(rs) * cell_h - int(cell_h * 0.10))
            bw = min(int(W * 0.38), min(W - bx - 10, (max(cs) - min(cs) + 1) * cell_w + int(cell_w * 0.35)))
            bh = min(int(H * 0.32), min(H - by - 10, (max(rs) - min(rs) + 1) * cell_h + int(cell_h * 0.30)))
            if bw <= 0 or bh <= 0:
                continue

            kinds = [int(cell_kind[cr, cc]) for cr, cc in visited]
            is_pothole_like = kinds.count(2) >= kinds.count(1)
            mean_score = float(np.mean([cell_score[cr, cc] for cr, cc in visited]))
            boxes.append([int(bx), int(by), int(bw), int(bh), mean_score, 2 if is_pothole_like else 1])

        return self._non_max_suppress(boxes)

    @staticmethod
    def _iou(box_a, box_b):
        xA, yA = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        xB = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
        yB = min(box_a[1] + box_a[3], box_b[1] + box_b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area_a, area_b = box_a[2] * box_a[3], box_b[2] * box_b[3]
        return inter / max(1e-6, area_a + area_b - inter)

    def _non_max_suppress(self, boxes, iou_thresh=0.35, containment_thresh=0.55, max_boxes=10):
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[4] * ((b[2] * b[3]) ** 0.20), reverse=True)
        max_area = max(b[2] * b[3] for b in boxes)
        selected = []
        for b in boxes:
            b_area = b[2] * b[3]
            if max_area > 8000 and b_area < 0.08 * max_area:
                continue  # small peripheral noise once a dominant defect exists
            keep = True
            for s in selected:
                s_area = s[2] * s[3]
                xA, yA = max(b[0], s[0]), max(b[1], s[1])
                xB, yB = min(b[0] + b[2], s[0] + s[2]), min(b[1] + b[3], s[1] + s[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                ios = inter / max(1.0, min(b_area, s_area))
                if self._iou(b[:4], s[:4]) > iou_thresh or ios > containment_thresh:
                    keep = False
                    break
            if keep:
                selected.append(b)
                if len(selected) >= max_boxes:
                    break
        return selected
