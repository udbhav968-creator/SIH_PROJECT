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
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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
            return image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        pil_img = pil_img.resize((self.target_w, self.target_h), Image.Resampling.BILINEAR)
        return np.array(pil_img, dtype=np.uint8)

    def detect_pedestrians(self, img_np):
        """
        Runs OpenCV's pretrained HOG+SVM person detector. Returns a list of
        detections; empty list if nobody is in frame - we don't invent one.
        """
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

        active = (cell_score >= 30.0).astype(np.uint8)
        if active.sum() < 2:
            return []

        num_labels, labels = cv2.connectedComponents(active, connectivity=4)
        boxes = []
        for label in range(1, num_labels):
            cells = np.argwhere(labels == label)
            if len(cells) < 2:
                continue
            min_r, max_r = cells[:, 0].min(), cells[:, 0].max()
            min_c, max_c = cells[:, 1].min(), cells[:, 1].max()

            bx = max(10, min_c * cell_w - int(cell_w * 0.2))
            by = max(roi_y0, roi_y0 + min_r * cell_h - int(cell_h * 0.15))
            bw = min(W - bx - 10, (max_c - min_c + 1) * cell_w + int(cell_w * 0.4))
            bh = min(H - by - 10, (max_r - min_r + 1) * cell_h + int(cell_h * 0.3))
            if bw <= 0 or bh <= 0:
                continue

            kinds = [int(cell_kind[cr, cc]) for cr, cc in cells]
            is_pothole_like = kinds.count(2) >= kinds.count(1)
            mean_score = float(np.mean([cell_score[cr, cc] for cr, cc in cells]))
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
