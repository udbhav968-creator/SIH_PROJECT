"""
DPDP-Act-style redaction of people and number plates before an image leaves
the device.

The Milestone 2 report claimed faces and plates were blurred on the bus before
transmission. No such code existed. This module is that step; what it does
and does not guarantee is stated here rather than implied:

  * people  - every "person" box from the COCO detector (models/onnx_object_detector.py)
              has its upper 35% (the head) blurred; if OpenCV's Haar face cascade
              is installed, any frontal face it finds is blurred as well.
  * plates  - inside every car / bus / truck / motorcycle box, the contour-based
              plate localiser of models/alpr_incident_tracker.py proposes a plate
              rectangle, which is blurred.
  * blur    - Gaussian, kernel scaled to the region (at least 15 x 15 px), so the
              result cannot be read back by sharpening.

NOT claimed: a recall figure. There is no annotated face / plate set in this
project to measure one against, so a missed face or plate is possible. The
returned report lists which detectors actually ran.
"""
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

VEHICLES = {"car", "bus", "truck", "motorcycle"}
HEAD_FRACTION = 0.35


def _blur(img, x, y, w, h):
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    roi = img[y0:y1, x0:x1]
    k = max(15, (max(x1 - x0, y1 - y0) // 3) | 1)  # odd kernel, >= 15
    if cv2 is not None:
        img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    else:  # crude fallback: mean fill, still unreadable
        img[y0:y1, x0:x1] = roi.reshape(-1, roi.shape[-1]).mean(axis=0).astype(img.dtype)
    return True


def _haar_faces(rgb):
    if cv2 is None:
        return None
    path = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not path:
        return None
    import os
    f = os.path.join(path, "haarcascade_frontalface_default.xml")
    if not os.path.exists(f):
        return None
    cascade = cv2.CascadeClassifier(f)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return [tuple(map(int, r)) for r in cascade.detectMultiScale(gray, 1.1, 5, minSize=(16, 16))]


def redact(image_rgb, detections=None, detector=None, plate_locator=None):
    """
    Returns (redacted_copy, report). `detections` is the COCO detector's output
    for this frame (pass it if the pipeline already ran the detector); otherwise
    `detector` is called if it is ready. The input array is never modified.
    """
    img = np.array(image_rgb, dtype=np.uint8, copy=True)
    report = {"people_blurred": 0, "faces_blurred": 0, "plates_blurred": 0,
              "detectors_used": [], "recall_measured": False}

    if detections is None and detector is not None and getattr(detector, "is_ready", False):
        detections = detector.detect(img)
    if detections is not None:
        report["detectors_used"].append("coco_object_detector")
    if plate_locator is None:
        from models.alpr_incident_tracker import ALPRIncidentTracker
        plate_locator = ALPRIncidentTracker().locate_plate_candidate

    for d in detections or []:
        x, y, w, h = d["bbox_pixels"]
        if d["class_name"] == "person":
            if _blur(img, x, y, w, max(2, h * HEAD_FRACTION)):
                report["people_blurred"] += 1
        elif d["class_name"] in VEHICLES:
            crop = img[max(0, y):y + h, max(0, x):x + w]
            box = plate_locator(crop) if crop.size else None
            if box is not None:
                px, py, pw, ph = box
                if _blur(img, x + px, y + py, pw, ph):
                    report["plates_blurred"] += 1
    if detections is not None:
        report["detectors_used"].append("contour_plate_localiser")

    faces = _haar_faces(img)
    if faces is not None:
        report["detectors_used"].append("haar_frontal_face")
        for (x, y, w, h) in faces:
            if _blur(img, x, y, w, h):
                report["faces_blurred"] += 1
    if not report["detectors_used"]:
        report["warning"] = ("No detector was available, so nothing could be found to redact. "
                             "Do not transmit this image as if it had been redacted.")
    return img, report
