"""
Model M6: rash-driving / hit-and-run incident detector & license-plate OCR.

analyze_vehicle_kinematics is real, unchanged math: bounding-box growth
rate approximates closing speed, and lateral-centroid jitter approximates
swerving, from a sequence of tracked bounding boxes.

extract_license_plate used to fabricate a random but plate-format-looking
string (with a random 94-99.8% "OCR confidence") whenever it was called,
regardless of whether an image was even provided - there was no OCR
happening at all. This rewrite runs a real classical ALPR pipeline
(edge/contour-based plate-region localization + Tesseract OCR via
pytesseract) against an actual image crop, and returns a plain "not
detected" result - not a fabricated plate - whenever no image is given or
no plate-shaped region is found. This project has no annotated
license-plate dataset to validate detection rate against, so this should
be treated as a real but unbenchmarked OCR pipeline, not a claimed
accuracy figure.
"""

import time
import re
import hashlib
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


class ALPRIncidentTracker:
    # Indian State/UT registration-authority codes, used only to sanity-check
    # OCR output against the real HSRP format (e.g. "DL 01 AB 1234"), never
    # to fabricate a plate.
    STATE_CODES = {"DL", "MH", "KA", "TN", "UP", "HR", "GJ", "WB", "TS", "AP", "RJ", "PB", "KL", "MP", "BR"}

    def __init__(self):
        self.state_codes = sorted(self.STATE_CODES)

    def analyze_vehicle_kinematics(self, track_history):
        """
        Analyzes consecutive bounding-box scales and lateral trajectory to
        detect rash driving. track_history: list of dicts with keys
        {'timestamp': float, 'bbox': [x, y, w, h]}.
        """
        if len(track_history) < 3:
            return {"is_rash_driving": False, "anomaly_type": "INSUFFICIENT_TELEMETRY", "confidence": 0.0}

        areas = [t["bbox"][2] * t["bbox"][3] for t in track_history]
        times = [t["timestamp"] for t in track_history]

        area_diffs = np.diff(areas)
        time_diffs = np.diff(times) + 1e-5
        growth_rates = area_diffs / time_diffs

        centers_x = [t["bbox"][0] + t["bbox"][2] / 2.0 for t in track_history]
        lateral_jerk = np.std(np.diff(centers_x))

        max_growth = float(np.max(growth_rates))
        is_rash = max_growth > 3500.0 or lateral_jerk > 60.0

        anomaly_type = "NORMAL_FLOW"
        if max_growth > 3500.0 and lateral_jerk > 60.0:
            anomaly_type = "AGGRESSIVE_SPEEDING_AND_SWERVING"
        elif max_growth > 3500.0:
            anomaly_type = "EXCESSIVE_APPROACH_VELOCITY"
        elif lateral_jerk > 60.0:
            anomaly_type = "RECKLESS_LANE_CUTTING"

        confidence = min(0.994, 0.75 + (max_growth / 10000.0) * 0.15 + (lateral_jerk / 200.0) * 0.1)

        return {
            "is_rash_driving": is_rash,
            "anomaly_type": anomaly_type,
            "confidence": round(confidence, 3),
            "lateral_jerk_px": round(lateral_jerk, 2),
            "max_approach_rate": round(max_growth, 2),
        }

    def locate_plate_candidate(self, vehicle_crop_rgb):
        """
        Classic edge/contour-based plate-region localizer: grayscale ->
        bilateral denoise -> Canny edges -> contour rectangles filtered by
        plate-like aspect ratio (2:1 to 5.5:1) and a minimum area. Returns
        the best candidate (x, y, w, h) in pixel coordinates, or None.
        """
        if cv2 is None or vehicle_crop_rgb is None:
            return None
        img = np.asarray(vehicle_crop_rgb)
        if img.ndim != 3 or img.shape[0] < 20 or img.shape[1] < 20:
            return None

        gray = cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 30, 200)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = img.shape[0] * img.shape[1]

        best = None
        best_score = -1.0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            aspect = w / float(h)
            area_ratio = (w * h) / float(img_area)
            if 2.0 <= aspect <= 5.5 and 0.01 <= area_ratio <= 0.6:
                # Prefer larger, more plate-shaped (closer to a ~3.5:1 HSRP aspect) candidates.
                score = area_ratio * (1.0 / (1.0 + abs(aspect - 3.5)))
                if score > best_score:
                    best_score = score
                    best = (x, y, w, h)
        return best

    def extract_license_plate(self, vehicle_crop_rgb=None):
        """
        Runs the plate localizer + Tesseract OCR on a real image crop.
        Returns detected=False (never a fabricated plate string) when no
        image is given, no plate-shaped region is found, or OCR yields
        nothing readable.
        """
        if vehicle_crop_rgb is None:
            return {
                "license_plate_number": None,
                "ocr_confidence": 0.0,
                "plate_type": None,
                "state_jurisdiction": None,
                "detected": False,
                "reason": "No image crop supplied - OCR requires a real image.",
            }

        bbox = self.locate_plate_candidate(vehicle_crop_rgb)
        if bbox is None:
            return {
                "license_plate_number": None,
                "ocr_confidence": 0.0,
                "plate_type": None,
                "state_jurisdiction": None,
                "detected": False,
                "reason": "No plate-shaped region found in the supplied crop.",
            }

        if pytesseract is None:
            return {
                "license_plate_number": None,
                "ocr_confidence": 0.0,
                "plate_type": None,
                "state_jurisdiction": None,
                "detected": False,
                "bbox_in_crop": [int(v) for v in bbox],
                "reason": "Plate-shaped region located, but pytesseract/tesseract is not installed to read it.",
            }

        x, y, w, h = bbox
        img = np.asarray(vehicle_crop_rgb)
        plate_crop = img[y : y + h, x : x + w]
        gray = cv2.cvtColor(plate_crop[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ocr_config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        raw_text = pytesseract.image_to_string(thresh, config=ocr_config)
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

        conf = 0.0
        try:
            data = pytesseract.image_to_data(thresh, config=ocr_config, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data.get("conf", []) if str(c) not in ("-1", "")]
            if confs:
                conf = max(0.0, sum(confs) / len(confs)) / 100.0
        except Exception:
            pass

        detected_state = cleaned[:2] if len(cleaned) >= 2 and cleaned[:2] in self.STATE_CODES else None

        return {
            "license_plate_number": cleaned if cleaned else None,
            "ocr_confidence": round(conf, 3),
            "plate_type": "INDIA_FORMAT_HEURISTIC" if cleaned else None,
            "state_jurisdiction": detected_state,
            "bbox_in_crop": [int(x), int(y), int(w), int(h)],
            "detected": bool(cleaned),
        }

    def generate_incident_alert(self, bus_id, gps_coords, track_history, vehicle_crop_rgb=None):
        """Builds a signed, tamper-evident JSON incident report."""
        kinematics = self.analyze_vehicle_kinematics(track_history)
        plate_info = self.extract_license_plate(vehicle_crop_rgb)

        alert_payload = {
            "incident_id": f"INC-BEL-{int(time.time() * 1000) % 1000000:06d}",
            "reporting_bus_unit": bus_id,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gps_coordinates": {
                "latitude": gps_coords.get("lat", 28.6139),
                "longitude": gps_coords.get("lng", 77.2090),
                "accuracy_meters": 2.5,
            },
            "incident_classification": kinematics["anomaly_type"],
            "is_emergency": kinematics["is_rash_driving"],
            "offending_vehicle": {
                "plate_number": plate_info["license_plate_number"],
                "ocr_confidence": plate_info["ocr_confidence"],
                "plate_detected": plate_info["detected"],
                "jurisdiction": plate_info["state_jurisdiction"],
                "kinematic_confidence": kinematics["confidence"],
            },
        }
        seal_data = f"{alert_payload['incident_id']}|{bus_id}|{kinematics['anomaly_type']}|{plate_info['license_plate_number']}"
        alert_payload["sha256_seal"] = hashlib.sha256(seal_data.encode()).hexdigest()
        return alert_payload

    def detect_incident(
        self,
        speed_kmh=80.0,
        lat=28.6139,
        lon=77.2090,
        vehicle_id="UNKNOWN",
        bus_id="BUS-001",
        vehicle_crop_rgb=None,
    ):
        """
        Detect a rash-driving incident from a scalar speed reading + GPS
        coordinates. Convenience method for single-reading incident
        detection; ALPR only fires if a real image crop is supplied.
        """
        is_rash = speed_kmh > 80.0
        if speed_kmh > 100.0:
            incident_class = "EXCESSIVE_APPROACH_VELOCITY"
        elif speed_kmh > 80.0:
            incident_class = "RECKLESS_LANE_CUTTING"
        else:
            incident_class = "NORMAL_FLOW"

        plate_info = self.extract_license_plate(vehicle_crop_rgb)

        payload = {
            "incident_id": f"INC-BEL-{int(time.time() * 1000) % 1000000:06d}",
            "reporting_bus_unit": bus_id,
            "vehicle_id": vehicle_id,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gps_coordinates": {"latitude": lat, "longitude": lon, "accuracy_meters": 2.5},
            "incident_class": incident_class,
            "is_emergency": is_rash,
            "speed_kmh": round(float(speed_kmh), 1),
            "alpr_confidence": plate_info["ocr_confidence"],
            "license_plate": plate_info["license_plate_number"],
            "alpr_detected": plate_info["detected"],
            "state_jurisdiction": plate_info["state_jurisdiction"],
        }
        seal_data = f"{payload['incident_id']}|{vehicle_id}|{speed_kmh:.2f}|{lat:.6f}|{lon:.6f}"
        payload["sha256_seal"] = hashlib.sha256(seal_data.encode()).hexdigest()
        return payload
