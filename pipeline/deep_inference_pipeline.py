"""
Deep inference pipeline: orchestrates the real models in this project into
one end-to-end road-photo audit.

Stages:
1. Decode the input image and standardize it to 640x480.
2. Texture gatekeeper - rejects non-pavement photos via a real luminance
   standard-deviation check (rolled asphalt has visible grain; screenshots,
   walls, and flat graphics don't).
3. Classical CV region proposals (CVCavityDetector): gradient/darkness grid
   clustering for candidate distress regions, OpenCV's pretrained HOG+SVM
   detector for pedestrians. Neither step uses a trained model - they're
   real, standard pre-deep-learning CV techniques.
4. VisionDistressNet classifies each candidate crop (a real scikit-learn
   model trained on the project's labeled photos - see
   training/train_vision.py for its honest, leakage-free held-out accuracy).
5. IPMHomographyEngine converts pixel boxes to real ground-plane area via
   pinhole-camera geometry, and prices asphalt tonnage from MoRTH Section
   500 material rates.
6. IMUShockClassifier scores a real accelerometer window when the caller
   supplies one. If no real IMU reading is available, this pipeline reports
   that plainly instead of fabricating one - the previous version invented
   an IMU signal by transforming the vision model's own output, which
   defeated the entire point of having two independent sensors agree.
7. BayesianFusionGate combines vision + IMU evidence (log-odds).
8. PavementConditionIndexEngine computes a real ASTM D6433-style PCI from
   the frame's actual detected distress, not a placeholder score.
9. PavementDeteriorationForecaster projects area growth over time.
10. MoRTHDispatchAgent issues a SHA-256-sealed work order for confirmed
    structural distress.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import time
import hashlib
import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from models.vision_distress_net import VisionDistressNet
from models.cv_cavity_detector import CVCavityDetector
from models.ipm_homography_engine import IPMHomographyEngine
from models.imu_shock_classifier import IMUShockClassifier
from models.bayesian_fusion_gate import BayesianFusionGate
from models.pci_regressor_net import PavementConditionIndexEngine
from models.pavement_deterioration_forecaster import PavementDeteriorationForecaster
from models.morth_dispatch_agent import MoRTHDispatchAgent
from models.multimodal_transformer_fusion import MultimodalTransformerFusionNet
from models.automotive_rl_policy_agent import AutomotiveADASPolicyAgent
from models.automotive_telematics_engine import AutomotiveTelematicsEngine

# Classes that represent an actual surface footprint (crack, pothole,
# waterlogging) get real IPM-derived area/depth/tonnage math. Classes 4-6
# (missing zebra crossing, missing divider, damaged sign) are presence/
# absence findings - there's no "how many square meters of missing paint"
# to compute, so we report the detection and its IRC remediation reference
# without inventing an asphalt-repair cost for them.
AREA_CLASSES = {1, 2, 3}

# There is no rut-bar or profilometer in this project, so rutting/IRI are
# reported as a documented proxy correlated with detected cavity severity,
# not a real physical measurement. This mirrors the honesty note already in
# pci_regressor_net.py for the deduct-value curves themselves.
ASSUMED_FRAME_PAVEMENT_AREA_M2 = 25.0  # rough visible-pavement extent in one dashcam frame, for density-% estimates


class DeepInferencePipeline:
    """Wires the project's real models together into one road-photo/telemetry audit."""

    def __init__(self, checkpoints_dir=None):
        self.ckpt_dir = checkpoints_dir or os.path.join(ENGINE_ROOT, "checkpoints")

        self.cv_detector = CVCavityDetector(target_size=(640, 480))

        self.vision_model = VisionDistressNet()
        vis_ckpt = os.path.join(self.ckpt_dir, "vision_distress_model.joblib")
        if os.path.exists(vis_ckpt):
            self.vision_model.load(vis_ckpt)
        else:
            print(f"[WARN] Vision model not found at: {vis_ckpt} - run training/train_vision.py first.")

        self.imu_model = IMUShockClassifier()
        imu_ckpt = os.path.join(self.ckpt_dir, "imu_shock_model.joblib")
        if os.path.exists(imu_ckpt):
            self.imu_model.load(imu_ckpt)
        else:
            print(f"[WARN] IMU model not found at: {imu_ckpt} - run training/train_imu.py first.")

        self.ipm_engine = IPMHomographyEngine(camera_height_m=1.45, pitch_deg=18.4)
        self.bayesian_gate = BayesianFusionGate(prior_pothole_prob=0.05, decision_threshold_log_odds=1.8)
        self.pci_model = PavementConditionIndexEngine()
        self.degrade_model = PavementDeteriorationForecaster()
        self.dispatch_agent = MoRTHDispatchAgent()
        self.multimodal_net = MultimodalTransformerFusionNet(imu_weight=0.4)
        self.rl_agent = AutomotiveADASPolicyAgent()
        self.telematics = AutomotiveTelematicsEngine(checkpoints_dir=self.ckpt_dir)

    # ------------------------------------------------------------------
    # Single-image audit
    # ------------------------------------------------------------------
    def audit_image(
        self,
        image_input,
        corridor_id="NH-44",
        latitude=28.7041,
        longitude=77.1025,
        chainage_km=108.4,
        imu_series=None,
        traffic_esal=7500,
        rain_mm=650.0,
        pavement_age_yr=3.5,
    ):
        """
        Runs the full pipeline on one image. image_input: file path, raw
        bytes, base64 string, PIL Image, or NumPy RGB array. imu_series, if
        given, must be a real (100, 3) accelerometer window - see the IMU
        stage below for what happens when it's omitted.
        """
        t0 = time.time()

        # STAGE 1: decode + standardize
        img_np = self.cv_detector.decode_image(image_input)
        H, W, _ = img_np.shape
        gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]

        # STAGE 2: texture gatekeeper
        roi_start_y = int(H * 0.35)
        road_gray = gray[roi_start_y:, :]
        mean_intensity = float(np.mean(road_gray))
        std_intensity = float(np.std(road_gray))

        if std_intensity < 6.5:
            return self._reject_non_pavement(mean_intensity, std_intensity, corridor_id, latitude, longitude, chainage_km, t0)

        # STAGE 3: region proposals (real classical CV, no trained model)
        pedestrians = self.cv_detector.detect_pedestrians(img_np)
        bboxes = self.cv_detector.extract_salient_regions(img_np, excluded_boxes=pedestrians)

        if not bboxes and not pedestrians:
            return self._normal_road(mean_intensity, std_intensity, corridor_id, latitude, longitude, chainage_km, t0)

        # STAGE 4: pedestrian entries (real HOG+SVM detections, no classifier involved)
        detections = [self._build_pedestrian_entry(p) for p in pedestrians]

        # STAGE 4/5: VisionDistressNet classification + IPM geometry per candidate region
        for bbox in bboxes:
            entry = self._classify_region(img_np, gray, bbox, W, H, mean_intensity)
            if entry is not None:
                detections.append(entry)

        if not detections:
            detections.append(self._normal_road_entry())

        ped_detections = [d for d in detections if d.get("is_pedestrian")]
        distress_detections = [d for d in detections if d.get("is_distress")]

        primary_pedestrian = ped_detections[0] if ped_detections else None
        if distress_detections:
            primary_distress = sorted(
                distress_detections,
                key=lambda d: (d.get("class_id") == 2, d.get("surface_area_m2", 0.0), d.get("confidence", 0.0)),
                reverse=True,
            )[0]
        else:
            primary_distress = detections[0]

        primary = primary_pedestrian if primary_pedestrian is not None else primary_distress
        has_dual_targets = bool(ped_detections) and bool(distress_detections)
        dual_target_summary = self._dual_target_summary(ped_detections, distress_detections, primary_pedestrian, primary_distress)

        # STAGE 6: IMU shock correlation - only real telemetry, never fabricated
        imu_report, delta_z, p_imu, imu_available = self._run_imu_stage(imu_series)

        # STAGE 7: Bayesian dual-sensor fusion
        p_vis = primary_distress.get("probabilities", {}).get("Pothole Cavity", 0.05)
        fusion_res = self.bayesian_gate.fuse(p_visual=p_vis, p_imu_shock=p_imu, delta_z_ms2=delta_z)
        fusion_res["imu_evidence_source"] = "real_sensor_window" if imu_available else "no_real_imu_data_neutral_prior"

        # STAGE 8: real ASTM D6433-style PCI from this frame's actual detections
        pci_result, rutting_mm, iri_roughness = self._compute_pci(detections, pavement_age_yr)

        # STAGE 9: deterioration forecast
        degrade_report = self.degrade_model.predict_lifecycle_roi(
            init_area_m2=max(0.5, primary_distress.get("surface_area_m2", 0.0)),
            depth_cm=max(3.0, primary_distress.get("depth_cm", 0.0)),
            esal_trucks=traffic_esal,
            rain_mm=rain_mm,
            age_yr=pavement_age_yr,
        )

        # MoRTH civil ledger (sum of real per-detection material costing)
        total_tonnage = round(float(sum(d.get("morth_tonnage_t", 0.0) for d in detections)), 3)
        total_repair_inr = round(float(sum(d.get("repair_cost_inr", 0.0) for d in detections)), 2)

        # STAGE 10/11: sealed work order for confirmed structural distress
        work_order = None
        if primary_distress.get("is_distress") and primary_distress.get("class_id") in AREA_CLASSES:
            work_order = self.dispatch_agent.generate_work_order(
                corridor_id=corridor_id,
                latitude=latitude,
                longitude=longitude,
                distress_class=primary_distress["class_name"],
                area_sqm=primary_distress.get("surface_area_m2", 0.0),
                depth_cm=primary_distress.get("depth_cm", 0.0),
                pci_score=int(pci_result["pci_score"]),
            )
            work_order["seal_verification_status"] = (
                "SEAL_VERIFIED_AUTHENTIC" if self.dispatch_agent.verify_work_order_seal(work_order) else "INVALID_SEAL"
            )

        elapsed_ms = round((time.time() - t0) * 1000.0, 2)

        return {
            "status": "ANALYSIS_COMPLETE",
            "gatekeeper_passed": True,
            "texture_metrics": {"road_roi_mean_lum": round(mean_intensity, 2), "road_roi_std_lum": round(std_intensity, 2)},
            "corridor_id": corridor_id,
            "location": {"lat": latitude, "lng": longitude, "chainage_km": chainage_km},
            "is_distress": len(distress_detections) > 0,
            "vulnerable_safety_alert": len(ped_detections) > 0,
            "pedestrians_count": len(ped_detections),
            "all_pedestrians": ped_detections,
            "has_dual_targets": has_dual_targets,
            "dual_target_summary": dual_target_summary,
            "primary_distress": primary_distress,
            "primary_pedestrian": primary_pedestrian,
            "primary_detection": primary,
            "all_detections": detections,
            "imu_shock_telemetry": imu_report,
            "bayesian_sensor_fusion": fusion_res,
            "astm_d6433_pci": {
                "pci_score": pci_result["pci_score"],
                "rating_category": pci_result["rating_category"],
                "description": pci_result["description"],
                "deduct_values": pci_result["deduct_values"],
                "rutting_mm": round(rutting_mm, 1),
                "iri_roughness": round(iri_roughness, 2),
                "note": "Per-frame proxy PCI (single dashcam frame), not a full-segment ASTM D6433 survey.",
            },
            "monsoon_deterioration_forecast": degrade_report,
            "morth_civil_ledger": {
                "asphalt_density_t_m3": 2.40,
                "compaction_factor": 1.15,
                "total_bitumen_tonnage_t": total_tonnage,
                "mix_rate_inr_per_tonne": 7500.0,
                "total_estimated_repair_inr": total_repair_inr,
            },
            "cryptographic_work_order": work_order,
            "deep_forensic_intelligence": {
                "shannon_entropy_bits": primary.get("shannon_entropy_bits", 0.0),
                "epistemic_uncertainty_rating": primary.get("uncertainty_rating", "LOW_UNCERTAINTY"),
                "astm_d6433_severity": primary_distress.get("astm_d6433_severity", "NONE"),
                "irc_standard_specification": primary.get("irc_standard_specification", ""),
                "top3_ranked_distress_hypotheses": primary.get("top3_ranked_predictions", []),
                "has_pedestrian_hazard": primary_pedestrian is not None,
                "pedestrians_detected_count": len(ped_detections),
                "pedestrian_alert_level": primary_pedestrian.get("alert_level") if primary_pedestrian else "NO_PEDESTRIAN_HAZARD",
            },
            "latency_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_pedestrian_entry(self, ped):
        """Wraps a real HOG+SVM pedestrian detection into the shared detection schema."""
        dist_m = ped["distance_meters"]
        if dist_m <= 8.0:
            alert_level, recommendation = "CRITICAL", "Emergency braking / evasive maneuver recommended."
        elif dist_m <= 18.0:
            alert_level, recommendation = "HIGH", "Reduce speed and monitor closely."
        else:
            alert_level, recommendation = "ADVISORY", "Pedestrian visible ahead; maintain awareness."

        return {
            "class_id": VisionDistressNet.PEDESTRIAN_CLASS_ID,
            "class_name": "Pedestrian / Vulnerable Road User",
            "confidence": ped["confidence"],
            "pedestrian_id": ped.get("pedestrian_id", 1),
            "detector": ped.get("detector", "opencv_hog_svm_person_detector"),
            "is_distress": False,
            "is_pedestrian": True,
            "alert_level": alert_level,
            "recommendation": recommendation,
            "bbox_pixels": ped["bbox_pixels"],
            "bbox_normalized": ped["bbox_normalized"],
            "distance_meters": dist_m,
            "surface_area_m2": 0.0,
            "depth_cm": 0.0,
            "morth_tonnage_t": 0.0,
            "repair_cost_inr": 0.0,
            "irc_standard_specification": VisionDistressNet.IRC_STANDARDS.get(VisionDistressNet.PEDESTRIAN_CLASS_ID, ""),
        }

    def _classify_region(self, img_np, gray, bbox, W, H, mean_intensity):
        """Crops one candidate region, runs the real vision classifier, and (for
        crack/pothole/waterlogging) prices the repair via real IPM geometry."""
        bx, by, bw, bh = bbox[:4]
        crop = img_np[by : by + bh, bx : bx + bw]
        if crop.size == 0:
            return None

        pred = self.vision_model.predict_image(crop)
        cls_id = pred["class_id"]
        if cls_id == 0:
            return None  # classifier says this region isn't actually a distress after all

        bx_norm, by_norm = round(bx / float(W), 4), round(by / float(H), 4)
        bw_norm, bh_norm = round(bw / float(W), 4), round(bh / float(H), 4)

        area_m2 = depth_cm = vol_m3 = tonnage_t = repair_cost_inr = 0.0
        dist_m = 0.0
        if cls_id in AREA_CLASSES:
            _, ground_y = self.ipm_engine.pixel_to_ground(bx + bw / 2.0, by + bh / 2.0)
            dist_m = max(1.8, min(30.0, float(ground_y)))
            area_m2 = self.ipm_engine.calculate_surface_area_sqm(bx, by, bw, bh)

            if cls_id == 2:  # Pothole Cavity - depth estimated from patch darkness relative to surroundings
                patch_gray = gray[by : by + bh, bx : bx + bw]
                patch_mean = float(np.mean(patch_gray)) if patch_gray.size > 0 else mean_intensity
                dark_contrast = max(0.0, (mean_intensity - patch_mean) / max(1.0, mean_intensity))
                # Coarse monocular estimate, not a measured depth - no stereo/LiDAR in this project.
                depth_cm = round(max(2.0, min(14.0, 6.0 * (dist_m / 8.0) * (dark_contrast + 0.5))), 1)
            elif cls_id == 1:  # Crack - shallow by definition; scale mildly with classifier confidence
                depth_cm = round(max(0.5, min(4.0, 1.5 + pred["confidence"] * 2.0)), 1)
            # Waterlogging (3): area is meaningful (hazard extent), depth is not asphalt depth.

            if cls_id in (1, 2):  # only crack/pothole get an asphalt repair costing
                materials = self.ipm_engine.estimate_repair_materials(area_m2, depth_cm=max(depth_cm, 1.0))
                vol_m3 = materials["volume_m3"]
                tonnage_t = materials["required_mass_tonnes"]
                repair_cost_inr = round(materials["total_cost_inr"], 2)

        return {
            "bbox_pixels": [bx, by, bw, bh],
            "bbox_normalized": [bx_norm, by_norm, bw_norm, bh_norm],
            "class_id": cls_id,
            "class_name": pred["class_name"],
            "confidence": pred["confidence"],
            "shannon_entropy_bits": pred["shannon_entropy_bits"],
            "uncertainty_rating": pred["uncertainty_rating"],
            "astm_d6433_severity": pred["astm_d6433_severity"],
            "irc_standard_specification": pred["irc_standard_specification"],
            "top3_ranked_predictions": pred["top3_ranked_predictions"],
            "is_distress": cls_id in AREA_CLASSES,
            "distance_meters": round(dist_m, 1),
            "surface_area_m2": round(area_m2, 2),
            "depth_cm": depth_cm,
            "volumetric_m3": vol_m3,
            "morth_tonnage_t": tonnage_t,
            "repair_cost_inr": repair_cost_inr,
            "physical_dimensions": {
                "surface_area_m2": round(area_m2, 2),
                "depth_cm": depth_cm,
                "bitumen_volume_m3": vol_m3,
                "morth_compacted_tonnage_t": tonnage_t,
                "estimated_repair_cost_inr": repair_cost_inr,
            },
            "probabilities": pred["all_class_probabilities"],
        }

    def _normal_road_entry(self):
        return {
            "class_id": 0,
            "class_name": "Normal Road / Sound Pavement",
            "confidence": 0.99,
            "shannon_entropy_bits": 0.0,
            "uncertainty_rating": "LOW_UNCERTAINTY",
            "astm_d6433_severity": "NONE",
            "irc_standard_specification": VisionDistressNet.IRC_STANDARDS.get(0, ""),
            "top3_ranked_predictions": [],
            "is_distress": False,
            "distance_meters": 0.0,
            "surface_area_m2": 0.0,
            "depth_cm": 0.0,
            "morth_tonnage_t": 0.0,
            "repair_cost_inr": 0.0,
            "probabilities": {"Normal Road / Sound Pavement": 0.99},
            "bbox_pixels": None,
            "bbox_normalized": None,
        }

    def _dual_target_summary(self, ped_detections, distress_detections, primary_pedestrian, primary_distress):
        if ped_detections and distress_detections:
            count_note = f"{len(ped_detections)} pedestrians" if len(ped_detections) > 1 else primary_pedestrian["class_name"]
            return (
                f"Co-occurring hazard: {count_note} (closest {primary_pedestrian['distance_meters']}m) "
                f"and {primary_distress['class_name']} ({primary_distress.get('surface_area_m2', 0.0)} m^2) in the same frame."
            )
        if len(ped_detections) > 1:
            return f"{len(ped_detections)} pedestrians detected (closest {primary_pedestrian['distance_meters']}m)."
        return ""

    def _run_imu_stage(self, imu_series):
        """
        Only ever scores a real accelerometer window. If none is supplied,
        this reports that honestly instead of deriving a fake one from the
        vision result - which is what the earlier version of this pipeline
        did, and which made "dual-sensor confirmation" meaningless (the two
        sensors were never actually independent).
        """
        if imu_series is None or not self.imu_model.is_ready:
            reason = "No real IMU telemetry provided for this frame." if imu_series is None else "IMU model not loaded."
            return (
                {"available": False, "reason": reason, "shock_classification": None, "peak_delta_z_ms2": 0.0, "pothole_shock_probability": 0.0},
                0.0,
                self.bayesian_gate.prior_p,  # neutral: fall back to the prior, not a fabricated confirmation
                False,
            )

        raw_imu = np.asarray(imu_series, dtype=np.float32)
        if raw_imu.ndim == 2:
            raw_imu = np.expand_dims(raw_imu, axis=0)
        delta_z = float(np.max(raw_imu[0, :, 2]) - np.min(raw_imu[0, :, 2]))

        preds, pothole_conf, _ = self.imu_model.predict(raw_imu)
        cls_name = IMUShockClassifier.CLASS_NAMES[int(preds[0])]
        p_imu = float(pothole_conf[0])
        return (
            {"available": True, "shock_classification": cls_name, "peak_delta_z_ms2": round(delta_z, 2), "pothole_shock_probability": round(p_imu, 4)},
            delta_z,
            p_imu,
            True,
        )

    def _compute_pci(self, detections, pavement_age_yr):
        crack_area = sum(d.get("surface_area_m2", 0.0) for d in detections if d.get("class_id") == 1)
        crack_count = sum(1 for d in detections if d.get("class_id") == 1)
        pothole_area = sum(d.get("surface_area_m2", 0.0) for d in detections if d.get("class_id") == 2)
        pothole_count = sum(1 for d in detections if d.get("class_id") == 2)

        crack_severities = [d["astm_d6433_severity"] for d in detections if d.get("class_id") == 1 and d.get("astm_d6433_severity") not in (None, "NONE")]
        pothole_severities = [d["astm_d6433_severity"] for d in detections if d.get("class_id") == 2 and d.get("astm_d6433_severity") not in (None, "NONE")]
        crack_severity = crack_severities[0] if crack_severities else "LOW"
        pothole_severity = pothole_severities[0] if pothole_severities else "MEDIUM"

        crack_density_pct = min(100.0, (crack_area / ASSUMED_FRAME_PAVEMENT_AREA_M2) * 100.0)
        pothole_density_pct = min(100.0, (pothole_area / ASSUMED_FRAME_PAVEMENT_AREA_M2) * 100.0)

        # Documented proxy, not a measured rut-bar/profilometer reading (see module docstring).
        rutting_mm = min(25.0, 3.0 + pothole_count * 3.5 + crack_area * 0.8)
        iri_roughness = min(9.0, 1.8 + pothole_count * 0.9)

        result = self.pci_model.compute(
            crack_density_pct=crack_density_pct,
            crack_severity=crack_severity,
            pothole_count=pothole_count,
            pothole_density_pct=pothole_density_pct,
            pothole_severity=pothole_severity,
            rutting_mm=rutting_mm,
            iri_roughness=iri_roughness,
            age_yr=pavement_age_yr,
        )
        return result, rutting_mm, iri_roughness

    def _reject_non_pavement(self, mean_intensity, std_intensity, corridor_id, latitude, longitude, chainage_km, t0):
        elapsed_ms = round((time.time() - t0) * 1000.0, 2)
        placeholder = {
            "class_name": "Non-Pavement Surface (Rejected)",
            "is_distress": False,
            "confidence": 0.99,
            "surface_area_m2": 0.0,
            "depth_cm": 0.0,
            "bbox_pixels": None,
            "bbox_normalized": None,
        }
        return {
            "status": "REJECTED_NON_PAVEMENT",
            "gatekeeper_passed": False,
            "texture_metrics": {"road_roi_mean_lum": round(mean_intensity, 2), "road_roi_std_lum": round(std_intensity, 2), "threshold_std": 6.5},
            "reason": f"Optical texture standard deviation ({std_intensity:.2f}) < 6.5 threshold. Rejected non-pavement surface.",
            "is_distress": False,
            "detections_count": 0,
            "primary_distress": placeholder,
            "primary_detection": placeholder,
            "all_detections": [],
            "corridor_id": corridor_id,
            "location": {"lat": latitude, "lng": longitude, "chainage_km": chainage_km},
            "latency_ms": elapsed_ms,
        }

    def _normal_road(self, mean_intensity, std_intensity, corridor_id, latitude, longitude, chainage_km, t0):
        elapsed_ms = round((time.time() - t0) * 1000.0, 2)
        entry = self._normal_road_entry()
        return {
            "status": "ROAD_INSPECTION_NORMAL",
            "gatekeeper_passed": True,
            "texture_metrics": {"road_roi_mean_lum": round(mean_intensity, 2), "road_roi_std_lum": round(std_intensity, 2)},
            "reason": "Road pavement verified; no anomalous cavity or distress contours detected.",
            "is_distress": False,
            "detections_count": 0,
            "primary_distress": entry,
            "primary_detection": entry,
            "all_detections": [entry],
            "pavement_pci": 96.0,
            "pci_category": "EXCELLENT",
            "corridor_id": corridor_id,
            "location": {"lat": latitude, "lng": longitude, "chainage_km": chainage_km},
            "latency_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def process_batch(self, image_source, max_samples=None, corridor_id="NH-44", **kwargs):
        """Processes a directory of images or a list of image paths, aggregating civil-engineering statistics."""
        t0 = time.time()

        if isinstance(image_source, str) and os.path.isdir(image_source):
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            image_paths = [
                os.path.join(image_source, fname)
                for fname in sorted(os.listdir(image_source))
                if os.path.splitext(fname)[1].lower() in valid_exts
            ]
        elif isinstance(image_source, (list, tuple)):
            image_paths = list(image_source)
        else:
            raise ValueError(f"Invalid image_source: {image_source}")

        if max_samples:
            image_paths = image_paths[:max_samples]

        records = []
        total_pavement_accepted = 0
        total_non_pavement_rejected = 0
        total_defects_count = 0
        total_tonnage = 0.0
        total_cost_inr = 0.0
        pci_scores = []
        latencies = []

        for idx, img_path in enumerate(image_paths):
            try:
                rec = self.audit_image(image_input=img_path, corridor_id=corridor_id, chainage_km=100.0 + idx * 0.25, **kwargs)
                rec["image_path"] = img_path
                rec["image_name"] = os.path.basename(img_path)
                records.append(rec)
                latencies.append(rec.get("latency_ms", 0.0))

                if rec["gatekeeper_passed"]:
                    total_pavement_accepted += 1
                    if rec["is_distress"]:
                        total_defects_count += len(rec.get("all_detections", []))
                        ledger = rec.get("morth_civil_ledger", {})
                        total_tonnage += ledger.get("total_bitumen_tonnage_t", 0.0)
                        total_cost_inr += ledger.get("total_estimated_repair_inr", 0.0)
                        pci_scores.append(rec["astm_d6433_pci"]["pci_score"])
                    else:
                        pci_scores.append(rec.get("pavement_pci", 95.0))
                else:
                    total_non_pavement_rejected += 1
            except Exception as e:
                records.append({"image_path": img_path, "status": f"ERROR: {str(e)}", "gatekeeper_passed": False})

        walltime_s = round(time.time() - t0, 3)
        avg_latency = round(float(np.mean(latencies)), 2) if latencies else 0.0
        mean_pci = round(float(np.mean(pci_scores)), 1) if pci_scores else 100.0

        return {
            "batch_summary": {
                "total_images_evaluated": len(image_paths),
                "pavements_accepted": total_pavement_accepted,
                "non_pavements_rejected": total_non_pavement_rejected,
                "total_defects_detected": total_defects_count,
                "total_bitumen_tonnage_tonnes": round(float(total_tonnage), 3),
                "total_repair_budget_inr": round(float(total_cost_inr), 2),
                "mean_pavement_pci": mean_pci,
                "mean_inference_latency_ms": avg_latency,
                "total_batch_walltime_s": walltime_s,
            },
            "records": records,
        }

    def verify_pipeline_integrity(self):
        """Checks that the trained model artifacts this pipeline depends on actually exist on disk."""
        expected = [
            ("Model M1 VisionDistressNet", "vision_distress_model.joblib"),
            ("Model M4 IMUShockClassifier", "imu_shock_model.joblib"),
        ]
        status = {}
        all_ok = True
        for name, fname in expected:
            fpath = os.path.join(self.ckpt_dir, fname)
            if os.path.exists(fpath):
                size_kb = round(os.path.getsize(fpath) / 1024.0, 1)
                with open(fpath, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                status[name] = {"status": "VERIFIED_OK", "size_kb": size_kb, "sha256": digest[:16] + "..."}
            else:
                status[name] = {"status": "MISSING"}
                all_ok = False
        status["Model PCI / Deterioration engines"] = {
            "status": "N/A_FORMULA_BASED",
            "note": "Deterministic ASTM D6433 / growth-model engines have no trained weights to verify.",
        }
        return {"all_models_verified": all_ok, "models": status}

    # ------------------------------------------------------------------
    # Automotive incident evaluation (ADAS policy + CAN + fusion)
    # ------------------------------------------------------------------
    def evaluate_automotive_incident(
        self,
        hazard_class_id=0,
        confidence=0.95,
        distance_m=45.0,
        vehicle_speed_kmh=65.0,
        surface_friction_mu=0.75,
        pothole_depth_mm=0.0,
        imu_z_shock_ms2=0.2,
        lateral_lane_margin_m=1.2,
        is_wet=False,
    ):
        """
        1. AutomotiveADASPolicyAgent picks an ADAS/active-suspension action
           (deterministic rule table over real physics - see that module).
        2. AutomotiveTelematicsEngine encodes the decision as a real CAN frame.
        3. MultimodalLateFusionNet combines the caller-supplied hazard
           classification with the caller-supplied IMU shock reading.
        """
        rl_res = self.rl_agent.evaluate_telemetry_state(
            hazard_class_id=hazard_class_id,
            confidence=confidence,
            distance_m=distance_m,
            vehicle_speed_kmh=vehicle_speed_kmh,
            surface_friction_mu=surface_friction_mu,
            pothole_depth_mm=pothole_depth_mm,
            imu_z_shock_ms2=imu_z_shock_ms2,
            lateral_lane_margin_m=lateral_lane_margin_m,
            is_wet=is_wet,
        )

        can_frame = self.telematics.generate_adas_can_packet(
            rl_decision=rl_res,
            hazard_class_id=hazard_class_id,
            ttc_sec=rl_res["telemetry_metrics"]["time_to_collision_sec"],
            speed_kmh=vehicle_speed_kmh,
        )

        # Build a real probability vector from the caller-supplied hazard
        # class + confidence (not a random tensor standing in for imaginary
        # LiDAR/CAN-bus sensors - see multimodal_transformer_fusion.py).
        n_classes = len(VisionDistressNet.CLASS_NAMES) + 1
        vision_probs = np.full(n_classes, (1.0 - confidence) / max(1, n_classes - 1), dtype=np.float64)
        vision_probs[min(hazard_class_id, n_classes - 1)] = confidence
        vision_probs = vision_probs / vision_probs.sum()

        # A single peak-shock scalar (not a full 100-sample window) can only
        # support a coarse heuristic, not the trained IMUShockClassifier -
        # documented here rather than silently treated as equivalent to it.
        imu_pothole_prob = float(np.clip(imu_z_shock_ms2 / 8.0, 0.0, 1.0))

        mm_res = self.multimodal_net.fuse(vision_probs, imu_pothole_prob=imu_pothole_prob, imu_shock_ms2=imu_z_shock_ms2)

        return {
            "rl_policy_decision": rl_res,
            "can_bus_telemetry": can_frame,
            "multimodal_fusion_status": mm_res,
            "automotive_standards_referenced": [
                "ISO 26262 ASIL-D functional-safety decision structure (rule table, not certified)",
                "SAE J1939 / ISO 11898-1 CAN 2.0B frame format",
            ],
        }
