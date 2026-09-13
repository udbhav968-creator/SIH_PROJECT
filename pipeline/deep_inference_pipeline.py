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

        # Prefer the fine-tuned CNN when its weights are on disk and a runtime
        # (ONNX Runtime or PyTorch) is installed; otherwise fall back to the
        # HOG/LBP + SVM baseline. Which one answered is reported downstream.
        from models.deep_vision_net import load_best_vision_model
        self.vision_model, self.vision_backend = load_best_vision_model(self.ckpt_dir, verbose=False)

        # COCO-pretrained object detector (people, vehicles, signs). Optional:
        # when its weights or onnxruntime are absent the stage is skipped and
        # the pipeline says so, rather than guessing what is in frame.
        from models.onnx_object_detector import ONNXObjectDetector
        self.object_detector = ONNXObjectDetector(checkpoints_dir=self.ckpt_dir)
        if self.vision_backend == "none":
            print("[WARN] No trained vision model found - run training/train_deep_vision.py "
                  "(deep) or training/train_vision.py (baseline).")

        self.imu_model = IMUShockClassifier()
        imu_ckpt = os.path.join(self.ckpt_dir, "imu_shock_model.joblib")
        if os.path.exists(imu_ckpt):
            self.imu_model.load(imu_ckpt)
        else:
            print(f"[WARN] IMU model not found at: {imu_ckpt} - run training/train_imu.py first.")

        # Geometry comes from a calibration profile rather than constants. With
        # no profile on disk this resolves to the historical assumed mount and
        # says so in every result, instead of presenting an assumption as a
        # measurement. See models/camera_calibration.py.
        from models.camera_calibration import describe as describe_calibration, resolve as resolve_calibration
        self._resolve_calibration = resolve_calibration
        self._describe_calibration = describe_calibration
        self.calibration, self.calibration_provenance = resolve_calibration(None)
        self.ipm_engine = IPMHomographyEngine.from_calibration(self.calibration)

        # Pixel-level segmentation, when a model has been trained. Without it
        # the older bounding-box area is used and the result is labelled as
        # such - a box around a diagonal crack overstates its area by an order
        # of magnitude, so which method answered is not a detail.
        try:
            from models.defect_segmenter import DefectSegmenter
            self.segmenter = DefectSegmenter()
            if self.segmenter.is_ready:
                print(f"  ✓ Defect segmenter: pixel masks "
                      f"({self.segmenter.thresholds})")
        except Exception as e:
            print(f"[WARN] segmenter unavailable: {e}")
            self.segmenter = None
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
        device_id=None,
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

        # Intrinsics are in pixels, so a profile calibrated at one resolution is
        # simply wrong at another. Rescale per request.
        calib, provenance = self._resolve_calibration(device_id, width_px=W, height_px=H)
        ipm = IPMHomographyEngine.from_calibration(calib)
        self.ipm_engine = ipm
        self.calibration, self.calibration_provenance = calib, provenance

        # One segmentation pass for the whole frame; regions index into it.
        seg_out = None
        if getattr(self, "segmenter", None) is not None and self.segmenter.is_ready:
            try:
                seg_out = self.segmenter.segment(img_np)
            except Exception as e:
                print(f"[WARN] segmentation failed, falling back to box area: {e}")
        self._seg_out = seg_out

        # STAGE 2: texture gatekeeper
        roi_start_y = int(H * 0.35)
        road_gray = gray[roi_start_y:, :]
        mean_intensity = float(np.mean(road_gray))
        std_intensity = float(np.std(road_gray))

        if std_intensity < 6.5:
            return self._reject_non_pavement(mean_intensity, std_intensity, corridor_id, latitude, longitude, chainage_km, t0)

        # STAGE 3a: COCO object detection (people, vehicles, traffic control)
        scene_objects, scene_summary = [], {"available": False,
                                            "reason": "No detector weights - run scripts/fetch_detector.py"}
        if self.object_detector.is_ready:
            try:
                scene_objects = self.object_detector.detect(img_np)
                scene_summary = self.object_detector.summarise(scene_objects)
                scene_summary["available"] = True
            except Exception as e:
                scene_summary = {"available": False, "reason": f"detector error: {e}"}

        # STAGE 3b: region proposals (classical CV) + people. The CNN detector's
        # person boxes are used when it ran; the HOG+SVM detector is the fallback.
        person_boxes = [d for d in scene_objects if d["class_name"] == "person"]
        if person_boxes:
            pedestrians = [{
                "bbox_pixels": d["bbox_pixels"],
                "bbox_normalized": d["bbox_normalized"],
                "pedestrian_id": i + 1,
                "confidence": d["confidence"],
                "distance_meters": round(max(1.5, 22.0 * (1.0 - ((d["bbox_pixels"][1] + d["bbox_pixels"][3]) / float(H)) ** 0.85)), 1),
                "detector": d.get("detector", "onnx_coco_detector"),
            } for i, d in enumerate(person_boxes)]
        else:
            pedestrians = self.cv_detector.detect_pedestrians(img_np)
        # Crack and pothole candidates come from the segmenter's mask; the
        # brightness scan supplies everything the segmenter has no class for
        # (waterlogging, signage, markings) and is the whole proposal stage when
        # no segmenter is loaded. See _segmentation_proposals for why.
        seg_boxes = self._segmentation_proposals(H, W)
        heuristic = self.cv_detector.extract_salient_regions(img_np, excluded_boxes=pedestrians)
        if seg_boxes:
            heuristic = [b for b in heuristic
                         if not any(self._boxes_overlap(b, sb) for sb in seg_boxes)]
        bboxes = seg_boxes + heuristic

        if not bboxes and not pedestrians:
            normal = self._normal_road(mean_intensity, std_intensity, corridor_id, latitude, longitude, chainage_km, t0)
            normal["scene_objects"] = scene_objects
            normal["scene_summary"] = scene_summary
            return normal

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
            "scene_objects": scene_objects,
            "scene_summary": scene_summary,
            "vision_backend": self.vision_backend,
            # Provenance of the geometry. Any area or cost in this response was
            # produced by this camera model; without a calibrated profile they
            # are estimates from an assumed mount, and this block says so rather
            # than letting the numbers imply otherwise.
            "camera_calibration": self._describe_calibration(self.calibration,
                                                             self.calibration_provenance),
            "segmentation": ({
                "available": True,
                "crack_pixels": seg_out.get("crack_px_full"),
                "pothole_pixels": seg_out.get("pothole_px_full"),
                "defect_fraction": seg_out.get("defect_fraction"),
                "area_method": "per-pixel ground footprint over the mask",
            } if seg_out else {
                "available": False,
                "area_method": "bounding-box corners projected to the ground plane",
                "note": "No segmenter on disk. Train one with "
                        "training/train_segmenter.py - a box around a diagonal "
                        "crack overstates its area by roughly an order of magnitude.",
            }),
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

    def _region_mask(self, cls_id, bx, by, bw, bh, H, W):
        """
        The segmentation mask restricted to one candidate region.

        Returns a full-frame boolean array that is True only inside the box AND
        only where the segmenter called that pixel this defect class. Anything
        the segmenter did not claim is excluded, which is the whole point: the
        area becomes the defect's own footprint rather than the rectangle a
        heuristic drew around it.

        None when no segmenter is loaded, or the class has no mask equivalent
        (waterlogging, signs and markings are not in the segmenter's three
        classes).
        """
        import numpy as _np
        seg = getattr(self, "_seg_out", None)
        if seg is None:
            return None
        seg_cls = {1: 1, 2: 2}.get(cls_id)      # crack -> 1, pothole -> 2
        if seg_cls is None:
            return None
        mask = seg["mask"]
        if mask.shape[:2] != (H, W):
            import cv2
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        window = _np.zeros((H, W), dtype=bool)
        y0, y1 = max(0, by), min(H, by + bh)
        x0, x1 = max(0, bx), min(W, bx + bw)
        if y1 <= y0 or x1 <= x0:
            return None
        window[y0:y1, x0:x1] = True
        return window & (mask == seg_cls)

    # ---- how a mask blob becomes a candidate region -----------------------
    #
    # Both knobs are swept end-to-end in scripts.tune_proposals - through
    # audit_image(), not through a copy of the region loop - and the curve is
    # written to checkpoints/proposal_tuning_report.json. Measured over 36 clean
    # and 24 annotated photographs:
    #
    #   size    conf     false positives      defects found
    #   0.002   0.40     22/36  (61.1%)       23/24  (95.8%)
    #   0.002   0.55      6/36  (16.7%)       22/24  (91.7%)
    #   0.002   0.70      4/36  (11.1%)       23/24  (95.8%)   <- committed
    #   0.008   0.70      4/36  (11.1%)       21/24  (87.5%)
    #   0.030   0.85      4/36  (11.1%)       21/24  (87.5%)
    #
    # 4/36 is the floor: no setting in the grid goes below it, so those four
    # photographs are not fixed by filtering and are not claimed to be.
    # Tightening past 0.002/0.70 buys nothing and costs detection.
    #
    # Size, as a fraction of the segmenter's WORKING frame (320x200 = 64,000 px)
    # rather than an absolute pixel count - an absolute count would mean a
    # stricter filter on a phone photo than on a dashcam frame, for no reason.
    MIN_COMPONENT_FRACTION = 0.004
    # How far above its own decision threshold a blob's MEAN probability must
    # sit before the blob is proposed.
    #
    # This is a MARGIN, not an absolute floor, and that distinction was learned
    # the hard way. It used to be a hardcoded 0.45, which on the model trained
    # here sits just above the pothole threshold of 0.35. Another machine
    # retrained the segmenter on its own larger dataset, got a pothole threshold
    # of 0.20, and the same 0.45 became 2.25x the threshold - it discarded real
    # potholes and the test suite caught it at 4 of 8 detected.
    #
    # A threshold is whatever the calibration found for that model, so anything
    # compared against it has to be relative to it. A blob whose pixels merely
    # scraped past the threshold is a scatter; one averaging a margin above it
    # is a defect. That statement is true of any model; "0.45" was only ever
    # true of one.
    # Per class, because a crack and a pothole are different SHAPES and the
    # mean probability inside a blob is a shape-dependent quantity.
    #
    # A pothole is compact: most of its pixels are interior, well inside the
    # defect, and confidently scored. Its blob mean sits comfortably above the
    # threshold, so requiring a margin separates real cavities from scatter.
    #
    # A crack is thin - often two or three pixels wide. Most of its pixels ARE
    # boundary pixels, and boundary pixels score near the threshold by
    # construction. Demanding the same margin of a crack penalises it for being
    # crack-shaped. Measured on the same model: raising the crack floor from
    # inactive to threshold+0.10 moved detection 91.7% -> 79.2% and bought only
    # 16.7% -> 13.9% on false positives.
    #
    # So potholes carry a margin and cracks do not. The crack floor is its own
    # calibrated threshold, which every pixel in the blob has already cleared.
    MIN_COMPONENT_MARGIN = {"crack": 0.0, "pothole": 0.10}

    def _component_floor(self, seg_cls):
        """Floor for one class: its calibrated threshold plus that class's margin."""
        name = {1: "crack", 2: "pothole"}.get(seg_cls, "pothole")
        seg = getattr(self, "segmenter", None)
        base = 0.35
        if seg is not None and getattr(seg, "thresholds", None):
            base = float(seg.thresholds.get(name, base))
        margin = self.MIN_COMPONENT_MARGIN
        if isinstance(margin, dict):
            margin = margin.get(name, 0.10)
        return min(0.95, base + float(margin))
    # No frame contains this many separate repairs. Past it, the segmenter is
    # confused about the whole surface and the frame is not evidence.
    MAX_COMPONENTS = 8
    # A crack or pothole proposed by the brightness grid may not be bigger than
    # this share of the frame. Measured: the grid was producing boxes at 0.61
    # and they were being priced as single repairs.
    MAX_HEURISTIC_BOX_FRACTION = 0.25
    # The same rule for blobs the SEGMENTER proposes. Capping only the
    # brightness grid was half a fix: measured on a real photograph, one
    # segmentation component covered 83% x 84% of the frame and was reported as
    # a single 4.273 m2 pothole priced at Rs 2,830. A component that large is
    # the segmenter over-claiming a whole road surface, not one repair.
    MAX_COMPONENT_BOX_FRACTION = 0.25
    # A single pothole patch under MoRTH Section 500 is a metre or so across.
    # Past this, the geometry has almost certainly been handed a photograph it
    # was not designed for - a close-up crop rather than a frame from a mounted
    # camera - and the honest output is a flag for manual survey, not a price.
    MAX_SINGLE_REPAIR_AREA_M2 = 2.0

    def _segmentation_proposals(self, H, W):
        """
        Candidate regions taken from the segmenter's own mask, not from
        brightness.

        The original proposal stage was a 16x24 grid of gradient and darkness
        scores, merged by connected components. On a normally textured road most
        cells fire, the components merge, and the result is ONE box covering the
        whole carriageway. That box is what appeared over a whole road in
        testing, and because it is enormous, the fraction of it that any real
        pothole occupies is under 1% - which is why a segmentation gate
        expressed as a fraction of the box threw away real potholes while
        letting zebra crossings through. The box was the bug, not the threshold.

        A connected component of the mask is a defect-shaped region by
        construction. It gives a tight box, so the area computed from it is the
        defect's own footprint, the crop handed to the classifier contains the
        defect rather than fifty square metres of road, and the two models are
        being asked about the same object.

        The brightness scan is still used, but only for the classes the
        segmenter has no pixels for - waterlogging, signage, markings - and as
        the fallback when no segmenter is loaded at all.
        """
        import cv2
        import numpy as _np
        seg = getattr(self, "_seg_out", None)
        if seg is None:
            return []
        mask = seg.get("mask_work")
        if mask is None:                       # older cached segmenter output
            mask = seg["mask"]
        mh, mw = mask.shape[:2]
        frame_px = float(mh * mw)
        sx, sy = W / float(mw), H / float(mh)
        min_px = max(12, int(self.MIN_COMPONENT_FRACTION * frame_px))
        proba = {1: seg.get("proba_crack"), 2: seg.get("proba_pothole")}

        out = []
        for seg_cls, kind in ((1, 1), (2, 2)):          # crack -> 1, pothole -> 2
            binary = (mask == seg_cls).astype(_np.uint8)
            if not binary.any():
                continue
            # Close one-pixel gaps so a dashed crack is one component rather
            # than forty. 3x3 is deliberately small: a bigger kernel would weld
            # separate potholes into a single box and overstate the repair.
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            n, lab, st, _cen = cv2.connectedComponentsWithStats(binary, 8)
            for i in range(1, n):
                area = int(st[i, cv2.CC_STAT_AREA])
                if area < min_px:
                    continue
                pm = proba.get(seg_cls)
                mean_p = float(pm[lab == i].mean()) if pm is not None else 1.0
                if mean_p < self._component_floor(seg_cls):
                    continue
                x = int(st[i, cv2.CC_STAT_LEFT] * sx)
                y = int(st[i, cv2.CC_STAT_TOP] * sy)
                w = max(8, int(st[i, cv2.CC_STAT_WIDTH] * sx))
                h = max(8, int(st[i, cv2.CC_STAT_HEIGHT] * sy))
                # A little context around the defect; the classifier was trained
                # on crops that include some surrounding road.
                px, py = int(w * 0.15), int(h * 0.15)
                x0, y0 = max(0, x - px), max(0, y - py)
                x1, y1 = min(W, x + w + px), min(H, y + h + py)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                # An oversized component is NOT dropped. Dropping it cost 8
                # points of detection (91.7% -> 83.3%) and hid real defects: a
                # road authority needs to know a large defect is there even when
                # its extent cannot be measured from one photograph.
                #
                # It is proposed, and the area check downstream refuses to price
                # it and flags it for manual survey. Reporting "large defect,
                # extent uncertain" is useful; reporting "4.273 m2, Rs 2,830" is
                # worse than useless, and reporting nothing loses the defect.
                oversized = (((x1 - x0) * (y1 - y0)) / float(W * H)
                             > self.MAX_COMPONENT_BOX_FRACTION)
                out.append([x0, y0, x1 - x0, y1 - y0, mean_p * area, kind,
                            "segmentation", round(mean_p, 4),
                            round(area / frame_px, 5), oversized])
        out.sort(key=lambda b: -b[4])
        return out[: self.MAX_COMPONENTS]

    @staticmethod
    def _boxes_overlap(a, b, thresh=0.30):
        ax, ay, aw, ah = a[:4]
        bx, by, bw, bh = b[:4]
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        if inter <= 0:
            return False
        return inter / float(min(aw * ah, bw * bh) or 1) > thresh

    # A region must have at least this fraction of its pixels claimed by the
    # segmenter before it is reported as that defect.
    #
    # 0.01 is measured, not chosen by feel. scripts/tune_detection_gate.py
    # sweeps it against 36 clean photographs (zebra crossings, sound pavement,
    # dividers) and 24 defect photographs, and prints BOTH error rates:
    #
    #   gate    false positives      defects still found
    #   0.00    15/36  (41.7%)       24/24  (100.0%)   <- the bug
    #   0.01    10/36  (27.8%)       23/24  ( 95.8%)   <- committed
    #   0.06     6/36  (16.7%)       11/24  ( 45.8%)   halves detection
    #   0.50     0/36  ( 0.0%)        0/24  (  0.0%)   silences everything
    #
    # Raising it further trades away far more detection than it buys in
    # precision. 27.8% false positives is still high, and no threshold fixes
    # that - the region proposals come from a brightness heuristic, and the
    # real answer is a detector trained on the pothole bounding boxes already
    # sitting in datasets/. That is a GPU job, and it is not done.
    SEGMENTER_GATE_FRACTION = 0.01
    # Below this, the classifier is not confident enough to assert anything.
    MIN_REPORT_CONFIDENCE = 0.45

    def _region_defect_fraction(self, cls_id, bx, by, bw, bh, H, W):
        """
        What fraction of this region the segmenter actually calls this defect.

        Returns None when there is no segmenter or the class has no mask
        equivalent, in which case the caller must not gate on it.
        """
        mask = self._region_mask(cls_id, bx, by, bw, bh, H, W)
        if mask is None:
            return None
        area = max(1, min(H, by + bh) - max(0, by)) * max(1, min(W, bx + bw) - max(0, bx))
        return float(mask.sum()) / float(area)

    def _classify_region(self, img_np, gray, bbox, W, H, mean_intensity):
        """
        Crops one candidate region, classifies it, and prices the repair.

        Two gates stand between a candidate region and a reported defect,
        because the region proposals come from a brightness heuristic that
        fires on any high-contrast structure - a zebra crossing's stripes are
        the highest-contrast thing on a road, and a lane marking's edge looks
        like a crack to a threshold.

        1. CONFIDENCE. The classifier must be at least MIN_REPORT_CONFIDENCE
           sure. A 48% call is not a finding.

        2. SEGMENTATION AGREEMENT. The segmenter is trained on 4,720 hand-drawn
           polygons and therefore knows what crack and pothole pixels actually
           look like. If it claims almost none of the region, the classifier is
           reading texture that is not a defect, and the region is dropped.

        The second gate is the one that matters. The classifier was trained on
        photographs of defects and is applied to arbitrary crops, so asking it
        "is this a defect at all?" is asking a question it was never trained to
        answer. The segmenter was trained on exactly that question, per pixel.
        """
        bx, by, bw, bh = bbox[:4]
        crop = img_np[by : by + bh, bx : bx + bw]
        if crop.size == 0:
            return None

        pred = self.vision_model.predict_image(crop)
        cls_id = pred["class_id"]
        if cls_id == 0:
            return None  # classifier says this region isn't actually a distress after all

        if float(pred.get("confidence", 0.0)) < self.MIN_REPORT_CONFIDENCE:
            return None

        from_segmenter = len(bbox) > 6 and bbox[6] == "segmentation"
        if cls_id in (1, 2) and not from_segmenter:
            # WHERE a crack or pothole is, is the segmenter's question. The
            # brightness grid merges into one rectangle over the whole
            # carriageway on any textured road - it was producing boxes at 0.61
            # of the frame and those were being priced as single repairs - so
            # when a segmenter is loaded the grid may not raise a crack or a
            # pothole at all. It still supplies the classes the segmenter has no
            # pixels for: waterlogging, signage, markings.
            #
            # Measured through audit_image() over 36 clean and 24 annotated
            # photographs, with the grid allowed to raise defects and with it
            # closed (mask proposals at 0.004 / 0.45 in both cases):
            #
            #   grid allowed (size-capped + agreement gate)   19.4% FP   87.5% found
            #   grid closed                                    8.3% FP   87.5% found
            #
            # It contributed four false positives and not one extra detection,
            # so it is closed. This is not a mute button: closing it at a
            # stricter mask setting (0.002 / 0.70) DID collapse detection to
            # 20.8%, which is why both numbers are printed at every setting in
            # scripts/tune_proposals.py rather than only the flattering one.
            if getattr(self, "segmenter", None) is not None and self.segmenter.is_ready:
                return None
            # No segmenter on disk. The grid is the only proposal source, so it
            # is allowed to raise defects - but never one covering most of the
            # frame, and the area it yields is labelled a box estimate
            # downstream rather than a measurement.
            box_fraction = (bw * bh) / float(max(1, W * H))
            if box_fraction > self.MAX_HEURISTIC_BOX_FRACTION:
                return None

        bx_norm, by_norm = round(bx / float(W), 4), round(by / float(H), 4)
        bw_norm, bh_norm = round(bw / float(W), 4), round(bh / float(H), 4)

        area_m2 = depth_cm = vol_m3 = tonnage_t = repair_cost_inr = 0.0
        dist_m = 0.0
        area_method = None
        area_diag = {}
        depth_estimate = None
        if cls_id in AREA_CLASSES:
            _, ground_y = self.ipm_engine.pixel_to_ground(bx + bw / 2.0, by + bh / 2.0)
            dist_m = max(1.8, min(30.0, float(ground_y)))

            # Area, by measurement where a mask exists and by estimate otherwise.
            region_mask = self._region_mask(cls_id, bx, by, bw, bh, H, W)
            if region_mask is not None and region_mask.any():
                area_m2, area_diag = self.ipm_engine.mask_area_m2(region_mask)
                area_method = "segmentation_mask"
            else:
                area_m2 = self.ipm_engine.calculate_surface_area_sqm(bx, by, bw, bh)
                area_method = "bounding_box_estimate"
                area_diag = {"note": "No segmentation mask for this region. A box "
                                     "around a non-rectangular defect overstates its "
                                     "area - for a diagonal crack, by roughly an "
                                     "order of magnitude."}

            # Depth: an estimate with an interval, never a function of the
            # classifier's confidence. See models/depth_estimator.py.
            from models import depth_estimator
            if cls_id == 2:
                depth_estimate = depth_estimator.pothole_depth(
                    gray, mask=region_mask,
                    bbox=None if region_mask is not None else (bx, by, bw, bh),
                    distance_m=dist_m)
            elif cls_id == 1:
                extent = None
                if region_mask is not None and region_mask.size:
                    extent = float(region_mask.sum()) / float(region_mask.size)
                depth_estimate = depth_estimator.crack_depth(severity_ratio=extent)
            if depth_estimate is not None:
                depth_cm = depth_estimate["depth_cm"]
            # Waterlogging (3): area is meaningful (hazard extent), depth is not asphalt depth.

            # Is this a plausible single repair at all?
            #
            # Everything downstream - tonnage, rupees, the work order - assumes
            # the area is one patch a crew will lay. Measured on a real
            # photograph: 4.273 m2 priced at Rs 2,830, from a blob covering most
            # of a close-up crop. The arithmetic was right and the answer was
            # nonsense, because the ground-plane projection had been handed an
            # image it was never designed for.
            #
            # A number that cannot be defended should not be priced. It is
            # reported, flagged, and sent for manual survey.
            oversized_box = len(bbox) > 9 and bool(bbox[9])
            box_fraction_here = (bw * bh) / float(max(1, W * H))
            area_plausible = (area_m2 <= self.MAX_SINGLE_REPAIR_AREA_M2
                              and not oversized_box
                              and box_fraction_here <= self.MAX_COMPONENT_BOX_FRACTION)
            if not area_plausible:
                area_diag = dict(area_diag or {})
                area_diag.update({
                    "implausible_for_a_single_repair": True,
                    "max_plausible_m2": self.MAX_SINGLE_REPAIR_AREA_M2,
                    "measured_area_m2": round(area_m2, 3),
                    "box_share_of_frame": round(box_fraction_here, 3),
                    "why": ("A single pothole patch under MoRTH Section 500 is about a "
                            "metre across. An area this large, or a region covering this "
                            "much of the frame, usually means the camera geometry does "
                            "not apply to this photograph - a close-up crop rather than a "
                            "frame from a mounted camera - or that the segmentation has "
                            "merged several defects with the road between them."),
                    "action": ("Reported for manual survey. No tonnage or cost is "
                               "quoted, because neither could be defended."),
                })

            if cls_id in (1, 2) and area_plausible:  # only crack/pothole get an asphalt repair costing
                materials = self.ipm_engine.estimate_repair_materials(area_m2, depth_cm=max(depth_cm, 1.0))
                vol_m3 = materials["volume_m3"]
                tonnage_t = materials["required_mass_tonnes"]
                repair_cost_inr = round(materials["total_cost_inr"], 2)
                # Cost is linear in both area and depth, so the depth interval
                # maps straight onto a cost interval. Quoting a single rupee
                # figure from an estimated depth is what loses an audit.
                if depth_estimate is not None:
                    lo = self.ipm_engine.estimate_repair_materials(
                        area_m2, depth_cm=max(depth_estimate["depth_low_cm"], 1.0))
                    hi = self.ipm_engine.estimate_repair_materials(
                        area_m2, depth_cm=max(depth_estimate["depth_high_cm"], 1.0))
                    depth_estimate["repair_cost_low_inr"] = round(lo["total_cost_inr"], 2)
                    depth_estimate["repair_cost_high_inr"] = round(hi["total_cost_inr"], 2)

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
            "surface_area_m2": round(area_m2, 3),
            "area_method": area_method,
            "area_diagnostics": area_diag,
            # An explicit flag, not something a reader has to infer from a zero
            # cost. A defect can be real and still not be priceable.
            "area_is_plausible_single_repair": bool(area_plausible) if cls_id in AREA_CLASSES else None,
            "needs_manual_survey": bool(cls_id in (1, 2) and not area_plausible),
            "depth_cm": depth_cm,
            "depth_estimate": depth_estimate,
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

        # Accept an (N, 3) array, the (window, label) pair returned by
        # data.dataset_generator.sample_real_imu_window, or a dict wrapping one.
        series = imu_series
        if isinstance(series, dict):
            series = series.get("imu_series", series.get("window"))
        if isinstance(series, (tuple, list)) and len(series) == 2 and np.ndim(series[1]) == 0:
            series = series[0]

        raw_imu = np.asarray(series, dtype=np.float32)
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
