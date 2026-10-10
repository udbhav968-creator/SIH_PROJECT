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
import threading
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
from models.urban_traffic_net import UrbanTrafficNet, PCU_WEIGHTS, IRC_STANDARDS as TRAFFIC_IRC_STANDARDS

# Classes that represent an actual surface footprint (crack, pothole,
# waterlogging) get real IPM-derived area/depth/tonnage math. Classes 4-6
# (missing zebra crossing, missing divider, damaged sign) are presence/
# absence findings - there's no "how many square meters of missing paint"
# to compute, so we report the detection and its IRC remediation reference
# without inventing an asphalt-repair cost for them.
AREA_CLASSES = {1, 2, 3}
DISTRESS_CLASSES = {1, 2, 3, 4, 5, 6}

VEHICLE_CLASS_MAP = {
    "car": "Car",
    "bus": "City Bus",
    "truck": "Heavy Truck",
    "train": "Heavy Truck",
    "motorcycle": "Two-Wheeler",
    "bicycle": "Two-Wheeler",
}

# There is no rut-bar or profilometer in this project, so rutting/IRI are
# reported as a documented proxy correlated with detected cavity severity,
# not a real physical measurement. This mirrors the honesty note already in
# pci_regressor_net.py for the deduct-value curves themselves.
ASSUMED_FRAME_PAVEMENT_AREA_M2 = 25.0  # rough visible-pavement extent in one dashcam frame, for density-% estimates


class DeepInferencePipeline:
    """Wires the project's real models together into one road-photo/telemetry audit."""

    # Typical values for the deterioration forecast when the caller supplies
    # none. Reported back as assumptions, never as measurements.
    DEFAULT_TRAFFIC_ESAL = 7500
    DEFAULT_RAIN_MM = 650.0
    DEFAULT_PAVEMENT_AGE_YR = 3.5

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
        # Road-damage boxes from the YOLOv8 model trained on RDD2022 India, when
        # it has been trained (training/train_rdd_detector.py). Independent
        # evidence shown beside the mask; it does not change area or cost.
        try:
            from models.road_damage_detector import RoadDamageDetector
            self.damage_detector = RoadDamageDetector(checkpoints_dir=self.ckpt_dir)
            if self.damage_detector.is_ready:
                print(f"  ✓ Road-damage detector: {self.damage_detector.backend}")
        except Exception as e:
            print(f"[WARN] road-damage detector unavailable: {e}")
            self.damage_detector = None
        if self.vision_backend == "none":
            print("[WARN] No trained vision model found - run training/train_deep_vision.py "
                  "(deep) or training/train_vision.py (baseline).")

        from models.imu_shock_classifier import load_served_imu_model
        self.imu_model, self.imu_backend = load_served_imu_model(self.ckpt_dir)
        if self.imu_model is None:
            self.imu_model = IMUShockClassifier()  # not ready; the IMU stage reports that plainly
            print("[WARN] IMU model not found - run training/train_imu.py first.")

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
            # The U-Net when checkpoints/segmenter_selection.json chose it and it
            # loads; otherwise the pixel classifier, exactly as before.
            from models.unet_segmenter import load_best_segmenter
            self.segmenter = load_best_segmenter()
            if self.segmenter.is_ready:
                kind = "U-Net" if type(self.segmenter).__name__ == "UNetSegmenter" else "pixel classifier"
                print(f"  ✓ Defect segmenter: {kind} masks "
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
        self.traffic_net = UrbanTrafficNet()
        # A second look at every crack component the segmenter proposes (models/crack_verifier.py). Loaded
        # only when scripts/select_crack_gate.py measured that it removes false alarms without losing a
        # detection; ROAD_SHIELD_CRACK_GATE=force|off overrides that for the measurement itself.
        self.crack_verifier = None
        gate_mode = os.environ.get("ROAD_SHIELD_CRACK_GATE", "auto")
        if gate_mode != "off":
            try:
                from models.crack_verifier import CrackVerifier
                cv_ = CrackVerifier(checkpoints_dir=self.ckpt_dir, require_served=(gate_mode != "force"))
                self.crack_verifier = cv_ if cv_.is_ready else None
                if cv_.is_ready:
                    print(f"  ✓ Crack verifier: threshold {cv_.threshold:.3f}")
            except Exception as e:
                print(f"[WARN] crack verifier unavailable: {e}")
        # Is the photograph one these models can be trusted on at all? (models/ood_guard.py)
        try:
            from models.ood_guard import OODGuard
            self.input_guard = OODGuard(checkpoints_dir=self.ckpt_dir)
            if self.input_guard.is_ready:
                print(f"  ✓ Input guard: {self.input_guard.meta.get('version')}")
        except Exception as e:
            print(f"[WARN] input guard unavailable: {e}")
            self.input_guard = None
        self.audit_hooks = []        # fn(img, result, source, projection, latency_ms): monitoring, active learning
        self._audit_lock = threading.RLock()
        self.model_versions = {}     # set by the server from the model registry
        try:
            from models.dan_dag_network import DANDAGNetwork
            self.dan_dag_net = DANDAGNetwork(
                checkpoints_dir=self.ckpt_dir,
                embedder=getattr(self.vision_model, "embedder", None),
            )
        except Exception as e:
            print(f"[WARN] DAN-DAG network unavailable: {e}")
            self.dan_dag_net = None

    # ------------------------------------------------------------------
    # Single-image audit
    # ------------------------------------------------------------------
    def audit_image(self, image_input, corridor_id="NH-44", latitude=None, longitude=None, **kwargs):
        """
        Runs the full pipeline on one image (see _audit_core), after the input guard (models/ood_guard.py)
        has said whether the photograph is one the models can be trusted on. The guard's answer is
        reported in "input_check"; it does not stop the analysis here (the caller decides: the citizen
        endpoint refuses photographs that are not of a road, an operator upload only shows the warning).
        Hooks (model monitoring, active learning) get the decoded image and the result afterwards.
        Pass _source="citizen" / "fleet" / "video" / "api" so the hooks know where it came from.
        """
        t0 = time.time()
        source = kwargs.pop("_source", "api")
        img = self.cv_detector.decode_image(image_input)
        check, proj = None, None
        guard = getattr(self, "input_guard", None)
        if guard is not None and guard.is_ready:
            try:
                # the guard's limits and the monitoring reference were set on uploads as decoded here (640x480);
                # bus and video frames arrive as arrays at camera resolution, so they are judged at the same size
                gimg = img
                if img.shape[0] != self.cv_detector.target_h or img.shape[1] != self.cv_detector.target_w:
                    from PIL import Image as _Image
                    gimg = np.asarray(_Image.fromarray(np.asarray(img, dtype=np.uint8)).resize(
                        (self.cv_detector.target_w, self.cv_detector.target_h), _Image.Resampling.BILINEAR))
                check = guard.check(gimg, with_projection=bool(getattr(self, "audit_hooks", None)))
                proj = check.pop("projection", None)
            except Exception as e:
                check = {"available": False, "reason": f"input guard failed: {e}"}
        # _audit_core keeps the frame being analysed on the instance (segmenter output, paint mask, crack gate,
        # forecast inputs), and the server shares one pipeline between request threads: one frame at a time,
        # so a request never reads another's state. The models are CPU-bound, so little throughput is lost.
        # rain at this location from Open-Meteo's history (services/road_context.py), when switched on; looked
        # up before taking the lock below, so a slow network never holds up other requests' analysis
        if kwargs.get("rain_mm") is None and latitude is not None and longitude is not None:
            try:
                from services import road_context
                if road_context.enabled():
                    mm, prov = road_context.default().forecast_rain_input(latitude, longitude)
                    if mm is not None:
                        kwargs["rain_mm"] = mm
                        kwargs["_context_inputs"] = {"seasonal_rain_mm": prov}
                    else:
                        kwargs["_context_inputs"] = {"seasonal_rain_mm": {"unavailable": prov.get("reason")}}
            except Exception as e:
                kwargs["_context_inputs"] = {"seasonal_rain_mm": {"unavailable": str(e)[:120]}}
        lock = self.__dict__.setdefault("_audit_lock", threading.RLock())
        with lock:
            res = self._audit_core(img, corridor_id=corridor_id, latitude=latitude, longitude=longitude, **kwargs)
        res["input_check"] = check or {"available": False,
                                       "reason": getattr(guard, "load_error", None) or "input guard not loaded"}
        if getattr(self, "model_versions", None):
            res["model_versions"] = dict(self.model_versions)
        total_ms = round((time.time() - t0) * 1000.0, 2)
        res["total_latency_ms"] = total_ms
        for hook in list(getattr(self, "audit_hooks", None) or []):
            try:
                hook(img, res, source, proj, total_ms)
            except Exception as e:
                print(f"[pipeline] audit hook failed: {e}")
        return res

    def _audit_core(
        self,
        image_input,
        corridor_id="NH-44",
        latitude=None,
        longitude=None,
        chainage_km=None,
        imu_series=None,
        traffic_esal=None,
        rain_mm=None,
        pavement_age_yr=None,
        device_id=None,
        vehicle_speed_kmh=45.0,
        **_extra,
    ):
        """
        Runs the full pipeline on one image. image_input: file path, raw
        bytes, base64 string, PIL Image, or NumPy RGB array. imu_series, if
        given, must be a real (100, 3) accelerometer window - see the IMU
        stage below for what happens when it's omitted.
        """
        t0 = time.time()
        # No location is ever invented. A photograph without GPS is reported
        # with lat/lng = None and any work order is held, not placed at a
        # default coordinate (this used to default to a point in Delhi).
        latitude = None if latitude is None else float(latitude)
        longitude = None if longitude is None else float(longitude)
        # The deterioration forecast needs traffic, rainfall and age. When the
        # caller does not know them, typical values are used and the response
        # lists them under "modelling_assumptions", so a forecast built on
        # assumed inputs cannot pass as one built on measured ones.
        self._assumed_inputs = {}
        self._context_inputs = dict(_extra.get("_context_inputs") or {})
        if traffic_esal is None:
            traffic_esal = self.DEFAULT_TRAFFIC_ESAL
            self._assumed_inputs["traffic_esal_per_day"] = traffic_esal
        if rain_mm is None:
            rain_mm = self.DEFAULT_RAIN_MM
            self._assumed_inputs["seasonal_rain_mm"] = rain_mm
        if pavement_age_yr is None:
            pavement_age_yr = self.DEFAULT_PAVEMENT_AGE_YR
            self._assumed_inputs["pavement_age_years"] = pavement_age_yr
        from models.dan_dag_network import DualAttentionModule, PipelineExecutionDAG
        dag_exec = PipelineExecutionDAG()

        # STAGE 1: decode + standardize
        img_np = self.cv_detector.decode_image(image_input)
        H, W, _ = img_np.shape
        gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]

        # STAGE 2: texture gatekeeper (fast check before running heavy models)
        roi_start_y = int(H * 0.35)
        road_gray = gray[roi_start_y:, :]
        mean_intensity = float(np.mean(road_gray))
        std_intensity = float(np.std(road_gray))

        if std_intensity < 6.5:
            return self._reject_non_pavement(mean_intensity, std_intensity, corridor_id, latitude, longitude, chainage_km, t0)
        dag_exec.mark("N0_scene_gate")

        # DAN Spatial Self-Attention (PAM) over 8x8 road patch grid
        try:
            _, _, self._dan_frame_telemetry = DualAttentionModule.extract_spatial_attention(img_np)
        except Exception:
            self._dan_frame_telemetry = {"domain_regime": "DRY_STANDARD_ASPHALT"}
        self._last_dag_trace = []
        dag_exec.mark("N2_dan_attention")

        # Intrinsics are in pixels, so a profile calibrated at one resolution is
        # simply wrong at another. Rescale per request.
        calib, provenance = self._resolve_calibration(device_id, width_px=W, height_px=H)
        ipm = IPMHomographyEngine.from_calibration(calib)
        self.ipm_engine = ipm
        self.calibration, self.calibration_provenance = calib, provenance

        # Painted markings, found geometrically. Two uses: they are reported as
        # objects in their own right, and their pixels are withheld from the
        # defect proposals. Every false positive in the end-to-end measurement
        # came from a zebra crossing or a divider - that is, from paint - and
        # the segmenter cannot separate paint from a cavity on lightness and
        # contrast alone, because on those features they are the same thing.
        try:
            from models import marking_detector
            self._markings = marking_detector.detect_zebra(
                img_np, roi_top_fraction=self.ROAD_ROI_TOP_FRACTION)
            self._paint_mask = (marking_detector.paint_mask(img_np, self._markings)
                                if self._markings.get("found") else None)
        except Exception as e:
            print(f"[WARN] marking detection failed: {e}")
            self._markings, self._paint_mask = {"found": False}, None

        self._classifier_failures = 0
        self._classifier_last_error = None

        # One segmentation pass for the whole frame; regions index into it.
        seg_out = None
        if getattr(self, "segmenter", None) is not None and self.segmenter.is_ready:
            try:
                seg_out = self.segmenter.segment(img_np)
            except Exception as e:
                print(f"[WARN] segmentation failed, falling back to box area: {e}")
        # Semantic gate: keep segmenter pothole pixels only where the CNN
        # classifier also sees a pothole (models/semantic_gate.py has the
        # measurement). Runs with either CNN classifier - the frozen-embedding head
        # or the fine-tuned network - since both expose predict_probabilities with
        # the same class order. It used to check for the head by name only, so
        # serving the fine-tuned CNN silently switched the gate off and zebra
        # crossings came back as potholes. Skipped with the hand-crafted fallback.
        if seg_out is not None and getattr(self, "vision_backend", "") in ("cnn_embeddings", "deep_cnn"):
            try:
                from models import semantic_gate
                seg_out = semantic_gate.apply(seg_out, self.vision_model, img_np)
            except Exception as e:
                print(f"[WARN] semantic gate skipped: {e}")
        self._seg_out = seg_out
        self._frame_rgb = img_np
        self._crack_gate = None
        dag_exec.mark("N3_pixel_segmenter", "COMPLETED" if seg_out else "SKIPPED")

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

        # STAGE 3a': road-damage boxes (YOLOv8, RDD2022 India), when trained.
        damage_boxes, damage_summary = [], {"available": False,
                                            "reason": "not trained - training/train_rdd_detector.py"}
        dd = getattr(self, "damage_detector", None)
        if dd is not None and dd.is_ready:
            try:
                damage_boxes = dd.detect(img_np)
                counts = {}
                for d in damage_boxes:
                    counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
                damage_summary = {"available": True, "boxes": len(damage_boxes), "counts_by_class": counts,
                                  "model": "YOLOv8 fine-tuned on RDD2022 India",
                                  "confidence_threshold": dd.conf_threshold,
                                  "test_map50": (dd.meta.get("test") or {}).get("map50"),
                                  "role": "locates damage; area and cost come from the segmentation mask"}
            except Exception as e:
                damage_summary = {"available": False, "reason": f"damage detector error: {e}"}
        self._damage = (damage_boxes, damage_summary)

        # STAGE 3a'': the classifier's opinion of the WHOLE frame. Reported beside
        # the region result, never instead of it: area and cost come only from a
        # measured region. Without it, a frame whose regions all fail the gates
        # (or a machine whose segmenter will not load) reads as "Normal Road"
        # even when the classifier is 90% sure it is looking at a pothole.
        frame_cls = self._frame_classification(img_np)

        # STAGE 3b: region proposals (classical CV) + people + vehicles.
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

        # Extract vehicles (Car, City Bus, Heavy Truck, Two-Wheeler) and compute IRC:106-1990 PCU congestion
        raw_vehicles = [d for d in scene_objects if d["class_name"] in VEHICLE_CLASS_MAP]
        vehicle_detections = [
            self._build_vehicle_entry(d, i + 1, H)
            for i, d in enumerate(raw_vehicles)
        ]
        vehicle_counts = {}
        for v in vehicle_detections:
            vtype = v["vehicle_type"]
            vehicle_counts[vtype] = vehicle_counts.get(vtype, 0) + 1
        traffic_analysis = self.traffic_net.calculate_congestion_index(vehicle_counts)
        traffic_analysis["vehicle_counts"] = vehicle_counts
        traffic_analysis["total_vehicles_detected"] = len(vehicle_detections)
        primary_vehicle = vehicle_detections[0] if vehicle_detections else None
        dag_exec.mark("N1_yolo_traffic")

        # Withhold the upper 82% of detected vehicles and pedestrians from the
        # road-defect segmentation mask so dark car tires, wheel arches, and
        # windshields are not proposed as pavement cavities, while leaving the
        # bottom 18% tire-to-road contact strip unmasked.
        obj_mask = np.zeros((H, W), dtype=bool)
        for obj in pedestrians + vehicle_detections:
            ox, oy, ow, oh = obj["bbox_pixels"]
            oy1 = max(oy, oy + int(oh * 0.82))
            obj_mask[max(0, oy):min(H, oy1), max(0, ox):min(W, ox + ow)] = True
        self._object_mask = obj_mask if obj_mask.any() else None

        # Crack and pothole candidates come from the segmenter's mask; the
        # brightness scan supplies everything the segmenter has no class for
        # (waterlogging, signage, markings), water-filled / rim-fragmented
        # cavities, and is the whole proposal stage when no segmenter is loaded.
        seg_boxes = self._segmentation_proposals(H, W)
        heuristic = self.cv_detector.extract_salient_regions(img_np, excluded_boxes=pedestrians)
        dag_exec.mark("N4_contour_proposer")

        # STAGE 4: pedestrian entries
        detections = [self._build_pedestrian_entry(p) for p in pedestrians]

        # STAGE 4/5: VisionDistressNet classification + IPM geometry per candidate region
        self._has_confirmed_seg_distress = False
        confirmed_seg_boxes = []
        for bbox in seg_boxes:
            entry = self._classify_region(img_np, gray, bbox, W, H, mean_intensity)
            if entry is not None:
                detections.append(entry)
                confirmed_seg_boxes.append(bbox)
                if entry.get("is_distress"):
                    self._has_confirmed_seg_distress = True

        if confirmed_seg_boxes:
            heuristic = [b for b in heuristic
                         if not any(self._boxes_overlap(b, sb) for sb in confirmed_seg_boxes)]
        bboxes = seg_boxes + heuristic
        for bbox in heuristic:
            entry = self._classify_region(img_np, gray, bbox, W, H, mean_intensity)
            if entry is not None:
                detections.append(entry)
                if entry.get("is_distress"):
                    self._has_confirmed_seg_distress = True

        # Corridor-level hazard check (Waterlogging, Missing Zebra Crossing, Missing Road Divider, Damaged Sign)
        # when no region-level distress was found.
        if not any(d.get("is_distress") for d in detections):
            scene_entry = self._classify_scene_fallback(img_np, gray, W, H, mean_intensity)
            if scene_entry is not None:
                detections.append(scene_entry)
        dag_exec.mark("N5_dag_classifier")
        dag_exec.mark("N6_pinhole_ipm")

        if not detections and not bboxes and not pedestrians:
            normal = self._normal_road(mean_intensity, std_intensity, corridor_id, latitude, longitude, chainage_km, t0)
            normal["scene_objects"] = scene_objects
            normal["scene_summary"] = scene_summary
            normal["road_damage_boxes"] = damage_boxes
            normal["road_damage_summary"] = damage_summary
            normal["frame_classification"] = frame_cls
            normal.setdefault("segmentation", {"available": seg_out is not None})
            normal["vehicles_count"] = len(vehicle_detections)
            normal["all_vehicles"] = vehicle_detections
            normal["primary_vehicle"] = primary_vehicle
            normal["urban_traffic_analysis"] = traffic_analysis
            dag_exec.mark("N7_astm_pci_hd4")
            dag_exec.mark("N8_irc_compliance")
            dag_exec.mark("N9_merkle_audit", "SKIPPED")
            normal["dan_dag_analysis"] = {
                "available": bool(getattr(self, "dan_dag_net", None) and self.dan_dag_net.is_ready),
                "dan_dual_attention": getattr(self, "_dan_frame_telemetry", {}),
                "dag_execution_graph": dag_exec.export(decision_dag_trace=getattr(self, "_last_dag_trace", [])),
            }
            return normal

        if not detections:
            detections.append(self._normal_road_entry())

        ped_detections = [d for d in detections if d.get("is_pedestrian")]
        distress_detections = [d for d in detections if d.get("is_distress")]

        primary_pedestrian = ped_detections[0] if ped_detections else None
        if distress_detections:
            primary_distress = sorted(
                distress_detections,
                key=lambda d: (
                    d.get("area_method") == "segmentation_mask",
                    round(float(d.get("confidence", 0.0)), 1),
                    d.get("surface_area_m2", 0.0),
                    d.get("class_id") == 2,
                ),
                reverse=True,
            )[0]
        else:
            primary_distress = self._normal_road_entry()

        # Prioritize road distress in primary_detection when a road defect is present so a bystander
        # does not mask a pothole/crack, while keeping primary_pedestrian and all_pedestrians populated.
        primary = primary_distress if distress_detections else (
            primary_pedestrian if primary_pedestrian is not None else primary_distress
        )
        has_dual_targets = bool(ped_detections) and bool(distress_detections)
        has_multi_modal_targets = sum([bool(distress_detections), bool(ped_detections), bool(vehicle_detections)]) >= 2
        dual_target_summary = self._dual_target_summary(ped_detections, distress_detections, primary_pedestrian, primary_distress)
        multi_target_summary = self._multi_target_summary(
            distress_detections, ped_detections, vehicle_detections,
            primary_distress, primary_pedestrian, traffic_analysis,
        )

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
        dag_exec.mark("N7_astm_pci_hd4")

        # MoRTH civil ledger (sum of real per-detection material costing)
        total_tonnage = round(float(sum(d.get("morth_tonnage_t", 0.0) for d in detections)), 3)
        total_repair_inr = round(float(sum(d.get("repair_cost_inr", 0.0) for d in detections)), 2)
        dag_exec.mark("N8_irc_compliance")

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
        dag_exec.mark("N9_merkle_audit", "COMPLETED" if work_order else "SKIPPED")

        elapsed_ms = round((time.time() - t0) * 1000.0, 2)
        dan_dag_block = {
            "available": bool(getattr(self, "dan_dag_net", None) and self.dan_dag_net.is_ready),
            "dan_dual_attention": {
                **getattr(self, "_dan_frame_telemetry", {}),
                "primary_region_channel_attention": (primary.get("dan_dag") or {}).get("top_attended_channels", []),
                "primary_region_consensus": (primary.get("dan_dag") or {}).get("consensus_agreement", True),
            },
            "dag_execution_graph": dag_exec.export(
                decision_dag_trace=(primary.get("dan_dag") or {}).get("decision_dag_path") or getattr(self, "_last_dag_trace", [])
            ),
        }

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
            "vehicles_count": len(vehicle_detections),
            "all_vehicles": vehicle_detections,
            "primary_vehicle": primary_vehicle,
            "urban_traffic_analysis": traffic_analysis,
            "has_dual_targets": has_dual_targets,
            "has_multi_modal_targets": has_multi_modal_targets,
            "dual_target_summary": dual_target_summary,
            "multi_target_summary": multi_target_summary,
            "primary_distress": primary_distress,
            "primary_pedestrian": primary_pedestrian,
            "primary_detection": primary,
            "all_detections": detections,
            "dan_dag_analysis": dan_dag_block,
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
                "asphalt_density_t_m3": IPMHomographyEngine.MATERIAL_PROPERTIES["DBM_SECTION_500"]["density_t_per_m3"],
                "compaction_factor": IPMHomographyEngine.COMPACTION_FACTOR,
                "total_bitumen_tonnage_t": total_tonnage,
                "mix_rate_inr_per_tonne": IPMHomographyEngine.MATERIAL_PROPERTIES["DBM_SECTION_500"]["cost_per_tonne_inr"],
                "rate_basis": IPMHomographyEngine.RATE_BASIS,
                "total_estimated_repair_inr": total_repair_inr,
            },
            "modelling_assumptions": {
                "assumed_inputs": dict(self._assumed_inputs),
                "inputs_from_public_apis": dict(getattr(self, "_context_inputs", {}) or {}),
                "note": ("Inputs listed here were not supplied by the caller; typical "
                         "values were used for the deterioration forecast only.")
                        if self._assumed_inputs else ("No typical values were used: every forecast input came "
                                                      "from the caller or from the public APIs listed."),
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
                "vehicles_detected_count": len(vehicle_detections),
                "pcu_equivalent": traffic_analysis.get("pcu_equivalent", 0.0),
            },
            "scene_objects": scene_objects,
            "scene_summary": scene_summary,
            "road_damage_boxes": damage_boxes,
            "road_damage_summary": damage_summary,
            "frame_classification": frame_cls,
            "crack_gate": getattr(self, "_crack_gate", None) or {
                "applied": False, "reason": (None if getattr(self, "crack_verifier", None) else "crack verifier not served")},
            "markings": getattr(self, "_markings", {"found": False}),
            # Non-zero means at least one region could not be classified, so the
            # number of detections below is a floor, not a count.
            "classifier_failures": getattr(self, "_classifier_failures", 0),
            "classifier_last_error": getattr(self, "_classifier_last_error", None),
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
                "note": self._segmenter_missing_note(),
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

    def _build_vehicle_entry(self, det, vehicle_id, H):
        """Wraps a real ONNX COCO vehicle detection into a structured traffic/PCU entry."""
        coco_cls = det["class_name"]
        vtype = VEHICLE_CLASS_MAP.get(coco_cls, "Car")
        bbox = det["bbox_pixels"]
        bottom_y = bbox[1] + bbox[3]
        dist_m = round(max(2.0, 26.0 * (1.0 - (bottom_y / float(max(1, H))) ** 0.85)), 1)
        return {
            "vehicle_id": vehicle_id,
            "coco_class": coco_cls,
            "vehicle_type": vtype,
            "class_name": f"Vehicle ({vtype})",
            "confidence": det["confidence"],
            "bbox_pixels": bbox,
            "bbox_normalized": det["bbox_normalized"],
            "distance_meters": dist_m,
            "pcu_weight": PCU_WEIGHTS.get(vtype, 1.0),
            "irc_standard_specification": TRAFFIC_IRC_STANDARDS.get(vtype, ""),
            "detector": det.get("detector", "onnx_coco_detector"),
            "is_vehicle": True,
            "is_distress": False,
            "is_pedestrian": False,
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
    # Per class, for the same reason the confidence margin is per class: a
    # pixel COUNT is a shape-dependent quantity.
    #
    # A pothole is compact - a 50 cm cavity is a solid blob of several hundred
    # pixels at working resolution. A crack of the same real extent is two or
    # three pixels wide, so it has an order of magnitude fewer pixels while
    # being just as real and just as expensive to seal.
    #
    # Measured on the three cracks this filter was missing:
    #     cpr_1159_2442   biggest crack blob  107 px   floor 256   rejected
    #     cpr_1144_2411   biggest crack blob   52 px   floor 256   rejected
    # One floor for both classes silently made the system pothole-only for thin
    # cracks, which are the majority of pavement distress by length.
    # Swept end-to-end through audit_image(); the curve is the reason for 0.0012
    # rather than something rounder:
    #
    #     crack floor   false positives   defects found
    #        0.0006       7/36 (19.4%)     23/24 (95.8%)
    #        0.0012       4/36 (11.1%)     22/24 (91.7%)   <- committed
    #        0.0020       4/36 (11.1%)     21/24 (87.5%)
    #        0.0040       4/36 (11.1%)     21/24 (87.5%)   one floor for both
    #
    # 0.0012 costs nothing in precision and recovers four points of detection
    # over the single shared floor.
    MIN_COMPONENT_FRACTION = {"crack": 0.0012, "pothole": 0.004}
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
    # Everything above this fraction of the frame is not the road surface in
    # front of the vehicle: it is sky, buildings, trees and other traffic. The
    # brightness grid has always restricted itself to below this line
    # (cv_cavity_detector.roi_y0); the segmentation proposals did not, and a
    # parked car at the top of a photograph was reported as "Pothole 100%"
    # while the crater filling the foreground was ignored.
    #
    # The features the segmenter uses - lightness, gradient, local variance -
    # describe a dark high-contrast patch. A car body against bright tarmac is
    # exactly that. No threshold distinguishes them; their POSITION does.
    ROAD_ROI_TOP_FRACTION = 0.35

    def _segmenter_missing_note(self):
        """Say WHY there is no mask: never trained, or on disk but unloadable here."""
        seg = getattr(self, "segmenter", None)
        detail = getattr(seg, "load_error_detail", None) or {}
        if seg is not None and getattr(seg, "file_exists", False):
            here, trained = detail.get("sklearn_here"), detail.get("sklearn_trained_with")
            why = (f"scikit-learn {here} here, model trained with {trained}" if here and trained
                   else (detail.get("likely_cause") or "it failed to load"))
            return (f"A segmenter is on disk but will not load on this machine ({why}). "
                    f"Install the pinned version: pip install \"scikit-learn>=1.8,<1.9\". "
                    "Until then areas are box estimates, which overstate a diagonal crack ~10x.")
        return ("No segmenter on disk. Train one with training/train_segmenter.py - a box around a "
                "diagonal crack overstates its area by roughly an order of magnitude.")

    def _frame_classification(self, img_np):
        """Whole-frame class probabilities from the served CNN, or None."""
        if getattr(self, "vision_backend", "") not in ("cnn_embeddings", "deep_cnn"):
            return None
        try:
            probs = [float(x) for x in np.asarray(self.vision_model.predict_probabilities(img_np)).ravel()]
            names = list(getattr(self.vision_model, "class_names", None) or
                         ["Normal Road / Sound Pavement", "Crack (Longitudinal / Transverse / Alligator)",
                          "Pothole Cavity", "Waterlogging / Flooding Hazard", "Missing Zebra Crossing",
                          "Missing Road Divider", "Damaged Traffic Sign"])[:len(probs)]
            top = int(np.argmax(probs))
            out = {"class_name": names[top], "confidence": round(probs[top], 4),
                   "probabilities": {n: round(p, 4) for n, p in zip(names, probs)},
                   "model": self.vision_backend,
                   "role": "whole-frame opinion of the classifier; does not set area, depth or cost"}
            route = getattr(self.vision_model, "last_route", None)
            if route:                       # an ensemble served as a cascade: which networks answered
                out["cascade"] = dict(route)
                out["model_detail"] = getattr(self.vision_model, "backend", None)
            return out
        except Exception as e:
            return {"error": f"frame classification failed: {e}"}

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
        # Withhold painted pixels. Only the bars themselves are removed, never
        # the asphalt between them - that is real road and can hold a real
        # pothole.
        paint = getattr(self, "_paint_mask", None)
        if paint is not None:
            import cv2 as _cv2
            pm = paint
            if pm.shape[:2] != mask.shape[:2]:
                pm = _cv2.resize(pm.astype(_np.uint8), (mask.shape[1], mask.shape[0]),
                                 interpolation=_cv2.INTER_NEAREST).astype(bool)
            mask = mask.copy()
            mask[pm] = 0
        obj_m = getattr(self, "_object_mask", None)
        if obj_m is not None:
            import cv2 as _cv2
            om = obj_m
            if om.shape[:2] != mask.shape[:2]:
                om = _cv2.resize(om.astype(_np.uint8), (mask.shape[1], mask.shape[0]),
                                 interpolation=_cv2.INTER_NEAREST).astype(bool)
            mask = mask.copy()
            mask[om] = 0
        mh, mw = mask.shape[:2]
        frame_px = float(mh * mw)
        sx, sy = W / float(mw), H / float(mh)
        def _min_px_for(name):
            frac = self.MIN_COMPONENT_FRACTION
            if isinstance(frac, dict):
                frac = frac.get(name, 0.004)
            return max(12, int(float(frac) * frame_px))
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
            min_px = _min_px_for("crack" if seg_cls == 1 else "pothole")
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
                # The blob's centre of mass must lie on the road, not above the
                # horizon. Centroid rather than the top edge, so a large defect
                # close to the camera - which legitimately reaches high up a
                # close-up frame - is kept, while a car or a sign sitting
                # entirely in the upper third is not.
                centroid_y = (y0 + y1) / 2.0
                if centroid_y < self.ROAD_ROI_TOP_FRACTION * H:
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
                if (x1 - x0) * (y1 - y0) > 0.55 * W * H:
                    scale = (0.50 * W * H / float((x1 - x0) * (y1 - y0))) ** 0.5
                    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                    nw = max(8, int((x1 - x0) * scale))
                    nh = max(8, int((y1 - y0) * scale))
                    x0 = max(0, min(W - nw, int(round(cx - nw / 2.0))))
                    y0 = max(0, min(H - nh, int(round(cy - nh / 2.0))))
                    x1, y1 = x0 + nw, y0 + nh
                out.append([x0, y0, x1 - x0, y1 - y0, mean_p * area, kind,
                            "segmentation", round(mean_p, 4),
                            round(area / frame_px, 5), oversized])
        out = self._verify_cracks(out)
        out.sort(key=lambda b: -b[4])
        return out[: self.MAX_COMPONENTS]

    def _verify_cracks(self, boxes):
        """Drop crack components the crack verifier does not see as a pavement crack (models/crack_verifier.py)."""
        cv_ = getattr(self, "crack_verifier", None)
        img = getattr(self, "_frame_rgb", None)
        cracks = [b for b in boxes if b[5] == 1]
        if cv_ is None or img is None or not cracks:
            return boxes
        from models.crack_verifier import crop_around
        try:
            p = cv_.proba([crop_around(img, b) for b in cracks])
        except Exception as e:
            self._crack_gate = {"applied": False, "error": str(e)[:200]}
            return boxes
        keep_ids = {id(b) for b, pi in zip(cracks, p) if pi >= cv_.threshold}
        self._crack_gate = {"applied": True, "threshold": round(cv_.threshold, 4),
                            "crack_components": len(cracks), "removed": len(cracks) - len(keep_ids),
                            "probabilities": [round(float(x), 3) for x in p]}
        return [b for b in boxes if b[5] != 1 or id(b) in keep_ids]

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

    def _verify_rim_fragmented_cavity(self, img_np, cls_id, bx, by, bw, bh, H, W, crop_conf):
        """
        Verifies a water-filled or rim-fragmented cavity when no single
        connected component in mask_work cleared the size floor on its own.
        Requires four independent checks to pass simultaneously:
          1. Region classifier confidence >= 0.60.
          2. Not painted road markings (_paint_mask covers <= 2% of region).
          3. Segmenter probability map inside [bx, by, bw, bh] has at least
             min_px pixels above the calibrated class threshold and max >= 0.50.
          4. Road ROI crop (y >= ROAD_ROI_TOP_FRACTION * H) independently
             classifies as the same defect class with confidence >= 0.65.
        """
        if crop_conf < 0.60:
            return None
        paint = getattr(self, "_paint_mask", None)
        if paint is not None:
            sub_paint = paint[max(0, by):min(H, by + bh), max(0, bx):min(W, bx + bw)]
            if sub_paint.size and (float(sub_paint.sum()) / float(sub_paint.size)) > 0.02:
                return None
        seg = getattr(self, "_seg_out", None)
        if seg is None:
            return None
        p_key = "proba_crack" if cls_id == 1 else "proba_pothole"
        t_key = "crack" if cls_id == 1 else "pothole"
        pp = seg.get(p_key)
        if pp is None:
            return None
        mh, mw = pp.shape[:2]
        wy0 = max(0, int(by * mh / float(H)))
        wy1 = min(mh, int(round((by + bh) * mh / float(H))))
        wx0 = max(0, int(bx * mw / float(W)))
        wx1 = min(mw, int(round((bx + bw) * mw / float(W))))
        sub_pp = pp[wy0:wy1, wx0:wx1]
        if sub_pp.size == 0:
            return None
        thresh = float((getattr(self.segmenter, "thresholds", None) or {}).get(
            t_key, 0.20 if cls_id == 2 else 0.50
        ))
        frac_cfg = (self.MIN_COMPONENT_FRACTION.get(t_key, 0.004)
                    if isinstance(self.MIN_COMPONENT_FRACTION, dict)
                    else float(self.MIN_COMPONENT_FRACTION))
        min_px = max(12, int(float(frac_cfg) * mh * mw))
        active_px = int((sub_pp >= thresh).sum())
        if active_px < min_px or float(sub_pp.max()) < max(0.50, self._component_floor(cls_id)):
            return None
        roi_crop = img_np[int(H * self.ROAD_ROI_TOP_FRACTION):, :]
        if roi_crop.size == 0:
            return None
        try:
            roi_pred = self.vision_model.predict_image(roi_crop)
        except Exception as e:
            self._classifier_failures = getattr(self, "_classifier_failures", 0) + 1
            self._classifier_last_error = str(e)
            return None
        if roi_pred.get("class_id") != cls_id or float(roi_pred.get("confidence", 0.0)) < 0.65:
            return None
        return max(crop_conf, float(roi_pred.get("confidence", crop_conf)))

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

        from_segmenter = len(bbox) > 6 and bbox[6] == "segmentation"
        try:
            pred = self.vision_model.predict_image(crop)
            if (not from_segmenter
                    and not getattr(self, "_has_confirmed_seg_distress", False)
                    and pred["class_id"] in (0, 3)
                    and (bw * bh) >= 0.08 * W * H):
                ctx_x0 = max(10, bx - int(bw * 0.15))
                ctx_y0 = max(int(H * self.ROAD_ROI_TOP_FRACTION), by - int(bh * 0.25))
                ctx_x1 = min(W - 10, bx + bw + int(bw * 0.15))
                ctx_y1 = min(H - 10, by + bh + int(bh * 0.10))
                ctx_crop = img_np[ctx_y0:ctx_y1, ctx_x0:ctx_x1]
                if ctx_crop.size > 0:
                    ctx_pred = self.vision_model.predict_image(ctx_crop)
                    if ctx_pred["class_id"] in (1, 2) and float(ctx_pred.get("confidence", 0.0)) >= 0.60:
                        pred = ctx_pred
                        bx, by = ctx_x0, ctx_y0
                        bw, bh = ctx_x1 - ctx_x0, ctx_y1 - ctx_y0
                        if bw * bh > 0.50 * W * H:
                            scale = (0.48 * W * H / float(bw * bh)) ** 0.5
                            cx, cy = bx + bw / 2.0, by + bh / 2.0
                            nw, nh = max(8, int(bw * scale)), max(8, int(bh * scale))
                            bx = max(0, min(W - nw, int(round(cx - nw / 2.0))))
                            by = max(0, min(H - nh, int(round(cy - nh / 2.0))))
                            bw, bh = nw, nh
        except Exception as e:
            # One region's classification failed. The frame is still worth
            # reporting, but the COUNT of detections in it is now unreliable,
            # so the failure is recorded on the result rather than swallowed.
            self._classifier_failures = getattr(self, "_classifier_failures", 0) + 1
            self._classifier_last_error = str(e)
            return None
        cls_id = pred["class_id"]
        if cls_id == 0:
            return None  # classifier says this region isn't actually a distress after all

        if float(pred.get("confidence", 0.0)) < self.MIN_REPORT_CONFIDENCE:
            return None

        if from_segmenter and cls_id not in (1, 2):
            return None
        if from_segmenter and cls_id in (1, 2) and int(bbox[5]) in (1, 2):
            seg_kind = int(bbox[5])
            if cls_id != seg_kind:
                own_frac = self._region_defect_fraction(cls_id, bx, by, bw, bh, H, W) or 0.0
                if own_frac < self.SEGMENTER_GATE_FRACTION:
                    cls_id = seg_kind
                    pred["class_id"] = cls_id
                    pred["class_name"] = VisionDistressNet.CLASS_NAMES[cls_id]
        if cls_id in (1, 2) and not from_segmenter:
            if getattr(self, "segmenter", None) is not None and self.segmenter.is_ready:
                if getattr(self, "_has_confirmed_seg_distress", False):
                    return None
                rescued_conf = self._verify_rim_fragmented_cavity(
                    img_np, cls_id, bx, by, bw, bh, H, W, float(pred.get("confidence", 0.0))
                )
                if rescued_conf is None:
                    return None
                pred["confidence"] = round(rescued_conf, 4)
                if bw * bh > 0.50 * W * H:
                    scale = (0.48 * W * H / float(bw * bh)) ** 0.5
                    cx, cy = bx + bw / 2.0, by + bh / 2.0
                    nw, nh = max(8, int(bw * scale)), max(8, int(bh * scale))
                    bx = max(0, min(W - nw, int(round(cx - nw / 2.0))))
                    by = max(0, min(H - nh, int(round(cy - nh / 2.0))))
                    bw, bh = nw, nh
            else:
                box_fraction = (bw * bh) / float(max(1, W * H))
                if box_fraction > self.MAX_HEURISTIC_BOX_FRACTION:
                    return None

        dan_dag_info = None
        if getattr(self, "dan_dag_net", None) is not None and self.dan_dag_net.is_ready:
            try:
                eval_crop = img_np[max(0, by):min(H, by + bh), max(0, bx):min(W, bx + bw)]
                if eval_crop.size > 0:
                    dan_res = self.dan_dag_net.predict_image(eval_crop)
                    self._last_dag_trace = dan_res.get("dag_trace", [])
                    consensus = (dan_res["class_id"] == cls_id) or (dan_res["dag_leaf_class_id"] == cls_id)
                    # The reported confidence stays the classifier's own
                    # probability. Agreement with DAN-DAG is recorded beside it
                    # (consensus_agreement) but does not raise the number: the
                    # old blend took max(p, 0.55p + 0.45q + 0.03), so it could
                    # only ever go up and carried an unjustified +0.03.
                    dan_dag_info = {
                        "dan_dag_class_id": dan_res["class_id"],
                        "dan_dag_class_name": dan_res["class_name"],
                        "dan_dag_confidence": dan_res["confidence"],
                        "dag_leaf_class_id": dan_res["dag_leaf_class_id"],
                        "dag_leaf_class_name": dan_res["dag_leaf_class_name"],
                        "consensus_agreement": bool(consensus),
                        "spatial_attention_peak": dan_res["dan_spatial"]["spatial_peak"],
                        "cavity_basin_score": dan_res["dan_spatial"]["cavity_basin_score"],
                        "domain_regime": dan_res["dan_spatial"]["domain_regime"],
                        "channel_gate_mean": dan_res["dan_channel"]["channel_gate_mean"],
                        "top_attended_channels": dan_res["dan_channel"]["top_attended_channels"],
                        "decision_dag_path": dan_res["dag_trace"],
                    }
            except Exception:
                dan_dag_info = None

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
            "dan_dag": dan_dag_info,
            "shannon_entropy_bits": pred["shannon_entropy_bits"],
            "uncertainty_rating": pred["uncertainty_rating"],
            "astm_d6433_severity": pred["astm_d6433_severity"],
            "irc_standard_specification": pred["irc_standard_specification"],
            "top3_ranked_predictions": pred["top3_ranked_predictions"],
            "is_distress": cls_id in DISTRESS_CLASSES,
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

    def _classify_scene_fallback(self, img_np, gray, W, H, mean_intensity):
        """
        Scene-level classification for corridor-wide urban safety hazards
        (3: Waterlogging, 4: Missing Zebra Crossing, 5: Missing Road Divider,
        6: Damaged Traffic Sign) when no localized region triggered a detection.
        Cracks (1) and Potholes (2) are intentionally excluded here so they are
        only ever reported from localized segmentation/region proposals.
        """
        try:
            pred = self.vision_model.predict_image(img_np)
        except Exception as e:
            self._classifier_failures = getattr(self, "_classifier_failures", 0) + 1
            self._classifier_last_error = str(e)
            return None
        cls_id = pred.get("class_id", 0)
        conf = float(pred.get("confidence", 0.0))
        if cls_id not in (3, 4, 5, 6) or conf < self.MIN_REPORT_CONFIDENCE:
            return None

        bx = int(W * 0.10)
        by = int(H * self.ROAD_ROI_TOP_FRACTION)
        bw = int(W * 0.80)
        bh = int(H * (0.95 - self.ROAD_ROI_TOP_FRACTION))
        bx_norm, by_norm = round(bx / float(W), 4), round(by / float(H), 4)
        bw_norm, bh_norm = round(bw / float(W), 4), round(bh / float(H), 4)

        area_m2 = 0.0
        dist_m = 8.0
        area_method = None
        area_diag = {}
        if cls_id == 3:
            _, ground_y = self.ipm_engine.pixel_to_ground(bx + bw / 2.0, by + bh / 2.0)
            dist_m = max(1.8, min(30.0, float(ground_y)))
            area_m2 = min(self.MAX_SINGLE_REPAIR_AREA_M2, self.ipm_engine.calculate_surface_area_sqm(bx, by, bw // 2, bh // 2))
            area_method = "corridor_roi_estimate"

        return {
            "bbox_pixels": [bx, by, bw, bh],
            "bbox_normalized": [bx_norm, by_norm, bw_norm, bh_norm],
            "class_id": cls_id,
            "class_name": pred["class_name"],
            "confidence": conf,
            "shannon_entropy_bits": pred["shannon_entropy_bits"],
            "uncertainty_rating": pred["uncertainty_rating"],
            "astm_d6433_severity": pred["astm_d6433_severity"],
            "irc_standard_specification": pred["irc_standard_specification"],
            "top3_ranked_predictions": pred["top3_ranked_predictions"],
            "is_distress": True,
            "distance_meters": round(dist_m, 1),
            "surface_area_m2": round(area_m2, 3),
            "area_method": area_method,
            "area_diagnostics": area_diag,
            "area_is_plausible_single_repair": True if cls_id in AREA_CLASSES else None,
            "needs_manual_survey": False,
            "depth_cm": 0.0,
            "depth_estimate": None,
            "volumetric_m3": 0.0,
            "morth_tonnage_t": 0.0,
            "repair_cost_inr": 0.0,
            "physical_dimensions": {
                "surface_area_m2": round(area_m2, 2),
                "depth_cm": 0.0,
                "bitumen_volume_m3": 0.0,
                "morth_compacted_tonnage_t": 0.0,
                "estimated_repair_cost_inr": 0.0,
            },
            "probabilities": pred["all_class_probabilities"],
        }

    def _normal_road_entry(self):
        return {
            "class_id": 0,
            "class_name": "Normal Road / Sound Pavement",
            # Not a model probability: no distress region survived the gates,
            # so there is no classifier score to report. (Was a fixed 0.99.)
            "confidence": None,
            "decision_basis": "no distress region survived proposal, classifier and plausibility gates",
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

    def _multi_target_summary(
        self, distress_detections, ped_detections, vehicle_detections,
        primary_distress, primary_pedestrian, traffic_analysis,
    ):
        parts = []
        if distress_detections:
            parts.append(
                f"{len(distress_detections)} road defect(s) [primary: {primary_distress['class_name']}, "
                f"{primary_distress.get('surface_area_m2', 0.0):.3f} m^2]"
            )
        if ped_detections and primary_pedestrian:
            parts.append(
                f"{len(ped_detections)} pedestrian(s) [closest {primary_pedestrian['distance_meters']}m, "
                f"alert: {primary_pedestrian.get('alert_level', 'ADVISORY')}]"
            )
        if vehicle_detections:
            vc_str = ", ".join(f"{k}: {v}" for k, v in sorted((traffic_analysis.get("vehicle_counts") or {}).items()))
            parts.append(
                f"{len(vehicle_detections)} vehicle(s) [{vc_str}; {traffic_analysis.get('pcu_equivalent', 0.0)} PCU]"
            )
        if not parts:
            return "Normal road surface; no defects, pedestrians, or vehicles detected."
        return "Simultaneous multi-target detection: " + " + ".join(parts) + "."

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
            {"available": True, "shock_classification": cls_name, "peak_delta_z_ms2": round(delta_z, 2),
             "pothole_shock_probability": round(p_imu, 4), "model": getattr(self, "imu_backend", "random_forest")},
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
            # Rule-based rejection (texture std below threshold), not a model
            # probability - so no confidence is claimed. (Was a fixed 0.99.)
            "confidence": None,
            "decision_basis": "luminance standard deviation below the texture-gate threshold",
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
                # chainage is whatever the caller supplies (default None) - it used
                # to be invented as 100 km + 250 m per file, which no survey records
                rec = self.audit_image(image_input=img_path, corridor_id=corridor_id, **kwargs)
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
