"""
ROAD-SHIELD AI Engine - HTTP API server.

A small multi-threaded REST server (stdlib http.server, no framework) that
exposes this project's real models: VisionDistressNet, IMUShockClassifier,
BayesianFusionGate, IPMHomographyEngine, the forensic duplicate/texture
auditors, MoRTHDispatchAgent, the ASTM D6433 PCI engine, the deterioration
forecaster, the automotive ADAS policy agent, and the deep inference
pipeline that wires them together.
"""
import sys
if sys.stdout is None:
    sys.stdout = open("server.log", "a", encoding="utf-8")
elif hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if sys.stderr is None:
    sys.stderr = open("server_err.log", "a", encoding="utf-8")
elif hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import time
import socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from models.vision_distress_net import VisionDistressNet
from models.imu_shock_classifier import IMUShockClassifier
from models.bayesian_fusion_gate import BayesianFusionGate
from models.ipm_homography_engine import IPMHomographyEngine
from models.forensic_audit_engine import ForensicMetricEmbedder, ForensicTextureAuditor
from models.morth_dispatch_agent import MoRTHDispatchAgent
from models.pci_regressor_net import PCIRegressorNet
from models.pavement_deterioration_forecaster import PavementDeteriorationForecaster
from data.dataset_generator import sample_real_imu_window
from data.benchmark_dataset_hub import BenchmarkDatasetHub
from training.mega_pipeline import run_training_suite, telemetry_streamer
from data.realworld_media_engine import RealWorldMediaEngine
from models.realworld_video_tracker import SpatialTemporalVideoTracker
from models.cv_cavity_detector import CVCavityDetector
from models.edge_model_exporter import EdgeModelExporter
from pipeline.deep_inference_pipeline import DeepInferencePipeline
from models.urban_traffic_net import UrbanTrafficNet
from models.alpr_incident_tracker import ALPRIncidentTracker
from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
from models.multimodal_transformer_fusion import MultimodalTransformerFusionNet
from models.automotive_rl_policy_agent import AutomotiveRLPolicyAgent
from models.automotive_telematics_engine import AutomotiveTelematicsEngine
import urllib.parse
from services.google_maps_service import google_maps_service


# ==============================================================================
# GLOBAL MODEL INITIALIZATION
# ==============================================================================
CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")

# On Vercel (and similar serverless hosts) the deployed files are read-only
# and only /tmp is writable - and /tmp is wiped between cold starts. Anything
# the server writes at runtime goes to WRITABLE_DIR; trained models are
# still read from CKPT_DIR.
IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
_ckpt_writable = os.access(CKPT_DIR, os.W_OK) if os.path.isdir(CKPT_DIR) else os.access(ENGINE_ROOT, os.W_OK)
WRITABLE_DIR = CKPT_DIR if (_ckpt_writable and not IS_SERVERLESS) else os.path.join("/tmp", "road_shield")

print("[AI Server] Loading trained models from:", CKPT_DIR)

vision_model = VisionDistressNet(model_path=os.path.join(CKPT_DIR, "vision_distress_model.joblib"))
print("  ✓ Vision distress classifier", "loaded" if vision_model.is_ready else "NOT TRAINED YET (run training/train_vision.py)")

imu_model = IMUShockClassifier(model_path=os.path.join(CKPT_DIR, "imu_shock_model.joblib"))
print("  ✓ IMU shock classifier", "loaded" if imu_model.is_ready else "NOT TRAINED YET (run training/train_imu.py)")

bayesian_gate = BayesianFusionGate(prior_pothole_prob=0.05, decision_threshold_log_odds=1.8)
ipm_engine = IPMHomographyEngine(camera_height_m=1.45, pitch_deg=18.4)
forensic_embedder = ForensicMetricEmbedder(hash_size=16, duplicate_hamming_threshold=6)
texture_auditor = ForensicTextureAuditor()
dispatch_agent = MoRTHDispatchAgent()
pci_model = PCIRegressorNet()
degrade_model = PavementDeteriorationForecaster()
dataset_hub = BenchmarkDatasetHub(seed=42)
media_engine = RealWorldMediaEngine()
cv_detector = CVCavityDetector()
edge_exporter = EdgeModelExporter(CKPT_DIR)
multimodal_net = MultimodalTransformerFusionNet(imu_weight=0.4)
rl_agent = AutomotiveRLPolicyAgent()
automotive_telematics = AutomotiveTelematicsEngine(checkpoints_dir=CKPT_DIR)
traffic_net = UrbanTrafficNet()
alpr_tracker = ALPRIncidentTracker()
deep_pipeline = DeepInferencePipeline(CKPT_DIR)
print("  ✓ Deep inference pipeline initialized.")

fleet_dedup_engine = FleetDeduplicationEngine(proximity_threshold_meters=10.0)
# Seed a handful of demo defects so the GIS map isn't empty on a fresh
# server start - these are labeled fixture data below, not live telemetry.
fleet_dedup_engine.ingest_fleet_detection("BUS-KA01-101", 12.9716, 77.5946, "Pothole Cavity", 42.0, 1.85, enrich_location=False)
fleet_dedup_engine.ingest_fleet_detection("BUS-KA01-204", 12.9717, 77.5945, "Pothole Cavity", 38.0, 2.10, enrich_location=False)  # near-duplicate -> confirmed hotspot
fleet_dedup_engine.ingest_fleet_detection("BUS-KA01-101", 12.9750, 77.5980, "Waterlogging / Flooding Hazard", 35.0, 5.20, enrich_location=False)
fleet_dedup_engine.ingest_fleet_detection("BUS-KA01-308", 12.9680, 77.5910, "Missing Zebra Crossing", 60.0, 3.40, enrich_location=False)
fleet_dedup_engine.ingest_fleet_detection("BUS-KA01-204", 12.9800, 77.6050, "Damaged Traffic Sign", 55.0, 0.80, enrich_location=False)
print("  ✓ Fleet deduplication engine initialized (seeded with 5 demo reports).")

active_feedback_counter = 0
video_trackers = {}
FEEDBACK_LOG_PATH = os.path.join(WRITABLE_DIR, "active_feedback_log.jsonl")


def find_or_export_artifact(filename):
    """Path to an exported spec/header, looking in checkpoints/ first, then
    WRITABLE_DIR; exports into WRITABLE_DIR if neither has it yet."""
    for folder in (CKPT_DIR, WRITABLE_DIR):
        candidate = os.path.join(folder, filename)
        if os.path.exists(candidate):
            return candidate
    edge_exporter.export_all_to_open_spec(output_dir=WRITABLE_DIR)
    candidate = os.path.join(WRITABLE_DIR, filename)
    return candidate if os.path.exists(candidate) else None


def get_session_tracker(session_id="default", reset=False):
    if reset or session_id not in video_trackers:
        video_trackers[session_id] = SpatialTemporalVideoTracker(iou_threshold=0.25, max_age_frames=5)
    return video_trackers[session_id]


# ==============================================================================
# HTTP REQUEST HANDLER WITH CORS SUPPORT
# ==============================================================================
class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class RoadShieldAPIHandler(BaseHTTPRequestHandler):

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, status_code, data):
        def _json_serial(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, "item"):
                return obj.item()
            return str(obj)

        try:
            payload = json.dumps(data, indent=2, default=_json_serial)
        except Exception as e:
            payload = json.dumps({"error": f"Serialization error: {str(e)}"}, indent=2)
            status_code = 500

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))

    def _read_json_body(self):
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            return {}
        body = self.rfile.read(content_len)
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return {}

    def log_message(self, format, *args):
        # Keep default stderr logging quiet in normal operation; uncomment for debugging.
        pass

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------
    def do_GET(self):
        full_path = self.path
        path = full_path.split("?")[0]
        t0 = time.time()

        accept_header = self.headers.get("Accept", "")
        if path in ["/dashboard", "/frontend", "/gui", "/app"] or (path == "/" and "text/html" in accept_header):
            frontend_path = os.path.join(ENGINE_ROOT, "road_shield_frontend.html")
            if os.path.exists(frontend_path):
                with open(frontend_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)
                return

        if path in ["/", "/api/v1/health"]:
            self._send_json(200, {
                "service": "ROAD-SHIELD AI Engine",
                "status": "ONLINE",
                "timestamp_utc": int(time.time()),
                "models": {
                    "vision_distress_net": "LOADED" if vision_model.is_ready else "NOT_TRAINED",
                    "imu_shock_classifier": "LOADED" if imu_model.is_ready else "NOT_TRAINED",
                    "bayesian_fusion_gate": "READY",
                    "ipm_homography_engine": "READY",
                    "forensic_audit_engine": "READY",
                    "morth_dispatch_agent": "READY",
                    "astm_d6433_pci_engine": "READY_FORMULA_BASED",
                    "deterioration_forecaster": "READY_FORMULA_BASED",
                    "urban_traffic_net": "READY_FORMULA_BASED",
                },
            })
            return

        # ---------------- Google Maps / GIS ----------------
        if path == "/api/v1/maps/status":
            self._send_json(200, google_maps_service.get_service_status())
            return

        if path == "/api/v1/maps/tile-layers":
            self._send_json(200, {"status": "OK", "default_layer": "google_roadmap", "tile_layers": google_maps_service.get_tile_layers()})
            return

        if path == "/api/v1/maps/geocode":
            q_params = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            q = q_params.get("query", q_params.get("address", [""]))[0]
            self._send_json(200, google_maps_service.geocode(q))
            return

        if path == "/api/v1/maps/reverse-geocode":
            q_params = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            lat = float(q_params.get("lat", [12.9716])[0])
            lon = float(q_params.get("lon", q_params.get("lng", [77.5946]))[0])
            self._send_json(200, google_maps_service.reverse_geocode(lat, lon))
            return

        if path == "/api/v1/maps/elevation":
            q_params = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            lat = float(q_params.get("lat", [12.9716])[0])
            lon = float(q_params.get("lon", q_params.get("lng", [77.5946]))[0])
            self._send_json(200, google_maps_service.get_elevation(lat, lon))
            return

        if path == "/api/v1/maps/places-nearby":
            q_params = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            lat = float(q_params.get("lat", [12.9716])[0])
            lon = float(q_params.get("lon", q_params.get("lng", [77.5946]))[0])
            fac_type = q_params.get("type", ["all"])[0]
            self._send_json(200, google_maps_service.find_nearby_civil_facilities(lat, lon, fac_type))
            return

        if path == "/api/v1/maps/streetview-url":
            q_params = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            lat = float(q_params.get("lat", [12.9716])[0])
            lon = float(q_params.get("lon", q_params.get("lng", [77.5946]))[0])
            self._send_json(200, google_maps_service.get_streetview_metadata(lat, lon))
            return

        # ---------------- Fleet / GIS dashboard ----------------
        if path == "/api/v1/gis/map-data":
            defects = fleet_dedup_engine.get_all_deduplicated_defects()
            for d in defects:
                if not d.get("address"):
                    geo = google_maps_service.reverse_geocode(d["lat"], d["lon"])
                    d["address"] = geo.get("formatted_address")
                    d["geocode_source"] = geo.get("provider")
                    d["google_maps_url"] = geo.get("google_maps_url", f"https://www.google.com/maps/search/?api=1&query={d['lat']},{d['lon']}")
                    d["street_view_url"] = geo.get("street_view_url", f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={d['lat']},{d['lon']}")
                    elev = google_maps_service.get_elevation(d["lat"], d["lon"])
                    d["elevation_m"] = elev.get("elevation_meters")
                    d["drainage_risk"] = elev.get("drainage_risk_category")
            civil_depots = google_maps_service.find_nearby_civil_facilities(12.9716, 77.5946)["facilities"]
            self._send_json(200, {
                "system": "ROAD-SHIELD Fleet & Defect GIS Dashboard",
                "note": "deduplicated_defects is real, computed state from fleet_dedup_engine (seeded with 5 demo reports at server startup). There is no live bus GPS/traffic feed in this project, so fleet_units/congestion figures below are not shown here - see /api/v1/fleet/telemetry for the real dedup registry stats instead.",
                "google_maps_status": google_maps_service.get_service_status(),
                "tile_layers": google_maps_service.get_tile_layers(),
                "default_tile_layer": "google_roadmap",
                "deduplicated_defects": defects,
                "civil_infrastructure": civil_depots,
                "timestamp_utc": int(time.time()),
            })
            return

        if path == "/api/v1/fleet/telemetry":
            stats = fleet_dedup_engine.get_deduplication_stats()
            self._send_json(200, {
                "unique_defects_registered": stats["unique_defects_registered"],
                "total_reports_ingested": stats["total_reports_ingested"],
                "deduplication_efficiency_pct": stats["deduplication_efficiency_pct"],
                "fleet_status": "DEMO_SEED_DATA_NO_LIVE_GPS_FEED",
            })
            return

        # ---------------- Training / models ----------------
        if path == "/api/v1/training/metrics":
            vis_report = os.path.join(CKPT_DIR, "vision_distress_report.json")
            imu_report = os.path.join(CKPT_DIR, "imu_shock_report.json")
            out = {}
            for name, path_ in (("vision", vis_report), ("imu", imu_report)):
                if os.path.exists(path_):
                    with open(path_, "r", encoding="utf-8") as f:
                        out[name] = json.load(f)
                else:
                    out[name] = {"status": "NOT_YET_TRAINED"}
            self._send_json(200, out)
            return

        if path == "/api/v1/datasets/benchmarks":
            self._send_json(200, dataset_hub.get_dataset_inventory())
            return

        if path == "/api/v1/training/status":
            self._send_json(200, telemetry_streamer.get_status())
            return

        if path == "/api/v1/models/registry":
            # Read-only: the dashboard polls this, so it must not rewrite files
            # on every call. Exporting is /api/v1/models/export-edge-spec's job.
            spec_path = find_or_export_artifact("road_shield_open_model_spec.json")
            header_path = find_or_export_artifact("road_shield_edge_inference.h")
            with open(spec_path, "r", encoding="utf-8") as fh:
                spec = json.load(fh)
            self._send_json(200, {
                "spec_json_path": spec_path,
                "c_header_path": header_path,
                "models_exported": list(spec.get("models", {}).keys()),
            })
            return

        if path == "/api/v1/ledger/defects":
            defects = fleet_dedup_engine.get_all_deduplicated_defects()
            enriched = []
            total_tonnage = 0.0
            total_cost = 0.0
            for d in defects:
                area = float(d.get("area_m2", 0.0))
                materials = ipm_engine.estimate_repair_materials(area, depth_cm=6.0) if area > 0 else None
                tonnage = materials["required_mass_tonnes"] if materials else 0.0
                cost = materials["total_cost_inr"] if materials else 0.0
                total_tonnage += tonnage
                total_cost += cost
                enriched.append({
                    "defect_id": d["defect_id"],
                    "distress_type": d["defect_class"],
                    "lat": d["lat"],
                    "lng": d["lon"],
                    "surface_area_m2": area,
                    "confirmations": d["confirmation_count"],
                    "is_verified_hotspot": d["is_verified_hotspot"],
                    "estimated_repair_tonnes": round(tonnage, 3),
                    "estimated_repair_inr": round(cost, 2),
                    "address": d.get("address"),
                })
            self._send_json(200, {
                "total_active_defects": len(enriched),
                "total_morth_tonnage_tonnes": round(total_tonnage, 3),
                "total_budget_inr": round(total_cost, 2),
                "defects": enriched,
            })
            return

        if path == "/api/v1/vision/curated-videos":
            self._send_json(200, media_engine.get_video_catalog())
            return

        if path == "/api/v1/models/export-edge-spec":
            exp_res = edge_exporter.export_all_to_open_spec(output_dir=WRITABLE_DIR)
            self._send_json(200, {
                "status": "SUCCESS_EXPORTED",
                "open_spec_json": exp_res["spec_json_path"],
                "c_header_library": exp_res["c_header_path"],
                "models_exported": exp_res["models_exported"],
                "export_latency_ms": round((time.time() - t0) * 1000.0, 3),
            })
            return

        if path == "/api/v1/models/download-c-header":
            c_path = find_or_export_artifact("road_shield_edge_inference.h")
            if c_path:
                with open(c_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/x-c")
                self.send_header("Content-Disposition", 'attachment; filename="road_shield_edge_inference.h"')
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
                return
            self._send_json(404, {"error": "road_shield_edge_inference.h not found"})
            return

        if path == "/api/v1/models/download-neural-spec":
            json_path = find_or_export_artifact("road_shield_open_model_spec.json")
            if json_path:
                with open(json_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", 'attachment; filename="road_shield_open_model_spec.json"')
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
                return
            self._send_json(404, {"error": "road_shield_open_model_spec.json not found"})
            return

        # ---------------- Automotive exports ----------------
        if path == "/api/v1/automotive/can-stream":
            scenario = self.path.split("scenario=")[1].split("&")[0] if "scenario=" in self.path else "highway_pothole"
            snapshot = automotive_telematics.get_simulated_telemetry_snapshot(scenario)
            rl_decision = rl_agent.evaluate_telemetry_state(
                hazard_class_id=snapshot["hazard_class_id"],
                confidence=0.96,
                distance_m=snapshot["hazard_distance_m"],
                vehicle_speed_kmh=snapshot["vehicle_speed_kmh"],
                surface_friction_mu=snapshot["friction_mu"],
                pothole_depth_mm=snapshot["pothole_depth_mm"],
                imu_z_shock_ms2=snapshot["imu_z_shock_ms2"],
            )
            can_frame = automotive_telematics.generate_adas_can_packet(
                rl_decision=rl_decision,
                hazard_class_id=snapshot["hazard_class_id"],
                ttc_sec=rl_decision["telemetry_metrics"]["time_to_collision_sec"],
                speed_kmh=snapshot["vehicle_speed_kmh"],
            )
            self._send_json(200, {
                "telemetry": snapshot,
                "rl_decision": rl_decision,
                "can_frame": can_frame,
                "timestamp_ms": int(time.time() * 1000),
            })
            return

        if path == "/api/v1/automotive/export-dbc":
            dbc_file = os.path.join(CKPT_DIR, "road_shield_can_spec.dbc")
            if not os.path.exists(dbc_file):
                dbc_file = automotive_telematics.generate_can_dbc()
            with open(dbc_file, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="road_shield_can_spec.dbc"')
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/api/v1/automotive/export-ecu-header":
            header_file = os.path.join(CKPT_DIR, "road_shield_automotive_ecu.h")
            if not os.path.exists(header_file):
                header_file = automotive_telematics.generate_cpp_ecu_header()
            with open(header_file, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="road_shield_automotive_ecu.h"')
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
            return

        self._send_json(404, {"error": f"Endpoint {path} not found"})

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------
    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read_json_body()
        t0 = time.time()

        # ---------------- Google Maps (POST) ----------------
        if path == "/api/v1/maps/config":
            self._send_json(200, google_maps_service.set_api_key(body.get("api_key", "")))
            return

        if path == "/api/v1/maps/directions":
            origin_lat = float(body.get("origin_lat", 12.9725))
            origin_lng = float(body.get("origin_lng", 77.5955))
            dest_lat = float(body.get("dest_lat", 12.9780))
            dest_lng = float(body.get("dest_lng", 77.6020))
            avoid_defects = bool(body.get("avoid_defects", False))
            known_defects = fleet_dedup_engine.get_all_deduplicated_defects()
            dirs = google_maps_service.get_directions(origin_lat, origin_lng, dest_lat, dest_lng, avoid_defects=avoid_defects, known_defects=known_defects)
            self._send_json(200, dirs)
            return

        if path == "/api/v1/maps/geocode":
            self._send_json(200, google_maps_service.geocode(body.get("query", body.get("address", ""))))
            return

        if path == "/api/v1/maps/reverse-geocode":
            lat = float(body.get("lat", body.get("latitude", 12.9716)))
            lon = float(body.get("lon", body.get("lng", body.get("longitude", 77.5946))))
            self._send_json(200, google_maps_service.reverse_geocode(lat, lon))
            return

        if path == "/api/v1/maps/elevation":
            lat = float(body.get("lat", body.get("latitude", 12.9716)))
            lon = float(body.get("lon", body.get("lng", body.get("longitude", 77.5946))))
            self._send_json(200, google_maps_service.get_elevation(lat, lon))
            return

        # ----------------------------------------------------------------------
        # Vision distress classification (Model M1) - real image in, real class out
        # ----------------------------------------------------------------------
        if path in ("/api/v1/detect/vision", "/api/v1/vision/predict"):
            img_b64 = body.get("image_base64") or body.get("image_path")
            if not img_b64:
                self._send_json(400, {
                    "error": "Missing image_base64 or image_path. This endpoint classifies a real image crop - "
                             "it no longer accepts a 'preferred_class' shortcut or a synthetic feature vector.",
                })
                return
            if not vision_model.is_ready:
                self._send_json(503, {"error": "Vision model not trained yet - run training/train_vision.py."})
                return
            try:
                img_np = cv_detector.decode_image(img_b64)
                pred = vision_model.predict_image(img_np)
            except Exception as e:
                self._send_json(500, {"error": f"Failed to classify image: {str(e)}"})
                return

            u_min, v_min, w_px, h_px = 0, 0, img_np.shape[1], img_np.shape[0]
            ground_area = ipm_engine.calculate_surface_area_sqm(u_min, v_min, w_px, h_px) if pred["class_id"] in (1, 2, 3) else 0.0

            self._send_json(200, {
                "model": "VisionDistressNet",
                "class_id": pred["class_id"],
                "distress_name": pred["class_name"],
                "confidence": pred["confidence"],
                "shannon_entropy_bits": pred["shannon_entropy_bits"],
                "epistemic_uncertainty_rating": pred["uncertainty_rating"],
                "astm_d6433_severity": pred["astm_d6433_severity"],
                "irc_standard_specification": pred["irc_standard_specification"],
                "top3_ranked_predictions": pred["top3_ranked_predictions"],
                "probabilities": pred["all_class_probabilities"],
                "is_distress": pred["class_id"] in (1, 2, 3),
                "ground_area_m2_whole_frame": round(ground_area, 3),
                "inference_latency_ms": round((time.time() - t0) * 1000.0, 3),
            })
            return

        # ----------------------------------------------------------------------
        # IMU telemetry classification (Model M4)
        # ----------------------------------------------------------------------
        if path == "/api/v1/telemetry/imu":
            data_source = "caller_supplied_real_series"
            if "raw_series" in body:
                raw = np.array(body["raw_series"], dtype=np.float32)
                if raw.ndim == 2:
                    raw = np.expand_dims(raw, axis=0)
            else:
                window, true_label = sample_real_imu_window(split="val", pothole_only=bool(body.get("simulate_shock", True)))
                if window is None:
                    self._send_json(503, {"error": "No IMU dataset found under datasets/04_mobile_imu_telemetry_100hz."})
                    return
                raw = window[np.newaxis, :, :]
                data_source = "real_historical_sample_from_val_dataset_not_live_telemetry"

            if not imu_model.is_ready:
                self._send_json(503, {"error": "IMU model not trained yet - run training/train_imu.py."})
                return

            preds, pothole_conf, probs = imu_model.predict(raw)
            cls_id = int(preds[0])
            delta_z = float(np.max(raw[0, :, 2]) - np.min(raw[0, :, 2]))

            self._send_json(200, {
                "model": "IMUShockClassifier",
                "data_source": data_source,
                "class_id": cls_id,
                "shock_classification": IMUShockClassifier.CLASS_NAMES[cls_id],
                "pothole_shock_probability": round(float(pothole_conf[0]), 4),
                "peak_delta_z_ms2": round(delta_z, 2),
                "probabilities": {name: round(float(probs[0][i]), 4) for i, name in enumerate(IMUShockClassifier.CLASS_NAMES)},
                "inference_latency_ms": round((time.time() - t0) * 1000.0, 3),
            })
            return

        # ----------------------------------------------------------------------
        # Bayesian dual-sensor fusion (Model M5)
        # ----------------------------------------------------------------------
        if path == "/api/v1/fusion/gate":
            p_vision = float(body.get("vision_pothole_prob", body.get("p_visual", 0.5)))
            accel_delta_z = float(body.get("peak_delta_z_ms2", body.get("delta_z_ms2", 0.0)))
            # Without a full 100-sample IMU window there's no real classifier
            # output to use, so an explicit p_imu_shock is required unless
            # the caller only wants the peak-shock heuristic below.
            p_imu = body.get("p_imu_shock")
            p_imu = float(p_imu) if p_imu is not None else (0.9 if accel_delta_z >= 3.5 else 0.05)

            fusion_res = bayesian_gate.fuse(p_visual=p_vision, p_imu_shock=p_imu, delta_z_ms2=accel_delta_z)
            fusion_res["model"] = "BayesianFusionGate"
            fusion_res["latency_ms"] = round((time.time() - t0) * 1000.0, 3)
            self._send_json(200, fusion_res)
            return

        # ----------------------------------------------------------------------
        # Civil volumetrics (Model M2 - real photogrammetry + material rates)
        # ----------------------------------------------------------------------
        if path == "/api/v1/civil/ipm-tonnage":
            area_m2 = float(body.get("area_m2", body.get("area_sqm", 1.8)))
            depth_cm = float(body.get("depth_cm", 6.5))
            mix_rate = float(body.get("mix_rate_inr_tonne", 7500.0))
            civil_res = ipm_engine.estimate_repair_materials(surface_area_sqm=area_m2, depth_cm=depth_cm, mix_rate_per_tonne_inr=mix_rate)
            civil_res["model"] = "IPMHomographyEngine"
            civil_res["latency_ms"] = round((time.time() - t0) * 1000.0, 3)
            self._send_json(200, civil_res)
            return

        # ----------------------------------------------------------------------
        # Forensic repair verification (Models M7/M8) - requires real before/after photos
        # ----------------------------------------------------------------------
        if path == "/api/v1/audit/verify-repair":
            before_b64 = body.get("before_image_base64")
            after_b64 = body.get("after_image_base64")
            dist_m = float(body.get("claimed_distance_m", 0.8))
            if not before_b64 or not after_b64:
                self._send_json(400, {
                    "error": "Missing before_image_base64 / after_image_base64. This audit compares two real "
                             "site photos - it no longer generates random noise images to fake a scenario.",
                })
                return
            try:
                before_rgb = cv_detector.decode_image(before_b64)
                after_rgb = cv_detector.decode_image(after_b64)
                before_gray = np.mean(before_rgb.astype(np.float32), axis=2)
                after_gray = np.mean(after_rgb.astype(np.float32), axis=2)
                audit_res = texture_auditor.verify_repair(before_gray, after_gray, embedder=forensic_embedder, claimed_dist_m=dist_m)
            except Exception as e:
                self._send_json(500, {"error": f"Failed to audit repair photos: {str(e)}"})
                return
            audit_res["model"] = "ForensicAuditEngine"
            audit_res["latency_ms"] = round((time.time() - t0) * 1000.0, 3)
            self._send_json(200, audit_res)
            return

        # ----------------------------------------------------------------------
        # MoRTH cryptographic work-order dispatch (Model M10)
        # ----------------------------------------------------------------------
        if path == "/api/v1/dispatch/work-order":
            work_order = dispatch_agent.generate_work_order(
                corridor_id=str(body.get("corridor_id", body.get("highway", "NH-44"))),
                latitude=float(body.get("latitude", body.get("lat", 28.7041))),
                longitude=float(body.get("longitude", body.get("lng", body.get("lon", 77.1025)))),
                distress_class=str(body.get("distress_class", body.get("distress_type", "Pothole Cavity"))),
                area_sqm=float(body.get("area_sqm", body.get("area_m2", 2.2))),
                depth_cm=float(body.get("depth_cm", 6.5)),
                pci_score=int(body.get("pci_score", 42)),
            )
            work_order["model"] = "MoRTHDispatchAgent"
            work_order["latency_ms"] = round((time.time() - t0) * 1000.0, 3)
            self._send_json(200, work_order)
            return

        if path == "/api/v1/dispatch/verify-seal":
            work_order = body.get("work_order", body)
            clean_wo = {k: v for k, v in work_order.items() if k not in ("model", "latency_ms")}
            is_valid = dispatch_agent.verify_work_order_seal(clean_wo)
            self._send_json(200, {
                "is_valid": is_valid,
                "work_order_id": clean_wo.get("work_order_id", "UNKNOWN"),
                "status": "SEAL_VERIFIED_AUTHENTIC" if is_valid else "CORRUPTED_OR_TAMPERED",
            })
            return

        # ----------------------------------------------------------------------
        # ASTM D6433 PCI (real deduct-value formula, not a trained regressor)
        # ----------------------------------------------------------------------
        if path == "/api/v1/pci/predict":
            result = pci_model.compute(
                crack_density_pct=float(body.get("crack_density_pct", 0.0)),
                crack_severity=str(body.get("crack_severity", "LOW")),
                pothole_count=int(body.get("pothole_count", 0)),
                pothole_density_pct=float(body.get("pothole_density_pct", 0.0)),
                pothole_severity=str(body.get("pothole_severity", "MEDIUM")),
                rutting_mm=float(body.get("rutting_mm", 0.0)),
                iri_roughness=float(body.get("iri_roughness", 0.0)),
                age_yr=float(body.get("age_years", body.get("age_yr", 0.0))),
            )
            result["model"] = "PavementConditionIndexEngine"
            result["astm_standard"] = "ASTM D6433-20 (deduct-value method; see module docstring for the curve-approximation note)"
            result["inference_latency_ms"] = round((time.time() - t0) * 1000.0, 3)
            self._send_json(200, result)
            return

        # ----------------------------------------------------------------------
        # Pavement deterioration lifecycle forecast (Model M_DEGRADE)
        # ----------------------------------------------------------------------
        if path == "/api/v1/forecast/deterioration":
            roi_res = degrade_model.predict_lifecycle_roi(
                init_area_m2=float(body.get("initial_area_m2", 1.5)),
                depth_cm=float(body.get("depth_cm", 6.0)),
                esal_trucks=float(body.get("esal_trucks", 6000)),
                rain_mm=float(body.get("rain_mm", 650.0)),
                age_yr=float(body.get("age_years", 3.5)),
            )
            roi_res["model"] = "PavementDeteriorationForecaster"
            roi_res["latency_ms"] = round((time.time() - t0) * 1000.0, 3)
            self._send_json(200, roi_res)
            return

        # ----------------------------------------------------------------------
        # Launch real training (vision + IMU) - runs the actual scikit-learn fits
        # ----------------------------------------------------------------------
        if path == "/api/v1/training/launch":
            if IS_SERVERLESS:
                self._send_json(503, {
                    "status": "TRAINING_UNAVAILABLE_ON_SERVERLESS",
                    "error": "Training needs the datasets/ folder and a writable checkpoints/ folder, neither of "
                             "which exists on this serverless deployment. Train locally with "
                             "`python -m training.train_mega_suite`, commit checkpoints/, and redeploy.",
                })
                return
            launch_res = run_training_suite(async_mode=True)
            self._send_json(200, launch_res)
            return

        # ----------------------------------------------------------------------
        # Real-photo demo classification via the deep pipeline
        # ----------------------------------------------------------------------
        if path == "/api/v1/vision/analyze-photo":
            class_id = body.get("class_id")
            preset = media_engine.get_photo_preset(class_id=class_id)
            if preset is None:
                self._send_json(503, {"error": "No labeled photos found under datasets/*/real_images."})
                return
            try:
                result = deep_pipeline.audit_image(image_input=preset["image_path"], corridor_id=f"Demo preset ({preset['class_name']})")
            except Exception as e:
                self._send_json(500, {"error": f"Failed to analyze preset photo: {str(e)}"})
                return
            result["preset_ground_truth_class"] = preset["class_name"]
            result["preset_image_path"] = preset["image_path"]
            result["latency_ms"] = round((time.time() - t0) * 1000.0, 3)
            self._send_json(200, result)
            return

        # ----------------------------------------------------------------------
        # Analyze an arbitrary uploaded field photograph via the deep pipeline
        # ----------------------------------------------------------------------
        if path == "/api/v1/vision/analyze-custom-photo":
            img_b64 = body.get("image_base64")
            filename = body.get("filename", "")
            highway = body.get("highway", "Custom Field Survey Location")
            if filename and filename not in highway:
                highway = f"{highway} {filename}"
            if not img_b64:
                self._send_json(400, {"error": "Missing image_base64 payload"})
                return
            try:
                analysis = deep_pipeline.audit_image(image_input=img_b64, corridor_id=highway)
                analysis["latency_ms"] = round((time.time() - t0) * 1000.0, 3)
                self._send_json(200, analysis)
            except Exception as e:
                self._send_json(500, {"error": f"Failed to analyze image: {str(e)}"})
            return

        # ----------------------------------------------------------------------
        # Analyze one video-stream frame + real IoU tracking across frames
        # ----------------------------------------------------------------------
        if path in ("/api/v1/vision/process-video-frame", "/api/v1/vision/analyze-custom-frame"):
            frame_b64 = body.get("frame_base64") or body.get("image_base64")
            frame_idx = int(body.get("frame_idx", 0))
            session_id = body.get("session_id", "default_stream")
            reset_tracker = bool(body.get("reset_tracker", False))

            if not frame_b64:
                self._send_json(400, {
                    "error": "Missing frame_base64. This project ships no real dashcam video dataset "
                             "(see /api/v1/vision/curated-videos), so this endpoint tracks real frames "
                             "you submit rather than a fabricated clip catalog.",
                })
                return

            try:
                tracker = get_session_tracker(session_id, reset=reset_tracker)
                analysis = deep_pipeline.audit_image(image_input=frame_b64, corridor_id=body.get("clip_id", "uploaded_stream"))
                adapted_dets = [
                    {
                        "bbox_normalized": d["bbox_normalized"],
                        "class_name": d["class_name"],
                        "confidence": d["confidence"],
                        "is_distress": d.get("is_distress", False),
                        "distance_meters": d.get("distance_meters", 0.0),
                        "surface_area_m2": d.get("surface_area_m2", 0.0),
                    }
                    for d in analysis.get("all_detections", [])
                    if d.get("bbox_normalized") is not None
                ]
                tracking_res = tracker.update(adapted_dets, frame_idx)
                self._send_json(200, {
                    "frame_idx": frame_idx,
                    "tracked_detections": tracking_res["tracked_detections"],
                    "anti_double_counting_metrics": {
                        "active_tracks_count": tracking_res["active_tracks_count"],
                        "total_unique_potholes_counted": tracking_res["total_unique_potholes_counted"],
                        "total_unique_cracks_counted": tracking_res["total_unique_cracks_counted"],
                    },
                    "latency_ms": round((time.time() - t0) * 1000.0, 3),
                })
            except Exception as e:
                self._send_json(500, {"error": f"Failed to analyze video frame: {str(e)}"})
            return

        # ----------------------------------------------------------------------
        # Active-feedback logging (real, durable log for the next retraining run)
        # ----------------------------------------------------------------------
        if path == "/api/v1/training/active-feedback":
            global active_feedback_counter
            true_cls = int(body.get("true_class", 0))
            img_b64 = body.get("image_base64")
            notes = body.get("notes", "")

            if true_cls < 0 or true_cls >= len(VisionDistressNet.CLASS_NAMES):
                self._send_json(400, {"error": f"true_class must be 0-{len(VisionDistressNet.CLASS_NAMES) - 1}"})
                return
            if not img_b64:
                self._send_json(400, {"error": "Missing image_base64 - a correction needs a real image to be useful later."})
                return

            active_feedback_counter += 1
            feedback_id = f"AFB-{int(time.time())}-{active_feedback_counter:04d}"
            record = {
                "feedback_id": feedback_id,
                "true_class_id": true_cls,
                "true_class_name": VisionDistressNet.CLASS_NAMES[true_cls],
                "notes": notes,
                "timestamp_unix": int(time.time()),
            }
            # Store the correction for the next real retraining pass rather than
            # claiming an in-place gradient update: SVC/RandomForest pipelines
            # don't support a meaningful single-sample online update, so
            # pretending one just happened (as the old endpoint did) would be
            # fabricating a result.
            try:
                os.makedirs(WRITABLE_DIR, exist_ok=True)
                with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                logged = True
            except Exception:
                logged = False

            current_pred = None
            if vision_model.is_ready:
                try:
                    img_np = cv_detector.decode_image(img_b64)
                    current_pred = vision_model.predict_image(img_np)
                except Exception:
                    pass

            self._send_json(200, {
                "status": "FEEDBACK_LOGGED_FOR_NEXT_RETRAINING_RUN" if logged else "FEEDBACK_RECEIVED_BUT_LOG_WRITE_FAILED",
                "feedback_id": feedback_id,
                "target_class": record["true_class_name"],
                "current_model_prediction": current_pred["class_name"] if current_pred else None,
                "current_model_confidence": current_pred["confidence"] if current_pred else None,
                "total_active_feedback_samples": active_feedback_counter,
                "note": "This correction is appended to checkpoints/active_feedback_log.jsonl and is not yet "
                        "reflected in the loaded model - trigger /api/v1/training/launch to retrain.",
                "latency_ms": round((time.time() - t0) * 1000.0, 3),
            })
            return

        # ----------------------------------------------------------------------
        # Full deep-inference pipeline audit
        # ----------------------------------------------------------------------
        if path == "/api/v1/pipeline/deep-audit":
            image_input = body.get("image_base64") or body.get("image_path")
            corridor = body.get("corridor_id", "NH-44")
            filename = body.get("filename", "")
            if filename and filename not in corridor:
                corridor = f"{corridor} {filename}"
            lat = float(body.get("latitude", body.get("lat", 28.7041)))
            lng = float(body.get("longitude", body.get("lng", 77.1025)))
            chainage = float(body.get("chainage_km", 108.4))
            traffic_esal = float(body.get("traffic_esal", 7500))
            rain_mm = float(body.get("rain_mm", 650.0))
            age_yr = float(body.get("age_years", 3.5))
            imu_series = body.get("imu_series")  # only used if the caller supplies a real (100,3) window

            if not image_input:
                default_pothole = os.path.join(ENGINE_ROOT, "datasets", "02_kaggle_pothole_600", "real_images", "1014628_RS_386_386RS124739_30065_RAW.jpg")
                if os.path.exists(default_pothole):
                    image_input = default_pothole
                else:
                    self._send_json(400, {"error": "Missing image_base64 or image_path payload"})
                    return

            try:
                audit_result = deep_pipeline.audit_image(
                    image_input=image_input,
                    corridor_id=corridor,
                    latitude=lat,
                    longitude=lng,
                    chainage_km=chainage,
                    imu_series=imu_series,
                    traffic_esal=traffic_esal,
                    rain_mm=rain_mm,
                    pavement_age_yr=age_yr,
                )
                self._send_json(200, audit_result)
            except Exception as e:
                self._send_json(500, {"error": f"Deep pipeline audit failed: {str(e)}"})
            return

        if path == "/api/v1/pipeline/deep-audit-batch":
            dir_path = body.get("directory_path")
            max_samples = int(body.get("max_samples", 10))
            corridor = body.get("corridor_id", "NH-44")
            if not dir_path or not os.path.isdir(dir_path):
                dir_path = os.path.join(ENGINE_ROOT, "datasets", "02_kaggle_pothole_600", "real_images")
            if not os.path.isdir(dir_path):
                self._send_json(503, {"error": "No image folder available for a batch audit on this deployment "
                                               "(datasets/ is not deployed). Use /api/v1/pipeline/deep-audit with your own photo."})
                return
            try:
                batch_result = deep_pipeline.process_batch(image_source=dir_path, max_samples=max_samples, corridor_id=corridor)
                self._send_json(200, batch_result)
            except Exception as e:
                self._send_json(500, {"error": f"Batch deep audit failed: {str(e)}"})
            return

        # ----------------------------------------------------------------------
        # Urban traffic congestion (real IRC:106-1990 PCU formula on caller-supplied counts)
        # ----------------------------------------------------------------------
        if path == "/api/v1/traffic/analyze":
            counts = body.get("vehicle_counts", {"Car": 16, "City Bus": 4, "Heavy Truck": 2, "Two-Wheeler": 10})
            capacity = body.get("road_capacity", 35)
            cong = traffic_net.calculate_congestion_index(counts, road_capacity=capacity)
            self._send_json(200, {
                "status": "SUCCESS",
                "vehicle_counts": counts,
                "congestion_analytics": cong,
                "bottleneck_identified": cong["congestion_index"] >= 0.80,
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            return

        # ----------------------------------------------------------------------
        # Pedestrian-situation risk (deterministic rules over caller-supplied facts)
        # ----------------------------------------------------------------------
        if path == "/api/v1/pedestrian/detect":
            ped_count = body.get("pedestrian_count", 0)
            is_school_zone = body.get("is_school_zone", False)
            is_outside_zebra = body.get("is_outside_crosswalk", False)

            risk_level = "LOW"
            if is_school_zone and is_outside_zebra and ped_count > 0:
                risk_level = "CRITICAL_CHILD_CROSSING_HAZARD"
            elif is_outside_zebra and ped_count > 0:
                risk_level = "MODERATE_JAYWALKING_ALERT"

            self._send_json(200, {
                "status": "SUCCESS",
                "pedestrians_detected": ped_count,
                "is_school_zone": is_school_zone,
                "is_outside_crosswalk": is_outside_zebra,
                "vulnerable_situation_alert": risk_level != "LOW",
                "alert_level": risk_level,
                "recommended_bus_action": "AUTONOMOUS_SLOWDOWN_CHIME" if risk_level == "CRITICAL_CHILD_CROSSING_HAZARD" else "MAINTAIN_VIGILANCE",
            })
            return

        # ----------------------------------------------------------------------
        # Fleet defect ingestion / deduplication
        # ----------------------------------------------------------------------
        if path == "/api/v1/fleet/report-defect":
            bus_id = body.get("bus_id", "BUS-UNKNOWN")
            lat = float(body.get("lat", 12.9716))
            lon = float(body.get("lon", 77.5946))
            cls_name = body.get("defect_class", "Pothole Cavity")
            pci = float(body.get("severity_pci", 42.0))
            area = float(body.get("area_m2", 1.85))
            res = fleet_dedup_engine.ingest_fleet_detection(bus_id, lat, lon, cls_name, pci, area)
            self._send_json(200, res)
            return

        # ----------------------------------------------------------------------
        # ALPR / rash-driving incident report (real kinematics + real OCR when an image is given)
        # ----------------------------------------------------------------------
        if path == "/api/v1/incidents/alpr":
            bus_id = body.get("bus_id", "BUS-KA01-204")
            gps = body.get("gps", {"lat": 12.9780, "lng": 77.6020})
            track_history = body.get("track_history") or [
                {"timestamp": 0.0, "bbox": [200, 150, 80, 60]},
                {"timestamp": 0.2, "bbox": [180, 160, 140, 110]},
                {"timestamp": 0.4, "bbox": [140, 175, 260, 210]},
            ]
            vehicle_crop = None
            if body.get("vehicle_image_base64"):
                try:
                    vehicle_crop = cv_detector.decode_image(body["vehicle_image_base64"])
                except Exception:
                    vehicle_crop = None
            incident = alpr_tracker.generate_incident_alert(bus_id, gps, track_history, vehicle_crop_rgb=vehicle_crop)
            self._send_json(200, incident)
            return

        # ----------------------------------------------------------------------
        # Automotive ADAS policy + CAN + fusion (Model RL-1 / MM-1)
        # ----------------------------------------------------------------------
        if path == "/api/v1/automotive/rl-action":
            h_cls = int(body.get("hazard_class_id", 2))
            conf = float(body.get("confidence", 0.95))
            dist = float(body.get("distance_m", 35.0))
            spd = float(body.get("vehicle_speed_kmh", 70.0))
            friction = float(body.get("surface_friction_mu", 0.75))
            depth = float(body.get("pothole_depth_mm", 45.0 if h_cls == 2 else 0.0))
            shock = float(body.get("imu_z_shock_ms2", 4.2 if h_cls == 2 else 0.1))
            lat_margin = float(body.get("lateral_lane_margin_m", 1.2))
            wet = bool(body.get("is_wet", False))

            res = deep_pipeline.evaluate_automotive_incident(
                hazard_class_id=h_cls,
                confidence=conf,
                distance_m=dist,
                vehicle_speed_kmh=spd,
                surface_friction_mu=friction,
                pothole_depth_mm=depth,
                imu_z_shock_ms2=shock,
                lateral_lane_margin_m=lat_margin,
                is_wet=wet,
            )
            res["model"] = "AutomotiveADASPolicyAgent"
            res["latency_ms"] = round((time.time() - t0) * 1000.0, 3)
            self._send_json(200, res)
            return

        if path == "/api/v1/automotive/multimodal-fusion":
            n_classes = len(VisionDistressNet.CLASS_NAMES) + 1
            if "vision_probabilities" in body and len(body["vision_probabilities"]) in (n_classes - 1, n_classes):
                vision_probs = np.array(body["vision_probabilities"], dtype=np.float64)
            elif body.get("image_base64") and vision_model.is_ready:
                img_np = cv_detector.decode_image(body["image_base64"])
                pred = vision_model.predict_image(img_np)
                vision_probs = np.array([pred["all_class_probabilities"][n] for n in VisionDistressNet.CLASS_NAMES], dtype=np.float64)
            else:
                self._send_json(400, {
                    "error": f"Provide either vision_probabilities (length {n_classes - 1} or {n_classes}) or image_base64 "
                             "with the vision model trained. This endpoint no longer fuses random placeholder tensors "
                             "for imaginary LiDAR/CAN-bus sensors this project doesn't have.",
                })
                return

            imu_pothole_prob = body.get("imu_pothole_prob")
            imu_shock_ms2 = body.get("imu_shock_ms2")
            if imu_pothole_prob is None and imu_shock_ms2 is not None:
                imu_pothole_prob = float(np.clip(float(imu_shock_ms2) / 8.0, 0.0, 1.0))

            fusion_res = multimodal_net.fuse(vision_probs, imu_pothole_prob=imu_pothole_prob, imu_shock_ms2=imu_shock_ms2)
            fusion_res["model"] = "MultimodalLateFusionNet"
            fusion_res["latency_ms"] = round((time.time() - t0) * 1000.0, 3)
            self._send_json(200, fusion_res)
            return

        self._send_json(404, {"error": f"Endpoint {path} not found"})


# ==============================================================================
# SERVER ENTRYPOINT
# ==============================================================================
def start_server(port=8000, host="0.0.0.0"):
    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, RoadShieldAPIHandler)
    print(f"\n{'=' * 70}")
    print(f"ROAD-SHIELD AI Engine running on http://127.0.0.1:{port}")
    print(f"CORS: enabled (all origins)")
    print(f"{'=' * 70}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[AI Server] Shutting down...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_server(port=port)
