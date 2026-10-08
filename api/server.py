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
import glob
import json
import time
import base64
import io
import socketserver
import queue
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
try:
    from training.mega_pipeline import run_training_suite, telemetry_streamer
except Exception:
    run_training_suite = None
    telemetry_streamer = None
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
# Explicit override: the test suite points this at a temporary directory so it
# never writes work orders or sightings into the real fleet ledger, and an
# operator can put the ledger on a mounted volume without moving checkpoints/.
if os.environ.get("ROAD_SHIELD_WRITABLE_DIR"):
    WRITABLE_DIR = os.environ["ROAD_SHIELD_WRITABLE_DIR"]
    os.makedirs(WRITABLE_DIR, exist_ok=True)

print("[AI Server] Loading trained models from:", CKPT_DIR)

from models.deep_vision_net import load_best_vision_model
vision_model, VISION_BACKEND = load_best_vision_model(CKPT_DIR)

from models.imu_shock_classifier import load_served_imu_model
imu_model, imu_backend = load_served_imu_model(CKPT_DIR)
if imu_model is None:
    imu_model = IMUShockClassifier()
print("  ✓ IMU shock classifier", f"loaded ({imu_backend})" if imu_model.is_ready else "NOT TRAINED YET (run training/train_imu.py)")

bayesian_gate = BayesianFusionGate(prior_pothole_prob=0.05, decision_threshold_log_odds=1.8)
# Standalone IPM for the /civil/ipm-tonnage endpoint. Built from the default
# calibration profile rather than repeating the constants here, so there is one
# place where an assumed mount is defined and it is labelled as assumed.
from models.camera_calibration import DEFAULT_PROFILE as _DEFAULT_CALIB
ipm_engine = IPMHomographyEngine.from_calibration(_DEFAULT_CALIB)
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

# The ledger is durable: a municipality's repair backlog cannot live in a
# process's memory. On a serverless filesystem the database goes to the
# writable tmp directory and is therefore per-instance, which is reported
# honestly by /api/v1/fleet/telemetry rather than hidden.
from pipeline.defect_store import DefectStore
defect_store = DefectStore(os.path.join(WRITABLE_DIR, "road_shield.db"))
fleet_dedup_engine = FleetDeduplicationEngine(proximity_threshold_meters=8.0,
                                              store=defect_store)

# Live map feed (server-sent events) and the encrypted link from the bus agents (edge/).
from pipeline.live_events import live_events, sse_frame
from pipeline.edge_ingest import EdgeIngest
edge_ingest = EdgeIngest(os.path.join(WRITABLE_DIR, "road_shield.db"), fleet_dedup_engine, live_events)
SSE_MAX_SECONDS = 1800

# Repair lifecycle (issue -> repair -> verified by the fleet) and alerts.
from pipeline.works import WorkOrders, WorkflowError, STATES as ORDER_STATES
from pipeline.alerts import Alerts
works = WorkOrders(os.path.join(WRITABLE_DIR, "road_shield.db"), fleet_dedup_engine, dispatch_agent, live_events)
alerts = Alerts(live_events)

# MLOps: model registry, production monitoring, shadow testing, active learning and the traffic estimate
# (mlops/, pipeline/traffic.py, api/mlops_routes.py).
from api import mlops_routes


def _reload_pipeline():
    """Load every model again from checkpoints/ (after a promotion or rollback) and swap them in: the pipeline
    and the stand-alone models the single-model endpoints use."""
    global deep_pipeline, vision_model, VISION_BACKEND, imu_model, imu_backend, pci_model, degrade_model
    new = DeepInferencePipeline(CKPT_DIR)
    vm, vb = load_best_vision_model(CKPT_DIR)
    im, ib = load_served_imu_model(CKPT_DIR)
    pm, dm = PCIRegressorNet(), PavementDeteriorationForecaster()
    deep_pipeline, vision_model, VISION_BACKEND = new, vm, vb
    imu_model, imu_backend = (im, ib) if im is not None else (IMUShockClassifier(), ib)
    pci_model, degrade_model = pm, dm
    return new


try:
    MLOPS = mlops_routes.MLOps(WRITABLE_DIR, CKPT_DIR, deep_pipeline, edge_ingest=edge_ingest, events=live_events,
                               alerts=alerts, reload_pipeline=_reload_pipeline)
except Exception as _e:
    print(f"[WARN] MLOps unavailable: {_e}")
    MLOPS = None


def _upload_source():
    """Where an uploaded photograph came from, for monitoring and active learning: an anonymous visitor to a
    public deployment is promised that nothing is kept."""
    return "public" if os.environ.get("ROAD_SHIELD_PUBLIC") == "1" else "api"


def _safe_priority(defect):
    # defined here, not with the other helpers: the demo seed below fires _on_sighting at import time
    from models import priority_index
    try:
        if MLOPS is not None:
            defect = MLOPS.traffic.enrich(defect)     # measured traffic per day where the fleet has counted it
        return priority_index.score_defect(defect) if defect.get("severity_pci") is not None else None
    except (ValueError, TypeError):
        return None


def _with_traffic(defects):
    """Ledger defects with the camera traffic estimate attached where a road cell has one."""
    if MLOPS is None:
        return defects
    return [MLOPS.traffic.enrich(d) for d in defects]


def _on_sighting(bus_id, result, at=None):
    works.on_sighting(result["defect_id"], bus_id, at=at)
    rec = next((d for d in fleet_dedup_engine.get_all_deduplicated_defects() if d["defect_id"] == result["defect_id"]), None)
    if rec is not None:
        alerts.check_defect(rec, _safe_priority(rec))


fleet_dedup_engine.listeners.append(_on_sighting)

# Citizen reports from the public Report page: pinned, and confirmed by a bus or an operator before the ledger.
from pipeline.citizen import CitizenReports, CitizenError
citizen = CitizenReports(os.path.join(WRITABLE_DIR, "road_shield.db"), fleet_dedup_engine, live_events)
fleet_dedup_engine.listeners.append(citizen.on_sighting)
edge_ingest.position_listeners.append(lambda bus, lat, lon, at=None: works.on_bus_position(bus, lat, lon, at=at))

# Operational guard rails: body limit, rate limit, Prometheus metrics, readiness (api/ops.py).
from api import ops
try:
    from api.openapi import spec as _openapi_spec
    _KNOWN_ROUTES = set(_openapi_spec()["paths"]) | {"/metrics", "/api/v1/ready"}
except Exception:
    _KNOWN_ROUTES = {"/metrics", "/api/v1/ready", "/api/v1/health"}
METRICS = ops.Metrics(_KNOWN_ROUTES)
LIMITER = ops.RateLimiter()


def _readiness():
    seg = getattr(deep_pipeline, "segmenter", None)
    return ops.readiness(vision_model.is_ready, bool(seg is not None and seg.is_ready),
                         imu_model.is_ready, defect_store is not None)

# Demo defects are OFF by default: five invented bus reports in a ledger that
# the dashboard presents as the city's repair backlog is fabricated data.
# Set ROAD_SHIELD_SEED_DEMO=1 to seed them for a demo (only when the ledger is
# empty - re-seeding a durable store would duplicate the fixtures).
SEED_DEMO = os.environ.get("ROAD_SHIELD_SEED_DEMO") == "1"
if SEED_DEMO and not fleet_dedup_engine.defect_registry:
    fleet_dedup_engine.ingest_fleet_detection("BUS-KA01-101", 12.9716, 77.5946, "Pothole Cavity", 42.0, 1.85, enrich_location=False)
    fleet_dedup_engine.ingest_fleet_detection("BUS-KA01-204", 12.97163, 77.59457, "Pothole Cavity", 38.0, 2.10, enrich_location=False)  # ~5 m away -> merges into the first, confirming it
    fleet_dedup_engine.ingest_fleet_detection("BUS-KA01-101", 12.9750, 77.5980, "Waterlogging / Flooding Hazard", 35.0, 5.20, enrich_location=False)
    fleet_dedup_engine.ingest_fleet_detection("BUS-KA01-308", 12.9680, 77.5910, "Missing Zebra Crossing", 60.0, 3.40, enrich_location=False)
    fleet_dedup_engine.ingest_fleet_detection("BUS-KA01-204", 12.9800, 77.6050, "Damaged Traffic Sign", 55.0, 0.80, enrich_location=False)
    print(f"  ✓ Fleet ledger seeded with 5 DEMO reports (ROAD_SHIELD_SEED_DEMO=1) "
          f"-> {defect_store.db_path}")
else:
    print(f"  ✓ Fleet ledger loaded: {len(fleet_dedup_engine.defect_registry)} defects "
          f"from {defect_store.db_path}")

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


# ------------------------------------------------------------------------------
# Static site
# ------------------------------------------------------------------------------
WEB_DIR = os.path.join(ENGINE_ROOT, "web")

# Clean URLs -> files in web/. Keeping this explicit rather than mapping any
# path to any file means the server cannot be talked into reading outside the
# web directory, which a naive static handler usually can.
PAGE_ROUTES = {
    "/": "index.html",
    "/inspect": "inspect.html",
    "/corridor": "corridor.html",
    "/report": "report.html",
    "/order": "order.html",
    "/mlops": "mlops.html",
    "/works": "works.html",
    "/models": "models.html",
    "/data": "data.html",
    "/system": "system.html",
    "/video": "video.html",
    "/architecture": "architecture.html",
    "/design": "design.html",
    "/impact": "impact.html",
    "/api-docs": "api-docs.html",
}

# Write endpoints that change shared state. When ROAD_SHIELD_API_KEY is set they
# require it (X-API-Key header, or Authorization: Bearer <key>); when it is not
# set - the default, and the demo - behaviour is unchanged. Analysis endpoints
# stay open: they read an image and return a result, and change nothing stored.
PROTECTED_POST = {
    "/api/v1/fleet/report-defect",
    "/api/v1/dispatch/work-order",
    "/api/v1/training/launch",
    "/api/v1/training/active-feedback",
    "/api/v1/incidents/alpr",
    "/api/v1/maps/config",
    "/api/v1/works/orders",
    "/api/v1/works/status",
    "/api/v1/video/ingest",            # writes sightings to the ledger, which can reopen repaired orders
    "/api/v1/citizen/review",
} | mlops_routes.PROTECTED_POST


# Public demo (ROAD_SHIELD_PUBLIC=1, set by deploy/huggingface/Dockerfile): anyone on the internet can reach
# the engine, so
#   * the write endpoints above are locked - with a random key if the operator set none - so a visitor
#     cannot add defects to the ledger, launch training or change settings;
#   * no request can name a file on the server outside datasets/ (image paths, video paths, batch folders):
#     uploads come in as base64, and the bundled photographs stay usable.
# Analysis endpoints stay open and store nothing.
PUBLIC = os.environ.get("ROAD_SHIELD_PUBLIC") == "1"
if PUBLIC and not os.environ.get("ROAD_SHIELD_API_KEY"):
    import secrets as _secrets
    os.environ["ROAD_SHIELD_API_KEY"] = _secrets.token_urlsafe(32)
    print("  ✓ Public demo: write endpoints locked (no ROAD_SHIELD_API_KEY was set, so a random one is in use)")
DATASETS_ROOT = os.path.realpath(os.path.join(ENGINE_ROOT, "datasets"))
PATH_KEYS = ("image_base64", "image_path", "before_image_base64", "after_image_base64", "before_base64",
             "after_base64", "frame_base64", "vehicle_image_base64", "video_path", "path", "directory_path")


def _server_path_refused(body):
    """In public mode: the first request field that names an existing server file outside datasets/, else None."""
    if not PUBLIC or not isinstance(body, dict):
        return None
    for k in PATH_KEYS:
        v = body.get(k)
        if not isinstance(v, str) or not v or len(v) > 4096:
            continue
        try:
            real = os.path.realpath(v if os.path.isabs(v) else os.path.join(ENGINE_ROOT, v))
            exists = os.path.exists(v) or os.path.exists(real)
        except (OSError, ValueError):
            continue
        if exists and not (real == DATASETS_ROOT or real.startswith(DATASETS_ROOT + os.sep)):
            return k
    return None


def _client_key(headers, client_address):
    """
    Who a request is from, for the rate limit. Behind a Cloudflare tunnel every request arrives from
    127.0.0.1, so in public mode the visitor address Cloudflare adds (CF-Connecting-IP, which Cloudflare
    sets itself) is used - but only for requests that come from this machine (the tunnel). A request
    straight over the network keeps its socket address, so it cannot pick its own rate-limit bucket.
    """
    peer = client_address[0] if client_address else "-"
    if PUBLIC and peer in ("127.0.0.1", "::1"):
        cf = (headers.get("CF-Connecting-IP") or "").strip()
        if cf and len(cf) <= 64:
            return cf
    return peer


def _api_key_ok(headers):
    key = os.environ.get("ROAD_SHIELD_API_KEY", "")
    if not key:
        return True
    import hmac
    given = headers.get("X-API-Key") or ""
    auth = headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        given = given or auth[7:].strip()
    return bool(given) and hmac.compare_digest(given.encode(), key.encode())
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8", ".json": "application/json",
    ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
    ".ico": "image/x-icon", ".webmanifest": "application/manifest+json",
}


def get_session_tracker(session_id="default", reset=False):
    if reset or session_id not in video_trackers:
        video_trackers[session_id] = SpatialTemporalVideoTracker(iou_threshold=0.25, max_age_frames=5)
    return video_trackers[session_id]


# ==============================================================================
# HTTP REQUEST HANDLER WITH CORS SUPPORT
# ==============================================================================
def _first(src, *keys):
    """First present, non-empty value among keys in a dict (or a parse_qs dict)."""
    for k in keys:
        if k in src:
            v = src[k]
            if isinstance(v, list):
                v = v[0] if v else None
            if v is not None and v != "":
                return v
    return None


def _text_or_none(v, n):
    return None if v is None or v == "" else str(v)[:n]


def _actor(body):
    a = _text_or_none(body.get("actor"), 40) if isinstance(body, dict) else None
    return a if a and all(c.isalnum() or c in " ._-@" for c in a) else "operator"


def _float_or_none(src, *keys):
    v = _first(src, *keys)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{keys[0]} must be a number, got {v!r}")


def _require_latlon(src, lat_keys=("lat", "latitude"), lon_keys=("lon", "lng", "longitude")):
    """(lat, lon, error). The API never substitutes a default city centre for a
    missing coordinate - that is how a request with no location used to come
    back with an address in Bengaluru or Delhi."""
    try:
        lat, lon = _float_or_none(src, *lat_keys), _float_or_none(src, *lon_keys)
    except ValueError as e:
        return None, None, str(e)
    if lat is None or lon is None:
        return None, None, f"{lat_keys[0]} and {lon_keys[0]} are required"
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None, None, "lat/lon out of range"
    return lat, lon, None


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class RoadShieldAPIHandler(BaseHTTPRequestHandler):
    """
    One note on error handling: a client that disconnects mid-response is not an
    error. Browsers cancel requests constantly - navigating away, closing a tab,
    a fetch superseded by the next one - and the default handler answers each
    with a traceback on stdout. During a demonstration that reads as a crash.
    Those three exceptions are caught and reported as a single quiet line.
    """


    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
        self.send_header("Access-Control-Expose-Headers", "X-Request-ID")
        if getattr(self, "_rid", None):
            self.send_header("X-Request-ID", self._rid)

    def send_response(self, code, message=None):
        BaseHTTPRequestHandler.send_response(self, code, message)
        self._status = code

    def log_request(self, code="-", size="-"):
        # Structured access log, one JSON line per request, when
        # ROAD_SHIELD_ACCESS_LOG=1. Off by default so a demo console stays quiet.
        if os.environ.get("ROAD_SHIELD_ACCESS_LOG") != "1":
            return
        try:
            print(json.dumps({"ts": round(time.time(), 3), "rid": getattr(self, "_rid", None),
                              "method": self.command, "path": self.path.split("?")[0],
                              "status": int(code) if str(code).isdigit() else code,
                              "ms": round((time.time() - getattr(self, "_t_req", time.time())) * 1000, 1)}),
                  flush=True)
        except Exception:
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _serve_static(self, name):
        """
        Serve one file from web/. Returns True if it was served.

        `name` is a filename resolved from a route, or a path relative to web/
        from a /web/ URL (web/samples/... included). Any ".." part is refused,
        and the realpath check refuses anything that still escapes web/.
        """
        parts = [p for p in str(name).replace("\\", "/").split("/") if p not in ("", ".")]
        if not parts or any(p == ".." for p in parts):
            return False
        root = os.path.realpath(WEB_DIR)
        full = os.path.realpath(os.path.join(root, *parts))
        if not full.startswith(root + os.sep) or not os.path.isfile(full):
            return False
        ext = os.path.splitext(full)[1].lower()
        with open(full, "rb") as fh:
            content = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", STATIC_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        if getattr(self, "_rid", None):
            self.send_header("X-Request-ID", self._rid)
        self.end_headers()
        self.wfile.write(content)
        return True

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

        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # The client went away mid-response. Entirely normal - a page
            # navigated, a tab closed, a fetch was cancelled - and there is
            # nothing to do about it. Left unhandled it printed a full
            # WinError 10053 traceback on the console, which looks like a crash
            # to anyone watching a demo and is not one.
            self._client_gone = True

    def handle_one_request(self):
        """Swallow client-side disconnects; let everything else behave normally."""
        import uuid
        # One id per request, returned as X-Request-ID and written to the access
        # log, so a client report can be matched to the server line that served it.
        self._rid = uuid.uuid4().hex[:16]
        self._t_req = time.time()
        self._status = None
        try:
            return BaseHTTPRequestHandler.handle_one_request(self)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True
        finally:
            if getattr(self, "command", None) and self._status is not None:
                METRICS.observe(self.path.split("?")[0], self.command, self._status, time.time() - self._t_req)

    def log_error(self, fmt, *args):
        # BaseHTTPRequestHandler routes broken pipes through here too.
        msg = fmt % args if args else fmt
        if any(k in str(msg) for k in ("10053", "10054", "Broken pipe", "aborted", "reset")):
            return
        BaseHTTPRequestHandler.log_error(self, fmt, *args)

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
        # ---------------- static site ----------------
        # The frontend is a real multi-page site under web/, not one HTML file.
        # Each page is served by name; unknown names fall through to the API.
        page = PAGE_ROUTES.get(path)
        if page is None and path.startswith("/web/"):
            page = path[len("/web/"):]
        if page:
            served = self._serve_static(page)
            if served:
                return
        if path in ("/dashboard", "/frontend", "/gui", "/app") or (path == "/" and "text/html" in accept_header):
            # Legacy single-file dashboard, kept so old links keep working.
            for candidate in ("index.html", "road_shield_frontend.html"):
                fp = os.path.join(ENGINE_ROOT, candidate)
                if os.path.exists(fp):
                    with open(fp, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(content)
                    return

        if path == "/metrics":
            text = METRICS.render(_readiness())
            if MLOPS is not None:
                try:
                    text += MLOPS.monitor.prometheus()
                except Exception as e:
                    text += f"# model monitoring unavailable: {e}\n"
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/v1/ready":
            r = _readiness()
            self._send_json(200 if r["ready"] else 503, r)
            return

        if path in ["/", "/api/v1/health"]:
            self._send_json(200, {
                "service": "ROAD-SHIELD AI Engine",
                "status": "ONLINE",
                "timestamp_utc": int(time.time()),
                "public_demo": PUBLIC,
                "models": {
                    "vision_distress_net": (f"LOADED ({VISION_BACKEND})" if vision_model.is_ready else "NOT_TRAINED"),
                    "coco_object_detector": (deep_pipeline.object_detector.backend
                                             if deep_pipeline.object_detector.is_ready
                                             else "NOT_INSTALLED (scripts/fetch_detector.py)"),
                    "imu_shock_classifier": "LOADED" if imu_model.is_ready else "NOT_TRAINED",
                    "bayesian_fusion_gate": "READY",
                    "ipm_homography_engine": "READY",
                    "forensic_audit_engine": "READY",
                    "morth_dispatch_agent": "READY",
                    "astm_d6433_pci_engine": ("LOADED (HistGradientBoostingRegressor+ASTM_D6433_20)"
                                              if getattr(pci_model, "is_ready", False)
                                              else "READY_TABLE_LOOKUP"),
                    "deterioration_forecaster": ("LOADED (HistGradientBoostingRegressor+HDM4_LTPP)"
                                                 if getattr(degrade_model, "is_ready", False)
                                                 else "READY_FORMULA_FALLBACK"),
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
            lat, lon, err = _require_latlon(q_params)
            if err:
                self._send_json(400, {"error": err})
                return
            self._send_json(200, google_maps_service.reverse_geocode(lat, lon))
            return

        if path == "/api/v1/maps/elevation":
            q_params = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            lat, lon, err = _require_latlon(q_params)
            if err:
                self._send_json(400, {"error": err})
                return
            self._send_json(200, google_maps_service.get_elevation(lat, lon))
            return

        if path == "/api/v1/maps/places-nearby":
            q_params = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            lat, lon, err = _require_latlon(q_params)
            if err:
                self._send_json(400, {"error": err})
                return
            fac_type = q_params.get("type", ["all"])[0]
            self._send_json(200, google_maps_service.find_nearby_civil_facilities(lat, lon, fac_type))
            return

        if path == "/api/v1/maps/streetview-url":
            q_params = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            lat, lon, err = _require_latlon(q_params)
            if err:
                self._send_json(400, {"error": err})
                return
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
            # Depots near the ledger's own defects (their centroid), not near a
            # hard-coded city centre; none at all when the ledger is empty.
            if defects:
                c_lat = sum(d["lat"] for d in defects) / len(defects)
                c_lon = sum(d["lon"] for d in defects) / len(defects)
                civil_depots = google_maps_service.find_nearby_civil_facilities(c_lat, c_lon)["facilities"]
            else:
                civil_depots = []
            repair = works.status_by_defect()
            for d in defects:
                d["repair"] = repair.get(d["defect_id"])
            self._send_json(200, {
                "system": "ROAD-SHIELD Fleet & Defect GIS Dashboard",
                "note": "deduplicated_defects is real, computed state from fleet_dedup_engine (demo reports only if the server was started with ROAD_SHIELD_SEED_DEMO=1). There is no live bus GPS/traffic feed in this project, so fleet_units/congestion figures below are not shown here - see /api/v1/fleet/telemetry for the real dedup registry stats instead.",
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
            # Pass the storage facts through unchanged. Whether the ledger is
            # durable, and where it lives, is exactly the sort of thing a pilot
            # needs to know and a demo tends to hide.
            self._send_json(200, {
                **stats,
                "fleet_status": "DEMO_SEED_DATA_NO_LIVE_GPS_FEED",
            })
            return

        # ---------------- Training / models ----------------
        if path == "/api/v1/training/metrics":
            out = {"active_vision_backend": VISION_BACKEND}
            reports = (
                ("vision", os.path.join(CKPT_DIR, "vision_distress_report.json")),
                ("imu", os.path.join(CKPT_DIR, "imu_shock_report.json")),
                ("pci", os.path.join(CKPT_DIR, "pci_model_report.json")),
                ("deterioration", os.path.join(CKPT_DIR, "deterioration_model_report.json")),
                ("depth_estimator", os.path.join(CKPT_DIR, "depth_estimator_report.json")),
                ("deep_vision", os.path.join(CKPT_DIR, "deep_vision_report.json")),
                ("dan_dag", os.path.join(CKPT_DIR, "dan_dag_report.json")),
            )
            for name, path_ in reports:
                if os.path.exists(path_):
                    with open(path_, "r", encoding="utf-8") as f:
                        out[name] = json.load(f)
                elif name != "deep_vision":
                    out[name] = {"status": "NOT_YET_TRAINED"}
            # CNN-embedding heads, one report per backbone; report the one the
            # server actually loaded rather than the best-looking file on disk.
            loaded = getattr(vision_model, "report", None) or {}
            for cnn_report in sorted(glob.glob(os.path.join(CKPT_DIR, "cnn_head_*_report.json"))):
                with open(cnn_report, "r", encoding="utf-8") as f:
                    blob = json.load(f)
                key = f"cnn_head_{blob.get('backbone', 'unknown')}"
                blob["active"] = bool(VISION_BACKEND == "cnn_embeddings"
                                      and loaded.get("backbone") == blob.get("backbone"))
                out[key] = blob
            from models.served_report import training_extras
            out.update(training_extras(CKPT_DIR))
            self._send_json(200, out)
            return

        if path == "/api/v1/calibration/profiles":
            from models.camera_calibration import CALIB_DIR, DEFAULT_PROFILE, describe, list_profiles
            profiles = list_profiles()
            self._send_json(200, {
                "profiles": [describe(p, "calibrated") for p in profiles],
                "count": len(profiles),
                "directory": CALIB_DIR,
                "active_default": describe(DEFAULT_PROFILE, "default"),
                "why_it_matters": (
                    "Every area, tonnage and cost is derived by projecting pixels onto "
                    "the ground plane using these numbers. Without a calibrated profile "
                    "the system assumes one mount for every vehicle; mounting the same "
                    "camera 30 cm higher changes a computed area by about 46%."),
                "how_to_add": "python -m scripts.calibrate_camera --device <id> "
                              "--hfov <deg> --width <px> --height-px <px> "
                              "--height <m> --pitch <deg>",
            })
            return

        if path == "/api/v1/models/served":
            from models.served_report import model_registry
            self._send_json(200, model_registry(CKPT_DIR))
            return

        if path == "/api/v1/openapi.json":
            from api.openapi import spec
            self._send_json(200, spec())
            return

        if path == "/api/v1/claims":
            # The claims registry, served live so every statement this project
            # makes can be checked against the engine that is actually running -
            # including the ones that were withdrawn or corrected.
            claims_path = os.path.join(CKPT_DIR, "claims.json")
            if not os.path.exists(claims_path):
                self._send_json(404, {"error": "claims.json not present",
                                      "fix": "python -m scripts.build_claims"})
                return
            with open(claims_path, "r", encoding="utf-8") as fh:
                claims = json.load(fh)
            # Attach LIVE status to each subsystem, so the page shows what is
            # loaded right now rather than what a document says should be.
            live = {
                "M1": bool(vision_model.is_ready),
                "M1-fallback": os.path.exists(os.path.join(CKPT_DIR, "vision_distress_model.joblib")),
                "M1-edge": os.path.exists(os.path.join(CKPT_DIR, "cnn_backbone_mobilenetv2.onnx")),
                "M_SEG": bool(getattr(deep_pipeline, "segmenter", None)
                              and deep_pipeline.segmenter.is_ready),
                "M_DET": bool(deep_pipeline.object_detector.is_ready),
                "M4": bool(imu_model.is_ready),
                "M2": True, "M_CALIB": True, "M_DEPTH": True, "M5": True,
                "M_PCI": True, "M_DEGRADE": True, "M_COST": True,
                "M_SEAL": True, "M_FORENSIC": True,
                "M_LEDGER": fleet_dedup_engine.store is not None,
                "M_VIDEO": True,
                "M_GATE": bool(getattr(deep_pipeline, "segmenter", None)
                               and deep_pipeline.segmenter.is_ready),
            }
            for sub in claims.get("subsystems", []):
                sub["live"] = live.get(sub["id"])
            claims["served_by"] = "live engine"
            claims["vision_backend"] = VISION_BACKEND
            self._send_json(200, claims)
            return

        if path == "/api/v1/segmentation/status":
            seg = getattr(deep_pipeline, "segmenter", None)
            if seg is None or not seg.is_ready:
                out = {
                    "available": False,
                    "area_method": "bounding-box corners projected to the ground plane",
                    "consequence": "A box around a diagonal crack overstates its area by "
                                   "roughly an order of magnitude, and cost is linear in area.",
                    "fix": "python -m training.train_segmenter",
                }
                if seg is not None and seg.file_exists:
                    # Trained, but will not load here (usually a scikit-learn
                    # version gap). Say that, and show the measured report,
                    # rather than claiming it was never trained.
                    out["model_on_disk"] = True
                    out["load_error"] = seg.load_error_detail
                    out["fix"] = (seg.load_error_detail or {}).get("fix", out["fix"])
                    rp = os.path.splitext(seg.model_path)[0] + "_report.json"
                    try:
                        with open(rp, "r", encoding="utf-8") as fh:
                            _rep = json.load(fh)
                        out["measured_report"] = _rep.get("iou")
                        out["trained_on"] = _rep.get("trained_on")
                        out["thresholds"] = _rep.get("thresholds")
                    except Exception:
                        pass
                self._send_json(200, out)
                return
            from models.unet_segmenter import read_selection
            sel = read_selection(CKPT_DIR) or {}
            self._send_json(200, {"available": True, **seg.describe(),
                                  "model": seg.describe().get("model") or
                                  "HistGradientBoosting pixel classifier on 11 features",
                                  "selection": {k: sel.get(k) for k in ("served", "rule", "why")} if sel else None})
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

        if path == "/api/v1/live/stream":
            raw = self.headers.get("Last-Event-ID") or _first(
                urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query), "since") or ""
            sub = live_events.subscribe(int(raw) if str(raw).isdigit() else None)
            if sub is None:
                self._send_json(503, {"error": "too many open live streams; retry shortly"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self._send_cors_headers()
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(b"retry: 3000\n\n")
                self.wfile.flush()
                deadline = time.time() + SSE_MAX_SECONDS
                while time.time() < deadline:
                    try:
                        self.wfile.write(sse_frame(sub.get(timeout=15)))
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                live_events.unsubscribe(sub)
            return

        if path == "/api/v1/works/orders":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            status = (q.get("status") or [None])[0]
            if status and status.upper() not in ORDER_STATES:
                self._send_json(400, {"error": f"status must be one of {', '.join(ORDER_STATES)}"})
                return
            orders = works.list(status)
            alerts.check_overdue(orders)
            self._send_json(200, {"orders": orders, "summary": works.summary()})
            return

        if path == "/api/v1/works/order":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            order = works.get((q.get("id") or [""])[0])
            if order is None:
                self._send_json(404, {"error": "no such work order"})
                return
            self._send_json(200, order)
            return

        if path == "/api/v1/citizen/reports":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            status = (q.get("status") or [None])[0]
            if status and status.upper() not in ("PENDING", "CONFIRMED", "DISMISSED"):
                self._send_json(400, {"error": "status must be pending, confirmed or dismissed"})
                return
            self._send_json(200, {"reports": citizen.list(status), "counts": citizen.counts()})
            return

        if path == "/api/v1/alerts":
            self._send_json(200, {"alerts": alerts.recent(), **alerts.status()})
            return

        if mlops_routes.handle_get(self, MLOPS, path, full_path, _api_key_ok):
            return

        if path == "/api/v1/ledger/export":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            fmt_ = (q.get("format") or ["geojson"])[0].lower()
            if fmt_ not in ("geojson", "csv"):
                self._send_json(400, {"error": "format must be geojson or csv"})
                return
            from pipeline.export import ledger_csv, ledger_geojson
            defects = fleet_dedup_engine.get_all_deduplicated_defects()
            repair = works.status_by_defect()
            body_bytes = (ledger_geojson(defects, _safe_priority, repair) if fmt_ == "geojson"
                          else ledger_csv(defects, _safe_priority, repair)).encode("utf-8")
            stamp = time.strftime("%Y%m%d")
            self.send_response(200)
            self.send_header("Content-Type", "application/geo+json" if fmt_ == "geojson" else "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="road_shield_defects_{stamp}.{fmt_}"')
            self.send_header("Content-Length", str(len(body_bytes)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body_bytes)
            return

        if path == "/api/v1/fleet/live":
            self._send_json(200, {
                "buses": edge_ingest.live_positions(),
                "imu_only_shocks": edge_ingest.recent_shocks(100),
                "known_buses": edge_ingest.buses(),
                "encrypted_link": "configured" if edge_ingest.configured else "ROAD_SHIELD_FLEET_KEY not set",
                "live_stream_subscribers": live_events.subscribers,
                "recent_events": live_events.recent()[-50:],
            })
            return

        if path == "/api/v1/priority/ranking":
            from models import priority_index
            q = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
            try:
                w = priority_index.parse_weights(q["weights"][0]) if "weights" in q else priority_index.default_weights()
                ranked = priority_index.rank(_with_traffic(fleet_dedup_engine.get_all_deduplicated_defects()), w)
                repair_status = works.status_by_defect()
                stability = (priority_index.rank_stability(_with_traffic(fleet_dedup_engine.get_all_deduplicated_defects()), w)
                             if q.get("stability", ["0"])[0] in ("1", "true", "yes") else None)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            self._send_json(200, {
                "formula": "PI = w1*(100-PCI) + w2*Vol + w3*Traffic, each term on 0-100",
                "weights": dict(zip(priority_index.TERMS, w)),
                "bands": [{"band": b, "from": f, "action": a} for f, b, a in priority_index.BANDS],
                "defects": [{
                    "rank": r["priority"]["rank"], "defect_id": r["defect_id"], "defect_class": r["defect_class"],
                    "lat": r["lat"], "lon": r["lon"], "severity_pci": r["severity_pci"],
                    "area_m2": r.get("area_m2"), "depth_cm": r.get("depth_cm"),
                    "confirmations": r.get("confirmation_count"), "address": r.get("address"),
                    "priority": r["priority"], "repair": repair_status.get(r["defect_id"]),
                } for r in ranked],
                "rank_stability": stability,
            })
            return

        if path == "/api/v1/ledger/defects":
            from models import priority_index
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
                    "priority": _safe_priority(d),
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
        try:
            declared = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            declared = -1
        if declared < 0 or declared > ops.MAX_BODY_BYTES:
            self.close_connection = True              # the body is not read, so the connection cannot be reused
            self._send_json(413, {"error": f"request body larger than {ops.MAX_BODY_BYTES // (1024 * 1024)} MB "
                                           f"(set ROAD_SHIELD_MAX_BODY_MB to change)"})
            return
        if path in ops.RATE_LIMITED:
            allowed, retry = LIMITER.check(_client_key(self.headers, self.client_address))
            if not allowed:
                self.close_connection = True
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Retry-After", str(retry))
                self._send_cors_headers()
                payload = json.dumps({"error": f"rate limit: {LIMITER.per_minute} model requests per minute per "
                                               f"client; retry in {retry} s"}).encode()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
        body = self._read_json_body()
        t0 = time.time()

        refused = _server_path_refused(body)
        if refused:
            self._send_json(403, {"error": f"'{refused}' names a file on the server. This public demo only reads "
                                           f"files under datasets/; send your own image as base64."})
            return

        if path in PROTECTED_POST and not _api_key_ok(self.headers):
            self._send_json(401, {"error": "this endpoint requires an API key (X-API-Key header or "
                                           "Authorization: Bearer <key>)"})
            return

        if mlops_routes.handle_post(self, MLOPS, path, body, _actor(body)):
            return

        # ---------------- Google Maps (POST) ----------------
        if path == "/api/v1/maps/config":
            self._send_json(200, google_maps_service.set_api_key(body.get("api_key", "")))
            return

        if path == "/api/v1/maps/directions":
            origin_lat, origin_lng, e1 = _require_latlon(body, ("origin_lat",), ("origin_lng", "origin_lon"))
            dest_lat, dest_lng, e2 = _require_latlon(body, ("dest_lat",), ("dest_lng", "dest_lon"))
            if e1 or e2:
                self._send_json(400, {"error": e1 or e2})
                return
            avoid_defects = bool(body.get("avoid_defects", False))
            known_defects = fleet_dedup_engine.get_all_deduplicated_defects()
            dirs = google_maps_service.get_directions(origin_lat, origin_lng, dest_lat, dest_lng, avoid_defects=avoid_defects, known_defects=known_defects)
            self._send_json(200, dirs)
            return

        if path == "/api/v1/maps/geocode":
            self._send_json(200, google_maps_service.geocode(body.get("query", body.get("address", ""))))
            return

        if path == "/api/v1/maps/reverse-geocode":
            lat, lon, err = _require_latlon(body)
            if err:
                self._send_json(400, {"error": err})
                return
            self._send_json(200, google_maps_service.reverse_geocode(lat, lon))
            return

        if path == "/api/v1/maps/elevation":
            lat, lon, err = _require_latlon(body)
            if err:
                self._send_json(400, {"error": err})
                return
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
                "model": "IMUShockClassifier" if imu_backend == "random_forest" else "IMUShockCNN (1-D CNN, ONNX)",
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
            from models.bayesian_fusion_gate import SEVERE_JOLT_MS2
            p_imu = float(p_imu) if p_imu is not None else (0.9 if accel_delta_z >= SEVERE_JOLT_MS2 else 0.05)

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
            # A sealed work order is a billing document. Every measured field
            # must come from the caller; the old defaults (a 2.2 m2, 6.5 cm
            # pothole at PCI 42 in Delhi) sealed an invented defect whenever a
            # field was missing - and web/works.html sends defect_class/pci,
            # which were silently ignored. GPS is optional: without it the
            # order is issued with dispatch_status HELD_NO_GPS.
            try:
                fields = {
                    "distress_class": _first(body, "distress_class", "distress_type", "defect_class"),
                    "area_sqm": _float_or_none(body, "area_sqm", "area_m2"),
                    "depth_cm": _float_or_none(body, "depth_cm"),
                    "pci_score": _float_or_none(body, "pci_score", "pci"),
                    "latitude": _float_or_none(body, "latitude", "lat"),
                    "longitude": _float_or_none(body, "longitude", "lng", "lon"),
                }
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            missing = [k for k in ("distress_class", "area_sqm", "depth_cm", "pci_score") if fields[k] is None]
            if missing:
                self._send_json(400, {"error": "missing required fields: " + ", ".join(missing),
                                      "required": ["distress_class", "area_sqm", "depth_cm", "pci_score"],
                                      "optional": ["latitude", "longitude", "corridor_id"]})
                return
            try:
                work_order = dispatch_agent.generate_work_order(
                    corridor_id=str(body.get("corridor_id", body.get("highway", "UNSPECIFIED"))),
                    **fields)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            work_order["model"] = "MoRTHDispatchAgent"
            work_order["latency_ms"] = round((time.time() - t0) * 1000.0, 3)
            live_events.publish("work_order", {k: work_order.get(k) for k in (
                "work_order_id", "coordinates", "distress_type", "priority", "allocated_budget_inr",
                "dispatch_status", "seal_algorithm")})
            self._send_json(200, work_order)
            return

        if path == "/api/v1/dispatch/verify-seal":
            work_order = body.get("work_order", body)
            if not isinstance(work_order, dict):
                self._send_json(400, {
                    "is_valid": False,
                    "status": "MALFORMED_WORK_ORDER",
                    "error": "work_order must be the JSON object returned by /api/v1/dispatch/work-order",
                })
                return
            status = dispatch_agent.check_work_order_seal(work_order)
            self._send_json(200, {
                "is_valid": status == "SEAL_VERIFIED_AUTHENTIC",
                "work_order_id": work_order.get("work_order_id", "UNKNOWN"),
                "seal_algorithm": work_order.get("seal_algorithm", "SHA-256"),
                "status": status,
            })
            return

        # ----------------------------------------------------------------------
        # ASTM D6433 PCI (real deduct-value formula, not a trained regressor)
        # ----------------------------------------------------------------------
        if path == "/api/v1/priority/score":
            from models import priority_index
            try:
                pci = _float_or_none(body, "pci", "severity_pci", "pci_score")
                if pci is None:
                    raise ValueError("pci is required")
                res = priority_index.score(
                    pci=pci,
                    volume_m3=_float_or_none(body, "volume_m3"),
                    area_m2=_float_or_none(body, "area_m2", "area_sqm"),
                    depth_cm=_float_or_none(body, "depth_cm"),
                    traffic_pcu_per_day=_float_or_none(body, "traffic_pcu_per_day", "pcu_per_day"),
                    reporting_buses=body.get("reporting_buses"),
                    weights=body.get("weights"),
                )
            except (ValueError, TypeError) as e:
                self._send_json(400, {"error": str(e)})
                return
            self._send_json(200, res)
            return

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
                result = deep_pipeline.audit_image(image_input=preset["image_path"], corridor_id=f"Demo preset ({preset['class_name']})",
                                                   _source="dataset")
                # Hand the dashboard the actual photo that was analysed, so the
                # overlay it draws is over the real input rather than a stand-in.
                try:
                    import base64 as _b64, mimetypes as _mt
                    with open(preset["image_path"], "rb") as _fh:
                        _raw = _fh.read()
                    _mime = _mt.guess_type(preset["image_path"])[0] or "image/jpeg"
                    result["image_data_url"] = f"data:{_mime};base64," + _b64.b64encode(_raw).decode("ascii")
                    result["image_source_file"] = os.path.basename(preset["image_path"])
                except Exception:
                    pass
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
                analysis = deep_pipeline.audit_image(
                    image_input=img_b64, corridor_id=highway,
                    device_id=body.get("device_id"), _source=_upload_source())
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
                analysis = deep_pipeline.audit_image(image_input=frame_b64, corridor_id=body.get("clip_id", "uploaded_stream"),
                                                     _source=_upload_source())
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
                        "total_unique_other_hazards_counted": tracking_res["total_unique_other_hazards_counted"],
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
            # Absent GPS stays absent (None) and absent forecast inputs are
            # filled by the pipeline and listed under modelling_assumptions.
            try:
                lat = _float_or_none(body, "latitude", "lat")
                lng = _float_or_none(body, "longitude", "lng", "lon")
                chainage = _float_or_none(body, "chainage_km")
                traffic_esal = _float_or_none(body, "traffic_esal")
                rain_mm = _float_or_none(body, "rain_mm")
                age_yr = _float_or_none(body, "age_years", "pavement_age_yr")
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            imu_series = body.get("imu_series")  # only used if the caller supplies a real (100,3) window
            device_id = body.get("device_id")  # camera calibration profile, if the caller has one

            if not image_input:
                # Used to audit a bundled pothole photograph instead, so an
                # empty request came back with a confident pothole detection.
                self._send_json(400, {"error": "Missing image_base64 or image_path payload"})
                return
            if not body.get("image_base64"):
                # image_path is a convenience for local datasets only; a remote
                # caller must not be able to make the server open any file.
                datasets_root = os.path.realpath(os.path.join(ENGINE_ROOT, "datasets"))
                real = os.path.realpath(os.path.join(ENGINE_ROOT, str(image_input)))
                if not real.startswith(datasets_root + os.sep) or not os.path.isfile(real):
                    self._send_json(400, {"error": "image_path must name an existing file under datasets/"})
                    return
                image_input = real

            try:
                audit_result = deep_pipeline.audit_image(
                    image_input=image_input,
                    _source="dataset" if not body.get("image_base64") else _upload_source(),
                    corridor_id=corridor,
                    latitude=lat,
                    longitude=lng,
                    chainage_km=chainage,
                    imu_series=imu_series,
                    traffic_esal=traffic_esal,
                    rain_mm=rain_mm,
                    pavement_age_yr=age_yr,
                    device_id=device_id,
                )
                self._send_json(200, audit_result)
            except Exception as e:
                self._send_json(500, {"error": f"Deep pipeline audit failed: {str(e)}"})
            return

        if path == "/api/v1/video/ingest":
            # Decodes a real video file, samples frames by ground distance and
            # feeds them through the pipeline into the durable ledger.
            src = body.get("video_path") or body.get("path")
            if not src:
                self._send_json(400, {
                    "error": "video_path is required",
                    "note": "This endpoint reads a video file from the server's "
                            "filesystem. Uploading multi-hundred-megabyte dashcam "
                            "footage through a JSON body is the wrong shape for the "
                            "problem; in a fleet deployment the bus uploads to "
                            "object storage and this is handed the key.",
                })
                return
            if not os.path.isfile(src):
                self._send_json(404, {"error": f"no such video: {src}"})
                return
            try:
                from pipeline.video_ingest import VideoIngestor
                ing = VideoIngestor(deep_pipeline, fleet_dedup_engine,
                                    traffic=MLOPS.traffic if MLOPS is not None else None)
                summary = ing.process(
                    src,
                    gps_track=body.get("gps_track"),
                    bus_id=body.get("bus_id", "UNKNOWN"),
                    sample_every_m=float(body.get("sample_every_m", 8.0)),
                    sample_every_s=float(body.get("sample_every_s", 1.0)),
                    max_frames=int(body.get("max_frames", 200)),
                    device_id=body.get("device_id"),
                    recorded_at=_float_or_none(body, "recorded_at_unix"),
                )
                # The frame-by-frame list can be enormous; return it only if asked.
                if not body.get("include_frames"):
                    summary["detections"] = summary["detections"][:25]
                    summary["detections_truncated_to"] = 25
                self._send_json(200, summary)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        if path == "/api/v1/video/probe":
            src = body.get("video_path") or body.get("path")
            if not src or not os.path.isfile(src):
                self._send_json(404, {"error": f"no such video: {src}"})
                return
            from pipeline.video_ingest import VideoIngestor
            self._send_json(200, VideoIngestor.probe(src))
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
                batch_result = deep_pipeline.process_batch(image_source=dir_path, max_samples=max_samples, corridor_id=corridor,
                                                           _source="dataset")
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
        # ----------------------------------------------------------------------
        # COCO object detection (people, vehicles, traffic control) - real
        # inference over published weights, or an honest "not installed".
        # ----------------------------------------------------------------------
        if path == "/api/v1/detect/objects":
            detector = deep_pipeline.object_detector
            if not detector.is_ready:
                self._send_json(503, {
                    "error": "No detector weights on this server.",
                    "fix": "python -m scripts.fetch_detector   (downloads COCO-pretrained YOLO and exports ONNX)",
                    "detector": detector.describe(),
                })
                return
            img_b64 = body.get("image_base64")
            if not img_b64:
                self._send_json(400, {"error": "Missing image_base64."})
                return
            try:
                img_np = cv_detector.decode_image(img_b64)
                conf = float(body.get("confidence_threshold", detector.conf_threshold))
                keep = body.get("classes")
                objects = detector.detect(img_np, conf_threshold=conf, keep_classes=set(keep) if keep else None)
                out = detector.summarise(objects)
                out["objects"] = objects
                out["latency_ms"] = round((time.time() - t0) * 1000.0, 3)
                self._send_json(200, out)
            except Exception as e:
                self._send_json(500, {"error": f"Detection failed: {e}"})
            return

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
            # Every field of a sighting comes from the bus. Missing GPS used to
            # become Bengaluru's city centre and a missing class a pothole.
            bus_id = body.get("bus_id", "BUS-UNKNOWN")
            lat, lon, err = _require_latlon(body)
            cls_name = _first(body, "defect_class")
            try:
                pci = _float_or_none(body, "severity_pci", "pci")
                area = _float_or_none(body, "area_m2", "area_sqm")
            except ValueError as e:
                err = err or str(e)
                pci = area = None
            missing = [n for n, v in (("defect_class", cls_name), ("severity_pci", pci), ("area_m2", area)) if v is None]
            if err or missing:
                self._send_json(400, {"error": err or ("missing required fields: " + ", ".join(missing))})
                return
            try:
                depth = _float_or_none(body, "depth_cm")
                traffic = _float_or_none(body, "traffic_pcu_per_day", "pcu_per_day")
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            try:
                res = fleet_dedup_engine.ingest_fleet_detection(bus_id, lat, lon, cls_name, pci, area,
                                                                depth_cm=depth, traffic_pcu_per_day=traffic)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            rec = next((d for d in fleet_dedup_engine.get_all_deduplicated_defects()
                        if d["defect_id"] == res["defect_id"]), None)
            live_events.publish("defect", {"bus_id": bus_id, "result": res, "defect": rec, "via": "api"})
            self._send_json(200, res)
            return

        if path == "/api/v1/citizen/report":
            b64 = body.get("image_base64") if isinstance(body, dict) else None
            if not b64:
                self._send_json(400, {"error": "image_base64 is required"})
                return
            try:
                lat, lon = _float_or_none(body, "lat", "latitude"), _float_or_none(body, "lon", "lng", "longitude")
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            if lat is None or lon is None:
                self._send_json(400, {"error": "a location is needed: allow location access, or use a photograph with GPS"})
                return
            try:
                audit = deep_pipeline.audit_image(b64, corridor_id="CITIZEN", latitude=lat, longitude=lon,
                                                  _source="citizen")
            except Exception as e:
                self._send_json(400, {"error": f"could not read the photograph: {e}"})
                return
            check = audit.get("input_check") or {}
            if check.get("verdict") in ("not_road", "poor_quality"):
                from models.ood_guard import OODGuard
                self._send_json(422, {"error": OODGuard.MESSAGES[check["verdict"]]
                                      + (" Please take it again in daylight, holding the phone steady."
                                         if check["verdict"] == "poor_quality" else
                                         " Photograph the damaged road surface itself."),
                                      "input_check": check, "report": None, "photo_stored": False})
                return
            try:
                report, message = citizen.submit(audit, lat, lon, client=_client_key(self.headers, self.client_address),
                                                 location_source=_text_or_none(body.get("location_source"), 40) or "unknown")
            except CitizenError as e:
                self._send_json(429 if "per hour" in str(e) else 400, {"error": str(e)})
                return
            top = audit.get("primary_distress") or {}
            self._send_json(200, {"report": report, "message": message,
                                  "finding": {"class": top.get("class_name"), "confidence": top.get("confidence"),
                                              "is_distress": audit.get("is_distress")},
                                  "photo_stored": False})
            return

        if path == "/api/v1/citizen/review":
            rid, action = _first(body, "report_id"), _first(body, "action")
            if not rid or not action:
                self._send_json(400, {"error": "report_id and action (promote or dismiss) are required"})
                return
            try:
                self._send_json(200, citizen.review(str(rid), str(action), actor=_actor(body)))
            except CitizenError as e:
                self._send_json(404 if str(e).startswith("no report") else 409, {"error": str(e)})
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
            return

        if path == "/api/v1/works/orders":
            defect_id = _first(body, "defect_id")
            if not defect_id:
                self._send_json(400, {"error": "defect_id is required (a defect from the ledger)"})
                return
            try:
                depth = _float_or_none(body, "depth_cm")
                order = works.issue(str(defect_id), actor=_actor(body), depth_cm=depth,
                                    note=_text_or_none(body.get("note"), 300))
            except WorkflowError as e:
                self._send_json(409 if "already has an open order" in str(e) else 400, {"error": str(e)})
                return
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            self._send_json(200, order)
            return

        if path == "/api/v1/works/status":
            oid, status = _first(body, "work_order_id", "order_id"), _first(body, "status")
            if not oid or not status:
                self._send_json(400, {"error": "work_order_id and status are required"})
                return
            try:
                order = works.transition(str(oid), str(status), actor=_actor(body),
                                         note=_text_or_none(body.get("note"), 300),
                                         contractor=_text_or_none(body.get("contractor"), 80))
            except WorkflowError as e:
                self._send_json(404 if str(e).startswith("no work order") else 409, {"error": str(e)})
                return
            self._send_json(200, order)
            return

        if path == "/api/v1/fleet/ingest-sealed":
            packets = body.get("packets") if isinstance(body, dict) else None
            if not isinstance(packets, list) or not packets or len(packets) > 500:
                self._send_json(400, {"error": "packets must be a list of 1-500 sealed envelopes"})
                return
            if not edge_ingest.configured:
                self._send_json(503, {"error": "ROAD_SHIELD_FLEET_KEY is not set on this server, so bus packets "
                                               "cannot be opened"})
                return
            self._send_json(200, edge_ingest.ingest(packets))
            return

        # ----------------------------------------------------------------------
        # ALPR / rash-driving incident report (real kinematics + real OCR when an image is given)
        # ----------------------------------------------------------------------
        if path == "/api/v1/privacy/redact":
            # Blur people (head region) and number plates before an image is
            # shared. Returns the redacted JPEG and what was found, plus the measured recall when
            # scripts/measure_redactor_recall.py has written one.
            b64 = body.get("image_base64")
            if not b64:
                self._send_json(400, {"error": "image_base64 is required"})
                return
            try:
                from models.privacy_redactor import redact
                from PIL import Image
                img = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
            except Exception as e:
                self._send_json(400, {"error": f"could not decode image: {e}"})
                return
            red, rep = redact(img, detector=deep_pipeline.object_detector)
            buf = io.BytesIO()
            Image.fromarray(red).save(buf, format="JPEG", quality=88)
            rep["latency_ms"] = round((time.time() - t0) * 1000.0, 1)
            self._send_json(200, {"redacted_image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                                  "report": rep})
            return

        if path == "/api/v1/incidents/alpr":
            bus_id = body.get("bus_id", "UNKNOWN")
            gps = body.get("gps")  # absent -> the report carries no location
            track_history = body.get("track_history")
            if not isinstance(track_history, list) or len(track_history) < 3 or not all(
                    isinstance(t, dict) and "timestamp" in t and isinstance(t.get("bbox"), list)
                    and len(t["bbox"]) == 4 for t in track_history):
                # This endpoint used to substitute a demo track and a fixed GPS
                # fix when none was sent, then seal the result as an incident.
                self._send_json(400, {
                    "error": "track_history is required: at least 3 entries of "
                             "{timestamp: seconds, bbox: [x, y, w, h]} from a real tracker",
                })
                return
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
def _overdue_loop(every_s=300):
    import threading as _t
    stop = _t.Event()

    def loop():
        while not stop.wait(every_s):
            try:
                works.tick()
                alerts.check_overdue(works.list())
            except Exception as e:
                print(f"[alerts] overdue check failed: {e}")
    _t.Thread(target=loop, daemon=True).start()
    return stop


def start_server(port=8000, host="0.0.0.0"):
    _overdue_loop(every_s=30)
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
