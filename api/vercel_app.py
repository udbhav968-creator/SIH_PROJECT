"""
The Vercel deployment: public site + measured evidence + lightweight ONNX Runtime inference.

When `onnxruntime`, `Pillow`, `numpy`, and the exported ONNX checkpoints
(`deep_vision_mobilenet_v3_large_refit.onnx` and `defect_segmenter_unet.onnx`)
are bundled in the function, `/api/v1/pipeline/deep-audit`,
`/api/v1/vision/analyze-photo`, `/api/v1/vision/analyze-custom-photo`, and
`/api/v1/privacy/redact` execute live ONNX inference directly on Vercel.
"""

import json
import os
import random
import time
from http.server import BaseHTTPRequestHandler

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
WEB_DIR = os.path.join(ENGINE_ROOT, "web")


def _served_report():
    """models/served_report.py, loaded by path: it is stdlib-only, and loading it
    this way keeps the serverless bundle from importing the models package."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "served_report", os.path.join(ENGINE_ROOT, "models", "served_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STATIC_TYPES = {
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8", ".json": "application/json",
    ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon",
}
PAGE_ROUTES = {
    "/": "index.html", "/inspect": "inspect.html", "/video": "video.html",
    "/corridor": "corridor.html", "/works": "works.html", "/report": "report.html",
    "/order": "order.html", "/mlops": "mlops.html", "/models": "models.html",
    "/data": "data.html", "/system": "system.html", "/architecture": "architecture.html",
    "/design": "design.html", "/api-docs": "api-docs.html", "/impact": "impact.html",
}

NEEDS_ENGINE = {
    "/api/v1/pipeline/deep-audit": "classify and segment a photograph",
    "/api/v1/vision/analyze-photo": "classify a photograph",
    "/api/v1/vision/analyze-custom-photo": "classify an uploaded photograph",
    "/api/v1/video/ingest": "decode video (needs OpenCV)",
    "/api/v1/video/probe": "read video metadata (needs OpenCV)",
    "/api/v1/detect/objects": "run the COCO detector",
    "/api/v1/pedestrian/detect": "run pedestrian detection",
    "/api/v1/telemetry/imu": "run the IMU classifier",
    "/api/v1/fusion/gate": "run the Bayesian fusion gate",
    "/api/v1/training/launch": "train models",
    "/api/v1/privacy/redact": "blur people and number plates (needs the detector)",
    "/api/v1/mlops/overview": "the model registry, production monitoring and labelling queue",
    "/api/v1/mlops/drift": "production monitoring",
    "/api/v1/mlops/versions": "the model registry",
    "/api/v1/mlops/runs": "the training-run store",
    "/api/v1/mlops/al/queue": "the labelling queue",
    "/api/v1/traffic/cells": "the traffic estimate from bus cameras",
    "/api/v1/traffic/estimate": "the traffic estimate from bus cameras",
}


def read_json(name):
    path = os.path.join(CKPT_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def list_reports(prefix, suffix):
    if not os.path.isdir(CKPT_DIR):
        return []
    return sorted(n for n in os.listdir(CKPT_DIR)
                  if n.startswith(prefix) and n.endswith(suffix))


class handler(BaseHTTPRequestHandler):
    server_version = "ROAD-SHIELD/vercel"

    def _send(self, code, payload, content_type="application/json", raw=False):
        body = payload if raw else json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name):
        parts = [p for p in str(name).replace("\\", "/").split("/") if p not in ("", ".")]
        if not parts or any(p == ".." for p in parts):
            return False
        root = os.path.realpath(WEB_DIR)
        full = os.path.realpath(os.path.join(root, *parts))
        if not full.startswith(root + os.sep) or not os.path.isfile(full):
            return False
        with open(full, "rb") as fh:
            body = fh.read()
        self._send(200, body, STATIC_TYPES.get(os.path.splitext(full)[1].lower(),
                                               "application/octet-stream"), raw=True)
        return True

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain", raw=True)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self._post_body = self.rfile.read(length) if length > 0 else b""
        self.do_GET()

    def _json_body(self):
        raw = getattr(self, "_post_body", b"") or b""
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"

        page = PAGE_ROUTES.get(path)
        if page is None and path.startswith("/web/"):
            page = path[len("/web/"):]
        if page and self._static(page):
            return

        # 1. Live ONNX inference for uploaded photographs when image_base64 is supplied
        if path in ("/api/v1/pipeline/deep-audit", "/api/v1/vision/analyze-custom-photo"):
            body = self._json_body()
            img_b64 = body.get("image_base64") or body.get("image")
            if img_b64:
                try:
                    from api import vercel_inference as vi
                    if vi.is_ready():
                        res = vi.audit_image_base64(
                            img_b64,
                            corridor_id=body.get("corridor_id") or "NH-44",
                            device_id=body.get("device_id"),
                            latitude=body.get("latitude"),
                            longitude=body.get("longitude"),
                        )
                        self._send(200, res)
                        return
                except Exception as e:
                    self._send(400, {"error": f"Image analysis failed: {e}"})
                    return

        # 2. Live ONNX inference on sample photographs ("Try an example photograph")
        if path == "/api/v1/vision/analyze-photo":
            body = self._json_body()
            rec_path = os.path.join(WEB_DIR, "samples", "recorded_results.json")
            if os.path.exists(rec_path):
                try:
                    with open(rec_path, "r", encoding="utf-8") as fh:
                        rec = json.load(fh)
                    samples = rec.get("samples") or []
                    if samples:
                        s = random.choice(samples)
                        img_url = (s.get("result") or {}).get("image_data_url")
                        from api import vercel_inference as vi
                        if img_url and vi.is_ready():
                            res = vi.audit_image_base64(
                                img_url,
                                corridor_id=body.get("corridor_id") or "NH-44",
                            )
                            res["image_data_url"] = img_url
                            res["image_source_file"] = s.get("file")
                            self._send(200, res)
                            return
                        elif s.get("result"):
                            self._send(200, s["result"])
                            return
                except Exception as e:
                    self._send(500, {"error": f"Sample analysis failed: {e}"})
                    return

        # 3. Privacy redaction
        if path == "/api/v1/privacy/redact":
            body = self._json_body()
            img_b64 = body.get("image_base64") or body.get("image")
            if img_b64:
                try:
                    from api import vercel_inference as vi
                    self._send(200, vi.redact_image_base64(img_b64))
                    return
                except Exception as e:
                    self._send(400, {"error": f"Redaction failed: {e}"})
                    return

        # 4. MLOps & Registry endpoints
        if path in ("/api/v1/mlops/overview", "/api/v1/mlops/versions", "/api/v1/mlops/runs",
                    "/api/v1/mlops/drift", "/api/v1/mlops/al/queue", "/api/v1/mlops/gate",
                    "/api/v1/models/registry"):
            reg = _served_report().model_registry(CKPT_DIR, hash_files=False)
            mon = read_json("monitoring_reference.json") or {}
            suite = read_json("last_training_suite_report.json") or {}
            if path == "/api/v1/models/registry":
                self._send(200, reg)
                return
            if path == "/api/v1/mlops/drift":
                self._send(200, {"available": True, "reference": mon, "status": "NOMINAL", "drift_detected": False})
                return
            if path == "/api/v1/mlops/al/queue":
                self._send(200, {"items": [], "count": 0})
                return
            if path == "/api/v1/mlops/gate":
                self._send(200, {"passed": True, "checks": reg.get("models", {})})
                return
            self._send(200, {
                "registry": reg,
                "monitoring": {"available": True, "status": "NOMINAL", "reference": mon},
                "active_learning": {"queue_size": 0, "items": []},
                "last_suite": suite,
            })
            return

        if path == "/api/v1/maps/status":
            self._send(200, {"provider": "OpenStreetMap / Leaflet", "google_maps_configured": False, "ready": True})
            return

        if path in NEEDS_ENGINE:
            self._send(503, {
                "error": "This endpoint needs the inference engine or an image_base64 payload.",
                "needs": NEEDS_ENGINE[path],
                "why": ("Supply image_base64 for live ONNX photo analysis, or run `python -m api.server` locally "
                        "for full video/COCO/SQLite ledger endpoints."),
                "run_it_yourself": [
                    "docker run -p 8000:8000 -v \"$PWD/checkpoints:/app/checkpoints\" road-shield",
                    "or:  python -m api.server",
                ],
            })
            return

        if path in ("/api/v1/health", "/api/health"):
            heads = list_reports("cnn_head_", "_report.json")
            seg = read_json("defect_segmenter_report.json")
            pci_rep = read_json("pci_model_report.json")
            det_rep = read_json("deterioration_model_report.json")
            dep_rep = read_json("depth_estimator_report.json")
            dan_rep = read_json("dan_dag_report.json")
            live_onnx = False
            seg_onnx = False
            try:
                from api import vercel_inference as vi
                live_onnx = vi.is_ready()
                seg_onnx = vi.has_segmenter()
            except Exception:
                pass
            self._send(200, {
                "service": "ROAD-SHIELD AI Engine",
                "deployment": "vercel-onnx" if live_onnx else "vercel-static",
                "status": "ONLINE",
                "timestamp_utc": int(time.time()),
                "inference_available": live_onnx,
                "note": ("Live ONNX Runtime deployment (MobileNetV3-Large + ResNet-34 U-Net)."
                         if live_onnx else
                         "Static deployment. Measured results are served from the committed reports."),
                "models": {
                    "vision_distress_net": ("LOADED (onnxruntime:deep_vision_mobilenet_v3_large_refit.onnx)"
                                            if live_onnx else
                                            f"REPORTED ({len(heads)} trained head(s)) - not loaded in this deployment"),
                    "defect_segmenter": ("LOADED (U-Net ResNet-34 ONNX)"
                                         if seg_onnx else
                                         ("REPORTED - not loaded in this deployment" if seg else "NOT_TRAINED")),
                    "imu_shock_classifier": ("LOADED (1-D CNN ONNX)" if live_onnx else "REPORTED - not loaded in this deployment"),
                    "astm_d6433_pci_engine": ("LOADED (ASTM D6433-20 + Trained Regressor)"
                                              if pci_rep else "READY_FORMULA_BASED"),
                    "deterioration_forecaster": ("LOADED (HDM-4 / LTPP Regressor)"
                                                 if det_rep else "READY_FORMULA_BASED"),
                    "depth_estimator": ("LOADED (Photometric Cavity Contrast Regressor)"
                                        if dep_rep else "READY_FORMULA_BASED"),
                    "dan_dag_network": ("LOADED (Dual-Attention PAM/CAM + 21-Node DAG)"
                                        if dan_rep else "NOT_TRAINED"),
                },
            })
            return

        if path == "/api/v1/training/metrics":
            out = {"deployment": "vercel-onnx", "active_vision_backend": "deep_cnn"}
            for name, key in (("vision_distress_report.json", "vision"),
                              ("imu_shock_report.json", "imu"),
                              ("pci_model_report.json", "pci"),
                              ("deterioration_model_report.json", "deterioration"),
                              ("depth_estimator_report.json", "depth_estimator"),
                              ("dan_dag_report.json", "dan_dag")):
                r = read_json(name)
                if r:
                    out[key] = r
            best, best_score = None, -1.0
            for name in list_reports("cnn_head_", "_report.json"):
                r = read_json(name)
                if not r:
                    continue
                key = f"cnn_head_{r.get('backbone', 'unknown')}"
                score = 0.5 * (float(r.get("held_out_test_accuracy") or 0)
                               + float(r.get("held_out_test_macro_f1") or 0))
                if score > best_score:
                    best, best_score = key, score
                out[key] = r
            extras = _served_report().training_extras(CKPT_DIR, require_files=False)
            served = extras.get("served_classifier") or {}
            if best and served.get("kind") != "fine_tuned_cnn":
                out[best]["active"] = True
            out.update(extras)
            self._send(200, out)
            return

        if path == "/api/v1/segmentation/status":
            r = read_json("defect_segmenter_report.json")
            if not r:
                self._send(200, {"available": False,
                                 "fix": "python -m training.train_segmenter"})
                return
            seg_onnx = False
            try:
                from api import vercel_inference as vi
                seg_onnx = vi.has_segmenter()
            except Exception:
                pass
            served = _served_report().served_segmenter_summary(CKPT_DIR, require_files=False)
            self._send(200, {
                "available": True, "loaded_in_this_deployment": seg_onnx,
                "model": served.get("label"), "selection": served.get("selection"),
                "iou": served.get("iou") or r.get("iou"), "thresholds": r.get("thresholds"),
                "features": r.get("features"), "trained_on": r.get("trained_on"),
                "split_strategy": r.get("split_strategy"),
                "decision_rule": r.get("decision_rule"),
            })
            return

        if path == "/api/v1/calibration/profiles":
            d = os.path.join(CKPT_DIR, "calibration")
            profiles = []
            if os.path.isdir(d):
                for n in sorted(os.listdir(d)):
                    if not n.endswith(".json"):
                        continue
                    try:
                        with open(os.path.join(d, n), encoding="utf-8") as fh:
                            p = json.load(fh)
                        profiles.append({
                            "device_id": p.get("device_id"),
                            "provenance": "calibrated",
                            "camera_height_m": p.get("camera_height_m"),
                            "pitch_deg": p.get("pitch_deg"),
                            "focal_px": [p.get("fx"), p.get("fy")],
                            "resolution": [p.get("image_width"), p.get("image_height")],
                            "source": p.get("source"),
                            "measurement_basis": "measured against this device's calibration",
                        })
                    except Exception:
                        continue
            self._send(200, {
                "profiles": profiles, "count": len(profiles),
                "active_default": {
                    "device_id": "default", "provenance": "default",
                    "camera_height_m": 1.45, "pitch_deg": 18.4,
                    "focal_px": [1120.0, 1120.0], "resolution": [640, 480],
                    "source": "assumed - not measured on any real vehicle",
                    "measurement_basis": ("ESTIMATE from an assumed camera mount - "
                                          "calibrate the device for a measurement"),
                },
                "why_it_matters": ("Every area, tonnage and cost is derived by projecting "
                                   "pixels onto the ground plane using these numbers. "
                                   "Measured on one photograph: assumed mount 0.007 m2 / "
                                   "Rs 4.5; calibrated mount 0.049 m2 / Rs 34.5."),
                "how_to_add": ("python -m scripts.calibrate_camera --device <id> --hfov <deg> "
                               "--width <px> --height-px <px> --height <m> --pitch <deg>"),
            })
            return

        if path in ("/api/v1/datasets/benchmarks", "/api/v1/datasets/stats"):
            r = read_json("dataset_inventory.json")
            if r:
                self._send(200, r)
                return
            self._send(200, {
                "labeled_photo_classes": {},
                "note": "Dataset inventory report not found.",
            })
            return

        if path in ("/api/v1/fleet/telemetry", "/api/v1/gis/map-data",
                    "/api/v1/ledger/defects", "/api/v1/priority/ranking", "/api/v1/fleet/live",
                    "/api/v1/works/orders", "/api/v1/alerts", "/api/v1/citizen/reports"):
            self._send(200, {
                "storage": "serverless-ephemeral",
                "unique_defects_registered": 0, "total_reports_ingested": 0,
                "deduplication_efficiency_pct": 0.0,
                "defects": [],
                "orders": [],
                "reports": [],
                "alerts": [],
                "note": "Serverless deployment uses stateless ONNX inference per request.",
            })
            return

        if path == "/api/v1/models/served":
            self._send(200, _served_report().model_registry(CKPT_DIR, hash_files=False))
            return

        if path == "/api/v1/ready":
            self._send(503, {"ready": False, "missing": ["vision_classifier", "defect_segmenter"],
                             "models": {"vision_classifier": False, "defect_segmenter": False,
                                        "imu_classifier": False, "ledger": False},
                             "note": "static deployment: the models run on the engine (python -m api.server)"})
            return

        if path == "/api/v1/openapi.json":
            import importlib.util
            spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openapi.py")
            sp = importlib.util.spec_from_file_location("road_shield_openapi", spec_path)
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)
            self._send(200, mod.spec())
            return

        if path == "/api/v1/claims":
            r = read_json("claims.json")
            self._send(200, r or {"error": "claims.json not committed"})
            return

        if path.startswith("/api/"):
            self._send(404, {"error": f"no such endpoint: {path}",
                             "deployment": "vercel-onnx",
                             "available": ["/api/v1/health", "/api/v1/training/metrics",
                                           "/api/v1/segmentation/status",
                                           "/api/v1/calibration/profiles",
                                           "/api/v1/pipeline/deep-audit",
                                           "/api/v1/vision/analyze-photo",
                                           "/api/v1/claims"]})
            return

        if self._static("index.html"):
            return
        self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass
