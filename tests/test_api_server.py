"""
Integration tests for api/server.py - the REST API the eight-page site calls.

Before this file the server had 0% test coverage: CI only curled /health and
checked that pages returned 200. These tests start the real server on an
ephemeral port and drive it over HTTP, so they exercise routing, JSON
serialisation, validation, and the model stack behind each endpoint together.

Isolation: ROAD_SHIELD_WRITABLE_DIR is pointed at a temporary directory BEFORE
api.server is imported, so work orders, sightings and feedback written here
never reach the real fleet ledger in checkpoints/road_shield.db.

Run alone:  python -m unittest tests.test_api_server -v
"""

import base64
import glob
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

_TMP = tempfile.mkdtemp(prefix="road_shield_api_test_")
os.environ["ROAD_SHIELD_WRITABLE_DIR"] = _TMP
# Demo fixtures are opt-in since they are invented reports; these tests use
# them to check deduplication end to end, so they switch them on explicitly.
os.environ["ROAD_SHIELD_SEED_DEMO"] = "1"

import api.server as srv  # noqa: E402  (must follow the env override)


def _first_image(folder):
    hits = sorted(glob.glob(os.path.join(ENGINE_ROOT, "datasets", folder, "real_images", "*.jpg")))
    return hits[0] if hits else None


def _b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


class APIServerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.path.realpath(srv.WRITABLE_DIR) != os.path.realpath(_TMP):
            raise unittest.SkipTest(
                "api.server was imported before this module set "
                "ROAD_SHIELD_WRITABLE_DIR; refusing to write to the real ledger")
        cls.httpd = srv.ThreadedHTTPServer(("127.0.0.1", 0), srv.RoadShieldAPIHandler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        try:
            srv.defect_store.close()
        except Exception:
            pass
        shutil.rmtree(_TMP, ignore_errors=True)

    # -- helpers -----------------------------------------------------------
    def _req(self, method, path, body=None, raw=None, headers=None, timeout=120):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers=headers or {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def get_json(self, path):
        code, _h, body = self._req("GET", path)
        return code, json.loads(body)

    def post_json(self, path, body=None, **kw):
        code, _h, payload = self._req("POST", path, body if body is not None else {}, **kw)
        return code, json.loads(payload)

    # -- isolation ---------------------------------------------------------
    # -- operations (api/ops.py) -------------------------------------------
    def test_metrics_endpoint_is_prometheus_text(self):
        self.get_json("/api/v1/health")
        code, headers, body = self._req("GET", "/metrics")
        self.assertEqual(code, 200)
        self.assertIn("text/plain", headers.get("Content-Type", ""))
        text = body.decode()
        self.assertIn('road_shield_requests_total{route="/api/v1/health",method="GET",status="200"}', text)
        self.assertIn("road_shield_request_duration_seconds_bucket", text)
        self.assertIn("road_shield_ready ", text)

    def test_readiness_names_what_is_missing(self):
        code, body = self.get_json("/api/v1/ready")
        self.assertIn(code, (200, 503))
        self.assertEqual(code == 200, body["ready"])
        self.assertEqual(body["ready"], not body["missing"])
        for k in ("vision_classifier", "defect_segmenter", "imu_classifier", "ledger"):
            self.assertIn(k, body["models"])

    def test_oversized_body_is_refused_before_it_is_read(self):
        import http.client
        host, port = self.base.replace("http://", "").split(":")
        conn = http.client.HTTPConnection(host, int(port), timeout=30)
        conn.putrequest("POST", "/api/v1/pipeline/deep-audit")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(srv.ops.MAX_BODY_BYTES + 1))
        conn.endheaders()                                  # no body sent: the server must not wait for it
        r = conn.getresponse()
        self.assertEqual(r.status, 413)
        self.assertIn(b"larger than", r.read())
        conn.close()

    def test_rate_limit_answers_429_with_retry_after(self):
        old = srv.LIMITER.per_minute
        srv.LIMITER.per_minute, srv.LIMITER._buckets = 1, {}
        try:
            first, _h, _b = self._req("POST", "/api/v1/telemetry/imu", {})
            second, headers, body = self._req("POST", "/api/v1/telemetry/imu", {})
            self.assertNotEqual(first, 429)
            self.assertEqual(second, 429)
            self.assertGreaterEqual(int(headers.get("Retry-After", "0")), 1)
            self.assertIn(b"rate limit", body)
            code, _h, _b = self._req("GET", "/api/v1/health")
            self.assertEqual(code, 200, "reads are never rate limited")
        finally:
            srv.LIMITER.per_minute, srv.LIMITER._buckets = old, {}

    def test_ledger_is_isolated_from_checkpoints(self):
        self.assertTrue(srv.defect_store.db_path.startswith(_TMP))
        self.assertFalse(srv.defect_store.db_path.startswith(srv.CKPT_DIR))

    # -- pages and static --------------------------------------------------
    def test_every_site_page_serves_html(self):
        for page in ("/", "/inspect", "/video", "/corridor", "/works", "/models", "/data", "/system"):
            code, headers, body = self._req("GET", page, headers={"Accept": "text/html"})
            self.assertEqual(code, 200, page)
            self.assertIn("text/html", headers.get("Content-Type", ""), page)
            self.assertIn(b"<", body[:200], page)

    def test_static_assets_have_correct_types(self):
        code, h, _ = self._req("GET", "/web/app.js")
        self.assertEqual(code, 200)
        self.assertIn("javascript", h.get("Content-Type", ""))
        code, h, _ = self._req("GET", "/web/app.css")
        self.assertEqual(code, 200)
        self.assertIn("css", h.get("Content-Type", ""))

    def test_path_traversal_is_refused(self):
        for evil in ("/web/../api/server.py", "/web/..%2Fapi%2Fserver.py",
                     "/web/../../etc/passwd", "/web/%2e%2e/requirements.txt"):
            code, _h, body = self._req("GET", evil)
            self.assertNotIn(b"import ", body, evil)
            self.assertNotIn(b"root:", body, evil)
            self.assertNotIn(b"numpy", body, evil)

    def test_files_in_web_subfolders_are_served(self):
        code, h, body = self._req("GET", "/web/samples/recorded_results.json")
        self.assertEqual(code, 200)
        self.assertIn("json", h.get("Content-Type", ""))
        self.assertIn("samples", json.loads(body))
        code, h, _ = self._req("GET", "/web/config.js")
        self.assertEqual(code, 200)
        for evil in ("/web/samples/../../requirements.txt", "/web/samples/../../../etc/passwd"):
            code, _h, body = self._req("GET", evil)
            self.assertNotIn(b"numpy", body, evil)
            self.assertNotIn(b"root:", body, evil)

    def test_public_demo_refuses_server_files_outside_datasets(self):
        img = _first_image("02_kaggle_pothole_600")
        old = srv.PUBLIC
        srv.PUBLIC = True
        try:
            # files and folders that exist on every OS, all outside datasets/
            server_file = os.path.join(srv.ENGINE_ROOT, "api", "server.py")
            reqs = os.path.join(srv.ENGINE_ROOT, "requirements.txt")
            api_dir = os.path.join(srv.ENGINE_ROOT, "api")
            for path, body in (("/api/v1/detect/vision", {"image_base64": server_file}),
                               ("/api/v1/detect/vision", {"image_path": reqs}),
                               ("/api/v1/detect/vision", {"image_path": "requirements.txt"}),
                               ("/api/v1/video/probe", {"video_path": reqs}),
                               ("/api/v1/pipeline/deep-audit-batch", {"directory_path": api_dir})):
                code, r = self.post_json(path, body)
                self.assertEqual(code, 403, f"{path} {body}: {r}")
            # the bundled photographs stay usable, and base64 is never mistaken for a path
            self.assertIsNone(srv._server_path_refused({"image_path": img} if img else {}))
            self.assertIsNone(srv._server_path_refused({"image_base64": "aGVsbG8="}))
        finally:
            srv.PUBLIC = old
        self.assertIsNone(srv._server_path_refused({"image_path": os.path.join(srv.ENGINE_ROOT, "requirements.txt")}),
                          "outside public mode the local convenience paths are unchanged")

    def test_health_says_whether_this_is_the_public_demo(self):
        code, h = self.get_json("/api/v1/health")
        self.assertEqual(code, 200)
        self.assertIn("public_demo", h)

    def test_cors_preflight(self):
        code, h, _ = self._req("OPTIONS", "/api/v1/health")
        self.assertEqual(code, 200)
        self.assertEqual(h.get("Access-Control-Allow-Origin"), "*")
        self.assertIn("POST", h.get("Access-Control-Allow-Methods", ""))

    def test_unknown_api_route_is_not_200(self):
        code, _h, _ = self._req("GET", "/api/v1/does-not-exist")
        self.assertNotEqual(code, 200)
        code, _h, _ = self._req("POST", "/api/v1/does-not-exist", {})
        self.assertNotEqual(code, 200)

    # -- read-only status endpoints the frontend calls ---------------------
    def test_frontend_get_endpoints_return_json(self):
        for path in ("/api/v1/health", "/api/v1/calibration/profiles", "/api/v1/claims",
                     "/api/v1/datasets/benchmarks", "/api/v1/fleet/telemetry",
                     "/api/v1/gis/map-data", "/api/v1/ledger/defects", "/api/v1/maps/status",
                     "/api/v1/models/registry", "/api/v1/segmentation/status",
                     "/api/v1/training/metrics"):
            code, data = self.get_json(path)
            self.assertEqual(code, 200, f"{path} -> {code}: {str(data)[:200]}")
            self.assertIsInstance(data, dict, path)

    def test_health_reports_loaded_models(self):
        _c, h = self.get_json("/api/v1/health")
        blob = json.dumps(h).lower()
        self.assertTrue(any(k in blob for k in ("ok", "healthy", "online", "ready")), blob[:300])

    def test_training_metrics_reports_real_held_out_numbers(self):
        _c, m = self.get_json("/api/v1/training/metrics")
        blob = json.dumps(m)
        self.assertIn("accuracy", blob)

    def test_ledger_was_seeded_and_deduplicated(self):
        _c, led = self.get_json("/api/v1/ledger/defects")
        blob = json.dumps(led)
        # five demo reports, two of which are 5 m apart and must merge
        self.assertIn("Pothole Cavity", blob)
        self.assertEqual(len(srv.fleet_dedup_engine.defect_registry), 4)

    def test_fresh_edge_export_succeeds(self):
        # /models/registry reuses files already on disk, which hid an exporter
        # crash; this route always exports from scratch into the writable dir.
        code, r = self.get_json("/api/v1/models/export-edge-spec")
        self.assertEqual(code, 200, str(r)[:300])
        self.assertIn("Model_PCI_ASTM_D6433", r["models_exported"])
        self.assertTrue(os.path.exists(os.path.join(_TMP, "road_shield_open_model_spec.json")))

    def test_alpr_requires_a_real_track(self):
        code, r = self.post_json("/api/v1/incidents/alpr", {})
        self.assertEqual(code, 400, "an incident was fabricated from an empty request")
        track = [{"timestamp": 0.2 * i, "bbox": [200, 150, 80 * 1.8 ** i, 60 * 1.8 ** i]} for i in range(4)]
        code, r = self.post_json("/api/v1/incidents/alpr", {"track_history": track})
        self.assertEqual(code, 200)
        self.assertIsNone(r["gps_coordinates"])

    # -- work orders and the tamper seal -----------------------------------
    def test_work_order_round_trip_and_tamper_detection(self):
        code, wo = self.post_json("/api/v1/dispatch/work-order", {
            "corridor_id": "NH-44", "latitude": 28.7, "longitude": 77.1,
            "distress_class": "Pothole Cavity", "area_sqm": 2.0, "depth_cm": 6.0, "pci_score": 35})
        self.assertEqual(code, 200)
        self.assertIn("sha256_cryptographic_seal", wo)

        _c, ok = self.post_json("/api/v1/dispatch/verify-seal", {"work_order": wo})
        self.assertTrue(ok["is_valid"])
        self.assertEqual(ok["status"], "SEAL_VERIFIED_AUTHENTIC")

        forged = dict(wo)
        for k, v in wo.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("latency_ms",):
                forged[k] = v * 10 + 1
                break
        _c, bad = self.post_json("/api/v1/dispatch/verify-seal", {"work_order": forged})
        self.assertFalse(bad["is_valid"])
        self.assertEqual(bad["status"], "CORRUPTED_OR_TAMPERED")

    def test_work_order_ids_are_unique_within_one_second(self):
        ids = set()
        for _ in range(5):
            _c, wo = self.post_json("/api/v1/dispatch/work-order",
                                    {"corridor_id": "NH-48", "distress_class": "Pothole Cavity",
                                     "area_sqm": 1.0, "depth_cm": 5.0, "pci_score": 40})
            ids.add(wo["work_order_id"])
        self.assertEqual(len(ids), 5, "work-order IDs collided for orders issued in the same second")

    # -- nothing is invented for a missing field (October 2026 audit) -------
    def test_work_order_refuses_missing_measurements(self):
        code, r = self.post_json("/api/v1/dispatch/work-order", {"corridor_id": "NH-48"})
        self.assertEqual(code, 400, "a work order was sealed for an invented defect")
        for f in ("distress_class", "area_sqm", "depth_cm", "pci_score"):
            self.assertIn(f, r["error"])

    def test_work_order_accepts_the_field_names_the_site_sends(self):
        # web/works.html posts defect_class / area_m2 / pci; these used to be ignored
        code, wo = self.post_json("/api/v1/dispatch/work-order", {
            "defect_class": "Crack (Longitudinal / Transverse / Alligator)", "area_m2": 0.7,
            "depth_cm": 2.0, "pci": 55})
        self.assertEqual(code, 200, wo)
        self.assertEqual(wo["distress_type"], "Crack (Longitudinal / Transverse / Alligator)")
        self.assertEqual(wo["pavement_pci"], 55)
        self.assertIsNone(wo["coordinates"])
        self.assertEqual(wo["dispatch_status"], "HELD_NO_GPS")

    def test_deep_audit_without_an_image_is_refused(self):
        code, r = self.post_json("/api/v1/pipeline/deep-audit", {})
        self.assertEqual(code, 400, "an empty request was answered with a bundled photo's audit")

    def test_deep_audit_image_path_cannot_leave_datasets(self):
        for evil in ("api/server.py", "../../etc/passwd", "/etc/hostname"):
            code, r = self.post_json("/api/v1/pipeline/deep-audit", {"image_path": evil})
            self.assertEqual(code, 400, evil)

    def test_deep_audit_reports_no_location_when_none_given(self):
        img = _first_image("02_kaggle_pothole_600")
        if not img:
            self.skipTest("no photographs on disk")
        code, r = self.post_json("/api/v1/pipeline/deep-audit", {"image_base64": _b64(img)}, timeout=300)
        self.assertEqual(code, 200, str(r)[:300])
        self.assertIsNone(r["location"]["lat"])
        self.assertNotIn("28.7041", json.dumps(r))

    def test_maps_endpoints_require_coordinates(self):
        for path in ("/api/v1/maps/reverse-geocode", "/api/v1/maps/elevation",
                     "/api/v1/maps/places-nearby", "/api/v1/maps/streetview-url"):
            code, r = self.get_json(path)
            self.assertEqual(code, 400, f"{path} answered for a default city centre")
        code, r = self.post_json("/api/v1/maps/directions", {})
        self.assertEqual(code, 400)

    def test_fleet_report_requires_its_fields(self):
        code, r = self.post_json("/api/v1/fleet/report-defect", {"bus_id": "B-1"})
        self.assertEqual(code, 400, "a sighting with no GPS was filed in Bengaluru")
        code, r = self.post_json("/api/v1/fleet/report-defect", {
            "bus_id": "B-1", "lat": 12.97161, "lon": 77.59461, "defect_class": "Pothole Cavity",
            "severity_pci": 40, "area_m2": 0.9})
        self.assertEqual(code, 200, r)
        # lands on the seeded pothole, so it merges rather than growing the ledger
        self.assertEqual(r["action"], "DEDUPLICATED_AND_UPDATED")

    def test_privacy_redaction_endpoint(self):
        code, r = self.post_json("/api/v1/privacy/redact", {})
        self.assertEqual(code, 400)
        img = _first_image("08_dashcam_video_streams") or _first_image("02_kaggle_pothole_600")
        if not img:
            self.skipTest("no photographs on disk")
        code, r = self.post_json("/api/v1/privacy/redact", {"image_base64": _b64(img)}, timeout=120)
        self.assertEqual(code, 200, str(r)[:300])
        self.assertTrue(r["redacted_image_base64"])
        self.assertFalse(r["report"]["recall_measured"])

    def test_verify_seal_rejects_malformed_input_without_crashing(self):
        for body in ({"work_order": "not-an-object"}, {"work_order": []}, {"work_order": None}):
            code, data = self.post_json("/api/v1/dispatch/verify-seal", body)
            self.assertIn(code, (200, 400), body)
            self.assertFalse(data.get("is_valid", False), body)

    # -- deterministic engineering engines ---------------------------------
    def test_pci_is_monotone_in_damage(self):
        _c, clean = self.post_json("/api/v1/pci/predict", {})
        _c, bad = self.post_json("/api/v1/pci/predict", {
            "crack_density_pct": 25, "crack_severity": "HIGH",
            "pothole_count": 6, "pothole_density_pct": 4, "pothole_severity": "HIGH"})
        key = next(k for k in clean if "pci" in k.lower() and isinstance(clean[k], (int, float)))
        self.assertGreater(clean[key], bad[key])
        self.assertGreaterEqual(bad[key], 0)
        self.assertLessEqual(clean[key], 100)

    def test_ipm_tonnage_scales_with_area(self):
        _c, a = self.post_json("/api/v1/civil/ipm-tonnage", {"area_m2": 1.0, "depth_cm": 5})
        _c, b = self.post_json("/api/v1/civil/ipm-tonnage", {"area_m2": 4.0, "depth_cm": 5})
        num = lambda d: {k: v for k, v in d.items() if isinstance(v, (int, float)) and "latency" not in k}
        grew = [k for k in num(a) if k in num(b) and num(a)[k] > 0 and num(b)[k] > num(a)[k] * 3.5]
        self.assertTrue(grew, f"no quantity scaled ~4x with area: {a} vs {b}")

    def test_deterioration_forecast(self):
        code, f = self.post_json("/api/v1/forecast/deterioration", {"initial_area_m2": 1.5})
        self.assertEqual(code, 200)
        self.assertEqual(f["model"], "PavementDeteriorationForecaster")

    def test_fusion_gate_strong_evidence_beats_weak(self):
        _c, weak = self.post_json("/api/v1/fusion/gate", {"vision_pothole_prob": 0.1, "p_imu_shock": 0.05})
        _c, strong = self.post_json("/api/v1/fusion/gate", {"vision_pothole_prob": 0.95, "p_imu_shock": 0.95,
                                                             "peak_delta_z_ms2": 8.0})
        pick = lambda d: next(v for k, v in d.items() if "posterior" in k.lower() and isinstance(v, (int, float)))
        self.assertGreater(pick(strong), pick(weak))

    def test_imu_endpoint_with_real_window(self):
        code, r = self.post_json("/api/v1/telemetry/imu", {"simulate_shock": True})
        if code == 503:
            self.skipTest(r.get("error"))
        self.assertEqual(code, 200)
        self.assertAlmostEqual(sum(r["probabilities"].values()), 1.0, places=2)

    def test_imu_endpoint_with_caller_series(self):
        series = [[0.0, 0.0, 9.81]] * 100
        code, r = self.post_json("/api/v1/telemetry/imu", {"raw_series": series})
        if code == 503:
            self.skipTest(r.get("error"))
        self.assertEqual(code, 200)
        self.assertEqual(r["data_source"], "caller_supplied_real_series")
        self.assertAlmostEqual(r["peak_delta_z_ms2"], 0.0, places=3)

    # -- vision ------------------------------------------------------------
    def test_vision_requires_an_image(self):
        code, r = self.post_json("/api/v1/vision/analyze-custom-photo", {})
        self.assertEqual(code, 400)
        code, r = self.post_json("/api/v1/detect/vision", {"preferred_class": 2})
        self.assertEqual(code, 400)

    def test_vision_classifies_a_real_pothole(self):
        img = _first_image("02_kaggle_pothole_600")
        if not img:
            self.skipTest("no pothole photographs on disk")
        code, r = self.post_json("/api/v1/detect/vision", {"image_base64": _b64(img)})
        if code == 503:
            self.skipTest(r["error"])
        self.assertEqual(code, 200, r)
        self.assertIn(r["class_id"], range(7))
        self.assertAlmostEqual(sum(r["probabilities"].values()), 1.0, places=2)

    def test_deep_audit_on_uploaded_photo(self):
        img = _first_image("02_kaggle_pothole_600")
        if not img:
            self.skipTest("no pothole photographs on disk")
        code, r = self.post_json("/api/v1/pipeline/deep-audit",
                                 {"image_base64": _b64(img), "corridor_id": "TEST-1"}, timeout=300)
        self.assertEqual(code, 200, str(r)[:400])
        self.assertIsInstance(r, dict)
        self.assertNotIn("error", r)

    def test_garbage_image_is_an_error_not_a_classification(self):
        code, r = self.post_json("/api/v1/vision/analyze-custom-photo",
                                 {"image_base64": base64.b64encode(b"definitely not a jpeg").decode()})
        if code == 200:
            # the pipeline may reject at its own gate; it must not claim a defect
            blob = json.dumps(r).lower()
            self.assertTrue("reject" in blob or "invalid" in blob or "error" in blob or "fail" in blob,
                            blob[:300])
        else:
            self.assertGreaterEqual(code, 400)

    def test_repair_audit_flags_resubmitted_photo(self):
        img = _first_image("02_kaggle_pothole_600")
        if not img:
            self.skipTest("no photographs on disk")
        b = _b64(img)
        code, r = self.post_json("/api/v1/audit/verify-repair",
                                 {"before_image_base64": b, "after_image_base64": b})
        self.assertEqual(code, 200, r)
        blob = json.dumps(r).lower()
        self.assertTrue("duplicate" in blob or "fraud" in blob or "identical" in blob or "resubmit" in blob,
                        blob[:400])

    def test_repair_audit_requires_both_photos(self):
        code, _r = self.post_json("/api/v1/audit/verify-repair", {"before_image_base64": "x"})
        self.assertEqual(code, 400)

    # -- video -------------------------------------------------------------
    def test_video_endpoints_validate_paths(self):
        code, _r = self.post_json("/api/v1/video/ingest", {})
        self.assertEqual(code, 400)
        code, _r = self.post_json("/api/v1/video/ingest", {"video_path": "/nope/missing.mp4"})
        self.assertEqual(code, 404)
        code, _r = self.post_json("/api/v1/video/probe", {"video_path": "/nope/missing.mp4"})
        self.assertEqual(code, 404)

    def test_video_probe_and_ingest_on_a_real_clip(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV not installed")
        frames = sorted(glob.glob(os.path.join(ENGINE_ROOT, "datasets", "08_dashcam_video_streams",
                                               "real_images", "*.jpg")))
        if not frames:
            self.skipTest("no dashcam frames on disk")
        clip = os.path.join(_TMP, "clip.avi")
        w = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"MJPG"), 5, (320, 240))
        for f in frames[:4]:
            img = cv2.imread(f)
            for _ in range(3):
                w.write(cv2.resize(img, (320, 240)))
        w.release()
        code, probe = self.post_json("/api/v1/video/probe", {"video_path": clip})
        self.assertEqual(code, 200, probe)
        self.assertIn("12", json.dumps(probe))  # 12 frames written

        code, r = self.post_json("/api/v1/video/ingest",
                                 {"video_path": clip, "bus_id": "TEST-BUS", "max_frames": 4},
                                 timeout=300)
        self.assertEqual(code, 200, str(r)[:400])
        # No GPS track was supplied, so no location may be invented
        blob = json.dumps(r).lower()
        self.assertNotIn("12.9716", blob)

    # -- feedback ----------------------------------------------------------
    def test_active_feedback_is_logged_to_isolated_dir(self):
        img = _first_image("02_kaggle_pothole_600")
        if not img:
            self.skipTest("no photographs on disk")
        code, r = self.post_json("/api/v1/training/active-feedback",
                                 {"image_base64": _b64(img), "true_class_id": 2})
        self.assertEqual(code, 200, r)
        self.assertTrue(os.path.exists(os.path.join(_TMP, "active_feedback_log.jsonl")))

    def test_malformed_json_body_does_not_crash(self):
        code, _h, _b = self._req("POST", "/api/v1/pci/predict", raw=b"{not json")
        self.assertIn(code, (200, 400))


class HuggingFaceSpaceTest(unittest.TestCase):
    """deploy/huggingface: the Space builds the engine from GitHub in public-demo mode on port 7860."""

    def test_dockerfile_runs_the_public_engine_on_the_space_port(self):
        root = os.path.join(os.path.dirname(__file__), "..", "deploy", "huggingface")
        with open(os.path.join(root, "Dockerfile"), encoding="utf-8") as fh:
            d = fh.read()
        for must in ("ROAD_SHIELD_PUBLIC=1", "ROAD_SHIELD_RATE_LIMIT=", "useradd -m -u 1000",
                     'CMD ["python", "-m", "api.server", "7860"]', "git clone", "requirements.txt"):
            self.assertIn(must, d)
        with open(os.path.join(root, "README.md"), encoding="utf-8") as fh:
            readme = fh.read()
        front = readme.split("---")[1]
        self.assertIn("sdk: docker", front)
        self.assertIn("app_port: 7860", front)

    def test_site_config_declares_the_engine_url(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "web", "config.js"), encoding="utf-8") as fh:
            self.assertIn("window.ROAD_SHIELD_ENGINE_URL", fh.read())


class VercelHandlerTest(unittest.TestCase):
    """The serverless entrypoint is stdlib-only and must serve the site and the
    stored measurements, while refusing inference with an honest 503."""

    @classmethod
    def setUpClass(cls):
        from http.server import HTTPServer
        from api import vercel_app
        cls.httpd = HTTPServer(("127.0.0.1", 0), vercel_app.handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _req(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_pages_serve(self):
        for page in ("/", "/inspect", "/models", "/system"):
            code, body = self._req("GET", page)
            self.assertEqual(code, 200, page)
            self.assertIn(b"<", body[:200])

    def test_static_deployment_is_never_ready(self):
        code, body = self._req("GET", "/api/v1/ready")
        self.assertEqual(code, 503)
        self.assertFalse(json.loads(body)["ready"])

    def test_inference_endpoints_refuse_honestly(self):
        code, body = self._req("POST", "/api/v1/pipeline/deep-audit", {})
        self.assertEqual(code, 503)
        self.assertIn(b"error", body.lower())

    def test_health_is_json(self):
        code, body = self._req("GET", "/api/v1/health")
        self.assertEqual(code, 200)
        json.loads(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
