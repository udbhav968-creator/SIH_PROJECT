"""
The static Vercel deployment (api/vercel_app.py): served over real HTTP.

This module has no coverage elsewhere, which is exactly how vercel.json
pointing at the wrong file (the ~440 MB engine, not this one) went unnoticed.
It also pins that this file stays import-light: Vercel's serverless limit is
250 MB, and the whole point of vercel_app.py is that it never gets near it.
"""

import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

HEAVY_MODULES = ("numpy", "sklearn", "cv2", "onnxruntime", "torch", "ultralytics", "scipy", "skimage")


class ImportWeightTests(unittest.TestCase):
    def test_vercel_app_imports_stdlib_only(self):
        """The module's own import statements, not just what happens to be installed here."""
        import ast
        path = os.path.join(os.path.dirname(__file__), "..", "api", "vercel_app.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        names = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        heavy = names & set(HEAVY_MODULES)
        self.assertEqual(heavy, set(), f"vercel_app.py must stay stdlib-only; found: {heavy}")


class VercelConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _read_vercel_json(self):
        with open(os.path.join(self.root, "vercel.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def test_vercel_json_points_at_the_lightweight_app(self):
        config = self._read_vercel_json()
        build_sources = [b["src"] for b in config["builds"] if b.get("use") == "@vercel/python"]
        self.assertEqual(build_sources, ["api/vercel_app.py"])
        api_routes = [r["dest"] for r in config["routes"] if r["src"] == "/api/(.*)"]
        self.assertEqual(api_routes, ["api/vercel_app.py"])

    def test_every_page_route_has_a_vercel_json_entry(self):
        from api.vercel_app import PAGE_ROUTES
        routed = {r["src"] for r in self._read_vercel_json()["routes"]}
        for route in PAGE_ROUTES:
            if route == "/":
                continue  # the root has its own catch-all entry, not a literal "/"
            self.assertIn(route, routed, f"{route} is in PAGE_ROUTES but vercel.json has no route for it")

    def test_detector_onnx_is_not_shipped_to_a_handler_that_cannot_load_it(self):
        with open(os.path.join(self.root, ".vercelignore"), encoding="utf-8") as handle:
            ignore = handle.read()
        self.assertIn("checkpoints/detectors/*.onnx", ignore)


class VercelAppOverHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api.vercel_app import handler
        cls.root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.server = HTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as response:
                return response.status, json.loads(response.read()), response.headers
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, json.loads(exc.read() or b"{}"), exc.headers

    def test_detect_page_is_served(self):
        with urllib.request.urlopen(self.base + "/detect", timeout=10) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/html", response.headers["Content-Type"])
            self.assertIn("/api/v1/perception/analyze", response.read().decode("utf-8"))

    def test_perception_status_matches_the_live_engines_response_shape(self):
        status, body, _ = self.get("/api/v1/perception/status")
        self.assertEqual(status, 200)
        self.assertEqual(set(body["models"]), {"traffic", "damage", "markings", "privacy"})
        # Nothing is ever loaded in this deployment: every head must say so.
        for head in body["models"].values():
            self.assertFalse(head["ready"])
            self.assertNotIn("backend", head)   # only ever present for a loaded detector

    def test_trained_heads_report_real_measured_scores(self):
        """Skips, visibly, on a checkout that has not trained these yet - it does not fabricate scores."""
        _, body, _ = self.get("/api/v1/perception/status")
        for key, card_name in (("damage", "road_damage"), ("markings", "crosswalk")):
            card_path = os.path.join(self.root, "checkpoints", "detectors", f"{card_name}.json")
            if not os.path.exists(card_path):
                self.skipTest(f"{card_name} not trained on this checkout")
            with open(card_path, encoding="utf-8") as handle:
                card = json.load(handle)
            reported = body["models"][key]["test_metrics"]["mAP50"]
            self.assertEqual(reported, card["metrics"]["test"]["mAP50"])

    def test_perception_analyze_and_traffic_analyze_explain_the_size_limit_instead_of_404(self):
        for path in ("/api/v1/perception/analyze", "/api/v1/traffic/analyze"):
            status, body, _ = self.get(path)
            self.assertEqual(status, 503, path)
            self.assertIn("250 MB", body["why"])
            self.assertIn("run_it_yourself", body)

    def test_health_reports_the_three_new_detectors(self):
        _, body, _ = self.get("/api/v1/health")
        for key in ("road_damage_detector", "crosswalk_detector", "license_plate_detector"):
            self.assertIn(key, body["models"])

    def test_unknown_api_path_lists_perception_status_as_available(self):
        status, body, _ = self.get("/api/v1/nonsense")
        self.assertEqual(status, 404)
        self.assertIn("/api/v1/perception/status", body["available"])

    def test_cors_headers_present_for_post(self):
        _, _, headers = self.get("/api/v1/health")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "*")


if __name__ == "__main__":
    unittest.main()
