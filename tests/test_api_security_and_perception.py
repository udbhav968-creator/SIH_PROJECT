"""
API hardening and the perception endpoints, exercised over real HTTP.

The server is started in-process on an ephemeral port with its writable
directory redirected to a temp folder, so the tracked ledger in checkpoints/
is never touched by a test run.
"""

import base64
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api import request_images as ri


def png_b64(width=64, height=48, colour=(90, 90, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class RequestImageUnitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "datasets")
        os.makedirs(os.path.join(self.root, "class_a"))
        self.inside = os.path.join(self.root, "class_a", "img.png")
        Image.new("RGB", (8, 8)).save(self.inside)
        self.outside = os.path.join(self.tmp.name, "secret.png")
        Image.new("RGB", (8, 8)).save(self.outside)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_path_in_the_base64_field_is_never_opened(self):
        with self.assertRaises(ri.RequestImageError) as ctx:
            ri.request_image({"image_base64": self.outside}, self.root)
        self.assertEqual(ctx.exception.status, 400)

    def test_path_outside_datasets_is_forbidden(self):
        for path in (self.outside, os.path.join(self.root, "..", "secret.png")):
            with self.assertRaises(ri.RequestImageError) as ctx:
                ri.request_image({"image_path": path}, self.root)
            self.assertEqual(ctx.exception.status, 403, path)

    def test_path_inside_datasets_is_allowed_absolute_or_relative(self):
        self.assertEqual(ri.request_image({"image_path": self.inside}, self.root), os.path.realpath(self.inside))
        self.assertEqual(ri.request_image({"image_path": "class_a/img.png"}, self.root),
                         os.path.realpath(self.inside))

    def test_missing_file_inside_datasets_is_404(self):
        with self.assertRaises(ri.RequestImageError) as ctx:
            ri.request_image({"image_path": "class_a/nope.png"}, self.root)
        self.assertEqual(ctx.exception.status, 404)

    def test_data_uri_and_wrapped_base64_decode(self):
        raw = ri.decode_base64_image("data:image/png;base64," + png_b64())
        self.assertTrue(raw.startswith(b"\x89PNG"))
        wrapped = "\n".join(png_b64()[i:i + 60] for i in range(0, len(png_b64()), 60))
        self.assertTrue(ri.decode_base64_image(wrapped).startswith(b"\x89PNG"))

    def test_oversized_base64_is_413_before_decoding(self):
        with self.assertRaises(ri.RequestImageError) as ctx:
            ri.decode_base64_image("A" * 2000, max_bytes=1000)
        self.assertEqual(ctx.exception.status, 413)

    def test_garbage_is_rejected(self):
        for value in ("not base64 at all!", "", 12345):
            with self.assertRaises(ri.RequestImageError):
                ri.decode_base64_image(value)

    def test_optional_image_absent_returns_none(self):
        self.assertIsNone(ri.request_image({}, self.root, required=False))
        with self.assertRaises(ri.RequestImageError):
            ri.request_image({}, self.root)

    def test_load_rgb_applies_exif_orientation(self):
        buffer = io.BytesIO()
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise on display
        Image.new("RGB", (40, 20)).save(buffer, format="JPEG", exif=exif)
        self.assertEqual(ri.load_rgb(buffer.getvalue()).shape, (40, 20, 3))

    def test_load_rgb_rejects_non_images(self):
        with self.assertRaises(ri.RequestImageError):
            ri.load_rgb(b"definitely not an image")


class ApiOverHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._data_dir = tempfile.TemporaryDirectory()
        os.environ["ROAD_SHIELD_DATA_DIR"] = cls._data_dir.name
        from api import server
        cls.server_module = server
        cls.httpd = server.ThreadedHTTPServer(("127.0.0.1", 0), server.RoadShieldAPIHandler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        os.environ.pop("ROAD_SHIELD_DATA_DIR", None)
        cls.server_module.defect_store.close()  # release the SQLite file so the temp dir can go
        cls._data_dir.cleanup()

    def call(self, path, body=None, headers=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request = urllib.request.Request(self.base + path, data=data, headers={
            "Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, json.loads(exc.read() or b"{}")

    def test_ledger_went_to_the_redirected_directory(self):
        self.assertTrue(self.server_module.defect_store.db_path.startswith(self._data_dir.name))

    def test_ledger_is_ranked_by_priority_index(self):
        status, body = self.call("/api/v1/ledger/defects")
        self.assertEqual(status, 200)
        defects = body["defects"]
        self.assertTrue(defects, "the server seeds demo defects into an empty ledger")
        scores = [d["priority_index"] for d in defects]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([d["priority_rank"] for d in defects], list(range(1, len(defects) + 1)))
        self.assertFalse(defects[0]["traffic_measured"])  # seeded reports carry no traffic count

    def test_edge_packet_round_trip_and_tamper_rejection(self):
        if not self.server_module.scene_perception.is_ready:
            self.skipTest("no perception model on this machine")
        request = {"image_base64": png_b64(320, 240), "bus_id": "BUS-T1",
                   "lat": 12.9716, "lon": 77.5946, "timestamp": 1790000000}
        with mock.patch.dict(os.environ, {"ROAD_SHIELD_DEVICE_KEY_BUS_T1": "per-bus-secret"}):
            status, body = self.call("/api/v1/edge/encode", request)
            self.assertEqual(status, 200, body)
            self.assertTrue(body["signed"])
            self.assertLessEqual(body["bytes"], 1024)
            status, verified = self.call("/api/v1/edge/verify", {"packet": body["packet"]})
            self.assertEqual(status, 200, verified)
            self.assertTrue(verified["signature_verified"])
            self.assertEqual((verified["bus_id"], verified["lat"]), ("BUS-T1", 12.9716))
            status, _ = self.call("/api/v1/edge/verify", {"packet": body["packet"].replace("12.9716", "13.0")})
            self.assertEqual(status, 400)  # moved the defect: signature no longer matches
        status, _ = self.call("/api/v1/edge/verify", {"packet": "not json"})
        self.assertEqual(status, 400)

    def test_edge_encode_requires_coordinates(self):
        status, _ = self.call("/api/v1/edge/encode", {"image_base64": png_b64()})
        self.assertIn(status, (400, 503))

    def test_health_reports_perception_models(self):
        status, body = self.call("/api/v1/health")
        self.assertEqual(status, 200)
        self.assertIn("road_damage_detector", body["models"])
        self.assertIn("crosswalk_detector", body["models"])

    def test_detect_page_is_served(self):
        with urllib.request.urlopen(self.base + "/detect", timeout=30) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/html", response.headers["Content-Type"])
            self.assertIn("/api/v1/perception/analyze", response.read().decode("utf-8"))

    def test_perception_status(self):
        status, body = self.call("/api/v1/perception/status")
        self.assertEqual(status, 200)
        self.assertEqual(set(body["models"]), {"traffic", "damage", "markings"})

    def test_server_file_cannot_be_read_through_image_path(self):
        target = os.path.abspath(__file__)
        for route in ("/api/v1/perception/analyze", "/api/v1/pipeline/deep-audit", "/api/v1/vision/predict"):
            status, body = self.call(route, {"image_path": target})
            self.assertEqual(status, 403, f"{route}: {body}")

    def test_server_file_cannot_be_read_through_base64_field(self):
        status, _ = self.call("/api/v1/detect/objects", {"image_base64": os.path.abspath(__file__)})
        self.assertIn(status, (400, 503))

    def test_oversized_body_is_refused_without_reading_it(self):
        huge = str(self.server_module.MAX_REQUEST_BYTES + 1)
        status, body = self.call("/api/v1/perception/analyze", raw=b"{}", headers={"Content-Length": huge})
        self.assertEqual(status, 413, body)

    def test_invalid_groups_rejected(self):
        if not self.server_module.scene_perception.is_ready:
            self.skipTest("no perception model on this machine")
        status, _ = self.call("/api/v1/perception/analyze", {"image_base64": png_b64(), "groups": ["lasers"]})
        self.assertEqual(status, 400)

    def test_analyze_returns_full_result_and_annotation(self):
        if not self.server_module.scene_perception.is_ready:
            self.skipTest("no perception model on this machine")
        status, body = self.call("/api/v1/perception/analyze",
                                 {"image_base64": png_b64(320, 240), "return_annotated": True})
        self.assertEqual(status, 200, body)
        for key in ("detections", "counts", "scene", "alerts", "models", "latency_ms", "image_size"):
            self.assertIn(key, body)
        self.assertEqual(body["image_size"], [320, 240])
        header, payload = body["annotated_image_base64"].split(",", 1)
        self.assertEqual(header, "data:image/jpeg;base64")
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(payload))).size, (320, 240))

    def test_pedestrian_endpoint_labels_where_its_facts_came_from(self):
        status, body = self.call("/api/v1/pedestrian/detect", {"pedestrian_count": 2, "is_outside_crosswalk": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["pedestrian_facts_source"], "caller_supplied")
        if self.server_module.scene_perception.is_ready:
            # An empty grey frame: detection overrides the caller's claim of two pedestrians.
            status, body = self.call("/api/v1/pedestrian/detect",
                                     {"pedestrian_count": 2, "image_base64": png_b64(320, 240)})
            self.assertEqual(body["pedestrian_facts_source"], "detected")
            self.assertEqual(body["pedestrians_detected"], 0)
            self.assertEqual(body["alert_level"], "LOW")


if __name__ == "__main__":
    unittest.main()
