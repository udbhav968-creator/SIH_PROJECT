"""Gateway features, the model registry and the API contract."""

import json
import os
import shutil
import tempfile
import unittest


class ApiKeyOnWriteEndpoints(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("ROAD_SHIELD_API_KEY", None)

    def test_open_when_no_key_is_configured(self):
        from api.server import _api_key_ok
        os.environ.pop("ROAD_SHIELD_API_KEY", None)
        self.assertTrue(_api_key_ok({}))

    def test_key_required_and_compared_exactly(self):
        from api.server import _api_key_ok
        os.environ["ROAD_SHIELD_API_KEY"] = "s3cret"
        self.assertFalse(_api_key_ok({}))
        self.assertFalse(_api_key_ok({"X-API-Key": "wrong"}))
        self.assertTrue(_api_key_ok({"X-API-Key": "s3cret"}))
        self.assertTrue(_api_key_ok({"Authorization": "Bearer s3cret"}))

    def test_protected_set_covers_state_changing_endpoints_only(self):
        from api.server import PROTECTED_POST
        self.assertIn("/api/v1/fleet/report-defect", PROTECTED_POST)
        self.assertIn("/api/v1/dispatch/work-order", PROTECTED_POST)
        self.assertNotIn("/api/v1/pipeline/deep-audit", PROTECTED_POST)
        self.assertNotIn("/api/v1/dispatch/verify-seal", PROTECTED_POST)


class ModelRegistry(unittest.TestCase):
    def test_registry_lists_served_models_with_checksums(self):
        from models.served_report import model_registry
        ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
        r = model_registry(ckpt)
        self.assertTrue(r["serving"])
        for m in r["models"]:
            for a in m["artefacts"]:
                if a.get("present"):
                    self.assertEqual(len(a["sha256"]), 64)

    def test_reports_only_mode_needs_no_files(self):
        from models.served_report import model_registry
        d = tempfile.mkdtemp()
        try:
            with open(os.path.join(d, "imu_shock_report.json"), "w") as fh:
                json.dump({"held_out_validation_accuracy": 0.87}, fh)
            r = model_registry(d, hash_files=False)
            names = [m["name"] for m in r["models"]]
            self.assertIn("IMU RandomForest", names)
            self.assertTrue(all("sha256" not in a for m in r["models"] for a in m["artefacts"]))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class OpenApiContract(unittest.TestCase):
    def test_spec_is_valid_json_and_lists_documented_paths(self):
        from api.openapi import spec
        s = spec()
        json.dumps(s)
        self.assertTrue(s["openapi"].startswith("3."))
        for p in ("/api/v1/pipeline/deep-audit", "/api/v1/dispatch/work-order", "/api/v1/models/served"):
            self.assertIn(p, s["paths"])

    def test_every_documented_path_is_routed_by_the_engine(self):
        from api.openapi import spec
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "server.py"),
                   encoding="utf-8").read()
        for p in spec()["paths"]:
            self.assertIn(f'"{p}"', src, p)


if __name__ == "__main__":
    unittest.main()
