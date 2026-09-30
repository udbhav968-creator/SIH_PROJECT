"""
Unified road-scene perception: scene geometry, threshold handling, degraded
mode, and (when the trained weights are present) a real end-to-end pass.

The geometry and orchestration tests use stub detectors so they run in CI
without any model weights. The integration tests skip, visibly, when the
weights have not been trained or fetched on this machine.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import road_scene_perception as rsp
from models.frame_gate import frame_quality


def det(class_name, bbox, confidence=0.9, model="traffic"):
    label, group, colour = rsp.CLASS_STYLE[class_name]
    return {"class_name": class_name, "label": label, "group": group, "confidence": confidence,
            "bbox_pixels": list(bbox), "bbox_normalized": [0, 0, 0, 0], "colour_hex": colour,
            "model": model}


class StubDetector:
    """Stands in for ONNXObjectDetector: returns fixed raw detections."""

    def __init__(self, raw, ready=True):
        self._raw = raw
        self.is_ready = ready
        self.backend = "stub"
        self.input_size = 640
        self.calls = []

    def detect(self, image, conf_threshold=None, keep_classes=None):
        self.calls.append({"conf_threshold": conf_threshold, "keep_classes": keep_classes})
        return [d for d in self._raw
                if d["confidence"] >= conf_threshold and (not keep_classes or d["class_name"] in keep_classes)]


def raw(class_name, bbox, confidence):
    return {"class_name": class_name, "confidence": confidence, "bbox_pixels": list(bbox),
            "bbox_normalized": [0, 0, 0, 0]}


class GeometryTests(unittest.TestCase):
    def test_ground_contact_is_bottom_centre(self):
        self.assertEqual(rsp.ground_contact_point([10, 20, 40, 100]), (30.0, 120))

    def test_point_in_box_respects_margin(self):
        box = [100, 100, 100, 50]
        self.assertTrue(rsp.point_in_box((150, 125), box))
        self.assertFalse(rsp.point_in_box((205, 125), box))
        self.assertTrue(rsp.point_in_box((205, 125), box, margin=0.1))

    def test_overlap_fraction(self):
        self.assertAlmostEqual(rsp.overlap_fraction([0, 0, 10, 10], [5, 0, 10, 10]), 0.5)
        self.assertEqual(rsp.overlap_fraction([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)
        self.assertEqual(rsp.overlap_fraction([0, 0, 0, 10], [0, 0, 10, 10]), 0.0)

    def test_ego_path(self):
        # 1000x1000 frame: centre-bottom is in path, far left or above horizon is not.
        self.assertTrue(rsp.in_ego_path([450, 800, 100, 100], 1000, 1000))
        self.assertFalse(rsp.in_ego_path([0, 800, 100, 100], 1000, 1000))
        self.assertFalse(rsp.in_ego_path([450, 100, 100, 100], 1000, 1000))


class SceneFactTests(unittest.TestCase):
    W, H = 1280, 720

    def test_pedestrian_standing_on_crossing(self):
        crossing = det("crosswalk", [300, 450, 700, 150], model="markings")
        on = det("person", [500, 350, 40, 180])      # feet at y=530, inside the crossing
        off = det("person", [1150, 100, 40, 120])    # far away on the pavement
        facts = rsp.derive_scene_facts([crossing, on, off], self.W, self.H)
        self.assertTrue(facts["zebra_crossing_visible"])
        self.assertEqual(facts["pedestrians_on_crossing"], 1)
        self.assertEqual(facts["vulnerable_road_users"], 2)
        alerts = rsp.alerts_from_facts(facts)
        self.assertEqual(alerts[0]["code"], "PEDESTRIAN_ON_CROSSING")
        self.assertEqual(alerts[0]["level"], "CRITICAL")

    def test_no_crossing_means_no_crossing_alert(self):
        facts = rsp.derive_scene_facts([det("person", [500, 350, 40, 180])], self.W, self.H)
        self.assertFalse(facts["zebra_crossing_visible"])
        self.assertEqual(facts["pedestrians_on_crossing"], 0)
        self.assertEqual(rsp.alerts_from_facts(facts), [])

    def test_vehicle_blocking_crossing(self):
        crossing = det("crosswalk", [300, 450, 700, 150], model="markings")
        car = det("car", [400, 420, 200, 150])
        facts = rsp.derive_scene_facts([crossing, car], self.W, self.H)
        self.assertEqual(facts["vehicles_blocking_crossing"], 1)
        self.assertIn("CROSSING_BLOCKED", [a["code"] for a in rsp.alerts_from_facts(facts)])

    def test_pothole_in_path_outranks_maintenance_log(self):
        pothole = det("pothole", [600, 600, 80, 50], model="damage")
        crack = det("longitudinal_crack", [50, 500, 30, 200], model="damage")
        facts = rsp.derive_scene_facts([pothole, crack], self.W, self.H)
        self.assertEqual(facts["road_damage_count"], 2)
        self.assertEqual(facts["potholes_in_ego_path"], 1)
        codes = [a["code"] for a in rsp.alerts_from_facts(facts)]
        self.assertEqual(codes, ["POTHOLE_IN_PATH"])

    def test_damage_off_path_is_logged_not_alarmed(self):
        crack = det("alligator_crack", [50, 500, 100, 100], model="damage")
        facts = rsp.derive_scene_facts([crack], self.W, self.H)
        self.assertEqual([a["level"] for a in rsp.alerts_from_facts(facts)], ["INFO"])

    def test_coverage_is_capped_at_100(self):
        big = det("pothole", [0, 0, self.W, self.H], model="damage")
        facts = rsp.derive_scene_facts([big, big], self.W, self.H)
        self.assertEqual(facts["damage_bbox_coverage_pct"], 100.0)

    def test_every_fact_documents_its_rule(self):
        facts = rsp.derive_scene_facts([], self.W, self.H)
        for key in ("pedestrians_on_crossing", "vehicles_blocking_crossing", "potholes_in_ego_path"):
            self.assertIn(key, facts["rules"])


class PerceptionOrchestrationTests(unittest.TestCase):
    def make(self, traffic_raw=(), damage_raw=None, markings_raw=None, thresholds=None):
        perception = rsp.RoadScenePerception.__new__(rsp.RoadScenePerception)
        thresholds = thresholds or {}
        perception.heads = {
            "traffic": rsp._Head("traffic", StubDetector(list(traffic_raw)),
                                 {n: 0.35 for n in rsp.TRAFFIC_CLASSES}, None),
            "damage": rsp._Head("damage", StubDetector(damage_raw) if damage_raw is not None else None,
                                thresholds.get("damage", {}), None),
            "markings": rsp._Head("markings", StubDetector(markings_raw) if markings_raw is not None else None,
                                  thresholds.get("markings", {}), None),
        }
        return perception

    def test_per_class_thresholds_are_applied(self):
        perception = self.make(
            damage_raw=[raw("pothole", [10, 10, 20, 20], 0.30), raw("longitudinal_crack", [50, 50, 5, 40], 0.30)],
            thresholds={"damage": {"pothole": 0.25, "longitudinal_crack": 0.40,
                                   "transverse_crack": 0.4, "alligator_crack": 0.4}},
        )
        result = perception.analyze(np.zeros((100, 100, 3), dtype=np.uint8), gate=False)
        self.assertEqual([d["class_name"] for d in result["detections"]], ["pothole"])
        # The detector is queried at the lowest per-class threshold, then filtered per class.
        self.assertEqual(perception.heads["damage"].detector.calls[0]["conf_threshold"], 0.25)

    def test_irrelevant_coco_classes_are_dropped(self):
        perception = self.make(traffic_raw=[raw("car", [0, 0, 50, 50], 0.9), raw("laptop", [0, 0, 5, 5], 0.99)])
        result = perception.analyze(np.zeros((100, 100, 3), dtype=np.uint8), gate=False)
        self.assertEqual(result["counts"], {"car": 1})

    def test_missing_models_are_reported_not_invented(self):
        perception = self.make(traffic_raw=[raw("person", [0, 0, 10, 30], 0.8)])
        result = perception.analyze(np.zeros((64, 64, 3), dtype=np.uint8), gate=False)
        self.assertEqual(sorted(result["unavailable_models"]), ["damage", "markings"])
        self.assertFalse(result["models"]["damage"])
        self.assertTrue(all(d["model"] == "traffic" for d in result["detections"]))

    def test_groups_restrict_which_heads_run(self):
        perception = self.make(traffic_raw=[raw("car", [0, 0, 50, 50], 0.9)],
                               damage_raw=[raw("pothole", [10, 10, 20, 20], 0.9)],
                               thresholds={"damage": {"pothole": 0.3}})
        result = perception.analyze(np.zeros((100, 100, 3), dtype=np.uint8), groups={"damage"}, gate=False)
        self.assertEqual(result["counts"], {"pothole": 1})
        self.assertEqual(perception.heads["traffic"].detector.calls, [])

    def test_rejects_non_rgb_input(self):
        perception = self.make()
        with self.assertRaises(ValueError):
            perception.analyze(np.zeros((10, 10), dtype=np.uint8), gate=False)

    def test_detections_sorted_by_confidence(self):
        perception = self.make(traffic_raw=[raw("car", [0, 0, 50, 50], 0.5), raw("bus", [0, 0, 90, 90], 0.95)])
        result = perception.analyze(np.zeros((100, 100, 3), dtype=np.uint8), gate=False)
        self.assertEqual([d["class_name"] for d in result["detections"]], ["bus", "car"])

    def test_annotate_returns_image_of_same_size(self):
        perception = self.make(traffic_raw=[raw("person", [5, 5, 20, 40], 0.9)])
        frame = np.full((80, 120, 3), 128, dtype=np.uint8)
        image = rsp.annotate(frame, perception.analyze(frame, gate=False))
        self.assertEqual(image.size, (120, 80))
        self.assertFalse(np.array_equal(np.asarray(image), frame))  # something was drawn


class FrameGateTests(unittest.TestCase):
    def road_like(self):
        rng = np.random.default_rng(0)
        frame = np.full((360, 640, 3), 110, dtype=np.uint8)
        frame[200:] = rng.integers(60, 160, (160, 640, 3), dtype=np.uint8)  # asphalt texture
        return frame

    def test_textured_road_is_analysable(self):
        self.assertTrue(frame_quality(self.road_like())["analysable"])

    def test_degenerate_frames_are_rejected_with_a_reason(self):
        cases = {
            "dark": np.full((360, 640, 3), 5, dtype=np.uint8),
            "glare": np.full((360, 640, 3), 255, dtype=np.uint8),
            "texture": np.full((360, 640, 3), 120, dtype=np.uint8),
        }
        for word, frame in cases.items():
            quality = frame_quality(frame)
            self.assertFalse(quality["analysable"], word)
            self.assertIn(word, " ".join(quality["reasons"]))

    def test_rejected_frames_skip_the_detectors(self):
        stub = StubDetector([raw("car", [0, 0, 50, 50], 0.9)])
        perception = rsp.RoadScenePerception.__new__(rsp.RoadScenePerception)
        perception.heads = {"traffic": rsp._Head("traffic", stub, {"car": 0.3}, None)}
        result = perception.analyze(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertTrue(result["skipped_by_quality_gate"])
        self.assertEqual((stub.calls, result["detections"]), ([], []))
        self.assertIn("too dark", result["frame_quality"]["reasons"][0])
        self.assertEqual(perception.analyze(self.road_like())["counts"], {"car": 1})


class PrivacyRedactionTests(unittest.TestCase):
    def test_people_are_blurred_and_nothing_else_changes(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)  # high-frequency texture
        result = {"detections": [det("person", [50, 40, 40, 100]), det("car", [200, 100, 60, 40])]}
        redacted = rsp.redact(frame, result)

        person = (slice(40, 140), slice(50, 90))
        car = (slice(100, 140), slice(200, 260))
        # Blurring removes pixel-to-pixel variation inside the person box.
        self.assertLess(np.abs(np.diff(redacted[person].astype(int), axis=1)).mean(),
                        np.abs(np.diff(frame[person].astype(int), axis=1)).mean() / 3)
        np.testing.assert_array_equal(redacted[car], frame[car])  # vehicles untouched
        np.testing.assert_array_equal(redacted[:, 150:], frame[:, 150:])
        self.assertFalse(np.shares_memory(redacted, frame))

    def test_plate_boxes_are_blurred_too(self):
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
        redacted = rsp.redact(frame, {"detections": []}, plate_boxes=[[200, 150, 60, 20]])
        plate = (slice(150, 170), slice(200, 260))
        self.assertLess(np.abs(np.diff(redacted[plate].astype(int), axis=1)).mean(),
                        np.abs(np.diff(frame[plate].astype(int), axis=1)).mean() / 3)
        np.testing.assert_array_equal(redacted[:100, :150], frame[:100, :150])

    def test_missing_plate_model_is_stated_not_implied(self):
        perception = rsp.RoadScenePerception.__new__(rsp.RoadScenePerception)
        perception.heads = {}
        perception.privacy = rsp._Head("license_plate", None, {}, None)
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        _, info = perception.redact(frame, {"detections": [det("person", [1, 1, 5, 10])]})
        self.assertEqual((info["people"], info["plates"]), (1, None))
        self.assertIn("NOT redacted", info["plates_note"])

    def test_boxes_at_the_border_are_clipped(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = {"detections": [det("person", [-10, -10, 30, 80])]}
        self.assertEqual(rsp.redact(frame, result).shape, frame.shape)


class TrafficCountTests(unittest.TestCase):
    def test_detections_map_to_irc_categories(self):
        from models.urban_traffic_net import UrbanTrafficNet, counts_from_detections
        dets = [det("car", [0, 0, 1, 1]), det("car", [0, 0, 1, 1]), det("bus", [0, 0, 1, 1]),
                det("motorcycle", [0, 0, 1, 1]), det("bicycle", [0, 0, 1, 1]), det("person", [0, 0, 1, 1])]
        counts = counts_from_detections(dets)
        self.assertEqual(counts, {"Car": 2, "City Bus": 1, "Two-Wheeler": 2})
        # 2*1.0 + 1*2.0 + 2*0.5 = 5 PCU; people do not occupy carriageway PCU
        self.assertEqual(UrbanTrafficNet().calculate_congestion_index(counts)["pcu_equivalent"], 5.0)


class ModelCardLoadingTests(unittest.TestCase):
    def test_untrained_detector_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(rsp.load_model_card("road_damage", tmp))
            head = rsp.RoadScenePerception._load_trained("road_damage", tmp)
            self.assertFalse(head.is_ready)

    def test_card_without_onnx_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "crosswalk.json"), "w", encoding="utf-8") as handle:
                json.dump({"classes": ["crosswalk"], "serving_thresholds": {"crosswalk": 0.4}}, handle)
            self.assertFalse(rsp.RoadScenePerception._load_trained("crosswalk", tmp).is_ready)


def _trained(name):
    return all(os.path.exists(os.path.join(rsp.DETECTOR_DIR, f"{name}.{ext}")) for ext in ("onnx", "json"))


class TrainedDetectorIntegrationTests(unittest.TestCase):
    """Run against whichever trained detectors are on disk; each skips on its own."""

    @classmethod
    def setUpClass(cls):
        cls.perception = rsp.RoadScenePerception()

    def _require(self, name):
        if not _trained(name):
            self.skipTest(f"{name} not trained; run training.train_detector configs/detectors/{name}.yaml")

    def _check_artifact(self, name, head_key):
        import hashlib
        import json
        card = json.loads(Path(rsp.DETECTOR_DIR, f"{name}.json").read_text(encoding="utf-8"))
        onnx = Path(rsp.DETECTOR_DIR, card["artifacts"]["onnx"]["file"])
        self.assertEqual(hashlib.sha256(onnx.read_bytes()).hexdigest(), card["artifacts"]["onnx"]["sha256"],
                         "the served ONNX file is not the one the model card describes")
        described = self.perception.describe()[head_key]
        self.assertTrue(described["ready"])
        for value in described["serving_thresholds"].values():
            self.assertTrue(0.0 < value < 1.0)
        self.assertIn("test", card["metrics"])

    def test_road_damage_artifact(self):
        self._require("road_damage")
        self._check_artifact("road_damage", "damage")

    def test_crosswalk_artifact(self):
        self._require("crosswalk")
        self._check_artifact("crosswalk", "markings")

    def test_crosswalk_found_in_a_held_out_frame(self):
        self._require("crosswalk")
        test_labels = Path(rsp.CKPT_DIR).parent / "datasets" / "crosswalk" / "labels" / "test"
        if not test_labels.is_dir():
            self.skipTest("crosswalk test split not prepared")
        from PIL import Image
        hits = 0
        frames = [lab for lab in sorted(test_labels.glob("*.txt")) if lab.read_text().startswith("0 ")][:10]
        for label in frames:
            image = Path(str(label).replace("labels", "images")).with_suffix(".jpg")
            result = self.perception.analyze(np.asarray(Image.open(image).convert("RGB")), groups={"markings"})
            hits += result["scene"]["zebra_crossing_visible"]
        self.assertGreaterEqual(hits, 6, f"found crossings in {hits}/10 labelled held-out frames")

    def test_featureless_frame_yields_no_road_findings(self):
        if not (_trained("road_damage") or _trained("crosswalk")):
            self.skipTest("no trained road detector")
        grey = np.full((720, 1280, 3), 110, dtype=np.uint8)
        result = self.perception.analyze(grey, groups={"damage", "markings"}, gate=False)
        self.assertEqual(result["detections"], [], "the model itself, not only the gate, must stay quiet")


if __name__ == "__main__":
    unittest.main()
