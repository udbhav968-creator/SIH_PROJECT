"""
Regression suite for the real system. Standard library only:

    python -m unittest tests.test_road_shield -v

Each test checks something that would actually break the product, and every
expected value is derived rather than hard-coded to whatever the code
currently returns. Tests that need a trained model skip cleanly when one
isn't on disk, so a fresh clone reports honestly instead of failing.

The earlier suite in tests/_legacy/ exercised behaviour that has been removed
on purpose - fabricated datasets, a classifier that could be told what to
answer, random licence plates. Those tests failing was the point.
"""

import os
import sys
import unittest

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import numpy as np

CKPT = os.path.join(ENGINE_ROOT, "checkpoints")


def sample_photo():
    from data.image_dataset import sample_photo_path
    try:
        path, _cls, _name = sample_photo_path(class_id=2)
        return path
    except Exception:
        return None


class DatasetIntegrity(unittest.TestCase):
    def test_no_photo_is_filed_under_two_classes(self):
        """The bug that held accuracy at 36.6%. It must not come back."""
        from scripts.fix_label_conflicts import index_dataset
        conflicts = {s: c for s, c in index_dataset().items() if len(c) > 1}
        self.assertEqual(conflicts, {}, f"photographs filed under several classes: {list(conflicts)[:5]}")

    def test_every_class_has_training_images(self):
        from data.image_dataset import dataset_inventory
        inv = dataset_inventory()["labeled_classes"]
        empty = [name for name, d in inv.items() if d["usable_photos"] == 0]
        self.assertEqual(empty, [], f"classes with no images: {empty}")


class VisionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.deep_vision_net import load_best_vision_model
        cls.model, cls.backend = load_best_vision_model(CKPT, verbose=False)
        cls.photo = sample_photo()

    def setUp(self):
        if self.backend == "none":
            self.skipTest("no trained vision model on disk")
        if not self.photo:
            self.skipTest("no dataset photo available")

    def test_prediction_shape_and_probabilities(self):
        from data.image_dataset import load_image
        from models.vision_distress_net import VisionDistressNet
        result = self.model.predict_image(load_image(self.photo))
        self.assertIn("class_name", result)
        self.assertIn(result["class_name"], VisionDistressNet.CLASS_NAMES)
        probs = result["all_class_probabilities"]
        self.assertEqual(len(probs), len(VisionDistressNet.CLASS_NAMES))
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)
        self.assertGreaterEqual(result["confidence"], max(probs.values()) - 1e-6)

    def test_prediction_is_deterministic(self):
        from data.image_dataset import load_image
        img = load_image(self.photo)
        a = self.model.predict_image(img)
        b = self.model.predict_image(img)
        self.assertEqual(a["class_name"], b["class_name"])
        self.assertAlmostEqual(a["confidence"], b["confidence"], places=6)

    def test_reported_accuracy_beats_random(self):
        import json
        path = os.path.join(CKPT, "vision_distress_report.json")
        if not os.path.exists(path):
            self.skipTest("no training report")
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        self.assertGreater(r["held_out_validation_accuracy"], r["random_guess_baseline"] * 2,
                           "held-out accuracy is not meaningfully above chance")


class CNNEmbeddingHead(unittest.TestCase):
    """
    A head trained on ResNet-50 embeddings applied to MobileNetV2 vectors would
    not raise - it would return confident nonsense. These tests pin the pairing.
    """

    def test_head_never_runs_on_a_foreign_backbone(self):
        import glob
        import joblib
        from models.cnn_embedder import BACKBONES
        from models.deep_vision_net import CNNHeadClassifier
        heads = glob.glob(os.path.join(CKPT, "cnn_head_*.joblib"))
        if not heads:
            self.skipTest("no CNN head trained")
        clf = CNNHeadClassifier(checkpoints_dir=CKPT)
        if not clf.is_ready:
            self.skipTest("no backbone on disk")
        self.assertEqual(clf.embedder.name, clf.report["backbone"],
                         "loaded a head against a backbone it was not trained on")
        # and the head it chose must be one that exists on disk
        trained_on = {joblib.load(p).get("backbone") for p in heads}
        self.assertIn(clf.embedder.name, trained_on)
        self.assertIn(clf.embedder.name, BACKBONES)

    def test_embedding_dimension_matches_what_the_head_expects(self):
        from models.deep_vision_net import CNNHeadClassifier
        clf = CNNHeadClassifier(checkpoints_dir=CKPT)
        if not clf.is_ready:
            self.skipTest("no CNN head available")
        photo = sample_photo()
        if not photo:
            self.skipTest("no dataset photo available")
        from data.image_dataset import load_image
        vec = clf.embedder.embed(load_image(photo))
        self.assertEqual(vec.ndim, 1)
        probs = clf.predict_probabilities(load_image(photo))
        self.assertEqual(len(probs), len(clf.CLASS_NAMES))
        self.assertAlmostEqual(float(sum(probs)), 1.0, places=4)

    def test_reported_accuracy_is_on_a_grouped_split(self):
        import glob
        import json
        reports = glob.glob(os.path.join(CKPT, "cnn_head_*_report.json"))
        if not reports:
            self.skipTest("no CNN head report")
        for path in reports:
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
            self.assertIn("grouped by source photograph", r["split_strategy"])
            self.assertGreater(r["held_out_test_accuracy"], r["random_guess_baseline"] * 2)
            self.assertLessEqual(r["held_out_test_photographs"], r["held_out_test_images"])


class KaggleIngest(unittest.TestCase):
    """
    The mapping from a Kaggle folder name to one of our seven classes decides
    what every ingested image is labelled. A wrong rule here silently poisons
    the training set, so the rules are pinned.
    """

    def test_folder_names_map_to_the_intended_classes(self):
        from scripts.fetch_kaggle_datasets import CRACK, NORMAL, POTHOLE, SIGN, classify_path
        cases = {
            "train/Positive/00042.jpg": CRACK,
            "train/Negative/00042.jpg": NORMAL,
            "dataset/potholes/img_7.png": POTHOLE,
            "dataset/plain road/img_7.png": NORMAL,
            "Cracks/alligator/x.jpg": CRACK,
            "road_sign/stop_12.jpg": SIGN,
        }
        for rel, expected in cases.items():
            self.assertEqual(classify_path(rel, {}), expected, f"{rel} mapped wrongly")

    def test_unrecognised_folders_are_skipped_not_guessed(self):
        from scripts.fetch_kaggle_datasets import classify_path
        for rel in ("annotations/x.xml.jpg", "misc/readme_images/logo.png", "x.jpg"):
            self.assertIsNone(classify_path(rel, {}),
                              f"{rel} should be skipped, not assigned a class")

    def test_overrides_beat_the_generic_rules(self):
        from scripts.fetch_kaggle_datasets import POTHOLE, classify_path
        # "images" means nothing generically, but a dataset can declare it
        self.assertIsNone(classify_path("images/a.jpg", {}))
        self.assertEqual(classify_path("images/a.jpg", {"images": POTHOLE}), POTHOLE)

    def _reupload_caught(self, index, path, scale, quality):
        import io
        from PIL import Image
        with Image.open(path) as im:
            original = im.convert("RGB")
            index.add(index.hash_image(original))
            buf = io.BytesIO()
            size = (max(16, int(original.width * scale)), max(16, int(original.height * scale)))
            original.resize(size).save(buf, "JPEG", quality=quality)
            buf.seek(0)
            with Image.open(buf) as reupload:
                return index.contains(index.hash_image(reupload.convert("RGB")))

    def test_recompressed_copies_are_always_caught(self):
        """A re-encoded copy is the common case and must never slip through."""
        from models.forensic_audit_engine import ForensicDuplicateHasher
        from scripts.fetch_kaggle_datasets import NearDuplicateIndex
        from data.image_dataset import sample_photo_path
        rng = np.random.default_rng(11)
        caught = 0
        for _ in range(8):
            try:
                path, _c, _n = sample_photo_path(class_id=int(rng.integers(0, 3)))
            except Exception:
                self.skipTest("no dataset photos available")
            index = NearDuplicateIndex(ForensicDuplicateHasher(hash_size=8))
            caught += self._reupload_caught(index, path, 1.0, 80)
        self.assertEqual(caught, 8, "a JPEG re-encode was not recognised as a duplicate")

    def test_resized_reuploads_are_caught_at_the_documented_rate(self):
        """Half-size re-uploads: measured at 100% for the chosen threshold."""
        from models.forensic_audit_engine import ForensicDuplicateHasher
        from scripts.fetch_kaggle_datasets import NearDuplicateIndex
        from data.image_dataset import sample_photo_path
        rng = np.random.default_rng(23)
        caught = 0
        for _ in range(10):
            try:
                path, _c, _n = sample_photo_path(class_id=int(rng.integers(0, 3)))
            except Exception:
                self.skipTest("no dataset photos available")
            index = NearDuplicateIndex(ForensicDuplicateHasher(hash_size=8))
            caught += self._reupload_caught(index, path, 0.5, 55)
        self.assertGreaterEqual(caught, 9, f"only {caught}/10 half-size re-uploads caught")

    def test_distinct_photographs_are_not_merged(self):
        """The other half of the trade-off: the threshold must not over-merge."""
        from models.forensic_audit_engine import ForensicDuplicateHasher
        from scripts.fetch_kaggle_datasets import NearDuplicateIndex
        from data.image_dataset import load_image, sample_photo_path
        from PIL import Image
        index = NearDuplicateIndex(ForensicDuplicateHasher(hash_size=8))
        collisions = 0
        for cls in (0, 1, 2):
            for _ in range(4):
                try:
                    path, _c, _n = sample_photo_path(class_id=cls)
                except Exception:
                    self.skipTest("no dataset photos available")
                with Image.open(path) as im:
                    digest = index.hash_image(im.convert("RGB"))
                if index.contains(digest):
                    collisions += 1
                index.add(digest)
        # augmented copies of one source photograph legitimately collide, so a
        # few are expected; most of 12 draws must still be distinct.
        self.assertLess(collisions, 6, f"{collisions}/12 distinct photographs collided")


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pipeline.deep_inference_pipeline import DeepInferencePipeline
        # Share the suite's single engine; see tests/test_geometry_and_storage.py
        # for why three of them exhausted a machine with 4.7 GB free.
        # unittest discover imports this as `test_road_shield`, not
        # `tests.test_road_shield`, so the sibling import has to be tried both
        # ways or the fallback silently builds a second engine - which is the
        # thing being avoided.
        shared = None
        for mod in ("tests.test_geometry_and_storage", "test_geometry_and_storage"):
            try:
                shared = __import__(mod, fromlist=["shared_pipeline"]).shared_pipeline
                break
            except Exception:
                continue
        cls.pipe = shared() if shared else DeepInferencePipeline(CKPT)
        cls.photo = sample_photo()

    def test_audit_returns_the_documented_shape(self):
        if not self.photo:
            self.skipTest("no dataset photo")
        r = self.pipe.audit_image(image_input=self.photo)
        for key in ("status", "all_detections", "imu_shock_telemetry",
                    "bayesian_sensor_fusion", "astm_d6433_pci", "latency_ms", "scene_summary"):
            self.assertIn(key, r)
        self.assertGreater(r["latency_ms"], 0)

    def test_imu_absence_is_reported_not_invented(self):
        """The central honesty fix: no IMU window means no IMU claim."""
        if not self.photo:
            self.skipTest("no dataset photo")
        r = self.pipe.audit_image(image_input=self.photo)
        imu = r["imu_shock_telemetry"]
        self.assertFalse(imu["available"])
        self.assertIsNone(imu["shock_classification"])
        self.assertIn("reason", imu)

    def test_imu_window_is_actually_used_when_supplied(self):
        if not self.photo:
            self.skipTest("no dataset photo")
        try:
            from data.dataset_generator import sample_real_imu_window
            window = sample_real_imu_window(split="val", pothole_only=True, seed=1)
        except Exception:
            self.skipTest("no IMU dataset on disk")
        r = self.pipe.audit_image(image_input=self.photo, imu_series=window)
        self.assertTrue(r["imu_shock_telemetry"]["available"])
        self.assertIsNotNone(r["imu_shock_telemetry"]["shock_classification"])

    def test_flat_grey_image_is_rejected_as_non_pavement(self):
        grey = np.full((480, 640, 3), 128, dtype=np.uint8)
        r = self.pipe.audit_image(image_input=grey)
        self.assertNotEqual(r.get("status"), "ANALYSIS_COMPLETE")


class WorkOrderSealing(unittest.TestCase):
    def setUp(self):
        from models.morth_dispatch_agent import MoRTHDispatchAgent
        self.agent = MoRTHDispatchAgent()
        self.order = self.agent.generate_work_order(
            corridor_id="NH-44", latitude=12.97, longitude=77.59,
            distress_class="Pothole Cavity", area_sqm=1.8, depth_cm=6.5, pci_score=42)

    def test_seal_verifies(self):
        self.assertTrue(self.agent.verify_work_order_seal(dict(self.order)))

    def test_tampering_breaks_the_seal(self):
        for field, value in (("allocated_budget_inr", 999999.0),
                             ("surface_area_sqm", 99.0),
                             ("corridor", "NH-99")):
            tampered = dict(self.order)
            tampered[field] = value
            self.assertFalse(self.agent.verify_work_order_seal(tampered),
                             f"editing {field} did not break the seal")

    def test_seal_is_a_sha256_digest(self):
        seal = self.order["sha256_cryptographic_seal"]
        self.assertEqual(len(seal), 64)
        int(seal, 16)  # raises if not hexadecimal


class FleetDeduplication(unittest.TestCase):
    def setUp(self):
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        self.engine = FleetDeduplicationEngine(proximity_threshold_meters=10.0)

    def test_nearby_reports_merge_and_distant_ones_do_not(self):
        a = self.engine.ingest_fleet_detection("BUS-1", 12.9716, 77.5946, "Pothole Cavity", 40, 1.5,
                                               enrich_location=False)
        b = self.engine.ingest_fleet_detection("BUS-2", 12.97163, 77.59457, "Pothole Cavity", 42, 1.6,
                                               enrich_location=False)
        c = self.engine.ingest_fleet_detection("BUS-3", 12.9800, 77.6050, "Pothole Cavity", 38, 2.0,
                                               enrich_location=False)
        self.assertEqual(a["action"], "REGISTERED_NEW_DEFECT")
        self.assertEqual(b["action"], "DEDUPLICATED_AND_UPDATED")
        self.assertEqual(b["defect_id"], a["defect_id"])
        self.assertEqual(c["action"], "REGISTERED_NEW_DEFECT")
        stats = self.engine.get_deduplication_stats()
        self.assertEqual(stats["total_reports_ingested"], 3)
        self.assertEqual(stats["unique_defects_registered"], 2)

    def test_haversine_matches_a_known_distance(self):
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine as F
        # one degree of latitude is about 111.2 km anywhere on Earth
        d = F.haversine_distance(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(d / 1000.0, 111.2, delta=0.5)


class ForensicAudit(unittest.TestCase):
    def setUp(self):
        from models.forensic_audit_engine import ForensicDuplicateHasher
        self.hasher = ForensicDuplicateHasher(hash_size=8)
        rng = np.random.default_rng(3)
        self.image = (rng.normal(120, 40, (256, 256, 3))).clip(0, 255).astype(np.uint8)

    def test_identical_images_hash_identically(self):
        h1 = self.hasher.compute_hash(self.image)
        h2 = self.hasher.compute_hash(self.image.copy())
        self.assertEqual(self.hasher.hamming_distance(h1, h2), 0)
        self.assertTrue(self.hasher.is_duplicate(h1, h2))

    def test_different_images_do_not_collide(self):
        rng = np.random.default_rng(99)
        other = (rng.normal(60, 30, (256, 256, 3))).clip(0, 255).astype(np.uint8)
        h1 = self.hasher.compute_hash(self.image)
        h2 = self.hasher.compute_hash(other)
        self.assertGreater(self.hasher.hamming_distance(h1, h2), 6)

    def test_resaving_an_image_still_matches(self):
        """A re-encoded copy must still be caught as the same photograph."""
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(self.image).save(buf, format="JPEG", quality=70)
        buf.seek(0)
        reencoded = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
        h1 = self.hasher.compute_hash(self.image)
        h2 = self.hasher.compute_hash(reencoded)
        self.assertLessEqual(self.hasher.hamming_distance(h1, h2), 8)


class PavementCondition(unittest.TestCase):
    def setUp(self):
        from models.pci_regressor_net import PavementConditionIndexEngine
        self.pci = PavementConditionIndexEngine()

    def test_perfect_road_scores_100(self):
        self.assertEqual(self.pci.compute()["pci_score"], 100.0)

    def test_more_damage_never_raises_the_score(self):
        previous = 101.0
        for density in (5, 15, 30, 50):
            score = self.pci.compute(crack_density_pct=density, crack_severity="HIGH")["pci_score"]
            self.assertLessEqual(score, previous)
            previous = score

    def test_score_stays_in_range(self):
        worst = self.pci.compute(crack_density_pct=100, crack_severity="HIGH",
                                 pothole_count=50, pothole_density_pct=100,
                                 pothole_severity="HIGH", rutting_mm=100, iri_roughness=20)
        self.assertGreaterEqual(worst["pci_score"], 0.0)
        self.assertLessEqual(worst["pci_score"], 100.0)


class ObjectDetector(unittest.TestCase):
    def test_detector_state_is_honest(self):
        from models.onnx_object_detector import ONNXObjectDetector
        det = ONNXObjectDetector(checkpoints_dir=CKPT)
        info = det.describe()
        if det.is_ready:
            self.assertIsNotNone(info["backend"])
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            self.assertEqual(det.detect(blank), [], "a blank frame must yield no detections")
        else:
            self.assertIsNone(info["backend"])
            with self.assertRaises(RuntimeError):
                det.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_non_maximum_suppression(self):
        from models.onnx_object_detector import nms
        boxes = np.array([[0, 0, 100, 100], [4, 4, 104, 104], [500, 500, 560, 560]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        self.assertEqual(nms(boxes, scores, 0.45), [0, 2])


class MapsService(unittest.TestCase):
    def test_unavailable_is_reported_not_invented(self):
        """With no network and no key, the service must say so, not guess."""
        from services.google_maps_service import GoogleMapsService
        svc = GoogleMapsService(api_key="")

        def fail(*_a, **_k):
            raise OSError("network disabled for this test")

        svc._get_json = fail
        far_from_india = svc.reverse_geocode(64.5, -21.9)  # Reykjavik
        self.assertIn(far_from_india["status"], ("UNAVAILABLE", "APPROXIMATE"))
        if far_from_india["status"] == "UNAVAILABLE":
            self.assertIsNone(far_from_india["formatted_address"])
        elevation = svc.get_elevation(64.5, -21.9)
        self.assertEqual(elevation["status"], "UNAVAILABLE")
        self.assertIsNone(elevation["elevation_meters"])

    def test_straight_line_route_is_labelled_as_such(self):
        from services.google_maps_service import GoogleMapsService
        svc = GoogleMapsService(api_key="")

        def fail(*_a, **_k):
            raise OSError("network disabled for this test")

        svc._get_json = fail
        route = svc.get_directions(12.97, 77.59, 12.93, 77.62)
        self.assertEqual(route["route_type"], "straight_line_estimate")
        self.assertEqual(len(route["polyline_coords"]), 2)
        self.assertIn("straight", route["provider"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
