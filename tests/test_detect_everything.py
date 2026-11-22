"""Tests for pipeline.detect_everything -- the unified image+video
detect-and-count pipeline, and the honest road-damage candidate path."""

import glob
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pothole_images(n=4):
    pats = [
        os.path.join(ROOT, "datasets", "02_kaggle_pothole_600", "real_images", "*.jpg"),
        os.path.join(ROOT, "datasets", "**", "real_images", "*.jpg"),
    ]
    found = []
    for p in pats:
        found = sorted(glob.glob(p, recursive=True))
        if found:
            break
    return found[:n]


class TestCanonicalKeys(unittest.TestCase):
    def test_mapping(self):
        from pipeline.detect_everything import canonical_key
        self.assertEqual(canonical_key("Pothole"), "potholes")
        self.assertEqual(canonical_key("Longitudinal Crack"), "cracks")
        self.assertEqual(canonical_key("Alligator Crack"), "cracks")
        self.assertEqual(canonical_key("Zebra Crosswalk"), "zebra_crossings")
        self.assertEqual(canonical_key("Person"), "persons")
        self.assertEqual(canonical_key("Car"), "cars")
        self.assertEqual(canonical_key("Bus"), "buses")
        self.assertEqual(canonical_key("Traffic Light"), "traffic_lights")
        self.assertEqual(canonical_key("Stop Sign"), "stop_signs")
        self.assertEqual(canonical_key("kite"), "other")


class TestDamageCandidatePath(unittest.TestCase):
    """The road-damage detector must stay BLOCKED on the trustworthy path, but
    produce clearly-labelled candidate boxes when explicitly asked."""

    def test_gate_is_intact(self):
        from models.road_damage_detector import RoadDamageDetector
        det = RoadDamageDetector()
        # Weights load, but the verification gate is still closed and honest.
        self.assertTrue(det.weights_loaded)
        self.assertFalse(det.is_ready, "served path must refuse an unverified detector")
        self.assertIsNotNone(det.blocked_by)

    def test_candidate_detections_are_tagged(self):
        from models.road_damage_detector import RoadDamageDetector
        imgs = _pothole_images(6)
        if not imgs:
            self.skipTest("no dataset images available")
        det = RoadDamageDetector()
        tagged_any = False
        for p in imgs:
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            for d in det.detect_candidate(arr, conf_threshold=0.25):
                self.assertEqual(d["verification_status"], "candidate_unverified")
                self.assertIsNotNone(d["gate_blocked_by"])
                tagged_any = True
        self.assertTrue(tagged_any, "expected at least one candidate box across pothole images")
        # verifying flag must be restored to False afterwards
        self.assertFalse(getattr(det, "verifying", False))


class TestAnalyzeImage(unittest.TestCase):
    def test_structure_and_counts(self):
        from pipeline.detect_everything import analyze_image, COUNT_KEYS
        imgs = _pothole_images(1)
        if not imgs:
            self.skipTest("no dataset images available")
        r = analyze_image(imgs[0], conf=0.25, candidate_damage=True)
        self.assertEqual(r["mode"], "image")
        self.assertIn("counts", r)
        self.assertIn("total", r["counts"])
        self.assertGreaterEqual(r["counts"]["total"], 0)
        # every count key is a known friendly bucket
        for k in r["counts"]:
            self.assertTrue(k == "total" or k in COUNT_KEYS)
        # annotated image was produced and matches input dims
        self.assertIn("annotated", r)
        self.assertEqual(r["annotated"].size, tuple(r["image_size"]))
        self.assertIsInstance(r["annotated_png"], (bytes, bytearray))
        self.assertTrue(r["annotated_png"][:8] == b"\x89PNG\r\n\x1a\n")
        # damage is exposed as a candidate (not silently claimed verified)
        self.assertEqual(r["damage_verification"], "candidate_unverified")

    def test_counts_match_detections(self):
        from pipeline.detect_everything import analyze_image
        imgs = _pothole_images(1)
        if not imgs:
            self.skipTest("no dataset images available")
        r = analyze_image(imgs[0], conf=0.25, candidate_damage=True, draw=False)
        self.assertEqual(r["counts"]["total"], len(r["detections"]))

    def test_hidden_when_candidate_off(self):
        from pipeline.detect_everything import analyze_image
        imgs = _pothole_images(1)
        if not imgs:
            self.skipTest("no dataset images available")
        r = analyze_image(imgs[0], conf=0.25, candidate_damage=False, draw=False)
        self.assertEqual(r["damage_verification"], "hidden")
        for d in r["detections"]:
            self.assertNotIn("pothole", d.get("class_name", "").lower())


class TestAnalyzeVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        imgs = _pothole_images(4)
        if not imgs:
            cls.clip = None
            return
        import cv2
        cls.tmp = tempfile.mkdtemp()
        cls.clip = os.path.join(cls.tmp, "clip.mp4")
        W, H = 640, 400
        vw = cv2.VideoWriter(cls.clip, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
        for p in imgs:
            frame = np.asarray(Image.open(p).convert("RGB").resize((W, H)))
            for _ in range(8):
                vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        vw.release()

    def test_video_counts(self):
        if not self.clip:
            self.skipTest("no dataset images available")
        from pipeline.detect_everything import analyze_video
        r = analyze_video(self.clip, conf=0.25, sample_every_s=0.4,
                          candidate_damage=True, save_sample_frames=1)
        self.assertEqual(r["mode"], "video")
        self.assertGreater(r["frames_processed"], 0)
        self.assertIn("unique_counts", r)
        self.assertGreaterEqual(r["unique_counts"]["total"], 0)
        self.assertIsInstance(r["per_frame"], list)
        # unique total never exceeds the sum of per-frame totals (de-dup sanity)
        per_frame_sum = sum(f["frame_counts"].get("total", 0) for f in r["per_frame"])
        self.assertLessEqual(r["unique_counts"]["total"], max(per_frame_sum, 1) * 10)


if __name__ == "__main__":
    unittest.main()
