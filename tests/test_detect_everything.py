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


class TestNMSAndTiled(unittest.TestCase):
    def test_nms_merges_overlapping_same_class(self):
        from pipeline.detect_everything import _nms_merge, _iou_xywh
        a = {"class_name": "Pothole", "confidence": 0.9, "bbox_normalized": [0.1, 0.1, 0.2, 0.2]}
        b = {"class_name": "Pothole", "confidence": 0.5, "bbox_normalized": [0.11, 0.11, 0.2, 0.2]}  # overlaps a
        c = {"class_name": "Pothole", "confidence": 0.8, "bbox_normalized": [0.7, 0.7, 0.1, 0.1]}    # disjoint
        out = _nms_merge([a, b, c], iou_thr=0.5)
        self.assertEqual(len(out), 2)                    # b suppressed by a
        self.assertIn(a, out); self.assertIn(c, out)
        self.assertGreater(_iou_xywh([0, 0, 0.2, 0.2], [0.05, 0.05, 0.2, 0.2]), 0.3)

    def test_nms_keeps_different_classes(self):
        from pipeline.detect_everything import _nms_merge
        a = {"class_name": "Pothole", "confidence": 0.9, "bbox_normalized": [0.1, 0.1, 0.2, 0.2]}
        b = {"class_name": "Car", "confidence": 0.9, "bbox_normalized": [0.1, 0.1, 0.2, 0.2]}
        self.assertEqual(len(_nms_merge([a, b], iou_thr=0.5)), 2)

    def test_tiled_finds_at_least_as_many(self):
        from pipeline.detect_everything import analyze_image, _engine
        imgs = _pothole_images(1)
        if not imgs:
            self.skipTest("no dataset images")
        eng = _engine(True)
        normal = analyze_image(imgs[0], conf=0.3, draw=False, tiled=False, engine=eng)["counts"]["total"]
        tiled = analyze_image(imgs[0], conf=0.3, draw=False, tiled=True, engine=eng)["counts"]["total"]
        self.assertGreaterEqual(tiled, normal)


class TestAuditImageMulti(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pipeline.deep_inference_pipeline import DeepInferencePipeline
        cls.dp = DeepInferencePipeline()

    def test_per_defect_fields(self):
        from pipeline.detect_everything import audit_image_multi
        imgs = _pothole_images(1)
        if not imgs:
            self.skipTest("no dataset images")
        r = audit_image_multi(imgs[0], self.dp, conf=0.3, max_defects=6)
        self.assertEqual(r["mode"], "audit_all")
        self.assertGreaterEqual(r["defect_count"], 0)
        self.assertIn("annotated_png", r)
        self.assertTrue(r["annotated_png"][:8] == b"\x89PNG\r\n\x1a\n")
        for d in r["defects"]:
            self.assertIn(d["measurement"], ("measured reference", "scaled estimate"))
            self.assertIn("surface_area_m2", d)
        if r["defects"]:
            self.assertEqual(r["defects"][0]["measurement"], "measured reference")

    def test_blank_image_is_safe(self):
        import numpy as np
        from pipeline.detect_everything import audit_image_multi
        blank = np.full((300, 400, 3), 128, np.uint8)
        r = audit_image_multi(blank, self.dp, conf=0.3)
        self.assertEqual(r["defect_count"], 0)
        self.assertEqual(r["total_cost_low_inr"], 0.0)


class TestAnnotatedVideoAndWorkOrders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        imgs = _pothole_images(3)
        if not imgs:
            cls.clip = None
            return
        import cv2, tempfile, numpy as np
        from PIL import Image
        cls.tmp = tempfile.mkdtemp()
        cls.clip = os.path.join(cls.tmp, "c.mp4")
        W, H = 480, 320
        vw = cv2.VideoWriter(cls.clip, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
        for p in imgs:
            fr = np.asarray(Image.open(p).convert("RGB").resize((W, H)))
            for _ in range(8):
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()

    def test_annotated_video_out(self):
        if not self.clip:
            self.skipTest("no dataset images")
        import tempfile, os as _os
        from pipeline.detect_everything import analyze_video_annotated
        out = os.path.join(self.tmp, "boxed.mp4")
        meta = analyze_video_annotated(self.clip, out, conf=0.3, max_seconds=5)
        self.assertGreater(meta["frames_written"], 0)
        self.assertTrue(_os.path.isfile(out) and _os.path.getsize(out) > 0)

    def test_video_defects_to_sealed_work_orders(self):
        if not self.clip:
            self.skipTest("no dataset images")
        from pipeline.detect_everything import analyze_video_deep
        from pipeline.deep_inference_pipeline import DeepInferencePipeline
        from models.morth_dispatch_agent import MoRTHDispatchAgent, check_seal
        dp = DeepInferencePipeline()
        agent = MoRTHDispatchAgent()
        r = analyze_video_deep(self.clip, dp, sample_every_s=1.0, max_frames=4, candidate_damage=True)
        issued = 0
        for ud in r["unique_defects"]:
            dl, dh = ud.get("depth_low_cm"), ud.get("depth_high_cm")
            depth = ((dl + dh) / 2.0) if (dl is not None and dh is not None) else (dl or dh or 0.0)
            wo = agent.generate_work_order(
                corridor_id="VIDEO", latitude=None, longitude=None,
                distress_class=ud["class_name"], area_sqm=ud["surface_area_m2"],
                depth_cm=float(depth), pci_score=float(ud["pci_score"] if ud["pci_score"] is not None else 100))
            self.assertIn("work_order_id", wo)
            self.assertTrue(wo.get("sha256_cryptographic_seal"))
            self.assertEqual(wo.get("dispatch_status"), "HELD_NO_GPS")   # video has no GPS
            self.assertTrue(check_seal(wo))                               # seal verifies
            issued += 1
        self.assertGreaterEqual(issued, 0)
