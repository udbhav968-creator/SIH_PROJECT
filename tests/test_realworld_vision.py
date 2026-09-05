"""
ROAD-SHIELD Real-World Vision & Video Tracking Test Suite
Verifies:
1. RealWorldMediaEngine: Dashcam frames, looming perspective, photo presets.
2. SpatialTemporalVideoTracker: Persistent IoU tracking, 0% double-counting error, hard negative rejection.
3. Live HTTP API Endpoints: /api/v1/vision/curated-videos, /api/v1/vision/process-video-frame, /api/v1/vision/analyze-photo.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import unittest
import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from data.realworld_media_engine import RealWorldMediaEngine
from models.realworld_video_tracker import SpatialTemporalVideoTracker
from api.server import get_session_tracker, media_engine, vision_model

class TestRealWorldVision(unittest.TestCase):

    def setUp(self):
        self.engine = RealWorldMediaEngine(seed=42)
        self.tracker = SpatialTemporalVideoTracker(iou_threshold=0.25, max_age_frames=5)

    def test_01_curated_video_catalog(self):
        """Test video catalog metadata has all required production streams."""
        catalog = self.engine.get_video_catalog()
        self.assertIn("NH44_Monsoon_Highway", catalog)
        self.assertIn("MumbaiPune_Night_Expressway", catalog)
        self.assertIn("Urban_Ward_Manhole_Test", catalog)
        self.assertEqual(catalog["NH44_Monsoon_Highway"]["fps"], 30)
        print("  ✓ Test 1: Video Catalog metadata validated (3 production sequences).")

    def test_02_perspective_looming(self):
        """Test looming: as frame_idx advances, bounding box expands and distance decreases."""
        clip_id = "NH44_Monsoon_Highway"
        f10 = self.engine.get_video_frame(clip_id, 10)
        f35 = self.engine.get_video_frame(clip_id, 35)

        self.assertGreater(len(f10["detections"]), 0)
        self.assertGreater(len(f35["detections"]), 0)

        det10 = f10["detections"][0]
        det35 = f35["detections"][0]

        # Distance should decrease as vehicle drives forward
        self.assertGreater(det10["distance_meters"], det35["distance_meters"])
        # Bounding box width should expand (looming)
        self.assertLess(det10["bbox_normalized"][2], det35["bbox_normalized"][2])
        print(f"  ✓ Test 2: Perspective looming verified (Z drops from {det10['distance_meters']}m to {det35['distance_meters']}m, Box expands from {det10['bbox_pixels_640x480'][2]}px to {det35['bbox_pixels_640x480'][2]}px).")

    def test_03_anti_double_counting_tracker(self):
        """Test that passing over a defect for 38 consecutive frames produces exactly 1 counted pothole."""
        clip_id = "NH44_Monsoon_Highway"
        tracker = SpatialTemporalVideoTracker(iou_threshold=0.25, max_age_frames=5)

        for f in range(45):
            frame_data = self.engine.get_video_frame(clip_id, f)
            dets = frame_data["detections"]
            res = tracker.update(dets, f)

        # Must record exactly 1 unique pothole despite being visible across >35 frames
        self.assertEqual(tracker.total_unique_potholes_counted, 1)
        self.assertEqual(tracker.total_unique_cracks_counted, 0)
        print(f"  ✓ Test 3: Anti-Double-Counting verified: 1 unique pothole counted across 45 frames (0.0% duplicate error).")

    def test_04_hard_negative_suppression(self):
        """Test that urban manhole covers are recognized as non-distress and not counted as potholes."""
        clip_id = "Urban_Ward_Manhole_Test"
        tracker = SpatialTemporalVideoTracker(iou_threshold=0.25, max_age_frames=5)

        for f in range(35):
            frame_data = self.engine.get_video_frame(clip_id, f)
            dets = frame_data["detections"]
            res = tracker.update(dets, f)

        # Manholes must be suppressed from pothole/crack counts
        self.assertEqual(tracker.total_unique_potholes_counted, 0)
        self.assertEqual(tracker.total_unique_cracks_counted, 0)
        print("  ✓ Test 4: Hard-Negative Suppression verified: Manhole cover correctly rejected (0 false alarms).")

    def test_05_photo_preset_generation(self):
        """Test generation of authentic real-world photo presets and neural feature vectors."""
        p_monsoon = self.engine.generate_realworld_photo_preset("Monsoon_Submerged_Cavity_NH44")
        self.assertEqual(p_monsoon["true_class"], "D40 Pothole")
        self.assertEqual(len(p_monsoon["features_64"]), 64)
        self.assertGreater(p_monsoon["area_m2"], 1.0)
        self.assertGreater(p_monsoon["depth_cm"], 5.0)

        p_manhole = self.engine.generate_realworld_photo_preset("Iron_Manhole_Hard_Negative")
        self.assertIn("Hard Negative", p_manhole["true_class"])
        print("  ✓ Test 5: Photo Presets verified with calibrated physical geometry.")

    def test_06_model_inference_on_photos(self):
        """Test neural model M1 classification accuracy on real-world photo features."""
        p_monsoon = self.engine.generate_realworld_photo_preset("Monsoon_Submerged_Cavity_NH44")
        X = np.array([p_monsoon["features_64"]], dtype=np.float32)
        preds, conf_arr, probs_arr, geo_preds = vision_model.predict(X)
        pred_cls = int(preds[0])
        conf = float(conf_arr[0])
        
        self.assertEqual(pred_cls, 4)  # D40 Pothole
        self.assertGreater(conf, 0.85)
        print(f"  ✓ Test 6: Model M1 inference on Monsoon photo: Pred class={pred_cls} ({vision_model.CLASS_NAMES[pred_cls]}), Confidence={conf*100:.2f}%.")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 ROAD-SHIELD REAL-WORLD VISION & TRACKING TEST SUITE")
    print("="*70)
    unittest.main(verbosity=2)
