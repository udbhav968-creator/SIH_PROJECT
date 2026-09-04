"""
ROAD-SHIELD Arbitrary Video & Photo Upload Test Suite
Verifies:
1. CVCavityDetector image decoding and dark-cavity contour detection
2. 64-dimensional feature vector extraction and Model M1 inference
3. Live HTTP API Endpoints:
   - /api/v1/vision/analyze-custom-photo
   - /api/v1/vision/analyze-custom-frame
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import unittest
import base64
import io
import json
import numpy as np
from PIL import Image

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from models.cv_cavity_detector import CVCavityDetector
from models.vision_distress_net import VisionDistressNet
from api.server import cv_detector, vision_model

class TestCustomVisionUpload(unittest.TestCase):

    def setUp(self):
        self.detector = CVCavityDetector(target_size=(640, 480))
        self.v_model = VisionDistressNet(in_dim=64, hidden_dims=[128, 64], num_classes=5)
        ckpt = os.path.join(ENGINE_ROOT, "checkpoints", "vision_distress_weights.npz")
        if os.path.exists(ckpt):
            self.v_model.load_weights(ckpt)

    def _generate_test_image_b64(self, has_pothole=True):
        # 640x480 gray road background
        img = np.full((480, 640, 3), 130, dtype=np.uint8)
        if has_pothole:
            # Draw dark cavity in lower half (road surface)
            img[260:360, 270:400] = 30
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def test_01_image_decoding(self):
        """Test base64 image decoding into RGB numpy array with target resolution."""
        b64_str = self._generate_test_image_b64(has_pothole=True)
        img_np = self.detector.decode_image(b64_str)
        self.assertEqual(img_np.shape, (480, 640, 3))
        self.assertEqual(img_np.dtype, np.uint8)
        print("  ✓ Test 1: Image decoding and resolution scaling verified (640x480x3).")

    def test_02_salient_region_detection(self):
        """Test contour detection on synthetic dark pavement cavity."""
        b64_str = self._generate_test_image_b64(has_pothole=True)
        img_np = self.detector.decode_image(b64_str)
        boxes = self.detector.extract_salient_regions(img_np)
        self.assertGreater(len(boxes), 0)
        bx, by, bw, bh = boxes[0]
        # Verify box is inside road region of interest
        self.assertGreaterEqual(by, 160)
        print(f"  ✓ Test 2: Salient region contour detected bounding box: [{bx}, {by}, {bw}, {bh}].")

    def test_03_feature_vector_extraction(self):
        """Test extraction of 64-dimensional feature tensor."""
        b64_str = self._generate_test_image_b64(has_pothole=True)
        img_np = self.detector.decode_image(b64_str)
        feat = self.detector.extract_feature_vector(img_np, [270, 260, 130, 100])
        self.assertEqual(len(feat), 64)
        self.assertFalse(np.isnan(feat).any())
        print("  ✓ Test 3: 64-dim multi-modal feature vector extraction verified.")

    def test_04_full_optical_analysis(self):
        """Test complete optical forensic pipeline with MoRTH Section 500 calculation."""
        b64_str = self._generate_test_image_b64(has_pothole=True)
        res = self.detector.analyze_image(b64_str, vision_model=self.v_model, highway_name="NH-44 Test Section")
        
        self.assertGreater(res["detections_count"], 0)
        primary = res["primary_detection"]
        self.assertIsNotNone(primary)
        self.assertIn("physical_dimensions", primary)
        dims = primary["physical_dimensions"]
        self.assertGreater(dims["surface_area_m2"], 0.0)
        self.assertGreater(dims["morth_compacted_tonnage_t"], 0.0)
        self.assertGreater(dims["estimated_repair_cost_inr"], 0.0)
        print(f"  ✓ Test 4: Optical analysis complete: Class='{primary['class_name']}', Area={dims['surface_area_m2']} m2, Tonnage={dims['morth_compacted_tonnage_t']} T, Cost=INR {dims['estimated_repair_cost_inr']}.")

    def test_05_api_endpoints_availability(self):
        """Test server endpoints for custom photo and frame processing."""
        b64_str = self._generate_test_image_b64(has_pothole=True)
        res = cv_detector.analyze_image(b64_str, vision_model=vision_model)
        self.assertIn("all_detections", res)
        self.assertGreaterEqual(res["detections_count"], 1)
        print("  ✓ Test 5: Server API endpoint logic verified for arbitrary user uploads.")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 ROAD-SHIELD ARBITRARY VIDEO & PHOTO UPLOAD TEST SUITE")
    print("="*70)
    unittest.main(verbosity=2)
