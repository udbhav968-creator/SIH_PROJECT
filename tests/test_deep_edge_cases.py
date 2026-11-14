"""
Comprehensive Edge-Case & Robustness Test Suite for ROAD-SHIELD.
Verifies that all models, pipelines, and converters handle malformed,
extreme, or boundary inputs gracefully without throwing uncaught exceptions.
"""

import unittest
import numpy as np
from PIL import Image


class TestDeepEdgeCases(unittest.TestCase):
    """Stress tests and boundary condition validation across neural & civil engines."""

    def test_yolo_video_engine_decode_image_edge_cases(self):
        from models.yolo_video_engine import YOLOVideoEngine

        # 1. 2D grayscale
        gray = np.full((120, 160), 128, dtype=np.uint8)
        decoded = YOLOVideoEngine.decode_image(gray)
        self.assertEqual(decoded.ndim, 3)
        self.assertEqual(decoded.shape, (120, 160, 3))
        self.assertEqual(decoded.dtype, np.uint8)

        # 2. 4-channel RGBA
        rgba = np.full((100, 100, 4), 200, dtype=np.uint8)
        decoded = YOLOVideoEngine.decode_image(rgba)
        self.assertEqual(decoded.shape, (100, 100, 3))
        self.assertEqual(decoded.dtype, np.uint8)

        # 3. 1-channel (H, W, 1)
        single = np.full((80, 80, 1), 75, dtype=np.uint8)
        decoded = YOLOVideoEngine.decode_image(single)
        self.assertEqual(decoded.shape, (80, 80, 3))

        # 4. Floating point [0..1]
        floats = np.full((60, 60, 3), 0.5, dtype=np.float32)
        decoded = YOLOVideoEngine.decode_image(floats)
        self.assertEqual(decoded.dtype, np.uint8)
        self.assertEqual(int(decoded[0, 0, 0]), 127)

        # 5. PIL Image
        pil_img = Image.new("RGB", (64, 48), color=(255, 0, 0))
        decoded = YOLOVideoEngine.decode_image(pil_img)
        self.assertEqual(decoded.shape, (48, 64, 3))
        self.assertEqual(decoded.dtype, np.uint8)

        # 6. Oversized image scaling
        huge = np.zeros((1200, 1600, 3), dtype=np.uint8)
        decoded = YOLOVideoEngine.decode_image(huge, max_dim=800)
        self.assertLessEqual(max(decoded.shape[:2]), 800)

    def test_yolo_video_engine_detect_frame_tiny_and_empty(self):
        from models.yolo_video_engine import get_yolo_video_engine
        eng = get_yolo_video_engine()

        # 1x1 image frame below minimum resolution
        tiny = np.zeros((1, 1, 3), dtype=np.uint8)
        res = eng.detect_frame(tiny, frame_index=0, timestamp_s=0.0)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["detections"], [])
        self.assertEqual(res["video_pci_score"], 100.0)

    def test_vimu_temporal_fusion_empty_and_malformed(self):
        from models.vimu_temporal_fusion import VIMUTemporalFusionEngine
        eng = VIMUTemporalFusionEngine()

        # 1. Empty array (0, 3)
        empty = np.zeros((0, 3), dtype=np.float64)
        self.assertIsNone(eng.extract_imu_features(empty))

        # 2. 1D array
        one_d = np.array([1.0, 2.0, 9.8])
        self.assertIsNone(eng.extract_imu_features(one_d))

        # 3. None / String
        self.assertIsNone(eng.extract_imu_features(None))
        self.assertIsNone(eng.extract_imu_features("invalid"))

        # 4. Valid 100 Hz shock window
        valid = np.full((100, 3), 0.1)
        valid[:, 2] = 9.81
        valid[50, 2] = 14.5  # pothole shock
        feats = eng.extract_imu_features(valid)
        self.assertIsNotNone(feats)
        self.assertGreater(feats["peak_vertical_shock_ms2"], 4.0)

    def test_metric_3d_depth_rgb_and_tiny_guard(self):
        from models.metric_3d_depth import Metric3DDepthEngine
        eng = Metric3DDepthEngine()

        # 1. 3-channel RGB image (should auto-convert without crashing)
        rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
        res = eng.evaluate_cavity_3d(rgb, bbox_pixels=[10, 10, 20, 20], class_id=2)
        self.assertIn("depth_cm", res)
        self.assertIn("volume_m3", res)
        self.assertIn("point_cloud_3d", res)

        # 2. Out-of-bounds bbox
        res_oob = eng.evaluate_cavity_3d(rgb, bbox_pixels=[-50, -50, 500, 500], class_id=1)
        self.assertIn("depth_cm", res_oob)

        # 3. Tiny 2x2 image
        tiny = np.zeros((2, 2), dtype=np.uint8)
        res_tiny = eng.evaluate_cavity_3d(tiny, bbox_pixels=[0, 0, 2, 2], class_id=2)
        self.assertIn("depth_cm", res_tiny)

    def test_surface_classifier_rgba_and_empty(self):
        from models.surface_classifier import SurfaceTypeClassifier
        clf = SurfaceTypeClassifier()

        # 1. 4-channel RGBA
        rgba = np.full((80, 80, 4), 90, dtype=np.uint8)
        res = clf.analyze_surface(rgba)
        self.assertIn("name", res)
        self.assertIn("civil_standard", res)

        # 2. Tiny image
        tiny = np.zeros((2, 2, 3), dtype=np.uint8)
        res_tiny = clf.analyze_surface(tiny)
        self.assertIn("name", res_tiny)

    def test_pinn_hdm4_none_and_zero_values(self):
        from models.pinn_deterioration import PhysicsInformedHDM4Forecaster
        pinn = PhysicsInformedHDM4Forecaster()

        # 1. None values
        traj = pinn.forecast_trajectory(
            init_area_m2=None, depth_cm=None, esal_trucks=None, rain_mm=None, pavement_age_yr=None
        )
        self.assertIn("trajectory_days", traj)
        self.assertIn("180", traj["trajectory_days"])
        self.assertGreater(traj["trajectory_days"]["180"]["area_m2"], 0.0)

        # 2. Zero values
        k_vel = pinn.compute_daily_breakout_velocity(
            esal_trucks=0, rain_mm=0, initial_area_m2=0, pavement_age_yr=0
        )
        self.assertGreater(k_vel, 0.0)

    def test_forensic_vlm_none_and_empty_state(self):
        from models.forensic_vlm_auditor import ForensicVLMAuditor
        auditor = ForensicVLMAuditor()

        # 1. None state
        rep_none = auditor.generate_audit_report(None)
        self.assertIn("forensic_report_id", rep_none)
        self.assertIn("executive_narrative", rep_none)
        self.assertIn("morth_schedule_of_rates", rep_none)

        # 2. Empty dict state
        rep_empty = auditor.generate_audit_report({})
        self.assertIn("forensic_report_id", rep_empty)
        self.assertIn("morth_schedule_of_rates", rep_empty)


if __name__ == "__main__":
    unittest.main()
