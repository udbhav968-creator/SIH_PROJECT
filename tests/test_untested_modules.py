"""
Unit tests for modules that had no coverage at all: the ALPR incident tracker,
the video tracker, the edge exporter, the augmentation pipeline, late fusion,
the ADAS policy agent and the CAN telematics encoder.

Several of these pin behaviour that was wrong before this file was written:
  - incident reports invented a Delhi GPS fix when none was supplied
  - a scalar speed reading was labelled "reckless lane cutting"
  - every non-pothole hazard was counted as a crack by the video tracker
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")


# ---------------------------------------------------------------------------
class ALPRIncidentTrackerTest(unittest.TestCase):
    def setUp(self):
        from models.alpr_incident_tracker import ALPRIncidentTracker
        self.t = ALPRIncidentTracker()

    def _track(self, growth=1.0, sway=0.0):
        return [{"timestamp": 0.2 * i,
                 "bbox": [200 + (sway if i % 2 else -sway), 150, 80 * growth ** i, 60 * growth ** i]}
                for i in range(4)]

    def test_short_track_is_insufficient(self):
        r = self.t.analyze_vehicle_kinematics(self._track()[:2])
        self.assertFalse(r["is_rash_driving"])
        self.assertEqual(r["anomaly_type"], "INSUFFICIENT_TELEMETRY")

    def test_steady_vehicle_is_normal(self):
        r = self.t.analyze_vehicle_kinematics(self._track(growth=1.0))
        self.assertFalse(r["is_rash_driving"])
        self.assertEqual(r["anomaly_type"], "NORMAL_FLOW")

    def test_fast_approach_is_flagged(self):
        r = self.t.analyze_vehicle_kinematics(self._track(growth=1.8))
        self.assertTrue(r["is_rash_driving"])
        self.assertEqual(r["anomaly_type"], "EXCESSIVE_APPROACH_VELOCITY")

    def test_swerving_is_flagged(self):
        r = self.t.analyze_vehicle_kinematics(self._track(sway=80))
        self.assertTrue(r["is_rash_driving"])
        self.assertEqual(r["anomaly_type"], "RECKLESS_LANE_CUTTING")

    def test_no_image_means_no_plate(self):
        p = self.t.extract_license_plate(None)
        self.assertFalse(p["detected"])
        self.assertIsNone(p["license_plate_number"])

    def test_blank_crop_finds_no_plate(self):
        self.assertIsNone(self.t.locate_plate_candidate(np.zeros((120, 200, 3), np.uint8)))

    def test_plate_shaped_region_is_located(self):
        import cv2
        img = np.full((200, 300, 3), 90, np.uint8)
        cv2.rectangle(img, (90, 120), (210, 155), (255, 255, 255), -1)   # ~3.4:1 white plate
        cv2.rectangle(img, (90, 120), (210, 155), (0, 0, 0), 2)
        box = self.t.locate_plate_candidate(img)
        self.assertIsNotNone(box)
        x, y, w, h = box
        self.assertTrue(2.0 <= w / h <= 5.5)
        self.assertLess(abs(x - 90), 10)

    def test_incident_without_gps_has_no_location(self):
        a = self.t.generate_incident_alert("BUS-1", {}, self._track(growth=1.8))
        self.assertIsNone(a["gps_coordinates"], "a location was invented for an incident with no GPS fix")
        a = self.t.generate_incident_alert("BUS-1", None, self._track(growth=1.8))
        self.assertIsNone(a["gps_coordinates"])

    def test_incident_with_gps_keeps_it(self):
        a = self.t.generate_incident_alert("BUS-1", {"lat": 12.97, "lng": 77.59}, self._track(growth=1.8))
        self.assertAlmostEqual(a["gps_coordinates"]["latitude"], 12.97)
        self.assertAlmostEqual(a["gps_coordinates"]["longitude"], 77.59)
        self.assertTrue(a["is_emergency"])
        self.assertEqual(len(a["sha256_seal"]), 64)

    def test_speed_reading_classes(self):
        self.assertEqual(self.t.detect_incident(speed_kmh=60)["incident_class"], "NORMAL_FLOW")
        self.assertEqual(self.t.detect_incident(speed_kmh=90)["incident_class"], "OVERSPEEDING")
        self.assertEqual(self.t.detect_incident(speed_kmh=120)["incident_class"], "EXCESSIVE_APPROACH_VELOCITY")
        self.assertFalse(self.t.detect_incident(speed_kmh=80)["is_emergency"])


# ---------------------------------------------------------------------------
class VideoTrackerTest(unittest.TestCase):
    def setUp(self):
        from models.realworld_video_tracker import SpatialTemporalVideoTracker
        self.tr = SpatialTemporalVideoTracker(iou_threshold=0.25, max_age_frames=2)

    def test_iou_identities(self):
        self.assertAlmostEqual(self.tr.compute_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(self.tr.compute_iou([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)
        self.assertAlmostEqual(self.tr.compute_iou([0, 0, 10, 10], [5, 0, 10, 10]), 50 / 150)

    def test_same_pothole_across_frames_is_counted_once(self):
        for f in range(5):
            out = self.tr.update([{"bbox_normalized": [0.40 + 0.01 * f, 0.5, 0.1, 0.1],
                                   "class_name": "Pothole Cavity", "is_distress": True}], f)
        self.assertEqual(out["total_unique_potholes_counted"], 1)
        self.assertEqual(out["tracked_detections"][0]["track_age_frames"], 5)

    def test_hazard_classes_are_counted_separately(self):
        dets = [
            {"bbox_normalized": [0.1, 0.1, 0.1, 0.1], "class_name": "Pothole Cavity", "is_distress": True},
            {"bbox_normalized": [0.4, 0.4, 0.1, 0.1], "class_name": "Crack (Longitudinal / Transverse / Alligator)", "is_distress": True},
            {"bbox_normalized": [0.7, 0.7, 0.1, 0.1], "class_name": "Waterlogging / Flooding Hazard", "is_distress": True},
            {"bbox_normalized": [0.7, 0.1, 0.1, 0.1], "class_name": "Missing Zebra Crossing", "is_distress": True},
        ]
        out = self.tr.update(dets, 0)
        self.assertEqual(out["total_unique_potholes_counted"], 1)
        self.assertEqual(out["total_unique_cracks_counted"], 1, "non-crack hazards were counted as cracks")
        self.assertEqual(out["total_unique_other_hazards_counted"], 2)

    def test_hard_negatives_are_tracked_but_not_counted(self):
        out = self.tr.update([{"bbox_normalized": [0.2, 0.2, 0.1, 0.1], "class_name": "Normal Road",
                               "is_distress": False}], 0)
        self.assertEqual(out["active_tracks_count"], 1)
        self.assertEqual(out["total_unique_potholes_counted"] + out["total_unique_cracks_counted"]
                         + out["total_unique_other_hazards_counted"], 0)

    def test_stale_tracks_are_evicted_and_return_as_new(self):
        d = [{"bbox_normalized": [0.4, 0.4, 0.1, 0.1], "class_name": "Pothole Cavity", "is_distress": True}]
        self.tr.update(d, 0)
        out = self.tr.update([], 5)
        self.assertEqual(out["active_tracks_count"], 0)
        out = self.tr.update(d, 6)
        self.assertEqual(out["total_unique_potholes_counted"], 2)


# ---------------------------------------------------------------------------
class AugmentationTest(unittest.TestCase):
    def test_expand_keeps_originals_shapes_and_labels(self):
        from data.augmentation_pipeline import CivilDataAugmentor
        rng = np.random.RandomState(0)
        imgs = [rng.randint(0, 255, (64, 80, 3), dtype=np.uint8) for _ in range(3)]
        out_i, out_l = CivilDataAugmentor(seed=1).expand(imgs, [0, 1, 2], copies_per_image=4)
        self.assertEqual(len(out_i), 15)
        self.assertEqual(list(out_l), [0] * 5 + [1] * 5 + [2] * 5)
        self.assertIs(out_i[0], imgs[0])
        for im in out_i:
            self.assertEqual(im.shape, (64, 80, 3))
            self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(np.array_equal(out_i[1], imgs[0]), "augmented copy identical to original")

    def test_seeded_runs_are_reproducible(self):
        from data.augmentation_pipeline import CivilDataAugmentor
        img = np.random.RandomState(3).randint(0, 255, (48, 48, 3), dtype=np.uint8)
        a = CivilDataAugmentor(seed=7).augment(img)
        b = CivilDataAugmentor(seed=7).augment(img)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
class LateFusionTest(unittest.TestCase):
    def setUp(self):
        from models.multimodal_transformer_fusion import MultimodalLateFusionNet, CLASS_NAMES
        self.net = MultimodalLateFusionNet(imu_weight=0.4)
        self.n = len(CLASS_NAMES)

    def _probs(self, top):
        p = np.full(self.n, 0.05)
        p[top] = 1.0 - 0.05 * (self.n - 1)
        return p

    def test_no_imu_means_no_fusion(self):
        r = self.net.fuse(self._probs(1))
        self.assertFalse(r["imu_fusion_applied"])
        self.assertEqual(r["predicted_class_id"], 1)

    def test_imu_shock_raises_pothole_and_stays_normalised(self):
        base = self._probs(1)
        r = self.net.fuse(base, imu_pothole_prob=1.0)
        self.assertTrue(r["imu_fusion_applied"])
        fused = list(r["fused_probabilities"].values())
        self.assertAlmostEqual(sum(fused), 1.0, places=3)
        self.assertGreater(fused[2], base[2])

    def test_wrong_length_is_rejected(self):
        with self.assertRaises(ValueError):
            self.net.fuse([0.5, 0.5])


# ---------------------------------------------------------------------------
class ADASPolicyAndCANTest(unittest.TestCase):
    def setUp(self):
        from models.automotive_rl_policy_agent import AutomotiveRLPolicyAgent
        from models.automotive_telematics_engine import AutomotiveTelematicsEngine
        self.agent = AutomotiveRLPolicyAgent()
        self.can = AutomotiveTelematicsEngine(checkpoints_dir=tempfile.mkdtemp())

    def test_probabilities_sum_to_one(self):
        r = self.agent.evaluate_telemetry_state(hazard_class_id=2, distance_m=40)
        self.assertAlmostEqual(sum(r["action_probabilities"]), 1.0, places=3)

    def test_imminent_collision_brakes(self):
        r = self.agent.evaluate_telemetry_state(hazard_class_id=2, distance_m=5, vehicle_speed_kmh=80)
        self.assertEqual(r["action_name"], "Emergency Autonomous Braking (AEB)")
        self.assertLess(r["telemetry_metrics"]["safety_margin_m"], 0)

    def test_stationary_vehicle_does_not_divide_by_zero(self):
        r = self.agent.evaluate_telemetry_state(hazard_class_id=2, distance_m=30, vehicle_speed_kmh=0)
        self.assertTrue(np.isfinite(r["telemetry_metrics"]["time_to_collision_sec"]))

    def test_wet_road_lengthens_stopping_distance(self):
        dry = self.agent.evaluate_telemetry_state(distance_m=60, surface_friction_mu=0.75)
        wet = self.agent.evaluate_telemetry_state(distance_m=60, surface_friction_mu=0.35)
        self.assertGreater(wet["telemetry_metrics"]["dynamic_stopping_distance_m"],
                           dry["telemetry_metrics"]["dynamic_stopping_distance_m"])

    def test_can_frame_is_eight_valid_bytes(self):
        dec = self.agent.evaluate_telemetry_state(hazard_class_id=2, distance_m=20)
        f = self.can.generate_adas_can_packet(dec, hazard_class_id=2, ttc_sec=99.0)
        self.assertEqual(f["dlc"], 8)
        self.assertEqual(len(f["data_bytes"]), 8)
        self.assertTrue(all(0 <= b <= 255 for b in f["data_bytes"]))
        self.assertEqual(f["data_bytes"][2], 255, "TTC must saturate, not overflow")
        b = f["data_bytes"]
        self.assertEqual(b[7] & 0x0F, (b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[4] ^ b[5] ^ b[6]) & 0x0F)

    def test_short_payload_is_padded(self):
        f = self.can.encode_can_frame(0x123, [1, 2])
        self.assertEqual(f["data_bytes"], [1, 2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(f["raw_hex"], "01 02 00 00 00 00 00 00")

    def test_dbc_and_header_are_well_formed(self):
        def text(out):
            # these generators write a file and return its path
            if isinstance(out, str) and os.path.isfile(out):
                with open(out, encoding="utf-8") as fh:
                    return fh.read()
            return out if isinstance(out, str) else json.dumps(out)
        dbc = text(self.can.generate_can_dbc())
        self.assertIn("BO_", dbc)
        self.assertIn("SG_", dbc)
        hdr = text(self.can.generate_cpp_ecu_header())
        self.assertIn("RoadShieldECUInference", hdr)


# ---------------------------------------------------------------------------
class EdgeExporterTest(unittest.TestCase):
    def test_export_writes_spec_and_header_from_trained_models(self):
        from models.edge_model_exporter import EdgeModelExporter
        if not os.path.exists(os.path.join(CKPT_DIR, "vision_distress_model.joblib")):
            self.skipTest("no trained vision model - run training.train_mega_suite")
        out = tempfile.mkdtemp()
        r = EdgeModelExporter(CKPT_DIR).export_all_to_open_spec(output_dir=out)
        self.assertTrue(os.path.exists(r["spec_json_path"]))
        self.assertTrue(os.path.exists(r["c_header_path"]))
        self.assertTrue(r["models_exported"])
        with open(r["spec_json_path"]) as fh:
            spec = json.load(fh)
        self.assertIsInstance(spec, dict)
        with open(r["c_header_path"]) as fh:
            header = fh.read()
        self.assertIn("#ifndef", header)
        self.assertIn("#endif", header)
        # the exporter must never write into checkpoints/ when told not to
        self.assertTrue(r["spec_json_path"].startswith(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
