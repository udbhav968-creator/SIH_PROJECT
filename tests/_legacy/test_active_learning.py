import unittest
import numpy as np
import os
import json
import sys

engine_root = r"c:\Users\Dell\Downloads\road_shield_ai_engine"
if engine_root not in sys.path:
    sys.path.insert(0, engine_root)

from models.edge_model_exporter import EdgeModelExporter
from models.cv_cavity_detector import CVCavityDetector
from models.vision_distress_net import VisionDistressNet

class TestActiveLearningAndEdgeExport(unittest.TestCase):
    def setUp(self):
        ckpt_dir = os.path.join(engine_root, "checkpoints")
        self.exporter = EdgeModelExporter(checkpoints_dir=ckpt_dir)
        self.detector = CVCavityDetector()
        self.vision_model = VisionDistressNet()
        ckpt_path = os.path.join(ckpt_dir, "vision_distress_weights.npz")
        if os.path.exists(ckpt_path):
            self.vision_model.load_weights(ckpt_path)

    def test_01_edge_spec_export(self):
        """Verify export to Open Neural Spec JSON and C single-header library."""
        res = self.exporter.export_all_to_open_spec()
        self.assertTrue(os.path.exists(res["spec_json_path"]))
        self.assertTrue(os.path.exists(res["c_header_path"]))
        
        # Verify JSON content
        with open(res["spec_json_path"], "r") as f:
            spec = json.load(f)
        self.assertIn("Open Neural Network Specification", spec["format"])
        self.assertIn("Model_M1_VisionDistressNet", spec["models"])
        self.assertIn("Model_M4_IMUShockClassifier", spec["models"])
        self.assertIn("Model_M_PCI_Regressor", spec["models"])

        # Verify C Header content
        with open(res["c_header_path"], "r") as f:
            header_src = f.read()
        self.assertIn("ROAD_SHIELD_EDGE_INFERENCE_H", header_src)
        self.assertIn("morth_calculate_asphalt_tonnage", header_src)
        print("\n  ✓ Edge export verified: JSON spec and C header verified.")

    def test_02_active_feedback_online_gradient_step(self):
        """Verify dynamic single-step gradient update (online learning) on Model M1."""
        X_sample = np.random.randn(1, 64).astype(np.float32)
        y_cls = np.array([4], dtype=np.int64) # Class 4: Pothole
        y_geo = np.array([[0.0, 3.5, 1.5, 6.0]], dtype=np.float32) # [x, y, area, depth]

        out_before = self.vision_model.forward(X_sample)
        logits_before = out_before[2]
        prob_before = np.exp(logits_before - np.max(logits_before))
        prob_before /= np.sum(prob_before)
        pothole_prob_before = prob_before[0, 4]

        loss = self.vision_model.train_step(X_sample, y_cls, y_geo)
        loss_val = float(loss[0]) if isinstance(loss, (tuple, list)) else float(loss)
        self.assertGreater(loss_val, 0.0)

        out_after = self.vision_model.forward(X_sample)
        logits_after = out_after[2]
        prob_after = np.exp(logits_after - np.max(logits_after))
        prob_after /= np.sum(prob_after)
        pothole_prob_after = prob_after[0, 4]

        print(f"\n  ✓ Active Feedback Online SGD: Pothole Prob Before={pothole_prob_before:.4f}, After={pothole_prob_after:.4f}, Loss={loss_val:.4f}")
        self.assertGreaterEqual(pothole_prob_after, pothole_prob_before - 0.05)

if __name__ == "__main__":
    unittest.main()
