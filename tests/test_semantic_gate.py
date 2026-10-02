import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import semantic_gate


class _Fake:
    """Classifier stub: says 'pothole' only for windows in the left half."""
    def __init__(self, W): self.W = W; self.calls = 0
    def predict_probabilities(self, crop):
        self.calls += 1
        p = np.zeros(7); p[2 if crop.mean() > 100 else 0] = 1.0
        return p


class SemanticGateTest(unittest.TestCase):
    def test_drops_pothole_pixels_outside_pothole_heat(self):
        img = np.zeros((100, 200, 3), np.uint8); img[:, :100] = 200     # bright left half = "pothole" windows
        mask = np.zeros((100, 200), np.uint8); mask[40:60, 10:30] = 2; mask[40:60, 170:190] = 2; mask[5:8, :] = 1
        out = semantic_gate.apply({"mask": mask, "pothole_px": int((mask == 2).sum())}, _Fake(200), img)
        self.assertTrue((out["mask"][40:60, 10:30] == 2).all(), "pothole inside the heat region must survive")
        self.assertFalse((out["mask"][40:60, 170:190] == 2).any(), "pothole outside the heat region must be dropped")
        self.assertTrue((out["mask"][5:8, :] == 1).all(), "cracks are never touched")
        self.assertEqual(out["pothole_px"], 400)
        self.assertGreater(out["semantic_gate"]["pothole_pixels_removed"], 0)

    def test_report_exists_and_shows_improvement(self):
        import json
        p = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "semantic_gate_report.json")
        r = json.load(open(p))["results"]
        self.assertGreater(r["C_seg_gated_0.5"]["pothole"]["iou"], r["A_segmenter"]["pothole"]["iou"])


if __name__ == "__main__":
    unittest.main()
