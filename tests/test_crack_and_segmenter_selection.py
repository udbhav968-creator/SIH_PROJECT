"""Crack verifier (serving rules, crops, pipeline gate), the deep-segmenter candidate rule, and the git-hosted
crack datasets."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class FakeEmbedder:
    is_ready, name = True, "mobilenetv2"

    def embed_batch(self, crops):
        # brightness of the crop's darkest 3%: thin dark lines (cracks) score high
        return np.array([[np.percentile(np.asarray(c, float), 3) < 60, 1.0] + [0.0] * 6 for c in crops], float)


def write_verifier(d, served=True, sha="auto"):
    from models.crack_verifier import MODEL_FILE, REPORT_FILE, SELECTION_FILE, _sha16
    np.savez(os.path.join(d, MODEL_FILE), mean=np.zeros(8), scale=np.ones(8),
             coef=np.array([8.0, -4.0] + [0.0] * 6), intercept=np.float64(0.0),
             meta_json=np.array(json.dumps({"threshold": 0.5})))
    with open(os.path.join(d, REPORT_FILE), "w") as fh:
        json.dump({"test": {}}, fh)
    sel = {"served": served, "why": "test",
           "model_sha16": _sha16(os.path.join(d, MODEL_FILE)) if sha == "auto" else sha}
    with open(os.path.join(d, SELECTION_FILE), "w") as fh:
        json.dump(sel, fh)


class CrackVerifierTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_served_only_when_decided_for_this_file(self):
        from models.crack_verifier import SELECTION_FILE, CrackVerifier
        write_verifier(self.d, served=False)
        self.assertFalse(CrackVerifier(self.d, embedder=FakeEmbedder()).is_ready)
        self.assertTrue(CrackVerifier(self.d, embedder=FakeEmbedder(), require_served=False).is_ready)
        write_verifier(self.d, served=True, sha="0" * 16)
        v = CrackVerifier(self.d, embedder=FakeEmbedder())
        self.assertFalse(v.is_ready, "a decision made for another model file does not serve this one")
        self.assertIn("changed", v.load_error)
        write_verifier(self.d, served=True)
        os.remove(os.path.join(self.d, SELECTION_FILE))
        self.assertFalse(CrackVerifier(self.d, embedder=FakeEmbedder()).is_ready, "no decision, not served")
        write_verifier(self.d, served=True)
        v = CrackVerifier(self.d, embedder=FakeEmbedder())
        self.assertTrue(v.is_ready)
        dark = np.full((60, 60, 3), 200, np.uint8)
        dark[:, 28:32] = 10
        p = v.proba([dark, np.full((60, 60, 3), 200, np.uint8)])
        self.assertGreater(p[0], 0.5)
        self.assertLess(p[1], 0.5)

    def test_crop_has_context_and_a_minimum_size(self):
        from models.crack_verifier import crop_around
        img = np.zeros((200, 300, 3), np.uint8)
        self.assertEqual(crop_around(img, (100, 100, 4, 4)).shape[:2], (48, 48))
        self.assertEqual(crop_around(img, (0, 0, 300, 200)).shape[:2], (200, 300))
        self.assertEqual(crop_around(img, (290, 190, 10, 10)).shape[:2], (48, 48), "shifted inwards at the border")

    def test_pipeline_drops_only_crack_components_it_rejects(self):
        from models.crack_verifier import CrackVerifier
        from pipeline.deep_inference_pipeline import DeepInferencePipeline
        write_verifier(self.d, served=True)
        pipe = DeepInferencePipeline.__new__(DeepInferencePipeline)
        pipe.crack_verifier = CrackVerifier(self.d, embedder=FakeEmbedder())
        img = np.full((200, 300, 3), 200, np.uint8)
        img[50:150, 50:53] = 5                         # a crack-like dark line at x 50
        pipe._frame_rgb = img
        boxes = [[40, 40, 30, 120, 9.0, 1, "segmentation"],     # crack over the line: kept
                 [200, 40, 30, 120, 8.0, 1, "segmentation"],    # crack over plain grey: dropped
                 [200, 40, 30, 120, 7.0, 2, "segmentation"]]    # pothole: never touched by this gate
        out = pipe._verify_cracks(boxes)
        self.assertEqual([b[4] for b in out], [9.0, 7.0])
        self.assertEqual(pipe._crack_gate["removed"], 1)
        pipe.crack_verifier = None
        self.assertEqual(len(pipe._verify_cracks(boxes)), 3, "no verifier: nothing changes")


class DeepSegmenterRuleTest(unittest.TestCase):
    def test_rule(self):
        from scripts.select_deep_segmenter import decide
        cur = {"detected_photos": 20, "false_positive_photos": 3}
        self.assertTrue(decide({"detected_photos": 21, "false_positive_photos": 3}, cur, None, None, True, False)[0])
        self.assertTrue(decide({"detected_photos": 20, "false_positive_photos": 2}, cur, None, None, True, False)[0])
        self.assertFalse(decide({"detected_photos": 22, "false_positive_photos": 4}, cur, None, None, True, False)[0],
                         "more detections never buy more false alarms")
        self.assertFalse(decide({"detected_photos": 19, "false_positive_photos": 0}, cur, None, None, True, False)[0])
        tie = {"detected_photos": 20, "false_positive_photos": 3}
        self.assertTrue(decide(tie, cur, 0.5, 0.4, True, False)[0])
        self.assertFalse(decide(tie, cur, 0.4, 0.5, True, False)[0])
        self.assertFalse(decide({"detected_photos": 24, "false_positive_photos": 0}, cur, 0.9, 0.1, False, False)[0],
                         "an ONNX file that does not reproduce the network is never served")
        self.assertFalse(decide({"detected_photos": 24, "false_positive_photos": 0}, cur, 0.9, 0.1, True, True)[0])


class GitCrackSourcesTest(unittest.TestCase):
    def test_crackforest_layout_from_a_local_git_repository(self):
        from PIL import Image
        from scipy.io import savemat
        from scripts import fetch_seg_datasets as fsd
        d = tempfile.mkdtemp()
        try:
            repo = os.path.join(d, "repo")
            os.makedirs(os.path.join(repo, "image"))
            os.makedirs(os.path.join(repo, "groundTruth"))
            for i in range(3):
                Image.fromarray(np.random.default_rng(i).integers(0, 255, (320, 480, 3), dtype=np.uint8)).save(
                    os.path.join(repo, "image", f"00{i}.jpg"))
                seg = np.ones((320, 480), np.uint8)
                seg[100:110, :] = 2                      # crack pixels are 2 in CFD's Segmentation
                gt = np.empty((1, 1), dtype=[("Segmentation", "O"), ("Boundaries", "O")])
                gt[0, 0] = (seg, np.zeros_like(seg))
                savemat(os.path.join(repo, "groundTruth", f"00{i}.mat"), {"groundTruth": gt})
            for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"]):
                subprocess.run(cmd, cwd=repo, check=True)
            out = os.path.join(d, "seg_multi")

            class NoLeak:
                def is_leak(self, img):
                    return False
            old_out, old_src = fsd.OUT, dict(fsd.GIT_SOURCES)
            fsd.OUT = out
            fsd.GIT_SOURCES["crackforest"] = dict(old_src["crackforest"], repo=repo)
            try:
                s = fsd.fetch_git_source("crackforest", NoLeak(), raw_root=os.path.join(d, "raw"))
            finally:
                fsd.OUT, fsd.GIT_SOURCES = old_out, old_src
            self.assertEqual((s["kept"], s["pairs_found"]), (3, 3))
            self.assertEqual(len(s["commit"]), 40)
            lab = np.asarray(Image.open(sorted(__import__("glob").glob(os.path.join(out, "crackforest", "lab", "*.png")))[0]))
            self.assertEqual(lab.shape, (fsd.IN_H, fsd.IN_W))
            self.assertEqual(set(np.unique(lab)) - {0, 1}, set(), "CFD's 2 becomes this project's crack class 1")
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
