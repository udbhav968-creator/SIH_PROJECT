"""Ensemble selection (validation-only choice, cascade threshold, leakage guard) and the cascade classifier."""
import glob
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
CKPT = os.path.join(ROOT, "checkpoints")


def _served_onnx():
    """The single network finetune_summary.json names (not merely the first file: train-only networks may sit beside it)."""
    try:
        with open(os.path.join(CKPT, "finetune_summary.json"), encoding="utf-8") as fh:
            tag = json.load(fh).get("served")
        p = os.path.join(CKPT, f"deep_vision_{tag}.onnx")
        if tag and os.path.exists(p) and os.path.exists(os.path.splitext(p)[0] + ".json"):
            return p
    except Exception:
        pass
    return None


class SelectEnsembleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rs_ens_")
        self.ens = os.path.join(self.tmp, "ensemble")
        os.makedirs(self.ens)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _member(self, name, noise, seed, yv, yt, cpu_ms=10.0, onnx=None):
        rng = np.random.default_rng(seed)

        def logits(y):
            z = rng.normal(0, noise, (len(y), 4))
            z[np.arange(len(y)), y] += 1.5
            return z.astype(np.float32)
        np.savez(os.path.join(self.ens, f"{name}_logits.npz"), val_logits=logits(yv), val_y=yv, test_logits=logits(yt),
                 test_y=yt, val_groups=np.arange(len(yv)).astype(str), test_groups=np.arange(len(yt)).astype(str))
        with open(os.path.join(self.ens, f"deep_vision_{name}.json"), "w") as fh:
            json.dump({"arch": name, "img_size": 224, "tta_flip": True, "resize_ratio": 1.15,
                       "class_names": ["a", "b", "c", "d"], "onnx": {"onnx_cpu_ms_per_image": cpu_ms}}, fh)
        if onnx:
            shutil.copyfile(onnx, os.path.join(self.ens, f"deep_vision_{name}.onnx"))
        else:
            open(os.path.join(self.ens, f"deep_vision_{name}.onnx"), "wb").close()

    def test_independent_members_make_a_better_ensemble_and_a_cheaper_cascade(self):
        from training import select_ensemble as se
        rng = np.random.default_rng(0)
        yv, yt = rng.integers(0, 4, 600), rng.integers(0, 4, 600)
        for i, (n, ms) in enumerate((("cnn", 8.0), ("vit", 20.0), ("levit", 12.0))):
            self._member(n, 1.6, 10 + i, yv, yt, cpu_ms=ms)
        rep = se.main(["--dir", self.ens, "--out", self.tmp])
        self.assertEqual(rep["decision"], "serve the cascade", rep["best_ensemble"])
        self.assertGreaterEqual(rep["cascade"]["validation_gain_over_best_single"], se.MIN_GAIN, "the served cascade clears the margin")
        self.assertEqual(rep["cascade"]["first"], "cnn", "the fastest member answers first")
        self.assertLess(rep["cascade"]["validation_escalation_rate"], 1.0)
        self.assertLess(rep["cascade"]["expected_cpu_ms"], rep["best_ensemble"]["cpu_ms"])
        lo, hi = rep["test_gain_of_ensemble_over_single"]["ci95"]
        self.assertLessEqual(lo, hi)
        cfg = json.load(open(os.path.join(self.tmp, "vision_ensemble.json")))
        self.assertEqual(sorted(cfg["members"]), sorted(rep["best_ensemble"]["members"]))
        for m in cfg["members"]:
            self.assertTrue(os.path.exists(os.path.join(self.tmp, f"deep_vision_ens_{m}.onnx")))

    def test_identical_members_do_not_get_served(self):
        from training import select_ensemble as se
        rng = np.random.default_rng(1)
        yv, yt = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
        self._member("a", 1.0, 5, yv, yt)
        self._member("b", 1.0, 5, yv, yt)              # same seed: the same network twice
        _write = os.path.join(self.tmp, "vision_ensemble.json")
        open(_write, "w").write("{}")
        rep = se.main(["--dir", self.ens, "--out", self.tmp])
        self.assertTrue(rep["decision"].startswith("keep the single network"))
        self.assertFalse(os.path.exists(_write), "a stale ensemble selection is removed")

    def test_members_scored_on_different_splits_are_refused(self):
        from training import select_ensemble as se
        rng = np.random.default_rng(2)
        yv, yt = rng.integers(0, 4, 100), rng.integers(0, 4, 100)
        self._member("a", 1.0, 1, yv, yt)
        self._member("b", 1.0, 2, np.roll(yv, 1), yt)
        with self.assertRaises(SystemExit):
            se.main(["--dir", self.ens, "--out", self.tmp])


class CascadeClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.onnx = _served_onnx()
        if not cls.onnx:
            raise unittest.SkipTest("no fine-tuned network on disk")
        cls.tmp = tempfile.mkdtemp(prefix="rs_casc_")
        side = os.path.splitext(cls.onnx)[0] + ".json"
        for n in ("fast", "slow"):
            shutil.copyfile(cls.onnx, os.path.join(cls.tmp, f"deep_vision_ens_{n}.onnx"))
            shutil.copyfile(side, os.path.join(cls.tmp, f"deep_vision_ens_{n}.json"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _cfg(self, tau):
        with open(os.path.join(self.tmp, "vision_ensemble.json"), "w") as fh:
            json.dump({"served": True, "members": ["fast", "slow"], "first": "fast", "tau": tau,
                       "temperatures": {"fast": 1.0, "slow": 2.0}}, fh)

    def test_routes_by_confidence_and_matches_the_single_network(self):
        from models.deep_vision_net import DeepVisionNet
        from models.ensemble_classifier import EnsembleClassifier
        img = (np.random.default_rng(0).uniform(0, 255, (240, 320, 3))).astype(np.uint8)
        single = DeepVisionNet(checkpoints_dir=CKPT).predict_probabilities(img)
        self._cfg(0.0)
        e = EnsembleClassifier(self.tmp)
        self.assertTrue(e.is_ready, e.load_error)
        p = e.predict_probabilities(img)
        self.assertFalse(e.last_route["escalated"])
        np.testing.assert_allclose(p, single, atol=1e-5)
        self._cfg(1.01)
        e = EnsembleClassifier(self.tmp)
        p2 = e.predict_probabilities(img)
        self.assertTrue(e.last_route["escalated"])
        self.assertEqual(e.last_route["networks"], ["fast", "slow"])
        self.assertAlmostEqual(float(p2.sum()), 1.0, places=5)
        self.assertEqual(int(p2.argmax()), int(single.argmax()), "temperature changes confidence, not the winner")
        self.assertIn("cascade", e.predict_image(img))

    def test_a_missing_member_falls_back(self):
        from models.ensemble_classifier import EnsembleClassifier
        with open(os.path.join(self.tmp, "vision_ensemble.json"), "w") as fh:
            json.dump({"members": ["fast", "nope"]}, fh)
        e = EnsembleClassifier(self.tmp)
        self.assertFalse(e.is_ready)
        self.assertIn("nope", e.load_error)


if __name__ == "__main__":
    unittest.main()


class TransformerDetectorTest(unittest.TestCase):
    def test_rtdetr_normalised_boxes_are_scaled_and_yolo_pixels_are_not(self):
        from models.onnx_object_detector import decode_output
        q = np.zeros((1, 300, 8), np.float32)                 # RT-DETR: (1, queries, 4 + 4 classes), boxes in 0-1
        q[0, 0, :4] = [0.5, 0.25, 0.1, 0.2]
        q[0, 0, 4 + 3] = 0.9
        boxes, conf, cls = decode_output(q, 4, 0.5, input_size=640)
        np.testing.assert_allclose(boxes[0], [320, 160, 64, 128])
        self.assertEqual((int(cls[0]), round(float(conf[0]), 2)), (3, 0.9))
        y = np.zeros((1, 8, 8400), np.float32)                # YOLOv8: (1, 4 + classes, anchors), boxes in pixels
        y[0, :4, 0] = [320, 160, 64, 128]
        y[0, 4 + 1, 0] = 0.8
        boxes, _c, _k = decode_output(y, 4, 0.5, input_size=640)
        np.testing.assert_allclose(boxes[0], [320, 160, 64, 128])

    def test_oversized_candidate_is_reported_not_served(self):
        from scripts.select_rdd_detector import decide
        inc = {"validation": {"map50": 0.3}, "data": {"photographs": {"valid": 100}}, "onnx": {"size_mb": 12}}
        cand = {"validation": {"map50": 0.5}, "data": {"photographs": {"valid": 100}}, "onnx": {"size_mb": 130}}
        self.assertEqual(decide(cand, inc)[0], "incumbent")
        cand["onnx"]["size_mb"] = 60
        self.assertEqual(decide(cand, inc)[0], "candidate")
