"""
The U-Net segmenter and the RDD2022 road-damage detector: selection rules,
fallbacks, and the shared post-processing. None of this needs PyTorch or the
trained weights, so it runs everywhere.
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np


def _write(d, name, obj):
    with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


class SharedMaskPostProcessing(unittest.TestCase):
    def test_pixel_classifier_and_unet_share_one_mask_rule(self):
        from models.defect_segmenter import WORK_H, WORK_W, mask_from_proba
        n = WORK_H * WORK_W
        p = np.zeros((n, 3), dtype=np.float32)
        p[:, 0] = 0.9
        p[:, 1] = 0.05
        p[:, 2] = 0.05
        p[1000:4000, 1] = 0.9           # a crack blob
        p[1000:4000, 0] = 0.05
        out = mask_from_proba(p, (WORK_H, WORK_W), (400, 640), {"crack": 0.5, "pothole": 0.5})
        self.assertEqual(out["mask"].shape, (400, 640))
        self.assertEqual(out["mask_work"].shape, (WORK_H, WORK_W))
        self.assertEqual(out["crack_px"], 3000)
        self.assertEqual(out["pothole_px"], 0)
        self.assertAlmostEqual(out["crack_mean_confidence"], 0.9, places=4)

    def test_specks_below_the_size_floor_are_dropped(self):
        from models.defect_segmenter import WORK_H, WORK_W, mask_from_proba
        p = np.zeros((WORK_H * WORK_W, 3), dtype=np.float32)
        p[:, 0] = 1.0
        p[500, :] = [0.0, 0.0, 1.0]     # one isolated pothole pixel
        out = mask_from_proba(p, (WORK_H, WORK_W), (200, 320), {"crack": 0.5, "pothole": 0.5})
        self.assertEqual(out["pothole_px"], 0)


def _fake_unet(tta=True):
    """_UNetBase with a deterministic forward pass: pothole probability rises left to right."""
    from models.unet_segmenter import IN_H, IN_W, _UNetBase

    class F(_UNetBase):
        is_ready = True

        def _forward(self, x):
            ramp = np.linspace(0, 1, IN_W, dtype=np.float32)[None, :].repeat(IN_H, 0)
            pot = ramp
            snd = 1 - ramp
            return np.stack([snd, np.zeros_like(ramp), pot])

    f = F()
    f.tta_flip = tta
    f.thresholds = {"crack": 0.5, "pothole": 0.6}
    return f


class UNetServingInterface(unittest.TestCase):
    def test_segment_returns_the_pixel_classifier_dictionary(self):
        img = (np.random.default_rng(0).random((360, 600, 3)) * 255).astype(np.uint8)
        out = _fake_unet(tta=False).segment(img)
        for k in ("mask", "mask_work", "proba_crack", "proba_pothole", "crack_px", "pothole_px",
                  "pothole_px_full", "defect_fraction", "scale_x", "scale_y"):
            self.assertIn(k, out)
        self.assertEqual(out["mask"].shape, (360, 600))
        self.assertGreater(out["pothole_px"], 0)        # right side of the frame is above 0.6

    def test_flip_tta_averages_the_mirrored_prediction(self):
        img = np.zeros((320, 512, 3), dtype=np.uint8)
        p = _fake_unet(tta=True).predict_work(img)
        # a left-right ramp averaged with its mirror is flat at 0.5
        self.assertTrue(np.allclose(p[:, 2], 0.5, atol=0.02))
        out = _fake_unet(tta=True).segment(img)
        self.assertEqual(out["pothole_px"], 0)          # 0.5 < 0.6 threshold everywhere


class SegmenterSelection(unittest.TestCase):
    def setUp(self):
        from training.train_unet_segmenter import decide_segmenter
        self.decide = decide_segmenter

    @staticmethod
    def v(c, p, fp=0.1):
        return {"crack": {"iou": c}, "pothole": {"iou": p}, "clean_false_blob_rate": fp}

    def test_unet_must_win_both_classes(self):
        self.assertEqual(self.decide(self.v(0.4, 0.3), self.v(0.2, 0.1))[0], "unet")
        self.assertEqual(self.decide(self.v(0.4, 0.09), self.v(0.2, 0.1))[0], "pixel_classifier")
        self.assertEqual(self.decide(self.v(0.19, 0.3), self.v(0.2, 0.1))[0], "pixel_classifier")

    def test_more_false_blobs_on_clean_roads_vetoes_it(self):
        self.assertEqual(self.decide(self.v(0.4, 0.3, fp=0.40), self.v(0.2, 0.1, fp=0.20))[0], "pixel_classifier")
        self.assertEqual(self.decide(self.v(0.4, 0.3, fp=0.24), self.v(0.2, 0.1, fp=0.20))[0], "unet")

    def test_smoke_runs_and_missing_baseline_never_serve(self):
        self.assertEqual(self.decide(self.v(0.9, 0.9), self.v(0.1, 0.1), smoke=True)[0], "pixel_classifier")
        self.assertEqual(self.decide(self.v(0.9, 0.9), None)[0], "pixel_classifier")


class SegmenterLoaderFallsBack(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_no_selection_file_means_pixel_classifier(self):
        from models.defect_segmenter import DefectSegmenter
        from models.unet_segmenter import load_best_segmenter
        self.assertIsInstance(load_best_segmenter(self.d), DefectSegmenter)

    def test_selection_says_unet_but_no_model_falls_back(self):
        from models.defect_segmenter import DefectSegmenter
        from models.unet_segmenter import load_best_segmenter
        _write(self.d, "segmenter_selection.json", {"served": "unet"})
        self.assertIsInstance(load_best_segmenter(self.d), DefectSegmenter)

    def test_served_summary_follows_the_selection(self):
        from models.served_report import served_segmenter_summary
        _write(self.d, "defect_segmenter_report.json", {"iou": {"crack": {"iou": 0.23}}})
        _write(self.d, "defect_segmenter_unet.json", {"iou": {"crack": {"iou": 0.41}}})
        _write(self.d, "segmenter_selection.json", {"served": "unet", "rule": "r", "why": "w"})
        # onnx missing -> the pixel classifier is what actually serves
        self.assertEqual(served_segmenter_summary(self.d)["kind"], "pixel_classifier")
        open(os.path.join(self.d, "defect_segmenter_unet.onnx"), "wb").close()
        s = served_segmenter_summary(self.d)
        self.assertEqual(s["kind"], "unet")
        self.assertEqual(s["iou"]["crack"]["iou"], 0.41)


class RoadDamageDetector(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_absent_model_is_not_ready_and_not_guessed(self):
        from models.road_damage_detector import RoadDamageDetector as R
        r = R(checkpoints_dir=self.d)
        self.assertFalse(r.is_ready)
        self.assertIsNone(r.weights_path)

    def test_coco_detector_never_picks_up_the_damage_model(self):
        """The COCO loader globs *detector*.onnx; the damage model must not match it."""
        from models.onnx_object_detector import ONNXObjectDetector
        open(os.path.join(self.d, "damage_rdd2022_india.onnx"), "wb").close()
        det = ONNXObjectDetector(checkpoints_dir=self.d)
        self.assertFalse(det.is_ready)
        self.assertIsNone(det.backend)

    def test_yolov8_four_class_output_decodes(self):
        from models.onnx_object_detector import decode_output
        raw = np.zeros((1, 8, 8400), dtype=np.float32)
        raw[0, :4, 7] = [320, 320, 100, 50]
        raw[0, 4 + 3, 7] = 0.9                      # D40 pothole
        boxes, conf, cls = decode_output(raw, 4, 0.25)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(int(cls[0]), 3)
        self.assertAlmostEqual(float(conf[0]), 0.9, places=5)

    def test_pipeline_reports_damage_boxes_as_unavailable_without_a_model(self):
        from pipeline.deep_inference_pipeline import DeepInferencePipeline
        p = DeepInferencePipeline.__new__(DeepInferencePipeline)
        self.assertFalse(getattr(p, "damage_detector", None) is not None and p.damage_detector.is_ready)


class ClaimsOnlyForTrainedModels(unittest.TestCase):
    def test_no_reports_no_new_claims(self):
        import scripts.build_claims as bc
        d = tempfile.mkdtemp()
        old = bc.CKPT
        try:
            bc.CKPT = d
            claims = {"subsystems": [{"id": "M_SEG", "measured": {}}], "corrections": []}
            bc._deep_extras(claims, lambda name: {})
            self.assertEqual([s["id"] for s in claims["subsystems"]], ["M_SEG"])
            self.assertNotIn("deep_alternative_tried", claims["subsystems"][0])
        finally:
            bc.CKPT = old
            shutil.rmtree(d, ignore_errors=True)

    def test_unet_that_lost_is_recorded_not_served(self):
        import scripts.build_claims as bc
        d = tempfile.mkdtemp()
        old = bc.CKPT
        reps = {"segmenter_selection.json": {"served": "pixel_classifier", "why": "lost on pothole",
                                             "test": {"unet": {"crack": {"iou": 0.3}},
                                                      "pixel_classifier": {"crack": {"iou": 0.23}}}},
                "defect_segmenter_unet.json": {"iou": {}}}
        try:
            bc.CKPT = d
            claims = {"subsystems": [{"id": "M_SEG", "measured": {"crack_iou": 0.23}}], "corrections": []}
            bc._deep_extras(claims, lambda name: reps.get(name, {}))
            seg = claims["subsystems"][0]
            self.assertIn("did NOT", seg["deep_alternative_tried"])
            self.assertEqual(seg["measured"]["crack_iou"], 0.23)
        finally:
            bc.CKPT = old
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
