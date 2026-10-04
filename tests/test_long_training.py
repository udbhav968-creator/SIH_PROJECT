"""Long runs across Colab sessions, and training on the user's own outlined photographs."""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np


class _NoLeak:
    ref = np.array([], dtype=np.uint64)
    threshold = 6

    def is_leak(self, img):
        return False


def _road(h=240, w=400, seed=0):
    """A distinct, smoothly varying picture per seed (pure noise would hash alike after downsampling)."""
    import cv2
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (6, 10, 3)).astype(np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


class ResumeRules(unittest.TestCase):
    def _args(self, **kw):
        from types import SimpleNamespace
        base = dict(encoder="resnet34", min_crop_scale=0.35, samples_per_epoch=6000, batch=8, lr=3e-4,
                    epochs=80, seed=42)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_same_setup_resumes(self):
        from training.train_unet_multi import check_resume_config, resume_config
        a = self._args()
        self.assertIsNone(check_resume_config(resume_config(a), resume_config(a)))

    def test_a_different_network_or_batch_refuses_to_resume(self):
        from training.train_unet_multi import check_resume_config, resume_config
        saved = resume_config(self._args())
        self.assertIn("encoder", check_resume_config(saved, resume_config(self._args(encoder="resnet18"))))
        self.assertIn("batch", check_resume_config(saved, resume_config(self._args(batch=16))))

    def test_time_budget_leaves_room_for_one_epoch_and_the_finish(self):
        from training.train_unet_multi import out_of_time
        h = 3600
        self.assertFalse(out_of_time(1 * h, 120, 0, 25))                    # no budget: never pauses
        self.assertFalse(out_of_time(1 * h, 150, 3.5, 25))                  # plenty of time
        # 3 h in, 150 s epochs: the next epoch fits, but not if it could also be the last one (25 min to finish)
        self.assertFalse(out_of_time(3 * h, 150, 3.5, 25, may_finish=False))
        self.assertTrue(out_of_time(3 * h, 150, 3.5, 25, may_finish=True))
        self.assertTrue(out_of_time(3.48 * h, 60, 3.5, 0, may_finish=False))  # 72 s left; the epoch + save needs 198 s

    def test_changing_epochs_or_learning_rate_refuses_to_resume(self):
        from training.train_unet_multi import check_resume_config, resume_config
        saved = resume_config(self._args())
        self.assertIn("epochs", check_resume_config(saved, resume_config(self._args(epochs=120))))
        self.assertIn("lr", check_resume_config(saved, resume_config(self._args(lr=1e-4))))
        self.assertIsNone(check_resume_config(saved, resume_config(self._args(seed=7))))

    def test_manifest_entries_without_files_are_skipped(self):
        import training.train_unet_multi as tum
        tmp = tempfile.mkdtemp()
        try:
            for sub in ("img", "lab"):
                os.makedirs(os.path.join(tmp, "src", sub))
            open(os.path.join(tmp, "src", "img", "a.jpg"), "wb").close()
            open(os.path.join(tmp, "src", "lab", "a.png"), "wb").close()
            with open(os.path.join(tmp, "manifest.json"), "w") as fh:
                json.dump({"items": {"src": [{"id": "a", "split": "train"}, {"id": "gone", "split": "test"}],
                                     "stale": [{"id": "x", "split": "train"}]}}, fh)
            with mock.patch.object(tum, "MULTI", tmp):
                man = tum.load_manifest()
            self.assertEqual(man, {"src": {"train": ["a"], "cal": [], "test": []}})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class OwnPhotographs(unittest.TestCase):
    def setUp(self):
        import scripts.fetch_seg_datasets as fsd
        self.fsd = fsd
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "seg_multi")
        self.patch = mock.patch.object(fsd, "OUT", self.out)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roboflow_names_and_spots(self):
        self.assertEqual(self.fsd.spot_of("IMG_1234_jpg.rf.abc123DEF"), "IMG_1234")
        self.assertEqual(self.fsd.spot_of("spot12__a_jpg.rf.9f8e"), "spot12")
        self.assertEqual(self.fsd.spot_of("spot12__b"), "spot12")
        self.assertEqual(self.fsd.spot_of("road7"), "road7")

    def test_class_names(self):
        self.assertEqual(self.fsd.class_of("Pothole"), 2)
        self.assertEqual(self.fsd.class_of("water-filled pothole"), 2)
        self.assertEqual(self.fsd.class_of("alligator crack"), 1)
        self.assertIsNone(self.fsd.class_of("manhole"))

    def test_pothole_is_painted_over_a_crack(self):
        lab = self.fsd.draw_shapes((100, 100), [
            (2, "polygon", [[20, 20], [60, 20], [60, 60], [20, 60]]),
            (1, "linestrip", [[0, 40], [99, 40]]),
        ])
        self.assertEqual(lab[40, 40], 2)          # inside the pothole, even where the crack line crosses it
        self.assertEqual(lab[40, 90], 1)          # the crack outside the pothole
        self.assertEqual(lab[90, 90], 0)

    def _write_coco(self, root, names):
        import cv2
        os.makedirs(root, exist_ok=True)
        images, anns = [], []
        for i, n in enumerate(names):
            cv2.imwrite(os.path.join(root, n), _road(seed=i))
            images.append({"id": i, "file_name": n, "width": 400, "height": 240})
            anns.append({"id": 10 * i, "image_id": i, "category_id": 1, "iscrowd": 0,
                         "segmentation": [[100, 60, 220, 60, 220, 160, 100, 160]], "bbox": [100, 60, 120, 100]})
            anns.append({"id": 10 * i + 1, "image_id": i, "category_id": 2, "iscrowd": 0,
                         "segmentation": [[10, 200, 390, 205, 390, 210, 10, 206]], "bbox": [10, 200, 380, 10]})
        coco = {"images": images, "annotations": anns,
                "categories": [{"id": 0, "name": "road-damage"}, {"id": 1, "name": "pothole"},
                               {"id": 2, "name": "crack"}]}
        with open(os.path.join(root, "_annotations.coco.json"), "w") as fh:
            json.dump(coco, fh)

    def test_coco_export_becomes_labelled_pairs_split_by_spot(self):
        import cv2
        src = os.path.join(self.tmp, "export")
        names = [f"spot{k}__{v}_jpg.rf.{k}{v}ab.jpg" for k in range(12) for v in ("a", "b")]
        self._write_coco(os.path.join(src, "train"), names)
        s = self.fsd.fetch_own_photos(_NoLeak(), src)
        self.assertEqual(s["format"], "coco")
        self.assertEqual(s["kept"], 24)
        self.assertEqual(s["spots"], 12)
        lab = cv2.imread(os.path.join(self.out, "own_india", "lab", s and os.listdir(
            os.path.join(self.out, "own_india", "lab"))[0]), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(set(np.unique(lab).tolist()), {0, 1, 2})
        # both photographs of a spot always land in the same split, also when the manifest is rebuilt
        with open(os.path.join(self.out, "own_india", "groups.json")) as fh:
            groups = json.load(fh)
        by_spot = {}
        for ident, spot in groups.items():
            by_spot.setdefault(spot, set()).add(self.fsd.split_of("own_india", spot))
        self.assertTrue(all(len(v) == 1 for v in by_spot.values()))

    def test_manifest_rebuild_uses_the_spot_split(self):
        src = os.path.join(self.tmp, "export")
        self._write_coco(src, [f"spot{k}__{v}.jpg" for k in range(10) for v in ("a", "b", "c")])
        with mock.patch.object(self.fsd, "LeakGuard", lambda: _NoLeak()):
            self.fsd.main(["--only", "own_india", "--own-photos", src])
        with open(os.path.join(self.out, "manifest.json")) as fh:
            man = json.load(fh)
        items = man["items"]["own_india"]
        self.assertEqual(len(items), 30)
        per_spot = {}
        for it in items:
            per_spot.setdefault(it["id"].split("__")[0], set()).add(it["split"])
        self.assertTrue(all(len(v) == 1 for v in per_spot.values()))

    def test_labelme_and_mask_folders(self):
        import cv2
        lm = os.path.join(self.tmp, "labelme")
        os.makedirs(lm)
        cv2.imwrite(os.path.join(lm, "r1.jpg"), _road())
        with open(os.path.join(lm, "r1.json"), "w") as fh:
            json.dump({"imagePath": "r1.jpg", "shapes": [
                {"label": "pothole", "shape_type": "polygon", "points": [[50, 50], [150, 50], [150, 120], [50, 120]]}]}, fh)
        self.assertEqual(self.fsd.fetch_own_photos(_NoLeak(), lm)["kept"], 1)

        mf = os.path.join(self.tmp, "maskdirs")
        os.makedirs(os.path.join(mf, "images"))
        os.makedirs(os.path.join(mf, "masks"))
        cv2.imwrite(os.path.join(mf, "images", "r2.jpg"), _road())
        m = np.zeros((240, 400), np.uint8)
        m[50:100, 50:150] = 2
        m[150:155, :] = 1
        cv2.imwrite(os.path.join(mf, "masks", "r2.png"), m)
        binary = (m > 0).astype(np.uint8) * 255                 # 0/255: crack or pothole cannot be told apart
        cv2.imwrite(os.path.join(mf, "images", "r3.jpg"), _road(seed=3))
        cv2.imwrite(os.path.join(mf, "masks", "r3.png"), binary)
        s = self.fsd.fetch_own_photos(_NoLeak(), mf)
        self.assertEqual(s["format"], "mask_folders")
        self.assertEqual(s["kept"], 1)
        self.assertEqual(s["detail"]["masks_with_unknown_classes"], 1)

    def test_cvat_layout_with_images_beside_annotations(self):
        import cv2
        src = os.path.join(self.tmp, "cvat")
        self._write_coco(os.path.join(src, "images"), ["r1.jpg", "r2.jpg"])
        os.makedirs(os.path.join(src, "annotations"))
        shutil.move(os.path.join(src, "images", "_annotations.coco.json"),
                    os.path.join(src, "annotations", "instances_default.json"))
        self.assertEqual(self.fsd.fetch_own_photos(_NoLeak(), src)["kept"], 2)

    def test_burst_shots_of_one_spot_share_a_split(self):
        import cv2
        src = os.path.join(self.tmp, "burst")
        names = [f"IMG_{1000 + i}.jpg" for i in range(8)]
        self._write_coco(src, names)
        base = _road(seed=99)
        for i, n in enumerate(names[:4]):                   # four near-identical frames of one spot
            frame = np.clip(base.astype(int) + i, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(src, n), frame)
        s = self.fsd.fetch_own_photos(_NoLeak(), src)
        self.assertEqual(s["kept"], 8)
        self.assertGreaterEqual(s["spots_merged_as_near_duplicates"], 3)
        with open(os.path.join(self.out, "own_india", "groups.json")) as fh:
            groups = json.load(fh)
        burst = {groups[k] for k in groups if k.startswith(("IMG_1000", "IMG_1001", "IMG_1002", "IMG_1003"))}
        self.assertEqual(len(burst), 1)

    def test_missing_path_is_skipped_not_an_error(self):
        self.assertIn("skipped", self.fsd.fetch_own_photos(_NoLeak(), os.path.join(self.tmp, "nope")))


class ColabDriver(unittest.TestCase):
    def test_week_script_parses_and_resumes_everything_from_drive(self):
        import ast
        path = os.path.join(os.path.dirname(__file__), "..", "scripts", "colab_week_training.py")
        src = open(path, encoding="utf-8").read()
        ast.parse(src)
        for must in ("--ckpt-dir", "--time-budget-hours", "unet_done.json", "yolo_done.json", "rdd2022_india.tar",
                     "seg_multi.tar", "data_signature.json", "RESUMED", "verify_rdd_detector --artefact"):
            self.assertIn(must, src)


if __name__ == "__main__":
    unittest.main()
