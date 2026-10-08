"""Training on more data: the multi-country detector set and the rules that decide which model is served."""

import glob
import json
import os
import shutil
import tempfile
import unittest
import zipfile


def _xml(w, h, objs):
    body = "".join(f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax>"
                   f"<ymax>{d}</ymax></bndbox></object>" for n, a, b, c, d in objs)
    return f"<annotation><size><width>{w}</width><height>{h}</height></size>{body}</annotation>"


def _write(path, data, mode="w"):
    with open(path, mode) as fh:
        fh.write(data)


def _jpg(path, w=64, h=48):
    from PIL import Image
    Image.new("RGB", (w, h), (90, 90, 90)).save(path)


class DetectorSelection(unittest.TestCase):
    def _rep(self, valid, test=0.1, photos=150):
        return {"validation": {"map50": valid}, "test": {"map50": test},
                "data": {"photographs": {"valid_india": photos}}}

    def test_better_india_validation_wins(self):
        from scripts.select_rdd_detector import decide
        self.assertEqual(decide(self._rep(0.60), self._rep(0.55))[0], "candidate")

    def test_equal_or_worse_keeps_served(self):
        from scripts.select_rdd_detector import decide
        self.assertEqual(decide(self._rep(0.55), self._rep(0.55))[0], "incumbent")
        self.assertEqual(decide(self._rep(0.50), self._rep(0.55))[0], "incumbent")

    def test_test_numbers_never_decide(self):
        from scripts.select_rdd_detector import decide
        self.assertEqual(decide(self._rep(0.50, test=0.99), self._rep(0.55, test=0.01))[0], "incumbent")

    def test_different_validation_photos_not_comparable(self):
        from scripts.select_rdd_detector import decide
        winner, why = decide(self._rep(0.9, photos=100), self._rep(0.5, photos=150))
        self.assertEqual(winner, "incumbent")
        self.assertIn("not comparable", why)

    def test_missing_reports(self):
        from scripts.select_rdd_detector import decide
        self.assertEqual(decide(None, self._rep(0.5))[0], "incumbent")
        self.assertEqual(decide(self._rep(0.5), None)[0], "candidate")

    def test_install_on_win(self):
        from scripts import select_rdd_detector as s
        d = tempfile.mkdtemp()
        try:
            _write(os.path.join(d, "damage_rdd2022_world.onnx"), b"WORLD", "wb")
            _write(os.path.join(d, "damage_rdd2022_india.onnx"), b"INDIA", "wb")
            _write(os.path.join(d, "damage_rdd2022_world.json"),
                   json.dumps({"names": ["D00"], "artefact_check": {"ok": True}}))
            _write(os.path.join(d, "road_damage_detector_world_report.json"),
                   json.dumps(dict(self._rep(0.6), tag="world", artefact_check={"ok": True})))
            _write(os.path.join(d, "road_damage_detector_report.json"), json.dumps(dict(self._rep(0.5), tag="india")))
            rec = s.main(ckpt=d)
            self.assertEqual(rec["served"], "world")
            with open(os.path.join(d, "damage_rdd2022_india.onnx"), "rb") as fh:
                self.assertEqual(fh.read(), b"WORLD")
            with open(os.path.join(d, "damage_rdd2022_india.json")) as fh:
                meta = json.load(fh)
            self.assertNotIn("artefact_check", meta)          # a new file must pass its own checks
            with open(os.path.join(d, "road_damage_detector_report.json")) as fh:
                rep = json.load(fh)
            self.assertEqual(rep["previous_model"]["tag"], "india")
            self.assertNotIn("artefact_check", rep)
        finally:
            shutil.rmtree(d)

    def test_names_for(self):
        from training.train_rdd_detector import names_for
        self.assertEqual(names_for("india")[0], "damage_rdd2022_india.onnx")
        self.assertEqual(names_for("world"), ("damage_rdd2022_world.onnx", "damage_rdd2022_world.json",
                                              "road_damage_detector_world_report.json"))


class WorldDataset(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        # a tiny India split, as prepare_rdd2022_voc writes it
        self.india = os.path.join(self.d, "india")
        for split, stems in (("train", ["India_000001", "India_000002"]), ("valid", ["India_000003"]),
                             ("test", ["India_000004"])):
            os.makedirs(os.path.join(self.india, "images", split))
            os.makedirs(os.path.join(self.india, "labels", split))
            for s in stems:
                _jpg(os.path.join(self.india, "images", split, s + ".jpg"))
                _write(os.path.join(self.india, "labels", split, s + ".txt"), "0 0.5 0.5 0.2 0.2")
        # a full-release zip holding a nested Japan archive
        inner = os.path.join(self.d, "Japan.zip")
        with zipfile.ZipFile(inner, "w") as z:
            for i in range(20):
                img = os.path.join(self.d, "tmp.jpg")
                _jpg(img, 64, 48)
                z.write(img, f"Japan/train/images/Japan_{i:06d}.jpg")
                objs = [("D40", 10, 10, 30, 30)] if i % 3 else []
                if i == 19:
                    objs = [("D43", 1, 1, 20, 20)]       # only an unkept class: skipped
                z.writestr(f"Japan/train/annotations/xmls/Japan_{i:06d}.xml", _xml(64, 48, objs))
        self.full = os.path.join(self.d, "RDD2022_all.zip")
        with zipfile.ZipFile(self.full, "w") as z:
            z.write(inner, "RDD2022_all_countries/Japan.zip")

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_build_keeps_india_split_and_tests_india_only(self):
        from scripts import prepare_rdd2022_world as w
        raw, out = os.path.join(self.d, "raw"), os.path.join(self.d, "out")
        self.assertEqual(w.extract_countries(self.full, ["Japan", "Czech"], dest=raw), ["Japan"])
        self.assertIsNotNone(w.country_root("Japan", raw))
        self.assertIsNone(w.country_root("Czech", raw))
        man = w.build(["Japan", "Czech"], india=self.india, out=out, raw=raw)

        def stems(split):
            return sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(out, "images", split)))
        self.assertEqual(stems("test"), ["India_000004"])
        self.assertEqual(stems("valid_india"), ["India_000003"])
        self.assertIn("India_000003", stems("valid"))
        japan = [s for s in stems("train") + stems("valid") if s.startswith("Japan")]
        self.assertEqual(len(japan), 19)                 # Japan_000019 had only an unkept class
        self.assertFalse(set(stems("train")) & set(stems("valid")))
        self.assertEqual(man["countries"]["Japan"]["valid"], 2)   # round(19 * 0.1)
        for lp in glob.glob(os.path.join(out, "labels", "*", "Japan_*.txt")):
            with open(lp) as fh:
                text = fh.read()
            for ln in text.split("\n"):
                if ln:
                    self.assertEqual(ln.split()[0], "3")   # D40 -> index 3
        self.assertTrue(os.path.exists(os.path.join(out, "data.yaml")))

    def test_split_is_deterministic(self):
        from scripts.prepare_rdd2022_world import split_others
        stems = [f"x{i}" for i in range(100)]
        self.assertEqual(split_others(stems), split_others(list(reversed(stems))))
        self.assertEqual(len(split_others(stems)[0]), 10)

    def test_large_photo_resized(self):
        from PIL import Image
        from scripts.prepare_rdd2022_world import copy_image
        src, dst = os.path.join(self.d, "big.jpg"), os.path.join(self.d, "small.jpg")
        _jpg(src, 3650, 2000)
        copy_image(src, dst)
        with Image.open(dst) as im:
            self.assertEqual(max(im.size), 1280)

    def test_yolo_lines_clip_and_drop_tiny(self):
        from scripts.prepare_rdd2022_world import yolo_lines
        lines = yolo_lines(100, 100, [("D00", -10, 0, 50, 50), ("D10", 5, 5, 6, 6)])
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("0 0.250000 0.250000 0.500000"))


class ClassifierSelection(unittest.TestCase):
    def test_halves_follow_photograph(self):
        from scripts.select_vision_candidate import half_of
        for photo in ("India_000123", "India_009999", "India_000001"):
            halves = {half_of(f"/x/crack/{photo}_{k}.jpg") for k in range(5)}
            self.assertEqual(len(halves), 1)
        both = {half_of(f"India_{i:06d}_0.jpg") for i in range(40)}
        self.assertEqual(both, {"select", "report"})

    def test_metrics(self):
        from scripts.select_vision_candidate import metrics
        m = metrics([0, 1, 2, 2], [0, 1, 2, 1])
        self.assertEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["macro_f1"], round((1 + 2 / 3 + 2 / 3) / 3, 4))
        self.assertEqual(metrics([], [])["images"], 0)

    def test_decide_needs_both_higher(self):
        from scripts.select_vision_candidate import decide
        served = {"images": 100, "accuracy": 0.90, "macro_f1": 0.85}
        self.assertEqual(decide({"images": 100, "accuracy": 0.92, "macro_f1": 0.86}, served)[0], "candidate")
        self.assertEqual(decide({"images": 100, "accuracy": 0.95, "macro_f1": 0.84}, served)[0], "served")
        self.assertEqual(decide({"images": 100, "accuracy": 0.90, "macro_f1": 0.90}, served)[0], "served")
        self.assertEqual(decide({"images": 0}, served)[0], "served")

    def test_world_crops_come_from_every_country(self):
        from scripts.ingest_rdd2022_world import country_of, interleave_countries
        rows = [(f"{c}_{i:06d}", None, None) for c, n in (("China_MotorBike", 50), ("Czech", 30), ("Japan", 40),
                                                         ("United_States", 20)) for i in range(n)]
        out = interleave_countries(rows)
        self.assertEqual(sorted(out), sorted(rows))
        self.assertEqual({country_of(r[0]) for r in out[:4]},
                         {"China_MotorBike", "Czech", "Japan", "United_States"})
        self.assertEqual(out, interleave_countries(list(reversed(rows))))

    def test_world_crops_grouped_by_photograph(self):
        from training.train_deep_vision import group_key
        self.assertEqual(group_key("rddw_Japan_000123_0.jpg"), group_key("rddw_Japan_000123_7.jpg"))
        self.assertNotEqual(group_key("rddw_Japan_000123_0.jpg"), group_key("rddw_Japan_000124_0.jpg"))
        self.assertEqual(group_key("rddin_India_000005_2.jpg"), "rddin_India_000005")


if __name__ == "__main__":
    unittest.main()
