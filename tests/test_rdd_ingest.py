"""RDD2022 India ingestion: class mapping, and the guarantee that the Indian
test split never reaches a training folder."""
import glob, os, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts.ingest_rdd2022_india as ing
from training.train_deep_vision import group_key


class RDDIngestTest(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, "rdd"); ds = os.path.join(self.tmp, "datasets")
        for split in ("train", "valid", "test"):
            os.makedirs(os.path.join(self.root, "images", split)); os.makedirs(os.path.join(self.root, "labels", split))
            for i in range(3):
                stem = f"India_{split}_{i}"
                Image.fromarray((np.random.RandomState(i).rand(200, 300, 3) * 255).astype(np.uint8)).save(
                    os.path.join(self.root, "images", split, stem + ".jpg"))
                lines = {0: "0 0.5 0.5 0.3 0.3\n3 0.2 0.7 0.2 0.2\n", 1: "", 2: "4 0.5 0.5 0.4 0.4\n"}[i]
                open(os.path.join(self.root, "labels", split, stem + ".txt"), "w").write(lines)
        open(os.path.join(self.root, "data.yaml"), "w").write("nc: 5\nnames: ['D00', 'D10', 'D20', 'D40', 'Repair']\n")
        self._saved = (ing.DATASETS, ing.EVAL_DIR)
        ing.DATASETS = ds; ing.EVAL_DIR = os.path.join(ds, "_eval_rdd2022_india")
        for cls in (0, 1, 2):
            os.makedirs(ing.train_dir(cls))

    def tearDown(self):
        ing.DATASETS, ing.EVAL_DIR = self._saved

    def test_mapping(self):
        self.assertEqual(ing.class_for("D40"), ing.POTHOLE)
        self.assertEqual(ing.class_for("pothole"), ing.POTHOLE)
        for n in ("D00", "D10", "D20", "longitudinal crack", "Alligator Crack"):
            self.assertEqual(ing.class_for(n), ing.CRACK, n)
        for n in ("Repair", "D43", "D50", "white line blur"):
            self.assertIsNone(ing.class_for(n), n)

    def test_yaml_dict_form(self):
        open(os.path.join(self.root, "data.yaml"), "w").write("names:\n  0: D00\n  1: D40\n")
        self.assertEqual(ing.read_names(self.root), ["D00", "D40"])

    def test_test_split_never_reaches_training(self):
        m = ing.main(["--root", self.root])
        train_files = [f for c in (0, 1, 2) for f in glob.glob(os.path.join(ing.train_dir(c), "*.jpg"))]
        self.assertTrue(train_files)
        self.assertFalse([f for f in train_files if "_test_" in f], "Indian test photographs leaked into training")
        eval_files = glob.glob(os.path.join(ing.EVAL_DIR, "*", "*.jpg"))
        self.assertTrue(eval_files and all("_test_" in f for f in eval_files))
        # per split: photo 0 -> 1 crack + 1 pothole, photo 1 -> normal band, photo 2 -> nothing ('Repair')
        self.assertEqual(m["training_crops"], {"normal": 2, "crack": 2, "pothole": 2})
        self.assertEqual(m["eval_crops"], {"normal": 1, "crack": 1, "pothole": 1})
        self.assertTrue(all(os.path.basename(f).startswith("rddin_") for f in train_files))

    def test_rerun_replaces_previous_crops(self):
        ing.main(["--root", self.root]); ing.main(["--root", self.root])
        n = sum(len(glob.glob(os.path.join(ing.train_dir(c), "*.jpg"))) for c in (0, 1, 2))
        self.assertEqual(n, 6)

    def test_crops_of_one_photo_share_a_group(self):
        self.assertEqual(group_key("rddin_India_000123_0.jpg"), group_key("rddin_India_000123_7.jpg"))
        self.assertNotEqual(group_key("rddin_India_000123_0.jpg"), group_key("rddin_India_000124_0.jpg"))


if __name__ == "__main__":
    unittest.main()
