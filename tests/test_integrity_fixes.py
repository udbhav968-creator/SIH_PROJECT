"""
Regression tests for the hard-coded / fabricated values removed in the
October 2026 audit. Each test pins one specific way the system used to invent
a number, a location or a confidence it did not have.

Run alone:  python -m unittest tests.test_integrity_fixes -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)


class WorkOrderHasNoDefaults(unittest.TestCase):
    def setUp(self):
        from models.morth_dispatch_agent import MoRTHDispatchAgent
        self.agent = MoRTHDispatchAgent()

    def test_measured_fields_are_required(self):
        import inspect
        sig = inspect.signature(self.agent.generate_work_order)
        for name in ("area_sqm", "depth_cm", "pci_score"):
            self.assertIs(sig.parameters[name].default, inspect.Parameter.empty,
                          f"{name} has a default - a sealed order could be built from it")
        with self.assertRaises(ValueError):
            self.agent.generate_work_order("NH-1", 12.9, 77.5, "Pothole Cavity", None, 5.0, 40)

    def test_missing_gps_is_held_not_invented(self):
        wo = self.agent.generate_work_order("NH-1", None, None, "Pothole Cavity", 1.0, 5.0, 40)
        self.assertIsNone(wo["coordinates"])
        self.assertEqual(wo["dispatch_status"], "HELD_NO_GPS")
        self.assertTrue(self.agent.verify_work_order_seal(dict(wo)))

    def test_with_gps_is_ready(self):
        wo = self.agent.generate_work_order("NH-1", 12.9, 77.5, "Pothole Cavity", 1.0, 5.0, 40)
        self.assertEqual(wo["coordinates"], {"lat": 12.9, "lon": 77.5})
        self.assertEqual(wo["dispatch_status"], "READY_FOR_DISPATCH")

    def test_out_of_range_inputs_are_refused(self):
        for args in ((95.0, 77.5, 1.0, 5.0, 40), (12.9, 77.5, -1.0, 5.0, 40), (12.9, 77.5, 1.0, 5.0, 140)):
            lat, lon, area, depth, pci = args
            with self.assertRaises(ValueError, msg=str(args)):
                self.agent.generate_work_order("NH-1", lat, lon, "Pothole Cavity", area, depth, pci)

    def test_order_prices_exactly_like_the_audit(self):
        """The work order and the pipeline's IPM engine must use one material table."""
        from models.ipm_homography_engine import IPMHomographyEngine
        wo = self.agent.generate_work_order("NH-1", 12.9, 77.5, "Pothole Cavity", 2.0, 5.0, 30)
        props = IPMHomographyEngine.MATERIAL_PROPERTIES[wo["asphalt_mix"]]
        expect_t = 2.0 * 0.05 * props["density_t_per_m3"] * IPMHomographyEngine.COMPACTION_FACTOR
        self.assertAlmostEqual(wo["required_mass_tonnes"], round(expect_t, 3), places=3)
        self.assertAlmostEqual(wo["allocated_budget_inr"], round(expect_t * props["cost_per_tonne_inr"], 2), delta=0.05)
        ipm = IPMHomographyEngine().compute_asphalt_procurement(2.0, 5.0, mix_type=wo["asphalt_mix"])
        self.assertAlmostEqual(ipm["required_mass_tonnes"], wo["required_mass_tonnes"], places=3)
        self.assertIn("Schedule of Rates", wo["rate_basis"])


class DeduplicationIsClassAware(unittest.TestCase):
    def test_different_defects_at_one_spot_stay_separate(self):
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        e = FleetDeduplicationEngine()
        a = e.ingest_fleet_detection("B1", 12.9716, 77.5946, "Pothole Cavity", 40, 1.0, enrich_location=False)
        b = e.ingest_fleet_detection("B2", 12.97161, 77.59461, "Damaged Traffic Sign", 60, 0.5, enrich_location=False)
        c = e.ingest_fleet_detection("B3", 12.97162, 77.59459, "Pothole Cavity", 42, 1.1, enrich_location=False)
        self.assertEqual(b["action"], "REGISTERED_NEW_DEFECT", "a sign was merged into a pothole 2 m away")
        self.assertEqual(c["action"], "DEDUPLICATED_AND_UPDATED")
        self.assertEqual(c["defect_id"], a["defect_id"])
        self.assertEqual(len(e.defect_registry), 2)

    def test_default_radius_is_8_m(self):
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        self.assertEqual(FleetDeduplicationEngine().proximity_threshold_m, 8.0)

    def test_bad_gps_is_refused(self):
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        with self.assertRaises(ValueError):
            FleetDeduplicationEngine().ingest_fleet_detection("B1", 123.0, 77.0, "Pothole Cavity", 40, 1.0,
                                                               enrich_location=False)


class PipelineInventsNothing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pipeline.deep_inference_pipeline import DeepInferencePipeline
        cls.pipe = DeepInferencePipeline()

    def _photo(self):
        import glob
        hits = sorted(glob.glob(os.path.join(ENGINE_ROOT, "datasets", "02_kaggle_pothole_600", "real_images", "*RAW.jpg")))
        if not hits:
            self.skipTest("no road photographs on disk")
        return hits[0]

    def test_no_gps_means_no_location(self):
        r = self.pipe.audit_image(image_input=self._photo())
        self.assertIsNone(r["location"]["lat"])
        self.assertIsNone(r["location"]["lng"])
        self.assertIsNone(r["location"]["chainage_km"])
        wo = r.get("cryptographic_work_order")
        if wo:
            self.assertIsNone(wo["coordinates"])
            self.assertEqual(wo["dispatch_status"], "HELD_NO_GPS")

    def test_assumed_forecast_inputs_are_declared(self):
        r = self.pipe.audit_image(image_input=self._photo())
        if r.get("status") != "ANALYSIS_COMPLETE":
            self.skipTest("photo rejected at the gate")
        assumed = r["modelling_assumptions"]["assumed_inputs"]
        self.assertEqual(set(assumed), {"traffic_esal_per_day", "seasonal_rain_mm", "pavement_age_years"})
        r2 = self.pipe.audit_image(image_input=self._photo(), traffic_esal=100, rain_mm=10, pavement_age_yr=1,
                                   latitude=12.9, longitude=77.5)
        self.assertEqual(r2["modelling_assumptions"]["assumed_inputs"], {})
        self.assertEqual(r2["location"]["lat"], 12.9)

    def test_rule_decisions_claim_no_confidence(self):
        grey = np.full((480, 640, 3), 128, dtype=np.uint8)  # featureless: rejected by the texture gate
        r = self.pipe.audit_image(image_input=grey)
        self.assertEqual(r["status"], "REJECTED_NON_PAVEMENT")
        self.assertIsNone(r["primary_distress"]["confidence"], "a rule decision reported a model confidence")
        self.assertIsNone(self.pipe._normal_road_entry()["confidence"])

    def test_ledger_uses_the_material_table(self):
        from models.ipm_homography_engine import IPMHomographyEngine
        r = self.pipe.audit_image(image_input=self._photo())
        if r.get("status") != "ANALYSIS_COMPLETE":
            self.skipTest("photo rejected at the gate")
        led = r["morth_civil_ledger"]
        self.assertEqual(led["compaction_factor"], IPMHomographyEngine.COMPACTION_FACTOR)
        self.assertEqual(led["mix_rate_inr_per_tonne"],
                         IPMHomographyEngine.MATERIAL_PROPERTIES["DBM_SECTION_500"]["cost_per_tonne_inr"])


class VisionSelectionRule(unittest.TestCase):
    HEAD = {"head": "svc_C3", "features": "flip_tta",
            "heads_compared": {"svc_C3|flip_tta": {"cv_accuracy": 0.80, "cv_macro_f1": 0.75, "score": 0.775}},
            "held_out_test_accuracy": 0.87, "held_out_test_macro_f1": 0.70}

    def ft(self, acc, f1):
        return {"chosen_arch": "efficientnet_b0", "archs": {"efficientnet_b0": {"val": {"accuracy": acc, "macro_f1": f1}}},
                "served_test": {"accuracy": 0.5, "macro_f1": 0.5}}

    def test_needs_both_metrics(self):
        from scripts.select_vision_model import decide
        self.assertEqual(decide(self.HEAD, self.ft(0.85, 0.80))[0], "deep_cnn")
        self.assertEqual(decide(self.HEAD, self.ft(0.85, 0.74))[0], "cnn_embeddings")
        self.assertEqual(decide(self.HEAD, self.ft(0.79, 0.90))[0], "cnn_embeddings")
        self.assertEqual(decide(self.HEAD, None)[0], "cnn_embeddings")

    def test_test_scores_do_not_enter_the_decision(self):
        from scripts.select_vision_model import decide
        a = self.ft(0.85, 0.80); b = self.ft(0.85, 0.80)
        b["served_test"] = {"accuracy": 0.01, "macro_f1": 0.01}
        self.assertEqual(decide(self.HEAD, a), decide(self.HEAD, b))

    def test_loader_falls_back_when_selected_model_is_missing(self):
        from models.deep_vision_net import load_best_vision_model
        tmp = tempfile.mkdtemp()
        try:
            for f in os.listdir(os.path.join(ENGINE_ROOT, "checkpoints")):
                if f.startswith(("cnn_head_mobilenetv2", "cnn_backbone_mobilenetv2", "vision_distress_model")):
                    shutil.copy(os.path.join(ENGINE_ROOT, "checkpoints", f), tmp)
            json.dump({"served": "deep_cnn"}, open(os.path.join(tmp, "vision_model_selection.json"), "w"))
            _clf, backend = load_best_vision_model(tmp, verbose=False)
            self.assertNotEqual(backend, "deep_cnn")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class RDD2022VocConversion(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.tmp = tempfile.mkdtemp()
        src = os.path.join(self.tmp, "India", "train")
        os.makedirs(os.path.join(src, "images")); os.makedirs(os.path.join(src, "annotations", "xmls"))
        spec = {0: [("D00", 10, 10, 60, 40), ("D40", 100, 100, 160, 150)],  # crack + pothole
                1: [],                                                     # clean photo
                2: [("D44", 10, 10, 50, 50)],                              # other class only -> skipped
                3: [("D20", 5, 5, 80, 90), ("Repair", 1, 1, 9, 9)]}        # mixed -> keep D20 only
        for i in range(20):
            stem = f"India_{i:06d}"
            Image.fromarray(np.zeros((200, 300, 3), np.uint8)).save(os.path.join(src, "images", stem + ".jpg"))
            root = ET.Element("annotation")
            sz = ET.SubElement(root, "size")
            ET.SubElement(sz, "width").text = "300"; ET.SubElement(sz, "height").text = "200"
            for name, x0, y0, x1, y1 in spec[i % 4]:
                o = ET.SubElement(root, "object"); ET.SubElement(o, "name").text = name
                bb = ET.SubElement(o, "bndbox")
                for k, v in zip(("xmin", "ymin", "xmax", "ymax"), (x0, y0, x1, y1)):
                    ET.SubElement(bb, k).text = str(v)
            ET.ElementTree(root).write(os.path.join(src, "annotations", "xmls", stem + ".xml"))
        self.src = os.path.join(self.tmp, "India")
        self.out = os.path.join(self.tmp, "yolo")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_layout_split_and_skips(self):
        from scripts.prepare_rdd2022_voc import convert
        import scripts.ingest_rdd2022_india as ing
        rep = convert(self.src, self.out)
        self.assertEqual(rep["skipped_other_classes_only"], 5)
        stems = {}
        for split in ("train", "valid", "test"):
            imgs = sorted(os.listdir(os.path.join(self.out, "images", split)))
            stems[split] = {os.path.splitext(f)[0] for f in imgs}
            for f in imgs:
                self.assertTrue(os.path.exists(os.path.join(self.out, "labels", split, os.path.splitext(f)[0] + ".txt")))
        self.assertEqual(sum(len(v) for v in stems.values()), 15)
        self.assertFalse(stems["train"] & stems["test"])
        self.assertFalse(stems["valid"] & stems["test"])
        names = ing.read_names(self.out)
        self.assertEqual(names, ["D00", "D10", "D20", "D40"])
        # the "Repair" box of a mixed photo is dropped, its D20 box kept
        all_lines = []
        for split in stems:
            for st in stems[split]:
                all_lines += open(os.path.join(self.out, "labels", split, st + ".txt")).read().split("\n")
        cls_ids = {int(l.split()[0]) for l in all_lines if l.strip()}
        self.assertEqual(cls_ids, {0, 2, 3})


class IMUSelection(unittest.TestCase):
    def test_random_forest_without_a_selection(self):
        from models.imu_shock_classifier import load_served_imu_model
        tmp = tempfile.mkdtemp()
        try:
            shutil.copy(os.path.join(ENGINE_ROOT, "checkpoints", "imu_shock_model.joblib"), tmp)
            m, name = load_served_imu_model(tmp)
            self.assertEqual(name, "random_forest")
            json.dump({"served": "cnn"}, open(os.path.join(tmp, "imu_model_selection.json"), "w"))
            m, name = load_served_imu_model(tmp)  # CNN named but absent -> forest, not a crash
            self.assertEqual(name, "random_forest")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class PrivacyRedaction(unittest.TestCase):
    """The report claimed faces and plates were blurred; this pins that they are."""

    def _scene(self):
        rng = np.random.RandomState(0)
        img = (rng.rand(240, 320, 3) * 255).astype(np.uint8)
        return img

    def test_person_head_and_plate_are_blurred_and_nothing_else(self):
        from models.privacy_redactor import redact
        img = self._scene()
        dets = [{"class_name": "person", "bbox_pixels": [20, 20, 60, 120]},
                {"class_name": "car", "bbox_pixels": [150, 100, 150, 120]}]
        plate = lambda crop: (40, 70, 60, 18)
        out, rep = redact(img, detections=dets, plate_locator=plate)
        self.assertEqual(rep["people_blurred"], 1)
        self.assertEqual(rep["plates_blurred"], 1)
        head = (slice(20, 20 + int(120 * 0.35)), slice(20, 80))
        self.assertGreater(np.abs(out[head].astype(int) - img[head].astype(int)).mean(), 10)
        pl = (slice(170, 188), slice(190, 250))
        self.assertGreater(np.abs(out[pl].astype(int) - img[pl].astype(int)).mean(), 10)
        untouched = (slice(200, 240), slice(0, 100))
        self.assertTrue(np.array_equal(out[untouched], img[untouched]))
        self.assertFalse(rep["recall_measured"])

    def test_input_is_not_modified(self):
        from models.privacy_redactor import redact
        img = self._scene(); ref = img.copy()
        redact(img, detections=[{"class_name": "person", "bbox_pixels": [0, 0, 50, 50]}])
        self.assertTrue(np.array_equal(img, ref))

    def test_no_detector_is_reported_not_hidden(self):
        from models import privacy_redactor as pr
        saved = pr._haar_faces
        pr._haar_faces = lambda rgb: None
        try:
            _out, rep = pr.redact(self._scene())
        finally:
            pr._haar_faces = saved
        self.assertIn("warning", rep)


class KinematicsClaimNoProbability(unittest.TestCase):
    def test_rule_output_has_no_confidence(self):
        from models.alpr_incident_tracker import ALPRIncidentTracker
        track = [{"timestamp": 0.2 * i, "bbox": [200, 150, 80 * 1.8 ** i, 60 * 1.8 ** i]} for i in range(4)]
        r = ALPRIncidentTracker().analyze_vehicle_kinematics(track)
        self.assertIsNone(r["confidence"])
        self.assertIn("threshold_ratio", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
