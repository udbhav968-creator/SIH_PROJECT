"""
Regression tests for the measurement chain and the durable ledger.

    python -m unittest tests.test_geometry_and_storage -v

These cover the parts of the system that turn a photograph into a rupee figure,
which is where a silent error is most expensive. Each test pins a property that
would actually break the product, and the expected values are derived from the
geometry rather than copied from whatever the code currently returns.
"""

import os
import sys
import tempfile
import unittest

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import numpy as np

CKPT = os.path.join(ENGINE_ROOT, "checkpoints")


class MaskArea(unittest.TestCase):
    """Area from a segmentation mask, which is what the cost is built on."""

    def setUp(self):
        from models.camera_calibration import DEFAULT_PROFILE
        from models.ipm_homography_engine import IPMHomographyEngine
        self.ipm = IPMHomographyEngine.from_calibration(DEFAULT_PROFILE)

    def test_rectangular_mask_agrees_with_the_box_method(self):
        """
        The two methods must agree where they should.

        A mask that IS a rectangle should give the same area as projecting that
        rectangle's four corners. If it does not, the per-pixel footprint sum is
        wrong and every mask-derived area is wrong with it.
        """
        mask = np.zeros((480, 640), dtype=bool)
        mask[380:440, 290:350] = True
        mask_area, _diag = self.ipm.mask_area_m2(mask)
        box_area = self.ipm.calculate_surface_area_sqm(290, 380, 60, 60)
        self.assertAlmostEqual(mask_area, box_area, delta=0.05 * box_area,
                               msg="mask and box areas disagree on a rectangle")

    def test_a_box_overstates_a_diagonal_defect(self):
        """
        The reason this module exists.

        A diagonal crack fills a small fraction of its bounding box. If the box
        were an acceptable proxy the two areas would be close; they are not, and
        the test pins that they are not, so nobody 'simplifies' back to a box.
        """
        mask = np.zeros((480, 640), dtype=bool)
        for i in range(60):
            mask[380 + i, 290 + i:295 + i] = True
        mask_area, _ = self.ipm.mask_area_m2(mask)
        box_area = self.ipm.calculate_surface_area_sqm(290, 380, 65, 60)
        self.assertGreater(box_area, 4 * mask_area,
                           "a bounding box should grossly overstate a diagonal crack")

    def test_pixels_further_away_cover_more_ground(self):
        """Perspective: a pixel near the horizon is a much larger patch of road."""
        per_row = self.ipm.row_pixel_area_m2(480, 640)
        near, far = per_row[460], per_row[300]
        self.assertGreater(far, near,
                           "a pixel higher in the frame must cover more ground area")

    def test_empty_mask_is_zero_not_a_floor_value(self):
        area, diag = self.ipm.mask_area_m2(np.zeros((480, 640), dtype=bool))
        self.assertEqual(area, 0.0)
        self.assertEqual(diag["mask_pixels"], 0)

    def test_clipping_is_reported_not_silent(self):
        """An area beyond the valid range means the geometry has failed; say so."""
        mask = np.ones((480, 640), dtype=bool)
        _area, diag = self.ipm.mask_area_m2(mask, max_area_m2=1.0)
        self.assertTrue(diag["clipped"], "a clipped area must be flagged")
        self.assertGreater(diag["raw_area_m2"], 1.0)


class Calibration(unittest.TestCase):
    def test_mount_height_changes_the_measured_area(self):
        """
        If this ever stops being true, calibration has been disconnected and
        every vehicle is silently sharing one assumed mount again.
        """
        from models.camera_calibration import DEFAULT_PROFILE
        from models.ipm_homography_engine import IPMHomographyEngine
        mask = np.zeros((480, 640), dtype=bool)
        mask[380:440, 290:350] = True
        low = dict(DEFAULT_PROFILE)
        high = dict(DEFAULT_PROFILE, camera_height_m=DEFAULT_PROFILE["camera_height_m"] + 0.3)
        a_low, _ = IPMHomographyEngine.from_calibration(low).mask_area_m2(mask)
        a_high, _ = IPMHomographyEngine.from_calibration(high).mask_area_m2(mask)
        self.assertGreater(abs(a_high - a_low) / a_low, 0.15,
                           "30 cm of mount height should move the area by >15%")

    def test_profile_is_rescaled_to_the_actual_resolution(self):
        """
        Intrinsics are in pixels. A profile calibrated at 1280x720 applied to a
        640x360 frame is wrong by a factor of two - straight through to area.
        """
        from models.camera_calibration import scaled_to
        p = {"device_id": "x", "camera_height_m": 1.5, "pitch_deg": 16.0,
             "fx": 1200.0, "fy": 1200.0, "cx": 640.0, "cy": 360.0,
             "image_width": 1280, "image_height": 720}
        s = scaled_to(p, 640, 360)
        self.assertAlmostEqual(s["fx"], 600.0)
        self.assertAlmostEqual(s["cx"], 320.0)
        self.assertEqual(s["rescaled_from"], [1280, 720])

    def test_absurd_values_are_rejected(self):
        from models.camera_calibration import CalibrationError, validate
        base = {"camera_height_m": 1.5, "pitch_deg": 16.0,
                "fx": 1200.0, "fy": 1200.0, "cx": 640.0, "cy": 360.0}
        validate(base)
        for bad in ({"camera_height_m": 45.0}, {"pitch_deg": 120.0}, {"fx": 3.0}):
            with self.assertRaises(CalibrationError):
                validate(dict(base, **bad))

    def test_provenance_says_when_it_is_an_assumption(self):
        from models.camera_calibration import describe, resolve
        profile, prov = resolve(None)
        self.assertEqual(prov, "default")
        self.assertIn("ESTIMATE", describe(profile, prov)["measurement_basis"])

    def test_round_trip_through_disk(self):
        from models.camera_calibration import from_field_of_view, load, save
        with tempfile.TemporaryDirectory() as d:
            p = from_field_of_view(1280, 720, 78.0, 1.52, 16.0, device_id="unit-test")
            save(p, calib_dir=d)
            back = load("unit-test", calib_dir=d)
            self.assertIsNotNone(back)
            self.assertAlmostEqual(back["fx"], p["fx"], places=3)
            self.assertIsNone(load("no-such-device", calib_dir=d))


class DepthHonesty(unittest.TestCase):
    """
    The old crack depth was `1.5 + confidence * 2.0` - the classifier's
    confidence with centimetres written after it. These tests make that
    impossible to reintroduce.
    """

    def test_depth_is_never_presented_as_a_measurement(self):
        from models import depth_estimator
        for est in (depth_estimator.crack_depth(),
                    depth_estimator.crack_depth(0.1),
                    depth_estimator.pothole_depth(np.full((100, 100), 120, np.float32))):
            self.assertFalse(est["is_measurement"])
            self.assertIn("ESTIMATE", est["caveat"])

    def test_every_estimate_carries_an_interval(self):
        from models import depth_estimator
        est = depth_estimator.crack_depth(0.05)
        self.assertLessEqual(est["depth_low_cm"], est["depth_cm"])
        self.assertGreaterEqual(est["depth_high_cm"], est["depth_cm"])

    def test_crack_depth_does_not_depend_on_classifier_confidence(self):
        """crack_depth has no confidence parameter at all - by design."""
        import inspect
        from models import depth_estimator
        params = inspect.signature(depth_estimator.crack_depth).parameters
        self.assertNotIn("confidence", params)
        self.assertIn("severity_ratio", params)

    def test_darker_cavity_reads_deeper_and_more_confidently(self):
        from models import depth_estimator
        road = np.full((200, 200), 150, np.float32)
        mask = np.zeros((200, 200), bool)
        mask[80:120, 80:120] = True
        shallow = road.copy(); shallow[80:120, 80:120] = 138
        deep = road.copy();    deep[80:120, 80:120] = 60
        a = depth_estimator.pothole_depth(shallow, mask=mask)
        b = depth_estimator.pothole_depth(deep, mask=mask)
        self.assertGreater(b["depth_cm"], a["depth_cm"])
        self.assertGreater(b["method_confidence"], a["method_confidence"])
        # and the interval should narrow as the cue strengthens
        self.assertLess(b["depth_high_cm"] - b["depth_low_cm"],
                        a["depth_high_cm"] - a["depth_low_cm"])

    def test_stereo_is_the_only_thing_allowed_to_claim_measurement(self):
        from models import depth_estimator
        est = depth_estimator.stereo_depth_from_disparity(40.0, 0.12, 1120.0)
        self.assertTrue(est["is_measurement"])
        with self.assertRaises(ValueError):
            depth_estimator.stereo_depth_from_disparity(0.0, 0.12, 1120.0)


class DurableLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")

    def test_defects_survive_a_restart(self):
        from pipeline.defect_store import DefectStore
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        e1 = FleetDeduplicationEngine(store=DefectStore(self.db))
        e1.ingest_fleet_detection("BUS-1", 12.9716, 77.5946, "Pothole Cavity",
                                  42.0, 0.8, enrich_location=False)
        e2 = FleetDeduplicationEngine(store=DefectStore(self.db))
        self.assertEqual(len(e2.get_all_deduplicated_defects()), 1,
                         "the ledger did not survive a restart")

    def test_ids_do_not_collide_after_a_restart(self):
        """
        The id counter has to be recovered from the database, not restarted at
        1001. If it is not, the first defect after a restart silently overwrites
        the first defect from before it - and since ids look like DEF-BLR-1001,
        recovering the maximum means parsing the numeric tail, not sorting the
        string (DEF-BLR-999 sorts above DEF-BLR-1001).
        """
        from pipeline.defect_store import DefectStore
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        e1 = FleetDeduplicationEngine(store=DefectStore(self.db))
        for i in range(3):
            # ~1 km apart so each is a genuinely distinct defect, not a merge
            e1.ingest_fleet_detection(f"BUS-{i}", 12.97 + i * 0.01, 77.59,
                                      "Pothole Cavity", 40.0, 1.0, enrich_location=False)
        ids_before = {d["defect_id"] for d in e1.get_all_deduplicated_defects()}
        self.assertEqual(len(ids_before), 3, "fixture should create three distinct defects")

        e2 = FleetDeduplicationEngine(store=DefectStore(self.db))
        # somewhere none of the originals are, so this must be a new record
        res = e2.ingest_fleet_detection("BUS-9", 13.10, 77.70, "Crack", 60.0, 0.2,
                                        enrich_location=False)
        self.assertEqual(res["action"], "REGISTERED_NEW_DEFECT")
        self.assertNotIn(res["defect_id"], ids_before,
                         "a new defect reused an id issued before the restart")
        ids_after = {d["defect_id"] for d in e2.get_all_deduplicated_defects()}
        self.assertEqual(len(ids_after), 4)

    def test_raw_sightings_are_kept_as_the_dedup_audit_trail(self):
        """Merged defects alone lose the evidence that merging happened."""
        from pipeline.defect_store import DefectStore
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        store = DefectStore(self.db)
        e = FleetDeduplicationEngine(store=store)
        e.ingest_fleet_detection("BUS-1", 12.97160, 77.59450, "Pothole Cavity",
                                 42.0, 0.8, enrich_location=False)
        e.ingest_fleet_detection("BUS-2", 12.971605, 77.594510, "Pothole Cavity",
                                 41.0, 0.9, enrich_location=False)
        s = store.stats()
        self.assertEqual(s["unique_defects"], 1)
        self.assertEqual(s["total_reports"], 2)
        self.assertEqual(s["merged_reports"], 1)
        self.assertEqual(s["deduplication_rate_pct"], 50.0)

    def test_stats_come_from_the_rows_not_a_counter(self):
        from pipeline.defect_store import DefectStore
        store = DefectStore(self.db)
        store.upsert_defect({"defect_id": "DEF-X-1", "lat": 1.0, "lon": 2.0,
                             "defect_class": "Crack", "confirmation_count": 1,
                             "reporting_buses": ["B"], "first_seen_timestamp": 1.0,
                             "last_seen_timestamp": 1.0})
        self.assertEqual(store.stats()["unique_defects"], 1)
        # upserting the same id must not create a second row
        store.upsert_defect({"defect_id": "DEF-X-1", "lat": 1.0, "lon": 2.0,
                             "defect_class": "Crack", "confirmation_count": 2,
                             "reporting_buses": ["B", "C"], "first_seen_timestamp": 1.0,
                             "last_seen_timestamp": 2.0})
        self.assertEqual(store.stats()["unique_defects"], 1)
        self.assertEqual(store.load_defects()["DEF-X-1"]["confirmation_count"], 2)


class VideoIngest(unittest.TestCase):
    def test_gps_track_interpolates_between_fixes(self):
        from pipeline.video_ingest import GPSTrack
        t = GPSTrack([{"t": 0, "lat": 10.0, "lon": 20.0},
                      {"t": 10, "lat": 10.001, "lon": 20.0}])
        mid = t.position_at(5.0)
        self.assertAlmostEqual(mid[0], 10.0005, places=6)
        # outside the track it clamps rather than extrapolating into fiction
        self.assertEqual(t.position_at(-5.0), (10.0, 20.0))
        self.assertEqual(t.position_at(99.0), (10.001, 20.0))

    def test_no_gps_means_no_invented_position(self):
        from pipeline.video_ingest import GPSTrack
        self.assertIsNone(GPSTrack([]).position_at(1.0))

    def test_haversine_matches_a_known_distance(self):
        from pipeline.video_ingest import GPSTrack
        # 0.001 degrees of latitude is about 111 m anywhere on Earth
        d = GPSTrack.haversine_m(12.9716, 77.5946, 12.9726, 77.5946)
        self.assertAlmostEqual(d, 111.0, delta=3.0)

    def test_decodes_a_real_video_and_samples_it(self):
        import shutil
        from pipeline.deep_inference_pipeline import DeepInferencePipeline
        from pipeline.video_ingest import VideoIngestor, synthesise_test_video
        tmp = tempfile.mkdtemp()
        try:
            path = synthesise_test_video(os.path.join(tmp, "clip.mp4"), frames=30, fps=15)
            info = VideoIngestor.probe(path)
            self.assertGreater(info["frame_count"], 0)
            ing = VideoIngestor(DeepInferencePipeline(CKPT))
            out = ing.process(path, gps_track=None, sample_every_s=0.5, max_frames=4)
            self.assertGreater(out["frames_analysed"], 0)
            self.assertFalse(out["gps"]["available"])
            self.assertIn("no position has been invented", out["gps"]["note"].lower())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class FalsePositiveGate(unittest.TestCase):
    """
    A zebra crossing was reported as Pothole Cavity, and a whole clean road was
    covered by one enormous pothole box.

    Two causes, both pinned here:

    1. Region proposals came from a 16x24 brightness grid. On a textured road
       most cells fire and merge into ONE box over the carriageway. Proposals
       now come from connected components of the segmentation mask.

    2. The segmenter had never seen road paint without a defect in frame -
       every DNIT training photograph contains one - so it learned
       "high-contrast patch = pothole". Clean photographs are now training data.

    Every test here pins BOTH halves of the trade-off. A build that reports
    nothing at all would pass the false-positive tests and be worthless, so the
    detection tests sit beside them deliberately.
    """

    @classmethod
    def setUpClass(cls):
        from pipeline.deep_inference_pipeline import DeepInferencePipeline
        cls.pipe = DeepInferencePipeline(CKPT)

    def _defects(self, path):
        r = self.pipe.audit_image(image_input=path)
        return [d for d in (r.get("all_detections") or [])
                if any(k in d.get("class_name", "") for k in ("Pothole", "Crack"))]

    def _sample(self, folder, n=4, seed=7):
        """
        A seeded shuffle, not the first n alphabetically.

        Taking the head of a sorted listing is not a sample: in this dataset the
        first entries are all large frames from one capture run, and a test
        written against them measured 3/6 while the same build measured 87.5%
        over a real sample. Same seed as scripts/measure_detection_quality.py so
        the test and the reported number are talking about the same thing.
        """
        import glob
        import random
        files = [p for p in sorted(glob.glob(os.path.join(ENGINE_ROOT, "datasets", folder,
                                                          "**", "*.jpg"), recursive=True))
                 if "_label_conflicts" not in p]
        random.Random(seed).shuffle(files)
        return files[:n]

    def _need_full_stack(self):
        """
        Refuse to judge the product while a weaker stand-in is loaded.

        The CNN backbone falls back to hand-crafted features when it cannot
        load - correct for the product, fatal for a test suite, because the
        suite then measures a classifier nobody ships. Observed on a machine out
        of memory: ONNX Runtime said "bad allocation", the fallback took over,
        and the zebra-crossing test failed 3 of 5. Nothing had regressed.
        """
        vm = getattr(self.pipe, "vision_model", None)
        fb = getattr(vm, "backbone_fallback", None)
        if fb:
            msg = (f"The CNN backbone ({fb['requested']}) is NOT loaded, so these tests "
                   f"would be measuring the hand-crafted fallback - a different, weaker "
                   f"model.\n  error: {fb['error']}")
            if fb.get("environment_failure"):
                self.skipTest(msg + "\n  This is MEMORY, not a code problem. Close other "
                                    "applications and any running api.server, then re-run.")
            self.fail(msg + "\n  Run: python -m scripts.fetch_cnn_backbone")

    def _need_segmenter(self):
        self._need_full_stack()
        """
        Skip only when no segmenter was ever trained. FAIL when one is on disk
        and will not load.

        This distinction is the whole point. On a machine with scikit-learn
        1.9.1 the model trained under 1.8.0 failed to unpickle, the pipeline
        fell back to brightness proposals - the configuration that reports zebra
        crossings as potholes - and this suite printed "OK (skipped=7)". A skip
        that hides a broken install is worse than having no test at all.
        """
        seg = getattr(self.pipe, "segmenter", None)
        if seg is None:
            self.skipTest("segmenter module not wired into the pipeline")
        if seg.is_ready:
            return
        if getattr(seg, "file_exists", False):
            d = getattr(seg, "load_error_detail", None) or {}
            self.fail(
                "A segmenter model is on disk but will not load, so the pipeline is "
                "running on brightness proposals and the false-positive fix is NOT "
                "active on this machine.\n"
                f"  error                : {d.get('error')}\n"
                f"  scikit-learn here    : {d.get('sklearn_here')}\n"
                f"  trained with         : {d.get('sklearn_trained_with')}\n"
                f"  likely cause         : {d.get('likely_cause')}\n"
                f"  fix                  : {d.get('fix')}")
        self.skipTest("no segmenter trained yet - run training.train_segmenter")

    # ---------------------------------------------------------------- wiring
    def test_proposal_thresholds_are_the_measured_ones(self):
        from pipeline.deep_inference_pipeline import DeepInferencePipeline as P
        # Both came from scripts/tune_proposals.py sweeping end-to-end through
        # audit_image(). Changing either without re-running that sweep is how
        # the previous regression happened.
        # Per class, because a pixel COUNT is shape-dependent: a crack of the
        # same real extent as a pothole has an order of magnitude fewer pixels.
        # One shared floor silently made the system pothole-only for thin
        # cracks - it was discarding blobs of 52 and 107 px against a 256 px
        # floor.
        self.assertIsInstance(P.MIN_COMPONENT_FRACTION, dict,
                              "the size floor must be per class")
        self.assertAlmostEqual(P.MIN_COMPONENT_FRACTION["pothole"], 0.004, places=4)
        self.assertAlmostEqual(P.MIN_COMPONENT_FRACTION["crack"], 0.0012, places=5)
        self.assertLess(P.MIN_COMPONENT_FRACTION["crack"],
                        P.MIN_COMPONENT_FRACTION["pothole"],
                        "a crack is thinner than a pothole and needs a lower floor")
        for name, frac in P.MIN_COMPONENT_FRACTION.items():
            self.assertGreater(frac, 0.0, f"a zero {name} floor floods a clean road")
            self.assertLess(frac, 0.05, f"a large {name} floor discards real defects")

        # The confidence floor must be a MARGIN, never an absolute probability.
        # An absolute 0.45 measured well on the model trained here (pothole
        # threshold 0.35) and discarded real potholes on a model calibrated to
        # 0.20, because 0.45 is 2.25x that threshold.
        self.assertIsInstance(P.MIN_COMPONENT_MARGIN, dict,
                              "the floor must be per class: a crack is thin, so "
                              "its blob mean sits near its threshold by shape")
        self.assertEqual(P.MIN_COMPONENT_MARGIN["crack"], 0.0)
        self.assertAlmostEqual(P.MIN_COMPONENT_MARGIN["pothole"], 0.10, places=2)

    def test_the_floor_follows_the_model_it_is_loaded_with(self):
        """Two models, two calibrations, and the floor must track each."""
        self._need_segmenter()
        real = self.pipe.segmenter.thresholds
        try:
            self.pipe.segmenter.thresholds = {"crack": 0.50, "pothole": 0.20}
            self.assertAlmostEqual(self.pipe._component_floor(2), 0.30, places=2)
            self.assertAlmostEqual(self.pipe._component_floor(1), 0.50, places=2)
            self.pipe.segmenter.thresholds = {"crack": 0.60, "pothole": 0.35}
            self.assertAlmostEqual(self.pipe._component_floor(2), 0.45, places=2)
            self.assertAlmostEqual(self.pipe._component_floor(1), 0.60, places=2)
        finally:
            self.pipe.segmenter.thresholds = real

    def test_proposals_come_from_the_mask_not_from_brightness(self):
        """The fix is architectural. If proposals silently revert to the
        brightness grid, the giant box comes back and no threshold helps."""
        self._need_segmenter()
        files = self._sample("02_kaggle_pothole_600", 3)
        if not files:
            self.skipTest("no pothole photographs on disk")
        saw_segmentation_proposal = False
        for path in files:
            img = self.pipe.cv_detector.decode_image(path)
            H, W, _ = img.shape
            self.pipe._seg_out = self.pipe.segmenter.segment(img)
            for b in self.pipe._segmentation_proposals(H, W):
                self.assertEqual(b[6], "segmentation")
                saw_segmentation_proposal = True
        self.assertTrue(saw_segmentation_proposal,
                        "no proposal came from the segmentation mask on any "
                        "pothole photograph")

    def test_reported_boxes_are_tight_not_whole_frame(self):
        """The regression users actually saw: one box over the entire road."""
        self._need_segmenter()
        files = self._sample("02_kaggle_pothole_600", 4) + self._sample("03_crack500_fatigue", 4)
        if not files:
            self.skipTest("no defect photographs on disk")
        offenders = []
        for path in files:
            for d in self._defects(path):
                bb = d.get("bbox_normalized")
                if not bb:
                    continue
                frac = float(bb[2]) * float(bb[3])
                if frac > 0.60:
                    offenders.append((os.path.basename(path), round(frac, 3)))
        self.assertEqual(offenders, [],
                         f"box covering more than 60% of the frame: {offenders}")

    # ------------------------------------------------------- false positives
    def test_zebra_crossings_are_not_reported_as_potholes(self):
        self._need_segmenter()
        files = self._sample("10_missing_zebra_crossing", 5)
        if not files:
            self.skipTest("no zebra-crossing photographs on disk")
        bad = [os.path.basename(p) for p in files if self._defects(p)]
        # Measured end-to-end: 3 of 36 clean photographs (8.3%) are still
        # reported. A hard zero would be a test that passes by luck on this
        # sample; the honest number lives in
        # checkpoints/detection_quality_report.json.
        self.assertLessEqual(len(bad), len(files) // 2,
                             f"{len(bad)}/{len(files)} zebra crossings reported as a "
                             f"crack or pothole: {bad}")

    def test_sound_pavement_is_not_reported_as_cracked(self):
        self._need_segmenter()
        files = self._sample("05_morth_civil_hard_negatives", 6)
        if not files:
            self.skipTest("no hard negatives on disk")
        bad = [os.path.basename(p) for p in files if self._defects(p)]
        self.assertLessEqual(len(bad), len(files) // 3,
                             f"{len(bad)}/{len(files)} sound-pavement photographs "
                             f"reported as defective: {bad}")

    # ------------------------------------------------------------- detection
    def test_real_defects_are_still_reported(self):
        """Without this, setting every threshold to 1.0 would pass the rest."""
        self._need_segmenter()
        files = self._sample("02_kaggle_pothole_600", 8)
        if not files:
            self.skipTest("no pothole photographs on disk")
        found = sum(1 for p in files if self._defects(p))
        # Measured end-to-end over 24 photographs: 87.5%. The bar here is 62.5%
        # because eight photographs is a small sample and a test that only
        # passes at the measured rate is a test that fails on noise. It is still
        # tight enough to catch the collapse to 20.8% that over-tightening the
        # proposal filter produced.
        self.assertGreaterEqual(found, (5 * len(files)) // 8,
                                f"only {found}/{len(files)} real defects detected - "
                                f"the proposal filter is too aggressive")

    def test_nothing_is_proposed_above_the_horizon(self):
        """
        A parked car was reported as "Pothole 100%" while the crater filling the
        foreground was ignored.

        The segmenter's features - lightness, gradient, local variance - describe
        a dark high-contrast patch, and a car body against bright tarmac is
        exactly that. No probability threshold separates them. Their POSITION
        does: the road surface in front of the vehicle is the lower part of the
        frame, and the brightness grid had always known that while the
        segmentation proposals did not.
        """
        self._need_segmenter()
        from pipeline.deep_inference_pipeline import DeepInferencePipeline as P
        self.assertGreater(P.ROAD_ROI_TOP_FRACTION, 0.0)
        self.assertLess(P.ROAD_ROI_TOP_FRACTION, 0.6,
                        "cutting too low would discard defects close to the camera")

        files = self._sample("02_kaggle_pothole_600", 6) + self._sample("10_missing_zebra_crossing", 4)
        if not files:
            self.skipTest("no photographs on disk")
        offenders = []
        for path in files:
            img = self.pipe.cv_detector.decode_image(path)
            H, W, _ = img.shape
            self.pipe._seg_out = self.pipe.segmenter.segment(img)
            for b in self.pipe._segmentation_proposals(H, W):
                centroid_y = b[1] + b[3] / 2.0
                if centroid_y < P.ROAD_ROI_TOP_FRACTION * H:
                    offenders.append((os.path.basename(path), round(centroid_y / H, 3)))
        self.assertEqual(offenders, [],
                         f"proposed a defect centred above the road ROI: {offenders}")

    # --------------------------------------------------- plausible to price
    def test_an_implausible_area_is_flagged_not_priced(self):
        """
        A number that cannot be defended must not carry a rupee figure.

        Measured on a real photograph before this guard: one segmentation blob
        covering 83% x 84% of the frame was reported as a single 4.273 m2
        pothole and priced at Rs 2,830. The arithmetic was correct; the answer
        was nonsense, because the ground-plane projection had been handed a
        close-up crop rather than a frame from a mounted camera.

        The defect is still REPORTED - a road authority needs to know it is
        there - but it is sent for manual survey instead of being costed.
        Dropping it outright was tried and cost 8 points of detection.
        """
        self._need_segmenter()
        from pipeline.deep_inference_pipeline import DeepInferencePipeline as P
        self.assertLessEqual(P.MAX_SINGLE_REPAIR_AREA_M2, 3.0,
                             "a single pothole patch is about a metre across")
        self.assertLessEqual(P.MAX_COMPONENT_BOX_FRACTION, 0.40,
                             "a region covering most of the frame is not one repair")

        files = self._sample("02_kaggle_pothole_600", 10) + self._sample("03_crack500_fatigue", 6)
        if not files:
            self.skipTest("no defect photographs on disk")
        priced_but_absurd = []
        for path in files:
            for d in self._defects(path):
                cost = float(d.get("repair_cost_inr") or 0.0)
                area = float(d.get("surface_area_m2") or 0.0)
                bb = d.get("bbox_normalized") or [0, 0, 0, 0]
                frac = float(bb[2]) * float(bb[3])
                if cost > 0 and (area > P.MAX_SINGLE_REPAIR_AREA_M2
                                 or frac > P.MAX_COMPONENT_BOX_FRACTION):
                    priced_but_absurd.append(
                        (os.path.basename(path), round(area, 3), round(frac, 3), cost))
        self.assertEqual(priced_but_absurd, [],
                         f"priced a defect whose area or extent cannot be defended: "
                         f"{priced_but_absurd}")

    # ------------------------------------------------------------- reporting
    def test_segmenter_report_measures_clean_road_false_positives(self):
        """IoU is computed only on photographs that contain a defect, so it is
        blind to inventing defects on clean roads. That blind spot is why the
        bug survived; the metric that closes it must not disappear."""
        import json
        path = os.path.join(CKPT, "defect_segmenter_report.json")
        if not os.path.exists(path):
            self.skipTest("no segmenter report")
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        fp = r.get("false_positives_on_clean_roads")
        self.assertIsNotNone(fp, "the segmenter report no longer measures false "
                                 "positives on clean roads")
        self.assertGreater(fp["clean_photographs_scored"], 0)
        self.assertLess(fp["photo_rate_any_blob"], 0.25,
                        "clean-road false blobs are back above a quarter of photographs")

    def test_the_withdrawn_number_is_not_quoted_anywhere(self):
        """27.8% was measured by a harness that reimplemented the pipeline. It
        was published once. It must not come back through a stale key."""
        import json
        path = os.path.join(CKPT, "claims.json")
        if not os.path.exists(path):
            self.skipTest("no claims registry")
        with open(path, encoding="utf-8") as fh:
            claims = json.load(fh)
        gate = [s for s in claims["subsystems"] if s["id"] == "M_GATE"]
        self.assertTrue(gate, "the proposal subsystem is missing from the claims registry")
        measured = gate[0].get("measured") or {}
        self.assertNotIn(0.278, [round(float(v), 3) for v in measured.values()
                                 if isinstance(v, (int, float))])
        self.assertIn("audit_image", measured.get("measured_through") or "",
                      "the gate's numbers must state that they came from the "
                      "real entry point, not from a harness that reimplements it")


class SegmenterContract(unittest.TestCase):

    def test_a_model_on_disk_actually_loads(self):
        """
        The deployment test.

        Every other segmenter test skips when nothing is loaded, which means a
        machine where the model file is present but unreadable passes the whole
        suite while running the buggy fallback. This one fails there, loudly,
        and names the scikit-learn versions involved.
        """
        from models.defect_segmenter import DefectSegmenter
        seg = DefectSegmenter()
        if not seg.file_exists:
            self.skipTest("no segmenter has been trained on this machine yet")
        self.assertTrue(seg.is_ready,
                        "checkpoints/defect_segmenter.joblib exists but will not load. "
                        "The pipeline is silently using brightness proposals, which is "
                        "the bug this model was trained to fix.\n"
                        f"  {getattr(seg, 'load_error_detail', None)}")

    def test_mask_shape_and_classes(self):
        from models.defect_segmenter import CLASS_CRACK, CLASS_POTHOLE, CLASS_SOUND, DefectSegmenter
        seg = DefectSegmenter()
        if not seg.is_ready:
            self.skipTest("no segmenter trained")
        img = (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)
        out = seg.segment(img)
        self.assertEqual(out["mask"].shape, (240, 320))
        self.assertTrue(set(np.unique(out["mask"])) <= {CLASS_SOUND, CLASS_CRACK, CLASS_POTHOLE})

    def test_thresholds_are_calibrated_not_argmax(self):
        from models.defect_segmenter import DefectSegmenter
        seg = DefectSegmenter()
        if not seg.is_ready:
            self.skipTest("no segmenter trained")
        self.assertIn("crack", seg.thresholds)
        self.assertIn("pothole", seg.thresholds)

    def test_reported_iou_is_from_a_grouped_split(self):
        import json
        path = os.path.join(CKPT, "defect_segmenter_report.json")
        if not os.path.exists(path):
            self.skipTest("no segmenter report")
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        self.assertIn("by photograph", r["split_strategy"])
        self.assertGreater(r["trained_on"]["test_photographs"], 0)
        for cls in ("crack", "pothole"):
            self.assertGreater(r["iou"][cls]["iou"], 0.0,
                               f"{cls} IoU should be positive on a trained model")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class MarkingDetection(unittest.TestCase):
    """
    Painted markings, found by geometry rather than by a model.

    Nineteen zebra photographs is not enough to train a detector on, and a
    crossing is the most regular structure on any road, so it is measured
    directly: parallel bright bars with regular spacing.
    """

    def _sample(self, folder, n, seed=7):
        import glob
        import random
        files = [p for p in sorted(glob.glob(os.path.join(ENGINE_ROOT, "datasets", folder,
                                                          "**", "*.jpg"), recursive=True))
                 if "_label_conflicts" not in p]
        random.Random(seed).shuffle(files)
        return files[:n]

    def _found(self, paths):
        import cv2
        from models.marking_detector import detect_zebra
        hits = 0
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            if detect_zebra(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["found"]:
                hits += 1
        return hits

    def test_it_does_not_fire_on_plain_pavement(self):
        """
        Specificity is the property that makes this worth having. A marking
        detector that fires on cracked asphalt would add noise to the exact
        photographs the product exists to analyse.
        """
        files = self._sample("02_kaggle_pothole_600", 14) + self._sample("03_crack500_fatigue", 14)
        if not files:
            self.skipTest("no defect photographs on disk")
        hits = self._found(files)
        self.assertLessEqual(hits, max(2, len(files) // 8),
                             f"{hits}/{len(files)} defect photographs claimed a painted "
                             f"marking - the detector is not specific enough to be useful")

    def test_it_finds_some_crossings(self):
        """
        Recall is poor and this test says so rather than hiding it: measured at
        6 of 19. The bar is set at "better than nothing" deliberately, because
        claiming more would be claiming something not measured.
        """
        files = self._sample("10_missing_zebra_crossing", 19)
        if not files:
            self.skipTest("no zebra photographs on disk")
        hits = self._found(files)
        self.assertGreaterEqual(hits, 3,
                                f"only {hits}/{len(files)} crossings found - the geometric "
                                f"test has stopped working altogether")

    def test_paint_mask_covers_bars_and_not_the_road_between(self):
        """
        The asphalt BETWEEN stripes is real road and can hold a real pothole.
        Masking the whole bounding box would blind the system to exactly the
        defect a pedestrian is most exposed to.
        """
        import cv2
        from models.marking_detector import detect_zebra, paint_mask
        for p in self._sample("10_missing_zebra_crossing", 19):
            img = cv2.imread(p)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            z = detect_zebra(rgb)
            if not z["found"]:
                continue
            m = paint_mask(rgb, z)
            b = z["bbox_pixels"]
            box_area = max(1, b[2] * b[3])
            covered = int(m[b[1]:b[1] + b[3], b[0]:b[0] + b[2]].sum())
            self.assertGreater(covered, 0, "the mask covers none of the bars")
            self.assertLess(covered / box_area, 0.95,
                            "the mask covers the whole crossing box, including the "
                            "road between the stripes")
            return
        self.skipTest("no crossing detected in the sample")
