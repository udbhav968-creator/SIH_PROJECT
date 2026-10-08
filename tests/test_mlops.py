"""MLOps: experiment tracking, model registry (gates, promotion, rollback), retraining pipeline,
monitoring and drift, shadow, active learning, the traffic estimate and the input guard."""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data) if not isinstance(data, str) else data)


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rs_mlops_")
        self.store = os.path.join(self.tmp, "store")
        self.ckpt = os.path.join(self.tmp, "ckpt")
        os.makedirs(self.ckpt)
        os.environ["ROAD_SHIELD_MLOPS_DIR"] = self.store
        os.environ["ROAD_SHIELD_MLFLOW"] = "0"

    def tearDown(self):
        os.environ.pop("ROAD_SHIELD_MLOPS_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)


class TrackingTest(_Tmp):
    def test_run_records_params_metrics_artifacts_and_failure(self):
        from mlops import tracking
        art = os.path.join(self.tmp, "model.bin")
        _write(art, "weights")
        with tracking.start_run("exp", params={"lr": 0.1}) as run:
            run.log_metric("loss", 0.9, step=1)
            run.log_metric("loss", 0.5, step=2)
            run.log_metrics({"f1": 0.8, "skip": "text"})
            run.log_artifact(art)
        r = tracking.get_run(run.run_id)
        self.assertEqual(r["status"], "FINISHED")
        self.assertEqual(r["params"]["lr"], 0.1)
        self.assertEqual(r["metrics"], {"loss": 0.5, "f1": 0.8})
        self.assertEqual([h["value"] for h in r["history"] if h["key"] == "loss"], [0.9, 0.5])
        self.assertEqual(r["artifacts"][0]["bytes"], 7)
        with self.assertRaises(RuntimeError):
            with tracking.start_run("exp") as bad:
                raise RuntimeError("boom")
        self.assertEqual(tracking.get_run(bad.run_id)["status"], "FAILED")
        self.assertIn("boom", tracking.get_run(bad.run_id)["error"])
        self.assertEqual(len(tracking.list_runs("exp")), 2)


class RegistryTest(_Tmp):
    def _model(self, auroc, refused=0.01, payload="v1"):
        _write(os.path.join(self.ckpt, "ood_guard.npz"), payload)
        _write(os.path.join(self.ckpt, "ood_guard_report.json"),
               {"test": {"combined": {"auroc": auroc, "in_distribution_flagged_rate": 0.04},
                         "refusals": {"road_photos_refused_rate": refused}}})

    def _reg(self):
        from mlops.registry import Registry
        return Registry(store=self.store, ckpt_dir=self.ckpt)

    def test_bootstrap_register_gate_promote_rollback(self):
        from mlops.registry import RegistryError
        self._model(0.95)
        reg = self._reg()
        self.assertEqual(reg.bootstrap(), {"ood_guard": 1})
        self.assertTrue(reg.served_state("ood_guard")["matches"])
        self.assertEqual(reg.production("ood_guard")["metrics"]["auroc_combined"], 0.95)

        self._model(0.97, payload="v2")                       # a better model
        v2 = reg.register("ood_guard")
        self.assertEqual((v2["version"], v2["stage"]), (2, "candidate"))
        self.assertEqual(reg.register("ood_guard")["version"], 2, "same bytes, same version")
        self._model(0.95)                                     # put production's files back
        self.assertTrue(reg.gate("ood_guard", 2)["passed"])
        res = reg.promote("ood_guard", 2, actor="test")
        self.assertEqual((res["version"], res["previous"]), (2, 1))
        with open(os.path.join(self.ckpt, "ood_guard.npz")) as fh:
            self.assertEqual(fh.read(), "v2", "promotion deployed the version's bytes")
        self.assertEqual(reg.get("ood_guard", 1)["stage"], "archived")

        back = reg.rollback("ood_guard", actor="test")
        self.assertEqual(back["version"], 1)
        with open(os.path.join(self.ckpt, "ood_guard.npz")) as fh:
            self.assertEqual(fh.read(), "v1")
        self.assertIn("rollback", [e["action"] for e in reg.events("ood_guard")])

        self._model(0.90, payload="v3")                       # worse than production by more than max_drop
        v3 = reg.register("ood_guard")
        self._model(0.95)
        g = reg.gate("ood_guard", v3["version"])
        self.assertFalse(g["passed"])
        with self.assertRaises(RegistryError):
            reg.promote("ood_guard", v3["version"])
        forced = reg.promote("ood_guard", v3["version"], force=True, reason="test")
        self.assertTrue(forced["forced"])

    def test_rollback_never_loses_a_hand_copied_model_and_walks_back(self):
        from mlops.registry import RegistryError
        self._model(0.95)
        reg = self._reg()
        reg.bootstrap()
        for i, a in ((2, 0.96), (3, 0.97)):
            self._model(a, payload=f"v{i}")
            reg.register("ood_guard")
            self._model(0.95 if i == 2 else 0.96, payload="v1" if i == 2 else "v2")
            reg.promote("ood_guard", i)
        self.assertEqual(reg.production("ood_guard")["version"], 3)
        self.assertEqual(reg.rollback("ood_guard")["version"], 2)
        self.assertEqual(reg.rollback("ood_guard")["version"], 1, "a second rollback goes further back")
        self._model(0.99, payload="copied by hand")
        with self.assertRaises(RegistryError):
            reg.promote("ood_guard", 3, force=True)          # force overrides the gate, not this
        res = reg.promote("ood_guard", 3, overwrite_unregistered=True)
        from mlops.paths import sha256_file
        self.assertTrue(res)
        import hashlib
        sha = hashlib.sha256(b"copied by hand").hexdigest()
        self.assertTrue(os.path.exists(reg._blob_path(sha)), "the overwritten file was kept in the blob store")

    def test_guard_metric_ceiling_blocks_promotion(self):
        self._model(0.95)
        reg = self._reg()
        reg.bootstrap()
        self._model(0.99, refused=0.2, payload="noisy")
        v = reg.register("ood_guard")
        self._model(0.95)
        g = reg.gate("ood_guard", v["version"])
        self.assertFalse(g["passed"])
        self.assertTrue(any(c["metric"] == "road_refused_rate" and not c["passed"] for c in g["checks"]))

    def test_hand_copied_files_are_never_overwritten_silently(self):
        from mlops.registry import RegistryError
        self._model(0.95)
        reg = self._reg()
        reg.bootstrap()
        self._model(0.97, payload="v2")
        v2 = reg.register("ood_guard")
        self._model(0.96, payload="someone copied this in")
        self.assertFalse(reg.served_state("ood_guard")["matches"])
        with self.assertRaises(RegistryError) as e:
            reg.promote("ood_guard", v2["version"])
        self.assertIn("register them first", str(e.exception))

    def test_materialize_and_manual_gate(self):
        _write(os.path.join(self.ckpt, "road_shield_detector.onnx"), "coco")
        reg = self._reg()
        reg.bootstrap()
        out = reg.materialize("coco_detector", 1, os.path.join(self.tmp, "m"))
        self.assertTrue(os.path.exists(os.path.join(out, "road_shield_detector.onnx")))
        self.assertTrue(reg.gate("coco_detector", 1)["manual"])


class RetrainTest(_Tmp):
    def test_candidate_is_registered_production_restored_and_collateral_undone(self):
        import mlops.retrain as rt
        from mlops.registry import Registry
        _write(os.path.join(self.ckpt, "imu_shock_model.joblib"), "old")
        _write(os.path.join(self.ckpt, "imu_shock_report.json"),
               {"held_out_validation_accuracy": 0.8, "per_class_report": {"macro avg": {"f1-score": 0.70},
                                                                         "Pothole Impact": {"recall": 0.9}}})
        _write(os.path.join(self.ckpt, "pci_model.joblib"), "pci-old")
        reg = Registry(store=self.store, ckpt_dir=self.ckpt)

        def fake_train(better):
            def run(cmd):
                _write(os.path.join(self.ckpt, "imu_shock_model.joblib"), f"new-{better}")
                _write(os.path.join(self.ckpt, "imu_shock_report.json"),
                       {"held_out_validation_accuracy": 0.85,
                        "per_class_report": {"macro avg": {"f1-score": 0.75 if better else 0.60},
                                             "Pothole Impact": {"recall": 0.92}}})
                _write(os.path.join(self.ckpt, "pci_model.joblib"), "pci-touched")   # collateral
                return type("P", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
            return run
        if True:
            res = rt.retrain("imu_classifier", reg, runner=fake_train(True))
            self.assertEqual(res["outcome"], "staged", res)
            with open(os.path.join(self.ckpt, "imu_shock_model.joblib")) as fh:
                self.assertEqual(fh.read(), "old", "production keeps serving until promotion")
            with open(os.path.join(self.ckpt, "pci_model.joblib")) as fh:
                self.assertEqual(fh.read(), "pci-old", "files of other models are put back")
            v = reg.get("imu_classifier", res["candidate"]["version"])
            self.assertEqual(v["stage"], "staging")
            self.assertEqual(v["metrics"]["held_out_macro_f1"], 0.75)
            self.assertEqual(v["run_id"], res["run_id"])

            res2 = rt.retrain("imu_classifier", reg, runner=fake_train(False))
            self.assertEqual(res2["outcome"], "gate_failed")

            res3 = rt.retrain("imu_classifier", reg, promote=True, runner=fake_train(True))
            self.assertIn(res3["outcome"], ("promoted", "no_change"))

    def test_crash_mid_training_restores_and_runtime_files_are_untouched(self):
        import mlops.retrain as rt
        from mlops.registry import Registry
        _write(os.path.join(self.ckpt, "imu_shock_model.joblib"), "good")
        _write(os.path.join(self.ckpt, "imu_shock_report.json"), {"held_out_validation_accuracy": 0.8})
        reg = Registry(store=self.store, ckpt_dir=self.ckpt)

        def crash(cmd):
            _write(os.path.join(self.ckpt, "imu_shock_model.joblib"), "PARTIAL")
            _write(os.path.join(self.ckpt, "al_images", "AL-0000000001.jpg"), "server wrote this meanwhile")
            raise TimeoutError("6 h")
        res = rt.retrain("imu_classifier", reg, runner=crash)
        self.assertEqual(res["outcome"], "training_failed")
        with open(os.path.join(self.ckpt, "imu_shock_model.joblib")) as fh:
            self.assertEqual(fh.read(), "good")
        self.assertTrue(os.path.exists(os.path.join(self.ckpt, "al_images", "AL-0000000001.jpg")),
                        "the server's own files in checkpoints/ are not model files")

    def test_gpu_models_explain_instead_of_training(self):
        from mlops.registry import Registry, RegistryError
        from mlops.retrain import retrain
        with self.assertRaises(RegistryError):
            retrain("vision_classifier", Registry(store=self.store, ckpt_dir=self.ckpt))


class MonitorTest(_Tmp):
    def _ref(self):
        rng = np.random.default_rng(0)
        from training.train_ood_guard import histogram_reference
        ref = {"built_from": "test", "features": {"brightness": histogram_reference(rng.normal(120, 20, 2000)),
                                                 "top_confidence": histogram_reference(rng.uniform(0.5, 1, 2000))},
               "class_mix": {"Normal Road / Sound Pavement": 0.6, "Pothole Cavity": 0.4}}
        p = os.path.join(self.tmp, "ref.json")
        _write(p, ref)
        return p

    def _result(self, brightness, cls="Normal Road / Sound Pavement", conf=0.9):
        return {"frame_classification": {"class_name": cls, "confidence": conf},
                "input_check": {"verdict": "ok", "novelty_score": 5.0,
                                "quality": {"brightness": brightness, "contrast": 30, "sharpness": 500,
                                            "clipped_fraction": 0.0}}}

    def test_stable_then_drift_and_alert(self):
        from mlops.monitor import Monitor
        rng = np.random.default_rng(1)
        sent = []

        class A:
            def notify(self, kind, key, text, data=None):
                sent.append(kind)
        clock = [time.time()]
        m = Monitor(os.path.join(self.tmp, "m.db"), self._ref(), alerts=A(), clock=lambda: clock[0])
        self.assertEqual(m.drift(24)["status"], "not enough data")
        for i in range(60):
            cls = "Pothole Cavity" if i % 5 < 2 else "Normal Road / Sound Pavement"
            m.record("api", self._result(float(rng.normal(120, 20)), cls, float(rng.uniform(.5, 1))), latency_ms=100 + i)
        d = m.drift(24)
        self.assertEqual(d["features"]["brightness"]["status"], "stable", d["features"])
        self.assertEqual(d["status"], "stable")
        self.assertEqual(d["latency_ms"]["p50"], 129.5)
        clock[0] += 2 * 86400
        for i in range(80):                                  # night-time camera: everything dark
            m.record("fleet", self._result(float(rng.normal(35, 8)), "Pothole Cavity", 0.55), latency_ms=90)
        d = m.drift(24)
        self.assertEqual(d["features"]["brightness"]["status"], "drift")
        self.assertEqual(d["status"], "drift")
        m._last_status = None
        m.check_and_alert()                                  # normally run in the background every 50 records
        self.assertIn("model_drift", sent)
        self.assertEqual(d["by_source"], {"fleet": 80})

    def test_rebaseline_makes_production_the_reference(self):
        from mlops.monitor import Monitor
        rng = np.random.default_rng(2)
        m = Monitor(os.path.join(self.tmp, "m.db"), self._ref())
        with self.assertRaises(ValueError):
            m.rebaseline(24)
        for i in range(260):                                   # this city's cameras are darker than the datasets
            m.record("fleet", self._result(float(rng.normal(70, 10))), latency_ms=80)
        self.assertEqual(m.drift(24)["features"]["brightness"]["status"], "drift")
        ref = m.rebaseline(24, actor="op")
        self.assertIn("260 production photographs", ref["built_from"])
        d = m.drift(24)
        self.assertEqual(d["reference_source"], "production")
        self.assertEqual(d["features"]["brightness"]["status"], "stable")
        again = Monitor(os.path.join(self.tmp, "m.db"), self._ref())
        self.assertEqual(again.reference["source"], "production", "kept across restarts")
        again.reset_reference()
        self.assertEqual(again.reference["source"], "training")

    def test_fleet_events_and_shadow_summary(self):
        from mlops.monitor import Monitor
        m = Monitor(os.path.join(self.tmp, "m.db"), self._ref())
        m.record_fleet_event("B1", "defect", {"cls": "Pothole Cavity", "conf": 0.8})
        m.record_fleet_event("B1", "cam", {"cls": "Pothole Cavity", "conf": 0.6})
        m.record_fleet_event("B1", "pos", {})
        m.record_fleet_event("B1", "defect", {"cls": "x", "conf": float("nan")})
        json.dumps(m.drift(24), allow_nan=False)
        f = m.drift(24)["fleet_events"]
        self.assertEqual((f["count"], f["mean_confidence"]), (3, 0.7))
        for a, b in (("x", "x"), ("x", "y"), ("x", "x"), ("x", "x")):
            m.record_shadow("vision_classifier", 1, 2, a, b, 0.9, 0.8, 12.0)
        s = m.shadow_summary("vision_classifier", 2)
        self.assertEqual((s["compared"], s["agreement"]), (4, 0.75))
        self.assertEqual(s["disagreements"], {"x -> y": 1})


class ActiveLearningTest(_Tmp):
    def _res(self, probs, distress=False, scene=None):
        names = ["Normal Road / Sound Pavement", "Crack (Longitudinal / Transverse / Alligator)", "Pothole Cavity"]
        top = int(np.argmax(probs))
        return {"frame_classification": {"class_name": names[top], "confidence": probs[top],
                                         "probabilities": dict(zip(names, probs))},
                "is_distress": distress, "primary_distress": {}, "scene_objects": scene or [],
                "scene_summary": {"available": True}}

    def test_scores_dedup_privacy_label_export(self):
        from mlops.active_learning import ActiveLearningQueue, acquisition
        self.assertLess(acquisition(self._res([0.96, 0.02, 0.02]))[0], 0.35)
        s, why = acquisition(self._res([0.05, 0.05, 0.9]))
        self.assertGreaterEqual(s, 0.3)
        self.assertTrue(any("no region measured" in w for w in why))
        q = ActiveLearningQueue(os.path.join(self.tmp, "al.db"), os.path.join(self.tmp, "img"), capacity=3)
        img = (np.random.default_rng(0).uniform(0, 255, (120, 160, 3))).astype(np.uint8)
        unsure = self._res([0.4, 0.35, 0.25])
        self.assertIsNone(q.consider(img, dict(unsure, scene_summary={"available": False}), "api"),
                          "without the people detector nothing can be blurred, so nothing is stored")
        self.assertIsNone(q.consider(img, unsure, "public"))
        self.assertIsNone(q.consider(img, unsure, "dataset"))
        self.assertIsNone(q.consider(img, unsure, "citizen"), "citizen photographs are never kept")
        self.assertIsNone(q.consider(img, dict(unsure, input_check={"verdict": "not_road"}), "api"))
        a = q.consider(img, unsure, "api", projection=[1, 0, 0])
        self.assertIsNotNone(a)
        self.assertIsNone(q.consider(img, unsure, "api", projection=[1, 0.001, 0]), "near-duplicate dropped")
        b = q.consider(img, self._res([0.34, 0.33, 0.33]), "video", projection=[0, 1, 0])
        self.assertIsNotNone(b)
        self.assertTrue(q.image(a).startswith(b"\xff\xd8"))
        self.assertIsNone(q.image("../../etc/passwd"))
        with self.assertRaises(ValueError):
            q.label(a, "banana")
        q.label(a, "Pothole Cavity", actor="op")
        q.label(b, "unusable")
        st = q.stats()
        self.assertEqual((st["pending"], st["labelled"]), (0, 2))
        batch, data, manifest = q.export()
        self.assertEqual([i["label"] for i in manifest["items"]], ["Pothole Cavity"], "unusable is not exported")
        import io
        import zipfile
        names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        self.assertIn(f"{batch}/pothole_cavity/{a}.jpg", names)
        self.assertIn(f"{batch}/manifest.json", names)
        self.assertEqual(q.get(a)["status"], "exported")

    def test_capacity_evicts_lowest_score(self):
        from mlops.active_learning import ActiveLearningQueue
        q = ActiveLearningQueue(os.path.join(self.tmp, "al.db"), os.path.join(self.tmp, "img"), capacity=2)
        img = np.zeros((60, 80, 3), np.uint8) + 100
        low = q.consider(img, self._res([0.6, 0.3, 0.1]), "api", projection=[1, 0, 0])
        mid = q.consider(img, self._res([0.45, 0.35, 0.2]), "api", projection=[0, 1, 0])
        high = q.consider(img, self._res([0.34, 0.33, 0.33]), "api", projection=[0, 0, 1])
        ids = [i["item_id"] for i in q.list()]
        self.assertEqual(set(ids), {mid, high})
        self.assertIsNone(q.get(low))


class TrafficTest(_Tmp):
    def test_flow_needs_moving_bus_and_enough_hours(self):
        from pipeline.traffic import TrafficEstimator, HOURLY_SHARE, VIEW_LENGTH_M, local_hour
        t = TrafficEstimator(os.path.join(self.tmp, "t.db"))
        base = 1_790_000_000
        base -= ((local_hour(base) - 7) % 24) * 3600          # 07:xx local: daytime, when flow is expanded
        o = t.observe(12.97, 77.59, {"Car": 2, "Two-Wheeler": 2}, speed_kmh=20, at=base)
        expect_density = 3.0 / (VIEW_LENGTH_M / 1000)
        self.assertAlmostEqual(o["density_pcu_per_km"], expect_density)
        self.assertAlmostEqual(o["flow_pcu_per_h"], expect_density * 20)
        self.assertEqual(o["daily_pcu_estimate"], round(expect_density * 20 / HOURLY_SHARE[local_hour(base)]))
        stopped = t.observe(12.97, 77.59, {"Car": 5}, speed_kmh=2, at=base)
        self.assertFalse(stopped["counted_for_flow"], "a bus at a stop says nothing about flow")
        est = t.estimate(12.97, 77.59, now=base + 10)
        self.assertFalse(est["usable_for_priority"])
        defect = {"lat": 12.97, "lon": 77.59, "severity_pci": 40}
        t._cache.clear()
        self.assertNotIn("traffic_pcu_per_day", t.enrich(defect))
        for h in range(6):
            t.observe(12.97001, 77.59001, {"Car": 2, "Two-Wheeler": 2}, speed_kmh=20, at=base + h * 2 * 3600)
        night = t.observe(12.97, 77.59, {"Car": 2}, speed_kmh=20, at=base - 5 * 3600)    # 02:xx
        self.assertFalse(night["counted_for_flow"], "night hours are not expanded to a day")
        t._cache.clear()
        t.clock = lambda: base + 30 * 3600
        est = t.estimate(12.97, 77.59)
        self.assertTrue(est["usable_for_priority"], est)
        e = t.enrich(defect)
        self.assertEqual(e["traffic_pcu_per_day"], est["daily_pcu_estimate"])
        from models import priority_index
        self.assertEqual(priority_index.score_defect(e)["traffic_basis"].split("_")[0], "measured")
        with self.assertRaises(ValueError):
            t.observe(12.97, 77.59, {"Car": 1e6})

    def test_counts_from_audit_needs_detector_and_respects_view_length(self):
        from pipeline.traffic import counts_from_audit
        self.assertIsNone(counts_from_audit({"scene_summary": {"available": False}}), "no detector is not zero traffic")
        a = {"scene_summary": {"available": True},
             "all_vehicles": [{"vehicle_type": "Car", "distance_meters": 10},
                              {"vehicle_type": "Car", "distance_meters": 80},
                              {"vehicle_type": "City Bus", "distance_meters": 20}]}
        self.assertEqual(counts_from_audit(a), {"Car": 1, "City Bus": 1})


class EdgeTrafficEventTest(_Tmp):
    def test_bus_agent_sends_one_traffic_event_per_cell_and_server_validates(self):
        from edge.bus_agent import BusAgent, traffic_event
        out = []
        q = type("Q", (), {"put": lambda self, ev: out.append(ev), "stats": lambda self: {"queued": 0}})()
        agent = BusAgent("B1", q, None, None, None)
        audit = {"scene_summary": {"available": True},
                 "all_vehicles": [{"vehicle_type": "Car", "distance_meters": 10}]}
        for i in range(3):
            agent._count_traffic(audit, {"lat": 12.9705, "lon": 77.5905, "speed_kmh": 30}, 100 + i, [])
        self.assertEqual(out, [])
        agent._count_traffic(audit, {"lat": 12.9725, "lon": 77.5905, "speed_kmh": 30}, 104, [])  # next cell
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["t"], out[0]["n"], out[0]["veh"], out[0]["spd"]), ("trf", 3, {"Car": 1.0}, 30.0))
        self.assertIsNone(traffic_event({"frames": 1}, 0))
        agent._count_traffic({"scene_summary": {"available": False}}, {"lat": 1, "lon": 1}, 200, [])

        from edge import crypto
        from pipeline.edge_ingest import EdgeIngest
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        key = crypto.new_key_hex()
        seen = []
        ei = EdgeIngest(os.path.join(self.tmp, "e.db"), FleetDeduplicationEngine(), key=key)
        ei.event_listeners.append(lambda bus, kind, ev, at: seen.append(kind))
        k = crypto.load_key(key)
        r = ei.ingest([crypto.pack(out[0], "B1", 1, k),
                       crypto.pack({"t": "trf", "lat": 1, "lon": 1, "veh": {"Car": 1e9}}, "B1", 2, k)])
        self.assertEqual([x["status"] for x in r["results"]], ["applied", "invalid_event"])
        self.assertEqual(seen, ["trf"])


class InputGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.ood_guard import OODGuard
        cls.g = OODGuard()
        if not cls.g.is_ready:
            raise unittest.SkipTest(cls.g.load_error)

    def test_road_photo_passes_and_dark_or_noise_does_not(self):
        import glob
        from PIL import Image
        paths = sorted(glob.glob(os.path.join(ROOT, "datasets", "02_kaggle_pothole_600", "real_images", "*.jpg")))
        if not paths:
            self.skipTest("no road photographs on disk")
        ok = 0
        for p in paths[:10]:
            img = np.asarray(Image.open(p).convert("RGB").resize((640, 480)))
            ok += self.g.check(img)["verdict"] in ("ok", "unusual")
        self.assertGreaterEqual(ok, 9)
        dark = (np.asarray(Image.open(paths[0]).convert("RGB").resize((640, 480))) * 0.1).astype(np.uint8)
        self.assertEqual(self.g.check(dark)["verdict"], "poor_quality")
        self.assertIn("too_dark", self.g.check(dark)["flags"])
        r = self.g.check(np.zeros((480, 640, 3), np.uint8) + 128, with_projection=True)
        self.assertNotEqual(r["verdict"], "ok")
        self.assertEqual(len(r["projection"]), 64)

    def test_report_numbers_meet_the_gate(self):
        rep = self.g.report["test"]
        self.assertGreaterEqual(rep["combined"]["auroc"], 0.9)
        self.assertLessEqual(rep["refusals"]["road_photos_refused_rate"], 0.03)


if __name__ == "__main__":
    unittest.main()
