"""Priority Index, keyed work-order seals, the bus agent and its encrypted queue, the live feed, and the
measurement scripts (benchmark, redactor recall, calibration)."""

import glob
import json
import os
import random
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock

import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class PriorityIndex(unittest.TestCase):
    def test_terms_and_weights(self):
        from models import priority_index as pi
        r = pi.score(pci=40, area_m2=1.25, depth_cm=10, traffic_pcu_per_day=40000, weights=(0.5, 0.2, 0.3))
        self.assertEqual(r["components"], {"severity": 60.0, "volume": 50.0, "traffic": 100.0})
        self.assertAlmostEqual(r["priority_index"], 0.5 * 60 + 0.2 * 50 + 0.3 * 100)
        self.assertEqual(r["missing"], [])
        self.assertEqual(r["band"], "P1")

    def test_missing_terms_share_their_weight(self):
        from models import priority_index as pi
        r = pi.score(pci=20, area_m2=1.0, weights=(0.5, 0.2, 0.3))
        self.assertEqual(r["missing"], ["volume", "traffic"])
        self.assertEqual(r["weights_applied"], {"severity": 1.0})
        self.assertEqual(r["priority_index"], 80.0)

    def test_bus_count_is_a_labelled_proxy(self):
        from models import priority_index as pi
        r = pi.score(pci=50, reporting_buses=["A", "B", "B"], weights=(0.5, 0.2, 0.3))
        self.assertEqual(r["traffic_basis"], "fleet_proxy_distinct_buses")
        self.assertEqual(r["components"]["traffic"], 10.0)

    def test_weights_are_validated(self):
        from models import priority_index as pi
        for bad in ("0.5,0.5", "0.6,0.6,0.6", "-0.1,0.6,0.5", (1, 2, 3)):
            with self.assertRaises(ValueError):
                pi.parse_weights(bad)
        self.assertEqual(pi.parse_weights("0.4, 0.3, 0.3"), (0.4, 0.3, 0.3))
        with mock.patch.dict(os.environ, {"ROAD_SHIELD_PI_WEIGHTS": "0.6,0.2,0.2"}):
            self.assertEqual(pi.default_weights(), (0.6, 0.2, 0.2))
        with mock.patch.dict(os.environ, {"ROAD_SHIELD_PI_WEIGHTS": "nonsense"}):
            self.assertEqual(pi.default_weights(), pi.DEFAULT_WEIGHTS)
        with self.assertRaises(ValueError):
            pi.score(pci=120)

    def test_rank_and_stability(self):
        from models import priority_index as pi
        defects = [
            {"defect_id": "A", "severity_pci": 20, "area_m2": 2, "depth_cm": 8, "reporting_buses": ["1"] * 1},
            {"defect_id": "B", "severity_pci": 70, "area_m2": 0.2, "depth_cm": 2, "reporting_buses": ["1", "2"]},
            {"defect_id": "C", "severity_pci": 45, "area_m2": 1, "depth_cm": 5, "reporting_buses": ["1", "2", "3"]},
            {"defect_id": "D", "severity_pci": None},
        ]
        ranked = pi.rank(defects)
        self.assertEqual([r["defect_id"] for r in ranked], ["A", "C", "B"])
        self.assertEqual([r["priority"]["rank"] for r in ranked], [1, 2, 3])
        st = pi.rank_stability(defects)
        self.assertEqual(st["defects"], 3)
        self.assertTrue(-1.0 <= st["min_kendall_tau"] <= 1.0)
        self.assertEqual(len(st["trials"]), 6)
        broken = defects + [{"defect_id": "E", "severity_pci": 150}]
        self.assertEqual([r["defect_id"] for r in pi.rank(broken)], ["A", "C", "B"], "one bad row is skipped")
        for bad in (-40, "ABC", float("nan"), True):
            with self.assertRaises(ValueError):
                pi.score(pci=50, reporting_buses=bad)

    def test_kendall_tau(self):
        from models.priority_index import kendall_tau
        self.assertEqual(kendall_tau("abcd", "abcd"), 1.0)
        self.assertEqual(kendall_tau("abcd", "dcba"), -1.0)
        self.assertEqual(kendall_tau("a", "a"), 1.0)


class KeyedSeal(unittest.TestCase):
    ORDER = dict(corridor_id="NH44", latitude=12.9, longitude=77.6, distress_class="Pothole Cavity",
                 area_sqm=1.0, depth_cm=5.0, pci_score=40)

    def test_plain_sha256_by_default(self):
        from models.morth_dispatch_agent import MoRTHDispatchAgent, check_seal
        with mock.patch.dict(os.environ, {"ROAD_SHIELD_SEAL_KEY": ""}):
            wo = MoRTHDispatchAgent().generate_work_order(**self.ORDER)
            self.assertEqual(wo["seal_algorithm"], "SHA-256")
            self.assertEqual(check_seal(wo), "SEAL_VERIFIED_AUTHENTIC")

    def test_hmac_needs_the_key_and_catches_edits(self):
        from models.morth_dispatch_agent import MoRTHDispatchAgent, check_seal
        a = MoRTHDispatchAgent(seal_key="municipal-secret")
        wo = a.generate_work_order(**self.ORDER)
        self.assertEqual(wo["seal_algorithm"], "HMAC-SHA256")
        self.assertTrue(a.verify_work_order_seal(wo))
        self.assertEqual(check_seal(wo, key=""), "KEY_REQUIRED_TO_VERIFY")
        self.assertEqual(check_seal(wo, key="wrong"), "CORRUPTED_OR_TAMPERED")
        edited = dict(wo, required_mass_tonnes=99.0)
        self.assertEqual(a.check_work_order_seal(edited), "CORRUPTED_OR_TAMPERED")

    def test_downgrade_to_unkeyed_is_refused(self):
        import hashlib
        from models.morth_dispatch_agent import MoRTHDispatchAgent, SEAL_FIELD, _canonical
        a = MoRTHDispatchAgent(seal_key="municipal-secret")
        wo = a.generate_work_order(**self.ORDER)
        forged = {k: v for k, v in wo.items() if k != SEAL_FIELD}
        forged["seal_algorithm"] = "SHA-256"
        forged["required_mass_tonnes"] = 99.0
        forged[SEAL_FIELD] = hashlib.sha256(_canonical(forged)).hexdigest()
        self.assertEqual(a.check_work_order_seal(forged), "UNKEYED_SEAL_REJECTED")

    def test_orders_sealed_before_the_algorithm_field_still_verify(self):
        import hashlib
        from models.morth_dispatch_agent import MoRTHDispatchAgent, SEAL_FIELD, _canonical
        old = {"work_order_id": "MORTH-WO-1", "pavement_pci": 40}
        old[SEAL_FIELD] = hashlib.sha256(_canonical(old)).hexdigest()
        self.assertTrue(MoRTHDispatchAgent(seal_key="").verify_work_order_seal(old))


class FleetCrypto(unittest.TestCase):
    def setUp(self):
        from edge import crypto
        self.c = crypto
        self.key = crypto.load_key("fleet-test")

    def test_roundtrip_and_binding(self):
        env = self.c.pack({"t": "pos", "lat": 1.0, "lon": 2.0}, "BUS-1", 7, self.key, epoch="ab12")
        self.assertEqual(self.c.unpack(env, self.key), ("BUS-1", "ab12", 7, {"lat": 1.0, "lon": 2.0, "t": "pos"}))
        for change in ({"bus": "BUS-2"}, {"seq": 8}, {"ep": "ab13"}, {"bus": "<b>x</b>"}):
            with self.assertRaises(self.c.PacketError):
                self.c.unpack(dict(env, **change), self.key)
        with self.assertRaises(self.c.PacketError):
            self.c.unpack(env, self.c.load_key("other"))
        ct = bytearray(__import__("base64").b64decode(env["ct"]))
        ct[0] ^= 1
        with self.assertRaises(self.c.PacketError):
            self.c.unpack(dict(env, ct=__import__("base64").b64encode(bytes(ct)).decode()), self.key)
        with self.assertRaises(self.c.PacketError):
            self.c.unpack({"v": 1, "bus": "x"}, self.key)

    def test_size_limit_and_keys(self):
        with self.assertRaises(self.c.PacketError):
            self.c.pack({"t": "defect", "pad": "x" * 2000}, "B", 1, self.key)
        self.assertEqual(len(self.c.load_key("ab" * 32)), 32)
        self.assertEqual(self.c.load_key("ab" * 32), bytes.fromhex("ab" * 32))
        self.assertEqual(self.c.load_key("pass"), self.c.load_key("pass"))
        with mock.patch.dict(os.environ, {"ROAD_SHIELD_FLEET_KEY": ""}):
            self.assertIsNone(self.c.load_key())


class StoreForward(unittest.TestCase):
    def setUp(self):
        from edge import crypto
        from edge.store_forward import StoreAndForward
        self.d = tempfile.mkdtemp()
        self.key = crypto.load_key("q-test")
        self.path = os.path.join(self.d, "q.db")
        self.q = StoreAndForward(self.path, "BUS-9", self.key, max_rows=5)

    def tearDown(self):
        self.q.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_in_order_delivery_and_backoff(self):
        from edge import crypto
        for i in range(3):
            self.assertEqual(self.q.put({"t": "defect", "i": i, "lat": 1, "lon": 1}), i + 1)
        got = []
        self.assertEqual(self.q.flush(lambda envs: (got.extend(envs), 1)[1], now=0), 1)
        self.assertEqual(self.q.stats()["queued"], 2)
        self.assertEqual(self.q.flush(lambda envs: 2, now=1), 0)          # backing off after the partial batch
        self.assertEqual(self.q.flush(lambda envs: len(envs), now=100), 2)
        self.assertEqual(self.q.stats()["queued"], 0)
        self.assertEqual(crypto.unpack(got[0], self.key)[2], 1)
        self.assertEqual(crypto.unpack(got[0], self.key)[1], self.q.epoch)

    def test_errors_keep_the_queue(self):
        self.q.put({"t": "pos", "lat": 1, "lon": 1})

        def boom(envs):
            raise OSError("no network")
        self.assertEqual(self.q.flush(boom, now=0), 0)
        self.assertEqual(self.q.stats()["queued"], 1)
        self.assertEqual(self.q.stats()["consecutive_failures"], 1)

    def test_bounded_and_encrypted_at_rest(self):
        self.q.put({"t": "defect", "secret": "pothole-at-home", "lat": 1, "lon": 1})
        for i in range(6):
            self.q.put({"t": "pos", "lat": i, "lon": i})
        st = self.q.stats()
        self.assertEqual(st["queued"], 5)
        self.assertEqual(st["dropped"], 2)
        con = sqlite3.connect(self.path)
        kinds = [k for (k,) in con.execute("SELECT kind FROM outbox ORDER BY seq")]
        con.close()
        self.assertIn("defect", kinds, "positions are dropped before events")
        with open(self.path, "rb") as fh:
            self.assertNotIn(b"pothole-at-home", fh.read())

    def test_sequence_and_epoch_survive_restart(self):
        from edge.store_forward import StoreAndForward
        self.q.put({"t": "pos", "lat": 1, "lon": 1})
        epoch = self.q.epoch
        self.q.close()
        self.q = StoreAndForward(self.path, "BUS-9", self.key)
        self.assertEqual(self.q.put({"t": "pos", "lat": 1, "lon": 1}), 2)
        self.assertEqual(self.q.epoch, epoch)
        fresh = StoreAndForward(os.path.join(self.d, "new_card.db"), "BUS-9", self.key)
        self.assertNotEqual(fresh.epoch, epoch, "a new SD card is a new epoch")
        fresh.close()

    def test_refused_packet_goes_to_dead_letter_but_outages_never_do(self):
        from edge.store_forward import MAX_REFUSALS
        self.q.put({"t": "defect", "lat": 1, "lon": 1})
        self.q.put({"t": "pos", "lat": 1, "lon": 1})

        def outage(envs):
            raise OSError("no network")
        for i in range(MAX_REFUSALS + 3):
            self.q.flush(outage, now=1000.0 * (i + 1))
        self.assertEqual(self.q.stats()["dead_letter"], 0)
        sent = []

        def refuse_first(envs):
            sent.append(len(envs))
            return 0 if len(envs) == 2 else len(envs)
        for i in range(MAX_REFUSALS):
            self.q.flush(refuse_first, now=10 ** 6 * (i + 1))
        st = self.q.stats()
        self.assertEqual(st["dead_letter"], 1)
        self.assertEqual(st["queued"], 0, "the queue moves on after the bad packet")


class EdgeIngestTest(unittest.TestCase):
    def setUp(self):
        from edge import crypto
        from pipeline.edge_ingest import EdgeIngest
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        from pipeline.live_events import LiveEvents
        self.c = crypto
        self.d = tempfile.mkdtemp()
        self.key = crypto.load_key("ingest-test")
        self.dedup = FleetDeduplicationEngine(store=None)
        self.events = LiveEvents()
        self.ing = EdgeIngest(os.path.join(self.d, "e.db"), self.dedup, self.events, key="ingest-test")

    def tearDown(self):
        self.ing._db.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def env(self, ev, seq, bus="B1"):
        return self.c.pack(ev, bus, seq, self.key)

    def test_new_epoch_is_not_a_replay(self):
        ev = {"t": "pos", "lat": 12.9, "lon": 77.6}
        self.assertEqual(self.ing.ingest([self.c.pack(ev, "B1", 5, self.key, epoch="aa")])["accepted_in_order"], 1)
        r = self.ing.ingest([self.c.pack(ev, "B1", 1, self.key, epoch="bb")])
        self.assertEqual(r["results"][0]["status"], "applied")
        r = self.ing.ingest([self.c.pack(ev, "B1", 3, self.key, epoch="aa")])
        self.assertEqual(r["results"][0]["status"], "duplicate")

    def test_fields_are_typed_before_reaching_the_map(self):
        r = self.ing.ingest([self.env({"t": "pos", "lat": 12.9, "lon": 77.6, "q": "<img src=x onerror=alert(1)>",
                                       "spd": "fast"}, 1),
                             self.env({"t": "shock", "lat": 12.9, "lon": 77.6, "p": "<b>", "dz": float("inf"),
                                       "why": "<script>" * 30}, 2)])
        self.assertEqual(r["accepted_in_order"], 2)
        p = self.ing.live_positions()[0]
        self.assertIsNone(p["queued_on_bus"])
        self.assertIsNone(p["speed_kmh"])
        s = self.ing.recent_shocks()[0]
        self.assertIsNone(s["p"])
        self.assertIsNone(s["dz"])
        self.assertLessEqual(len(s["camera_unusable_because"]), 60)

    def test_bad_pci_never_reaches_the_ledger(self):
        for pci in (150, -1, float("nan"), "x"):
            r = self.ing.ingest([self.env({"t": "defect", "lat": 12.9, "lon": 77.6, "cls": "Pothole Cavity",
                                           "pci": pci, "area": 1.0}, self.ing._last_seq("B1", "0") + 1)])
            self.assertEqual(r["results"][0]["status"], "invalid_event", pci)
        self.assertEqual(self.dedup.get_all_deduplicated_defects(), [])

    def test_apply_dedupe_replay(self):
        defect = {"t": "defect", "lat": 12.9, "lon": 77.6, "cls": "Pothole Cavity", "pci": 30, "area": 1.0, "depth": 5}
        with mock.patch("services.google_maps_service.google_maps_service.reverse_geocode", side_effect=Exception):
            r = self.ing.ingest([self.env(defect, 1), self.env({"t": "pos", "lat": 12.9, "lon": 77.6}, 2),
                                 self.env({"t": "shock", "lat": 12.9, "lon": 77.6, "p": 0.9, "dz": 7}, 3)])
        self.assertEqual(r["accepted_in_order"], 3)
        self.assertEqual(len(self.dedup.get_all_deduplicated_defects()), 1)
        self.assertEqual(self.dedup.get_all_deduplicated_defects()[0]["depth_cm"], 5.0)
        r2 = self.ing.ingest([self.env(defect, 1)])
        self.assertEqual(r2["results"][0]["status"], "duplicate")
        self.assertEqual(self.dedup.total_reports_ingested, 1, "a replayed packet is not applied twice")
        self.assertEqual(len(self.ing.live_positions()), 1)
        self.assertEqual(len(self.ing.shocks), 1)
        self.assertEqual([e["type"] for e in self.events.recent()], ["defect", "bus_position", "shock"])

    def test_stops_at_forgery_and_accepts_bad_content(self):
        bad = dict(self.env({"t": "pos", "lat": 1, "lon": 1}, 1), bus="B2")
        r = self.ing.ingest([bad, self.env({"t": "pos", "lat": 1, "lon": 1}, 2)])
        self.assertEqual(r["accepted_in_order"], 0)
        self.assertEqual(len(r["results"]), 1)
        r = self.ing.ingest([self.env({"t": "teleport", "lat": 1, "lon": 1}, 5)])
        self.assertEqual(r["results"][0]["status"], "invalid_event")
        self.assertEqual(r["accepted_in_order"], 1, "an authentic but unusable event must not block the queue")

    def test_needs_a_key(self):
        from pipeline.edge_ingest import EdgeIngest
        with mock.patch.dict(os.environ, {"ROAD_SHIELD_FLEET_KEY": ""}):
            ing = EdgeIngest(":memory:", self.dedup)
        self.assertFalse(ing.configured)
        with self.assertRaises(RuntimeError):
            ing.ingest([{}])
        ing._db.close()


class LiveFeed(unittest.TestCase):
    def test_publish_subscribe_resume_and_caps(self):
        from pipeline.live_events import LiveEvents, sse_frame
        bus = LiveEvents(recent=3, per_subscriber=2, max_subscribers=1)
        q = bus.subscribe()
        self.assertIsNone(bus.subscribe(), "over capacity")
        for i in range(4):
            bus.publish("defect", {"i": i})
        got = [q.get_nowait()["data"]["i"] for _ in range(q.qsize())]
        self.assertEqual(got, [2, 3], "a slow subscriber loses its oldest events")
        bus.unsubscribe(q)
        q2 = bus.subscribe(last_event_id=2)
        self.assertEqual([q2.get_nowait()["id"] for _ in range(q2.qsize())], [3, 4])
        frame = sse_frame(bus.recent()[-1]).decode()
        self.assertTrue(frame.startswith("id: 4\nevent: defect\ndata: "))
        self.assertTrue(frame.endswith("\n\n"))


class FrameQuality(unittest.TestCase):
    def _road(self, seed=0):
        rng = np.random.default_rng(seed)
        img = np.clip(120 + rng.normal(0, 30, (480, 640, 3)), 0, 255).astype(np.uint8)
        img[300:320, 100:500] = 40
        return img

    def test_detects_blur_dark_and_glare(self):
        import cv2
        from edge import frame_quality as fq
        img = self._road()
        self.assertTrue(fq.assess(img)["usable"])
        self.assertIn("blurred", fq.assess(cv2.GaussianBlur(img, (41, 41), 0))["reasons"])
        self.assertIn("too_dark", fq.assess((img * 0.1).astype(np.uint8))["reasons"])
        self.assertIn("overexposed", fq.assess(np.full_like(img, 250))["reasons"])

    def test_tune_from_good_frames(self):
        from edge import frame_quality as fq
        t = fq.tune([self._road(s) for s in range(10)])
        self.assertLess(t["min_brightness"], t["max_brightness"])

    def test_pass_rate_on_project_photographs(self):
        from PIL import Image
        from edge import frame_quality as fq
        files = glob.glob(os.path.join(ENGINE_ROOT, "datasets", "0[1-3]_*", "**", "*.jpg"), recursive=True)
        if len(files) < 200:
            self.skipTest("road photographs not on disk")
        random.Random(0).shuffle(files)
        bad = 0
        for f in files[:200]:
            with Image.open(f) as im:
                bad += not fq.assess(np.asarray(im.convert("RGB")))["usable"]
        self.assertLessEqual(bad, 6, f"{bad}/200 good road photographs rejected")


class Sensors(unittest.TestCase):
    def test_nmea(self):
        from edge.sensors import parse_nmea

        def nmea(body):
            cs = 0
            for ch in body:
                cs ^= ord(ch)
            return f"${body}*{cs:02X}"
        rmc = parse_nmea(nmea("GPRMC,123519,A,1258.3354,N,07735.7650,E,13.5,84.4,230394,003.1,W"))
        self.assertAlmostEqual(rmc["lat"], 12 + 58.3354 / 60, places=6)
        self.assertAlmostEqual(rmc["lon"], 77 + 35.7650 / 60, places=6)
        self.assertAlmostEqual(rmc["speed_kmh"], 13.5 * 1.852, places=3)
        self.assertIsNone(parse_nmea(nmea("GPRMC,123519,V,,,,,,,230394,,")), "no fix")
        gga = parse_nmea(nmea("GNGGA,123519,1258.3354,S,07735.7650,W,1,08,0.9,545.4,M,46.9,M,,"))
        self.assertLess(gga["lat"], 0)
        self.assertLess(gga["lon"], 0)
        self.assertEqual(gga["satellites"], 8)
        self.assertIsNone(parse_nmea("$GPRMC,123519,A,1258.3354,N,07735.7650,E,13.5,84.4,230394,003.1,W*00"))

    def test_replay_sources(self):
        from edge.sensors import ReplayGps, ReplayImu
        d = tempfile.mkdtemp()
        try:
            with open(os.path.join(d, "r.csv"), "w") as fh:
                fh.write("t,lat,lon,speed_kmh\n0,12.0,77.0,20\n10,12.0,77.001,20\n")
            g = ReplayGps(os.path.join(d, "r.csv"))
            g.fix(now=100.0)
            f = g.fix(now=105.0)
            self.assertAlmostEqual(f["lon"], 77.0005, places=6)
            with open(os.path.join(d, "i.csv"), "w") as fh:
                fh.write("Time,Ax,Ay,Az\n" + "".join(f"{i},0,0,{9.8 + (i == 150)}\n" for i in range(300)))
            imu = ReplayImu(os.path.join(d, "i.csv"))
            self.assertEqual(imu.window(100, now=0.0).shape, (100, 3))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class BusAgentTest(unittest.TestCase):
    AUDIT = {"is_distress": True,
             "primary_distress": {"class_id": 2, "is_distress": True, "class_name": "Pothole Cavity",
                                  "surface_area_m2": 0.8, "depth_cm": 6.0, "confidence": 0.91},
             "bayesian_sensor_fusion": {"verdict": "CONFIRMED_POTHOLE", "imu_evidence_source": "real_sensor_window"},
             "astm_d6433_pci": {"pci_score": 38.0}}

    def test_events(self):
        from edge import crypto
        from edge.bus_agent import defect_event, position_event, shock_event
        fix = {"lat": 12.9716, "lon": 77.5946, "speed_kmh": 22.0}
        ev = defect_event(self.AUDIT, fix, 1000)
        self.assertEqual((ev["cls"], ev["pci"], ev["area"], ev["imu"]), ("Pothole Cavity", 38.0, 0.8, True))
        rejected = dict(self.AUDIT, bayesian_sensor_fusion={"verdict": "REJECTED_OPTICAL_FALSE_ALARM"})
        self.assertIsNone(defect_event(rejected, fix, 1000))
        self.assertIsNone(defect_event(dict(self.AUDIT, is_distress=False), fix, 1000))
        sign = dict(self.AUDIT, primary_distress=dict(self.AUDIT["primary_distress"], class_id=6))
        self.assertIsNone(defect_event(sign, fix, 1000), "a damaged sign has no area to repair")
        strong = {"available": True, "pothole_shock_probability": 0.9, "peak_delta_z_ms2": 2.0}
        self.assertEqual(shock_event(strong, fix, 1000, ["too_dark"])["why"], "too_dark")
        self.assertIsNone(shock_event(dict(strong, pothole_shock_probability=0.2), fix, 1000, []))
        for e in (ev, shock_event(strong, fix, 1000, ["blurred"]), position_event(fix, 1000, 3)):
            crypto.pack(e, "BUS-LONG-IDENTIFIER-0001", 10 ** 9, crypto.load_key("k"))

    def test_tick_paths(self):
        from edge import crypto
        from edge.bus_agent import BusAgent
        from edge.store_forward import StoreAndForward
        d = tempfile.mkdtemp()
        try:
            q = StoreAndForward(os.path.join(d, "q.db"), "B", crypto.load_key("k"))

            class Cam:
                frame_ = None

                def frame(self):
                    return self.frame_

            class Gps:
                fix_ = {"lat": 12.9, "lon": 77.6, "speed_kmh": 20.0}

                def fix(self, now=None):
                    return self.fix_

            class Imu:
                def window(self, n=100):
                    return np.zeros((100, 3), np.float32)

            class Pipe:
                def audit_image(self_, frame, **kw):
                    return BusAgentTest.AUDIT

                def _run_imu_stage(self_, w):
                    return ({"available": True, "pothole_shock_probability": 0.95, "peak_delta_z_ms2": 8.0,
                             "shock_classification": "Pothole Impact"},)

            cam, gps = Cam(), Gps()
            agent = BusAgent("B", q, cam, Imu(), gps, pipeline=Pipe(), position_every=10)
            rng = np.random.default_rng(0)
            cam.frame_ = np.clip(120 + rng.normal(0, 30, (240, 320, 3)), 0, 255).astype(np.uint8)
            self.assertEqual(agent.tick(now=0), ["pos", "defect"])
            cam.frame_ = np.zeros((240, 320, 3), np.uint8)
            self.assertEqual(agent.tick(now=1), ["shock"])
            gps.fix_ = None
            self.assertEqual(agent.tick(now=2), [])
            self.assertEqual(agent.counters["skipped_no_gps"], 1)
            self.assertEqual(q.stats()["queued"], 3)
            q.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Measurements(unittest.TestCase):
    def test_calibration_maths(self):
        from scripts.measure_calibration import fit_temperature, reliability, softmax, summarise
        rng = np.random.default_rng(0)
        n = 4000
        true_logits = rng.normal(0, 1.5, (n, 3))
        y = np.array([rng.choice(3, p=p) for p in softmax(true_logits)])
        over = true_logits * 3.0                                    # an over-confident model
        t = fit_temperature(over, y)
        self.assertGreater(t, 2.0)
        self.assertLess(t, 4.0)
        before, after = summarise(over, y, 1.0), summarise(over, y, t)
        self.assertEqual(before["accuracy"], after["accuracy"])
        self.assertLess(after["ece"], before["ece"])
        ece, mce, rows = reliability(softmax(true_logits), y)
        self.assertLess(ece, 0.05)
        self.assertEqual(sum(r["count"] for r in rows), n)

    def test_redaction_scoring_helpers(self):
        from scripts.measure_redactor_recall import coco_items, covered_fraction, wider_items
        mask = np.zeros((100, 100), bool)
        mask[0:50, 0:50] = True
        self.assertEqual(covered_fraction((0, 0, 50, 50), mask), 1.0)
        self.assertEqual(covered_fraction((25, 0, 50, 50), mask), 0.5)
        self.assertEqual(covered_fraction((200, 200, 5, 5), mask), 0.0)
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, "wider_face_split"))
            os.makedirs(os.path.join(d, "WIDER_val", "images", "0--Parade"))
            with open(os.path.join(d, "wider_face_split", "wider_face_val_bbx_gt.txt"), "w") as fh:
                fh.write("0--Parade/a.jpg\n2\n1 2 30 40 0 0 0 0 0 0\n5 5 9 9 0 0 0 1 0 0\n"
                         "0--Parade/b.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n")
            items = wider_items(d)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0][1], [(1.0, 2.0, 30.0, 40.0)], "invalid boxes are skipped")
            with open(os.path.join(d, "_annotations.coco.json"), "w") as fh:
                json.dump({"images": [{"id": 1, "file_name": "p.jpg"}, {"id": 2, "file_name": "q.jpg"}],
                           "annotations": [{"image_id": 1, "bbox": [1, 2, 3, 4]}], "categories": []}, fh)
            self.assertEqual(coco_items(d), [(os.path.join(d, "p.jpg"), [(1, 2, 3, 4)])])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_redactor_returns_regions(self):
        from models.privacy_redactor import redact
        img = np.full((200, 200, 3), 128, np.uint8)
        dets = [{"class_name": "person", "bbox_pixels": [20, 20, 60, 120]}]
        _, rep = redact(img, detections=dets, plate_locator=lambda crop: None, return_regions=True)
        self.assertEqual(rep["regions"][0][4], "person_head")
        _, rep = redact(img, detections=dets, plate_locator=lambda crop: None)
        self.assertNotIn("regions", rep)

    def test_benchmark_times_a_model(self):
        from scripts.benchmark_edge import concrete_shape, device_info, time_model
        self.assertEqual(concrete_shape(["batch", 3, "h", "w"], "x.onnx"), [1, 3, 224, 224])
        self.assertEqual(concrete_shape([None, 3, None], "imu_shock_cnn.onnx"), [1, 3, 100])
        path = os.path.join(ENGINE_ROOT, "checkpoints", "imu_shock_cnn.onnx")
        if not os.path.exists(path):
            self.skipTest("no ONNX model on disk")
        r = time_model(path, threads=1, runs=3, warmup=1)
        self.assertGreater(r["median_ms"], 0)
        self.assertEqual(r["precision"], "fp32")
        self.assertIn("onnxruntime", device_info())



class HotspotNeedsTwoSources(unittest.TestCase):
    def test_same_source_twice_is_not_verification(self):
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        e = FleetDeduplicationEngine(store=None)
        args = dict(lat=12.9, lon=77.6, defect_class="Pothole Cavity", severity_pci=40, area_m2=1.0,
                    enrich_location=False)
        e.ingest_fleet_detection("photo-upload", **args)
        r = e.ingest_fleet_detection("photo-upload", **args)
        self.assertEqual(r["confirmations"], 2)
        self.assertFalse(r["is_hotspot"], "one source repeating itself")
        r = e.ingest_fleet_detection("BMTC-201", **args)
        self.assertTrue(r["is_hotspot"])
        with self.assertRaises(ValueError):
            e.ingest_fleet_detection("BMTC-201", **dict(args, severity_pci=float("nan")))


if __name__ == "__main__":
    unittest.main()
