"""Repair lifecycle (issue -> repair -> verified by the fleet), alerts and the GIS export."""

import csv
import io
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _ledger(defects=(("Pothole Cavity", 12.9716, 77.5946, 30, 1.2, 6.0),)):
    from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
    e = FleetDeduplicationEngine(store=None)
    for cls, lat, lon, pci, area, depth in defects:
        e.ingest_fleet_detection("BUS-1", lat, lon, cls, pci, area, enrich_location=False, depth_cm=depth)
    return e


class Lifecycle(unittest.TestCase):
    def setUp(self):
        from models.morth_dispatch_agent import MoRTHDispatchAgent
        from pipeline.live_events import LiveEvents
        from pipeline.works import WorkOrders
        self.d = tempfile.mkdtemp()
        self.clock = Clock()
        self.ledger = _ledger()
        self.events = LiveEvents()
        self.w = WorkOrders(os.path.join(self.d, "w.db"), self.ledger, MoRTHDispatchAgent(seal_key="k"), self.events,
                            verify_passes=2, verify_min_hours=1, clock=self.clock)
        self.defect = self.ledger.get_all_deduplicated_defects()[0]["defect_id"]

    def tearDown(self):
        self.w.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def _to_repaired(self):
        from pipeline.works import WorkflowError
        o = self.w.issue(self.defect)
        oid = o["work_order_id"]
        with self.assertRaises(WorkflowError):
            self.w.transition(oid, "ASSIGNED")                           # needs a contractor
        self.w.transition(oid, "ASSIGNED", contractor="Acme Roads")
        self.w.transition(oid, "IN_PROGRESS")
        self.clock.t += 3600
        self.w.transition(oid, "REPAIRED", note="patched")
        return oid

    def test_issue_uses_ledger_measurements_and_seals(self):
        from pipeline.works import WorkflowError
        o = self.w.issue(self.defect)
        wo = o["work_order"]
        self.assertEqual((wo["surface_area_sqm"], wo["depth_cm"], wo["pavement_pci"]), (1.2, 6.0, 30))
        self.assertEqual(o["seal_status"], "SEAL_VERIFIED_AUTHENTIC")
        self.assertTrue(o["history_chain_intact"])
        with self.assertRaises(WorkflowError):
            self.w.issue(self.defect)                                    # one open order per defect
        with self.assertRaises(WorkflowError):
            self.w.issue("DEF-NOPE")

    def test_issue_needs_a_depth(self):
        from models.morth_dispatch_agent import MoRTHDispatchAgent
        from pipeline.works import WorkflowError, WorkOrders
        led = _ledger((("Pothole Cavity", 12.9, 77.5, 30, 1.0, None),))
        w = WorkOrders(":memory:", led, MoRTHDispatchAgent(), clock=self.clock)
        did = led.get_all_deduplicated_defects()[0]["defect_id"]
        with self.assertRaises(WorkflowError):
            w.issue(did)
        self.assertEqual(w.issue(did, depth_cm=5)["work_order"]["depth_cm"], 5.0)
        w.close()

    def test_illegal_transitions(self):
        from pipeline.works import WorkflowError
        oid = self.w.issue(self.defect)["work_order_id"]
        for bad in ("REPAIRED", "VERIFIED", "nonsense"):
            with self.assertRaises(WorkflowError):
                self.w.transition(oid, bad)
        self.w.transition(oid, "CANCELLED")
        with self.assertRaises(WorkflowError):
            self.w.transition(oid, "ASSIGNED")
        self.assertEqual(self.w.issue(self.defect)["status"], "ISSUED", "a new order after a cancelled one")

    def test_fleet_verifies_after_enough_clean_passes(self):
        oid = self._to_repaired()
        lat, lon = self.defect_latlon()
        self.assertEqual(self.w.on_bus_position("BUS-7", lat + 0.003, lon), [], "300 m away is not a pass")
        self.assertEqual(self.w.on_bus_position("BUS-7", lat + 0.00005, lon), [])
        self.clock.t += 60
        self.assertEqual(self.w.on_bus_position("BUS-7", lat, lon), [], "same bus within 10 minutes counts once")
        self.clock.t += 120
        self.assertEqual(self.w.on_bus_position("BUS-8", lat, lon), [], "two passes but not yet an hour")
        self.clock.t += 3600
        self.assertEqual(self.w.on_bus_position("BUS-9", lat, lon), [], "not before the settle window")
        self.clock.t += 30
        self.assertEqual(self.w.tick(), [])
        self.clock.t += 31
        self.assertEqual(self.w.tick(), [oid])
        o = self.w.get(oid)
        self.assertEqual(o["status"], "VERIFIED")
        self.assertEqual(o["history"][-1]["actor"], "fleet")
        self.assertEqual(len(o["history"][-1]["evidence"]["passes"]), 3)
        self.assertTrue(o["history_chain_intact"])

    def test_a_pass_between_two_fixes_counts(self):
        oid = self._to_repaired()
        lat, lon = self.defect_latlon()
        step = 0.0004                                   # ~44 m: neither fix is within 15 m of the defect
        self.assertEqual(self.w.on_bus_position("BUS-5", lat, lon - step), [])
        self.clock.t += 6
        self.w.on_bus_position("BUS-5", lat, lon + step)
        self.assertEqual(self.w.get(oid)["fleet_passes_since_repair"], 1)
        self.clock.t += 6
        self.w.on_bus_position("BUS-6", lat + 0.002, lon - step)
        self.clock.t += 6
        self.w.on_bus_position("BUS-6", lat + 0.002, lon + step)      # parallel road 220 m away
        self.assertEqual(self.w.get(oid)["fleet_passes_since_repair"], 1)
        self.clock.t += 600
        self.w.on_bus_position("BUS-6", lat, lon - step)               # a jump after a long gap is not a track
        self.assertEqual(self.w.get(oid)["fleet_passes_since_repair"], 1)

    def test_segment_distance(self):
        from pipeline.works import _segment_distance_m
        self.assertLess(_segment_distance_m(12.97, 77.60, (12.97, 77.599), (12.97, 77.601)), 0.5)
        self.assertAlmostEqual(_segment_distance_m(12.97, 77.60, (12.9701, 77.599), (12.9701, 77.601)), 11.1, delta=0.3)
        self.assertAlmostEqual(_segment_distance_m(12.97, 77.60, (12.97, 77.601), (12.97, 77.602)), 108.5, delta=1.0)

    def test_a_bus_that_still_sees_the_defect_does_not_verify_it(self):
        oid = self._to_repaired()
        self.w.verify_min_hours = 0
        lat, lon = self.defect_latlon()
        self.w.on_bus_position("BUS-7", lat, lon)
        self.clock.t += 5
        self.w.on_bus_position("BUS-8", lat, lon)                    # enough passes...
        self.clock.t += 2
        self.w.on_sighting(self.defect, "BUS-8")                     # ...but BUS-8's camera still sees it
        self.clock.t += 120
        self.assertEqual(self.w.tick(), [])
        self.assertEqual(self.w.get(oid)["status"], "REOPENED")

    def test_one_bus_cannot_verify_alone(self):
        oid = self._to_repaired()
        self.w.verify_min_hours = 0
        lat, lon = self.defect_latlon()
        for _ in range(4):
            self.w.on_bus_position("BUS-7", lat, lon)
            self.clock.t += 700                                   # past the per-bus cooldown each time
        self.assertEqual(self.w.tick(), [])
        self.assertEqual(self.w.get(oid)["fleet_passes_since_repair"], 1, "passes count different buses")

    def test_late_uploads_from_before_the_repair_are_not_evidence(self):
        before = self.clock.t
        oid = self._to_repaired()
        lat, lon = self.defect_latlon()
        self.assertIsNone(self.w.on_sighting(self.defect, "BUS-2", at=before), "seen before the repair")
        self.assertEqual(self.w.get(oid)["status"], "REPAIRED")
        self.w.on_bus_position("BUS-3", lat, lon, at=before)
        self.assertEqual(self.w.get(oid)["fleet_passes_since_repair"], 0)

    def test_facts_come_from_the_chain(self):
        oid = self._to_repaired()
        self.w._db.execute("UPDATE order_details SET repaired_at=0, contractor='Other' WHERE order_id=?", (oid,))
        o = self.w.get(oid)
        self.assertEqual(o["contractor"], "Acme Roads", "an edited cache column is not believed")
        self.assertGreater(o["repaired_at"], 0)
        self.w._db.execute("UPDATE work_orders SET status='VERIFIED' WHERE order_id=?", (oid,))
        self.assertFalse(self.w.verify_chain(oid), "a status that disagrees with the history breaks it")

    def test_cancelled_after_failed_repair_is_not_a_met_sla(self):
        oid = self._to_repaired()
        self.w.on_sighting(self.defect, "BUS-2")
        self.w.transition(oid, "CANCELLED")
        o = self.w.get(oid)
        self.assertIsNone(o["met_sla"])
        self.assertIsNone(o["repaired_at"])

    def test_bad_depth_is_refused(self):
        from pipeline.works import WorkflowError
        for bad in (float("nan"), float("inf"), -1, 0, 500):
            with self.assertRaises(WorkflowError):
                self.w.issue(self.defect, depth_cm=bad)

    def test_seen_again_reopens(self):
        oid = self._to_repaired()
        self.assertIsNone(self.w.on_sighting("DEF-OTHER", "BUS-2"))
        self.assertEqual(self.w.on_sighting(self.defect, "BUS-2"), oid)
        self.assertEqual(self.w.get(oid)["status"], "REOPENED")
        self.w.transition(oid, "IN_PROGRESS")
        self.w.transition(oid, "REPAIRED")
        self.w.transition(oid, "VERIFIED", actor="inspector")
        self.assertEqual(self.w.on_sighting(self.defect, "BUS-3"), oid, "a verified repair that failed reopens")

    def test_sighting_before_repair_changes_nothing(self):
        oid = self.w.issue(self.defect)["work_order_id"]
        self.assertIsNone(self.w.on_sighting(self.defect, "BUS-2"))
        self.assertEqual(self.w.get(oid)["status"], "ISSUED")

    def test_history_tampering_is_detected(self):
        oid = self._to_repaired()
        self.assertTrue(self.w.verify_chain(oid))
        self.w._db.execute("UPDATE order_events SET at = at - 86400 WHERE order_id=? AND status='REPAIRED'", (oid,))
        self.assertFalse(self.w.verify_chain(oid), "backdating a step breaks the chain")

    def test_sla_and_summary(self):
        o = self.w.issue(self.defect)
        self.assertEqual(o["sla_hours"], 24)                               # a pothole is a 24 h order
        self.assertFalse(o["overdue"])
        self.clock.t += 25 * 3600
        self.assertTrue(self.w.list()[0]["overdue"])
        s = self.w.summary()
        self.assertEqual(s["by_status"]["ISSUED"], 1)
        self.assertEqual(s["overdue"], 1)
        self.assertEqual(self.w.status_by_defect()[self.defect]["status"], "ISSUED")

    def test_events_are_published(self):
        q = self.events.subscribe()
        oid = self.w.issue(self.defect)["work_order_id"]
        self.w.transition(oid, "CANCELLED")
        kinds = [q.get_nowait()["data"]["status"] for _ in range(q.qsize())]
        self.assertEqual(kinds, ["ISSUED", "CANCELLED"])

    def defect_latlon(self):
        d = self.ledger.get_all_deduplicated_defects()[0]
        return d["lat"], d["lon"]


class AlertsTest(unittest.TestCase):
    def test_once_per_subject_and_webhook(self):
        from pipeline.alerts import Alerts
        got = []

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass
        httpd = HTTPServer(("127.0.0.1", 0), H)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            a = Alerts(webhook=f"http://127.0.0.1:{httpd.server_address[1]}/hook", timeout=5)
            d = {"defect_id": "D1", "lat": 12.9, "lon": 77.6, "defect_class": "Pothole Cavity", "severity_pci": 10}
            self.assertIsNotNone(a.check_defect(d, {"band": "P1", "priority_index": 88.0}))
            self.assertIsNone(a.check_defect(d, {"band": "P1", "priority_index": 88.0}), "sent once")
            self.assertIsNone(a.check_defect(dict(d, defect_id="D2"), {"band": "P3", "priority_index": 30}))
            self.assertEqual(len(a.check_overdue([{"work_order_id": "W1", "overdue": True, "sla_hours": 24},
                                                  {"work_order_id": "W2", "overdue": False}])), 1)
            for _ in range(100):
                if a.status()["delivered"] >= 2:
                    break
                threading.Event().wait(0.05)
            self.assertEqual(len(got), 2)
            texts = sorted(g["text"] for g in got)          # sent in parallel, so in either order
            self.assertTrue(texts[0].startswith("ROAD-SHIELD: P1"))
            self.assertIn("past its 24 h SLA", texts[1])
            self.assertEqual(a.status()["delivered"], 2)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_only_https_or_local(self):
        from pipeline.alerts import webhook_allowed, Alerts
        self.assertTrue(webhook_allowed("https://hooks.slack.com/services/x"))
        self.assertTrue(webhook_allowed("http://localhost:9000/x"))
        self.assertFalse(webhook_allowed("http://example.com/x"))
        self.assertFalse(webhook_allowed("file:///etc/passwd"))
        self.assertIn("rejected", Alerts(webhook="http://example.com/x").status()["webhook"])


class Export(unittest.TestCase):
    def test_geojson_and_csv(self):
        from pipeline.export import ledger_csv, ledger_geojson
        defects = [{"defect_id": "D1", "defect_class": "=HYPERLINK(\"x\")", "lat": 12.9, "lon": 77.6,
                    "severity_pci": 40, "area_m2": 1.0, "confirmation_count": 2, "reporting_buses": ["A", "B"],
                    "first_seen_timestamp": 0, "last_seen_timestamp": 86400}]
        pr = lambda d: {"priority_index": 55.0, "band": "P2"}
        repair = {"D1": {"status": "IN_PROGRESS", "work_order_id": "W1"}}
        g = json.loads(ledger_geojson(defects, pr, repair))
        f = g["features"][0]
        self.assertEqual(f["geometry"]["coordinates"], [77.6, 12.9], "GeoJSON is lon, lat")
        self.assertEqual(f["properties"]["repair_status"], "IN_PROGRESS")
        self.assertEqual(f["properties"]["distinct_sources"], 2)
        text = ledger_csv(defects, pr, repair)
        self.assertTrue(text.startswith("\ufeff"), "BOM for Excel")
        rows = list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))
        self.assertTrue(rows[0]["defect_class"].startswith("'="), "no spreadsheet formula injection")
        self.assertEqual(rows[0]["priority_band"], "P2")
        self.assertEqual(rows[0]["last_seen_utc"], "1970-01-02T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
