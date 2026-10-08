"""Citizen reports: pinned, confirmed by a bus or an operator, rate-limited, photograph never stored."""

import os
import shutil
import tempfile
import unittest


def audit(cls="Pothole Cavity", class_id=2, distress=True):
    return {"is_distress": distress,
            "primary_distress": {"class_id": class_id, "class_name": cls, "is_distress": distress,
                                 "surface_area_m2": 0.9, "depth_cm": 6.0, "confidence": 0.88},
            "astm_d6433_pci": {"pci_score": 35.0}}


class Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


class CitizenFlow(unittest.TestCase):
    def setUp(self):
        from pipeline.citizen import CitizenReports
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        self.d = tempfile.mkdtemp()
        self.ledger = FleetDeduplicationEngine(store=None)
        self.clock = Clock()
        self.c = CitizenReports(os.path.join(self.d, "c.db"), self.ledger, clock=self.clock)
        self.ledger.listeners.append(self.c.on_sighting)

    def tearDown(self):
        self.c.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_pending_until_a_bus_confirms(self):
        rep, msg = self.c.submit(audit(), 12.95, 77.61, client="1.2.3.4", location_source="photo GPS")
        self.assertEqual(rep["status"], "PENDING")
        self.assertNotIn("sender", rep, "the sender hash is never returned")
        self.assertEqual(self.ledger.get_all_deduplicated_defects(), [], "not in the ledger yet")
        self.ledger.ingest_fleet_detection("BUS-1", 12.95005, 77.61, "Pothole Cavity", 40, 1.0, enrich_location=False)
        rep = self.c.get(rep["report_id"])
        self.assertEqual(rep["status"], "CONFIRMED")
        self.assertEqual(rep["reviewed_by"], "fleet:BUS-1")
        self.assertIsNotNone(rep["defect_id"])

    def test_other_class_or_far_bus_does_not_confirm(self):
        rep, _ = self.c.submit(audit(), 12.95, 77.61, client="a")
        self.ledger.ingest_fleet_detection("BUS-1", 12.95, 77.61, "Crack (Longitudinal / Transverse / Alligator)", 40, 1.0,
                                           enrich_location=False)
        self.ledger.ingest_fleet_detection("BUS-1", 12.951, 77.61, "Pothole Cavity", 40, 1.0, enrich_location=False)
        self.assertEqual(self.c.get(rep["report_id"])["status"], "PENDING")

    def test_already_in_the_ledger_is_confirmed_at_once(self):
        self.ledger.ingest_fleet_detection("BUS-1", 12.95, 77.61, "Pothole Cavity", 40, 1.0, enrich_location=False)
        rep, msg = self.c.submit(audit(), 12.95003, 77.61, client="a")
        self.assertEqual(rep["status"], "CONFIRMED")
        self.assertIn("already in the repair ledger", msg)

    def test_no_defect_nothing_kept(self):
        rep, msg = self.c.submit(audit(distress=False), 12.95, 77.61, client="a")
        self.assertIsNone(rep)
        rep, _ = self.c.submit(audit("Damaged Traffic Sign", 6), 12.95, 77.61, client="a")
        self.assertIsNone(rep, "only defects with an area to repair")
        self.assertEqual(self.c.list(), [])

    def test_operator_review(self):
        from pipeline.citizen import CitizenError
        a, _ = self.c.submit(audit(), 12.95, 77.61, client="a")
        b, _ = self.c.submit(audit(), 12.96, 77.62, client="a")
        r = self.c.review(a["report_id"], "promote", actor="engineer")
        self.assertEqual(r["status"], "CONFIRMED")
        self.assertEqual(len(self.ledger.get_all_deduplicated_defects()), 1)
        self.assertEqual(self.c.review(b["report_id"], "dismiss")["status"], "DISMISSED")
        for bad in ((a["report_id"], "promote"), ("CR-NOPE", "dismiss"), (b["report_id"], "explode")):
            with self.assertRaises(CitizenError):
                self.c.review(*bad)
        self.assertEqual(self.c.counts(), {"PENDING": 0, "CONFIRMED": 1, "DISMISSED": 1})

    def test_rate_limit_and_location(self):
        from pipeline.citizen import CitizenError, MAX_PER_HOUR
        for i in range(MAX_PER_HOUR):
            self.c.submit(audit(), 12.9 + i * 0.01, 77.6, client="spammer")
        with self.assertRaises(CitizenError):
            self.c.submit(audit(), 13.5, 77.6, client="spammer")
        self.c.submit(audit(), 13.5, 77.6, client="someone-else")
        self.clock.t += 3601
        self.c.submit(audit(), 13.6, 77.6, client="spammer")
        for lat, lon in ((None, 77.6), (95, 77.6), (float("nan"), 1)):
            with self.assertRaises(CitizenError):
                self.c.submit(audit(), lat, lon, client="x")


if __name__ == "__main__":
    unittest.main()
