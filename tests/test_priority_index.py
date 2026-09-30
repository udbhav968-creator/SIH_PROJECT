"""Repair Priority Index: bounds, monotonicity, honesty about missing traffic, and fairness."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import priority_index as pi


class PriorityIndexTests(unittest.TestCase):
    def test_bounds(self):
        best = pi.priority_index(100.0, 0.0, 0)
        worst = pi.priority_index(0.0, 1e6, 10 * pi.REFERENCE_AADT)
        self.assertEqual(best["priority_index"], 0.0)
        self.assertAlmostEqual(worst["priority_index"], 100.0, delta=0.01)
        self.assertEqual((best["band"], worst["band"]), ("LOW", "CRITICAL"))

    def test_each_term_increases_priority(self):
        base = pi.priority_index(60, 0.05, 5000)["priority_index"]
        self.assertGreater(pi.priority_index(30, 0.05, 5000)["priority_index"], base)
        self.assertGreater(pi.priority_index(60, 0.20, 5000)["priority_index"], base)
        self.assertGreater(pi.priority_index(60, 0.05, 50000)["priority_index"], base)

    def test_volume_saturates(self):
        small = pi.priority_index(50, pi.REFERENCE_VOLUME_M3)["components"]["volume"]
        huge = pi.priority_index(50, 40 * pi.REFERENCE_VOLUME_M3)["components"]["volume"]
        self.assertAlmostEqual(small, 0.5, places=3)
        self.assertLess(huge / small, 2.0)

    def test_unmeasured_traffic_is_dropped_not_guessed(self):
        result = pi.priority_index(40, 0.1)
        self.assertFalse(result["traffic_measured"])
        self.assertNotIn("traffic", result["components"])
        self.assertAlmostEqual(sum(result["effective_weights"].values()), 1.0, places=3)
        self.assertAlmostEqual(result["effective_weights"]["condition"], 0.5 / 0.7, places=3)

    def test_weights_must_be_a_distribution(self):
        with self.assertRaises(ValueError):
            pi.PriorityWeights(0.5, 0.5, 0.5)
        with self.assertRaises(ValueError):
            pi.PriorityWeights(1.2, -0.1, -0.1)

    def test_invalid_inputs_rejected(self):
        for args in ((120, 0.1), (-1, 0.1), (50, -0.1)):
            with self.assertRaises(ValueError):
                pi.priority_index(*args)
        with self.assertRaises(ValueError):
            pi.priority_index(50, 0.1, -5)

    def test_ranking_orders_and_numbers(self):
        ranked = pi.rank_defects([
            {"id": "mild", "pci": 85, "volume_m3": 0.01, "daily_traffic": 800},
            {"id": "severe", "pci": 20, "volume_m3": 0.30, "daily_traffic": 30000},
            {"id": "middle", "pci": 55, "volume_m3": 0.06, "daily_traffic": 6000},
        ])
        self.assertEqual([r["id"] for r in ranked], ["severe", "middle", "mild"])
        self.assertEqual([r["priority_rank"] for r in ranked], [1, 2, 3])

    def test_fairness_locality_cannot_change_priority(self):
        """Identical measurements, different neighbourhoods: identical priority."""
        measured = {"pci": 45, "volume_m3": 0.08, "daily_traffic": 4000}
        ranked = pi.rank_defects([
            {**measured, "id": "outer-colony", "ward": "Ward 198", "address": "Sector 115, outer ring"},
            {**measured, "id": "vip-road", "ward": "Ward 1", "address": "Near Minister's residence",
             "complaints": 500, "constituency": "Central"},
        ])
        self.assertEqual(ranked[0]["priority_index"], ranked[1]["priority_index"])
        # Ties keep input order, so the VIP road does not jump the queue.
        self.assertEqual([r["id"] for r in ranked], ["outer-colony", "vip-road"])


if __name__ == "__main__":
    unittest.main()
