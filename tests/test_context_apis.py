"""Rain (Open-Meteo) and road (OpenStreetMap Overpass) lookups, with a fake network."""
import datetime
import json
import os
import sys
import unittest
import urllib.parse
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import road_context as rc  # noqa: E402

TODAY = datetime.date(2026, 6, 1)


def fake_archive(mm_per_day_monsoon=12.0, mm_per_day_dry=0.5):
    def fetch(url, data=None, timeout=4.0):
        if url.startswith(rc.OVERPASS_URL):
            q = urllib.parse.parse_qs(data.decode())["data"][0]
            assert "around:30" in q
            return {"elements": [
                # a long primary road whose centre is far away, but which passes 2 m from the point
                {"tags": {"highway": "primary", "name": "MG Road", "surface": "asphalt", "lanes": "4",
                          "maxspeed": "40", "oneway": "yes"},
                 "geometry": [{"lat": 12.97162, "lon": 77.580}, {"lat": 12.97162, "lon": 77.640}]},
                # a residential lane 25 m away
                {"tags": {"highway": "residential", "name": "4th Cross"},
                 "geometry": [{"lat": 12.97182, "lon": 77.5940}, {"lat": 12.97182, "lon": 77.5950}]}]}
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        d0, d1 = datetime.date.fromisoformat(q["start_date"]), datetime.date.fromisoformat(q["end_date"])
        days, rain = [], []
        d = d0
        while d <= d1:
            days.append(d.isoformat())
            rain.append(mm_per_day_monsoon if d.month in (6, 7, 8, 9) else mm_per_day_dry)
            d += datetime.timedelta(days=1)
        return {"daily": {"time": days, "precipitation_sum": rain}}
    return fetch


class RainTest(unittest.TestCase):
    def test_climatology_of_the_next_180_days_and_clipping(self):
        ctx = rc.RoadContext(fetch=fake_archive(), today=TODAY)
        r = ctx.rainfall(12.9716, 77.5946)
        self.assertTrue(r["available"])
        # 1 Jun + 180 days: 122 monsoon days at 12 mm, 58 dry days at 0.5 mm, in each of three years
        self.assertAlmostEqual(r["expected_next_180_days_mm"], 122 * 12 + 58 * 0.5, delta=1.0)
        self.assertEqual(r["years_averaged"], 3)
        self.assertAlmostEqual(r["observed_last_30_days_mm"], 30 * 0.5, delta=6.0)
        mm, prov = ctx.forecast_rain_input(12.9716, 77.5946)
        self.assertEqual(mm, rc.TRAINED_RAIN_MAX_MM, "beyond the forecaster's range the input is held, not extrapolated")
        self.assertIn("lower bound", prov["clipped"])

    def test_dry_place_is_not_clipped_and_failures_are_reported_not_filled(self):
        ctx = rc.RoadContext(fetch=fake_archive(2.0, 1.0), today=TODAY)
        mm, prov = ctx.forecast_rain_input(28.6, 77.2)
        self.assertLess(mm, rc.TRAINED_RAIN_MAX_MM)
        self.assertNotIn("clipped", prov)

        calls = []

        def broken(url, data=None, timeout=4.0):
            calls.append(url)
            raise OSError("offline")
        ctx = rc.RoadContext(fetch=broken, today=TODAY)
        mm, why = ctx.forecast_rain_input(12.9, 77.6)
        self.assertIsNone(mm)
        self.assertIn("not reachable", why["reason"])
        ctx.forecast_rain_input(12.9, 77.6)
        self.assertEqual(len(calls), 1, "a failure is cached for a while, not retried on every photograph")


class RoadTest(unittest.TestCase):
    def test_nearest_road_by_its_line(self):
        ctx = rc.RoadContext(fetch=fake_archive(), today=TODAY)
        r = ctx.road(12.9716, 77.5946)
        self.assertEqual((r["highway"], r["name"], r["lanes"], r["oneway"]), ("primary", "MG Road", 4, True))
        self.assertLess(r["distance_m"], 5)
        self.assertIn("ODbL", r["source"])
        line = [{"lat": 12.9718, "lon": 77.594}, {"lat": 12.9718, "lon": 77.595}]
        self.assertAlmostEqual(rc.distance_to_line_m(12.9716, 77.5946, line), 22.1, delta=0.5)

    def test_overloaded_overpass_is_a_failure_not_no_road(self):
        ctx = rc.RoadContext(fetch=lambda url, data=None, timeout=4: {"remark": "runtime error: Query timed out",
                                                                       "elements": []}, today=TODAY)
        r = ctx.road(12.9, 77.6)
        self.assertFalse(r["available"])

    def test_circuit_breaker_stops_lookups_after_repeated_failures(self):
        calls = []

        def broken(url, data=None, timeout=4):
            calls.append(url)
            raise OSError("offline")
        ctx = rc.RoadContext(fetch=broken, today=TODAY)
        for i in range(6):
            ctx.road(12.9 + i * 0.01, 77.6)                     # six different places
        self.assertEqual(len(calls), rc.BREAKER_FAILURES, "after three failures the lookups pause")
        self.assertIn("paused", ctx.road(13.5, 77.6)["reason"])

    def test_malformed_answer_does_not_crash(self):
        ctx = rc.RoadContext(fetch=lambda url, data=None, timeout=4: {"daily": {"time": ["x"], "precipitation_sum": [1]}},
                             today=TODAY)
        self.assertFalse(ctx.rainfall(12.9, 77.6)["available"])

    def test_no_road(self):
        ctx = rc.RoadContext(fetch=lambda url, data=None, timeout=4.0: {"elements": []}, today=TODAY)
        self.assertFalse(ctx.road(10.0, 70.0)["found"])


class PipelineUsesRain(unittest.TestCase):
    def test_forecast_input_comes_from_the_api_when_switched_on(self):
        from pipeline.deep_inference_pipeline import DeepInferencePipeline
        pipe = DeepInferencePipeline()
        img = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets",
                           "08_dashcam_video_streams")
        import glob
        frame = sorted(glob.glob(os.path.join(img, "**", "*.jpg"), recursive=True))[0]
        ctx = rc.RoadContext(fetch=fake_archive(2.0, 1.0), today=TODAY)
        with mock.patch.dict(os.environ, {"ROAD_SHIELD_CONTEXT_APIS": "1"}), mock.patch.object(rc, "_DEFAULT", ctx):
            res = pipe.audit_image(frame, latitude=28.6, longitude=77.2)
        ma = res["modelling_assumptions"]
        self.assertNotIn("seasonal_rain_mm", ma["assumed_inputs"])
        self.assertIn("Open-Meteo", ma["inputs_from_public_apis"]["seasonal_rain_mm"]["from"])
        res = pipe.audit_image(frame, latitude=28.6, longitude=77.2)       # switched off (the test default)
        self.assertIn("seasonal_rain_mm", res["modelling_assumptions"]["assumed_inputs"])
        self.assertEqual(res["modelling_assumptions"]["inputs_from_public_apis"], {})


if __name__ == "__main__":
    unittest.main()
