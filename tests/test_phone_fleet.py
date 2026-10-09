"""Phone dashcam (pipeline/phone_fleet.py): axes, timing, and the bus agent's rules on a phone's data."""
import base64
import io
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import phone_fleet as pf  # noqa: E402


def _readings(up, fwd, n=300, rate=100.0, t0=1_000_000.0, bump_at=None, brake=0.0, brake_from=0):
    """Phone readings for a vehicle with the phone's 'up' and camera-forward directions given in phone axes."""
    up, fwd = np.asarray(up, float), np.asarray(fwd, float)
    up, fwd = up / np.linalg.norm(up), fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    rows = []
    for i in range(n):
        vert = 9.81 + (6.0 if bump_at is not None and abs(i - bump_at) < 3 else 0.0)
        a = vert * up + (brake if i >= brake_from else 0.0) * fwd + 0.0 * right
        rows.append([t0 + i * 1000.0 / rate, *a])
    return rows


def _jpeg_b64(img):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


class VehicleFrame(unittest.TestCase):
    def test_landscape_on_the_windscreen(self):
        # landscape, screen to the driver: the phone's x axis is up, the rear camera (-z) looks forward
        t, v, info = pf.to_vehicle_frame(_readings(up=[1, 0, 0], fwd=[0, 0, -1]))
        self.assertAlmostEqual(float(v[:, 2].mean()), 9.81, places=2)
        self.assertAlmostEqual(float(np.abs(v[:, :2]).max()), 0.0, places=6)
        # braking for the last 2 of 3 seconds: oriented by the running gravity, the axes do not tilt
        braking = _readings(up=[1, 0, 0], fwd=[0, 0, -1], brake=-3.0, brake_from=100)
        _, v, _ = pf.to_vehicle_frame(braking, gravity=[9.81, 0, 0])
        self.assertAlmostEqual(float(v[150:, 1].mean()), -3.0, places=3)
        self.assertAlmostEqual(float(v[150:, 2].mean()), 9.81, places=3)
        _, naive, _ = pf.to_vehicle_frame(braking)
        self.assertGreater(abs(float(naive[150:, 1].mean()) + 3.0), 0.05, "a batch-only average would be tilted")
        g = None
        for _ in range(3):
            g = pf.running_gravity(g, [9.81, 0, 0])
        g = pf.running_gravity(g, [9.6, 0, -2.0])
        self.assertLess(abs(float(np.degrees(np.arctan2(-g[2], g[0])))), 4, "one braking batch moves it a little")
        self.assertEqual(list(pf.running_gravity(g, [0, 9.81, 0])), [0, 9.81, 0], "re-mounted: start again")
        self.assertAlmostEqual(info["rate_hz"], 100.0, places=0)
        self.assertAlmostEqual(info["camera_pitch_deg"], 0.0, places=1)

    def test_tilted_down_and_negated_readings_keep_the_vertical(self):
        p = np.radians(15)                          # camera (-z) looking 15 degrees below the horizon
        up = np.array([0, np.cos(p), np.sin(p)])    # world up in phone axes
        fwd = np.array([0, np.sin(p), -np.cos(p)])  # the road ahead, level, in phone axes
        _, v, info = pf.to_vehicle_frame(_readings(up=up, fwd=fwd, bump_at=150))
        self.assertAlmostEqual(float(np.median(v[:, 2])), 9.81, places=2)
        self.assertGreater(float(v[:, 2].max() - v[:, 2].min()), 5.5, "the bump is on the vertical axis")
        self.assertAlmostEqual(info["camera_pitch_deg"], -15.0, places=0)
        neg = [[r[0], -r[1], -r[2], -r[3]] for r in _readings(up=up, fwd=fwd, bump_at=150)]
        _, v2, _ = pf.to_vehicle_frame(neg)
        np.testing.assert_allclose(v2[:, 2], v[:, 2], atol=1e-6)

    def test_refuses_what_it_cannot_use(self):
        with self.assertRaises(ValueError):
            pf.to_vehicle_frame([[0, 0, 0, 9.8]] * 5)
        with self.assertRaises(ValueError):            # linear acceleration only: no gravity in it
            pf.to_vehicle_frame([[i * 10.0, 0, 0, 0.1] for i in range(100)])
        with self.assertRaises(ValueError):            # 5 readings a second
            pf.to_vehicle_frame([[i * 200.0, 0, 0, 9.8] for i in range(30)])
        t, v, _ = pf.to_vehicle_frame(_readings([1, 0, 0], [0, 0, -1]) + [[float("nan"), 1, 1, 1]])
        self.assertTrue(np.isfinite(v).all())

    def test_window_at_resamples_to_100hz(self):
        imu = pf.PhoneImu()
        t_ms, v, _ = pf.to_vehicle_frame(_readings([1, 0, 0], [0, 0, -1], n=180, rate=60.0, bump_at=90))
        t_s = t_ms / 1000.0 - t_ms[0] / 1000.0 + 50.0              # 50.0 .. 52.98 s
        imu.add(t_s, v)
        w = imu.window_at(51.5)
        self.assertEqual(w.shape, (100, 3))
        self.assertGreater(float(w[:, 2].max()), 14.0, "the bump at 51.5 s is inside the second around it")
        self.assertIsNone(imu.window_at(49.0), "no readings that early")
        self.assertEqual(imu.window().shape, (100, 3))
        imu.add(t_s, v)                                            # a resend adds nothing
        self.assertEqual(len(imu.t), 180)
        later = pf.PhoneImu()
        later.add(t_s, v)
        later.add(t_s + 5.0, v)                                    # 2 s with no readings between the batches
        self.assertIsNone(later.window_at(54.0), "a second with a hole in it is not interpolated across")
        self.assertEqual(later.window_at(56.5).shape, (100, 3))


class PhoneTicks(unittest.TestCase):
    AUDIT = {"is_distress": True, "input_check": {"verdict": "ok"},
             "primary_distress": {"class_id": 2, "is_distress": True, "class_name": "Pothole Cavity",
                                  "surface_area_m2": 0.8, "depth_cm": 6.0, "confidence": 0.91,
                                  "probabilities": {"Pothole Cavity": 0.9}},
             "astm_d6433_pci": {"pci_score": 38.0}}

    def setUp(self):
        from models.bayesian_fusion_gate import BayesianFusionGate
        from pipeline.edge_ingest import EdgeIngest
        from pipeline.fleet_deduplication_engine import FleetDeduplicationEngine
        self.d = tempfile.mkdtemp()
        self.dedup = FleetDeduplicationEngine(store=None)
        self.ing = EdgeIngest(os.path.join(self.d, "e.db"), self.dedup, None, key="phone-test")
        audit = self

        class Pipe:
            bayesian_gate = BayesianFusionGate()
            windows = []

            def audit_image(self_, frame, **kw):
                assert kw["imu_series"] is None
                return audit.audit

            def _run_imu_stage(self_, w):
                Pipe.windows.append(w)
                dz = float(np.max(w[:, 2]) - np.min(w[:, 2]))
                return ({"available": True, "pothole_shock_probability": 0.95 if dz > 4 else 0.05,
                         "peak_delta_z_ms2": dz, "shock_classification": "Pothole Impact"},)

        self.pipe = Pipe()
        self.audit = dict(self.AUDIT)
        self.fleet = pf.PhoneFleet(self.ing, lambda: self.pipe)
        rng = np.random.default_rng(0)
        self.frame = _jpeg_b64(np.clip(120 + rng.normal(0, 40, (360, 640, 3)), 0, 255).astype(np.uint8))

    def tearDown(self):
        self.ing._db.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def body(self, now, bump=False, acc=8.0, lag=0.0, **kw):
        sent_ms = now * 1000.0
        rows = _readings([1, 0, 0], [0, 0, -1], n=300, t0=sent_ms - 3000.0 - lag * 1000.0,
                         bump_at=250 if bump else None)
        b = {"device_id": "abc123", "frame_base64": self.frame, "sent_ms": sent_ms, "imu": rows,
             "gps": {"lat": 12.9716, "lon": 77.5946, "accuracy_m": acc, "speed_mps": 6.0}}
        b.update(kw)
        return b

    def test_seen_then_felt_becomes_a_defect(self):
        r = self.fleet.tick(self.body(1000.0), now=1000.0)
        self.assertEqual(r["bus_id"], "PHONE-abc123")
        self.assertEqual(r["seen"]["class"], "Pothole Cavity")
        self.assertEqual(r["waiting_for_wheels"], 1, "the camera saw it ahead; the wheels have not reached it")
        self.assertEqual(self.dedup.get_all_deduplicated_defects(), [])
        # 2 s later, with a bump in the accelerometer second around when the wheels got there (6 m at 6 m/s)
        r = self.fleet.tick(self.body(1002.0, bump=True), now=1002.0)
        self.assertIn("defect", r["events"])
        self.assertEqual([a["type"] for a in r["applied"] if a["type"] != "pos"], ["defect"])
        self.assertEqual(len(self.dedup.get_all_deduplicated_defects()), 1)
        self.assertEqual(r["applied"][-1]["key"], "phone")

    def test_seen_not_felt_is_held(self):
        self.fleet.tick(self.body(1000.0), now=1000.0)
        r = self.fleet.tick(self.body(1002.0, bump=False), now=1002.0)
        self.assertIn("cam", r["events"], "a smooth second at the spot: held until another vehicle sees it")
        self.assertEqual(self.dedup.get_all_deduplicated_defects(), [])

    def test_no_reading_at_the_defect_is_held_not_added(self):
        self.fleet.tick(self.body(1000.0), now=1000.0)
        b = self.body(1010.0, bump=True)              # the phone was out of signal: nothing for 1001-1007 s
        r = self.fleet.tick(b, now=1010.0)
        self.assertIn("cam", r["events"])
        self.assertEqual(self.dedup.get_all_deduplicated_defects(), [], "not felt because not measured: held")

    def test_a_late_second_is_waited_for_and_upload_time_does_not_shift_it(self):
        self.fleet.tick(self.body(1000.0), now=1000.0)
        # the readings in this frame stop 1.5 s before it was sent: the moment the wheels reach the defect
        # (1001.33) is not in them yet, so the sighting waits instead of being judged on an earlier second
        r = self.fleet.tick(self.body(1002.0, lag=1.5), now=1002.0)
        self.assertNotIn("cam", r["events"])
        self.assertNotIn("defect", r["events"])
        self.assertGreaterEqual(r["waiting_for_wheels"], 1)
        # a slow upload: received 0.7 s after it was sent. The clock offset keeps the smallest delay seen,
        # so the bump stays where the phone measured it and is found at the defect
        r = self.fleet.tick(self.body(1004.0, bump=True), now=1004.7)
        self.assertEqual((r["counters"]["defects"], r["counters"]["camera_only"]), (1, 1),
                         "first sighting: smooth second, held; second: the bump, felt")

    def test_accelerometer_trouble_does_not_cost_the_frame(self):
        b = self.body(1000.0)
        b["imu"] = [[1_000_000.0 + i * 10, 0.0, 0.0, 0.1] for i in range(100)]   # no gravity in it
        r = self.fleet.tick(b, now=1000.0)
        self.assertIn("error", r["imu"])
        self.assertEqual(r["seen"]["class"], "Pothole Cavity", "the camera frame is still analysed")
        r = self.fleet.tick(self.body(1002.0), now=1002.0)
        self.assertNotIn("error", r["imu"], "a bad batch does not poison the gravity estimate")
        r = self.fleet.tick(dict(self.body(1004.0), gps=None), now=1004.0)
        self.assertEqual(r["gps"], "no GPS fix in this frame")
        self.assertEqual(r["counters"]["skipped_no_gps"], 1, "an old fix is not reused for a moving phone")

    def test_input_guard_and_coarse_gps_send_nothing(self):
        self.audit = dict(self.AUDIT, input_check={"verdict": "not_road"})
        r = self.fleet.tick(self.body(1000.0), now=1000.0)
        self.assertEqual(r["waiting_for_wheels"], 0, "the input check said this is not a road")
        self.audit = dict(self.AUDIT)
        r = self.fleet.tick(self.body(1002.0, acc=80.0), now=1002.0)
        self.assertIn("80 m", r["gps"])
        self.assertEqual(r["events"], [])
        self.assertEqual(r["counters"]["skipped_no_gps"], 1)

    def test_rules(self):
        self.fleet.tick(self.body(1000.0), now=1000.0)
        with self.assertRaisesRegex(ValueError, "too fast"):
            self.fleet.tick(self.body(1000.3), now=1000.3)
        with self.assertRaises(ValueError):
            self.fleet.tick(dict(self.body(1003.0), device_id="<x>"), now=1003.0)
        self.ing.revoke("PHONE-abc123", "test")
        with self.assertRaises(PermissionError):
            self.fleet.tick(self.body(1005.0), now=1005.0)
        self.ing.reinstate("PHONE-abc123")
        r = self.fleet.tick({"device_id": "abc123", "stop": True}, now=1006.0)
        self.assertTrue(r["stopped"])
        self.assertIn("cam", [a.get("type") for a in r["applied"]], "stopping flushes what was waiting")
        self.assertEqual(self.fleet.devices(now=1006.0)[0]["bus_id"], "PHONE-abc123")


if __name__ == "__main__":
    unittest.main()
