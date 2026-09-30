"""IMU band-pass: what it passes, what it removes, and that old checkpoints still run."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import imu_shock_classifier as imu


def tone(hz, amplitude=1.0, samples=100, fs=100.0):
    t = np.arange(samples) / fs
    return amplitude * np.sin(2 * np.pi * hz * t)


class BandpassTests(unittest.TestCase):
    def window(self, signal):
        return np.repeat(signal[None, :, None], 3, axis=2).astype(np.float32)

    def test_removes_gravity_offset(self):
        filtered = imu.bandpass_windows(self.window(np.full(100, 9.81)))
        self.assertLess(np.abs(filtered).max(), 0.05)

    def test_removes_engine_vibration(self):
        filtered = imu.bandpass_windows(self.window(tone(38)))
        self.assertLess(filtered[0, 20:80, 0].std(), 0.2)  # >80% of the 38 Hz amplitude gone

    def test_keeps_impact_band(self):
        filtered = imu.bandpass_windows(self.window(tone(8)))
        self.assertGreater(filtered[0, 20:80, 0].std(), 0.6)  # 8 Hz wheel-impact band survives

    def test_shape_and_dtype(self):
        out = imu.bandpass_windows(np.zeros((4, 100, 3)))
        self.assertEqual((out.shape, out.dtype), ((4, 100, 3), np.float32))


class PipelineCompatibilityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(0, 1, (40, 100, 3)).astype(np.float32)
        self.X[:, :, 2] += 9.81
        self.y = np.arange(40) % 4
        self.X[self.y == 3, 45:55, 2] += 12.0  # "pothole" spikes

    def test_new_models_filter_inside_the_pipeline(self):
        model = imu.IMUShockClassifier(n_estimators=10).fit(self.X, self.y)
        self.assertTrue(model.uses_bandpass)
        _preds, _pothole_conf, probs = model.predict(self.X)
        self.assertEqual(probs.shape, (40, 4))

    def test_checkpoints_from_before_the_filter_still_run(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        legacy = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=10))])
        legacy.fit(imu.IMUShockClassifier.extract_temporal_features(self.X), self.y)
        model = imu.IMUShockClassifier()
        model.pipeline = legacy
        self.assertFalse(model.uses_bandpass)
        self.assertEqual(model.predict(self.X)[2].shape, (40, 4))


if __name__ == "__main__":
    unittest.main()
