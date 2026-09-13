"""
Model M4: 100 Hz 3-axis IMU shock classifier.

Turns a 1-second window (100 samples x 3 axes) of accelerometer readings into
one of four classes: smooth asphalt, expansion joint, rumble strip, or a
pothole impact. The time -> feature step (extract_temporal_features) is
plain, well-understood signal processing (mean/std/energy/zero-crossing-rate/
jerk per axis); the classifier on top of it is a real scikit-learn model
trained on datasets/04_mobile_imu_telemetry_100hz.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class IMUShockClassifier:
    CLASS_NAMES = ["Smooth Asphalt", "Expansion Joint", "Rumble Strip", "Pothole Impact"]
    POTHOLE_CLASS_ID = 3

    def __init__(self, model_path=None, n_estimators=300, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.pipeline = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @property
    def is_ready(self):
        return self.pipeline is not None

    @staticmethod
    def extract_temporal_features(X_raw, vehicle_speed_kmh=None):
        """(B, T, 3) raw accelerometer window -> (B, 36) real time-domain features.
        When vehicle_speed_kmh is provided, normalizes the z-axis acceleration by (40.0 / v)^0.75
        so slow deep pothole traversals and fast expansion joint impacts are scaled consistently.
        """
        X_raw = np.asarray(X_raw, dtype=np.float32).copy()
        if vehicle_speed_kmh is not None and float(vehicle_speed_kmh) > 0:
            speed_ratio = float((40.0 / max(15.0, min(100.0, float(vehicle_speed_kmh)))) ** 0.75)
            X_raw[:, :, 2] *= speed_ratio
        B, T, C = X_raw.shape
        feats = []
        for c in range(C):
            sig = X_raw[:, :, c]
            mean = sig.mean(axis=1, keepdims=True)
            std = sig.std(axis=1, keepdims=True)
            var = sig.var(axis=1, keepdims=True)
            mx = sig.max(axis=1, keepdims=True)
            mn = sig.min(axis=1, keepdims=True)
            ptp = mx - mn
            centered = sig - mean
            zcr = np.mean(np.abs(np.diff(np.sign(centered), axis=1)) > 0, axis=1, keepdims=True)
            energy = np.mean(sig**2, axis=1, keepdims=True)
            diff1 = np.diff(sig, axis=1)
            jerk_max = np.max(np.abs(diff1), axis=1, keepdims=True)
            jerk_mean = np.mean(np.abs(diff1), axis=1, keepdims=True)
            p1 = np.mean(sig[:, : T // 2] ** 2, axis=1, keepdims=True)
            p2 = np.mean(sig[:, T // 2 :] ** 2, axis=1, keepdims=True)
            ratio = (p2 + 1e-5) / (p1 + 1e-5)
            feats.extend([mean, std, var, mx, mn, ptp, zcr, energy, jerk_max, jerk_mean, p1, ratio])
        return np.concatenate(feats, axis=1).astype(np.float32)

    def _build_pipeline(self):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=14,
                        class_weight="balanced",
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, X_raw, y):
        feats = self.extract_temporal_features(X_raw)
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(feats, np.asarray(y, dtype=np.int64))
        return self

    def evaluate(self, X_raw, y):
        if not self.is_ready:
            raise RuntimeError("Model has not been trained or loaded yet.")
        feats = self.extract_temporal_features(X_raw)
        preds = self.pipeline.predict(feats)
        y = np.asarray(y, dtype=np.int64)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "confusion_matrix": confusion_matrix(y, preds).tolist(),
            "per_class_report": classification_report(y, preds, target_names=self.CLASS_NAMES, output_dict=True, zero_division=0),
        }

    def predict(self, X_raw, vehicle_speed_kmh=None):
        """X_raw: (B, 100, 3). Returns (pred_ids, pothole_confidence, prob_matrix)."""
        if not self.is_ready:
            raise RuntimeError("Model has not been trained or loaded yet - call fit() or load().")
        feats = self.extract_temporal_features(X_raw, vehicle_speed_kmh=vehicle_speed_kmh)
        probs_partial = self.pipeline.predict_proba(feats)
        full_probs = np.zeros((probs_partial.shape[0], len(self.CLASS_NAMES)), dtype=np.float32)
        for i, cls in enumerate(self.pipeline.named_steps["clf"].classes_):
            full_probs[:, cls] = probs_partial[:, i]
        preds = np.argmax(full_probs, axis=-1)
        pothole_conf = full_probs[:, self.POTHOLE_CLASS_ID]
        return preds, pothole_conf, full_probs

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path):
        self.pipeline = joblib.load(path)
