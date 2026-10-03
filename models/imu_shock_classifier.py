"""
Model M4: 100 Hz 3-axis IMU shock classifier.

Turns a 1-second window (100 samples x 3 axes) of accelerometer readings into
one of four classes: smooth asphalt, unmarked speed breaker / bump, marked
speed breaker, or a pothole impact. The time -> feature step (extract_temporal_features) is
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
    # Names follow the source logs (plain_road / unmarked_sb / marked_sb /
    # potholes in VishalSingh25/Pothole-Project). Classes 1 and 2 used to be
    # called "Expansion Joint" and "Rumble Strip", which the data never contained:
    # they are unmarked and marked speed breakers.
    CLASS_NAMES = ["Smooth Asphalt", "Unmarked Speed Breaker / Bump", "Marked Speed Breaker", "Pothole Impact"]
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
    def extract_temporal_features(X_raw):
        """(B, T, 3) raw accelerometer window -> time-domain & spectral features."""
        X_raw = np.asarray(X_raw, dtype=np.float32)
        B, T, C = X_raw.shape
        feats = []
        for c in range(C):
            sig = X_raw[:, :, c]
            mean = sig.mean(axis=1, keepdims=True)
            std = sig.std(axis=1, keepdims=True) + 1e-6
            var = sig.var(axis=1, keepdims=True)
            mx = sig.max(axis=1, keepdims=True)
            mn = sig.min(axis=1, keepdims=True)
            ptp = mx - mn
            centered = sig - mean
            zcr = np.mean(np.abs(np.diff(np.sign(centered), axis=1)) > 0, axis=1, keepdims=True)
            energy = np.mean(sig**2, axis=1, keepdims=True)
            ac_energy = np.mean(centered**2, axis=1, keepdims=True)
            diff1 = np.diff(sig, axis=1)
            jerk_max = np.max(np.abs(diff1), axis=1, keepdims=True)
            jerk_mean = np.mean(np.abs(diff1), axis=1, keepdims=True)
            jerk_std = np.std(diff1, axis=1, keepdims=True)
            p1 = np.mean(centered[:, : T // 2] ** 2, axis=1, keepdims=True)
            p2 = np.mean(centered[:, T // 2 :] ** 2, axis=1, keepdims=True)
            ratio = (p2 + 1e-5) / (p1 + 1e-5)
            # Higher-order shape & impulsiveness (crest factor, skewness, kurtosis)
            rms = np.sqrt(ac_energy + 1e-6)
            crest = np.max(np.abs(centered), axis=1, keepdims=True) / rms
            skew = np.mean((centered / std) ** 3, axis=1, keepdims=True)
            kurt = np.mean((centered / std) ** 4, axis=1, keepdims=True)
            # Spectral sub-band energies via rFFT (low, mid, high frequency bands)
            fft_mag = np.abs(np.fft.rfft(centered, axis=1))[:, 1:]  # drop DC
            nb = max(1, fft_mag.shape[1] // 3)
            b_low = np.mean(fft_mag[:, :nb] ** 2, axis=1, keepdims=True)
            b_mid = np.mean(fft_mag[:, nb : 2 * nb] ** 2, axis=1, keepdims=True)
            b_high = np.mean(fft_mag[:, 2 * nb :] ** 2, axis=1, keepdims=True)
            spec_tot = b_low + b_mid + b_high + 1e-6
            feats.extend([
                mean, std, var, mx, mn, ptp, zcr, energy, ac_energy,
                jerk_max, jerk_mean, jerk_std, p1, ratio,
                crest, skew, kurt,
                b_low / spec_tot, b_mid / spec_tot, b_high / spec_tot,
            ])
        # Total 3D vector magnitude dynamic excursion
        vmag = np.sqrt(np.sum(X_raw**2, axis=2))
        v_ptp = (vmag.max(axis=1, keepdims=True) - vmag.min(axis=1, keepdims=True))
        v_std = vmag.std(axis=1, keepdims=True)
        feats.extend([v_ptp, v_std])
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

    def predict(self, X_raw):
        """X_raw: (B, 100, 3). Returns (pred_ids, pothole_confidence, prob_matrix)."""
        if not self.is_ready:
            raise RuntimeError("Model has not been trained or loaded yet - call fit() or load().")
        feats = self.extract_temporal_features(X_raw)
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


class IMUShockCNN:
    """ONNX Runtime inference for the 1-D CNN trained by training/train_imu_deep.py.

    Same predict() contract as IMUShockClassifier, so the pipeline does not
    care which one answers. Served only when checkpoints/imu_model_selection.json
    says so (a rule fixed before the held-out windows were scored)."""

    CLASS_NAMES = IMUShockClassifier.CLASS_NAMES
    POTHOLE_CLASS_ID = IMUShockClassifier.POTHOLE_CLASS_ID

    def __init__(self, onnx_path, sidecar_path):
        import json
        import onnxruntime as ort
        with open(sidecar_path) as fh:
            self.meta = json.load(fh)
        self.scale = np.asarray(self.meta["axis_scale"], dtype=np.float32)
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.path = onnx_path

    @property
    def is_ready(self):
        return self.session is not None

    def predict(self, X_raw):
        X = np.asarray(X_raw, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        Xc = X - X.mean(axis=1, keepdims=True)
        x = np.transpose(Xc / self.scale[None, None, :], (0, 2, 1)).astype(np.float32)
        probs = self.session.run(None, {self.input_name: x})[0]
        preds = np.argmax(probs, axis=-1)
        return preds, probs[:, self.POTHOLE_CLASS_ID], probs


def load_served_imu_model(checkpoints_dir):
    """(model, name). The CNN when the recorded selection names it and its files
    load; otherwise the RandomForest checkpoint; (None, 'none') if neither."""
    import json
    sel_path = os.path.join(checkpoints_dir, "imu_model_selection.json")
    sel = {}
    if os.path.exists(sel_path):
        try:
            with open(sel_path) as fh:
                sel = json.load(fh)
        except Exception:
            sel = {}
    if sel.get("served") == "cnn":
        onnx_path = os.path.join(checkpoints_dir, "imu_shock_cnn.onnx")
        side = os.path.join(checkpoints_dir, "imu_shock_cnn.json")
        if os.path.exists(onnx_path) and os.path.exists(side):
            try:
                return IMUShockCNN(onnx_path, side), "cnn"
            except Exception as e:
                print(f"[IMU] CNN selected but would not load ({e}); using the RandomForest")
    rf = IMUShockClassifier(model_path=os.path.join(checkpoints_dir, "imu_shock_model.joblib"))
    return (rf, "random_forest") if rf.is_ready else (None, "none")
