"""
Model M1: Pavement / road-context photo classifier.

Classifies a road photo (or a candidate crop from CVCavityDetector) into one
of the 7 classes below, using the classic hand-engineered feature vector from
data/feature_extraction.py (HOG + LBP + color histogram) reduced with PCA and
fed to a scikit-learn RBF-kernel SVM.

Why not a CNN: a convolutional network would very likely do better, but it
needs either a GPU + a lot more labeled images than we have, or pretrained
ImageNet weights - and this project has to run on plain laptops with no
internet access to a model zoo. An engineered-feature classifier is the
standard fallback for that situation and, importantly, it's a real model
that is actually trained on the ~160 real labeled photos under datasets/,
not a hand-initialized network that was never fit to anything. See
training/train_vision.py for the honest, leakage-free held-out accuracy
this was chosen against (SVM+PCA tied or beat RandomForest/GradientBoosting/
LogisticRegression at this dataset size).
"""

import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data.feature_extraction import extract_image_features


def compute_box_iou(box1, box2):
    """IoU between two [x, y, w, h] boxes, or batches of them (N, 4)."""
    b1 = np.atleast_2d(box1)
    b2 = np.atleast_2d(box2)
    xA = np.maximum(b1[:, 0], b2[:, 0])
    yA = np.maximum(b1[:, 1], b2[:, 1])
    xB = np.minimum(b1[:, 0] + b1[:, 2], b2[:, 0] + b2[:, 2])
    yB = np.minimum(b1[:, 1] + b1[:, 3], b2[:, 1] + b2[:, 3])
    inter = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    area1 = np.maximum(1e-6, b1[:, 2] * b1[:, 3])
    area2 = np.maximum(1e-6, b2[:, 2] * b2[:, 3])
    union = area1 + area2 - inter
    ious = inter / np.maximum(1e-6, union)
    return float(ious[0]) if (b1.shape[0] == 1 and b2.shape[0] == 1) else ious


class VisionDistressNet:
    CLASS_NAMES = [
        "Normal Road / Sound Pavement",
        "Crack (Longitudinal / Transverse / Alligator)",
        "Pothole Cavity",
        "Waterlogging / Flooding Hazard",
        "Missing Zebra Crossing",
        "Missing Road Divider",
        "Damaged Traffic Sign",
    ]

    # Vulnerable-road-user detections come from CVCavityDetector's HOG person
    # detector, not from this classifier - they're appended downstream with
    # this class id so the combined detection list has one consistent scheme.
    PEDESTRIAN_CLASS_ID = 7

    IRC_STANDARDS = {
        0: "IRC:82-2015 Cl. 3.1: Routine visual survey, non-distress pavement",
        1: "IRC:SP:72-2015 Cl. 5.3/5.4: Bituminous joint sealant / crack injection",
        2: "IRC:82-2015 Cl. 4.2: Mechanical pothole patching with BC + VG-30 tack coat",
        3: "IRC:SP:42-2014 Sec. 8: Camber correction & cross-drainage culvert",
        4: "IRC:35-2015 Cl. 7.2: Retroreflective thermoplastic zebra crossing marking",
        5: "IRC:79-2019 Sec. 4: W-beam crash barrier / retroreflective road divider",
        6: "IRC:67-2012: High-intensity microprismatic sign retrofit",
        7: "IRC:103-2012: Pedestrian facilities - signalized crossing / refuge island",
    }

    CLASS_COLORS = {
        0: {"hex": "#10b981", "label": "NORMAL ROAD"},
        1: {"hex": "#a855f7", "label": "CRACK"},
        2: {"hex": "#f59e0b", "label": "POTHOLE"},
        3: {"hex": "#0ea5e9", "label": "WATERLOGGING"},
        4: {"hex": "#eab308", "label": "MISSING ZEBRA"},
        5: {"hex": "#10b981", "label": "MISSING DIVIDER"},
        6: {"hex": "#3b82f6", "label": "DAMAGED SIGN"},
        7: {"hex": "#06b6d4", "label": "PEDESTRIAN / VRU"},
    }

    def __init__(self, model_path=None, svm_c=20.0, random_state=42):
        self.random_state = random_state
        self.svm_c = svm_c
        self.pipeline = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @property
    def is_ready(self):
        return self.pipeline is not None

    def _build_pipeline(self, n_features, n_samples):
        # Keep PCA components comfortably below the sample count so we don't
        # just memorize a ~160-photo dataset. An RBF-kernel SVM on top of the
        # PCA projection beat a random forest and gradient boosting in a
        # held-out comparison at this dataset size (see training/train_vision.py).
        n_components = max(8, min(64, n_features, n_samples - 1))
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=self.random_state)),
                (
                    "clf",
                    SVC(
                        C=self.svm_c,
                        kernel="rbf",
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.pipeline = self._build_pipeline(X.shape[1], X.shape[0])
        self.pipeline.fit(X, y)
        return self

    def evaluate(self, X, y):
        """Real held-out metrics - no fabricated numbers. Call with data the model was NOT fit on."""
        if not self.is_ready:
            raise RuntimeError("Model has not been trained or loaded yet.")
        y = np.asarray(y, dtype=np.int64)
        preds = self.pipeline.predict(X)
        report = classification_report(
            y, preds, target_names=self.CLASS_NAMES, output_dict=True, zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "confusion_matrix": confusion_matrix(y, preds).tolist(),
            "per_class_report": report,
        }

    def predict(self, X):
        """X: (N, D) feature matrix. Returns (pred_ids, confidences, prob_matrix)."""
        if not self.is_ready:
            raise RuntimeError("Model has not been trained or loaded yet - call fit() or load().")
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        probs = self.pipeline.predict_proba(X)
        # RandomForest.classes_ may be a subset of range(len(CLASS_NAMES)) if a
        # class had zero training examples; project back onto the full space.
        full_probs = np.zeros((probs.shape[0], len(self.CLASS_NAMES)), dtype=np.float32)
        for i, cls in enumerate(self.pipeline.named_steps["clf"].classes_):
            full_probs[:, cls] = probs[:, i]
        preds = np.argmax(full_probs, axis=-1)
        conf = np.max(full_probs, axis=-1)
        return preds, conf, full_probs

    def predict_deep(self, X):
        """Rich per-sample forensic output: entropy, IRC standard, severity, color, top-3."""
        preds, conf, probs = self.predict(X)
        results = []
        for i in range(len(preds)):
            p_vec = probs[i]
            pred_id = int(preds[i])
            c = float(conf[i])
            entropy = float(-np.sum(p_vec * np.log2(p_vec + 1e-10)))
            uncertainty = "LOW_UNCERTAINTY" if entropy < 1.0 else ("MODERATE_UNCERTAINTY" if entropy < 2.0 else "HIGH_UNCERTAINTY_OOD")

            if pred_id == 0:
                severity = "NONE"
            elif pred_id == 2:
                severity = "HIGH" if c > 0.70 else ("MEDIUM" if c > 0.45 else "LOW")
            else:
                severity = "HIGH" if c > 0.75 else ("MEDIUM" if c > 0.50 else "LOW")

            order = np.argsort(p_vec)[::-1][:3]
            top3 = [
                {
                    "rank": rank + 1,
                    "class_id": int(idx),
                    "class_name": self.CLASS_NAMES[idx],
                    "probability": round(float(p_vec[idx]), 4),
                    "color_hex": self.CLASS_COLORS.get(int(idx), {}).get("hex", "#f59e0b"),
                }
                for rank, idx in enumerate(order)
            ]

            color_meta = self.CLASS_COLORS.get(pred_id, {"hex": "#f59e0b", "label": "DISTRESS"})
            results.append(
                {
                    "class_id": pred_id,
                    "class_name": self.CLASS_NAMES[pred_id],
                    "confidence": round(c, 4),
                    "shannon_entropy_bits": round(entropy, 3),
                    "uncertainty_rating": uncertainty,
                    "astm_d6433_severity": severity,
                    "irc_standard_specification": self.IRC_STANDARDS.get(pred_id, "MoRTH general infrastructure guideline"),
                    "color_hex": color_meta["hex"],
                    "hud_label": color_meta["label"],
                    "top3_ranked_predictions": top3,
                    "all_class_probabilities": {self.CLASS_NAMES[k]: round(float(p_vec[k]), 4) for k in range(len(p_vec))},
                }
            )
        return results

    def predict_image(self, image_rgb):
        """Convenience: extract features from a raw RGB crop and classify it. Returns predict_deep()[0]."""
        feat = extract_image_features(image_rgb)
        return self.predict_deep(feat.reshape(1, -1))[0]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path):
        self.pipeline = joblib.load(path)
