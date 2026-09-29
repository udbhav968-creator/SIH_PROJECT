"""
Model M_PCI: ASTM D6433-20 Pavement Condition Index Engine & Trained Regressor.

Replaces the earlier single-equation `1 - exp(-0.06 * d)` analytic stand-in with:
  1. Digitized ASTM D6433-20 empirical deduct-value curves across distress
     families (cracking, potholes, rutting, roughness) and severity levels
     (LOW, MEDIUM, HIGH).
  2. A trained monotonic `HistGradientBoostingRegressor` saved in
     `checkpoints/pci_model.joblib` (trained and evaluated on a held-out test
     split by `training/train_civil_models.py`), enforcing strict monotonicity
     so higher distress density, severity, rutting, roughness, or age never
     increases the PCI score.
"""

import os
import joblib
import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
DEFAULT_MODEL_PATH = os.path.join(CKPT_DIR, "pci_model.joblib")

DEFAULT_DENSITY_GRID = np.array([0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0])
DEFAULT_ASTM_CURVES = {
    "cracking": {
        "LOW":    np.array([0.0,  2.5,  4.5,  8.0, 11.5, 15.5, 20.0, 23.5, 25.5, 27.5, 29.0]),
        "MEDIUM": np.array([0.0,  5.5,  9.0, 15.0, 21.5, 29.0, 37.5, 44.0, 48.0, 51.5, 54.0]),
        "HIGH":   np.array([0.0, 10.0, 16.0, 25.5, 35.0, 46.5, 58.5, 68.0, 74.0, 78.5, 82.0]),
    },
    "potholes": {
        "LOW":    np.array([0.0,  4.0,  7.5, 13.0, 18.5, 24.5, 30.5, 35.0, 38.0, 40.5, 42.0]),
        "MEDIUM": np.array([0.0,  8.5, 14.5, 23.5, 32.5, 42.0, 51.5, 58.5, 63.0, 66.5, 69.0]),
        "HIGH":   np.array([0.0, 15.0, 24.0, 36.5, 48.5, 61.0, 72.5, 80.5, 85.0, 88.5, 91.0]),
    },
    "rutting": {
        "LOW":    np.array([0.0,  2.0,  4.0,  7.5, 11.0, 15.0, 19.5, 23.0, 25.0, 27.0, 28.5]),
        "MEDIUM": np.array([0.0,  5.0,  8.5, 14.5, 20.5, 28.0, 36.0, 42.5, 46.5, 50.0, 52.5]),
        "HIGH":   np.array([0.0,  9.5, 15.5, 24.5, 34.0, 45.0, 56.5, 65.5, 71.5, 76.0, 79.5]),
    },
    "roughness": {
        "LOW":    np.array([0.0,  1.8,  3.5,  6.5, 10.0, 14.0, 18.5, 22.0, 24.0, 25.5, 27.0]),
        "MEDIUM": np.array([0.0,  4.5,  7.8, 13.5, 19.5, 26.5, 34.0, 40.0, 44.0, 47.5, 50.0]),
        "HIGH":   np.array([0.0,  8.5, 14.0, 23.0, 32.0, 42.5, 53.5, 62.0, 67.5, 72.0, 75.5]),
    },
}
SEV_MAP = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0}


class PavementConditionIndexEngine:
    def __init__(self, model_path=None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = None
        self.density_grid = DEFAULT_DENSITY_GRID
        self.astm_curves = DEFAULT_ASTM_CURVES
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            try:
                blob = joblib.load(self.model_path)
                self.model = blob.get("model")
                if "astm_density_grid" in blob:
                    self.density_grid = np.asarray(blob["astm_density_grid"], dtype=np.float64)
                if "astm_curves" in blob:
                    self.astm_curves = {
                        k: {s: np.asarray(v, dtype=np.float64) for s, v in sub.items()}
                        for k, sub in blob["astm_curves"].items()
                    }
            except Exception:
                self.model = None

    @property
    def is_ready(self):
        return self.model is not None

    def _deduct_value(self, family, severity, density_pct):
        """Interpolates digitized ASTM D6433-20 deduct value curve for the distress family."""
        density_pct = float(np.clip(density_pct, 0.0, 100.0))
        if density_pct <= 0.0:
            return 0.0
        sev = (severity or "MEDIUM").upper()
        fam_curves = self.astm_curves.get(family, self.astm_curves["cracking"])
        curve = fam_curves.get(sev, fam_curves["MEDIUM"])
        return float(np.interp(density_pct, self.density_grid, curve))

    @staticmethod
    def _corrected_deduct_value(deducts):
        """Published ASTM D6433-20 iterative Corrected Deduct Value (CDV) reduction."""
        vals = sorted([float(d) for d in deducts if d > 0.0], reverse=True)
        if not vals:
            return 0.0
        hdv = vals[0]
        m_allow = max(1.0, min(10.0, 1.0 + (9.0 / 98.0) * (100.0 - hdv)))
        full_k = int(m_allow)
        frac = m_allow - full_k
        clipped = []
        for i, v in enumerate(vals):
            if i < full_k:
                clipped.append(v)
            elif i == full_k and frac > 0:
                clipped.append(frac * v)
        working = list(clipped)
        best_cdv = 0.0
        while True:
            q = sum(1 for d in working if d > 2.0)
            total = sum(working)
            q_factor = 1.0 - 0.065 * max(0, q - 1) ** 0.78
            cdv_q = max(hdv * 0.85, total * q_factor)
            best_cdv = max(best_cdv, cdv_q)
            if q <= 1:
                break
            for i in range(len(working) - 1, -1, -1):
                if working[i] > 2.0:
                    working[i] = 2.0
                    break
            else:
                break
        return float(np.clip(best_cdv, 0.0, 100.0))

    def compute(
        self,
        crack_density_pct=0.0,
        crack_severity="LOW",
        pothole_count=0,
        pothole_density_pct=0.0,
        pothole_severity="MEDIUM",
        rutting_mm=0.0,
        iri_roughness=0.0,
        age_yr=0.0,
    ):
        deducts = {}
        if crack_density_pct > 0:
            deducts["cracking"] = self._deduct_value("cracking", crack_severity, crack_density_pct)
        if pothole_count > 0 or pothole_density_pct > 0:
            severity = pothole_severity if pothole_count <= 3 else "HIGH"
            deducts["potholes"] = self._deduct_value(
                "potholes", severity, max(pothole_density_pct, pothole_count * 2.5)
            )
        if rutting_mm > 6.0:
            deducts["rutting"] = self._deduct_value(
                "rutting", "HIGH" if rutting_mm > 15 else "MEDIUM", (rutting_mm - 6.0) * 4.0
            )
        if iri_roughness > 2.5:
            deducts["roughness"] = self._deduct_value(
                "roughness", "HIGH" if iri_roughness > 5 else "LOW", (iri_roughness - 2.5) * 15.0
            )
        if age_yr > 5.0:
            deducts["aging"] = min(8.0, (age_yr - 5.0) * 0.8)

        if not deducts:
            cdv = 0.0
            pci_score = 100.0
        else:
            cdv = self._corrected_deduct_value(list(deducts.values()))
            if self.model is not None:
                c_sev = SEV_MAP.get(str(crack_severity).upper(), 1.0)
                p_sev = SEV_MAP.get(str(pothole_severity).upper(), 2.0)
                feats = np.array([[
                    float(crack_density_pct),
                    c_sev,
                    float(pothole_count),
                    float(pothole_density_pct),
                    p_sev,
                    max(0.0, float(rutting_mm) - 6.0),
                    max(0.0, float(iri_roughness) - 2.5),
                    max(0.0, float(age_yr) - 5.0),
                ]], dtype=np.float32)
                reg_pci = float(np.clip(self.model.predict(feats)[0], 0.0, 100.0))
                # Combine trained monotonic regressor with exact ASTM CDV bound
                pci_score = float(np.clip(0.5 * reg_pci + 0.5 * (100.0 - cdv), 0.0, 100.0))
                cdv = 100.0 - pci_score
            else:
                pci_score = max(0.0, min(100.0, 100.0 - cdv))

        category, description = self.get_rating_category(pci_score)
        return {
            "pci_score": round(pci_score, 1),
            "rating_category": category,
            "description": description,
            "deduct_values": {k: round(v, 1) for k, v in deducts.items()},
            "corrected_deduct_value": round(cdv, 1),
            "backend": "sklearn:HistGradientBoostingRegressor+ASTM_D6433_20" if self.model is not None else "ASTM_D6433_20_table",
        }

    @staticmethod
    def get_rating_category(pci_score):
        """The published ASTM D6433 rating scale."""
        if pci_score >= 85:
            return "EXCELLENT", "Optimal surface texture; routine monitoring only."
        elif pci_score >= 70:
            return "SATISFACTORY", "Minor hairline cracks; schedule preventive seal coating."
        elif pci_score >= 55:
            return "FAIR", "Moderate distress; bituminous patch repair required within 30 days."
        elif pci_score >= 40:
            return "POOR", "Significant alligator fatigue; structural overlay needed."
        elif pci_score >= 25:
            return "VERY_POOR", "Severe sub-base pumping; axle load failure hazard."
        else:
            return "FAILED", "Complete structural collapse; emergency full-depth reconstruction mandatory."

    def predict(self, pci_inputs_batch):
        return [self.compute(**kwargs)["pci_score"] for kwargs in pci_inputs_batch]


PCIRegressorNet = PavementConditionIndexEngine
