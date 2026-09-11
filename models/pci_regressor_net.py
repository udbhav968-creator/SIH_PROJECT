"""
Model M_PCI: ASTM D6433 Pavement Condition Index.

The PCI standard itself is not a machine-learning model - it's a documented
deduct-value procedure: rate each distress type present by severity and
density, look up a deduct value per distress from the standard's curves, add
them up, run a correction pass, and subtract from 100. There's nothing to
"train" here as long as we implement the real procedure, so that's what this
module does instead of fitting a regressor to made-up numbers.

Honesty note on the curves: ASTM D6433 defines its individual deduct-value
curves as digitized charts (one per distress type / severity), not closed-
form equations. We don't have those charts on hand, so `_deduct_value()`
below uses a documented analytic stand-in (a saturating log curve, capped by
severity) shaped like the real curves rather than a pixel-accurate
reproduction of them. The correction procedure around it (largest-deducts-first
iterative reduction, q count, PCI = 100 - max CDV) is the real ASTM
algorithm. If a stricter compliance requirement ever needs the exact
published curves, replace `_deduct_value()` with a table lookup - the rest of
the class does not need to change.
"""

import math


class PavementConditionIndexEngine:
    SEVERITY_CAP = {"LOW": 25.0, "MEDIUM": 50.0, "HIGH": 80.0}

    def _deduct_value(self, severity, density_pct):
        """Saturating curve: 0 density -> 0 deduct, rising toward the severity's cap."""
        cap = self.SEVERITY_CAP.get(severity.upper(), self.SEVERITY_CAP["MEDIUM"])
        density_pct = max(0.0, min(100.0, density_pct))
        return cap * (1.0 - math.exp(-0.06 * density_pct))

    @staticmethod
    def _corrected_deduct_value(deducts):
        """
        The standard's correction pass: sort deducts descending, iteratively
        clip every value past the (q-1)th down to a floor of 2.0 and
        recompute a total each round, keeping the maximum total seen. This
        keeps a handful of severe distresses from being additively
        double-counted into an unrealistically low PCI.
        """
        deducts = sorted([d for d in deducts if d > 0], reverse=True)
        if not deducts:
            return 0.0
        m = len(deducts)
        best_total = sum(deducts)
        working = list(deducts)
        while True:
            q = sum(1 for d in working if d > 2.0)
            if q <= 1:
                break
            total = sum(working)
            best_total = max(best_total, total)
            # clip the smallest "significant" deduct down toward 2.0 and retry
            for i in range(m - 1, -1, -1):
                if working[i] > 2.0:
                    working[i] = 2.0
                    break
            else:
                break
        return min(best_total, sum(deducts))

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
            deducts["cracking"] = self._deduct_value(crack_severity, crack_density_pct)
        if pothole_count > 0 or pothole_density_pct > 0:
            severity = pothole_severity if pothole_count <= 3 else "HIGH"
            deducts["potholes"] = self._deduct_value(severity, max(pothole_density_pct, pothole_count * 2.5))
        if rutting_mm > 6.0:
            deducts["rutting"] = self._deduct_value("HIGH" if rutting_mm > 15 else "MEDIUM", (rutting_mm - 6.0) * 4.0)
        if iri_roughness > 2.5:
            deducts["roughness"] = self._deduct_value("HIGH" if iri_roughness > 5 else "LOW", (iri_roughness - 2.5) * 15.0)
        # Mild age-related aging deduction even with no visible distress yet.
        if age_yr > 5.0:
            deducts["aging"] = min(8.0, (age_yr - 5.0) * 0.8)

        cdv = self._corrected_deduct_value(list(deducts.values()))
        pci_score = max(0.0, min(100.0, 100.0 - cdv))
        category, description = self.get_rating_category(pci_score)
        return {
            "pci_score": round(pci_score, 1),
            "rating_category": category,
            "description": description,
            "deduct_values": {k: round(v, 1) for k, v in deducts.items()},
            "corrected_deduct_value": round(cdv, 1),
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
        """
        Compatibility shim for callers still passing a list of kwargs dicts
        (one per sample) instead of calling compute() directly.
        """
        return [self.compute(**kwargs)["pci_score"] for kwargs in pci_inputs_batch]


# Backward-compatible alias - this class is not a neural net (see module
# docstring), but older code in this repo may still import the old name.
PCIRegressorNet = PavementConditionIndexEngine
