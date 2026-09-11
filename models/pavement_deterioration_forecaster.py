"""
Model M_DEGRADE: pavement deterioration forecaster.

Projects how much a cavity's surface area is expected to grow at 30/60/90/180
days. This uses a standard pavement-engineering growth model rather than a
trained network: unrepaired distress area grows roughly exponentially over
time, and the growth rate is driven by two physical factors that civil
engineering literature consistently points to for accelerated deterioration:

  * traffic loading, in Equivalent Single Axle Loads (ESAL/day) - more heavy
    axle passes over a cavity edge accelerates ravelling
  * moisture ingress, approximated here by seasonal rainfall (mm) - water
    trapped under the surface weakens the sub-base beneath a defect

area(t) = area0 * exp(k * t),  k = k_base + k_esal(esal) + k_rain(rain_mm)

The constants below are engineering assumptions (documented per line) tuned
to give plausible multi-month growth factors, not measurements from a field
trial - there's no public dataset of "same pothole, photographed monthly for
6 months" for us to fit against. Treat the day-180 number as an order-of-
magnitude planning figure, not a certified prediction.
"""

import math


class PavementDeteriorationForecaster:
    HORIZONS = [30, 60, 90, 180]

    # Daily area growth rate with negligible traffic/rain (baseline weathering).
    K_BASE = 0.0025
    # Marginal growth rate contributed by traffic loading, per 1000 ESAL/day.
    K_PER_1000_ESAL = 0.0009
    # Marginal growth rate contributed by monsoon rainfall, per 100mm/season.
    K_PER_100MM_RAIN = 0.0014
    # A cavity's depth deepens faster than its footprint once water pools in it.
    DEPTH_GROWTH_MULTIPLIER_180D = 2.2

    ASPHALT_DENSITY_T_M3 = 2.40
    COMPACTION_FACTOR = 1.15
    MIX_RATE_INR_PER_TONNE = 7500.0

    def _growth_rate_per_day(self, esal_trucks, rain_mm):
        k = self.K_BASE
        k += (max(0.0, esal_trucks) / 1000.0) * self.K_PER_1000_ESAL
        k += (max(0.0, rain_mm) / 100.0) * self.K_PER_100MM_RAIN
        return k

    def forecast_area(self, init_area_m2, esal_trucks, rain_mm):
        """Returns {30: area_m2, 60: ..., 90: ..., 180: ...}."""
        k = self._growth_rate_per_day(esal_trucks, rain_mm)
        return {h: round(init_area_m2 * math.exp(k * h), 3) for h in self.HORIZONS}

    def predict_lifecycle_roi(self, init_area_m2, depth_cm, esal_trucks, rain_mm, age_yr=3.5):
        """
        Compares the repair bill today against the repair bill if the same
        defect is left until day 180 (larger area, and - once water has been
        pooling in it through a monsoon - a deeper cavity too).
        """
        areas = self.forecast_area(init_area_m2, esal_trucks, rain_mm)

        mass_today = init_area_m2 * (depth_cm / 100.0) * self.ASPHALT_DENSITY_T_M3 * self.COMPACTION_FACTOR
        cost_today = round(mass_today * self.MIX_RATE_INR_PER_TONNE, 2)

        depth_180 = depth_cm * self.DEPTH_GROWTH_MULTIPLIER_180D
        mass_180 = areas[180] * (depth_180 / 100.0) * self.ASPHALT_DENSITY_T_M3 * self.COMPACTION_FACTOR
        cost_180 = round(mass_180 * self.MIX_RATE_INR_PER_TONNE, 2)

        return {
            "initial_area_m2": round(init_area_m2, 2),
            "forecast_30_days_area_m2": areas[30],
            "forecast_60_days_area_m2": areas[60],
            "forecast_90_days_area_m2": areas[90],
            "forecast_180_days_area_m2": areas[180],
            "immediate_repair_cost_inr": cost_today,
            "delayed_repair_cost_180d_inr": cost_180,
            "municipal_savings_preventive_inr": round(cost_180 - cost_today, 2),
            "growth_factor_180d": round(areas[180] / max(1e-5, init_area_m2), 2),
            "assumptions": {
                "growth_model": "area(t) = area0 * exp(k*t), k driven by ESAL traffic + rainfall",
                "depth_growth_multiplier_180d": self.DEPTH_GROWTH_MULTIPLIER_180D,
                "note": "Engineering planning estimate, not a field-calibrated forecast (see module docstring).",
            },
        }
