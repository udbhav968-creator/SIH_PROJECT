"""
Model M5: dual-sensor Bayesian fusion gate.

Combines the vision model's "this looks like a pothole" probability with the
IMU model's "this felt like a pothole" probability using log-odds (Bayesian)
evidence combination, to cut down on false positives that only one sensor
would produce on its own - a tree shadow can fool the camera, a bad
expansion joint can fool the accelerometer, but both agreeing is a much
stronger signal.
"""

import numpy as np


class BayesianFusionGate:
    def __init__(self, prior_pothole_prob=0.05, decision_threshold_log_odds=1.8):
        self.prior_p = prior_pothole_prob
        self.prior_log_odds = np.log(self.prior_p / (1.0 - self.prior_p))
        self.tau = decision_threshold_log_odds

    def fuse(self, p_visual, p_imu_shock, delta_z_ms2=0.0):
        """
        p_visual: float [0,1] probability from the vision model
        p_imu_shock: float [0,1] probability from the IMU shock classifier
        delta_z_ms2: peak vertical acceleration shock in m/s^2 (0 if no real IMU reading was available)
        """
        pv = np.clip(p_visual, 0.005, 0.995)
        pa = np.clip(p_imu_shock, 0.005, 0.995)

        l_vis = np.log(pv / (1.0 - pv))
        l_imu = np.log(pa / (1.0 - pa))
        l_total = self.prior_log_odds + l_vis + l_imu
        posterior_p = 1.0 / (1.0 + np.exp(-l_total))

        if l_total >= self.tau and (pa >= 0.40 or delta_z_ms2 >= 3.5):
            verdict = "CONFIRMED_POTHOLE"
            reason = f"Vision ({pv:.2f}) and IMU shock ({delta_z_ms2:.1f} m/s^2) agree."
        elif pv >= 0.70 and pa < 0.15 and delta_z_ms2 < 1.0:
            verdict = "REJECTED_OPTICAL_FALSE_ALARM"
            reason = f"Visual score is high ({pv:.2f}) but no matching suspension shock ({delta_z_ms2:.1f} m/s^2) - likely shadow or road marking."
        elif pv < 0.40 and (pa >= 0.85 or delta_z_ms2 >= 6.0):
            verdict = "SUBMERGED_MONSOON_POTHOLE"
            reason = f"Visual score is low ({pv:.2f}, likely waterlogged/obscured) but IMU shows a severe impact ({delta_z_ms2:.1f} m/s^2)."
        elif l_total >= self.tau:
            verdict = "CONFIRMED_POTHOLE"
            reason = f"Combined log-odds ({l_total:.2f}) exceeds the decision threshold ({self.tau})."
        else:
            verdict = "BELOW_CONFIDENCE_THRESHOLD"
            reason = f"Insufficient combined evidence (log-odds {l_total:.2f} < {self.tau})."

        return {
            "posterior_probability": float(posterior_p),
            "log_odds_score": float(l_total),
            "gate_passed": verdict in ("CONFIRMED_POTHOLE", "SUBMERGED_MONSOON_POTHOLE"),
            "verdict": verdict,
            "reason": reason,
        }

    def fuse_evidence(self, p_vision, accel_delta_z, vehicle_speed_kmh=55.0, is_wet_monsoon=False):
        """Convenience wrapper matching the naming used by telemetry callers."""
        p_imu = 0.96 if accel_delta_z >= 3.5 else 0.05
        return self.fuse(p_visual=p_vision, p_imu_shock=p_imu, delta_z_ms2=accel_delta_z)
