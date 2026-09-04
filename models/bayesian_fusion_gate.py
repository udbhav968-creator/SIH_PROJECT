"""
Model M5: Recursive Bayesian Multimodal Sensor Fusion Gate
Fuses optical visual probability P(V|H) and 100Hz physical shock P(A|H)
using log-odds decision theory to eliminate false positives (e.g. tree shadows, road markings).
"""
import numpy as np

class BayesianFusionGate:
    def __init__(self, prior_pothole_prob=0.05, decision_threshold_log_odds=1.8):
        self.prior_p = prior_pothole_prob
        self.prior_log_odds = np.log(self.prior_p / (1.0 - self.prior_p))
        self.tau = decision_threshold_log_odds

    def fuse(self, p_visual, p_imu_shock, delta_z_ms2=0.0):
        """
        Computes the posterior probability of a genuine road cavity.
        Parameters:
          p_visual: float [0, 1] probability from YOLO/Vision net
          p_imu_shock: float [0, 1] probability from 1D-CNN IMU classifier
          delta_z_ms2: peak Z-axis vertical acceleration shock in m/s^2
        Returns:
          dict with posterior_prob, log_odds, verdict, and explanation.
        """
        # Numerical clamping to avoid log(0)
        pv = np.clip(p_visual, 0.005, 0.995)
        pa = np.clip(p_imu_shock, 0.005, 0.995)
        
        # Log-odds contributions
        l_vis = np.log(pv / (1.0 - pv))
        l_imu = np.log(pa / (1.0 - pa))
        
        # Combined posterior log-odds
        l_total = self.prior_log_odds + l_vis + l_imu
        posterior_p = 1.0 / (1.0 + np.exp(-l_total))
        
        # Verdict decision logic
        if l_total >= self.tau and (pa >= 0.40 or delta_z_ms2 >= 3.5):
            verdict = "CONFIRMED_POTHOLE"
            reason = f"Dual-sensor convergence: optical visual ({pv:.2f}) and physical impact ({delta_z_ms2:.1f} m/s^2) matched."
        elif pv >= 0.70 and pa < 0.15 and delta_z_ms2 < 1.0:
            verdict = "REJECTED_OPTICAL_FALSE_ALARM"
            reason = f"Tree shadow or surface paint marking detected: visual is high ({pv:.2f}) but suspension shock is zero ({delta_z_ms2:.1f} m/s^2)."
        elif pv < 0.40 and (pa >= 0.85 or delta_z_ms2 >= 6.0):
            verdict = "SUBMERGED_MONSOON_POTHOLE"
            reason = f"Waterlogged cavity: optical visibility obscured ({pv:.2f}), but severe physical bottom-out ({delta_z_ms2:.1f} m/s^2) confirmed."
        elif l_total >= self.tau:
            verdict = "CONFIRMED_POTHOLE"
            reason = f"High statistical log-odds ({l_total:.2f}) exceeds gate threshold."
        else:
            verdict = "BELOW_CONFIDENCE_THRESHOLD"
            reason = f"Insufficient statistical evidence (L = {l_total:.2f} < {self.tau})."
            
        return {
            "posterior_probability": float(posterior_p),
            "log_odds_score": float(l_total),
            "gate_passed": bool(verdict in ["CONFIRMED_POTHOLE", "SUBMERGED_MONSOON_POTHOLE"]),
            "verdict": verdict,
            "reason": reason
        }

    def fuse_evidence(self, p_vision, accel_delta_z, vehicle_speed_kmh=55.0, is_wet_monsoon=False):
        """Convenience method matching external sensor telemetry signature."""
        p_imu = 0.96 if accel_delta_z >= 3.5 else 0.05
        return self.fuse(p_visual=p_vision, p_imu_shock=p_imu, delta_z_ms2=accel_delta_z)
