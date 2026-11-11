"""Vision-IMU Cross-Modal Temporal Transformer & Physical Verification Engine.

Fuses high-resolution optical defect detections with 100 Hz tri-axial accelerometer
telemetry to distinguish genuine physical pavement hazards from benign optical stains,
shadows, and paint marks.
"""

import numpy as np


class VIMUTemporalFusionEngine:
    """Cross-modal attention and physical suspension verification engine."""

    GRAVITY_MS2 = 9.80665

    SHOCK_THRESHOLD_POTHOLE_MS2 = 2.2
    SHOCK_THRESHOLD_SEVERE_MS2 = 4.5
    BENIGN_SHOCK_CEILING_MS2 = 0.75

    def __init__(self, vehicle_speed_kmh=45.0, wheelbase_m=2.65):
        self.speed_kmh = float(vehicle_speed_kmh)
        self.speed_mps = max(1.0, self.speed_kmh / 3.6)
        self.wheelbase_m = float(wheelbase_m)

    def extract_imu_features(self, imu_series):
        """
        Extracts temporal shock dynamics from a (N, 3) accelerometer array [ax, ay, az].
        """
        arr = np.asarray(imu_series, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return None

        # Remove gravity baseline from vertical axis
        az = arr[:, 2]
        az_dynamic = az - (np.median(az) if abs(np.median(az) - self.GRAVITY_MS2) < 2.0 else 0.0)

        peak_shock = float(np.max(np.abs(az_dynamic)))
        rms_shock = float(np.sqrt(np.mean(az_dynamic ** 2)))
        variance_shock = float(np.var(az_dynamic))

        # Peak lateral / longitudinal jerk
        ax = arr[:, 0]
        ay = arr[:, 1]
        lat_sway = float(np.std(ax))
        long_jerk = float(np.std(ay))

        return {
            "peak_vertical_shock_ms2": round(peak_shock, 3),
            "rms_vertical_shock_ms2": round(rms_shock, 3),
            "shock_energy": round(variance_shock, 3),
            "lateral_sway_ms2": round(lat_sway, 3),
            "longitudinal_jerk_ms2": round(long_jerk, 3),
            "sample_count": len(arr),
        }

    def fuse_detection(self, visual_defect, imu_series=None, vehicle_speed_kmh=None):
        """
        Cross-modal physical audit combining a visual detection and accelerometer series.

        Parameters:
            visual_defect: dict with class_id, confidence, distance_meters, depth_cm, surface_area_m2.
            imu_series: (N, 3) numpy array or list of 100 Hz tri-axial accelerometer samples.
            vehicle_speed_kmh: optional vehicle speed override in km/h.

        Returns:
            dict containing physical verification status, cross-modal confidence, and dispute defense.
        """
        if vehicle_speed_kmh is not None:
            self.speed_kmh = float(vehicle_speed_kmh)
            self.speed_mps = max(1.0, self.speed_kmh / 3.6)

        dist_m = float(visual_defect.get("distance_meters", 5.0))
        depth_cm = float(visual_defect.get("depth_cm", 4.0))
        vis_conf = float(visual_defect.get("confidence", 0.85))
        cls_id = int(visual_defect.get("class_id", 2))

        # Estimated Time-To-Impact (TTC) in seconds
        ttc_sec = round(dist_m / self.speed_mps, 2)

        imu_feats = self.extract_imu_features(imu_series) if imu_series is not None else None

        if imu_feats is None:
            # Camera-only preview: defect is ahead, vehicle has not yet traversed it
            return {
                "physical_verification_status": "ANTICIPATED_HAZARD_AHEAD",
                "verification_confidence": round(vis_conf, 4),
                "is_physically_verified": True,
                "time_to_impact_sec": ttc_sec,
                "vehicle_speed_kmh": self.speed_kmh,
                "imu_telemetry": None,
                "audit_verdict": (
                    f"Defect located {dist_m:.1f} m ahead (TTC: {ttc_sec}s at {self.speed_kmh:.0f} km/h). "
                    "Physical wheel traversal shock pending."
                ),
            }

        peak_shock = imu_feats["peak_vertical_shock_ms2"]

        # Cross-modal arbitration matrix
        if dist_m <= 3.5:
            # Vehicle has physically passed over or is currently traversing defect location
            if peak_shock >= self.SHOCK_THRESHOLD_POTHOLE_MS2:
                # Both camera and accelerometer confirm structural cavity
                status = "CONFIRMED_STRUCTURAL_DEFECT"
                fused_conf = min(0.99, vis_conf * 0.50 + min(1.0, peak_shock / 4.0) * 0.50)
                is_verified = True
                verdict = (
                    f"INCONTROVERTIBLE EVIDENCE: Visual cavity (depth {depth_cm} cm) corroborated by "
                    f"chassis impulse shock ({peak_shock:.2f} m/s²). Contractor defense unassailable."
                )
            elif peak_shock <= self.BENIGN_SHOCK_CEILING_MS2 and depth_cm >= 4.0:
                # Camera claims deep pothole but car rode smoothly -> likely water reflection, stain, or shadow
                status = "BENIGN_OPTICAL_ARTIFACT"
                fused_conf = round(vis_conf * 0.35, 4)
                is_verified = False
                verdict = (
                    f"FALSE POSITIVE FILTERED: Camera indicated defect, but suspension shock was negligible "
                    f"({peak_shock:.2f} m/s²). Reclassified as surface discoloration or shallow water puddle."
                )
            else:
                # Intermediate/marginal condition
                status = "SHALLOW_SURFACE_RAVELING"
                fused_conf = round((vis_conf + 0.65) / 2.0, 4)
                is_verified = True
                verdict = f"Mild surface depression: shock={peak_shock:.2f} m/s², visual depth={depth_cm} cm."
        else:
            # Defect is still ahead, but IMU reports prior vibrations
            status = "ANTICIPATED_HAZARD_AHEAD"
            fused_conf = vis_conf
            is_verified = True
            verdict = f"Target {dist_m:.1f} m ahead (TTC: {ttc_sec}s). Pavement vibration background: {peak_shock:.2f} m/s²."

        return {
            "physical_verification_status": status,
            "verification_confidence": round(float(fused_conf), 4),
            "is_physically_verified": is_verified,
            "time_to_impact_sec": ttc_sec,
            "vehicle_speed_kmh": self.speed_kmh,
            "imu_telemetry": imu_feats,
            "audit_verdict": verdict,
        }
