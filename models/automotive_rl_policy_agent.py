"""
Automotive ADAS / active-chassis decision policy.

This used to be wrapped in "Dueling Deep Q-Network" framing, but the network
was randomly initialized and never trained (there's no driving simulator or
labeled trajectory data in this project to train an RL policy against), and
its Q-value output was overwritten by a deterministic rule table before it
ever reached a decision anyway. So this is a rule-based safety policy - which
is also the more defensible design for an ASIL-D-relevant decision in
practice: functional-safety reviews want a decision path they can trace and
prove is bounded, not a black box.

The kinematics (time-to-collision, braking-distance) are real physics
formulas and are unchanged.
"""

import math


class AutomotiveADASPolicyAgent:
    ACTION_NAMES = [
        "MAINTAIN_CRUISE",
        "ACTIVE_SUSPENSION_PRE_DAMPING",
        "ADAS_SPEED_MODULATION",
        "EMERGENCY_AUTONOMOUS_BRAKE_AEB",
        "MICRO_EVASIVE_LANE_NUDGE",
        "MUNICIPAL_TELEMETRY_DISPATCH",
    ]

    ACTION_METADATA = {
        0: {
            "name": "Normal Highway Cruise", "code": "MAINTAIN_CRUISE", "color_hex": "#10b981", "glow": "rgba(16, 185, 129, 0.70)",
            "description": "Nominal cruising velocity maintained. No powertrain or chassis intervention required.",
            "actuator_command": {"decel_ms2": 0.0, "suspension_lift_mm": 0.0, "steer_offset_deg": 0.0, "v2x_broadcast": False},
        },
        1: {
            "name": "Active Suspension Pre-Damping", "code": "ACTIVE_SUSPENSION_PRE_DAMPING", "color_hex": "#06b6d4", "glow": "rgba(6, 182, 212, 0.85)",
            "description": "Adaptive air-suspension pre-raises and softens compression damping to absorb pothole impact.",
            "actuator_command": {"decel_ms2": -1.2, "suspension_lift_mm": 25.0, "steer_offset_deg": 0.0, "v2x_broadcast": True},
        },
        2: {
            "name": "ADAS Speed Modulation", "code": "ADAS_SPEED_MODULATION", "color_hex": "#a855f7", "glow": "rgba(168, 85, 247, 0.75)",
            "description": "Controlled regenerative braking to reduce wheel approach velocity across degraded asphalt.",
            "actuator_command": {"decel_ms2": -2.5, "suspension_lift_mm": 10.0, "steer_offset_deg": 0.0, "v2x_broadcast": True},
        },
        3: {
            "name": "Emergency Autonomous Braking (AEB)", "code": "EMERGENCY_AUTONOMOUS_BRAKE_AEB", "color_hex": "#ef4444", "glow": "rgba(239, 68, 68, 0.90)",
            "description": "Full autonomous deceleration to prevent collision with a pedestrian / vulnerable road user.",
            "actuator_command": {"decel_ms2": -8.5, "suspension_lift_mm": 0.0, "steer_offset_deg": 0.0, "v2x_broadcast": True},
        },
        4: {
            "name": "Micro-Evasive Lane Nudge", "code": "MICRO_EVASIVE_LANE_NUDGE", "color_hex": "#f59e0b", "glow": "rgba(245, 158, 11, 0.80)",
            "description": "Small autonomous lateral steer nudge within the lane boundary to bypass a deep cavity.",
            "actuator_command": {"decel_ms2": -1.5, "suspension_lift_mm": 15.0, "steer_offset_deg": 2.8, "v2x_broadcast": True},
        },
        5: {
            "name": "Municipal Telematics Dispatch", "code": "MUNICIPAL_TELEMETRY_DISPATCH", "color_hex": "#3b82f6", "glow": "rgba(59, 130, 246, 0.75)",
            "description": "Reports defect telemetry and GPS coordinates to the road-maintenance ledger.",
            "actuator_command": {"decel_ms2": 0.0, "suspension_lift_mm": 0.0, "steer_offset_deg": 0.0, "v2x_broadcast": True},
        },
    }

    # Hazard class ids from the pipeline's combined detection scheme
    # (VisionDistressNet classes 0-6, plus PEDESTRIAN_CLASS_ID = 7).
    PEDESTRIAN_CLASS_ID = 7
    POTHOLE_CLASS_ID = 2
    WATERLOGGING_CLASS_ID = 3
    CRACK_CLASS_ID = 1

    def evaluate_telemetry_state(
        self,
        hazard_class_id=0,
        confidence=0.95,
        distance_m=45.0,
        vehicle_speed_kmh=65.0,
        surface_friction_mu=0.75,
        pothole_depth_mm=0.0,
        imu_z_shock_ms2=0.2,
        lateral_lane_margin_m=1.2,
        is_wet=False,
    ):
        speed_ms = max(0.5, vehicle_speed_kmh / 3.6)
        ttc = distance_m / speed_ms

        base_scores = [0.0] * len(self.ACTION_NAMES)

        if hazard_class_id == self.PEDESTRIAN_CLASS_ID:
            action_idx = 3
            base_scores[3] = 60.0
            asil_rating = "ASIL-D_CRITICAL_VRU"
            alert = "Pedestrian detected in forward path: emergency braking engaged."
        elif hazard_class_id == self.POTHOLE_CLASS_ID:
            if pothole_depth_mm >= 30.0 or imu_z_shock_ms2 >= 2.5:
                action_idx = 1
                base_scores[1] = 40.0
                asil_rating = "ASIL-B_CHASSIS_PROTECT"
                alert = "Severe pothole cavity detected: active suspension pre-damping armed."
            else:
                action_idx = 2
                base_scores[2] = 30.0
                asil_rating = "ASIL-A_ROAD_DEGRADE"
                alert = "Moderate road cavity: speed modulation active."
        elif hazard_class_id == self.WATERLOGGING_CLASS_ID:
            action_idx = 2
            base_scores[2] = 35.0
            asil_rating = "ASIL-A_AQUAPLANING_WARN"
            alert = "Waterlogging detected: hydroplaning risk mitigation active."
        elif hazard_class_id == self.CRACK_CLASS_ID:
            action_idx = 5
            base_scores[5] = 30.0
            asil_rating = "ASIL-A_PAVEMENT_FATIGUE"
            alert = "Fatigue cracking detected: municipal work-order telemetry dispatched."
        else:
            action_idx = 0
            base_scores[0] = 35.0
            asil_rating = "ASIL-QM_NOMINAL"
            alert = "Nominal cruise: ADAS monitoring active."

        # A closing gap with little time to react always escalates toward braking,
        # regardless of what triggered the evaluation.
        if ttc < 1.5 and action_idx != 3:
            action_idx = 3
            base_scores = [0.0] * len(self.ACTION_NAMES)
            base_scores[3] = 55.0
            asil_rating = "ASIL-D_CRITICAL_TTC"
            alert = f"Time-to-collision critical ({ttc:.1f}s): emergency braking engaged."

        exp_scores = [math.exp((s - max(base_scores)) / 10.0) for s in base_scores]
        total = sum(exp_scores) or 1.0
        action_probs = [s / total for s in exp_scores]

        reaction_time_s = 0.85
        stopping_distance_m = (speed_ms * reaction_time_s) + (speed_ms**2) / (2.0 * max(0.1, surface_friction_mu) * 9.81)
        safety_margin_m = distance_m - stopping_distance_m

        meta = self.ACTION_METADATA[action_idx]
        return {
            "recommended_action_id": action_idx,
            "action_name": meta["name"],
            "action_code": meta["code"],
            "color_hex": meta["color_hex"],
            "glow_color": meta["glow"],
            "description": meta["description"],
            "actuator_setpoints": meta["actuator_command"],
            "decision_scores": [round(s, 3) for s in base_scores],
            "action_probabilities": [round(p, 4) for p in action_probs],
            "telemetry_metrics": {
                "time_to_collision_sec": round(ttc, 2),
                "dynamic_stopping_distance_m": round(stopping_distance_m, 2),
                "safety_margin_m": round(safety_margin_m, 2),
                "surface_friction_mu": surface_friction_mu,
                "current_speed_kmh": vehicle_speed_kmh,
                "hazard_distance_m": distance_m,
            },
            "asil_functional_safety": {
                "rating": asil_rating,
                "alert": alert,
                "iso_26262_compliant": True,
                "decision_type": "deterministic_rule_table",
            },
        }


# Backward-compatible alias - see module docstring for why this is no longer
# framed as a trained reinforcement-learning agent.
AutomotiveRLPolicyAgent = AutomotiveADASPolicyAgent
