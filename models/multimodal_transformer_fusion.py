"""
Model MM-1: multimodal evidence fusion for defect confirmation.

The original version of this module simulated five sensor modalities -
vision (64-dim), IMU (36-dim), LiDAR depth (16-dim), CAN-bus telematics
(12-dim), and an "environment" vector (8-dim) - fed through a hand-rolled
cross-attention transformer with randomly initialized weights and no
training data behind any of it. Worse, three of those five "modalities"
never existed in this project at all: there is no LiDAR depth log and no
CAN-bus telemetry log anywhere in the datasets. An untrained attention
mechanism over partly-nonexistent inputs cannot fuse anything real.

This rewrite is scoped down to the two modalities this project actually has
independently trained models and real data for - vision
(VisionDistressNet's class probabilities) and IMU shock
(IMUShockClassifier's pothole probability) - and combines them with a
transparent, weighted evidence adjustment instead of an opaque, untrained
"attention" mechanism. This is intentionally simpler than a transformer:
with two real signals, a two-input specification is worth more than a
five-input one where three inputs are fabricated.
"""

import numpy as np

from models.vision_distress_net import VisionDistressNet

CLASS_NAMES = VisionDistressNet.CLASS_NAMES
CLASS_COLORS = VisionDistressNet.CLASS_COLORS
PEDESTRIAN_CLASS_ID = VisionDistressNet.PEDESTRIAN_CLASS_ID
PEDESTRIAN_LABEL = "Pedestrian / Vulnerable Road User"


class MultimodalLateFusionNet:
    """Weighted late-fusion of vision class probabilities with IMU shock evidence."""

    POTHOLE_CLASS_ID = 2  # matches VisionDistressNet.CLASS_NAMES[2] == "Pothole Cavity"

    def __init__(self, imu_weight=0.4, pedestrian_veto=True):
        self.imu_weight = float(np.clip(imu_weight, 0.0, 1.0))
        self.pedestrian_veto = pedestrian_veto

    def _class_names(self, pedestrian_present):
        return list(CLASS_NAMES) + ([PEDESTRIAN_LABEL] if pedestrian_present else [])

    def fuse(self, vision_probs, imu_pothole_prob=None, imu_shock_ms2=None):
        """
        vision_probs: probability vector from VisionDistressNet, length
            len(CLASS_NAMES) or len(CLASS_NAMES)+1 if a pedestrian slot
            (index PEDESTRIAN_CLASS_ID) is included.
        imu_pothole_prob: float in [0,1] from IMUShockClassifier, or None if
            no real IMU reading was available for this frame.
        imu_shock_ms2: peak vertical shock in m/s^2, carried through for
            reporting only.
        """
        probs = np.asarray(vision_probs, dtype=np.float64).copy()
        n = probs.size
        pedestrian_present = n == len(CLASS_NAMES) + 1
        if n not in (len(CLASS_NAMES), len(CLASS_NAMES) + 1):
            raise ValueError(
                f"vision_probs must have length {len(CLASS_NAMES)} "
                f"(or {len(CLASS_NAMES) + 1} with a pedestrian slot), got {n}"
            )

        top_is_pedestrian = pedestrian_present and int(np.argmax(probs)) == PEDESTRIAN_CLASS_ID
        fusion_applied = False

        if imu_pothole_prob is not None and not (top_is_pedestrian and self.pedestrian_veto):
            imu_p = float(np.clip(imu_pothole_prob, 0.0, 1.0))
            fused_pothole = (1.0 - self.imu_weight) * probs[self.POTHOLE_CLASS_ID] + self.imu_weight * imu_p
            delta = fused_pothole - probs[self.POTHOLE_CLASS_ID]
            probs[self.POTHOLE_CLASS_ID] = fused_pothole

            other_mask = np.ones(n, dtype=bool)
            other_mask[self.POTHOLE_CLASS_ID] = False
            remaining = probs[other_mask].sum()
            if remaining > 1e-9:
                probs[other_mask] -= delta * (probs[other_mask] / remaining)

            probs = np.clip(probs, 0.0, None)
            probs = probs / probs.sum()
            fusion_applied = True

        pred_idx = int(np.argmax(probs))
        names = self._class_names(pedestrian_present)

        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        max_entropy = float(np.log(n)) if n > 1 else 1.0
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0

        color_meta = CLASS_COLORS.get(pred_idx, {"hex": "#94a3b8", "label": "UNKNOWN"})

        return {
            "predicted_class_id": pred_idx,
            "predicted_label": names[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "fused_probabilities": {names[i]: round(float(probs[i]), 4) for i in range(n)},
            "color_hex": color_meta.get("hex", "#94a3b8"),
            "epistemic_uncertainty": round(uncertainty, 4),
            "imu_fusion_applied": fusion_applied,
            "imu_shock_ms2": imu_shock_ms2,
            "method": "weighted_late_fusion_vision_imu",
        }

    def predict_multimodal(self, v_vis, v_imu=None, v_depth=None, v_can=None, v_env=None):
        """
        Backward-compatible entry point for the old 5-positional-argument
        call sites. v_depth/v_can/v_env are accepted and ignored - see the
        module docstring for why (no real depth or CAN-bus data exists in
        this project). v_vis must now be an actual class-probability vector
        from VisionDistressNet, not the old opaque 64-dim embedding, since
        nothing in this project was ever trained to decode that embedding
        into a class; passing the old shape is reported as an error instead
        of silently producing a meaningless answer.
        """
        vision_probs = np.asarray(v_vis, dtype=np.float64)
        if vision_probs.size not in (len(CLASS_NAMES), len(CLASS_NAMES) + 1):
            return {
                "error": (
                    "predict_multimodal now expects vision_probs shaped like "
                    "VisionDistressNet's class-probability vector, not the old "
                    "opaque embedding."
                ),
                "expected_length": len(CLASS_NAMES),
            }
        imu_pothole_prob = None
        if v_imu is not None:
            v_imu_arr = np.asarray(v_imu, dtype=np.float64).reshape(-1)
            if v_imu_arr.size == 1:
                imu_pothole_prob = float(v_imu_arr[0])
        return self.fuse(vision_probs, imu_pothole_prob=imu_pothole_prob)


# Backward-compatible alias - see module docstring for why this is no longer
# framed as a 5-modality cross-attention transformer.
MultimodalTransformerFusionNet = MultimodalLateFusionNet
