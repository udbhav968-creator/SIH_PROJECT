"""
Defect depth, estimated honestly with uncertainty intervals, backed by the
monotonic gradient-boosted photometric regressor trained in
`training/train_civil_models.py` (`checkpoints/depth_estimator_model.joblib`).

A single monocular photograph carries no direct metric range channel, so every
monocular estimate returned here carries:
  - `depth_cm`, `depth_low_cm`, `depth_high_cm`
  - `is_measurement: False` and `"ESTIMATE"` in `caveat`
  - `method`, `basis`, and `method_confidence`

Only `stereo_depth_from_disparity()` (calibrated stereo triangulation `Z = f*B/d`)
returns `is_measurement: True`.
"""

import os
import joblib
import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
MODEL_PATH = os.path.join(CKPT_DIR, "depth_estimator_model.joblib")

CRACK_SEAL_DEPTH_CM = {"central": 2.0, "low": 1.0, "high": 3.5,
                       "basis": "IRC:82 crack sealing - specification depth, not measured"}
POTHOLE_PATCH_DEPTH_CM = {"low": 2.5, "high": 12.0,
                          "basis": "IRC:SP:83 pothole patching band, placed by trained photometric contrast regressor"}

_MODEL_CACHE = None


def _get_models():
    global _MODEL_CACHE
    if _MODEL_CACHE is None and os.path.exists(MODEL_PATH):
        try:
            _MODEL_CACHE = joblib.load(MODEL_PATH)
        except Exception:
            _MODEL_CACHE = False
    return _MODEL_CACHE if isinstance(_MODEL_CACHE, dict) else None


class DepthEstimate(dict):
    """A depth with its interval and its provenance. Dict so it serialises directly."""

    @property
    def central(self):
        return self["depth_cm"]


def _estimate(central, low, high, method, basis, confidence, **extra):
    central = float(np.clip(central, low, high))
    return DepthEstimate({
        "depth_cm": round(central, 1),
        "depth_low_cm": round(float(low), 1),
        "depth_high_cm": round(float(high), 1),
        "is_measurement": False,
        "method": method,
        "basis": basis,
        "method_confidence": round(float(confidence), 2),
        "caveat": "ESTIMATE. This system has no depth sensor; a single photograph "
                  "carries no depth information. Cost scales linearly with this "
                  "number - use the interval, not the central value, for disputes.",
        **extra,
    })


def crack_depth(severity_ratio=None):
    """
    Crack repair depth.

    `severity_ratio` is optionally the crack's mask area as a fraction of the
    inspected road area - a genuine observable, unlike classifier confidence.
    Uses the trained monotonic `crack_depth_model` regressor when available.
    """
    band = CRACK_SEAL_DEPTH_CM
    if severity_ratio is None:
        return _estimate(band["central"], band["low"], band["high"],
                         method="irc82_specification_midband",
                         basis=band["basis"], confidence=0.30,
                         driver="none supplied")
    models = _get_models()
    if models and "crack_depth_model" in models:
        x = np.array([[float(np.clip(severity_ratio, 0.0, 0.30))]], dtype=np.float32)
        central = float(np.clip(models["crack_depth_model"].predict(x)[0], band["low"], band["high"]))
        method = "irc82_trained_monotonic_regressor"
    else:
        r = float(np.clip(severity_ratio, 0.0, 0.25)) / 0.25
        central = band["low"] + r * (band["high"] - band["low"])
        method = "irc82_specification_scaled_by_extent"
    return _estimate(central, band["low"], band["high"],
                     method=method,
                     basis=band["basis"] + "; placed within the band by the measured "
                                           "fraction of road area affected",
                     confidence=0.45, driver="segmented crack extent",
                     severity_ratio=round(float(severity_ratio), 4))


def pothole_depth(image_gray, mask=None, bbox=None, distance_m=None):
    """
    Pothole depth from the optical darkness and shadow gradient of the cavity,
    evaluated through the monotonic HistGradientBoostingRegressor trained on
    real pothole and crack crops in `training/train_civil_models.py`.
    """
    g = np.asarray(image_gray, dtype=np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)

    if mask is not None and np.asarray(mask, dtype=bool).any():
        sel = np.asarray(mask, dtype=bool)
        inside = g[sel]
        import cv2
        ring = cv2.dilate(sel.astype(np.uint8), np.ones((25, 25), np.uint8), 1).astype(bool) & ~sel
        outside = g[ring] if ring.any() else g[~sel]
        source = "segmentation mask"
    elif bbox is not None:
        x, y, w, h = (int(v) for v in bbox)
        inside = g[max(0, y):y + h, max(0, x):x + w].ravel()
        outside = g.ravel()
        source = "bounding box"
    else:
        return _estimate(6.0, POTHOLE_PATCH_DEPTH_CM["low"], POTHOLE_PATCH_DEPTH_CM["high"],
                         method="band_midpoint_no_observation",
                         basis=POTHOLE_PATCH_DEPTH_CM["basis"], confidence=0.15,
                         driver="no mask or box supplied")

    if inside.size == 0 or outside.size == 0:
        return _estimate(6.0, POTHOLE_PATCH_DEPTH_CM["low"], POTHOLE_PATCH_DEPTH_CM["high"],
                         method="band_midpoint_empty_region",
                         basis=POTHOLE_PATCH_DEPTH_CM["basis"], confidence=0.15,
                         driver="empty region")

    surround = float(np.median(outside))
    cavity = float(np.median(inside))
    contrast = float(np.clip((surround - cavity) / max(surround, 1.0), 0.0, 0.6)) / 0.6

    lo, hi = POTHOLE_PATCH_DEPTH_CM["low"], POTHOLE_PATCH_DEPTH_CM["high"]
    dist_flag = 1.0 if (distance_m is not None and float(distance_m) > 12.0) else 0.0
    grad_norm = float(np.clip(np.std(inside) / 40.0, 0.0, 1.0))

    models = _get_models()
    if models and "pothole_central_model" in models:
        x = np.array([[contrast, grad_norm, dist_flag]], dtype=np.float32)
        central = float(np.clip(models["pothole_central_model"].predict(x)[0], lo, hi))
        span = float(np.clip(models["pothole_span_model"].predict(x)[0], 0.5, hi - lo))
        conf = float(np.clip(models["pothole_conf_model"].predict(x)[0], 0.10, 0.60))
        method = "trained_photometric_cavity_regressor"
    else:
        central = lo + contrast * (hi - lo)
        conf = 0.25 + 0.35 * contrast
        if dist_flag > 0:
            conf *= 0.6
        span = 0.45 * (hi - lo) * (1.0 - 0.4 * contrast)
        method = "optical_darkness_to_irc_band"

    extra = {"relative_darkness": round(contrast, 3),
             "cavity_median_intensity": round(cavity, 1),
             "surround_median_intensity": round(surround, 1),
             "region_source": source}
    if distance_m is not None:
        extra["distance_m"] = round(float(distance_m), 2)
        if float(distance_m) > 12.0:
            extra["distance_penalty"] = "beyond 12 m the cue is unreliable"
    return _estimate(central, max(lo, central - span), min(hi, central + span),
                     method=method,
                     basis=POTHOLE_PATCH_DEPTH_CM["basis"],
                     confidence=min(conf, 0.6), driver="measured cavity contrast", **extra)


def stereo_depth_from_disparity(disparity_px, baseline_m, focal_px):
    """
    Calibrated stereo triangulation: Z = f * B / d.
    The only function in this module that returns `is_measurement: True`.
    """
    d = float(disparity_px)
    if d <= 0:
        raise ValueError("disparity must be positive")
    z = float(focal_px) * float(baseline_m) / d
    return DepthEstimate({
        "depth_cm": round(z * 100.0, 1),
        "depth_low_cm": round(z * 100.0 * 0.95, 1),
        "depth_high_cm": round(z * 100.0 * 1.05, 1),
        "is_measurement": True,
        "method": "stereo_triangulation",
        "basis": "Z = f*B/d from a calibrated stereo pair",
        "method_confidence": 0.9,
        "caveat": "Measured, not estimated. Interval reflects disparity quantisation.",
    })
