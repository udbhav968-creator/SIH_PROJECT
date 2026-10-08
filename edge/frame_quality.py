"""
Is this camera frame worth analysing?

A bus camera spends much of its time producing frames the vision models cannot judge: motion blur over
speed breakers, night, low sun straight into the lens, rain on the windscreen. Analysing them wastes the
Pi's time and produces confident nonsense, so each frame is checked first on the road part of the image
(below ROAD_TOP of the height):

    sharpness   variance of the Laplacian; low means blurred
    exposure    mean brightness, and the share of pixels crushed to black or blown to white
    contrast    standard deviation of brightness; a flat grey frame is fog, glare or a dirty lens

When a frame fails, the agent does not drop the moment: it falls back to the IMU alone (the SIH design's
"IMU takes precedence in rain or darkness"), and reports a strong enough shock as an IMU-only sighting,
clearly labelled as such.

The thresholds sit near the 1st percentile of 600 of the project's road photographs (datasets 01-03, a
fixed random sample, seed 0): 6 of those 600 (1%) are turned away. Those are still photographs, not footage from a moving bus; record a day
of the real camera and re-tune with tune() before trusting the pass rate on the road.
"""
import numpy as np

ROAD_TOP = 0.35
DEFAULTS = {
    "min_sharpness": 25.0,
    "min_brightness": 35.0,
    "max_brightness": 225.0,
    "max_clipped_fraction": 0.35,
    "min_contrast": 10.0,
}


def _gray(img):
    a = np.asarray(img)
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return a.astype(np.float32)


def laplacian_variance(gray):
    g = gray
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var()) if lap.size else 0.0


def measure(img):
    g = _gray(img)
    h = g.shape[0]
    road = g[int(h * ROAD_TOP):, :]
    if road.shape[0] > 480:
        step = int(np.ceil(road.shape[0] / 480))
        road = road[::step, ::step]
    clipped = float(np.mean((road <= 8) | (road >= 247))) if road.size else 1.0
    return {
        "sharpness": round(laplacian_variance(road), 2),
        "brightness": round(float(road.mean()) if road.size else 0.0, 2),
        "contrast": round(float(road.std()) if road.size else 0.0, 2),
        "clipped_fraction": round(clipped, 4),
    }


def assess(img, thresholds=None):
    t = dict(DEFAULTS, **(thresholds or {}))
    m = measure(img)
    reasons = []
    if m["sharpness"] < t["min_sharpness"]:
        reasons.append("blurred")
    if m["brightness"] < t["min_brightness"]:
        reasons.append("too_dark")
    if m["brightness"] > t["max_brightness"]:
        reasons.append("overexposed")
    if m["clipped_fraction"] > t["max_clipped_fraction"]:
        reasons.append("clipped")
    if m["contrast"] < t["min_contrast"]:
        reasons.append("low_contrast")
    return {"usable": not reasons, "reasons": reasons, "metrics": m}


def tune(good_frames, quantile=0.05):
    """Thresholds from frames a person marked as usable: each limit sits at the given quantile of the good
    frames, so about that share of good frames would be rejected by each test."""
    ms = [measure(f) for f in good_frames]
    if not ms:
        raise ValueError("no frames")
    col = {k: np.array([m[k] for m in ms]) for k in ms[0]}
    return {
        "min_sharpness": float(np.quantile(col["sharpness"], quantile)),
        "min_brightness": float(np.quantile(col["brightness"], quantile)),
        "max_brightness": float(np.quantile(col["brightness"], 1 - quantile)),
        "max_clipped_fraction": float(np.quantile(col["clipped_fraction"], 1 - quantile)),
        "min_contrast": float(np.quantile(col["contrast"], quantile)),
    }
