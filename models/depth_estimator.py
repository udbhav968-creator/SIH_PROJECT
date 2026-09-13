"""
Defect depth, estimated honestly, with the uncertainty attached.

The problem this replaces
-------------------------
Crack depth used to be computed as

    depth_cm = 1.5 + classifier_confidence * 2.0

That is not an estimate of depth. It is the classifier's confidence with
centimetres written after it: a photograph the model was sure about produced a
"deeper" crack than one it was unsure about, which is not how roads work. It
fed straight into tonnage and cost.

Pothole depth was a darkness heuristic - better founded, since a cavity really
is darker than the road around it, but reported as a single number with no
indication that it was a guess.

What this does instead
----------------------
A single photograph from a single camera contains no depth information. That is
geometry, not a limitation of the code: any monocular depth number is an
inference from priors, and the only honest treatment is to say so and to carry
an interval.

So every estimate here returns a central value AND a plausible range AND the
basis it was derived from:

  cracks    IRC:82 classifies sealed cracking by width, and depth for costing
            purposes is the seal/fill depth, which is a specification choice,
            not a measurement. We use the specification band and say so.

  potholes  Optical depth cue: a cavity is darker than surrounding pavement
            because less light escapes it. The relationship is monotonic but
            weakly calibrated - it depends on sun angle, surface wetness and
            camera exposure. We map relative darkness onto the IRC:SP:83 band
            for pothole patching depth and report a wide interval.

Both are labelled `ESTIMATE`. Nothing in this module returns a measurement,
because nothing in this system measures depth. Getting a measurement requires
stereo, structured light, LiDAR, or a reference object of known size in frame -
`stereo_depth_from_disparity()` is here for when a second camera exists.

Why this matters commercially
-----------------------------
Tonnage is linear in depth. A 2 cm error on a 6 cm patch is a 33% error in the
asphalt bill. A contractor disputing an invoice will ask how the depth was
determined, and "the model was 84% confident" is not an answer that survives
that conversation. A stated range with a stated basis is.
"""

import numpy as np

# IRC specification bands for repair depth, in centimetres.
# These are specification choices for a repair, not measurements of a defect.
CRACK_SEAL_DEPTH_CM = {"central": 2.0, "low": 1.0, "high": 3.5,
                       "basis": "IRC:82 crack sealing - specification depth, not measured"}
POTHOLE_PATCH_DEPTH_CM = {"low": 2.5, "high": 12.0,
                          "basis": "IRC:SP:83 pothole patching band, placed by optical darkness"}


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
    Wider, more extensive cracking is sealed deeper. With nothing supplied, the
    mid-band is returned with a correspondingly wider interval.
    """
    band = CRACK_SEAL_DEPTH_CM
    if severity_ratio is None:
        return _estimate(band["central"], band["low"], band["high"],
                         method="irc82_specification_midband",
                         basis=band["basis"], confidence=0.30,
                         driver="none supplied")
    r = float(np.clip(severity_ratio, 0.0, 0.25)) / 0.25
    central = band["low"] + r * (band["high"] - band["low"])
    return _estimate(central, band["low"], band["high"],
                     method="irc82_specification_scaled_by_extent",
                     basis=band["basis"] + "; placed within the band by the measured "
                                           "fraction of road area affected",
                     confidence=0.45, driver="segmented crack extent",
                     severity_ratio=round(float(severity_ratio), 4))


def pothole_depth(image_gray, mask=None, bbox=None, distance_m=None):
    """
    Pothole depth from the optical darkness of the cavity.

    Darkness is measured over the defect's own pixels (from the segmentation
    mask where one exists, otherwise the box) relative to the surrounding
    pavement. The relationship between that contrast and true depth is
    monotonic but weakly calibrated, so the interval returned is wide on
    purpose.
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
    inside_std = float(np.std(inside))

    # Check for specular water reflection: water-filled cavity has near-zero interior texture
    # and reflects sky light, so optical darkness contrast is inverted or neutralized.
    is_specular_water = (inside_std < 7.5) and (cavity >= surround - 5.0)

    # Relative darkness in [0, 1]. Normalised by the surrounding brightness so
    # the measure is invariant to overall exposure, which it must be: the same
    # pothole photographed at noon and at dusk has the same depth.
    contrast = float(np.clip((surround - cavity) / max(surround, 1.0), 0.0, 0.6)) / 0.6

    lo, hi = POTHOLE_PATCH_DEPTH_CM["low"], POTHOLE_PATCH_DEPTH_CM["high"]
    if is_specular_water:
        central = 5.0
        contrast = 0.35
        method_name = "specular_water_cavity_irc_band"
        basis_note = "IRC:SP:83 standing water cavity - optical darkness inverted by sky reflection"
    else:
        central = lo + contrast * (hi - lo)
        method_name = "optical_darkness_to_irc_band"
        basis_note = POTHOLE_PATCH_DEPTH_CM["basis"]

    # Confidence in the METHOD, not in the value: strong contrast means the cue
    # is present, it never means the number is accurate.
    conf = 0.25 + 0.35 * contrast
    span = 0.45 * (hi - lo) * (1.0 - 0.4 * contrast)   # interval narrows as the cue strengthens
    extra = {"relative_darkness": round(contrast, 3),
             "cavity_median_intensity": round(cavity, 1),
             "surround_median_intensity": round(surround, 1),
             "region_source": source}
    if is_specular_water:
        extra["water_specular_reflection"] = True
    if distance_m is not None:
        extra["distance_m"] = round(float(distance_m), 2)
        if float(distance_m) > 12.0:
            conf *= 0.6
            extra["distance_penalty"] = "beyond 12 m the cue is unreliable"
    return _estimate(central, max(lo, central - span), min(hi, central + span),
                     method=method_name,
                     basis=basis_note,
                     confidence=min(conf, 0.6), driver="measured cavity contrast", **extra)


def stereo_depth_from_disparity(disparity_px, baseline_m, focal_px):
    """
    The real thing, for when a second camera exists.

    Z = f * B / d. This IS a measurement, and it is the only function in this
    module that returns `is_measurement: True`. Nothing currently calls it -
    it is here so the interface exists the day a stereo rig is fitted.
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
