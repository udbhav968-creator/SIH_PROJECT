"""
Frame-quality gate: skip frames not worth running the detectors on.

On a bus the camera sometimes sees nothing useful - a lens covered by mud or
a hand, the black of an unlit road, the whiteout of low sun, a frame blurred
by a pothole jolt. Running three detectors on those wastes the edge CPU and
can produce detections from noise. Three cheap checks on the road region
(the bottom 45% of the frame, downscaled to 320 px wide) decide:

  texture     variance of the Laplacian   < 5.0   defocused / featureless
  darkness    mean luminance              < 15    covered lens, unlit road
  glare       share of pixels >= 250      > 0.50  whiteout

The thresholds are calibrated, not assumed (scripts/calibrate_frame_gate.py).
On 450 real road frames from three camera sources (RDD2022 India, CDSet
dashcam, the project's own photographs) they keep 100%, and on degraded
copies of the same frames they catch 100% of covered-lens and severely
defocused frames, 98% of whiteouts and 89% of moderately defocused ones.

A fixed Laplacian threshold of 42.5, as first proposed, would have dropped
10% of those real frames: texture varies by an order of magnitude between
cameras (median 1,192 on CDSet, 127 on the project photographs), so a
threshold tuned on one camera does not transfer.

Known gap: horizontal motion blur mostly passes (vertical edges survive
it). The detectors' own confidence thresholds are the defence there.
"""

import numpy as np

TEXTURE_MIN = 5.0
DARK_MEAN_MAX = 15.0
GLARE_FRACTION_MAX = 0.50
ROI_TOP = 0.55
ANALYSIS_WIDTH = 320


def frame_quality(image_rgb):
    """Measure the road region of an RGB frame and decide whether to analyse it."""
    import cv2

    frame = np.asarray(image_rgb, dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    height, width = gray.shape
    scaled_height = max(2, round(height * ANALYSIS_WIDTH / width))
    gray = cv2.resize(gray, (ANALYSIS_WIDTH, scaled_height), interpolation=cv2.INTER_AREA)
    roi = gray[int(scaled_height * ROI_TOP):]

    texture = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    luminance = float(roi.mean())
    glare = float((roi >= 250).mean())

    reasons = []
    if luminance < DARK_MEAN_MAX:
        reasons.append("too dark: covered lens or unlit road")
    if glare > GLARE_FRACTION_MAX:
        reasons.append("glare: most of the road region is saturated")
    if texture < TEXTURE_MIN and not reasons:
        reasons.append("no texture: defocused, fogged or featureless")

    return {
        "analysable": not reasons,
        "reasons": reasons,
        "texture_laplacian_var": round(texture, 2),
        "mean_luminance": round(luminance, 1),
        "glare_fraction": round(glare, 3),
        "thresholds": {"texture_min": TEXTURE_MIN, "dark_mean_max": DARK_MEAN_MAX,
                       "glare_fraction_max": GLARE_FRACTION_MAX},
    }
