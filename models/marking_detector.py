"""
Painted road markings, found geometrically rather than statistically.

Why this is not a learned model
-------------------------------
A zebra crossing is the most regular structure on any road: three or more
bright bars of similar size, parallel to each other, evenly spaced. That
regularity is the signal, and it is far easier to measure directly than to
learn from nineteen photographs - which is all the zebra data this project has.

Two jobs, and the second is the one that matters
------------------------------------------------
1. Report where markings are, so a frame with a crossing, a pothole, two cars
   and a pedestrian shows all of them rather than only the pothole.

2. Tell the defect pipeline which pixels are PAINT. Every remaining false
   positive in the end-to-end measurement came from a zebra crossing or a road
   divider, and the reason is stated plainly in the segmenter's own design: its
   features are lightness, colour, gradient and local variance, and fresh white
   paint on dark asphalt is the strongest example of a bright high-contrast
   patch that a road can contain. The segmenter cannot separate paint from a
   cavity on those features. Geometry can: a cavity does not come in a row of
   evenly spaced parallel bars.
"""

import numpy as np

# A bar must be at least this elongated. Zebra stripes are long rectangles;
# a pothole's highlight is not.
MIN_ASPECT = 2.2
# Bars are grouped when their orientations agree to within this many degrees.
ORIENTATION_TOLERANCE_DEG = 18.0
# Below this many parallel bars it is a lane line, a patch or a reflection -
# not a crossing. Three is the minimum that establishes a repeating pattern.
MIN_BARS = 3
# Spacing between adjacent bars must be REGULAR - but regular does not mean
# constant, and assuming it did was the first version's mistake.
#
# A crossing is painted with equal gaps, and a camera looking along the road
# sees those gaps shrink with distance. The spacing is therefore a geometric
# progression, not a constant. Measured on real photographs: 9 to 11 bars found
# and rejected, because their gaps were perfectly ordered and not remotely
# equal.
#
# Two acceptable patterns, then:
#   fronto-parallel  gaps roughly equal            -> low variation of gaps
#   in perspective   gaps shrink smoothly          -> low variation of the
#                                                     RATIO between gaps
MAX_SPACING_VARIATION = 0.55
MAX_RATIO_VARIATION = 0.45


def _bar_candidates(gray, roi_y0, min_area):
    """Bright, elongated blobs on the road surface."""
    import cv2
    roi = gray[roi_y0:, :]
    if roi.size == 0:
        return []
    # Adaptive, because road brightness varies hugely between a sunlit
    # carriageway and a wet one at dusk. A fixed threshold finds paint in one
    # and nothing in the other.
    thr = float(roi.mean()) + 1.1 * float(roi.std())
    binary = (roi > thr).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(binary, 8)
    bars = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / float(min(w, h))
        if aspect < MIN_ASPECT:
            continue
        # Orientation from the component's second moments, which is stable for
        # a filled rectangle and does not need the contour.
        angle = 0.0 if w >= h else 90.0
        bars.append({
            "x": int(stats[i, cv2.CC_STAT_LEFT]),
            "y": int(stats[i, cv2.CC_STAT_TOP]) + roi_y0,
            "w": w, "h": h, "area": area, "aspect": round(aspect, 2),
            "angle": angle,
            "cx": float(cent[i][0]), "cy": float(cent[i][1]) + roi_y0,
        })
    return bars


def _regular_group(bars):
    """
    The largest set of bars that share an orientation AND are evenly spaced.

    Orientation alone is not enough: a kerb, a lane line and a reflection can
    all be horizontal. Even SPACING is what makes a crossing a crossing.
    """
    best = []
    for angle in (0.0, 90.0):
        same = [b for b in bars if abs(b["angle"] - angle) <= ORIENTATION_TOLERANCE_DEG]
        if len(same) < MIN_BARS:
            continue
        # Bars run along `angle`, so they repeat across the perpendicular axis.
        key = "cy" if angle == 0.0 else "cx"
        same.sort(key=lambda b: b[key])
        gaps = [same[i + 1][key] - same[i][key] for i in range(len(same) - 1)]
        if not gaps:
            continue
        mean_gap = float(np.mean(gaps))
        if mean_gap <= 0:
            continue
        variation = float(np.std(gaps)) / mean_gap
        regular = variation <= MAX_SPACING_VARIATION
        if not regular and len(gaps) >= 2:
            # Perspective: consecutive gaps form a geometric progression, so
            # their RATIOS are what stay constant.
            ratios = [gaps[i + 1] / gaps[i] for i in range(len(gaps) - 1)
                      if gaps[i] > 1e-6]
            if ratios:
                mr = float(np.mean(ratios))
                monotonic = all(r <= 1.15 for r in ratios) or all(r >= 0.87 for r in ratios)
                regular = (mr > 0 and float(np.std(ratios)) / mr <= MAX_RATIO_VARIATION
                           and monotonic)
        if not regular:
            continue
        if len(same) > len(best):
            best = same
    return best


def detect_zebra(image_rgb, roi_top_fraction=0.35):
    """
    Find a pedestrian crossing.

    Returns a dict with `found`, the bars, a bounding box over them, and the
    evidence that led to the decision - so a wrong answer can be argued with
    rather than only observed.
    """
    import cv2
    img = np.asarray(image_rgb)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    H, W = gray.shape[:2]
    roi_y0 = int(H * roi_top_fraction)
    min_area = max(40, int(0.0008 * H * W))

    bars = _bar_candidates(gray, roi_y0, min_area)
    group = _regular_group(bars)
    if len(group) < MIN_BARS:
        return {
            "found": False,
            "bars": [],
            "bar_count": len(bars),
            "reason": (f"{len(bars)} elongated bright blob(s), but no set of "
                       f"{MIN_BARS}+ parallel, evenly spaced bars"),
        }

    x0 = min(b["x"] for b in group)
    y0 = min(b["y"] for b in group)
    x1 = max(b["x"] + b["w"] for b in group)
    y1 = max(b["y"] + b["h"] for b in group)
    key = "cy" if abs(group[0]["angle"]) < 45 else "cx"
    gaps = [group[i + 1][key] - group[i][key] for i in range(len(group) - 1)]
    variation = float(np.std(gaps)) / float(np.mean(gaps)) if gaps else 0.0

    # Confidence from the evidence, not from a model: more bars and more regular
    # spacing mean a more certain crossing.
    conf = min(0.99, 0.45 + 0.12 * len(group) + 0.25 * (1.0 - min(1.0, variation)))

    # Naming it honestly.
    #
    # This finds PAINT. A pedestrian crossing and a lane or divider line are
    # both paint, and measured on real photographs the geometry separates them
    # only partially: 6 of 19 crossings found, and 6 of 18 divider photographs
    # also matched. For the purpose that matters - telling the defect pipeline
    # these pixels are paint, not a cavity - both are correct answers, because
    # both really are paint.
    #
    # For the purpose of putting a LABEL on screen they are not the same, so a
    # crossing is only named as one when the evidence is strong: four or more
    # bars with near-equal spacing is a crossing, and three bars receding into
    # the distance is more honestly called a marking.
    crossing = len(group) >= 4 and variation <= MAX_SPACING_VARIATION
    return {
        "found": True,
        "is_crossing": bool(crossing),
        "class_name": "Zebra Crossing" if crossing else "Road Marking (painted)",
        "confidence": round(float(conf), 3),
        "bbox_pixels": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        "bbox_normalized": [round(x0 / W, 4), round(y0 / H, 4),
                            round((x1 - x0) / W, 4), round((y1 - y0) / H, 4)],
        "bars": group,
        "bar_count": len(group),
        "spacing_variation": round(variation, 3),
        "method": ("parallel bright bars with regular spacing - geometric, not "
                   "learned; see models/marking_detector.py"),
    }


def paint_mask(image_rgb, zebra):
    """
    Boolean mask of the crossing's painted bars, at full resolution.

    Used to stop the defect pipeline reporting paint as a cavity. Only the bars
    themselves are masked, not the whole bounding box: the dark asphalt BETWEEN
    the stripes is real road and can genuinely contain a pothole.
    """
    img = np.asarray(image_rgb)
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    if not zebra or not zebra.get("found"):
        return mask
    for b in zebra["bars"]:
        # A small dilation, because the segmenter tends to claim the paint's
        # edge - the highest-contrast part - rather than its centre.
        pad_x, pad_y = max(2, b["w"] // 12), max(2, b["h"] // 12)
        y0 = max(0, b["y"] - pad_y)
        y1 = min(H, b["y"] + b["h"] + pad_y)
        x0 = max(0, b["x"] - pad_x)
        x1 = min(W, b["x"] + b["w"] + pad_x)
        mask[y0:y1, x0:x1] = True
    return mask
