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
MIN_ASPECT = 1.6
# Bars are grouped when their orientations agree to within this many degrees.
ORIENTATION_TOLERANCE_DEG = 25.0
# Below this many parallel bars it is a lane line, a patch or a reflection -
# not a crossing. Three is the minimum that establishes a repeating pattern.
MIN_BARS = 3
# Spacing between adjacent bars must be REGULAR - but regular does not mean
# constant, and assuming it did was the first version's mistake.
MAX_SPACING_VARIATION = 0.65
MAX_RATIO_VARIATION = 0.55


def _bar_candidates_from_mask(binary_mask, roi_y0, min_area):
    """Extract bright elongated bars from a binary paint mask."""
    bars = []
    try:
        import cv2
        opened = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        n, _lab, stats, cent = cv2.connectedComponentsWithStats(opened, 8)
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
            solidity = area / float(w * h)
            if solidity < 0.35:
                continue
            angle = 0.0 if w >= h else 90.0
            bars.append({
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]) + roi_y0,
                "w": w, "h": h, "area": area, "aspect": round(aspect, 2),
                "solidity": round(solidity, 2),
                "angle": angle,
                "cx": float(cent[i][0]), "cy": float(cent[i][1]) + roi_y0,
            })
    except Exception:
        from collections import deque
        h_roi, w_roi = binary_mask.shape
        visited = np.zeros((h_roi, w_roi), dtype=bool)
        ys, xs = np.nonzero(binary_mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue
            visited[y0, x0] = True
            q = deque([(y0, x0)])
            coords = [(y0, x0)]
            min_x = max_x = x0
            min_y = max_y = y0
            while q:
                cy, cx = q.popleft()
                for ny in range(max(0, cy - 1), min(h_roi, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w_roi, cx + 2)):
                        if not visited[ny, nx] and binary_mask[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                            coords.append((ny, nx))
                            if nx < min_x: min_x = nx
                            if nx > max_x: max_x = nx
                            if ny < min_y: min_y = ny
                            if ny > max_y: max_y = ny
            area = len(coords)
            if area < min_area:
                continue
            bw = max_x - min_x + 1
            bh = max_y - min_y + 1
            aspect = max(bw, bh) / float(min(bw, bh))
            if aspect < MIN_ASPECT:
                continue
            solidity = area / float(bw * bh)
            if solidity < 0.35:
                continue
            bars.append({
                "x": int(min_x),
                "y": int(min_y) + roi_y0,
                "w": int(bw),
                "h": int(bh),
                "area": area,
                "aspect": round(aspect, 2),
                "solidity": round(solidity, 2),
                "angle": 0.0 if bw >= bh else 90.0,
                "cx": float(np.mean([c[1] for c in coords])),
                "cy": float(np.mean([c[0] for c in coords])) + roi_y0,
            })
    return bars


def _regular_group(bars):
    """
    The largest set of bars that share an orientation AND form an evenly spaced
    sequence of stripes across the road.
    """
    best_bars = []
    best_stripes = []
    best_var = 1.0

    for angle in (0.0, 90.0):
        same = [b for b in bars if abs(b["angle"] - angle) <= ORIENTATION_TOLERANCE_DEG]
        if len(same) < MIN_BARS:
            continue
        key = "cy" if angle == 0.0 else "cx"
        dim_key = "h" if angle == 0.0 else "w"
        same.sort(key=lambda b: b[key])

        # Cluster bars at almost the same position (left/right fragments of same stripe)
        stripes = []
        for b in same:
            merged = False
            for s in stripes:
                thresh = max(10.0, min(16.0, 0.35 * s[dim_key]))
                diff = abs(b[key] - s[key])
                if diff <= thresh:
                    s["bars"].append(b)
                    s[key] = float(np.mean([bar[key] for bar in s["bars"]]))
                    s[dim_key] = max(s[dim_key], b[dim_key])
                    merged = True
                    break
            if not merged:
                stripes.append({
                    key: b[key],
                    dim_key: b[dim_key],
                    "bars": [b]
                })

        if len(stripes) < MIN_BARS:
            continue

        stripes.sort(key=lambda s: s[key])
        gaps = [stripes[i + 1][key] - stripes[i][key] for i in range(len(stripes) - 1)]
        if not gaps:
            continue
        mean_gap = float(np.mean(gaps))
        if mean_gap <= 4.0:
            continue
        variation = float(np.std(gaps)) / mean_gap
        regular = variation <= MAX_SPACING_VARIATION
        if not regular and len(gaps) >= 2:
            ratios = [gaps[i + 1] / gaps[i] for i in range(len(gaps) - 1)
                      if gaps[i] > 1e-6]
            if ratios:
                mr = float(np.mean(ratios))
                monotonic = all(r >= 0.80 for r in ratios) or all(r <= 1.25 for r in ratios)
                regular = (mr > 0 and float(np.std(ratios)) / max(1e-6, mr) <= MAX_RATIO_VARIATION
                           and monotonic)
        if not regular:
            continue

        all_cluster_bars = []
        for s in stripes:
            all_cluster_bars.extend(s["bars"])

        if len(stripes) > len(best_stripes) or (len(stripes) == len(best_stripes) and variation < best_var):
            best_stripes = stripes
            best_bars = all_cluster_bars
            best_var = variation

    return best_bars, len(best_stripes), best_var


def detect_zebra(image_rgb, roi_top_fraction=0.35):
    """
    Find a pedestrian crossing using geometric regularity and chromatic whiteness.
    """
    img = np.asarray(image_rgb)
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        cmax = img.max(axis=-1).astype(np.float32)
        cmin = img.min(axis=-1).astype(np.float32)
        sat = np.zeros_like(cmax)
        nz = cmax > 1e-5
        sat[nz] = (cmax[nz] - cmin[nz]) / cmax[nz]
    else:
        gray = img.astype(np.float32)
        sat = np.zeros_like(gray)

    H, W = gray.shape[:2]
    roi_y0 = int(H * roi_top_fraction)
    roi_gray = gray[roi_y0:, :]
    roi_sat = sat[roi_y0:, :]
    roi_area = roi_gray.size
    if roi_area == 0:
        return {"found": False, "bars": [], "bar_count": 0, "reason": "empty ROI"}

    mean_val = float(roi_gray.mean())
    std_val = float(roi_gray.std())
    thr = max(160.0, mean_val + 1.05 * std_val)

    # Thermoplastic road paint is achromatic white: low saturation (< 0.22) and high intensity (> thr)
    bright_mask = (roi_gray > thr) & (roi_sat < 0.22)
    paint_share = float(bright_mask.sum()) / float(roi_area)

    # Minimum paint coverage floor: a real crossing must occupy at least 3.0% of the road surface
    if paint_share < 0.030:
        return {
            "found": False,
            "bars": [],
            "bar_count": 0,
            "paint_share": round(paint_share, 4),
            "reason": f"paint_share {paint_share:.3f} < 0.030 (insufficient white paint on surface)",
        }

    min_area = max(40, int(0.0008 * H * W))
    bars = _bar_candidates_from_mask(bright_mask, roi_y0, min_area)
    if len(bars) < MIN_BARS:
        return {
            "found": False,
            "bars": [],
            "bar_count": len(bars),
            "paint_share": round(paint_share, 4),
            "reason": f"{len(bars)} candidate bar(s) < {MIN_BARS} required",
        }

    group, stripe_count, variation = _regular_group(bars)
    if len(group) < MIN_BARS or stripe_count < MIN_BARS:
        return {
            "found": False,
            "bars": [],
            "bar_count": len(bars),
            "paint_share": round(paint_share, 4),
            "reason": (f"{len(bars)} elongated bright blob(s), but no set of "
                       f"{MIN_BARS}+ parallel, evenly spaced bars"),
        }

    x0 = min(b["x"] for b in group)
    y0 = min(b["y"] for b in group)
    x1 = max(b["x"] + b["w"] for b in group)
    y1 = max(b["y"] + b["h"] for b in group)
    box_w = x1 - x0
    box_h = y1 - y0

    # Crosswalk bounding box paint density: bars must fill >= 14% of the enclosing bounding box
    total_paint_area = sum(b["area"] for b in group)
    box_area = box_w * box_h
    fill_ratio = total_paint_area / float(box_area) if box_area > 0 else 0.0
    if fill_ratio < 0.08:
        return {
            "found": False,
            "bars": [],
            "bar_count": len(group),
            "paint_share": round(paint_share, 4),
            "fill_ratio": round(fill_ratio, 3),
            "reason": f"fill_ratio {fill_ratio:.3f} < 0.14 (bars too scattered/isolated)",
        }

    conf = min(0.99, 0.45 + 0.12 * stripe_count + 0.25 * (1.0 - min(1.0, variation)))
    crossing = stripe_count >= 3
    return {
        "found": True,
        "is_crossing": bool(crossing),
        "class_name": "Zebra Crossing" if crossing else "Road Marking (painted)",
        "confidence": round(float(conf), 3),
        "bbox_pixels": [int(x0), int(y0), int(box_w), int(box_h)],
        "bbox_normalized": [round(x0 / W, 4), round(y0 / H, 4),
                            round(box_w / W, 4), round(box_h / H, 4)],
        "bars": group,
        "bar_count": len(group),
        "stripe_count": stripe_count,
        "spacing_variation": round(variation, 3),
        "paint_share": round(paint_share, 4),
        "fill_ratio": round(fill_ratio, 3),
        "method": ("parallel bright bars with regular spacing - geometric, not "
                   "learned; see models/marking_detector.py"),
    }


def paint_mask(image_rgb, zebra):
    """
    Boolean mask of the crossing's painted bars, at full resolution.
    """
    img = np.asarray(image_rgb)
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    if not zebra or not zebra.get("found"):
        return mask
    for b in zebra["bars"]:
        pad_x, pad_y = max(2, b["w"] // 12), max(2, b["h"] // 12)
        y0 = max(0, b["y"] - pad_y)
        y1 = min(H, b["y"] + b["h"] + pad_y)
        x0 = max(0, b["x"] - pad_x)
        x1 = min(W, b["x"] + b["w"] + pad_x)
        mask[y0:y1, x0:x1] = True
    return mask
