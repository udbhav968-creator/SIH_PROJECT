"""Semantic gate verifying pixel masks against CNN patch predictions."""
import numpy as np

POTHOLE, CRACK = 2, 1
DEFAULT_THRESHOLD = 0.5
SCALES = (0.45, 0.30)
STRIDE = 0.5


def heatmaps(classifier, rgb, scales=SCALES, stride=STRIDE):
    """Per-pixel class probabilities averaged over overlapping windows."""
    H, W = rgb.shape[:2]
    acc, cnt = None, np.zeros((H, W), np.float32)
    for s in scales:
        win = max(32, int(s * min(H, W)))
        step = max(8, int(win * stride))
        for y in range(0, max(1, H - win) + 1, step):
            for x in range(0, max(1, W - win) + 1, step):
                p = np.asarray(classifier.predict_probabilities(rgb[y:y + win, x:x + win]),
                               np.float32).reshape(-1)
                if acc is None:
                    acc = np.zeros((p.size, H, W), np.float32)
                acc[:, y:y + win, x:x + win] += p[:, None, None]
                cnt[y:y + win, x:x + win] += 1
    return acc / np.maximum(cnt, 1)[None]


def _resize(a, shape):
    import cv2
    return cv2.resize(a.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def apply(seg_out, classifier, rgb, threshold=DEFAULT_THRESHOLD):
    """Drop segmenter pothole pixels the classifier does not see as pothole.
    Returns seg_out (modified in place) with a 'semantic_gate' record."""
    hm = heatmaps(classifier, rgb)
    pot = hm[POTHOLE]
    removed = 0
    for key in ("mask", "mask_work"):
        m = seg_out.get(key)
        if m is None:
            continue
        p = pot if p_shape_ok(pot, m) else _resize(pot, m.shape[:2])
        drop = (m == POTHOLE) & (p < threshold)
        if key == "mask":
            removed = int(drop.sum())
        m = m.copy(); m[drop] = 0
        seg_out[key] = m
    mk = seg_out.get("mask")
    if mk is not None:
        for name, cls in (("crack_px", CRACK), ("pothole_px", POTHOLE)):
            if name in seg_out:
                seg_out[name] = int((mk == cls).sum())
    if "pixel_counts" in seg_out and seg_out.get("mask") is not None:
        mk = seg_out["mask"]
        seg_out["pixel_counts"] = {"crack": int((mk == CRACK).sum()), "pothole": int((mk == POTHOLE).sum())}
    seg_out["semantic_gate"] = {"threshold": threshold, "pothole_pixels_removed": removed,
                                "pothole_heat_max": round(float(pot.max()), 3)}
    return seg_out


def p_shape_ok(p, m):
    return p.shape[:2] == m.shape[:2]
