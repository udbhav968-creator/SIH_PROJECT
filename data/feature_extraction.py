"""Handcrafted visual feature extraction for pavement textures."""

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

PATCH_SIZE = (96, 96)

_HOG_ORIENTATIONS = 9
_HOG_PIXELS_PER_CELL = (8, 8)
_HOG_CELLS_PER_BLOCK = (2, 2)

_LBP_POINTS = 24
_LBP_RADIUS = 3
_LBP_BINS = _LBP_POINTS + 2

_COLOR_BINS = 16


def _to_gray_patch(image_rgb):
    """Resizes an RGB uint8 image to PATCH_SIZE and returns (gray, hsv)."""
    resized = cv2.resize(image_rgb, PATCH_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    return gray, hsv


def extract_image_features(image_rgb):
    """
    Computes a fixed-length real-valued feature vector for one RGB image (or
    crop). Accepts any size/aspect ratio - it gets resized internally.

    Returns a 1D float32 numpy array.
    """
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    if image_rgb.ndim == 2:
        image_rgb = np.stack([image_rgb] * 3, axis=-1)

    gray, hsv = _to_gray_patch(image_rgb)

    hog_feat = hog(
        gray,
        orientations=_HOG_ORIENTATIONS,
        pixels_per_cell=_HOG_PIXELS_PER_CELL,
        cells_per_block=_HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    lbp = local_binary_pattern(gray, _LBP_POINTS, _LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=_LBP_BINS, range=(0, _LBP_BINS), density=True)
    lbp_hist = lbp_hist.astype(np.float32)

    h_hist = cv2.calcHist([hsv], [0], None, [_COLOR_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [_COLOR_BINS], [0, 256]).flatten()
    h_hist = (h_hist / (h_hist.sum() + 1e-6)).astype(np.float32)
    s_hist = (s_hist / (s_hist.sum() + 1e-6)).astype(np.float32)

    stats = np.array(
        [
            gray.mean(),
            gray.std(),
            np.percentile(gray, 10),
            np.percentile(gray, 90),
            cv2.Laplacian(gray, cv2.CV_64F).var(),
        ],
        dtype=np.float32,
    )
    stats /= 255.0

    return np.concatenate([hog_feat, lbp_hist, h_hist, s_hist, stats])


def feature_vector_length():
    """Returns the length of the vector extract_image_features() produces."""
    dummy = np.zeros((PATCH_SIZE[1], PATCH_SIZE[0], 3), dtype=np.uint8)
    return extract_image_features(dummy).shape[0]


def extract_batch(images_rgb):
    """Vectorized convenience wrapper: list of RGB images -> (N, D) array."""
    return np.stack([extract_image_features(img) for img in images_rgb]).astype(np.float32)
