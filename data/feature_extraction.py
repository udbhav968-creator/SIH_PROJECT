"""
Image feature extraction for the pavement classifier.

We don't have a GPU or internet access to pull pretrained CNN weights in every
deployment environment this project runs in (some of our machines are locked
down college lab PCs), so instead of a deep network we build a classic
hand-engineered feature vector per image crop and hand it to a scikit-learn
classifier. It's the same recipe used in a lot of pre-2015 pothole detection
papers (HOG + color + a texture descriptor), just implemented with today's
libraries (skimage/opencv) instead of MATLAB.

Feature vector = [HOG] + [LBP histogram] + [HSV color histogram] + [gray
statistics]. Everything is computed on a fixed-size crop so the vector length
never changes.
"""

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

# Every crop is resized to this before any feature is computed. Small enough
# to keep HOG cheap, big enough to keep crack-shaped texture visible.
PATCH_SIZE = (96, 96)

_HOG_ORIENTATIONS = 9
_HOG_PIXELS_PER_CELL = (8, 8)
_HOG_CELLS_PER_BLOCK = (2, 2)

_LBP_POINTS = 24
_LBP_RADIUS = 3
_LBP_BINS = _LBP_POINTS + 2  # uniform LBP codes + one bucket for the rest

_COLOR_BINS = 16  # per channel, on the H and S channels of HSV


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

    # 1. HOG - captures the edge/gradient shape of cracks, potholes rims etc.
    hog_feat = hog(
        gray,
        orientations=_HOG_ORIENTATIONS,
        pixels_per_cell=_HOG_PIXELS_PER_CELL,
        cells_per_block=_HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    # 2. Local Binary Patterns - texture descriptor, good at telling smooth
    #    asphalt apart from rough/broken surface or standing water glare.
    lbp = local_binary_pattern(gray, _LBP_POINTS, _LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=_LBP_BINS, range=(0, _LBP_BINS), density=True)
    lbp_hist = lbp_hist.astype(np.float32)

    # 3. Color histogram (Hue + Saturation) - waterlogging, road paint, and
    #    rusty traffic signs all have fairly distinct color signatures.
    h_hist = cv2.calcHist([hsv], [0], None, [_COLOR_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [_COLOR_BINS], [0, 256]).flatten()
    h_hist = (h_hist / (h_hist.sum() + 1e-6)).astype(np.float32)
    s_hist = (s_hist / (s_hist.sum() + 1e-6)).astype(np.float32)

    # 4. A handful of plain gray-level statistics (contrast, brightness) -
    #    cheap but genuinely useful for the "is this even pavement" gate.
    stats = np.array(
        [
            gray.mean(),
            gray.std(),
            np.percentile(gray, 10),
            np.percentile(gray, 90),
            cv2.Laplacian(gray, cv2.CV_64F).var(),  # blur/roughness proxy
        ],
        dtype=np.float32,
    )
    stats /= 255.0  # keep everything on a similar scale to the histograms

    return np.concatenate([hog_feat, lbp_hist, h_hist, s_hist, stats])


def feature_vector_length():
    """Returns the length of the vector extract_image_features() produces."""
    dummy = np.zeros((PATCH_SIZE[1], PATCH_SIZE[0], 3), dtype=np.uint8)
    return extract_image_features(dummy).shape[0]


def extract_batch(images_rgb):
    """Vectorized convenience wrapper: list of RGB images -> (N, D) array."""
    return np.stack([extract_image_features(img) for img in images_rgb]).astype(np.float32)
