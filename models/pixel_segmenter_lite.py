"""
The served pixel segmenter without scikit-learn or OpenCV, for the Vercel build
(numpy + Pillow + onnxruntime only).

- Features: a numpy port of models/defect_segmenter.extract_pixel_features
  (area resize, sRGB->CIELAB, Sobel, Laplacian, box blur with OpenCV's default
  reflect-101 border).
- Classifier: checkpoints/defect_segmenter_pixel.onnx, built from the fitted
  HistGradientBoostingClassifier by scripts/export_pixel_segmenter_onnx.py
  (probabilities agree with scikit-learn to < 1e-6).

How closely this reproduces the OpenCV/scikit-learn segmenter is measured by
scripts/export_pixel_segmenter_onnx.py --verify, not assumed.
"""
import json
import os

import numpy as np

WORK_W, WORK_H = 320, 200
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
ONNX_PATH = os.path.join(CKPT_DIR, "defect_segmenter_pixel.onnx")
META_PATH = os.path.join(CKPT_DIR, "defect_segmenter_pixel.json")


def _area_weights(n_in, n_out):
    """(n_out, n_in) matrix averaging each output cell's footprint - OpenCV INTER_AREA when shrinking."""
    scale = n_in / n_out
    W = np.zeros((n_out, n_in), dtype=np.float64)
    for i in range(n_out):
        a, b = i * scale, (i + 1) * scale
        lo, hi = int(np.floor(a)), int(np.ceil(b))
        for j in range(lo, min(hi, n_in)):
            W[i, j] = min(b, j + 1) - max(a, j)
        W[i] /= W[i].sum()
    return W


def _resize_area_uint8(img, w, h):
    H, W_ = img.shape[:2]
    if W_ < w or H < h:  # enlarging: OpenCV's INTER_AREA is bilinear here
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((w, h), Image.Resampling.BILINEAR))
    Wy, Wx = _area_weights(H, h), _area_weights(W_, w)
    out = np.einsum("ij,jkc->ikc", Wy, img.astype(np.float64))
    out = np.einsum("lk,ikc->ilc", Wx, out)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _rgb_to_lab_uint8(img):
    """OpenCV COLOR_RGB2LAB for 8-bit input: sRGB gamma, D65, L*255/100, a+128, b+128."""
    c = img.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]])
    xyz = lin @ M.T
    xyz[..., 0] /= 0.950456
    xyz[..., 2] /= 1.088754
    eps = 0.008856
    f = np.where(xyz > eps, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = np.where(xyz[..., 1] > eps, 116.0 * np.cbrt(xyz[..., 1]) - 16.0, 903.3 * xyz[..., 1])
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    lab = np.stack([L * 255.0 / 100.0, a + 128.0, b + 128.0], axis=-1)
    return np.clip(np.rint(lab), 0, 255).astype(np.float32)


def _conv3(src, k):
    p = np.pad(src, 1, mode="reflect")  # numpy "reflect" == OpenCV BORDER_REFLECT_101
    H, W = src.shape
    out = np.zeros_like(src, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            if k[dy][dx]:
                out += k[dy][dx] * p[dy:dy + H, dx:dx + W]
    return out


def _box(src, k):
    r = k // 2
    p = np.pad(src.astype(np.float64), r, mode="reflect")
    cs = np.cumsum(np.cumsum(np.pad(p, ((1, 0), (1, 0))), axis=0), axis=1)
    H, W = src.shape
    s = cs[k:k + H, k:k + W] - cs[0:H, k:k + W] - cs[k:k + H, 0:W] + cs[0:H, 0:W]
    return (s / (k * k)).astype(np.float32)


def extract_pixel_features(image_rgb):
    img = np.asarray(image_rgb)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    img = np.clip(img, 0, 255).astype(np.uint8)[:, :, :3]
    img = _resize_area_uint8(img, WORK_W, WORK_H)
    lab = _rgb_to_lab_uint8(img)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    gx = _conv3(L, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = _conv3(L, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad = np.sqrt(gx * gx + gy * gy)
    lap = np.abs(_conv3(L, [[2, 0, 2], [0, -8, 0], [2, 0, 2]]))

    def local_std(src, k):
        m = _box(src, k)
        sq = _box(src * src, k)
        return np.sqrt(np.maximum(sq - m * m, 0.0))

    h, w = WORK_H, WORK_W
    rows = np.repeat(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], w, axis=1)
    cols = np.repeat(np.abs(np.linspace(-1.0, 1.0, w, dtype=np.float32))[None, :], h, axis=0)
    stack = np.stack([L, a, b, grad, lap, local_std(L, 5), local_std(L, 15), L - _box(L, 31),
                      rows, cols, L - float(L.mean())], axis=-1)
    return stack.reshape(-1, stack.shape[-1]).astype(np.float32), (h, w)


_SESSION = None


def available():
    return os.path.exists(ONNX_PATH) and os.path.exists(META_PATH)


def load():
    """(onnxruntime session, metadata) or (None, None)."""
    global _SESSION
    if _SESSION is None and available():
        import onnxruntime as ort
        _SESSION = (ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"]),
                    json.load(open(META_PATH, encoding="utf-8")))
    return _SESSION or (None, None)


def predict_proba(image_rgb):
    """(H*W, 3) probabilities [sound, crack, pothole] at 320x200, or None."""
    sess, _meta = load()
    if sess is None:
        return None
    feats, _ = extract_pixel_features(image_rgb)
    return sess.run(None, {"X": feats})[1]
