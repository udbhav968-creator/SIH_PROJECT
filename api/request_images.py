"""
Turning an image field in an API request into something safe to decode.

Two problems this closes
------------------------
1. ``CVCavityDetector.decode_image`` treats any string that names an existing
   file as a path. Passing request fields to it directly meant a client could
   send ``"image_base64": "C:/Users/.../anything.png"`` and have the server
   open files from its own disk. Here, base64 fields are decoded to bytes
   before they reach the decoder, so a base64 field can never be read as a path.

2. ``image_path`` is a real feature (the demo pages analyse photographs from
   the bundled corpus by path), but it accepted any path on the machine. It is
   now confined to the ``datasets/`` directory, checked after resolving
   symlinks and ``..`` segments.

Sizes are capped before decoding, and the pixel count is capped by Pillow's
decompression-bomb guard, so a small PNG that inflates to gigabytes is refused.
"""

import base64
import binascii
import os

from PIL import Image

MAX_IMAGE_BYTES = 20 * 1024 * 1024
# 40 MP covers an 8K frame (33 MP). Pillow raises DecompressionBombError past
# twice this; the default (89 MP) allows a single request ~270 MB of RGB.
Image.MAX_IMAGE_PIXELS = 40_000_000


class RequestImageError(ValueError):
    """The request's image field is missing, malformed, too large or out of bounds."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def decode_base64_image(value, max_bytes=MAX_IMAGE_BYTES):
    """Strict base64 (optionally a data: URI) to raw bytes."""
    if not isinstance(value, str):
        raise RequestImageError("image must be a base64 string")
    if value.startswith("data:"):
        value = value.split(",", 1)[1] if "," in value else ""
    value = "".join(value.split())  # tolerate line-wrapped base64
    # base64 inflates by 4/3; reject before allocating the decoded buffer.
    if len(value) * 3 // 4 > max_bytes:
        raise RequestImageError(f"image exceeds {max_bytes // (1024 * 1024)} MB", status=413)
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise RequestImageError("image is not valid base64") from None
    if not raw:
        raise RequestImageError("image is empty")
    return raw


def confined_path(path, root):
    """`path` resolved inside `root`, or RequestImageError. Relative paths resolve against root."""
    if not isinstance(path, str) or not path:
        raise RequestImageError("image_path must be a non-empty string")
    root_real = os.path.realpath(root)
    candidate = path if os.path.isabs(path) else os.path.join(root_real, path)
    real = os.path.realpath(candidate)
    try:
        inside = os.path.commonpath([root_real, real]) == root_real
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        raise RequestImageError("image_path must point inside the datasets directory", status=403)
    if not os.path.isfile(real):
        raise RequestImageError("image_path does not exist", status=404)
    return real


def load_rgb(image):
    """
    Bytes or a confined path from `request_image` to an HxWx3 uint8 array at
    native resolution, with EXIF orientation applied (phone photos are often
    stored sideways and rotated only by metadata).
    """
    import io

    import numpy as np
    from PIL import ImageOps, UnidentifiedImageError

    try:
        source = io.BytesIO(image) if isinstance(image, (bytes, bytearray)) else image
        with Image.open(source) as img:
            return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise RequestImageError(f"could not decode image: {exc}") from None
    except Image.DecompressionBombError:
        raise RequestImageError("image has too many pixels", status=413) from None


def request_image(body, datasets_root, b64_keys=("image_base64",), path_key="image_path",
                  required=True):
    """
    The image a request refers to: raw bytes for a base64 field, or a
    confined filesystem path for `path_key`. Returns None when absent and not
    required. Both forms are accepted by CVCavityDetector.decode_image.
    """
    for key in b64_keys:
        if body.get(key):
            return decode_base64_image(body[key])
    if path_key and body.get(path_key):
        return confined_path(body[path_key], datasets_root)
    if required:
        names = " or ".join([*b64_keys, *([path_key] if path_key else [])])
        raise RequestImageError(f"missing {names}")
    return None
