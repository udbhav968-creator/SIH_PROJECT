"""
Per-vehicle camera calibration.

Every square metre, tonne and rupee this system reports comes from projecting
pixels onto the ground plane, and that projection needs four numbers: how high
the camera is, how far it is tilted down, and its focal length in pixels.

Until this module existed those numbers were constants in the source -
1.45 m, 18.4 degrees, fx = fy = 1120 - which is to say every bus in a fleet was
assumed to carry an identically mounted, identically lensed camera. It does not
matter for a demo. It matters enormously for a bill: mounting the same camera
30 cm higher changes the computed ground area of a defect by roughly 20%, and
the asphalt tonnage with it.

A profile is a small JSON file:

    {
      "device_id": "DL1PC1234-front",
      "camera_height_m": 1.52,
      "pitch_deg": 16.0,
      "fx": 1180.0, "fy": 1180.0,
      "cx": 640.0, "cy": 360.0,
      "image_width": 1280, "image_height": 720,
      "source": "checkerboard, 2026-09-12",
      "notes": "Ashok Leyland Viking, windscreen mount, driver side"
    }

stored in `checkpoints/calibration/<device_id>.json`.

Where the numbers come from
---------------------------
`fx`/`fy`/`cx`/`cy` are camera intrinsics. Three routes, in descending order of
trustworthiness:

  1. A checkerboard calibration - `scripts/calibrate_camera.py` runs OpenCV's
     standard routine over a dozen photographs of a printed chessboard.
  2. The horizontal field of view, if the manufacturer publishes it:
     fx = (W/2) / tan(HFOV/2). `from_field_of_view()` does this.
  3. The default guess below, which assumes a ~60 degree horizontal FOV.

Height and pitch are measured physically - a tape measure and a phone's
inclinometer are enough, and are more accurate than any estimate from a single
image.

Nothing here guesses silently. A profile records `source`, and the API reports
which profile answered, so a number can always be traced to the mount it
assumed.
"""

import json
import os

CALIB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "calibration")

# The historical hard-coded values, kept as an explicitly-labelled fallback so
# that behaviour does not change silently for anyone who has no profile yet.
DEFAULT_PROFILE = {
    "device_id": "default",
    "camera_height_m": 1.45,
    "pitch_deg": 18.4,
    "fx": 1120.0, "fy": 1120.0,
    "cx": 320.0, "cy": 240.0,
    "image_width": 640, "image_height": 480,
    "source": "assumed - not measured on any real vehicle",
    "notes": "Fallback. Areas and costs derived from this are estimates for an "
             "assumed mount, not measurements of a calibrated one.",
}

REQUIRED = ("camera_height_m", "pitch_deg", "fx", "fy", "cx", "cy")


class CalibrationError(ValueError):
    pass


def validate(profile):
    """Raise if a profile is unusable. Cheap insurance against silent nonsense."""
    missing = [k for k in REQUIRED if k not in profile]
    if missing:
        raise CalibrationError(f"calibration profile missing {missing}")
    if not 0.2 <= float(profile["camera_height_m"]) <= 6.0:
        raise CalibrationError("camera_height_m outside 0.2-6.0 m; check the units")
    if not -5.0 <= float(profile["pitch_deg"]) <= 80.0:
        raise CalibrationError("pitch_deg outside -5 to 80 degrees")
    for k in ("fx", "fy"):
        if not 100.0 <= float(profile[k]) <= 10000.0:
            raise CalibrationError(f"{k} outside 100-10000 px; that is not a focal length")
    return True


def from_field_of_view(width_px, height_px, hfov_deg, camera_height_m, pitch_deg,
                       device_id="from_fov", notes=""):
    """
    Intrinsics from the published horizontal field of view.

    Less accurate than a checkerboard, far better than a guess, and every phone
    and dashcam datasheet quotes an FOV.
    """
    import math
    if not 20.0 < float(hfov_deg) < 170.0:
        raise CalibrationError("hfov_deg outside 20-170 degrees")
    fx = (width_px / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
    return {
        "device_id": device_id,
        "camera_height_m": float(camera_height_m),
        "pitch_deg": float(pitch_deg),
        "fx": float(fx), "fy": float(fx),          # square pixels assumed
        "cx": width_px / 2.0, "cy": height_px / 2.0,
        "image_width": int(width_px), "image_height": int(height_px),
        "source": f"horizontal FOV {hfov_deg} deg, square pixels assumed",
        "notes": notes,
    }


def scaled_to(profile, width_px, height_px):
    """
    A profile rescaled to the resolution an image actually arrived at.

    Intrinsics are in pixels, so a profile calibrated at 1280x720 is simply
    wrong when applied to a 640x360 frame of the same scene - by a factor of
    two, straight through to the area. This is the single easiest way to be
    quietly incorrect, so it is handled explicitly rather than hoped about.
    """
    pw = float(profile.get("image_width") or width_px)
    ph = float(profile.get("image_height") or height_px)
    sx, sy = width_px / pw, height_px / ph
    out = dict(profile)
    out.update({
        "fx": float(profile["fx"]) * sx, "fy": float(profile["fy"]) * sy,
        "cx": float(profile["cx"]) * sx, "cy": float(profile["cy"]) * sy,
        "image_width": int(width_px), "image_height": int(height_px),
        "rescaled_from": [int(pw), int(ph)],
    })
    return out


def save(profile, calib_dir=None):
    validate(profile)
    d = calib_dir or CALIB_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{profile['device_id']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)
    return path


def load(device_id, calib_dir=None):
    """A stored profile, or None. Never falls back silently - the caller decides."""
    d = calib_dir or CALIB_DIR
    path = os.path.join(d, f"{device_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            profile = json.load(fh)
        validate(profile)
        return profile
    except Exception as e:
        print(f"[calibration] {path} is unusable: {e}")
        return None


def list_profiles(calib_dir=None):
    d = calib_dir or CALIB_DIR
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            p = load(os.path.splitext(name)[0], calib_dir=d)
            if p:
                out.append(p)
    return out


def resolve(device_id=None, width_px=None, height_px=None, calib_dir=None):
    """
    The profile to use, plus how it was chosen.

    Returns (profile, provenance) where provenance is one of:
        "calibrated"   a stored profile for this device
        "default"      the assumed mount; areas are estimates, and say so
    """
    profile, provenance = None, "default"
    if device_id:
        profile = load(device_id, calib_dir=calib_dir)
        if profile:
            provenance = "calibrated"
    if profile is None:
        profile = dict(DEFAULT_PROFILE)
    if width_px and height_px:
        profile = scaled_to(profile, width_px, height_px)
    return profile, provenance


def describe(profile, provenance):
    """What the API returns alongside any measured area, so it can be traced."""
    return {
        "device_id": profile.get("device_id"),
        "provenance": provenance,
        "camera_height_m": profile.get("camera_height_m"),
        "pitch_deg": profile.get("pitch_deg"),
        "focal_px": [profile.get("fx"), profile.get("fy")],
        "principal_point": [profile.get("cx"), profile.get("cy")],
        "resolution": [profile.get("image_width"), profile.get("image_height")],
        "rescaled_from": profile.get("rescaled_from"),
        "source": profile.get("source"),
        "measurement_basis": (
            "measured against this device's calibration" if provenance == "calibrated"
            else "ESTIMATE from an assumed camera mount - calibrate the device for a "
                 "measurement"),
    }
