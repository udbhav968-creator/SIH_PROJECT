"""
Create a camera calibration profile for one vehicle.

Three ways, most accurate first:

    # 1. Checkerboard - the proper way. Print an A4 chessboard, photograph it
    #    a dozen times from different angles with the mounted camera.
    python -m scripts.calibrate_camera --device DL1PC1234-front \
        --checkerboard photos/calib/ --height 1.52 --pitch 16.0

    # 2. From the published field of view - one number from the datasheet.
    python -m scripts.calibrate_camera --device DL1PC1234-front \
        --hfov 78 --width 1920 --height-px 1080 --height 1.52 --pitch 16.0

    # 3. Inspect what is already stored.
    python -m scripts.calibrate_camera --list
    python -m scripts.calibrate_camera --show DL1PC1234-front

Measuring height and pitch
--------------------------
Height: tape measure, ground to the centre of the lens, vehicle unladen.
Pitch:  the angle below horizontal. A phone inclinometer laid against the
        camera body is accurate to about a degree, which is plenty - a degree
        of pitch error moves a defect's computed distance by roughly 3%.

Both are worth measuring properly once per vehicle. Everything downstream -
area, tonnage, cost - inherits them.
"""

import argparse
import glob
import json
import os
import sys

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from models.camera_calibration import (CALIB_DIR, CalibrationError, describe,
                                       from_field_of_view, list_profiles, load, save, validate)


def from_checkerboard(image_dir, pattern=(9, 6), square_mm=25.0):
    """
    OpenCV's standard intrinsic calibration over a folder of chessboard photos.

    `pattern` is the number of INNER corners, so a board with 10x7 squares is
    (9, 6). Twelve photographs from varied angles and distances is plenty;
    fewer than five and the result is not worth having.
    """
    import cv2
    import numpy as np

    paths = sorted(sum((glob.glob(os.path.join(image_dir, e))
                        for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")), []))
    if len(paths) < 5:
        raise CalibrationError(f"only {len(paths)} images in {image_dir}; need at least 5")

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * (square_mm / 1000.0)

    obj_points, img_points, shape, used = [], [], None, []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not found:
            print(f"  no board found in {os.path.basename(p)}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        used.append(os.path.basename(p))

    if len(obj_points) < 5:
        raise CalibrationError(f"board found in only {len(obj_points)} images; need 5+")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, shape, None, None)
    print(f"  board found in {len(used)}/{len(paths)} images, RMS reprojection error {rms:.3f} px")
    if rms > 1.0:
        print("  WARNING: RMS above 1 px. Reshoot with more varied angles before trusting this.")
    return {
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "image_width": int(shape[0]), "image_height": int(shape[1]),
        "rms_reprojection_px": round(float(rms), 4),
        "images_used": len(used),
        "distortion": [round(float(x), 6) for x in dist.ravel()],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="device id, e.g. DL1PC1234-front")
    ap.add_argument("--checkerboard", help="folder of chessboard photographs")
    ap.add_argument("--pattern", default="9x6", help="inner corners, e.g. 9x6")
    ap.add_argument("--square-mm", type=float, default=25.0, help="chessboard square size in mm")
    ap.add_argument("--hfov", type=float, help="horizontal field of view in degrees")
    ap.add_argument("--width", type=int, help="image width in pixels (with --hfov)")
    ap.add_argument("--height-px", type=int, help="image height in pixels (with --hfov)")
    ap.add_argument("--height", type=float, help="camera height above ground, metres")
    ap.add_argument("--pitch", type=float, help="downward tilt, degrees below horizontal")
    ap.add_argument("--notes", default="", help="vehicle, mount position, anything useful")
    ap.add_argument("--list", action="store_true", help="list stored profiles")
    ap.add_argument("--show", help="print one stored profile")
    args = ap.parse_args()

    if args.list:
        profiles = list_profiles()
        if not profiles:
            print(f"No calibration profiles in {CALIB_DIR}.\n"
                  "Every area and cost is therefore an estimate from an assumed mount.")
            return
        print(f"{len(profiles)} profile(s) in {CALIB_DIR}:")
        for p in profiles:
            print(f"  {p['device_id']:28s} h={p['camera_height_m']}m "
                  f"pitch={p['pitch_deg']}deg fx={p['fx']:.0f}  [{p.get('source', '')}]")
        return

    if args.show:
        p = load(args.show)
        if not p:
            sys.exit(f"No profile named {args.show!r}.")
        print(json.dumps(describe(p, "calibrated"), indent=2))
        return

    if not args.device:
        sys.exit("--device is required. Use --list to see what is stored.")
    if args.height is None or args.pitch is None:
        sys.exit("--height (metres) and --pitch (degrees) are required: they cannot be "
                 "recovered from the images and they drive every distance in the system.")

    if args.checkerboard:
        pw, ph = (int(x) for x in args.pattern.lower().split("x"))
        print(f"[calibrate] checkerboard in {args.checkerboard}, inner corners {pw}x{ph}")
        intr = from_checkerboard(args.checkerboard, (pw, ph), args.square_mm)
        profile = {
            "device_id": args.device,
            "camera_height_m": float(args.height),
            "pitch_deg": float(args.pitch),
            "source": f"checkerboard, {intr['images_used']} images, "
                      f"RMS {intr['rms_reprojection_px']} px",
            "notes": args.notes,
            **{k: v for k, v in intr.items() if k not in ("images_used",)},
        }
    elif args.hfov:
        if not (args.width and args.height_px):
            sys.exit("--hfov needs --width and --height-px")
        profile = from_field_of_view(args.width, args.height_px, args.hfov,
                                     args.height, args.pitch,
                                     device_id=args.device, notes=args.notes)
    else:
        sys.exit("Give either --checkerboard <folder> or --hfov <degrees>.")

    validate(profile)
    path = save(profile)
    print(f"\nsaved -> {path}")
    print(json.dumps(describe(profile, "calibrated"), indent=2))
    print("\nUse it by passing device_id on the analysis request, e.g.:")
    print(f'  {{"image_base64": "...", "device_id": "{args.device}"}}')


if __name__ == "__main__":
    main()
