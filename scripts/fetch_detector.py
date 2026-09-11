"""
Fetch a COCO-pretrained object detector and export it to ONNX for this project.

    python -m scripts.fetch_detector                 # yolo11n, the balanced default
    python -m scripts.fetch_detector --model yolo11s # more accurate, ~3x slower
    python -m scripts.fetch_detector --model yolov8n # if you prefer v8

What happens: the Ultralytics package downloads the published weights (trained
on COCO: 330k images, 80 classes) and exports them to ONNX. The result lands
in checkpoints/road_shield_detector.onnx, where models/onnx_object_detector.py
picks it up automatically. Nothing is trained here — these are the authors'
published weights, used for inference.

Size and speed on a laptop CPU, 640x640 input, roughly:

    yolo11n   ~10 MB   40-70 ms per image   (default)
    yolo11s   ~35 MB   90-150 ms
    yolo11m   ~75 MB   250-400 ms

Licence note: Ultralytics YOLO models are AGPL-3.0. That is fine for a
hackathon, a demonstration or research. If this is ever deployed commercially
you need either their commercial licence or a permissively-licensed detector
(RT-DETR and DETR under Apache-2.0 are the usual substitutes).
"""

import argparse
import os
import shutil
import sys

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
TARGET_NAME = "road_shield_detector.onnx"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11n",
                    help="yolo11n/s/m, yolov8n/s/m, or a path to your own .pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--keep-name", action="store_true",
                    help="keep the exported file's own name instead of road_shield_detector.onnx")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "The ultralytics package is needed to download and export the weights:\n"
            "    pip install ultralytics\n"
            "It is only needed for this one-off export; serving uses onnxruntime alone."
        )

    os.makedirs(CKPT_DIR, exist_ok=True)
    name = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    print(f"[detector] loading {name} (downloads on first use)")
    model = YOLO(name)

    print(f"[detector] exporting to ONNX at {args.imgsz}x{args.imgsz}, opset {args.opset}")
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=False)
    exported = str(exported)
    if not os.path.exists(exported):
        sys.exit(f"Export reported success but {exported} is missing.")

    target = os.path.join(CKPT_DIR, os.path.basename(exported) if args.keep_name else TARGET_NAME)
    shutil.copy2(exported, target)
    size_mb = os.path.getsize(target) / 1e6
    print(f"[detector] ready: {target}  ({size_mb:.1f} MB)")

    try:
        from models.onnx_object_detector import ONNXObjectDetector
        det = ONNXObjectDetector(checkpoints_dir=CKPT_DIR)
        print(f"[detector] load check: {det.describe()}")
        if det.is_ready:
            import numpy as np
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            print(f"[detector] smoke test on a blank frame: {len(det.detect(blank))} detections "
                  f"(0 is the correct answer for an empty image)")
    except Exception as e:
        print(f"[detector] exported, but the load check failed: {e}")
        print("           install onnxruntime:  pip install onnxruntime")

    print("\nRestart the API and the detector is used automatically:")
    print("    python -m api.server")


if __name__ == "__main__":
    sys.path.insert(0, ENGINE_ROOT)
    main()
