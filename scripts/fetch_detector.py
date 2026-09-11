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
    ap.add_argument("--onnx-url", default=None,
                    help="download an already-exported .onnx from this URL instead of using PyTorch. "
                         "Use when torch will not load on this machine, or when you exported the model "
                         "elsewhere (Colab, the DGX, another laptop).")
    ap.add_argument("--onnx-file", default=None,
                    help="copy an already-exported .onnx from this path into checkpoints/")
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    target = os.path.join(CKPT_DIR, TARGET_NAME)

    # Routes that need no PyTorch at all -------------------------------------
    if args.onnx_file:
        if not os.path.exists(args.onnx_file):
            sys.exit(f"No such file: {args.onnx_file}")
        shutil.copy2(args.onnx_file, target)
        print(f"[detector] copied {args.onnx_file} -> {target}")
        return _verify(target)

    if args.onnx_url:
        import urllib.request
        print(f"[detector] downloading {args.onnx_url}")
        try:
            with urllib.request.urlopen(args.onnx_url, timeout=120) as resp, open(target, "wb") as fh:
                shutil.copyfileobj(resp, fh)
        except Exception as e:
            sys.exit(f"Download failed: {e}")
        print(f"[detector] saved -> {target} ({os.path.getsize(target) / 1e6:.1f} MB)")
        return _verify(target)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "The ultralytics package is needed to download and export the weights:\n"
            "    pip install ultralytics\n"
            "It is only needed for this one-off export; serving uses onnxruntime alone."
        )
    except OSError as e:
        sys.exit(
            f"PyTorch failed to load on this machine:\n    {e}\n\n"
            "Serving the detector only needs onnxruntime, not PyTorch. Either fix torch, or\n"
            "export the model somewhere else (Colab, the DGX, another PC) with:\n"
            "    from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640, opset=13)\n"
            "then bring the file over and run:\n"
            "    python -m scripts.fetch_detector --onnx-file path\\to\\yolo11n.onnx"
        )

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
    print(f"[detector] ready: {target}  ({os.path.getsize(target) / 1e6:.1f} MB)")
    return _verify(target)


def _verify(target):
    """Load the exported graph through ONNX Runtime and run one blank frame."""
    try:
        from models.onnx_object_detector import ONNXObjectDetector
        det = ONNXObjectDetector(checkpoints_dir=CKPT_DIR, weights_path=target)
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
