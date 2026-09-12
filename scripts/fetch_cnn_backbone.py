"""
Download an ImageNet-pretrained CNN for use as a feature extractor.

    python -m scripts.fetch_cnn_backbone                  # ResNet-50, 98 MB
    python -m scripts.fetch_cnn_backbone --model mobilenetv2   # 14 MB, faster

Weights come from the official ONNX Model Zoo (github.com/onnx/models), which
publishes the reference exports of these networks. They run under ONNX
Runtime, so no PyTorch is needed - which matters on machines where PyTorch
will not load.

    resnet50      98 MB   ~35 ms per image on a laptop CPU   best accuracy
    mobilenetv2   14 MB   ~10 ms per image                   for slower machines

After this, train the classifier head:  python -m training.train_cnn_head
"""

import argparse
import os
import sys
import urllib.request

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
BASE = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification"
MODELS = {
    "resnet50": (f"{BASE}/resnet/model/resnet50-v1-12.onnx", "cnn_backbone_resnet50.onnx", 98),
    "mobilenetv2": (f"{BASE}/mobilenet/model/mobilenetv2-12.onnx", "cnn_backbone_mobilenetv2.onnx", 14),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="resnet50", choices=sorted(MODELS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    url, filename, mb = MODELS[args.model]
    os.makedirs(CKPT_DIR, exist_ok=True)
    target = os.path.join(CKPT_DIR, filename)
    if os.path.exists(target) and not args.force:
        print(f"[backbone] {filename} already present ({os.path.getsize(target) / 1e6:.0f} MB). "
              f"Use --force to download again.")
    else:
        print(f"[backbone] downloading {args.model} (~{mb} MB) from the ONNX Model Zoo ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ROAD-SHIELD/3.0"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(target, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    print(f"\r    {os.path.getsize(target) / 1e6:6.1f} MB", end="", flush=True)
            print()
        except Exception as e:
            sys.exit(f"\nDownload failed: {e}")
        print(f"[backbone] saved -> {target} ({os.path.getsize(target) / 1e6:.0f} MB)")

    try:
        from models.cnn_embedder import CNNEmbedder
        import numpy as np
        emb = CNNEmbedder(prefer=(args.model,))
        print(f"[backbone] load check: {emb.describe()}")
        if emb.is_ready:
            vec = emb.embed(np.zeros((480, 640, 3), dtype=np.uint8))
            print(f"[backbone] smoke test: embedding vector of {vec.shape[0]} values")
            print("\nNext:  python -m training.train_cnn_head --compare")
    except Exception as e:
        print(f"[backbone] downloaded, but the load check failed: {e}")
        print("          install onnxruntime:  pip install onnxruntime")


if __name__ == "__main__":
    main()
