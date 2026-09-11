"""
Inference for the fine-tuned CNN produced by training/train_deep_vision.py.

Two runtimes are supported, tried in this order:

  1. ONNX Runtime on checkpoints/deep_vision_<arch>.onnx  - ~50 MB dependency,
     fast on CPU, and what a deployment should use.
  2. PyTorch on checkpoints/deep_vision_<arch>.pt         - used when ONNX
     Runtime isn't installed but torch is.

If neither the weights nor a runtime are present, `is_ready` stays False and
the caller falls back to the scikit-learn classifier in
models/vision_distress_net.py. Nothing here fabricates a prediction when the
model is missing.

Output dictionaries are identical in shape to VisionDistressNet's, so the
pipeline and the dashboard don't care which classifier answered - except that
`backend` says which one did.
"""

import glob
import os

import numpy as np

from models.vision_distress_net import VisionDistressNet

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class DeepVisionNet(VisionDistressNet):
    """Fine-tuned CNN classifier with the same output contract as the baseline."""

    def __init__(self, checkpoints_dir=None, img_size=224):
        super().__init__()
        self.ckpt_dir = checkpoints_dir or CKPT_DIR
        self.img_size = img_size
        self.backend = None
        self.weights_path = None
        self.class_names = list(self.CLASS_NAMES)
        self._session = None
        self._torch_model = None
        self._load()

    # ------------------------------------------------------------------
    @property
    def is_ready(self):
        return self._session is not None or self._torch_model is not None

    def _find(self, pattern):
        matches = sorted(glob.glob(os.path.join(self.ckpt_dir, pattern)))
        return matches[0] if matches else None

    def _load(self):
        onnx_path = self._find("deep_vision_*.onnx")
        if onnx_path:
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
                shape = self._session.get_inputs()[0].shape
                if isinstance(shape[-1], int) and shape[-1] > 0:
                    self.img_size = int(shape[-1])
                self.backend = f"onnxruntime:{os.path.basename(onnx_path)}"
                self.weights_path = onnx_path
                return
            except ImportError:
                pass
            except Exception as e:
                print(f"[DeepVisionNet] ONNX model found but could not be loaded: {e}")

        pt_path = self._find("deep_vision_*.pt")
        if pt_path:
            try:
                import torch
                from training.train_deep_vision import build_model
                blob = torch.load(pt_path, map_location="cpu")
                arch = blob.get("arch", "resnet18")
                names = blob.get("class_names") or self.CLASS_NAMES
                model, _head, _block = build_model(arch, len(names))
                model.load_state_dict(blob["state_dict"])
                model.eval()
                self._torch_model = model
                self.class_names = list(names)
                self.img_size = int(blob.get("img_size", self.img_size))
                self.backend = f"torch:{os.path.basename(pt_path)}"
                self.weights_path = pt_path
            except ImportError:
                pass
            except Exception as e:
                print(f"[DeepVisionNet] PyTorch weights found but could not be loaded: {e}")

    # ------------------------------------------------------------------
    def _preprocess(self, image_rgb):
        """Resize shorter side, centre-crop, normalise - matches the eval transform used in training."""
        from PIL import Image
        img = image_rgb
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img, dtype=np.uint8))
        img = img.convert("RGB")
        target = int(self.img_size * 1.15)
        w, h = img.size
        scale = target / min(w, h)
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)
        w, h = img.size
        left, top = (w - self.img_size) // 2, (h - self.img_size) // 2
        img = img.crop((left, top, left + self.img_size, top + self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)

    def predict_probabilities(self, image_rgb):
        if not self.is_ready:
            raise RuntimeError("No fine-tuned CNN available. Train one with training/train_deep_vision.py")
        batch = self._preprocess(image_rgb)
        if self._session is not None:
            logits = self._session.run(None, {self._input_name: batch})[0]
        else:
            import torch
            with torch.no_grad():
                logits = self._torch_model(torch.from_numpy(batch)).numpy()
        return _softmax(logits)[0]

    def predict_image(self, image_rgb):
        probs = self.predict_probabilities(image_rgb)
        result = self.format_probabilities(probs.reshape(1, -1))[0]
        result["backend"] = self.backend
        return result

    def predict_batch(self, images):
        return [self.predict_image(im) for im in images]

    def describe(self):
        return {
            "ready": self.is_ready,
            "backend": self.backend,
            "weights_path": self.weights_path,
            "img_size": self.img_size,
            "classes": self.class_names,
        }


def load_best_vision_model(checkpoints_dir=None, verbose=True):
    """
    Returns the strongest classifier actually available on disk: the
    fine-tuned CNN when its weights and a runtime are present, otherwise the
    scikit-learn baseline. The second return value names which one it is.
    """
    ckpt = checkpoints_dir or CKPT_DIR
    deep = DeepVisionNet(checkpoints_dir=ckpt)
    if deep.is_ready:
        if verbose:
            print(f"  ✓ Vision classifier: fine-tuned CNN ({deep.backend})")
        return deep, "deep_cnn"
    baseline = VisionDistressNet(model_path=os.path.join(ckpt, "vision_distress_model.joblib"))
    if verbose:
        print("  ✓ Vision classifier:", "HOG/LBP + SVM baseline" if baseline.is_ready else "NOT TRAINED YET")
    return baseline, ("sklearn_baseline" if baseline.is_ready else "none")
