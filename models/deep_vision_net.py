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


class CNNHeadClassifier(VisionDistressNet):
    """
    ImageNet CNN embeddings (ONNX Runtime) + the scikit-learn head trained by
    training/train_cnn_head.py. Same output contract as every other classifier
    here, so the pipeline and dashboard don't need to know which one answered.

    A head is only usable with the backbone it was trained on - the embedding
    spaces of ResNet-50 and MobileNetV2 are unrelated, and feeding one head the
    other's vectors produces confident nonsense. `_load` therefore pairs each
    saved head with its own backbone file and skips any head whose backbone is
    missing, rather than silently substituting whatever happens to be on disk.
    """

    def __init__(self, checkpoints_dir=None):
        super().__init__()
        self.ckpt_dir = checkpoints_dir or CKPT_DIR
        self.backend = None
        self.head = None
        self.embedder = None
        self.report = None
        self._load()

    @property
    def is_ready(self):
        return self.head is not None and self.embedder is not None and self.embedder.is_ready

    def _candidates(self):
        """Saved heads, newest naming first, de-duplicated by path."""
        paths = sorted(glob.glob(os.path.join(self.ckpt_dir, "cnn_head_*.joblib")))
        legacy = os.path.join(self.ckpt_dir, "cnn_head_model.joblib")
        if os.path.exists(legacy) and legacy not in paths:
            paths.append(legacy)
        return paths

    def _load(self):
        try:
            import joblib
            from models.cnn_embedder import BACKBONES, CNNEmbedder
        except ImportError as e:
            print(f"[CNNHead] dependency missing: {e}")
            return

        best = None  # (macro_f1, path, blob, backbone)
        for path in self._candidates():
            try:
                blob = joblib.load(path)
            except Exception as e:
                print(f"[CNNHead] {os.path.basename(path)} could not be read: {e}")
                continue
            backbone = blob.get("backbone")
            spec = BACKBONES.get(backbone)
            if spec is None:
                continue
            if not os.path.exists(os.path.join(self.ckpt_dir, spec["file"])):
                # head trained on a backbone this machine doesn't have - skip it
                continue
            # Mean of accuracy and macro-F1. Macro-F1 alone decides on classes
            # with two or three held-out examples, where a single image moves it
            # by 0.1; accuracy alone ignores the rare classes entirely.
            score = 0.5 * (float(blob.get("macro_f1") or 0.0) + float(blob.get("accuracy") or 0.0))
            if best is None or score > best[0]:
                best = (score, path, blob, backbone)

        if best is None:
            if self._candidates():
                print("[CNNHead] head found but its backbone is missing - "
                      "run: python -m scripts.fetch_cnn_backbone")
            return

        _score, path, blob, backbone = best
        embedder = CNNEmbedder(checkpoints_dir=self.ckpt_dir, prefer=(backbone,))
        if not embedder.is_ready or embedder.name != backbone:
            # Record WHY, and whether the machine or the file is at fault.
            #
            # This fallback is silent by design - a missing backbone should not
            # stop the product - but it swaps the classifier for a different,
            # weaker one. On a machine that had run out of memory, ONNX Runtime
            # answered "bad allocation", the hand-crafted path took over, and a
            # test suite then reported zebra crossings as potholes. The suite was
            # measuring a model nobody intended to ship.
            err = getattr(embedder, "load_error", "") or ""
            self.backbone_fallback = {
                "requested": backbone,
                "loaded": getattr(embedder, "name", None),
                "error": err,
                "environment_failure": any(k in err.lower() for k in
                                           ("bad allocation", "memory", "alloc")),
            }
            print(f"[CNNHead] backbone {backbone} would not load - falling back")
            if self.backbone_fallback["environment_failure"]:
                print("[CNNHead] the reason is MEMORY, not the model file. The "
                      "hand-crafted classifier is now active, and it is a "
                      "different, weaker model - any accuracy measured now is "
                      "not this system's accuracy.")
            return
        self.backbone_fallback = None
        self.embedder = embedder
        self.head = blob["head"]
        self.report = {k: v for k, v in blob.items() if k != "head"}
        self.backend = f"cnn:{backbone}+{blob.get('head_name', 'head')}"

    def predict_probabilities(self, image_rgb):
        if not self.is_ready:
            raise RuntimeError("CNN head not available")
        vec = self.embedder.embed(image_rgb).reshape(1, -1)
        if hasattr(self.head, "predict_proba"):
            return self.head.predict_proba(vec)[0]
        # decision_function fallback, squashed to a distribution
        scores = np.asarray(self.head.decision_function(vec)).reshape(-1)
        e = np.exp(scores - scores.max())
        return e / e.sum()

    def predict_image(self, image_rgb):
        probs = self.predict_probabilities(image_rgb)
        result = self.format_probabilities(np.asarray(probs).reshape(1, -1))[0]
        result["backend"] = self.backend
        return result

    def predict_batch(self, images):
        return [self.predict_image(im) for im in images]

    def describe(self):
        return {"ready": self.is_ready, "backend": self.backend,
                "held_out_accuracy": (self.report or {}).get("accuracy"),
                "held_out_macro_f1": (self.report or {}).get("macro_f1"),
                "embedder": self.embedder.describe() if self.embedder else None}


def load_best_vision_model(checkpoints_dir=None, verbose=True):
    """
    Returns the strongest classifier actually available on disk: the
    fine-tuned CNN when its weights and a runtime are present, otherwise the
    scikit-learn baseline. The second return value names which one it is.
    """
    ckpt = checkpoints_dir or CKPT_DIR

    # 1. fine-tuned CNN (PyTorch/ONNX), if one was trained
    # 2. ImageNet CNN embeddings + trained head  <- usually available
    # 3. hand-crafted features + SVM baseline
    cnn_head = CNNHeadClassifier(checkpoints_dir=ckpt)
    if cnn_head.is_ready:
        if verbose:
            print(f"  ✓ Vision classifier: deep CNN embeddings ({cnn_head.backend})")
        return cnn_head, "cnn_embeddings"

    deep = DeepVisionNet(checkpoints_dir=ckpt)
    if deep.is_ready:
        if verbose:
            print(f"  ✓ Vision classifier: fine-tuned CNN ({deep.backend})")
        return deep, "deep_cnn"
    baseline = VisionDistressNet(model_path=os.path.join(ckpt, "vision_distress_model.joblib"))
    if verbose:
        print("  ✓ Vision classifier:", "HOG/LBP + SVM baseline" if baseline.is_ready else "NOT TRAINED YET")
    return baseline, ("sklearn_baseline" if baseline.is_ready else "none")
