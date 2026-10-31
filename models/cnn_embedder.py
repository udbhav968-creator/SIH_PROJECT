"""Convolutional neural network feature embedder for pavement patches."""

import glob
import os

import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BACKBONES = {
    "mobilenetv2": {"file": "cnn_backbone_mobilenetv2.onnx", "size": 224, "dim": 1000},
    "resnet50": {"file": "cnn_backbone_resnet50.onnx", "size": 224, "dim": 1000},
}


def _session_options(ort):
    """
    ONNX Runtime options chosen for a machine that is also doing other things.

    The CPU memory ARENA is the important one. By default ORT reserves a large
    pool up front and never returns it, which is right for a server doing
    millions of inferences on one fixed shape and wrong here: this process holds
    several sessions, the crops vary in size, and the arena's reservations
    stack. Observed on a laptop with 4.7 GB free - not a small machine - ORT
    still answered "bad allocation" while loading a 98 MB model, because three
    pipelines had each taken an arena first.

    Disabling it costs a few milliseconds per call and makes the difference
    between loading and not.
    """
    opts = ort.SessionOptions()

    opts.enable_cpu_mem_arena = False

    opts.enable_mem_pattern = True

    opts.intra_op_num_threads = max(1, min(2, (os.cpu_count() or 2)))
    opts.inter_op_num_threads = 1

    if os.environ.get("ROAD_SHIELD_LITE_ORT") == "1":
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return opts


class CNNEmbedder:
    """ImageNet CNN used as a frozen feature extractor."""

    def __init__(self, checkpoints_dir=None, prefer=("resnet50", "mobilenetv2"), batch_size=16):
        self.load_error = None
        self.runtime_failures = 0
        self.last_runtime_error = None
        self.ckpt_dir = checkpoints_dir or CKPT_DIR
        self.batch_size = batch_size
        self.name = None
        self.input_size = 224
        self.dim = 1000
        self._session = None
        self._input_name = None
        self._load(prefer)

    @property
    def is_ready(self):
        return self._session is not None or getattr(self, "_cv_net", None) is not None

    def _load(self, prefer):
        self._cv_net = None
        ort = None
        try:
            try:
                import sklearn
            except Exception:
                pass
            import onnxruntime as _ort
            ort = _ort
        except ImportError as e:
            self.load_error = str(e)

        for name in prefer:
            spec = BACKBONES.get(name)
            if not spec:
                continue
            path = os.path.join(self.ckpt_dir, spec["file"])
            if not os.path.exists(path):
                continue
            if ort is not None:
                try:
                    self._session = ort.InferenceSession(path, _session_options(ort),
                                                         providers=["CPUExecutionProvider"])
                    self._input_name = self._session.get_inputs()[0].name
                    self.name = name
                    self.input_size = spec["size"]
                    self.dim = spec["dim"]
                    self.load_error = None
                    return
                except Exception as e:
                    self.load_error = str(e)
                    print(f"[CNNEmbedder] {path} ORT failed, trying cv2.dnn: {e}")
            try:
                import cv2
                self._cv_net = cv2.dnn.readNetFromONNX(path)
                self.name = name
                self.input_size = spec["size"]
                self.dim = spec["dim"]
                self.load_error = None
                return
            except Exception as e:
                self.load_error = str(e)
                print(f"[CNNEmbedder] {path} failed to load: {e}")

        for path in sorted(glob.glob(os.path.join(self.ckpt_dir, "cnn_backbone_*.onnx"))):
            if ort is not None:
                try:
                    self._session = ort.InferenceSession(path, _session_options(ort),
                                                         providers=["CPUExecutionProvider"])
                    self._input_name = self._session.get_inputs()[0].name
                    self.name = os.path.basename(path)
                    self.load_error = None
                    return
                except Exception:
                    pass
            try:
                import cv2
                self._cv_net = cv2.dnn.readNetFromONNX(path)
                self.name = os.path.basename(path)
                self.load_error = None
                return
            except Exception:
                continue

    def preprocess(self, image_rgb):
        """Resize shorter side to 256, centre-crop 224, normalise - the standard recipe."""
        from PIL import Image
        img = image_rgb
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img, dtype=np.uint8))
        img = img.convert("RGB")
        target = int(self.input_size * 256 / 224)
        w, h = img.size
        scale = target / min(w, h)
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)
        w, h = img.size
        left, top = (w - self.input_size) // 2, (h - self.input_size) // 2
        img = img.crop((left, top, left + self.input_size, top + self.input_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)

    def _run_batch(self, batch):
        if self._session is not None:
            try:
                return self._session.run(None, {self._input_name: batch})[0].astype(np.float32)
            except Exception as e:
                if getattr(self, "_cv_net", None) is None:
                    try:
                        import cv2
                        spec = BACKBONES.get(self.name)
                        if spec:
                            self._cv_net = cv2.dnn.readNetFromONNX(os.path.join(self.ckpt_dir, spec["file"]))
                    except Exception:
                        pass
                if getattr(self, "_cv_net", None) is not None:
                    self._cv_net.setInput(batch)
                    return self._cv_net.forward().astype(np.float32)
                raise e
        self._cv_net.setInput(batch)
        return self._cv_net.forward().astype(np.float32)

    def embed(self, image_rgb):
        if not self.is_ready:
            raise RuntimeError("No CNN backbone on disk. Run scripts/fetch_cnn_backbone.py")
        batch = self.preprocess(image_rgb)[None, ...]
        try:
            return self._run_batch(batch)[0]
        except Exception as e:
            self.runtime_failures += 1
            self.last_runtime_error = str(e)
            raise

    def embed_batch(self, images, progress_every=0):
        """(N, dim) embeddings. Batched for throughput on CPU."""
        if not self.is_ready:
            raise RuntimeError("No CNN backbone on disk. Run scripts/fetch_cnn_backbone.py")
        out = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start:start + self.batch_size]
            batch = np.stack([self.preprocess(im) for im in chunk])
            out.append(self._run_batch(batch))
            if progress_every and (start // self.batch_size) % progress_every == 0:
                print(f"    embedded {min(start + self.batch_size, len(images))}/{len(images)}", flush=True)
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), dtype=np.float32)

    def describe(self):
        return {"ready": self.is_ready, "backbone": self.name,
                "input_size": self.input_size, "embedding_dim": self.dim,
                "runtime": "onnxruntime" if self.is_ready else None}
