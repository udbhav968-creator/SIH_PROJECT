"""
Deep CNN image embeddings through ONNX Runtime - no PyTorch required.

The classifier's original features were hand-engineered: HOG gradients, local
binary patterns and colour histograms. This module replaces them with the
output of a convolutional network trained on ImageNet (1.28 million images),
which encodes far richer visual structure than any hand-written descriptor.

Why ONNX rather than PyTorch: the published weights run under ONNX Runtime,
a 50 MB dependency that installs cleanly everywhere, including the laptops
where PyTorch's DLLs refuse to load. Training the head on top of those
embeddings is ordinary scikit-learn, so the whole pipeline stays light.

Weights come from the official ONNX Model Zoo (github.com/onnx/models) and
are downloaded once by scripts/fetch_cnn_backbone.py into checkpoints/.

    embedder = CNNEmbedder()                    # picks whatever is on disk
    vec = embedder.embed(image_rgb)             # (1000,) float32
    mat = embedder.embed_batch(list_of_images)  # (N, 1000)

The embedding is the network's final layer. Using it as a feature vector is
standard transfer learning: the values describe what the network recognises
in the image, and a small classifier on top learns to map those descriptions
onto road-distress classes with a few hundred examples per class instead of
the hundreds of thousands a network needs from scratch.
"""

import glob
import os

import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

# ImageNet preprocessing, as the ONNX Model Zoo documents for these models.
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
    opts.enable_mem_pattern = False
    opts.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2)))
    opts.inter_op_num_threads = 1
    return opts


class CNNEmbedder:
    """ImageNet CNN used as a frozen feature extractor."""

    def __init__(self, checkpoints_dir=None, prefer=("resnet50", "mobilenetv2"), batch_size=16):
        self.load_error = None
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
        return self._session is not None

    def _load(self, prefer):
        try:
            import onnxruntime as ort
        except ImportError:
            return
        for name in prefer:
            spec = BACKBONES.get(name)
            if not spec:
                continue
            path = os.path.join(self.ckpt_dir, spec["file"])
            if not os.path.exists(path):
                continue
            try:
                self._session = ort.InferenceSession(path, _session_options(ort),
                                                     providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
                self.name = name
                self.input_size = spec["size"]
                self.dim = spec["dim"]
                return
            except Exception as e:
                # Kept so a caller can tell "the file is wrong" from "this
                # machine could not allocate" - ONNX Runtime says "bad
                # allocation" for the second, and the two need opposite
                # responses.
                self.load_error = str(e)
                print(f"[CNNEmbedder] {path} failed to load: {e}")
        # last resort: any backbone file that happens to be present
        for path in sorted(glob.glob(os.path.join(self.ckpt_dir, "cnn_backbone_*.onnx"))):
            try:
                self._session = ort.InferenceSession(path, _session_options(ort),
                                                     providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
                self.name = os.path.basename(path)
                return
            except Exception:
                continue

    # ------------------------------------------------------------------
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

    def embed(self, image_rgb):
        if not self.is_ready:
            raise RuntimeError("No CNN backbone on disk. Run scripts/fetch_cnn_backbone.py")
        batch = self.preprocess(image_rgb)[None, ...]
        return self._session.run(None, {self._input_name: batch})[0][0].astype(np.float32)

    def embed_batch(self, images, progress_every=0):
        """(N, dim) embeddings. Batched for throughput on CPU."""
        if not self.is_ready:
            raise RuntimeError("No CNN backbone on disk. Run scripts/fetch_cnn_backbone.py")
        out = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start:start + self.batch_size]
            batch = np.stack([self.preprocess(im) for im in chunk])
            out.append(self._session.run(None, {self._input_name: batch})[0].astype(np.float32))
            if progress_every and (start // self.batch_size) % progress_every == 0:
                print(f"    embedded {min(start + self.batch_size, len(images))}/{len(images)}", flush=True)
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), dtype=np.float32)

    def describe(self):
        return {"ready": self.is_ready, "backbone": self.name,
                "input_size": self.input_size, "embedding_dim": self.dim,
                "runtime": "onnxruntime" if self.is_ready else None}
