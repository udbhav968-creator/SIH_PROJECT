"""
Crack verifier: a second look at every crack the segmenter proposes.

The segmenter marks crack pixels from local texture. Paint edges, tar seams, shadows of wires and the joints
of a zebra crossing are the same kind of thin dark line, and the end-to-end measurement showed it: the
false alarms that survived on clean roads were "Crack" reports (scripts/measure_detection_quality.py). The
semantic gate (models/semantic_gate.py) checks potholes against the CNN classifier; nothing checked cracks.

This model looks at the crop around each crack component and answers one question - is this a crack in a
pavement - with a logistic regression on the MobileNetV2 ImageNet embedding (the same backbone and the same
normalisation as the input guard, models/ood_guard.py). Trained by training/train_crack_verifier.py on
    crack windows      DeepCrack and CrackForest images with pixel masks (scripts/fetch_seg_datasets.py)
    not-crack windows  crack-free areas of the same pavements, and road scenes from the clean-road folders
                       (zebra crossings, dividers, sound pavement) - never the photographs the end-to-end
                       measurement uses
Saved as numbers only (crack_verifier.npz), so it loads under any scikit-learn version.

It is used only when checkpoints/crack_verifier_selection.json says "served": true for this very model file
(its SHA-256). scripts/select_crack_gate.py writes that file after measuring the whole pipeline with and
without the verifier: it is served only if it removes false alarms without losing a single detected defect
on that sample. The decision lives in its own file, so the model's files (and their registry version) do not
change when it is made.
"""
import json
import os

import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
MODEL_FILE = "crack_verifier.npz"
REPORT_FILE = "crack_verifier_report.json"
SELECTION_FILE = "crack_verifier_selection.json"     # the serving decision, tied to the model file by SHA-256
BACKBONE = "mobilenetv2"
MIN_CROP = 48


def _sha16(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def serving_state(ckpt_dir=None):
    """{"served": bool, "why": str}: served only on a decision made for the model file now on disk."""
    ckpt_dir = ckpt_dir or CKPT_DIR
    path = os.path.join(ckpt_dir, MODEL_FILE)
    if not os.path.exists(path):
        return {"served": False, "why": f"{MODEL_FILE} not trained"}
    try:
        with open(os.path.join(ckpt_dir, SELECTION_FILE), encoding="utf-8") as fh:
            sel = json.load(fh)
    except Exception:
        return {"served": False, "why": "no serving decision yet: python -m scripts.select_crack_gate"}
    if sel.get("model_sha16") != _sha16(path):
        return {"served": False, "why": "the model file changed after the serving decision: run "
                                        "python -m scripts.select_crack_gate again"}
    return {"served": bool(sel.get("served")), "why": str(sel.get("why") or ""), "decision": sel}


def normalise(emb):
    x = np.atleast_2d(np.asarray(emb, dtype=np.float64))
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def crop_around(img, box, pad=0.25, min_side=MIN_CROP):
    """The box (x, y, w, h) with some context, at least min_side pixels on each side, clipped to the image."""
    H, W = img.shape[:2]
    x, y, w, h = [int(v) for v in box[:4]]
    px, py = int(w * pad), int(h * pad)
    cx, cy = x + w / 2.0, y + h / 2.0
    hw, hh = max(min_side / 2.0, w / 2.0 + px), max(min_side / 2.0, h / 2.0 + py)
    # near the border the window is shifted inwards rather than cut, so it keeps its size where it can
    x0 = int(min(max(0.0, cx - hw), max(0.0, W - 2 * hw)))
    y0 = int(min(max(0.0, cy - hh), max(0.0, H - 2 * hh)))
    return img[y0:min(H, int(y0 + 2 * hh)), x0:min(W, int(x0 + 2 * hw))]


class CrackVerifier:
    def __init__(self, checkpoints_dir=None, embedder=None, require_served=True):
        self.ckpt_dir = checkpoints_dir or CKPT_DIR
        self.params, self.meta, self.report, self.load_error = None, {}, {}, None
        self.embedder = embedder
        path = os.path.join(self.ckpt_dir, MODEL_FILE)
        if not os.path.exists(path):
            self.load_error = f"{MODEL_FILE} not trained - python -m training.train_crack_verifier"
            return
        try:
            with open(os.path.join(self.ckpt_dir, REPORT_FILE), encoding="utf-8") as fh:
                self.report = json.load(fh)
        except Exception:
            self.report = {}
        self.selection = serving_state(self.ckpt_dir)
        if require_served and not self.selection["served"]:
            self.load_error = "not served: " + self.selection["why"]
            return
        try:
            with np.load(path, allow_pickle=False) as z:
                p = {k: z[k] for k in z.files}
            self.meta = json.loads(str(p.pop("meta_json")))
            self.params = p
        except Exception as e:
            self.load_error = f"could not read {MODEL_FILE}: {e}"
            return
        if self.embedder is None:
            from models.cnn_embedder import CNNEmbedder
            self.embedder = CNNEmbedder(checkpoints_dir=self.ckpt_dir, prefer=(BACKBONE,))
        if not getattr(self.embedder, "is_ready", False) or getattr(self.embedder, "name", BACKBONE) != BACKBONE:
            self.load_error = f"the {BACKBONE} backbone it was trained on is not loaded"
            self.params = None

    @property
    def is_ready(self):
        return self.params is not None

    @property
    def threshold(self):
        return float(self.meta.get("threshold", 0.5))

    def proba_from_embeddings(self, emb):
        p = self.params
        z = (normalise(emb) - p["mean"]) / p["scale"]
        logit = z @ p["coef"] + float(p["intercept"])
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))

    def proba(self, crops):
        crops = [c for c in crops]
        if not crops:
            return np.zeros(0)
        return self.proba_from_embeddings(self.embedder.embed_batch(crops))

    def describe(self):
        return {"ready": self.is_ready, "reason": self.load_error, "threshold": self.meta.get("threshold"),
                "test": (self.report.get("test") or {}).get("summary"), "trained_on": self.meta.get("trained_on")}
