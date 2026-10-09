"""
Ensemble of fine-tuned CNNs and vision transformers, served as a confidence cascade.

    first network ──(top probability >= tau)──► answer
          │
          └──(less sure)──► every member ──► mean of temperature-scaled probabilities ──► answer

Which networks, their temperatures, the first network and tau all come from checkpoints/vision_ensemble.json,
written by training/select_ensemble.py only when, on validation, the ensemble beat the best single network by
the margin its rule requires; the measurement (validation choice, one test scoring, bootstrap interval of the
gain) is in checkpoints/ensemble_report.json. Every member is an ONNX file with the same preprocessing sidecar
as the single network (models/deep_vision_net.DeepVisionNet loads each one).

Same output contract as the other classifiers; the route each photograph took is in last_route (per thread).
"""
import json
import os
import threading
import time

import numpy as np

from models.deep_vision_net import CKPT_DIR, DeepVisionNet, _softmax
from models.vision_distress_net import VisionDistressNet


class EnsembleClassifier(VisionDistressNet):
    def __init__(self, checkpoints_dir=None):
        super().__init__()
        self.ckpt_dir = checkpoints_dir or CKPT_DIR
        self.members, self.temperatures = {}, {}
        self.first, self.tau = None, 1.01
        self.load_error = None
        self.backend = None
        self.config = {}
        self._route = threading.local()
        try:
            with open(os.path.join(self.ckpt_dir, "vision_ensemble.json"), encoding="utf-8") as fh:
                self.config = json.load(fh)
        except Exception as e:
            self.load_error = f"vision_ensemble.json unreadable: {e}"
            return
        for arch in self.config.get("members") or []:
            m = DeepVisionNet(checkpoints_dir=self.ckpt_dir,
                              onnx_path=os.path.join(self.ckpt_dir, f"deep_vision_ens_{arch}.onnx"))
            if not m.is_ready:
                self.load_error = f"member {arch} did not load"
                self.members = {}
                return
            self.members[arch] = m
            self.temperatures[arch] = float((self.config.get("temperatures") or {}).get(arch, 1.0)) or 1.0
        if len(self.members) < 2:
            self.load_error = self.load_error or "fewer than two members"
            self.members = {}
            return
        self.first = self.config.get("first") if self.config.get("first") in self.members else next(iter(self.members))
        self.tau = float(self.config.get("tau", 1.01))
        names = [m.class_names for m in self.members.values()]
        if any(n != names[0] for n in names):
            self.load_error = "members disagree on the class list"
            self.members = {}
            return
        self.class_names = list(names[0])
        self.backend = "ensemble:" + "+".join(self.members) + f" (cascade from {self.first}, tau {self.tau:g})"

    @property
    def is_ready(self):
        return len(self.members) >= 2

    def _probs(self, arch, image_rgb):
        return _softmax(self.members[arch].predict_logits(image_rgb) / self.temperatures[arch])[0]

    def predict_probabilities(self, image_rgb):
        if not self.is_ready:
            raise RuntimeError(f"ensemble not available: {self.load_error}")
        t0 = time.time()
        p = self._probs(self.first, image_rgb)
        if float(p.max()) >= self.tau:
            self._route.last = {"escalated": False, "networks": [self.first], "ms": round((time.time() - t0) * 1000, 1)}
            return p
        ps = [p] + [self._probs(a, image_rgb) for a in self.members if a != self.first]
        self._route.last = {"escalated": True, "networks": list(self.members), "ms": round((time.time() - t0) * 1000, 1),
                            "first_confidence": round(float(p.max()), 4)}
        return np.mean(ps, axis=0)

    @property
    def last_route(self):
        return getattr(self._route, "last", None)

    def predict_image(self, image_rgb):
        probs = self.predict_probabilities(image_rgb)
        result = self.format_probabilities(np.asarray(probs).reshape(1, -1))[0]
        result["backend"] = self.backend
        result["cascade"] = self.last_route
        return result

    def predict_batch(self, images):
        return [self.predict_image(im) for im in images]

    def describe(self):
        return {"ready": self.is_ready, "backend": self.backend, "members": list(self.members),
                "first": self.first, "tau": self.tau, "temperatures": self.temperatures,
                "report": self.config.get("report"), "error": self.load_error}
