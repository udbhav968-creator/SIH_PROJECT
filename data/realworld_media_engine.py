"""
Media utilities for the demo/UI endpoints.

The earlier version of this module claimed a catalog of "curated real-world
dashcam video sequences" and "realistic photo defect presets". Both were
entirely fabricated: the "video" bounding boxes came from a hand-written
looming-perspective formula with invented confidences, and the "photo
presets" were random-noise feature vectors shaped to trigger a specific
class in the old (also fake) classifier - not real images at all. Worse,
datasets/08_dashcam_video_streams, where real video clips would have to
live for a "curated video catalog" to mean anything, is empty. There is no
real video dataset in this project.

This rewrite is honest about both facts: get_video_catalog() reports
whatever real video files actually exist on disk (none, currently) instead
of fabricated titles, and get_photo_preset() hands back a real labeled
photo from datasets/*/real_images - the same photos VisionDistressNet is
trained and evaluated on - instead of a synthetic vector engineered to
produce a chosen answer.
"""
import os
import random

from data.image_dataset import sample_photo_path, dataset_inventory

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_DIR = os.path.join(ENGINE_ROOT, "datasets", "08_dashcam_video_streams")


class RealWorldMediaEngine:
    """Serves what this project actually has: real labeled photos, and an honest (empty) video catalog."""

    def __init__(self, seed=42):
        self._rng = random.Random(seed)

    def get_video_catalog(self):
        """
        Reports real dashcam video files if any exist under
        datasets/08_dashcam_video_streams. That folder currently ships
        empty, so this returns an explicit "nothing here" result instead of
        fabricated clip titles and ground-truth tracks.
        """
        files = []
        if os.path.isdir(VIDEO_DIR):
            files = sorted(f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")))
        return {
            "available": bool(files),
            "sequences": files,
            "reason": None if files else "No real dashcam video files are present in datasets/08_dashcam_video_streams.",
        }

    def get_photo_preset(self, class_id=None):
        """
        Returns one real labeled photo (path, class_id, class_name), picked
        from datasets/*/real_images - the project's actual training data -
        instead of a synthetic feature vector built to produce a chosen
        class.
        """
        choice = sample_photo_path(class_id=class_id, rng=self._rng)
        if choice is None:
            return None
        path, cid, name = choice
        return {"image_path": path, "class_id": cid, "class_name": name}

    def get_dataset_summary(self):
        """Real per-class photo counts, straight off disk - see data/image_dataset.py."""
        return dataset_inventory()
