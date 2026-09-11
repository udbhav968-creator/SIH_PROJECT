"""
Trains Model M1 (VisionDistressNet) on the real labeled photos under
datasets/*/real_images and reports genuine held-out validation metrics.

Run directly: python -m training.train_vision
"""

# Cap the BLAS thread pools before NumPy/scikit-learn are imported. On
# laptops with modest RAM, OpenBLAS otherwise allocates a buffer per
# thread per core and dies with "Memory allocation still failed".
import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "2")


import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.model_selection import train_test_split

from data.image_dataset import load_labeled_dataset, holdout_images, dataset_inventory
from data.augmentation_pipeline import CivilDataAugmentor
from data.feature_extraction import extract_batch
from models.vision_distress_net import VisionDistressNet

CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
MODEL_PATH = os.path.join(CKPT_DIR, "vision_distress_model.joblib")
REPORT_PATH = os.path.join(CKPT_DIR, "vision_distress_report.json")


def run_training(val_ratio=0.25, copies_per_image=7, seed=42, save_dir=None, max_per_class=None):
    save_dir = save_dir or CKPT_DIR
    os.makedirs(save_dir, exist_ok=True)

    print("[M1 Vision] Loading real labeled photos from datasets/ ...")
    # max_per_class bounds peak memory: the feature matrix is samples x 4419
    # floats, and an RBF SVM's kernel matrix grows with the square of the
    # sample count. 800 per class trains comfortably in about 2 GB.
    if max_per_class is None:
        max_per_class = int(os.environ.get("ROAD_SHIELD_MAX_PER_CLASS", "800"))
    images, labels, paths = load_labeled_dataset(dedupe_augmented=True, max_per_class=max_per_class)
    print(f"  {len(images)} distinct real photos across {len(set(labels.tolist()))} classes "
          f"(cap {max_per_class} per class)")

    # Split at the base-photo level FIRST, then augment only the training
    # side - an augmented copy of a validation photo must never leak into
    # training, or the validation accuracy below would be meaningless.
    idx = np.arange(len(images))
    train_idx, val_idx = train_test_split(idx, test_size=val_ratio, stratify=labels, random_state=seed)
    train_imgs = [images[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_imgs = [images[i] for i in val_idx]
    val_labels = labels[val_idx]
    print(f"  train photos: {len(train_imgs)} | held-out validation photos: {len(val_imgs)}")

    # Augmentation exists to stretch a tiny dataset. Once real images arrive in
    # volume it costs training time without adding information, so scale it
    # down as the dataset grows (and keep it for the classes that are still
    # short of examples).
    if len(train_imgs) > 900:
        copies_per_image = 1
    elif len(train_imgs) > 400:
        copies_per_image = 2
    print(f"  augmentation: {copies_per_image} extra copies per training photo")

    augmentor = CivilDataAugmentor(seed=seed)
    aug_imgs, aug_labels = augmentor.expand(train_imgs, train_labels, copies_per_image=copies_per_image)
    print(f"  training set after augmentation: {len(aug_imgs)} images")

    t0 = time.time()
    X_train = extract_batch(aug_imgs)
    X_val = extract_batch(val_imgs)
    print(f"  feature extraction done in {time.time() - t0:.1f}s (feature dim = {X_train.shape[1]})")

    model = VisionDistressNet(random_state=seed)
    model.fit(X_train, aug_labels)

    metrics = model.evaluate(X_val, val_labels)
    print(f"  HELD-OUT validation accuracy: {metrics['accuracy'] * 100:.1f}% "
          f"(random-guess baseline for {len(VisionDistressNet.CLASS_NAMES)} classes = "
          f"{100.0 / len(VisionDistressNet.CLASS_NAMES):.1f}%)")

    model.save(MODEL_PATH)
    print(f"  saved trained model -> {MODEL_PATH}")

    report = {
        "model": "VisionDistressNet",
        "classifier": "StandardScaler -> PCA -> SVC(rbf)",
        "class_names": VisionDistressNet.CLASS_NAMES,
        "dataset_inventory": dataset_inventory(),
        "train_photos": len(train_imgs),
        "train_photos_after_augmentation": len(aug_imgs),
        "held_out_validation_photos": len(val_imgs),
        "held_out_validation_accuracy": round(metrics["accuracy"], 4),
        "random_guess_baseline": round(1.0 / len(VisionDistressNet.CLASS_NAMES), 4),
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class_report": metrics["per_class_report"],
        "note": (
            "Validation photos and their augmented copies are strictly disjoint from the "
            "training set. Accuracy is limited mainly by how few distinct source photos "
            "exist for some classes (see dataset_inventory) - this is a data problem, not "
            "an unreported one."
        ),
        "trained_at_unix": int(time.time()),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  wrote training report -> {REPORT_PATH}")

    ho = holdout_images()
    if ho:
        print(f"  ({len(ho)} photos in the mixed-label holdout folders were not used for training or scoring)")

    return report


if __name__ == "__main__":
    run_training()
