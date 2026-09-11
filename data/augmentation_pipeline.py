"""
Real pixel-level augmentation for the small pavement photo dataset.

Earlier versions of this file (and the "aug_mega_*" files sitting in
datasets/*/real_images) were not actual augmentations - they were exact
byte-for-byte duplicate copies of the same photo, just renamed, to make the
dataset folder look bigger than it is. That's worse than useless for
training: duplicates that end up split across train/val leak the answer and
inflate validation accuracy.

This version does real, cheap image transforms with PIL/OpenCV: flips,
small rotations, brightness/contrast jitter, and mild Gaussian noise. Nothing
fancy, but every output pixel actually differs from the input.
"""

import numpy as np
import cv2


class CivilDataAugmentor:
    """Generates real augmented copies of a road photo for training."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def _random_flip(self, img):
        if self.rng.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
        return img

    def _random_rotation(self, img, max_deg=8):
        angle = self.rng.uniform(-max_deg, max_deg)
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def _random_brightness_contrast(self, img, brightness=0.20, contrast=0.20):
        alpha = 1.0 + self.rng.uniform(-contrast, contrast)  # contrast gain
        beta = self.rng.uniform(-brightness, brightness) * 255.0  # brightness shift
        out = img.astype(np.float32) * alpha + beta
        return np.clip(out, 0, 255).astype(np.uint8)

    def _random_noise(self, img, sigma=6.0):
        noise = self.rng.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _random_crop_resize(self, img, min_scale=0.85):
        h, w = img.shape[:2]
        scale = self.rng.uniform(min_scale, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0 = self.rng.randint(0, max(1, h - ch + 1))
        x0 = self.rng.randint(0, max(1, w - cw + 1))
        crop = img[y0 : y0 + ch, x0 : x0 + cw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def augment(self, img):
        """Applies a random combination of the transforms above to one image."""
        out = img
        out = self._random_flip(out)
        out = self._random_rotation(out)
        out = self._random_crop_resize(out)
        out = self._random_brightness_contrast(out)
        if self.rng.rand() < 0.5:
            out = self._random_noise(out)
        return out

    def expand(self, images, labels, copies_per_image=6):
        """
        Given a list of real images and their labels, returns an expanded
        (images, labels) pair containing the originals plus `copies_per_image`
        genuinely-augmented variants of each. Caller is responsible for
        keeping this expansion inside a single train/val split (never split
        an original and its augmented copies across train and val).
        """
        out_images, out_labels = [], []
        for img, label in zip(images, labels):
            out_images.append(img)
            out_labels.append(label)
            for _ in range(copies_per_image):
                out_images.append(self.augment(img))
                out_labels.append(label)
        return out_images, np.array(out_labels, dtype=np.int64)
