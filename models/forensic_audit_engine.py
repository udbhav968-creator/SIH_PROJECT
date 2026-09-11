"""
Models M7 & M8: forensic anti-fraud and repair-verification engine.

- ForensicDuplicateHasher: flags duplicate/near-duplicate photo re-submission
  ("ghost billing", where the same site photo is uploaded twice to draw two
  payouts). This used to be framed as a "DINOv2-style deep metric embedder",
  but the network was randomly initialized weights with no training data to
  fit it on - it would have produced meaningless embeddings. Duplicate-image
  detection does not need a trained model at all: a perceptual hash (a
  DCT-based average hash, in the spirit of pHash) is deterministic, needs no
  training data, and is the standard real-world tool for exactly this job -
  it is robust to re-compression and minor exposure/crop differences while
  still being sensitive to genuine content changes.
- ForensicTextureAuditor: SSIM (structural similarity) + Laplacian-variance
  surface-texture comparison between a before/after repair photo pair. This
  part of the original module was already real, standard image-processing
  math; it now optionally uses scikit-image / OpenCV's own implementations
  where available for accuracy, falling back to the original hand-written
  versions if those libraries are missing.
"""

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from skimage.metrics import structural_similarity as _sk_ssim
except ImportError:  # pragma: no cover
    _sk_ssim = None


class ForensicDuplicateHasher:
    """
    DCT-based perceptual hash ("pHash") for duplicate / near-duplicate
    photo detection.

    Two photos of the *same* physical repair, even shot moments apart with a
    different phone angle or JPEG quality, hash to a very small Hamming
    distance. Two photos of genuinely different sites do not. This is the
    right tool for "did the contractor upload the same picture twice",
    which is a similarity/duplication problem, not a classification problem
    - so it needs no labeled training set.
    """

    def __init__(self, hash_size=16, highfreq_factor=4, duplicate_hamming_threshold=6):
        self.hash_size = int(hash_size)
        self.highfreq_factor = int(highfreq_factor)
        self.duplicate_hamming_threshold = int(duplicate_hamming_threshold)

    def _to_gray_float(self, img):
        arr = np.asarray(img)
        if arr.ndim == 3:
            if cv2 is not None:
                arr = cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        return arr.astype(np.float32)

    def _resize(self, arr, side):
        if cv2 is not None:
            return cv2.resize(arr, (side, side), interpolation=cv2.INTER_AREA)
        # Nearest-neighbour fallback resize (no cv2 available).
        h, w = arr.shape
        ys = (np.arange(side) * h / side).astype(np.int32)
        xs = (np.arange(side) * w / side).astype(np.int32)
        return arr[ys][:, xs]

    @staticmethod
    def _dct2(block):
        # Separable 2D DCT-II built from a real 1D DCT-II matrix, so this
        # has no dependency on scipy being present.
        n = block.shape[0]
        k = np.arange(n).reshape(-1, 1)
        i = np.arange(n).reshape(1, -1)
        basis = np.cos(np.pi / n * (i + 0.5) * k)
        basis[0, :] *= 1.0 / np.sqrt(2.0)
        basis *= np.sqrt(2.0 / n)
        return basis @ block @ basis.T

    def compute_hash(self, img):
        """Returns a (hash_size*hash_size,) boolean array - the perceptual fingerprint."""
        side = self.hash_size * self.highfreq_factor
        gray = self._resize(self._to_gray_float(img), side)
        dct = self._dct2(gray)
        low_freq = dct[: self.hash_size, : self.hash_size]
        # DC term (top-left) dominates scale, exclude it from the median used
        # for thresholding, same as the standard pHash recipe.
        med = np.median(low_freq.flatten()[1:])
        return (low_freq > med).flatten()

    @staticmethod
    def hamming_distance(hash_a, hash_b):
        return int(np.count_nonzero(hash_a != hash_b))

    def similarity(self, img_a, img_b):
        """1.0 = pixel-identical fingerprint, 0.0 = maximally different."""
        ha, hb = self.compute_hash(img_a), self.compute_hash(img_b)
        dist = self.hamming_distance(ha, hb)
        return 1.0 - (dist / ha.size)

    def is_duplicate(self, img_a, img_b):
        ha, hb = self.compute_hash(img_a), self.compute_hash(img_b)
        dist = self.hamming_distance(ha, hb)
        return {
            "hamming_distance": dist,
            "hash_bits": int(ha.size),
            "similarity": round(1.0 - dist / ha.size, 4),
            "is_duplicate_or_near_duplicate": dist <= self.duplicate_hamming_threshold,
        }

    # --- backward-compatible surface matching the old fake-embedder API ---
    def cosine_similarity(self, emb_or_img1, emb_or_img2):
        """
        Compatibility shim for callers that previously compared two
        embedding vectors. Given two images, returns the perceptual-hash
        similarity above instead of a meaningless dot product of untrained
        embeddings.
        """
        return self.similarity(emb_or_img1, emb_or_img2)


class ForensicTextureAuditor:
    """SSIM & Laplacian surface-texture analyzer for before/after repair photos."""

    @staticmethod
    def compute_ssim(img1, img2):
        """
        Structural Similarity Index between two single-channel 2D matrices.
        Uses scikit-image's windowed SSIM when available (the standard,
        locally-windowed implementation); otherwise falls back to a single
        global-statistics SSIM, which is the same formula applied over the
        whole image at once rather than in sliding windows.
        """
        a = np.asarray(img1, dtype=np.float64)
        b = np.asarray(img2, dtype=np.float64)
        if a.shape != b.shape:
            if cv2 is not None:
                b = cv2.resize(b.astype(np.float32), (a.shape[1], a.shape[0])).astype(np.float64)
            else:
                raise ValueError("compute_ssim requires equally-shaped inputs when cv2 is unavailable")

        if _sk_ssim is not None:
            data_range = float(max(a.max(), b.max()) - min(a.min(), b.min())) or 255.0
            return float(np.clip(_sk_ssim(a, b, data_range=data_range), -1.0, 1.0))

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        mu1, mu2 = np.mean(a), np.mean(b)
        var1, var2 = np.var(a), np.var(b)
        cov = np.mean((a - mu1) * (b - mu2))
        num = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
        den = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
        return float(np.clip(num / den, -1.0, 1.0))

    @staticmethod
    def compute_laplacian_variance(img):
        """
        Discrete 2D Laplacian surface variance (sigma^2).
        Smooth rolled bitumen has low high-frequency variance (< 500);
        jagged, uncompacted patch material has high variance (> 1000).
        Uses cv2.Laplacian when available; otherwise a manually convolved
        3x3 Laplacian kernel gives an identical result.
        """
        arr = np.asarray(img, dtype=np.float32)
        if cv2 is not None:
            return float(np.var(cv2.Laplacian(arr, cv2.CV_32F, ksize=3)))

        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        h, w = arr.shape
        padded = np.pad(arr, 1, mode="edge")
        lap = np.zeros((h, w), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                lap += kernel[i, j] * padded[i:i + h, j:j + w]
        return float(np.var(lap))

    def evaluate_repair(self, img_before, img_after, claimed_dist_m=0.8):
        """Complete forensic decision gate for contractor work-order verification."""
        ssim = self.compute_ssim(img_before, img_after)
        lap_before = self.compute_laplacian_variance(img_before)
        lap_after = self.compute_laplacian_variance(img_after)

        geofence_passed = claimed_dist_m <= 2.5
        physical_alteration_passed = ssim <= 0.75  # SSIM > 0.75 => images too identical, no repair happened
        smooth_compaction_passed = lap_after <= 600.0  # smooth, well-compacted bitumen surface

        passed = geofence_passed and physical_alteration_passed and smooth_compaction_passed

        if not geofence_passed:
            verdict = "REJECTED_GEOFENCE_VIOLATION"
            reason = f"Uploaded photo coordinates are {claimed_dist_m:.1f}m away (threshold 2.5m)."
        elif not physical_alteration_passed:
            verdict = "REJECTED_GHOST_CLAIM"
            reason = f"SSIM index ({ssim:.2f} > 0.75) indicates no physical alteration occurred."
        elif not smooth_compaction_passed:
            verdict = "REJECTED_POOR_COMPACTION"
            reason = f"Surface variance ({lap_after:.1f} > 600) indicates incomplete compaction/crumbling asphalt."
        else:
            verdict = "VERIFIED_COMPLETED_REPAIR"
            reason = f"Repair verified: physical alteration confirmed (SSIM={ssim:.2f}), compaction passed (sigma^2={lap_after:.1f})."

        return {
            "ssim_index": round(ssim, 3),
            "laplacian_variance_before": round(lap_before, 1),
            "laplacian_variance_after": round(lap_after, 1),
            "geofence_distance_m": round(claimed_dist_m, 2),
            "audit_passed": passed,
            "verdict": verdict,
            "reason": reason,
        }

    def verify_repair(self, img_before, img_after, embedder=None, claimed_dist_m=0.8):
        """
        Full verification: runs the texture audit, and - when a duplicate
        hasher is supplied - also checks whether the "after" photo is just
        the "before" photo re-submitted unchanged (the clearest possible
        fraud signal, distinct from a merely-high SSIM on a real photo).
        """
        result = self.evaluate_repair(img_before, img_after, claimed_dist_m=claimed_dist_m)
        if embedder is not None:
            dup = embedder.is_duplicate(img_before, img_after)
            result["duplicate_submission_check"] = dup
            if dup["is_duplicate_or_near_duplicate"] and result["verdict"] == "VERIFIED_COMPLETED_REPAIR":
                result["verdict"] = "REJECTED_GHOST_CLAIM"
                result["audit_passed"] = False
                result["reason"] = (
                    f"Before/after photos are near-duplicates (Hamming distance "
                    f"{dup['hamming_distance']}) despite passing the SSIM check - likely the same photo re-submitted."
                )
        return result


# Backward-compatible alias - see class docstring for why this is no longer
# framed as a trained "deep metric embedder".
ForensicMetricEmbedder = ForensicDuplicateHasher
