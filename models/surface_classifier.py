"""Pavement material and road domain classifier (SurfaceTypeNet).

Classifies roadway surfaces into standard civil engineering pavement categories:
  1. Flexible Bituminous Asphalt (IRC:37 / IRC:82 / MoRTH Section 500)
  2. Rigid Cement Concrete (IRC:58 / IRC:15 / MoRTH Section 600)
  3. Unpaved Rural Earthen / Dirt / Gravel Track (IRC:SP:72 / IRC:SP:20 - Rural Road Manual)
  4. Interlocking Concrete Paver Blocks / ICBP (IRC:SP:63)

Switches civil engineering repair standards, cost ledgers, and patch limits
dynamically based on the physical surface material.
"""

import os
import warnings
import joblib
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

_CKPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "advanced_5_models.joblib"))
_CACHED_BUNDLE = None
_CACHED_MTIME = None


def load_advanced_5_bundle():
    """Loads and caches `checkpoints/advanced_5_models.joblib` once per mtime."""
    global _CACHED_BUNDLE, _CACHED_MTIME
    if not os.path.exists(_CKPT_PATH):
        return {}
    try:
        mtime = os.path.getmtime(_CKPT_PATH)
        if _CACHED_BUNDLE is not None and _CACHED_MTIME == mtime:
            return _CACHED_BUNDLE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _CACHED_BUNDLE = joblib.load(_CKPT_PATH) or {}
            _CACHED_MTIME = mtime
        return _CACHED_BUNDLE
    except Exception:
        return {}


SURFACE_CLASSES = {
    0: {
        "id": 0,
        "code": "flexible_bituminous_asphalt",
        "name": "Flexible Bituminous Pavement (Asphalt)",
        "civil_standard": "IRC:37-2018 / IRC:82-2015",
        "morth_section": "MoRTH Section 500 (Bituminous Base & Surface Courses)",
        "repair_method": "Mechanical saw-cutting, VG-30 tack coat, and hot-mix Bituminous Concrete (BC) patching",
        "bitumen_patching_applicable": True,
        "max_single_patch_m2": 2.0,
        "base_unit_rate_inr_per_tonne": 7500.0,
    },
    1: {
        "id": 1,
        "code": "rigid_cement_concrete",
        "name": "Rigid Cement Concrete Pavement",
        "civil_standard": "IRC:58-2015 / IRC:15-2017",
        "morth_section": "MoRTH Section 600 (Concrete Pavement Construction)",
        "repair_method": "Full-depth slab replacement, dowel bar retrofitting, rapid-hardening epoxy mortar stitch",
        "bitumen_patching_applicable": False,
        "max_single_patch_m2": 4.5,
        "base_unit_rate_inr_per_tonne": 9200.0,
    },
    2: {
        "id": 2,
        "code": "unpaved_rural_earthen_dirt",
        "name": "Unpaved Rural Earthen / Dirt / Gravel Road",
        "civil_standard": "IRC:SP:72-2015 / IRC:SP:20 (PMGSY Rural Road Manual)",
        "morth_section": "IRC:SP:20 Cl. 8 (Maintenance of Earthen & Gravel Roads)",
        "repair_method": "Motor-grader scarifying, subgrade camber reshaping, gravel replenishment, and drainage ditch restoration (Bituminous patching NOT applicable)",
        "bitumen_patching_applicable": False,
        "max_single_patch_m2": 15.0,
        "base_unit_rate_inr_per_tonne": 0.0,
        "rural_grading_cost_per_m2": 45.0,
    },
    3: {
        "id": 3,
        "code": "interlocking_concrete_paver",
        "name": "Interlocking Concrete Block Pavement (ICBP)",
        "civil_standard": "IRC:SP:63-2018",
        "morth_section": "IRC:SP:63 Cl. 9 (Maintenance & Re-laying of Paver Blocks)",
        "repair_method": "Lifting damaged blocks, regrading bed sand layer, relaying blocks, and vibro-compaction joint sand replenishment",
        "bitumen_patching_applicable": False,
        "max_single_patch_m2": 6.0,
        "base_unit_rate_inr_per_tonne": 0.0,
        "paver_relay_cost_per_m2": 380.0,
    },
}


class SurfaceTypeClassifier:
    """Multispectral colorimetry and structural texture analyzer for pavement classification."""

    @property
    def is_ready(self):
        return True

    def __init__(self):
        self.classes = SURFACE_CLASSES
        bundle = load_advanced_5_bundle()
        self._trained_model = (bundle.get("surface_classifier") or {}).get("classifier")

    def analyze_surface(self, img_rgb, roi_top_fraction=0.38):
        """
        Classifies the pavement surface from the road region of interest (ROI).

        Parameters:
            img_rgb: np.ndarray (H, W, 3) in uint8 RGB format.
            roi_top_fraction: float, fraction from top where road surface begins.

        Returns:
            dict containing surface classification, civil standards, and repair applicability.
        """
        img = np.asarray(img_rgb, dtype=np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        H, W = img.shape[:2]

        y0 = int(H * np.clip(roi_top_fraction, 0.20, 0.60))
        road_roi = img[y0:, :]
        if road_roi.size == 0:
            return dict(self.classes[0], confidence=0.50, features={}, warning="Empty ROI")

        r = road_roi[:, :, 0].astype(np.float32)
        g = road_roi[:, :, 1].astype(np.float32)
        b = road_roi[:, :, 2].astype(np.float32)

        mean_r = float(np.mean(r))
        mean_g = float(np.mean(g))
        mean_b = float(np.mean(b))
        mean_lum = float(0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b)

        rb_ratio = float(mean_r / (mean_b + 1e-4))
        rg_ratio = float(mean_r / (mean_g + 1e-4))

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        sat = np.zeros_like(cmax)
        nz = cmax > 1e-4
        sat[nz] = delta[nz] / cmax[nz]
        mean_sat = float(np.mean(sat))
        sat_p75 = float(np.percentile(sat, 75))

        if cv2 is not None:
            gray_roi = cv2.cvtColor(road_roi, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray_roi, cv2.CV_32F, ksize=3)
            texture_energy = float(lap.var())
            gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)
            high_freq_ratio = float((grad_mag > 25.0).mean())
        else:
            gray_roi = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
            texture_energy = float(np.var(gray_roi))
            high_freq_ratio = 0.05

        # Extract green foliage / rural canopy context from upper borders
        hsv_full = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) if cv2 is not None else None
        if hsv_full is not None:
            # Green foliage: Hue 25-85 in OpenCV scale (0-180), Saturation >= 25
            is_green = (hsv_full[:, :, 0] >= 25) & (hsv_full[:, :, 0] <= 85) & (hsv_full[:, :, 1] >= 25)
            upper_h = int(H * 0.45)
            foliage_left = float(is_green[:upper_h, :int(W * 0.35)].mean())
            foliage_right = float(is_green[:upper_h, int(W * 0.65):].mean())
            foliage_upper = float(is_green[:upper_h, :].mean())
            rural_foliage_evidence = max(0.0, min(1.0, (max(foliage_left, foliage_right, foliage_upper) - 0.20) / 0.35))
        else:
            rural_foliage_evidence = 0.0

        scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        # 1. Unpaved Rural Earthen / Dirt Road Evidence
        # Characteristics: rural foliage borders, warm soil tone, puddle mud contrast, unsealed track
        earth_tone_evidence = max(0.0, min(1.0, (rb_ratio - 1.01) / 0.15))
        mud_saturation_evidence = max(0.0, min(1.0, (mean_sat - 0.08) / 0.15))
        scores[2] = (
            0.45 * rural_foliage_evidence
            + 0.30 * earth_tone_evidence
            + 0.25 * mud_saturation_evidence
        )

        # 2. Rigid Cement Concrete Pavement Evidence
        # Characteristics: High luminance (light gray/off-white), low color saturation, neutral R ≈ G ≈ B, low foliage
        concrete_lum_evidence = max(0.0, min(1.0, (mean_lum - 145.0) / 35.0))
        neutral_tint_evidence = max(0.0, min(1.0, 1.0 - abs(rb_ratio - 1.0) / 0.08))
        low_sat_evidence = max(0.0, min(1.0, (0.10 - mean_sat) / 0.06))
        scores[1] = (0.45 * concrete_lum_evidence + 0.35 * neutral_tint_evidence + 0.20 * low_sat_evidence) * (1.0 - 0.7 * rural_foliage_evidence)

        # 3. Interlocking Concrete Paver Blocks Evidence
        # Characteristics: high repetitive frequency, moderate luminance, urban/industrial context (low rural foliage)
        paver_edge_evidence = max(0.0, min(1.0, (high_freq_ratio - 0.20) / 0.25))
        scores[3] = (paver_edge_evidence * 0.75) * (1.0 - 0.8 * rural_foliage_evidence)

        # 4. Flexible Bituminous Asphalt Evidence
        # Default engineered highway surface: dark to medium gray, neutral tint, moderate saturation, asphalt binder
        asphalt_lum_evidence = max(0.0, min(1.0, (140.0 - mean_lum) / 45.0))
        asphalt_neutral = max(0.0, min(1.0, 1.0 - abs(rb_ratio - 1.01) / 0.10))
        scores[0] = (0.55 * asphalt_lum_evidence + 0.45 * asphalt_neutral) * (1.0 - 0.6 * rural_foliage_evidence)

        total = sum(scores.values()) + 1e-8
        probs = {k: round(v / total, 4) for k, v in scores.items()}

        if self._trained_model is not None:
            try:
                feat_vec = np.array([[
                    mean_lum / 255.0,
                    rb_ratio,
                    rg_ratio,
                    mean_sat,
                    sat_p75,
                    min(1.0, texture_energy / 800.0),
                    high_freq_ratio,
                    rural_foliage_evidence,
                ]], dtype=np.float32)
                ml_probs = self._trained_model.predict_proba(feat_vec)[0]
                for k in range(min(4, len(ml_probs))):
                    probs[k] = round(0.55 * probs.get(k, 0.0) + 0.45 * float(ml_probs[k]), 4)
            except Exception:
                pass

        best_cls_id = int(max(probs, key=probs.get))
        best_conf = float(probs[best_cls_id])

        # Hysteresis threshold: only declare unpaved or concrete if confidence exceeds baseline floor
        if best_cls_id == 2 and (best_conf < 0.42 or rural_foliage_evidence < 0.25):
            best_cls_id = 0
            best_conf = probs[0]
        elif best_cls_id in (1, 3) and best_conf < 0.45:
            best_cls_id = 0
            best_conf = probs[0]

        meta = dict(self.classes[best_cls_id])
        meta["confidence"] = round(best_conf, 4)
        meta["trained_checkpoint_loaded"] = bool(self._trained_model is not None)
        meta["probabilities"] = {self.classes[k]["code"]: probs[k] for k in probs}
        meta["features"] = {
            "mean_luminance": round(mean_lum, 1),
            "red_to_blue_ratio": round(rb_ratio, 3),
            "mean_saturation": round(mean_sat, 3),
            "saturation_p75": round(sat_p75, 3),
            "texture_energy": round(texture_energy, 1),
            "high_freq_edge_ratio": round(high_freq_ratio, 4),
        }

        # Civil Engineering Ledger Advice
        if best_cls_id == 2:
            meta["engineering_advisory"] = (
                "IRC:SP:72 COMPLIANCE: Carriageway is an unpaved rural earthen/gravel road. "
                "MoRTH Section 500 Bituminous Concrete (BC) patching is physically inapplicable. "
                "Defects require motor-grader shaping, gravel compaction, and drainage clearing."
            )
        elif best_cls_id == 1:
            meta["engineering_advisory"] = (
                "IRC:58 COMPLIANCE: Rigid concrete carriageway identified. "
                "Joint spalling and slab cracking require rapid-hardening epoxy mortar or slab stitching per IRC:15."
            )
        elif best_cls_id == 3:
            meta["engineering_advisory"] = (
                "IRC:SP:63 COMPLIANCE: Interlocking block pavement detected. "
                "Surface distress requires block lifting, bed sand regrading, and joint sand vibro-compaction."
            )
        else:
            meta["engineering_advisory"] = (
                "IRC:82 COMPLIANCE: Flexible bituminous highway pavement verified. "
                "Defects governed by MoRTH Section 500 hot-mix asphalt patching equations."
            )

        return meta
