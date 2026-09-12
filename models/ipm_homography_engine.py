"""
Model M2: Inverse Perspective Mapping (IPM) & MoRTH Section 500 material
volumetrics.

Maps a 2D bounding box in a dashcam frame to a real ground-plane area (m^2)
using a pinhole-camera + flat-road-plane assumption, then converts that area
into an asphalt tonnage and repair cost using MoRTH Section 500 material
specs. This is standard photogrammetry, not a learned model - the only
"calibration" involved is knowing (or estimating) the camera's mount height
and pitch angle.
"""

import numpy as np


class IPMHomographyEngine:
    MATERIAL_PROPERTIES = {
        "DBM_SECTION_500": {
            "name": "Dense Bituminous Macadam (MoRTH Sec 500)",
            "density_t_per_m3": 2.40,
            "cost_per_tonne_inr": 7500.0,
            "description": "Base/binder course for heavy commercial highway loading",
        },
        "BC_SECTION_508": {
            "name": "Bituminous Concrete (MoRTH Sec 508)",
            "density_t_per_m3": 2.35,
            "cost_per_tonne_inr": 8200.0,
            "description": "High-grade surface wearing course with dense aggregate grading",
        },
        "IRC_SP_79_COLD_EMULSION": {
            "name": "Cold Mix Asphalt Emulsion (IRC:SP:79)",
            "density_t_per_m3": 2.20,
            "cost_per_tonne_inr": 6800.0,
            "description": "Emergency monsoon all-weather pothole patching",
        },
    }

    def __init__(self, camera_height_m=1.45, pitch_deg=18.4, fx=1120.0, fy=1120.0, cx=320.0, cy=240.0):
        self.h = float(camera_height_m)
        self.pitch_rad = float(np.radians(pitch_deg))
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)

    @classmethod
    def from_calibration(cls, profile):
        """Build from a camera_calibration profile dict."""
        return cls(camera_height_m=profile["camera_height_m"], pitch_deg=profile["pitch_deg"],
                   fx=profile["fx"], fy=profile["fy"], cx=profile["cx"], cy=profile["cy"])

    def pixel_to_ground(self, u, v):
        """Pixel (u, v) -> ground-plane (x, y) meters. y = forward distance, x = lateral offset."""
        alpha = np.arctan((v - self.cy) / self.fy)
        total_angle = np.clip(self.pitch_rad + alpha, 0.05, np.pi / 2 - 0.05)
        y_ground = self.h / np.tan(total_angle)
        x_ground = y_ground * (u - self.cx) / (self.fx * np.cos(self.pitch_rad))
        return float(x_ground), float(y_ground)

    def calculate_surface_area_sqm(self, u_min, v_min, width_px, height_px):
        """Shoelace-formula ground area of a pixel bounding box's 4 corners."""
        u_max, v_max = u_min + width_px, v_min + height_px
        corners = [
            self.pixel_to_ground(u_min, v_min),
            self.pixel_to_ground(u_max, v_min),
            self.pixel_to_ground(u_max, v_max),
            self.pixel_to_ground(u_min, v_max),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
        return float(np.clip(area, 0.05, 25.0))

    # ------------------------------------------------------------------
    def row_pixel_area_m2(self, image_height, image_width, row_stride=1):
        """
        Ground area of ONE pixel, per image row.

        Perspective means a pixel near the bottom of the frame covers a few
        square centimetres of road and a pixel near the horizon covers square
        metres. Any area computed by counting pixels has to weight them by row,
        or it is meaningless. Returned as an array indexed by row so a mask can
        be turned into an area with a single dot product.
        """
        rows = np.arange(0, image_height, row_stride, dtype=np.float64)
        u0, u1 = self.cx - 0.5, self.cx + 0.5
        out = np.zeros(rows.shape[0], dtype=np.float64)
        for i, v in enumerate(rows):
            (x0, y0) = self.pixel_to_ground(u0, v - 0.5)
            (x1, y1) = self.pixel_to_ground(u1, v - 0.5)
            (_x2, y2) = self.pixel_to_ground(u0, v + 0.5)
            width_m = abs(x1 - x0)
            depth_m = abs(y2 - y0)
            out[i] = width_m * depth_m
        return out

    def mask_area_m2(self, mask, max_area_m2=25.0):
        """
        Ground area of a boolean pixel mask - a measurement, not a box estimate.

        Each pixel contributes the ground area of its own footprint, which
        depends on its row. This is what replaces
        `calculate_surface_area_sqm`'s four-corner rectangle: a defect is
        rarely rectangular, and a padded box around a diagonal crack can
        overstate its area several times over.

        Returns (area_m2, diagnostics). The clip is reported rather than
        applied silently, because a clipped area means the geometry has gone
        out of its valid range and the number should not be trusted.
        """
        m = np.asarray(mask, dtype=bool)
        if m.ndim != 2:
            raise ValueError("mask must be 2-D")
        h, w = m.shape
        per_row = self.row_pixel_area_m2(h, w)
        counts = m.sum(axis=1).astype(np.float64)
        raw = float(np.dot(counts, per_row))
        clipped = float(np.clip(raw, 0.0, max_area_m2))
        return clipped, {
            "raw_area_m2": round(raw, 4),
            "clipped": bool(raw > max_area_m2),
            "mask_pixels": int(m.sum()),
            "image_shape": [int(h), int(w)],
            "method": "per-pixel ground footprint summed over the segmentation mask",
        }

    def compute_asphalt_procurement(self, area_sqm, depth_cm=6.5, mix_type="DBM_SECTION_500", compaction_margin=1.15):
        """M = Area (m^2) * Depth (m) * Density (T/m^3) * compaction margin."""
        props = self.MATERIAL_PROPERTIES.get(mix_type, self.MATERIAL_PROPERTIES["DBM_SECTION_500"])
        volume_m3 = area_sqm * (depth_cm / 100.0)
        mass_tonnes = volume_m3 * props["density_t_per_m3"] * compaction_margin
        cost_inr = mass_tonnes * props["cost_per_tonne_inr"]
        return {
            "surface_area_sqm": round(area_sqm, 3),
            "depth_cm": round(depth_cm, 1),
            "volume_m3": round(volume_m3, 4),
            "bituminous_mix": props["name"],
            "density_t_per_m3": props["density_t_per_m3"],
            "compaction_factor": compaction_margin,
            "required_mass_tonnes": round(mass_tonnes, 4),
            "estimated_cost_inr": round(cost_inr, 2),
        }

    def estimate_repair_materials(self, surface_area_sqm, depth_cm=6.5, mix_rate_per_tonne_inr=7500.0):
        res = self.compute_asphalt_procurement(surface_area_sqm, depth_cm)
        res["total_mix_mass_tonnes"] = res["required_mass_tonnes"]
        res["total_cost_inr"] = res["required_mass_tonnes"] * mix_rate_per_tonne_inr
        return res
