"""
Model M2: Inverse Perspective Mapping (IPM) & MoRTH Section 500 Material Engine
Transforms 2D camera image coordinates into metric ground space (m^2)
and calculates required bituminous asphalt mass (Tonnes) and procurement budget (INR).
"""
import numpy as np

class IPMHomographyEngine:
    def __init__(self, camera_height_m=1.45, pitch_deg=18.4, fx=1120.0, fy=1120.0, cx=320.0, cy=240.0):
        self.h = float(camera_height_m)
        self.pitch_rad = float(np.radians(pitch_deg))
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        
        # MoRTH Material Specifications
        self.MATERIAL_PROPERTIES = {
            "DBM_SECTION_500": {
                "name": "Dense Bituminous Macadam (MoRTH Sec 500)",
                "density_t_per_m3": 2.40,
                "cost_per_tonne_inr": 7500.0,
                "description": "Base/binder course for heavy commercial highway loading"
            },
            "BC_SECTION_508": {
                "name": "Bituminous Concrete (MoRTH Sec 508)",
                "density_t_per_m3": 2.35,
                "cost_per_tonne_inr": 8200.0,
                "description": "High-grade surface wearing course with dense aggregate grading"
            },
            "IRC_SP_79_COLD_EMULSION": {
                "name": "Cold Mix Asphalt Emulsion (IRC:SP:79)",
                "density_t_per_m3": 2.20,
                "cost_per_tonne_inr": 6800.0,
                "description": "Emergency monsoon all-weather pothole patching"
            }
        }

    def pixel_to_ground(self, u, v):
        """
        Transforms pixel coordinate (u, v) into ground coordinate (X, Y) in meters.
        Y is longitudinal distance forward; X is lateral distance.
        """
        alpha = np.arctan((v - self.cy) / self.fy)
        total_angle = self.pitch_rad + alpha
        total_angle = np.clip(total_angle, 0.05, np.pi/2 - 0.05)
        
        y_ground = self.h / np.tan(total_angle)
        x_ground = y_ground * (u - self.cx) / (self.fx * np.cos(self.pitch_rad))
        return float(x_ground), float(y_ground)

    def calculate_surface_area_sqm(self, u_min, v_min, width_px, height_px):
        """
        Computes metric ground plane area (m^2) for a given bounding box.
        """
        # 4 corners of bounding box in pixel space
        u_max = u_min + width_px
        v_max = v_min + height_px
        
        x1, y1 = self.pixel_to_ground(u_min, v_min)
        x2, y2 = self.pixel_to_ground(u_max, v_min)
        x3, y3 = self.pixel_to_ground(u_max, v_max)
        x4, y4 = self.pixel_to_ground(u_min, v_max)
        
        # Shoelace polygon area formula
        xs = [x1, x2, x3, x4]
        ys = [y1, y2, y3, y4]
        area = 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
        return float(np.clip(area, 0.05, 25.0))

    def compute_asphalt_procurement(self, area_sqm, depth_cm=6.5, mix_type="DBM_SECTION_500", compaction_margin=1.15):
        """
        Computes required asphalt mass according to MoRTH Section 500 standards.
        M = Area (m^2) * Depth (m) * Density (T/m^3) * Compaction Factor
        """
        if mix_type not in self.MATERIAL_PROPERTIES:
            mix_type = "DBM_SECTION_500"
            
        props = self.MATERIAL_PROPERTIES[mix_type]
        depth_m = depth_cm / 100.0
        volume_m3 = area_sqm * depth_m
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
            "estimated_cost_inr": round(cost_inr, 2)
        }

    def estimate_repair_materials(self, surface_area_sqm, depth_cm=6.5, mix_rate_per_tonne_inr=7500.0):
        """Convenience alias for procurement estimation."""
        res = self.compute_asphalt_procurement(surface_area_sqm, depth_cm)
        res["total_mix_mass_tonnes"] = res["required_mass_tonnes"]
        res["total_cost_inr"] = res["estimated_cost_inr"]
        return res
