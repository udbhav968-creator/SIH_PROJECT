"""
Urban traffic density / Passenger Car Unit (PCU) engine.

The PCU-weighted congestion formula below is the real IRC:106-1990 method
for scoring urban road capacity - a deterministic, standards-based
computation, not something to train a network on. What used to sit in front
of it (a hand-initialized "vehicle classifier" that turned meaningless noise
features into fake car/bus/truck counts) has been removed: we don't have a
trained vehicle detector in this project, so this module now expects real
vehicle counts as input - either typed in by an operator, or produced by
whatever object detector a deployment actually has available - rather than
inventing them.
"""

IRC_STANDARDS = {
    "Car": "IRC:106-1990: Passenger Car Unit basis (1.0 PCU)",
    "City Bus": "IRC:106-1990 Sec. 4: Dedicated bus lane / BRT priority (2.0 PCU)",
    "Heavy Truck": "IRC:37-2018 / IRC:106: Axle load enforcement & route restriction (2.5 PCU)",
    "Two-Wheeler": "IRC:86-1983 / IRC:106: Segregated non-motorized/two-wheeler lane (0.5 PCU)",
}

PCU_WEIGHTS = {"Car": 1.0, "City Bus": 2.0, "Heavy Truck": 2.5, "Two-Wheeler": 0.5}


class UrbanTrafficNet:
    """Deterministic PCU / Urban Congestion Index calculator (IRC:106-1990)."""

    def calculate_congestion_index(self, vehicle_counts, road_capacity=40):
        pcu = sum(PCU_WEIGHTS.get(name, 1.0) * count for name, count in vehicle_counts.items())
        density_ratio = min(1.0, round(pcu / max(1.0, road_capacity), 3))

        if density_ratio >= 0.80:
            status, color = "SEVERELY_CONGESTED_BOTTLENECK", "#ef4444"
        elif density_ratio >= 0.50:
            status, color = "MODERATE_FLOW", "#f59e0b"
        else:
            status, color = "OPTIMAL_FREE_FLOW", "#10b981"

        return {
            "pcu_equivalent": round(pcu, 1),
            "congestion_index": density_ratio,
            "status": status,
            "indicator_color": color,
            "estimated_delay_mins": round(density_ratio * 18.5, 1),
            "irc_reference": "IRC:106-1990 Guidelines for Capacity of Urban Roads in Plain Areas",
        }

    def count_vehicle_blobs(self, frame_bgr, background_bgr, min_blob_area=350):
        """
        A genuinely simple vehicle-presence estimate via background
        subtraction + connected components (no trained classifier - it can't
        tell a car from a bus). Useful as a coarse "how much is moving in
        this frame" signal when no real object detector is deployed; do not
        treat its per-class breakdown as authoritative.
        """
        import cv2
        import numpy as np

        diff = cv2.absdiff(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY))
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_blob_area]
        return {"moving_object_count": len(blobs), "blob_areas_px": blobs, "method": "background_subtraction_connected_components"}
