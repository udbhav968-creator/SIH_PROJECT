"""
Model M10: municipal work-order generator with a cryptographic tamper seal.

Builds a MoRTH Section 500 / IRC:SP:72 -styled digital work order (SLA tier,
asphalt mix, material tonnage and budget), then SHA-256-hashes its canonical
JSON so any later edit to the order is detectable. This part was already
real - genuine hashing, genuine material-cost arithmetic - only the
docstring's "certified" language and the old 10-class distress labels
needed cleaning up.
"""
import hashlib
import json
import time
import uuid

class MoRTHDispatchAgent:
    def __init__(self, authority="National Highways Authority of India (NHAI) / MoRTH"):
        self.authority = authority

    def generate_work_order(self, corridor_id, latitude, longitude, distress_class, area_sqm, depth_cm, pci_score):
        """Generates a work order dict and seals it with a SHA-256 digest.

        Every measured field must be supplied by the caller - there are no
        defaults for area, depth or PCI, because a sealed order built from a
        default is a fabricated document. GPS may be None (a photograph with no
        location); the order is then issued but held, with
        dispatch_status = "HELD_NO_GPS", instead of being placed at a made-up
        coordinate.
        """
        from models.ipm_homography_engine import IPMHomographyEngine

        if area_sqm is None or depth_cm is None or pci_score is None or not distress_class:
            raise ValueError("distress_class, area_sqm, depth_cm and pci_score are all required")

        # Cast inputs to native Python types
        area_sqm = float(area_sqm)
        depth_cm = float(depth_cm)
        pci_score = int(round(float(pci_score)))
        has_gps = latitude is not None and longitude is not None
        if has_gps:
            latitude, longitude = float(latitude), float(longitude)
            if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
                raise ValueError("latitude/longitude out of range")
        if area_sqm < 0 or depth_cm < 0 or not (0 <= pci_score <= 100):
            raise ValueError("area and depth must be >= 0 and PCI must be 0-100")

        # Determine SLA based on severity & PCI
        if distress_class in ("Pothole Cavity", "D40 Pothole", "D40") or pci_score < 40:
            sla_hours = 24
            priority = "CRITICAL_TIER_1"
            mix = "DBM_SECTION_500"
        elif distress_class in ("Crack (Longitudinal / Transverse / Alligator)", "D20 Alligator", "D20") or pci_score < 60:
            sla_hours = 48
            priority = "HIGH_TIER_2"
            mix = "BC_SECTION_508"
        else:
            sla_hours = 72
            priority = "MEDIUM_TIER_3"
            mix = "IRC_SP_79_COLD_EMULSION"

        timestamp_utc = int(time.time())
        # The time prefix keeps IDs human-sortable; the random suffix makes them
        # unique. Time alone collided for every order issued on one corridor in
        # the same second - routine when several buses report one pothole.
        order_uuid = (f"MORTH-WO-{timestamp_utc % 1000000:06d}-{str(corridor_id)[:4]}"
                      f"-{uuid.uuid4().hex[:8].upper()}")

        # Materials come from the one table the inference pipeline also uses,
        # so a work order can never price a repair differently from the audit
        # that produced it (they used to carry separate copies of the numbers).
        props = IPMHomographyEngine.MATERIAL_PROPERTIES[mix]
        density = props["density_t_per_m3"]
        rate = props["cost_per_tonne_inr"]
        volume_m3 = float(area_sqm * (depth_cm / 100.0))
        tonnage = float(volume_m3 * density * IPMHomographyEngine.COMPACTION_FACTOR)
        allocated_budget_inr = float(tonnage * rate)

        # Canonical dictionary for cryptographic hashing
        canonical_data = {
            "work_order_id": str(order_uuid),
            "issuing_authority": str(self.authority),
            "corridor": str(corridor_id),
            "coordinates": {"lat": round(latitude, 6), "lon": round(longitude, 6)} if has_gps else None,
            "dispatch_status": "READY_FOR_DISPATCH" if has_gps else "HELD_NO_GPS",
            "distress_type": str(distress_class),
            "pavement_pci": pci_score,
            "surface_area_sqm": round(area_sqm, 2),
            "depth_cm": round(depth_cm, 1),
            "asphalt_mix": mix,
            "required_mass_tonnes": round(tonnage, 3),
            "allocated_budget_inr": round(allocated_budget_inr, 2),
            "rate_basis": IPMHomographyEngine.RATE_BASIS,
            "priority": priority,
            "sla_resolution_hours": int(sla_hours),
            "timestamp_created": timestamp_utc
        }

        # Compute SHA-256 cryptographic seal
        serialized = json.dumps(canonical_data, sort_keys=True, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        sha256_seal = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        canonical_data["sha256_cryptographic_seal"] = sha256_seal

        return canonical_data

    @staticmethod
    def verify_work_order_seal(work_order):
        """Recomputes the SHA-256 digest and checks it against the stored seal."""
        if not isinstance(work_order, dict):
            return False
        order_copy = dict(work_order)
        original_seal = order_copy.pop("sha256_cryptographic_seal", None)
        if not original_seal:
            return False
        # Remove any ephemeral server response wrappers
        order_copy.pop("model", None)
        order_copy.pop("latency_ms", None)
        serialized = json.dumps(order_copy, sort_keys=True, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        computed_seal = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return computed_seal == original_seal
