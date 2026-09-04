"""
Model M10: MoRTH Autonomous Tender & Cryptographic Dispatch Agent
Generates MoRTH Section 500 / IRC:SP:72 certified Work Orders
sealed with immutable SHA-256 digests to prevent contractor collusion.
"""
import hashlib
import json
import time

class MoRTHDispatchAgent:
    def __init__(self, authority="National Highways Authority of India (NHAI) / MoRTH"):
        self.authority = authority

    def generate_work_order(self, corridor_id, latitude, longitude, distress_class, area_sqm, depth_cm=6.5, pci_score=42):
        """
        Generates certified digital work order with cryptographic digest.
        """
        # Determine SLA based on severity & PCI
        if distress_class in ["D40 Pothole", "D40"] or pci_score < 40:
            sla_hours = 24
            priority = "CRITICAL_TIER_1"
            mix = "DBM_SECTION_500"
        elif distress_class in ["D20 Alligator", "D20"] or pci_score < 60:
            sla_hours = 48
            priority = "HIGH_TIER_2"
            mix = "BC_SECTION_508"
        else:
            sla_hours = 72
            priority = "MEDIUM_TIER_3"
            mix = "IRC_SP_79_COLD_EMULSION"
            
        timestamp_utc = int(time.time())
        order_uuid = f"MORTH-WO-{timestamp_utc % 1000000:06d}-{corridor_id[:4]}"
        
        # Calculate materials
        density = 2.40 if mix == "DBM_SECTION_500" else (2.35 if mix == "BC_SECTION_508" else 2.20)
        volume_m3 = area_sqm * (depth_cm / 100.0)
        tonnage = volume_m3 * density * 1.15
        rate = 7500.0 if mix == "DBM_SECTION_500" else (8200.0 if mix == "BC_SECTION_508" else 6800.0)
        allocated_budget_inr = tonnage * rate
        
        # Canonical dictionary for cryptographic hashing
        canonical_data = {
            "work_order_id": order_uuid,
            "issuing_authority": self.authority,
            "corridor": corridor_id,
            "coordinates": {"lat": round(latitude, 6), "lon": round(longitude, 6)},
            "distress_type": distress_class,
            "pavement_pci": pci_score,
            "surface_area_sqm": round(area_sqm, 2),
            "depth_cm": round(depth_cm, 1),
            "asphalt_mix": mix,
            "required_mass_tonnes": round(tonnage, 3),
            "allocated_budget_inr": round(allocated_budget_inr, 2),
            "priority": priority,
            "sla_resolution_hours": sla_hours,
            "timestamp_created": timestamp_utc
        }
        
        # Compute SHA-256 cryptographic seal
        serialized = json.dumps(canonical_data, sort_keys=True)
        sha256_seal = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        canonical_data["sha256_cryptographic_seal"] = sha256_seal
        
        return canonical_data

    @staticmethod
    def verify_work_order_seal(work_order):
        """
        Validates whether a work-order JSON has been tampered with.
        """
        order_copy = dict(work_order)
        original_seal = order_copy.pop("sha256_cryptographic_seal", None)
        if not original_seal:
            return False
        # Remove any ephemeral server response wrappers
        order_copy.pop("model", None)
        order_copy.pop("latency_ms", None)
        serialized = json.dumps(order_copy, sort_keys=True)
        computed_seal = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return computed_seal == original_seal
