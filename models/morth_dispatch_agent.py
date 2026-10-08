"""
Model M10: municipal work-order generator with a cryptographic tamper seal.

Builds a MoRTH Section 500 / IRC:SP:72 -styled digital work order (SLA tier,
asphalt mix, material tonnage and budget), then seals its canonical JSON so any
later edit to the order is detectable.

Two kinds of seal
-----------------
    SHA-256        a plain digest. Detects an edit, but anyone who edits the
                   order can also recompute the digest, so it proves nothing
                   about who issued it.
    HMAC-SHA256    a keyed digest (RFC 2104). Only someone holding the issuing
                   key can produce a valid seal, so a contractor cannot alter
                   the tonnage and re-seal. Used whenever ROAD_SHIELD_SEAL_KEY
                   is set (the SIH submission claimed HMAC; this is it).

The algorithm is written into the sealed fields, so it cannot be switched after
issue without breaking the seal. A verifier that holds a key refuses plain
SHA-256 seals: otherwise anyone could strip an HMAC order down to an unkeyed
one and re-seal it.
"""
import hashlib
import hmac
import json
import os
import time
import uuid

SEAL_FIELD = "sha256_cryptographic_seal"
EPHEMERAL = ("model", "latency_ms", "seal_verification_status")


def _seal_key(key=None):
    key = key if key is not None else os.environ.get("ROAD_SHIELD_SEAL_KEY", "")
    return key.encode("utf-8") if isinstance(key, str) else (key or b"")


def _canonical(order):
    return json.dumps(order, sort_keys=True,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode("utf-8")


def compute_seal(order, key=None):
    """(algorithm, hex digest) for an order dict that does not contain the seal."""
    k = _seal_key(key)
    algo = order.get("seal_algorithm", "SHA-256")
    data = _canonical(order)
    if algo == "HMAC-SHA256":
        if not k:
            raise KeyError("an HMAC-SHA256 seal needs ROAD_SHIELD_SEAL_KEY")
        return algo, hmac.new(k, data, hashlib.sha256).hexdigest()
    return "SHA-256", hashlib.sha256(data).hexdigest()


def check_seal(work_order, key=None):
    """Status string: SEAL_VERIFIED_AUTHENTIC, CORRUPTED_OR_TAMPERED, KEY_REQUIRED_TO_VERIFY,
    UNKEYED_SEAL_REJECTED or MALFORMED_WORK_ORDER."""
    if not isinstance(work_order, dict):
        return "MALFORMED_WORK_ORDER"
    order = {k: v for k, v in work_order.items() if k not in EPHEMERAL}
    given = order.pop(SEAL_FIELD, None)
    if not given or not isinstance(given, str):
        return "MALFORMED_WORK_ORDER"
    has_key = bool(_seal_key(key))
    algo = order.get("seal_algorithm", "SHA-256")
    if algo == "HMAC-SHA256" and not has_key:
        return "KEY_REQUIRED_TO_VERIFY"
    if algo != "HMAC-SHA256" and has_key:
        return "UNKEYED_SEAL_REJECTED"
    try:
        _, expected = compute_seal(order, key)
    except KeyError:
        return "KEY_REQUIRED_TO_VERIFY"
    return "SEAL_VERIFIED_AUTHENTIC" if hmac.compare_digest(expected, given) else "CORRUPTED_OR_TAMPERED"

class MoRTHDispatchAgent:
    def __init__(self, authority="National Highways Authority of India (NHAI) / MoRTH", seal_key=None):
        self.authority = authority
        self.seal_key = seal_key

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
            "timestamp_created": timestamp_utc,
            "seal_algorithm": "HMAC-SHA256" if _seal_key(self.seal_key) else "SHA-256",
        }
        _, seal = compute_seal(canonical_data, self.seal_key)
        canonical_data[SEAL_FIELD] = seal
        return canonical_data

    def check_work_order_seal(self, work_order):
        return check_seal(work_order, self.seal_key)

    def verify_work_order_seal(self, work_order):
        """True only when the seal checks out under this agent's key policy."""
        return check_seal(work_order, self.seal_key) == "SEAL_VERIFIED_AUTHENTIC"
