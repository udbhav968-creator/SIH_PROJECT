"""
Repair Priority Index: which defect gets fixed first, decided by measurements.

    PI = 100 * (w_c * condition + w_v * volume + w_t * traffic)

  condition = (100 - PCI) / 100                        how broken the pavement is
  volume    = V / (V + V_ref)                          how much material is missing
  traffic   = log(1 + AADT) / log(1 + 10 * AADT_ref)   how many people it endangers

Every term is scaled to [0, 1] before weighting, so the weights mean what
they say: with the defaults, pavement condition carries half the decision.
Volume saturates (a 2 m^3 crater is not forty times more urgent than a
0.05 m^3 pothole) and traffic is logarithmic (the ten-thousandth vehicle adds
less risk than the tenth).

Why this exists (CSET485 / AI and Society)
------------------------------------------
Repair queues are political: arterial roads near influential addresses get
resurfaced while outer colonies wait. The index accepts ONLY measured road
quantities. It has no parameter for ward, constituency, locality name or who
complained, so it cannot be steered by them, and a test pins that.

When traffic has not been measured for a location, the traffic term is left
out and the remaining weights are renormalised. The result says so. Filling
the gap with a guessed traffic count would make the ranking look more
objective than it is.
"""

import math
from dataclasses import asdict, dataclass

REFERENCE_VOLUME_M3 = 0.06  # 1 m^2 at 6 cm: a typical urban pothole
REFERENCE_AADT = 10_000  # vehicles/day on a busy urban arterial
BANDS = ((75.0, "CRITICAL"), (55.0, "HIGH"), (35.0, "MEDIUM"), (0.0, "LOW"))


@dataclass(frozen=True)
class PriorityWeights:
    condition: float = 0.5
    volume: float = 0.2
    traffic: float = 0.3

    def __post_init__(self):
        values = (self.condition, self.volume, self.traffic)
        if any(v < 0 for v in values):
            raise ValueError("priority weights must be non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError(f"priority weights must sum to 1, got {sum(values):.3f}")


DEFAULT_WEIGHTS = PriorityWeights()


def _band(score):
    return next(label for floor, label in BANDS if score >= floor)


def priority_index(pci, volume_m3, daily_traffic=None, weights=DEFAULT_WEIGHTS):
    """
    Priority for one defect.

    pci           ASTM D6433 condition, 0 (failed) to 100 (perfect)
    volume_m3     missing material, area x depth
    daily_traffic annual average daily traffic (vehicles/day), or None if unmeasured
    """
    if not 0.0 <= pci <= 100.0:
        raise ValueError(f"PCI must be in [0, 100], got {pci}")
    if volume_m3 < 0:
        raise ValueError("volume must be non-negative")
    if daily_traffic is not None and daily_traffic < 0:
        raise ValueError("daily traffic must be non-negative")

    terms = {
        "condition": (100.0 - pci) / 100.0,
        "volume": volume_m3 / (volume_m3 + REFERENCE_VOLUME_M3),
    }
    used = {"condition": weights.condition, "volume": weights.volume}
    if daily_traffic is not None:
        terms["traffic"] = min(1.0, math.log1p(daily_traffic) / math.log1p(10 * REFERENCE_AADT))
        used["traffic"] = weights.traffic

    total = sum(used.values())
    if total <= 0:
        raise ValueError("the weights of the available terms sum to zero")
    effective = {k: w / total for k, w in used.items()}
    score = 100.0 * sum(effective[k] * terms[k] for k in terms)

    return {
        "priority_index": round(score, 2),
        "band": _band(score),
        "components": {k: round(v, 4) for k, v in terms.items()},
        "effective_weights": {k: round(v, 4) for k, v in effective.items()},
        "traffic_measured": daily_traffic is not None,
        "formula": "PI = 100 * (w_c*(100-PCI)/100 + w_v*V/(V+0.06) + w_t*log(1+AADT)/log(1+100000))",
    }


def rank_defects(defects, weights=DEFAULT_WEIGHTS):
    """
    Rank defect records, most urgent first.

    Each record needs `pci` and `volume_m3`; `daily_traffic` is optional. Any
    other keys (ids, addresses, ward names) are carried through untouched and
    never read. Ties keep their input order, so the ranking is deterministic.
    """
    scored = []
    for record in defects:
        result = priority_index(record["pci"], record["volume_m3"], record.get("daily_traffic"), weights)
        scored.append({**record, **result})
    scored.sort(key=lambda r: -r["priority_index"])  # stable: equal scores keep input order
    for position, record in enumerate(scored, start=1):
        record["priority_rank"] = position
    return scored


def describe(weights=DEFAULT_WEIGHTS):
    return {"weights": asdict(weights), "reference_volume_m3": REFERENCE_VOLUME_M3,
            "reference_aadt": REFERENCE_AADT, "bands": dict((label, floor) for floor, label in BANDS)}
