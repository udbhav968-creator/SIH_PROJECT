"""
Repair Priority Index, the formula proposed in the Milestone 2 report (section 3.5):

    PI = w1 * (100 - PCI) + w2 * Vol + w3 * Traffic

The report gives the formula but no scales and no weights, so a number could not be computed from it.
Here each term is first put on a 0-100 scale, which makes the weights the share of the index each term
can contribute and keeps PI itself on 0-100.

    Severity  100 - PCI                       PCI from the ASTM D6433 engine (0 = failed, 100 = perfect)
    Vol       repair volume, area x depth     100 at VOLUME_FULL_M3 (0.25 m3, e.g. 2.5 m2 at 10 cm deep)
    Traffic   how much the road is used       measured PCU/day when the caller has a count (log scale,
                                              100 at PCU_FULL); otherwise the fleet's own exposure:
                                              distinct buses that reported the defect, 100 at BUSES_FULL

Which traffic basis was used is returned with every score, because a bus count is a proxy (it measures
how many routes pass, not how many vehicles) and should not be read as a traffic survey.

Weights are a policy decision, not something learned from data. The defaults (0.5 / 0.2 / 0.3) put the
condition of the pavement first and traffic second; a municipality can change them with
ROAD_SHIELD_PI_WEIGHTS="w1,w2,w3" or per request. They must be non-negative and sum to 1.

When a term cannot be computed (no depth, so no volume; no traffic count and no bus list) its weight is
shared out over the other terms in proportion, and the term is listed under "missing". Nothing is filled
in with a typical value, because a ranking built on invented inputs is worse than a shorter formula.

Deliberately not inputs: ward, constituency, road name or any "VIP road" flag. The point of the index in
the report is that repair order follows measured hazard and use, not influence.

rank_stability() answers the obvious objection to any weighted index ("you chose the weights to get the
order you wanted"): it re-ranks under every +/-0.1 shift of weight between two terms and reports Kendall's
tau against the default order. A tau near 1 means the order barely depends on the policy choice.
"""
import math
import os

DEFAULT_WEIGHTS = (0.5, 0.2, 0.3)
VOLUME_FULL_M3 = 0.25
PCU_FULL = 40000.0
BUSES_FULL = 20
BANDS = ((70.0, "P1", "repair within 24 h"), (50.0, "P2", "repair within 7 days"),
         (30.0, "P3", "schedule in the next maintenance cycle"), (0.0, "P4", "monitor"))
TERMS = ("severity", "volume", "traffic")


def parse_weights(value):
    """'0.5,0.2,0.3' or a 3-sequence -> validated tuple of floats."""
    if isinstance(value, str):
        parts = [p for p in value.replace(";", ",").split(",") if p.strip()]
        value = [float(p) for p in parts]
    w = tuple(float(x) for x in value)
    if len(w) != 3:
        raise ValueError("weights need exactly three numbers: w1 (severity), w2 (volume), w3 (traffic)")
    if any(x < 0 or math.isnan(x) for x in w):
        raise ValueError("weights must be non-negative")
    if abs(sum(w) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1 (got {sum(w):.3f})")
    return w


def default_weights():
    env = os.environ.get("ROAD_SHIELD_PI_WEIGHTS")
    if env:
        try:
            return parse_weights(env)
        except ValueError:
            pass
    return DEFAULT_WEIGHTS


def severity_score(pci):
    pci = float(pci)
    if not (math.isfinite(pci) and 0.0 <= pci <= 100.0):
        raise ValueError("PCI must be between 0 and 100")
    return 100.0 - pci


def volume_score(volume_m3=None, area_m2=None, depth_cm=None):
    """(score, volume_m3) or (None, None) when the volume cannot be computed."""
    if volume_m3 is None and area_m2 is not None and depth_cm is not None:
        volume_m3 = float(area_m2) * float(depth_cm) / 100.0
    if volume_m3 is None:
        return None, None
    volume_m3 = float(volume_m3)
    if not math.isfinite(volume_m3) or volume_m3 < 0:
        raise ValueError("volume must be a non-negative number")
    return min(100.0, 100.0 * volume_m3 / VOLUME_FULL_M3), volume_m3


def traffic_score(traffic_pcu_per_day=None, reporting_buses=None):
    """(score, basis) - basis says whether a measured count or the fleet proxy was used."""
    if traffic_pcu_per_day is not None:
        pcu = float(traffic_pcu_per_day)
        if not math.isfinite(pcu) or pcu < 0:
            raise ValueError("traffic_pcu_per_day must be a non-negative number")
        return min(100.0, 100.0 * math.log1p(pcu) / math.log1p(PCU_FULL)), "measured_pcu_per_day"
    if reporting_buses is not None:
        if isinstance(reporting_buses, (list, tuple, set)):
            n = len(set(map(str, reporting_buses)))
        elif isinstance(reporting_buses, (int, float)) and not isinstance(reporting_buses, bool) \
                and math.isfinite(reporting_buses) and reporting_buses >= 0:
            n = reporting_buses
        else:
            raise ValueError("reporting_buses must be a list of bus ids or a non-negative count")
        return min(100.0, 100.0 * float(n) / BUSES_FULL), "fleet_proxy_distinct_buses"
    return None, None


def band_for(pi):
    for floor, name, action in BANDS:
        if pi >= floor:
            return name, action
    return BANDS[-1][1], BANDS[-1][2]


def score(pci, volume_m3=None, area_m2=None, depth_cm=None, traffic_pcu_per_day=None,
          reporting_buses=None, weights=None):
    w = parse_weights(weights) if weights is not None else default_weights()
    vol, vol_m3 = volume_score(volume_m3, area_m2, depth_cm)
    trf, basis = traffic_score(traffic_pcu_per_day, reporting_buses)
    comps = {"severity": severity_score(pci), "volume": vol, "traffic": trf}
    present = {t: wt for t, wt in zip(TERMS, w) if comps[t] is not None}
    total_w = sum(present.values())
    if total_w <= 0:
        raise ValueError("every term with a non-zero weight is missing; supply depth or traffic, or change weights")
    used = {t: present[t] / total_w for t in present}
    pi = sum(used[t] * comps[t] for t in used)
    band, action = band_for(pi)
    return {
        "priority_index": round(pi, 2),
        "band": band,
        "action": action,
        "components": {t: (round(v, 2) if v is not None else None) for t, v in comps.items()},
        "weights_requested": dict(zip(TERMS, w)),
        "weights_applied": {t: round(v, 4) for t, v in used.items()},
        "missing": [t for t in TERMS if comps[t] is None],
        "volume_m3": round(vol_m3, 4) if vol_m3 is not None else None,
        "traffic_basis": basis,
        "formula": "PI = w1*(100-PCI) + w2*Vol + w3*Traffic, each term on 0-100",
    }


def score_defect(defect, weights=None):
    """Score a ledger record (fleet_deduplication_engine shape)."""
    return score(
        pci=defect["severity_pci"],
        area_m2=defect.get("area_m2"),
        depth_cm=defect.get("depth_cm"),
        traffic_pcu_per_day=defect.get("traffic_pcu_per_day"),
        reporting_buses=defect.get("reporting_buses"),
        weights=weights,
    )


def rank(defects, weights=None):
    """Ledger records with their score, highest priority first; ties broken by older first sighting.
    A record that cannot be scored (no PCI, or a value out of range) is left out, never allowed to break
    the ranking of the others."""
    out = []
    w = parse_weights(weights) if weights is not None else default_weights()
    for d in defects:
        if d.get("severity_pci") is None:
            continue
        try:
            s = score_defect(d, w)
        except (ValueError, TypeError):
            continue
        out.append({**d, "priority": s})
    out.sort(key=lambda r: (-r["priority"]["priority_index"], r.get("first_seen_timestamp") or 0,
                            str(r.get("defect_id"))))
    for i, r in enumerate(out, 1):
        r["priority"]["rank"] = i
    return out


def kendall_tau(order_a, order_b):
    """Kendall's tau-a between two orderings of the same ids."""
    pos_b = {k: i for i, k in enumerate(order_b)}
    ids = [k for k in order_a if k in pos_b]
    n = len(ids)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos_b[ids[i]] < pos_b[ids[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


def rank_stability(defects, weights=None, step=0.1):
    """How much the repair order depends on the weights: tau for every +/-step shift between two terms."""
    base_w = parse_weights(weights) if weights is not None else default_weights()
    base = [r["defect_id"] for r in rank(defects, base_w)]
    trials = []
    for i in range(3):
        for j in range(3):
            if i == j or base_w[j] < step - 1e-9:
                continue
            w = list(base_w)
            w[i] += step
            w[j] -= step
            w = tuple(round(x, 6) for x in w)
            order = [r["defect_id"] for r in rank(defects, w)]
            trials.append({"weights": dict(zip(TERMS, w)), "kendall_tau": round(kendall_tau(base, order), 4),
                           "top_changed": bool(base and order and base[0] != order[0])})
    taus = [t["kendall_tau"] for t in trials]
    return {
        "defects": len(base),
        "step": step,
        "min_kendall_tau": min(taus) if taus else 1.0,
        "mean_kendall_tau": round(sum(taus) / len(taus), 4) if taus else 1.0,
        "top_priority_changes_in": sum(t["top_changed"] for t in trials),
        "trials": trials,
    }
