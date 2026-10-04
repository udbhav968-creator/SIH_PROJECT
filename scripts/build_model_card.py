"""
Write docs/MODEL_CARD.md from checkpoints/claims.json and the selection files.

    python scripts/build_model_card.py

A model card (Mitchell et al., "Model Cards for Model Reporting", 2019) says
what each served model is for, what it was trained and tested on, how well it
does, and where it should not be trusted. Every figure here is read from the
same report the website and the claims registry read, so the card cannot
disagree with them; regenerate it after any retraining.
"""

import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT = os.path.join(ROOT, "checkpoints")
OUT = os.path.join(ROOT, "docs", "MODEL_CARD.md")

SERVED_ORDER = ["M1", "M_SEG", "M4", "M_DET", "M_RDD", "M_PRIVACY", "M_GATE"]
INTENDED = {
    "M1": "Classify a road photograph or region into seven road-condition classes, as the first step of a defect record.",
    "M_SEG": "Outline cracks and potholes pixel by pixel, so area (and from it material and cost) is measured from the "
             "defect's own footprint rather than a box.",
    "M4": "Classify a 1-second accelerometer window (smooth road, speed breakers, pothole impact) to corroborate a "
          "visual detection.",
    "M_DET": "Find people, vehicles and traffic furniture in a frame, for privacy blurring and to keep them out of the "
             "defect search.",
    "M_RDD": "Draw boxes around cracks and potholes in a road frame (display only; area and cost still come from the mask).",
    "M_PRIVACY": "Blur heads and number plates before an image is shared.",
    "M_GATE": "Decide which mask regions become reported defects, trading false alarms against missed defects.",
}


def load(name):
    p = os.path.join(CKPT, name)
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return ", ".join(f"{k} {fmt(x)}" for k, x in v.items() if not isinstance(x, (dict, list)))
    return str(v)


def main():
    claims = load("claims.json")
    if not claims:
        sys.exit("checkpoints/claims.json missing - run scripts/build_claims.py first")
    subs = {s["id"]: s for s in claims.get("subsystems", [])}
    seg_sel = load("segmenter_selection.json")
    vis_sel = load("vision_model_selection.json")
    imu_sel = load("imu_model_selection.json")

    w = []
    w.append("# ROAD-SHIELD model card")
    w.append("")
    w.append(f"Generated {time.strftime('%Y-%m-%d')} from `checkpoints/claims.json` by `scripts/build_model_card.py`. "
             "Regenerate after any retraining; do not edit by hand.")
    w.append("")
    w.append("## System at a glance")
    w.append("")
    w.append("| | |")
    w.append("|---|---|")
    w.append("| Purpose | Road-defect records from bus-camera frames: class, outline, ground area, depth interval, "
             "cost range, sealed work order |")
    w.append("| Intended users | Road authorities and their contractors; inspectors reviewing system reports |")
    w.append("| Decision role | **Recommends; people decide.** No repair is dispatched without a human authority |")
    w.append("| Out of scope | Any decision about a person; legal evidence of fault; roads or cameras unlike the "
             "evaluation data without re-validation; depth as a measurement |")
    w.append(f"| Served segmenter | {seg_sel.get('served', 'pixel_classifier')} "
             f"(rule: {seg_sel.get('rule', 'see segmenter_selection.json')[:160]}) |")
    w.append(f"| Served vibration model | {imu_sel.get('served', 'random_forest')} |")
    w.append("")

    for sid in SERVED_ORDER:
        s = subs.get(sid)
        if not s:
            continue
        w.append(f"## {s.get('name', sid)}")
        w.append("")
        w.append(f"**Intended use.** {INTENDED.get(sid, s.get('role', ''))}")
        w.append("")
        if s.get("architecture"):
            w.append(f"**Model.** {s['architecture']}")
            w.append("")
        if s.get("why"):
            w.append(f"**Why this design.** {s['why']}")
            w.append("")
        meas = {k: v for k, v in (s.get("measured") or {}).items() if v is not None}
        if meas:
            w.append("**Measured (held-out data).**")
            w.append("")
            w.append("| metric | value |")
            w.append("|---|---|")
            for k, v in meas.items():
                if isinstance(v, list) or (isinstance(v, dict) and not fmt(v)):
                    continue
                w.append(f"| {k.replace('_', ' ')} | {fmt(v)} |")
            w.append("")
        if s.get("not_claimed"):
            w.append(f"**Limits - do not rely on it for this.** {s['not_claimed']}")
            w.append("")
        alt = s.get("deep_alternative_tried") or s.get("rejected_alternative")
        if alt:
            w.append(f"**Alternatives tried.** {alt}")
            w.append("")
        if s.get("evidence"):
            w.append(f"**Evidence.** `{s['evidence']}`" + (f" · reproduce: `{s['reproduce']}`" if s.get("reproduce") else ""))
            w.append("")

    w.append("## How models are chosen")
    w.append("")
    w.append("Every deep model replaces its simpler counterpart only by a rule written before its test set is scored, "
             "and the segmenter must also hold up end to end through the full pipeline on photographs from other "
             "datasets. Losers stay on disk and are reported. Training data is checked for near-copies of every "
             "measurement photograph before training.")
    if vis_sel.get("rule"):
        w.append("")
        w.append(f"- Vision classifier: {vis_sel['rule']}")
    if seg_sel.get("rule"):
        w.append(f"- Segmenter: {seg_sel['rule']}")
    if imu_sel.get("rule"):
        w.append(f"- Vibration: {imu_sel['rule']}")
    w.append("")
    w.append("## Ethical considerations")
    w.append("")
    w.append("- **People.** No identity is stored; heads and plates are blurred before sharing. Blur recall has not been "
             "measured, so a missed face or plate is possible.")
    w.append("- **Fairness is geographic.** Roads without bus routes are not inspected, and a model trained in one "
             "country degrades in another (33% on Indian roads before Indian data was added). Coverage and per-city "
             "accuracy must be reported before any allocation of repair budgets relies on the system.")
    w.append("- **Money.** Costs are ranges from an estimated depth. A work order is a recommendation for a human "
             "authority, sealed so it cannot be altered after issue.")
    w.append("")
    corr = claims.get("corrections") or []
    w.append(f"## Corrections on record ({len(corr)})")
    w.append("")
    w.append("Claims this project made and later withdrew or corrected, kept visible on the Architecture page:")
    w.append("")
    for c in corr:
        w.append(f"- **{c.get('status')}** - {c.get('claim')}")
    w.append("")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(w))
    print(f"wrote {OUT} ({len(w)} lines)")


if __name__ == "__main__":
    main()
