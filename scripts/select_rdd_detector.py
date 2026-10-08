"""
Decide which road-damage detector the server runs: the India-only one or the multi-country candidate.

    python -m scripts.select_rdd_detector            # after training/train_rdd_detector.py --tag world

Rule (written before the candidate's test numbers are read)
-----------------------------------------------------------
Serve the multi-country detector only if its mAP@0.5 on RDD2022 India's VALIDATION photographs is higher
than the served detector's on the same photographs. Both models were trained with India's identical split
(scripts/prepare_rdd2022_voc.py, seed 42), so the comparison is like for like. The India TEST numbers of
both are copied into the record for reporting; they do not enter the decision.

The winner still has to pass scripts/verify_rdd_detector.py (--artefact and --clean-roads) before the
server shows its boxes, exactly like the India-only model.

If it wins, its files replace the served names (damage_rdd2022_india.onnx/.json,
road_damage_detector_report.json) and the old report is kept inside the new one as "previous_model".
"""
import json
import os
import shutil
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT = os.path.join(ENGINE_ROOT, "checkpoints")
SERVED = ("damage_rdd2022_india.onnx", "damage_rdd2022_india.json", "road_damage_detector_report.json")
SELECTION = "rdd_detector_selection.json"
RULE = ("serve the multi-country detector only if its mAP@0.5 on RDD2022 India's validation photographs is "
        "higher than the served detector's on the same photographs; test numbers reported, never used to choose")


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _map50(rep, key):
    return float(((rep or {}).get(key) or {}).get("map50") or 0.0)


def india_valid_photos(rep):
    """How many India validation photographs a report's 'validation' was measured on."""
    n = ((rep or {}).get("data") or {}).get("photographs") or {}
    return n.get("valid_india") or n.get("valid")


def decide(candidate, incumbent):
    """('candidate' | 'incumbent', why). Pure function of the two reports - unit-tested."""
    if not candidate:
        return "incumbent", "no multi-country candidate report"
    if not incumbent:
        return "candidate", "no India-only detector has been trained; the candidate is the only model"
    a, b = india_valid_photos(candidate), india_valid_photos(incumbent)
    if a and b and a != b:
        return "incumbent", (f"not comparable: the candidate was validated on {a} India photographs, the served "
                             f"detector on {b}")
    c, i = _map50(candidate, "validation"), _map50(incumbent, "validation")
    if c > i:
        return "candidate", f"India validation mAP@0.5 {c:.4f} > served {i:.4f}"
    return "incumbent", f"India validation mAP@0.5 {c:.4f} <= served {i:.4f}"


def main(argv=None, ckpt=CKPT, tag="world"):
    cand_files = (f"damage_rdd2022_{tag}.onnx", f"damage_rdd2022_{tag}.json", f"road_damage_detector_{tag}_report.json")
    candidate = _load(os.path.join(ckpt, cand_files[2]))
    incumbent = _load(os.path.join(ckpt, SERVED[2]))
    winner, why = decide(candidate, incumbent)
    record = {
        "rule": RULE, "decided_before_test": True, "decided_unix": int(time.time()),
        "candidate": tag, "winner": winner, "why": why,
        "validation_map50_india": {"candidate": _map50(candidate, "validation"),
                                   "served_before": _map50(incumbent, "validation") if incumbent else None},
        "test_map50_india_reported_only": {"candidate": _map50(candidate, "test"),
                                           "served_before": _map50(incumbent, "test") if incumbent else None},
    }
    if winner == "candidate":
        if not all(os.path.exists(os.path.join(ckpt, f)) for f in cand_files):
            sys.exit(f"candidate files missing in {ckpt}: {cand_files}")
        shutil.copy(os.path.join(ckpt, cand_files[0]), os.path.join(ckpt, SERVED[0]))
        meta = _load(os.path.join(ckpt, cand_files[1]))
        meta.pop("artefact_check", None)            # a new file has to pass its own checks
        meta.pop("deployment_check", None)
        with open(os.path.join(ckpt, SERVED[1]), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        rep = dict(candidate)
        rep.pop("artefact_check", None)
        rep.pop("deployment_check", None)
        rep["onnx"] = dict(rep.get("onnx") or {}, file=SERVED[0])
        rep["selection"] = record
        rep["previous_model"] = {k: v for k, v in (incumbent or {}).items() if k != "previous_model"} or None
        with open(os.path.join(ckpt, SERVED[2]), "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2)
        record["served"] = tag
        print(f"[select-detector] SERVING the {tag} detector: {why}")
        print("  next: python -m scripts.verify_rdd_detector --artefact --clean-roads "
              f"--run-name rdd_{tag} --data-dir datasets/rdd2022_{tag}")
    else:
        record["served"] = (incumbent or {}).get("tag", "india")
        print(f"[select-detector] keeping the served detector: {why}")
    with open(os.path.join(ckpt, SELECTION), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return record


if __name__ == "__main__":
    main()
