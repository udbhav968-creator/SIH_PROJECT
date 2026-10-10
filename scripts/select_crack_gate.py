"""
Decide whether the crack verifier (models/crack_verifier.py) is served: measure the whole pipeline with and
without it, through audit_image(), on the same photographs.

    python -m scripts.select_crack_gate                 # 40 photographs per folder (~20-30 min on a laptop)
    python -m scripts.select_crack_gate --per-folder 12  # the standard detection-quality sample, quicker

Rule (fixed before measuring): served only if, with the gate, FEWER clean photographs get a crack or pothole
report AND NOT ONE annotated defect photograph that was found without the gate is missed with it. The
photographs are those of scripts/measure_detection_quality.py (seed 7), which training/train_crack_verifier.py
keeps out of its training and threshold data.

The decision is written to checkpoints/crack_verifier_selection.json with the SHA-256 of the model file it
was made for, so a retrained verifier is never served on an older decision.
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

CKPT = os.path.join(ROOT, "checkpoints")


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run(mode, photos):
    from pipeline.deep_inference_pipeline import DeepInferencePipeline
    from scripts.measure_detection_quality import defects_in
    before = os.environ.get("ROAD_SHIELD_CRACK_GATE")
    os.environ["ROAD_SHIELD_CRACK_GATE"] = mode                # read once, when the pipeline is built
    try:
        pipe = DeepInferencePipeline(CKPT)
    finally:
        if before is None:
            os.environ.pop("ROAD_SHIELD_CRACK_GATE", None)
        else:
            os.environ["ROAD_SHIELD_CRACK_GATE"] = before
    if mode == "force" and pipe.crack_verifier is None:
        raise SystemExit("the crack verifier did not load: python -m training.train_crack_verifier first")
    out = {}
    for path, expected in photos:
        res = pipe.audit_image(path)
        out[path] = {"expected": expected, "found": bool(defects_in(res)),
                     "gate": res.get("crack_gate")}
    del pipe
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-folder", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    from models.crack_verifier import MODEL_FILE, SELECTION_FILE
    from scripts.measure_detection_quality import CLEAN_FOLDERS, DEFECT_FOLDERS, sample
    model_path = os.path.join(CKPT, MODEL_FILE)
    if not os.path.exists(model_path):
        raise SystemExit("no crack verifier: python -m training.train_crack_verifier")
    photos = [(p, "clean") for p, _ in sample(CLEAN_FOLDERS, a.per_folder, a.seed)] + \
             [(p, "defect") for p, _ in sample(DEFECT_FOLDERS, a.per_folder, a.seed)]
    t0 = time.time()
    off = run("off", photos)
    on = run("force", photos)

    def tally(r):
        return {"false_positive_photos": sum(v["found"] for v in r.values() if v["expected"] == "clean"),
                "detected_photos": sum(v["found"] for v in r.values() if v["expected"] == "defect")}
    t_off, t_on = tally(off), tally(on)
    lost = [os.path.relpath(p, ROOT) for p in off if off[p]["expected"] == "defect" and off[p]["found"]
            and not on[p]["found"]]
    fixed = [os.path.relpath(p, ROOT) for p in off if off[p]["expected"] == "clean" and off[p]["found"]
             and not on[p]["found"]]
    gained = [os.path.relpath(p, ROOT) for p in off if off[p]["expected"] == "clean" and not off[p]["found"]
              and on[p]["found"]]
    serve = t_on["false_positive_photos"] < t_off["false_positive_photos"] and not lost
    clean_n = sum(1 for _, e in photos if e == "clean")
    defect_n = len(photos) - clean_n
    why = (f"without the gate {t_off['false_positive_photos']}/{clean_n} clean photographs had a false alarm and "
           f"{t_off['detected_photos']}/{defect_n} defects were found; with it {t_on['false_positive_photos']}/{clean_n} "
           f"and {t_on['detected_photos']}/{defect_n}" + (f"; it lost {len(lost)} detected defect(s)" if lost else ""))
    selection = {
        "rule": "served only if fewer clean photographs get a defect report and no detected defect is lost",
        "served": serve, "why": why, "without_gate": t_off, "with_gate": t_on,
        "clean_photographs": clean_n, "defect_photographs": defect_n,
        "false_alarms_removed": fixed, "false_alarms_added": gained, "detections_lost": lost,
        "model_sha16": sha16(model_path), "per_folder": a.per_folder, "seed": a.seed,
        "seconds": round(time.time() - t0, 1), "decided_unix": int(time.time()),
    }
    with open(os.path.join(CKPT, SELECTION_FILE), "w", encoding="utf-8") as fh:
        json.dump(selection, fh, indent=1)
    print(json.dumps({k: v for k, v in selection.items() if k not in ("false_alarms_removed",)}, indent=1))
    return selection


if __name__ == "__main__":
    main()
