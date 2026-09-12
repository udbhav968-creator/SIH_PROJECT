"""
The whole thing, end to end, in one command.

    python -m scripts.run_full_pipeline                  # everything
    python -m scripts.run_full_pipeline --fast           # skip the slow validation sweep
    python -m scripts.run_full_pipeline --backbone mobilenetv2
    python -m scripts.run_full_pipeline --from train_cnn # resume from a stage

Stages, in order, each one a script that already exists and can be run alone:

    1. audit        scripts/fix_label_conflicts.py       quarantine photos filed
                                                         under two classes at once
    2. data         scripts/fetch_cracks_potholes_dataset.py
                                                         1,667 annotated defect
                                                         crops from the DNIT set
    2b. kaggle      scripts/fetch_kaggle_datasets.py     Kaggle road-condition
                                                         datasets, mapped to the
                                                         seven classes and
                                                         de-duplicated
    3. train_base   training/train_mega_suite.py         hand-crafted features,
                                                         IMU classifier, the rest
    4. backbone     scripts/fetch_cnn_backbone.py        ImageNet CNN weights
    5. train_cnn    training/train_cnn_head.py --compare embeddings + head, and
                                                         the baseline on the same
                                                         split for comparison
    6. validate     scripts/validate_models.py           leakage, grouped CV,
                                                         calibration, latency,
                                                         robustness
    7. test         tests/test_road_shield.py            regression suite

A stage that fails does not abort the run unless it is one the later stages
depend on; the summary at the end says what passed, what failed and what the
consequence is. Every number printed is produced by the stage that printed it -
this script computes no metrics of its own.

Output: checkpoints/pipeline_run_report.json, which the dashboard's System tab
reads to show when the pipeline last ran and how each stage went.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
REPORT_PATH = os.path.join(CKPT_DIR, "pipeline_run_report.json")

# name, argv, required-by-later-stages, one-line purpose
STAGES = [
    ("audit", ["-m", "scripts.fix_label_conflicts"], False,
     "quarantine photographs filed under more than one class"),
    ("data", ["-m", "scripts.fetch_cracks_potholes_dataset", "--limit", "2235"], False,
     "download real annotated crack and pothole crops"),
    ("kaggle", ["-m", "scripts.fetch_kaggle_datasets"], False,
     "download and ingest the Kaggle road-condition datasets"),
    ("train_base", ["-m", "training.train_mega_suite"], True,
     "train the hand-crafted-feature and IMU models"),
    ("backbone", ["-m", "scripts.fetch_cnn_backbone"], False,
     "download the ImageNet CNN used as a feature extractor"),
    ("train_cnn", ["-m", "training.train_cnn_head", "--compare"], False,
     "train the classifier head on CNN embeddings"),
    ("validate", ["-m", "scripts.validate_models"], False,
     "leakage audit, grouped cross-validation, calibration, latency"),
    ("test", ["-m", "unittest", "tests.test_road_shield"], False,
     "regression suite"),
]

CONSEQUENCE = {
    "audit": "duplicate labels stay in the training data and accuracy is overstated",
    "data": "training runs on whatever images are already on disk",
    "kaggle": "no Kaggle images are added - usually missing credentials; "
              "training continues on the DNIT data already downloaded",
    "train_base": "no models on disk - the server cannot classify anything",
    "backbone": "the CNN path is unavailable; the server serves the hand-crafted baseline",
    "train_cnn": "the server serves the hand-crafted baseline instead of the CNN head",
    "validate": "no leakage or calibration evidence for the judges",
    "test": "regressions in the pipeline would go unnoticed",
}


def run_stage(name, argv, tail_lines=14):
    print(f"\n{'=' * 72}\n[{name}] python {' '.join(argv)}\n{'=' * 72}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable] + argv, cwd=ENGINE_ROOT,
                          capture_output=True, text=True)
    seconds = time.time() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    for ln in lines[-tail_lines:]:
        print("   ", ln)
    ok = proc.returncode == 0
    print(f"    -> {'ok' if ok else 'FAILED (exit %d)' % proc.returncode} in {seconds:.0f}s", flush=True)
    return {"ok": ok, "exit_code": proc.returncode, "seconds": round(seconds, 1),
            "tail": lines[-tail_lines:]}


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def collect_measurements():
    """Read what the stages wrote. Nothing here recomputes or estimates."""
    import glob
    out = {}
    base = read_json(os.path.join(CKPT_DIR, "vision_distress_report.json"))
    if base:
        out["handcrafted_features"] = {
            "held_out_accuracy": base.get("held_out_validation_accuracy"),
            "held_out_photos": base.get("held_out_validation_photos"),
        }
    for path in sorted(glob.glob(os.path.join(CKPT_DIR, "cnn_head_*_report.json"))):
        rep = read_json(path)
        if rep:
            out[f"cnn_{rep.get('backbone')}"] = {
                "held_out_accuracy": rep.get("held_out_test_accuracy"),
                "held_out_macro_f1": rep.get("held_out_test_macro_f1"),
                "held_out_images": rep.get("held_out_test_images"),
                "held_out_photographs": rep.get("held_out_test_photographs"),
                "head": rep.get("head"),
            }
    val = read_json(os.path.join(CKPT_DIR, "validation_report.json"))
    if val:
        out["validation"] = {k: val[k] for k in ("leakage", "cross_validation", "calibration", "latency")
                             if k in val}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "mobilenetv2"])
    ap.add_argument("--fast", action="store_true", help="skip the validation sweep")
    ap.add_argument("--from", dest="start_at", default=None,
                    help="resume from this stage name")
    ap.add_argument("--only", default=None, help="run just this stage")
    ap.add_argument("--list", action="store_true", help="print the stages and exit")
    args = ap.parse_args()

    if args.list:
        for name, argv, required, why in STAGES:
            print(f"  {name:12s} {why}")
        return

    stages = list(STAGES)
    if args.backbone != "resnet50":
        stages = [(n, (a + ["--model", args.backbone]) if n == "backbone"
                   else (a + ["--backbone", args.backbone]) if n == "train_cnn" else a, r, w)
                  for n, a, r, w in stages]
    if args.fast:
        stages = [s for s in stages if s[0] != "validate"]
    if args.only:
        stages = [s for s in stages if s[0] == args.only]
        if not stages:
            sys.exit(f"No stage named {args.only!r}. Use --list.")
    elif args.start_at:
        names = [s[0] for s in stages]
        if args.start_at not in names:
            sys.exit(f"No stage named {args.start_at!r}. Use --list.")
        stages = stages[names.index(args.start_at):]

    print(f"ROAD-SHIELD full pipeline - {len(stages)} stages, backbone {args.backbone}")
    started = time.time()
    results = {}
    for name, argv, required, why in stages:
        results[name] = run_stage(name, argv)
        results[name]["purpose"] = why
        if not results[name]["ok"]:
            results[name]["consequence"] = CONSEQUENCE.get(name, "")
            if required:
                print(f"\n[{name}] is required by the stages after it. Stopping here.")
                break

    report = {
        "finished_unix": int(time.time()),
        "total_seconds": round(time.time() - started, 1),
        "backbone": args.backbone,
        "stages": results,
        "measurements": collect_measurements(),
    }
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n{'=' * 72}\nSUMMARY   ({report['total_seconds']:.0f}s total)\n{'=' * 72}")
    for name, res in results.items():
        mark = "ok    " if res["ok"] else "FAILED"
        print(f"  {mark} {name:12s} {res['seconds']:6.0f}s   {res['purpose']}")
        if not res["ok"]:
            print(f"         -> {res.get('consequence', '')}")

    m = report["measurements"]
    if m:
        print("\nMeasured (read back from what each stage wrote):")
        for key, val in m.items():
            if key == "validation":
                continue
            acc = val.get("held_out_accuracy")
            f1 = val.get("held_out_macro_f1")
            if acc is not None:
                print(f"  {key:24s} accuracy {acc * 100:5.1f}%"
                      + (f"   macro-F1 {f1:.3f}" if f1 is not None else ""))
    print(f"\nreport -> {REPORT_PATH}")
    failed = [n for n, r in results.items() if not r["ok"]]
    sys.exit(1 if any(req for n, _a, req, _w in stages if n in failed) else 0)


if __name__ == "__main__":
    main()
