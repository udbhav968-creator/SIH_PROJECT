"""
Retraining pipeline: validate data -> train -> register -> restore production -> gate -> stage/promote.

    python -m mlops retrain imu_classifier                 # candidate only, production untouched
    python -m mlops retrain imu_classifier --promote       # promote if the gate passes
    python -m mlops retrain ood_guard --ood-dir ../imagenet-sample-images

The training scripts write straight into checkpoints/. To keep production serving while a candidate
is judged, the pipeline:

  1. validates the data the script reads (photographs or windows per folder, label-conflict folder,
     a fingerprint of which files) and records it on the run
  2. stores every file in checkpoints/ in the blob store (content-addressed: unchanged files cost
     nothing) and makes sure the current files are registered
  3. runs the training command; its output goes to the run log
  4. registers the files of this model that changed as a new candidate version, with its metrics
  5. puts every changed file back as it was, the model's own and any other the script touched
     (training/train_civil_models.py, for example, also rewrites the segmenter)
  6. runs the gate against production: passed -> staging (and promoted with --promote); failed ->
     the candidate stays in the registry with the reasons, and production is unchanged

GPU models (the CNN, the U-Net, the YOLO detector) train in Colab; copy the outputs back
(scripts/apply_colab_outputs.ps1) and run `python -m mlops register <model>` and `gate` / `promote`.
"""
import glob
import os
import subprocess
import sys
import time

from mlops.paths import ENGINE_ROOT, sha256_file
from mlops.registry import Registry, RegistryError
from mlops.specs import SPECS, spec
from mlops.tracking import start_run

DATA = {
    "imu_classifier": ["datasets/04_mobile_imu_telemetry_100hz"],
    "ood_guard": ["datasets/*/real_images"],
    "pci_regressor": ["datasets/06_astm_d6433_pci_benchmark", "datasets/02_kaggle_pothole_600"],
    "depth_estimator": ["datasets/02_kaggle_pothole_600", "datasets/03_crack500_fatigue"],
    "deterioration_forecaster": ["datasets/07_monsoon_pavement_deterioration"],
}


def validate_data(model):
    """What the training script will read: files per folder and a digest of which files."""
    import hashlib
    folders = {}
    for pat in DATA.get(model, []):
        for d in sorted(glob.glob(os.path.join(ENGINE_ROOT, pat))):
            if not os.path.isdir(d):
                continue
            files = sorted(os.path.relpath(p, ENGINE_ROOT).replace("\\", "/")
                           for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                           if os.path.isfile(p) and "_label_conflicts" not in p)
            h = hashlib.sha256("\n".join(files).encode()).hexdigest()[:16]
            folders[os.path.relpath(d, ENGINE_ROOT).replace("\\", "/")] = {"files": len(files), "digest": h}
    conflicts = len(glob.glob(os.path.join(ENGINE_ROOT, "datasets", "_label_conflicts", "**", "*.*"), recursive=True))
    problems = []
    if DATA.get(model) and not folders:
        problems.append("none of the data folders exist on this machine")
    for f, v in folders.items():
        if v["files"] == 0:
            problems.append(f"{f} is empty")
    return {"folders": folders, "label_conflicts_set_aside": conflicts, "problems": problems,
            "passed": not problems}


def _model_files(reg):
    """Every file in checkpoints/ that belongs to some model (by the specs' patterns). Runtime files the server
    keeps there (the ledger, labelling photographs, shadow copies, a production drift reference) are not
    model files and are never snapshotted, reverted or deleted."""
    out = {}
    for name in SPECS:
        out.update(reg.files_on_disk(name))
    return out


def _snapshot(reg):
    return {rel: {"path": rel, "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
            for rel, p in _model_files(reg).items()}


def retrain(model, registry=None, promote=False, actor="cli", ood_dir=None, runner=None, timeout=6 * 3600):
    s = spec(model)
    if isinstance(s["train"], str):
        raise RegistryError(f"{model} is trained with: {s['train']}. Copy the outputs into checkpoints/, then "
                            f"python -m mlops register {model}")
    cmd = [sys.executable if c == "python" else c for c in s["train"]]
    if "{ood_dir}" in cmd:
        if not ood_dir:
            raise RegistryError("--ood-dir is required (git clone --depth 1 "
                                "https://github.com/EliSchwartz/imagenet-sample-images)")
        cmd = [ood_dir if c == "{ood_dir}" else c for c in cmd]
    reg = registry or Registry()
    reg.bootstrap(actor=actor)
    runner = runner or (lambda c: subprocess.run(c, cwd=ENGINE_ROOT, capture_output=True, text=True, timeout=timeout))
    summary = {"model": model, "steps": []}

    def step(name, **kw):
        summary["steps"].append({"step": name, "at": round(time.time(), 1), **kw})
        print(f"[retrain] {name}: " + ", ".join(f"{k}={v}" for k, v in kw.items() if not isinstance(v, (dict, list))),
              flush=True)

    with start_run(f"retrain/{model}", params={"command": " ".join(cmd), "promote": promote},
                   tags={"pipeline": "mlops.retrain"}) as run:
        summary["run_id"] = run.run_id
        data = validate_data(model)
        run.log_params({"data": data["folders"], "label_conflicts_set_aside": data["label_conflicts_set_aside"]})
        step("validate_data", passed=data["passed"], folders=data["folders"], problems=data["problems"])
        if not data["passed"]:
            run.set_tag("outcome", "data_invalid")
            summary["outcome"] = "data_invalid"
            return summary

        before = _snapshot(reg)
        reg.keep(list(before.values()))
        prod = reg.production(model)
        step("snapshot", files=len(before), production=prod["version"] if prod else None)
        changed, cand = [], None
        try:
            t0 = time.time()
            try:
                proc = runner(cmd)
            except Exception as e:           # a timeout or a crash still ends with production restored
                proc = type("Failed", (), {"returncode": -1, "stdout": "", "stderr": f"{type(e).__name__}: {e}"})()
            out = (getattr(proc, "stdout", "") or "") + (getattr(proc, "stderr", "") or "")
            log = os.path.join(reg.store, "runs", f"{run.run_id}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, "w", encoding="utf-8") as fh:
                fh.write(out)
            run.log_artifact(log)
            run.log_metric("train_seconds", round(time.time() - t0, 1))
            step("train", returncode=proc.returncode, seconds=round(time.time() - t0, 1),
                 log=os.path.relpath(log, ENGINE_ROOT))

            after = _snapshot(reg)
            changed = sorted(p for p in set(before) | set(after)
                             if (before.get(p) or {}).get("sha256") != (after.get(p) or {}).get("sha256"))
            mine = [p for p in changed if reg._belongs(model, p)]
            if proc.returncode != 0:
                run.set_tag("outcome", "training_failed")
                summary["outcome"] = "training_failed"
                summary["log_tail"] = out[-2000:]
                return summary
            if not mine:
                run.set_tag("outcome", "no_change")
                summary["outcome"] = "no_change"
                step("register", note="the training run did not change this model's files")
                return summary
            cand = reg.register(model, stage="candidate", run_id=run.run_id, source="retrain", actor=actor,
                                note=f"retrained; changed {', '.join(mine[:6])}")
            run.log_metrics(cand["metrics"])
            run.set_tag("candidate_version", cand["version"])
            summary["candidate"] = {"version": cand["version"], "metrics": cand["metrics"]}
            step("register", version=cand["version"], changed=mine, metrics=cand["metrics"])
        finally:
            if not changed:                  # the run died before comparing: compare now
                now = _snapshot(reg)
                changed = sorted(p for p in set(before) | set(now)
                                 if (before.get(p) or {}).get("sha256") != (now.get(p) or {}).get("sha256"))
            reg.restore([before[p] for p in changed if p in before])
            for p in changed:
                if p not in before:
                    try:
                        os.remove(os.path.join(reg.ckpt_dir, p))
                    except OSError:
                        pass
            step("restore_production", files=len(changed),
                 others_touched=[p for p in changed if not reg._belongs(model, p)])

        gate = reg.gate(model, cand["version"])
        summary["gate"] = gate
        run.log_metric("gate_passed", 1.0 if gate["passed"] else 0.0)
        step("gate", passed=gate["passed"], checks=gate["checks"])
        if not gate["passed"]:
            run.set_tag("outcome", "gate_failed")
            summary["outcome"] = "gate_failed"
            return summary
        reg.set_stage(model, cand["version"], "staging", actor=actor)
        if promote:
            res = reg.promote(model, cand["version"], actor=actor, reason=f"retrain run {run.run_id}")
            step("promote", version=cand["version"], previous=res["previous"])
            run.set_tag("outcome", "promoted")
            summary["outcome"] = "promoted"
        else:
            run.set_tag("outcome", "staged")
            summary["outcome"] = "staged"
            step("stage", version=cand["version"], note=f"promote with: python -m mlops promote {model} {cand['version']}")
    return summary
