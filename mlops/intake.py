"""
Bring a training run's outputs (the Colab zip) into this checkout through the registry, not by copying.

    python -m mlops intake road_shield_all_outputs.zip           # or a folder it was unpacked into
    python -m mlops intake outputs.zip --no-promote               # register and gate only

For each model in mlops/specs.py that has files in the outputs:

    1. its candidate is the production version's files with the new ones laid over them (a run that
       retrained only the classifier's ONNX does not drop the model's other files)
    2. the candidate is registered (SHA-256 of every file, metrics from its own reports, source "intake")
    3. it is gated against production (floor, max drop, guard metrics)
    4. passed -> promoted (deployed into checkpoints/; the old version archived, so `rollback` brings it back)
       failed -> stays a candidate, checkpoints/ unchanged, reason printed

A model whose new files came without the report its gate reads is not taken in ("needs_report"): its
new weights would otherwise be judged on production's own numbers. Model weights (.onnx, .joblib, ...)
that belong to no spec are not copied, since no gate covers them; they are listed. Other files (logs,
the claim registry, the rebuilt report, measurement reports such as privacy_redaction_report.json) are
copied in as they are. Nothing is deleted: a file the run renamed or removed stays in production's set. Afterwards rebuild the
claims from what is now served: `python -m scripts.build_claims`.

An outputs zip may only write under checkpoints/, logs/ and the report .docx; anything else in it, and any
path that tries to leave this folder, is refused.
"""
import os
import shutil
import tempfile
import zipfile

from mlops.paths import ENGINE_ROOT, sha256_file
from mlops.registry import Registry, RegistryError
from mlops.specs import SPECS

ALLOWED_TOP = ("checkpoints/", "logs/")
WEIGHT_EXT = (".onnx", ".joblib", ".npz", ".pt", ".pth", ".pkl", ".h5", ".tflite")
ALLOWED_FILES = ("CSET485_ROAD_SHIELD_Milestone2_Report_Audited_TrackedChanges.docx",)


def _safe_members(names):
    ok, refused = [], []
    for n in names:
        rel = n.replace("\\", "/")
        if rel.endswith("/"):
            continue
        norm = os.path.normpath(rel).replace("\\", "/")
        if norm.startswith("../") or os.path.isabs(norm) or ".." in norm.split("/"):
            refused.append(n)
        elif norm.startswith(ALLOWED_TOP) or norm in ALLOWED_FILES:
            ok.append((n, norm))
        else:
            refused.append(n)
    return ok, refused


def unpack(src, dest):
    """Unpack a zip (or copy a folder) into dest, keeping only allowed paths. Returns the refused names."""
    if os.path.isdir(src):
        names = [os.path.relpath(os.path.join(r, f), src) for r, _, fs in os.walk(src) for f in fs]
        ok, refused = _safe_members(names)
        for n, norm in ok:
            p = os.path.join(dest, norm)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            shutil.copyfile(os.path.join(src, n), p)
        return refused
    with zipfile.ZipFile(src) as z:
        ok, refused = _safe_members(z.namelist())
        for n, norm in ok:
            p = os.path.join(dest, norm)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with z.open(n) as fin, open(p, "wb") as fout:
                shutil.copyfileobj(fin, fout)
    return refused


def intake(src, reg=None, promote=True, repo_root=ENGINE_ROOT, actor="intake", note=None):
    reg = reg or Registry()
    reg.bootstrap()
    report = {"models": {}, "copied": [], "unchanged_files": 0, "refused": [], "skipped_weights": []}
    with tempfile.TemporaryDirectory(prefix="rs_intake_") as tmp:
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)
        report["refused"] = unpack(src, raw)
        new_ckpt = os.path.join(raw, "checkpoints")
        claimed = set()
        for name in SPECS:
            new_files = reg.files_on_disk(name, new_ckpt) if os.path.isdir(new_ckpt) else {}
            if not new_files:
                continue
            claimed |= set(new_files)
            entry = {"new_files": sorted(new_files)}
            prod = reg.production(name)
            prod_sha = {f["path"]: f["sha256"] for f in (prod or {}).get("files", [])}
            changed = {rel for rel, p in new_files.items() if sha256_file(p) != prod_sha.get(rel)}
            if prod is not None and not changed:
                report["models"][name] = dict(entry, outcome="unchanged", version=prod["version"])
                continue
            cand_root = os.path.join(tmp, "cand_" + name)
            os.makedirs(cand_root)
            if prod is not None:                                   # production's stored files first (not whatever
                reg.materialize(name, prod["version"], cand_root)  # is in checkpoints/: a hand edit stays out) ...
            for rel, p in new_files.items():                       # ... with the run's files over them
                os.makedirs(os.path.dirname(os.path.join(cand_root, rel)), exist_ok=True)
                shutil.copyfile(p, os.path.join(cand_root, rel))
            # The gate reads the report of the model the candidate SERVES (mlops/specs.metric_sources). New weights
            # of that model next to its old report would be judged on production's own numbers and pass, so they
            # are not taken in without their report. Changed reports or selection files alone are fine.
            from mlops.specs import metric_sources, served_weights
            sources = metric_sources(name, cand_root)
            weights = [f for f in changed if f.lower().endswith(WEIGHT_EXT) and served_weights(name, cand_root, f)]
            missing = [f for f in sources if f not in changed]
            if weights and missing:
                report["models"][name] = dict(entry, outcome="needs_report", reason=(
                    f"the run changed {', '.join(sorted(weights)[:3])} but not {', '.join(missing)}, which the gate "
                    f"reads for the model it serves; review it, then register and promote by hand if it should be served"))
                continue
            try:
                v = reg.register(name, root=cand_root, stage="candidate", source="intake", actor=actor,
                                 note=note or f"intake from {os.path.basename(str(src))}")
            except RegistryError as e:
                report["models"][name] = dict(entry, outcome="error", reason=str(e))
                continue
            entry.update(version=v["version"], metrics=v["metrics"])
            if v["stage"] == "production":
                report["models"][name] = dict(entry, outcome="unchanged")
                continue
            gate = reg.gate(name, v["version"])
            entry["gate"] = gate
            if not gate.get("passed"):
                report["models"][name] = dict(entry, outcome="kept_as_candidate",
                                              reason=gate.get("reason") or [c for c in gate.get("checks", [])
                                                                            if not c.get("passed")])
                continue
            if not promote:
                reg.set_stage(name, v["version"], "staging")
                report["models"][name] = dict(entry, outcome="staged")
                continue
            try:
                res = reg.promote(name, v["version"], actor=actor, reason="intake: gate passed")
                report["models"][name] = dict(entry, outcome="promoted", previous=res["previous"])
            except RegistryError as e:
                report["models"][name] = dict(entry, outcome="kept_as_candidate", reason=str(e))
        # everything that is not a model file goes in as it is
        for r, _, fs in os.walk(raw):
            for f in fs:
                p = os.path.join(r, f)
                rel = os.path.relpath(p, raw).replace("\\", "/")
                if rel.startswith("checkpoints/") and rel[len("checkpoints/"):] in claimed:
                    continue
                if rel.startswith("checkpoints/") and rel.lower().endswith(WEIGHT_EXT):
                    report["skipped_weights"].append(rel)   # weights no spec covers get no gate: not copied
                    continue
                dst = (os.path.join(reg.ckpt_dir, rel[len("checkpoints/"):]) if rel.startswith("checkpoints/")
                       else os.path.join(repo_root, rel))
                if os.path.exists(dst) and sha256_file(dst) == sha256_file(p):
                    report["unchanged_files"] += 1
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(p, dst + ".intake")
                os.replace(dst + ".intake", dst)
                report["copied"].append(rel)
    return report


def print_report(rep):
    print(f"{'model':<26} {'outcome':<18} version  detail")
    for name, m in rep["models"].items():
        detail = ""
        if m["outcome"] == "kept_as_candidate":
            r = m.get("reason")
            detail = r if isinstance(r, str) else "; ".join(
                f"{c.get('metric')}: {c['why']}" if c.get("why") else
                f"{c.get('metric')}={c.get('value')} (limit {c.get('limit')}, {c.get('rule')})" for c in (r or []))
        elif m["outcome"] == "promoted":
            detail = f"replaced v{m.get('previous')}" if m.get("previous") else "first production version"
        elif m["outcome"] in ("error", "needs_report"):
            detail = m.get("reason", "")
        print(f"{name:<26} {m['outcome']:<18} v{m.get('version', '-'):<6} {detail[:150]}")
    print(f"\nother files copied: {len(rep['copied'])}, already identical: {rep['unchanged_files']}")
    if rep.get("skipped_weights"):
        print(f"model weights that belong to no registered model, not copied (no gate covers them): "
              f"{len(rep['skipped_weights'])} e.g. {rep['skipped_weights'][:3]}")
    if rep["refused"]:
        print(f"refused (outside checkpoints/, logs/ or the report): {len(rep['refused'])} e.g. {rep['refused'][:3]}")
