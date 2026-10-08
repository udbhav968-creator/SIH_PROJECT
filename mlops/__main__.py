"""
python -m mlops <command>

    status                         every model: production version, metrics, do the served files match
    bootstrap                      register what is in checkpoints/ now as version 1 (done automatically)
    versions MODEL                 all versions of a model
    register MODEL [--stage staging] [--note TEXT]
                                   record the model's current files in checkpoints/ as a new version
    gate MODEL VERSION             would this version pass promotion?
    stage MODEL VERSION STAGE      candidate | staging | archived
    promote MODEL VERSION [--force --reason TEXT]
    rollback MODEL                 put the previous production version back
    retrain MODEL [--promote] [--ood-dir DIR]
    runs [--experiment NAME]       training runs and their metrics
    run RUN_ID                     one run in full
    events [MODEL]                 the registry's history

A running server keeps serving the models it loaded; after promote or rollback from here, restart it,
or use the MLOps page (which reloads the models in place).
"""
import argparse
import json
import sys
import time


def _t(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mlops", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("bootstrap")
    p = sub.add_parser("versions"); p.add_argument("model")
    p = sub.add_parser("register"); p.add_argument("model"); p.add_argument("--stage", default="candidate")
    p.add_argument("--note")
    p = sub.add_parser("gate"); p.add_argument("model"); p.add_argument("version", type=int)
    p = sub.add_parser("stage"); p.add_argument("model"); p.add_argument("version", type=int); p.add_argument("stage")
    p = sub.add_parser("promote"); p.add_argument("model"); p.add_argument("version", type=int)
    p.add_argument("--force", action="store_true"); p.add_argument("--reason")
    p = sub.add_parser("rollback"); p.add_argument("model"); p.add_argument("--reason")
    p = sub.add_parser("retrain"); p.add_argument("model"); p.add_argument("--promote", action="store_true")
    p.add_argument("--ood-dir")
    p = sub.add_parser("runs"); p.add_argument("--experiment"); p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("run"); p.add_argument("run_id")
    p = sub.add_parser("events"); p.add_argument("model", nargs="?")
    a = ap.parse_args(argv)

    from mlops.registry import Registry, RegistryError
    from mlops import tracking
    reg = Registry()
    try:
        if a.cmd == "status":
            reg.bootstrap()
            for m in reg.overview():
                prod = m["production"]
                head = f"{m['model']:<26} " + (f"v{prod['version']:<3}" if prod else "--  ")
                match = "" if not prod else ("files match" if m["served_files_match"] else
                                              "FILES CHANGED: " + ", ".join(m["changed_files"][:3]))
                stg = f"  staging v{m['staging']['version']}" if m["staging"] else ""
                mets = ", ".join(f"{k}={v:.4g}" for k, v in (prod or {}).get("metrics", {}).items())
                print(f"{head} {match}{stg}\n{'':27}{mets or ('not on disk' if not m['on_disk'] else '')}")
        elif a.cmd == "bootstrap":
            print(json.dumps(reg.bootstrap(), indent=2))
        elif a.cmd == "versions":
            for v in reg.versions(a.model):
                print(f"v{v['version']:<3} {v['stage']:<10} {_t(v['created_at'])}  {v['source'] or '':<9} "
                      f"run={v['run_id'] or '-'}  {json.dumps(v['metrics'])}")
        elif a.cmd == "register":
            v = reg.register(a.model, stage=a.stage, note=a.note, source="register")
            print(f"{a.model} v{v['version']} ({v['stage']}) {json.dumps(v['metrics'])}")
        elif a.cmd == "gate":
            print(json.dumps(reg.gate(a.model, a.version), indent=2))
        elif a.cmd == "stage":
            v = reg.set_stage(a.model, a.version, a.stage)
            print(f"{a.model} v{v['version']} -> {v['stage']}")
        elif a.cmd == "promote":
            print(json.dumps(reg.promote(a.model, a.version, force=a.force, reason=a.reason), indent=2))
        elif a.cmd == "rollback":
            print(json.dumps(reg.rollback(a.model, reason=a.reason), indent=2))
        elif a.cmd == "retrain":
            from mlops.retrain import retrain
            res = retrain(a.model, reg, promote=a.promote, ood_dir=a.ood_dir)
            print(json.dumps({k: v for k, v in res.items() if k != "steps"}, indent=2))
            return 0 if res.get("outcome") in ("staged", "promoted", "no_change") else 1
        elif a.cmd == "runs":
            for r in tracking.list_runs(a.experiment, a.limit):
                mets = ", ".join(f"{k}={v:.4g}" for k, v in list(r["metrics"].items())[:5])
                print(f"{r['run_id']}  {r['experiment']:<24} {r['status']:<9} {_t(r['started_at'])}  "
                      f"{r['duration_s'] or '-':>7}s  {r['git_commit'] or ''}  {mets}")
        elif a.cmd == "run":
            print(json.dumps(tracking.get_run(a.run_id), indent=2, default=str))
        elif a.cmd == "events":
            for e in reg.events(a.model):
                print(f"{_t(e['at'])}  {e['model']:<24} v{e['version'] or '-':<3} {e['action']:<14} "
                      f"{e['actor'] or ''}  {json.dumps(e['detail'])[:160]}")
    except (RegistryError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
