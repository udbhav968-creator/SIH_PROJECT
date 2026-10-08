"""
Model quality check for CI: python -m mlops.ci_check

Reads the metrics of every model whose files are committed in checkpoints/ (the same report paths the
registry uses) and fails if any is below its promotion floor or above a guard ceiling. A pull request
that commits a worse model, or a report that no longer matches its spec, fails here before review.
Models with no automatic gate (published weights) and models not on disk are listed and skipped.
"""
import sys

from mlops.paths import CKPT_DIR
from mlops.registry import Registry
from mlops.specs import SPECS


def check(ckpt_dir=CKPT_DIR, store=None):
    import tempfile
    reg = Registry(store=store or tempfile.mkdtemp(prefix="rs_ci_"), ckpt_dir=ckpt_dir)
    rows, failed = [], False
    for name, s in SPECS.items():
        if not reg.files_on_disk(name):
            rows.append((name, "skip", "not committed"))
            continue
        g = s.get("gate") or {}
        if not g.get("primary"):
            rows.append((name, "skip", g.get("manual", "no automatic gate")))
            continue
        m = reg.read_metrics(name)
        v = m.get(g["primary"])
        problems = []
        if v is None:
            problems.append(f"{g['primary']} missing from its report")
        elif g.get("floor") is not None and (v < g["floor"] if g.get("direction", "max") == "max" else v > g["floor"]):
            problems.append(f"{g['primary']} {v:.4g} below floor {g['floor']}")
        for gk, rule in (g.get("guards") or {}).items():
            if rule.get("ceiling") is None:
                continue
            if m.get(gk) is None:
                problems.append(f"{gk} missing from its report (it has a ceiling of {rule['ceiling']})")
            elif m[gk] > rule["ceiling"]:
                problems.append(f"{gk} {m[gk]:.4g} above ceiling {rule['ceiling']}")
        failed |= bool(problems)
        rows.append((name, "FAIL" if problems else "ok",
                     "; ".join(problems) or f"{g['primary']} = {v:.4g} (floor {g.get('floor')})"))
    return rows, failed


def main():
    rows, failed = check()
    for name, st, why in rows:
        print(f"{st:<5} {name:<26} {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
