"""
Rebuild the audited Milestone 2 report from the checkpoints on disk.

    python scripts/report_audit/build_report.py
    python scripts/report_audit/build_report.py --tests-log logs/tests_after.txt

Input : CSET485_ROAD_SHIELD_Milestone2_Report_Corrected_TrackedChanges.docx (the
        earlier fact-checked report, committed in the repo root)
Output: CSET485_ROAD_SHIELD_Milestone2_Report_Audited_TrackedChanges.docx

Every number written into the report is read from a checkpoint report
(classifier, fine-tuning summary, model selection, Indian-roads evaluation,
IMU comparison, end-to-end benchmark) - none is typed by hand. Changes are
tracked under the author "Claude (audit 2026-10-03)", on top of the earlier
tracked changes, so a reader can accept or reject each one in Word.
"""
import argparse
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import track         # noqa: E402
import edits_part1   # noqa: E402
import edits_part2   # noqa: E402
import facts         # noqa: E402

SRC = os.path.join(ROOT, "CSET485_ROAD_SHIELD_Milestone2_Report_Corrected_TrackedChanges.docx")
DST = os.path.join(ROOT, "CSET485_ROAD_SHIELD_Milestone2_Report_Audited_TrackedChanges.docx")


def tests_line_from_log(path):
    """'Ran 176 tests ... OK (skipped=2)' -> a sentence; None if the log is unusable."""
    if not path or not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    ran = re.findall(r"Ran (\d+) tests?", txt)
    if not ran:
        return None
    n = int(ran[-1])
    tail = txt[txt.rfind("Ran "):]
    if re.search(r"^OK", tail, re.M):
        sk = re.search(r"skipped=(\d+)", tail)
        return f"all {n} automated tests passing" + (f" ({sk.group(1)} skipped by design)" if sk else "")
    fails = re.search(r"failures=(\d+)", tail)
    errs = re.search(r"errors=(\d+)", tail)
    bad = int(fails.group(1) if fails else 0) + int(errs.group(1) if errs else 0)
    return f"{n} automated tests, {n - bad} passing and {bad} failing (see logs/tests_after.txt)"


def bench_from_report(ck):
    p = os.path.join(ck, "real_distinct_images_deep_audit_report.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p))
    out = {"n": b["total_real_images"], "mean_ms": b["mean_latency_ms"], "cost": b["total_repair_cost_inr"],
           "where": (f"a {b['cpu_count']}-thread CPU" if b.get("cpu_count") else "the recorded benchmark machine")}
    if "total_tonnage_t" in b:
        out["tonnes"] = b["total_tonnage_t"]
        out["seal_n"] = b.get("work_orders_issued")
        out["seal_ok"] = b.get("work_orders_seal_verified")
    return out


def build(ck, tests_line, src=SRC, dst=DST):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(src) as z:
        z.extractall(tmp)
    doc = os.path.join(tmp, "word", "document.xml")
    xml = open(doc, encoding="utf-8").read()
    track._id[0] = 12000
    for old, new in edits_part1.E:
        xml = track.edit(xml, old, new)
    F = facts.build(ck, tests_line, bench_from_report(ck))
    xml = edits_part2.apply(xml, F)
    cnt = itertools.count(30001)
    xml = re.sub(r'(<w:(?:ins|del) w:id=")\d+(")', lambda m: f"{m.group(1)}{next(cnt)}{m.group(2)}", xml)
    open(doc, "w", encoding="utf-8").write(xml)
    if os.path.exists(dst):
        os.remove(dst)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _dirs, files in os.walk(tmp):
            for f in files:
                full = os.path.join(base, f)
                z.write(full, os.path.relpath(full, tmp))
    shutil.rmtree(tmp, ignore_errors=True)
    return F["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--tests-log", default=os.path.join(ROOT, "logs", "tests_after.txt"))
    ap.add_argument("--tests-line", default=None, help="override the sentence about the test suite")
    a = ap.parse_args()
    line = a.tests_line or tests_line_from_log(a.tests_log) or "an automated test suite (run it with python -m unittest discover -s tests -t .)"
    s = build(a.checkpoints, line)
    print(f"wrote {DST}")
    print(json.dumps({k: v for k, v in s.items() if k != "ind"}, indent=1, default=str))


if __name__ == "__main__":
    main()
