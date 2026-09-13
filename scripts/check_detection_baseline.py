"""
Does checkpoints/detection_quality_report.json still describe this machine?

Prints FRESH, or STALE followed by the reason. Exit code is 0 either way -
staleness is a normal state, not an error, and a non-zero code here would stop
the release script for something it is about to fix.

    python -m scripts.check_detection_baseline
"""

import json
import os
import sys

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

CKPT = os.path.join(ENGINE_ROOT, "checkpoints")


def main():
    path = os.path.join(CKPT, "detection_quality_report.json")
    if not os.path.exists(path):
        print("STALE no baseline has ever been measured here")
        return
    try:
        rep = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"STALE baseline unreadable: {e}")
        return

    from models.defect_segmenter import DefectSegmenter
    from pipeline.corpus_fingerprint import fingerprint, describe_mismatch

    seg = DefectSegmenter()
    if not seg.is_ready:
        print("STALE the segmenter does not load, so nothing can be fingerprinted")
        return
    folders = (rep.get("clean_folders") or []) + (rep.get("defect_folders") or [])
    why = describe_mismatch(rep.get("fingerprint"), fingerprint(seg, folders))
    if why:
        print(f"STALE {why}")
    else:
        print(f"FRESH detection {float(rep.get('detection_rate') or 0) * 100:.1f}% "
              f"over {rep.get('defect_photographs')} photographs")


if __name__ == "__main__":
    main()
