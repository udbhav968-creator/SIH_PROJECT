"""
Record real engine results so the static site can show them, labelled as recorded.

    python -m scripts.record_samples                 # 6 photographs, ~1 minute

Why
---
The public site (Vercel) cannot run the models: the stack is ~440 MB and a
serverless function is capped at 250 MB. Without this, "Use a dataset
photograph" on the public site can only say "not deployed here". With it, the
site shows what the engine actually computed on a real photograph, with the date
and the models that computed it - never an analysis it pretends to run.

Which photographs
-----------------
The first defect and clean-road photographs of the same seeded sample that
scripts/measure_detection_quality.py scores, taken IN ORDER, not chosen by how
the result looks. The measured rates for the whole sample are on the Models
page; these are examples of what the screen shows.

Output: web/samples/recorded_results.json (images downscaled to 800 px).
"""

import argparse
import base64
import json
import os
import sys
import time

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

CKPT = os.path.join(ENGINE_ROOT, "checkpoints")
OUT = os.path.join(ENGINE_ROOT, "web", "samples", "recorded_results.json")


def small_jpeg_data_url(path, max_side=800):
    import cv2
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defect", type=int, default=4)
    ap.add_argument("--clean", type=int, default=2)
    a = ap.parse_args(argv)

    from pipeline.deep_inference_pipeline import DeepInferencePipeline
    from scripts.measure_detection_quality import CLEAN_FOLDERS, DEFECT_FOLDERS, defects_in, sample

    pipe = DeepInferencePipeline(CKPT)
    seg = getattr(pipe, "segmenter", None)
    if not (seg and seg.is_ready):
        sys.exit("The segmenter did not load (scikit-learn must be 1.8.x); recorded results would be misleading.")
    seg_name = "U-Net" if type(seg).__name__ == "UNetSegmenter" else "pixel classifier"

    defect = sample(DEFECT_FOLDERS, 12, 7)[: a.defect]
    clean = sample(CLEAN_FOLDERS, 12, 7)[: a.clean]
    out = []
    for (path, folder), expected in [(p, "defect") for p in defect] + [(p, "clean") for p in clean]:
        t0 = time.time()
        res = pipe.audit_image(path, corridor_id="Recorded example")
        ms = round((time.time() - t0) * 1000)
        res = json.loads(json.dumps(res, default=str))       # plain JSON, exactly what the API would send
        res["image_data_url"] = small_jpeg_data_url(path)
        res["image_source_file"] = os.path.basename(path)
        found = defects_in(res)
        out.append({"file": os.path.basename(path), "folder": folder, "expected": expected,
                    "engine_reported_defect": bool(found), "engine_ms": ms, "result": res})
        print(f"  {expected:6s} {os.path.basename(path)[:40]:42s} -> "
              f"{', '.join(d['class_name'].split('(')[0].strip() for d in found) or 'no defect'}  ({ms} ms)")

    doc = {
        "recorded_unix": int(time.time()),
        "recorded_on": time.strftime("%Y-%m-%d"),
        "engine": {"vision_backend": getattr(pipe, "vision_backend", None), "segmenter": seg_name,
                   "segmenter_thresholds": getattr(seg, "thresholds", None)},
        "selection": ("the first photographs of the seeded detection-quality sample, in order - not chosen "
                      "by result; measured rates for the whole sample are on the Models page"),
        "samples": out,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(f"\nwrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB, {len(out)} recorded results, segmenter: {seg_name})")


if __name__ == "__main__":
    main()
