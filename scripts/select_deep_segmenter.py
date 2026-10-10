"""
Should a deep-segmenter candidate (e.g. SegFormer) replace the U-Net? Decided end to end, then the usual
U-Net-versus-pixel-classifier check decides what is served.

    python -m training.train_unet_multi --arch segformer-b1 --out checkpoints/segformer_candidate
    python -m scripts.select_deep_segmenter --candidate checkpoints/segformer_candidate

Rule (fixed before measuring): through audit_image() on the detection-quality photographs (seed 7, the same
ones scripts/segmenter_deployment_check.py uses), the candidate must find at least as many defect
photographs AND raise no more false alarms than the current deep segmenter, AND be strictly better on one
of the two or, if tied on both, have the higher mean calibration IoU (crack + pothole, DNIT calibration
split, recorded by its own training run). Its exported ONNX file must have reproduced the network on the
test split, and a smoke run never replaces anything.

It must also have won its own IoU selection against the pixel classifier (otherwise it could never be served).
If it passes, the current files are backed up to checkpoints/segmenter_backup/<time>/, and its ONNX file,
metadata and selection record are copied into checkpoints/ (the deep-segmenter slot; restored if anything
after that fails), and scripts/segmenter_deployment_check.py then decides between it and the pixel classifier exactly as
for the U-Net. The comparison is recorded in segmenter_selection.json under "deep_candidate_check" either way.
On your laptop, `python -m mlops intake` gates the result again against production.
"""
import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

CKPT = os.path.join(ROOT, "checkpoints")


def cal_iou(meta_or_sel):
    v = ((meta_or_sel or {}).get("validation") or {}).get("unet") or {}
    vals = [((v.get(c) or {}).get("iou")) for c in ("crack", "pothole")]
    return sum(vals) / 2 if all(isinstance(x, (int, float)) for x in vals) else None


def decide(cand, cur, cand_cal, cur_cal, verified, smoke):
    if smoke:
        return False, "smoke run: never replaces anything"
    if not verified:
        return False, "the candidate's exported ONNX file did not reproduce the network on the test split"
    d_hit = cand["detected_photos"] - cur["detected_photos"]
    d_fp = cand["false_positive_photos"] - cur["false_positive_photos"]
    if d_hit < 0 or d_fp > 0:
        return False, (f"end to end it found {cand['detected_photos']} defect photographs with "
                       f"{cand['false_positive_photos']} false alarms; the current deep segmenter "
                       f"{cur['detected_photos']} with {cur['false_positive_photos']}")
    if d_hit > 0 or d_fp < 0:
        return True, (f"end to end {cand['detected_photos']} found / {cand['false_positive_photos']} false alarms against "
                      f"{cur['detected_photos']} / {cur['false_positive_photos']}")
    if cand_cal is not None and cur_cal is not None and cand_cal > cur_cal:
        return True, f"tied end to end; higher mean calibration IoU ({cand_cal:.3f} vs {cur_cal:.3f})"
    return False, "tied end to end and not better on calibration IoU"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--per-folder", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    from models.unet_segmenter import META_NAME, ONNX_NAME, SELECTION_NAME, UNetSegmenter
    from pipeline.deep_inference_pipeline import DeepInferencePipeline
    from scripts.measure_detection_quality import CLEAN_FOLDERS, DEFECT_FOLDERS, sample
    from scripts.segmenter_deployment_check import run

    cand_dir = os.path.abspath(a.candidate)
    if cand_dir == os.path.abspath(CKPT):
        sys.exit("--candidate must be the candidate's own folder, not checkpoints/")
    cand = UNetSegmenter(cand_dir)
    cur = UNetSegmenter(CKPT)
    if not cand.is_ready:
        sys.exit(f"candidate did not load from {cand_dir}: {cand.load_error or 'no ' + ONNX_NAME}")
    with open(os.path.join(cand_dir, META_NAME), encoding="utf-8") as fh:
        cand_meta = json.load(fh)
    cand_sel, cur_sel = {}, {}
    for d, target in ((cand_dir, cand_sel), (CKPT, cur_sel)):
        p = os.path.join(d, SELECTION_NAME)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                target.update(json.load(fh))

    pipe = DeepInferencePipeline(CKPT)
    clean, defect = sample(CLEAN_FOLDERS, a.per_folder, a.seed), sample(DEFECT_FOLDERS, a.per_folder, a.seed)
    print(f"[deep candidate] {cand_meta.get('model')} vs the current deep segmenter, "
          f"{len(clean)} clean / {len(defect)} defect photographs")
    res = {"candidate": run(pipe, cand, clean, defect, "candidate")}
    if cur.is_ready:
        res["current"] = run(pipe, cur, clean, defect, "current U-Net")
        ok, why = decide(res["candidate"], res["current"], cal_iou(cand_sel), cal_iou(cur_sel),
                         bool(cand_meta.get("onnx_verified_on_test")), bool(cand_meta.get("smoke")))
    else:
        res["current"] = {"note": "no deep segmenter in checkpoints/"}
        ok = bool(cand_meta.get("onnx_verified_on_test")) and not cand_meta.get("smoke")
        why = ("there was no deep segmenter to compare with" if ok else
               "smoke run, or the exported ONNX file did not reproduce the network")
    # Replacing the deep segmenter only matters if the candidate can then be served: its own IoU selection
    # against the pixel classifier (same rule as the U-Net's) must have chosen it.
    if ok and cand_sel.get("iou_selection_served") != "unet":
        ok, why = False, why + "; but it lost its own IoU selection against the pixel classifier, so it could not be served"
    record = {"candidate": cand_meta.get("model"), "candidate_dir": os.path.relpath(cand_dir, ROOT),
              "replaced_current": ok, "why": why, "results": res,
              "calibration_mean_iou": {"candidate": cal_iou(cand_sel), "current": cal_iou(cur_sel)},
              "checked_unix": int(time.time())}
    print(f"\n  {'REPLACES' if ok else 'does NOT replace'} the current deep segmenter: {why}")
    if ok:
        backup = os.path.join(CKPT, "segmenter_backup", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(backup, exist_ok=True)
        names = (ONNX_NAME, META_NAME, SELECTION_NAME)
        for name in names:                                     # everything this step changes, kept first
            if os.path.exists(os.path.join(CKPT, name)):
                shutil.copyfile(os.path.join(CKPT, name), os.path.join(backup, name))
        record["backup"] = os.path.relpath(backup, ROOT)
        try:
            for name in (ONNX_NAME, META_NAME):
                shutil.copyfile(os.path.join(cand_dir, name), os.path.join(CKPT, name + ".tmp"))
                os.replace(os.path.join(CKPT, name + ".tmp"), os.path.join(CKPT, name))
            new_sel = dict(cand_sel)
            new_sel["deep_candidate_check"] = record
            new_sel["previous_deep_segmenter"] = {k: cur_sel.get(k) for k in ("served", "why", "test") if k in cur_sel}
            with open(os.path.join(CKPT, SELECTION_NAME), "w", encoding="utf-8") as fh:
                json.dump(new_sel, fh, indent=2, default=float)
            from scripts import segmenter_deployment_check
            segmenter_deployment_check.main(["--per-folder", str(a.per_folder), "--seed", str(a.seed)])
        except BaseException:
            for name in names:                                 # put the previous state back, then report
                b = os.path.join(backup, name)
                if os.path.exists(b):
                    shutil.copyfile(b, os.path.join(CKPT, name))
            print(f"  the switch failed; the previous segmenter files were restored from {backup}")
            raise
    elif cur_sel:
        cur_sel["deep_candidate_check"] = record
        with open(os.path.join(CKPT, SELECTION_NAME), "w", encoding="utf-8") as fh:
            json.dump(cur_sel, fh, indent=2, default=float)
    return record


if __name__ == "__main__":
    main()
