# Detect-everything pipeline (image & video)

One entry point takes an **image or a video** and returns **every road object
detected, boxed and counted**: potholes, cracks, zebra crossings, road arrows,
people, cars, buses, trucks, motorcycles, bicycles, traffic lights and stop signs.

It does not add any new model. It fuses the three pretrained detectors already in
this repository, through `models/yolo_video_engine.py`:

| Detector | Weights | Classes |
|---|---|---|
| Road damage (RDD2022) | `checkpoints/damage_rdd2022_india.onnx` | Pothole, Alligator / Longitudinal / Transverse Crack |
| COCO scene & traffic | `checkpoints/road_shield_detector.onnx` | person, car, bus, truck, motorcycle, bicycle, traffic light, stop sign |
| Crosswalk & markings | `checkpoints/crosswalk_rdd.onnx` | Zebra Crosswalk, Road Guide Arrow |

For video, a lightweight IoU tracker (`YOLOMultiObjectTracker`) gives each object a
persistent id, so an object that appears in many frames is **counted once**.

## Use it

Python:

```python
from pipeline.detect_everything import analyze_image, analyze_video

# Image -> counts + boxes + an annotated PIL image
r = analyze_image("road.jpg")            # or a numpy RGB array, or base64
print(r["summary"])                      # "Detected 1 pothole, 1 crack, 4 persons, 1 bus."
print(r["counts"])                       # {"potholes": 1, "cracks": 1, "persons": 4, "buses": 1, "total": 7}
r["annotated"].save("road_boxed.png")    # the drawn image

# Video -> UNIQUE counts across the clip + a per-frame trace + sample frames
v = analyze_video("drive.mp4", sample_every_s=0.5)
print(v["unique_counts"])                # counted once per tracked object
```

HTTP (while `python -m api.server` is running):

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/detect/everything \
  -H 'Content-Type: application/json' \
  -d '{"image_base64": "<...>", "candidate_damage": true, "annotate": true}'
```

The response carries `counts`, `summary`, `detections`, `models_active`,
`damage_verification`, and (when `annotate`) `annotated_image_base64`.

## The honest part: `candidate` vs `verified` road-damage boxes

The COCO and crosswalk detectors are pretrained and used as published, so their
boxes are shown normally.

The **road-damage detector has not passed this repository's verification gate**
(`scripts/verify_rdd_detector.py`). Its exported ONNX scored mAP@0.5 **0.0211**
above the trained network — just over the pre-registered **0.02** budget — and the
clean-road check was never run. So on the trustworthy ("served") path the detector
reports `is_ready == False` and its boxes stay hidden. That gate is deliberate and
is **left intact**.

This pipeline can still show those boxes, but only as an **unverified candidate**:

- `candidate_damage=True` (the default here) runs the detector and labels every
  road-damage box with `verification_status: "candidate_unverified"` and the name
  of the outstanding check (`gate_blocked_by`).
- In the annotated image, candidate boxes are drawn **dashed with a `?`** so they
  never read as confirmed.
- `damage_verification` in the result is `"candidate_unverified"` (or `"hidden"`
  when `candidate_damage=False`, or `"verified"` once the gate passes).

Nothing here flips the gate or claims validation. `candidate_damage=False`
reproduces the original, fully-gated behaviour.

## How to graduate the road-damage detector to `verified`

This needs the labelled RDD2022-India detection set and the trained weights, which
are **not** committed to the repo (they are large, and RDD2022 is distributed from
figshare). On a machine with that data and `torch`/`ultralytics` installed:

1. Place the training output at `runs/rdd_india/weights/best.pt` and the test split
   under `datasets/rdd2022_india/` (images + YOLO labels, `test` split), matching
   the paths in `scripts/verify_rdd_detector.py`.
2. Run the artefact check — does the served ONNX behave like the trained network?
   ```bash
   python -m scripts.verify_rdd_detector --artefact
   ```
   If it still fails by a hair, the most likely cause is that the two mAP numbers
   were produced by **different scorers** (Ultralytics' validator for `best.pt`
   vs. this repo's decoder for the ONNX). Score **both** through the same harness
   on the same images before trusting the gap, and re-export with a pinned opset
   and fixed input size if the discrepancy is in the export.
3. Run the clean-road check — does it draw damage on roads that have none?
   ```bash
   python -m scripts.verify_rdd_detector --clean-roads
   ```
4. Both checks write their result into `checkpoints/damage_rdd2022_india.json`.
   Once both record `"passed": true`, `RoadDamageDetector.is_ready` becomes `True`
   on its own, the boxes are served as `verified`, and `candidate_damage` is no
   longer needed to show them.

## What is and isn't proven

- COCO and crosswalk detections: pretrained models used as published.
- Road-damage detections: a candidate detector; **treat as indicative, not
  validated**, until the two checks above pass.
- Counts are exact counts of the boxes the detectors drew — they inherit each
  detector's own precision/recall, which is not perfect (expect occasional false
  positives, especially on bare-pavement close-ups).
