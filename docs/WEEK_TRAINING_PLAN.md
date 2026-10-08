# One-week training plan

The goal is better outlines on Indian roads, especially water-filled potholes, without breaking the
evaluation protocol. The script is `scripts/colab_week_training.py`. It resumes from Google Drive, so the
training can be spread over as many Colab sessions as the free GPU quota allows.

## Why your own photographs matter most

Every model here learned outlines from Brazilian (DNIT), benchmark (CrackSeg9k) and mixed web photographs.
None of them shows a monsoon-filled Indian pothole the way a bus camera sees it. More epochs on the same data
gives small gains. A few hundred outlined Indian photographs teach exactly the missing case. They also give
the first Indian-road outline score this project can report, from the held-out test share of your own photos.

## Day 1–2: take and outline photographs (≈ 200–400)

**Taking the photographs**

- Use a phone held at chest height, looking ahead and down at the road, the way a bus camera would. Don't
  shoot straight down.
- Cover the variety you need:
  - dry potholes
  - water-filled potholes
  - cracks of all kinds
  - patched spots
  - shadows
  - some clean road
- Photograph different spots. If you take several photos of one spot, name them with the same prefix and a
  double underscore, for example `spot12__a.jpg` and `spot12__b.jpg`. The split keeps a spot in either train
  or test, never both.
  - Burst shots of one spot are grouped automatically, but naming them is safer.
- Avoid faces and number plates where you can.

**Outlining them.** Use any one of these tools:

| Tool | How |
|---|---|
| CVAT (cvat.ai, free, private) | New task → upload photos → labels `pothole`, `crack` → polygon tool → Export → **COCO 1.0** |
| Roboflow (free) | Instance Segmentation project → classes `pothole`, `crack` → Smart Polygon → Export → **COCO Segmentation**. The free plan may make the project public; use CVAT if that matters. |
| LabelMe (desktop, offline) | Open the folder → Create Polygons → labels `pothole`, `crack` → it saves a `.json` next to each photo |

Labelling rules:

- **Water-filled pothole:** outline the whole pothole, water included. This is the exact case the current
  model gets wrong.
- **Crack:** a thin polygon along the crack. In LabelMe, a line strip also works.
- Class names only need to *contain* `pothole` or `crack` ("water pothole" and "alligator crack" both work).
  Other classes are ignored.
- Rough time: 1–2 minutes per photograph. 200 photographs is an afternoon or two.

Upload the export (zip or folder) to Google Drive, for example `MyDrive/road_photos_coco.zip`.

## Day 3 onwards: train in Colab

**1. Push the code first.** The cell checks that GitHub has it.

**2. Paste `scripts/colab_week_training.py` into one Colab cell** (Runtime → T4 GPU), then set:

```python
OWN_PHOTOS = "/content/drive/MyDrive/road_photos_coco.zip"
POTHOLE_MIX_DRIVE = "/content/drive/MyDrive/pothole-mix-v1.0-20220526.zip"
```

**3. Run it.** Allow Drive access when it asks. The first session:

- prepares the datasets and caches them in Drive (~30–45 min, once);
- runs a 2-minute smoke run that also proves resuming works;
- starts the U-Net.

**4. When the session ends**, either because it says `PAUSED` or because Colab disconnects, open a new
session and run the **same cell** again. It continues from the last checkpoint. Repeat until it prints
`ALL TRAINING FINISHED` and downloads `road_shield_run1_results.zip`.

**Rough GPU time (estimates):**

| Step | Time | Note |
|---|---|---|
| U-Net | 3–5 h | ResNet-34, up to 80 epochs; early stopping usually ends it sooner |
| YOLOv8s, India | 2.5–4 h | Up to 100 epochs |
| Multi-country data | ~1 h, once | Downloads the full RDD2022 release (~13 GB) and prepares Japan, Czech, USA and China; ~4–5 GB cached in Drive |
| YOLOv8s, multi-country | 5–7 h | About 4× the photographs, up to 50 epochs; resumes across sessions |
| Image classifier | 3–4 h | Three architectures, up to 40 epochs each; every finished one is kept in Drive |

That is 5–8 sessions in total. To skip a stage, set `TRAIN_YOLO_WORLD = False` or `TRAIN_CLASSIFIER = False`.

**What the two new stages add, and when they are served.** Each newly trained model is a *candidate*. It
replaces the served one only under a rule fixed before it is scored:

- **Multi-country detector** (`scripts/select_rdd_detector.py`): served only if its mAP@0.5 on India's
  validation photographs is higher than the India-only detector's on the same photographs. India's
  train/validation/test split is copied unchanged, so the comparison is like for like. India's test
  photographs are reported and never used to choose. If it wins but its exported file fails the artefact
  check, the India-only detector stays.
- **Image classifier** (`scripts/select_vision_candidate.py`): the Indian held-out crops are split by
  photograph into a *selection* half and a *report* half. The candidate is served only if its accuracy and
  its macro-F1 are both higher on the selection half. The number published for it comes from the report
  half.

If a candidate loses, nothing changes. Both outcomes are recorded (`rdd_detector_selection.json`,
`vision_candidate_selection.json`).

**Not retrained:** the IMU model. No public dataset matches its sensor setup, so more data would have to
come from your own recorded rides.

**If you get stuck:**

- **"GPU usage limit":** come back later (usually within a day). The checkpoints wait in Drive.
- **Drive full:** empty the Drive trash. Checkpoints are rewritten every 30 minutes.
- **New run:** to train again with different photos or settings, change `RUN_NAME`. The old run stays on
  record. A run that has started never changes its data. If the data differs, it refuses to resume and
  says why.

## After training: on the laptop, once

```powershell
cd C:\Users\Dell\SIH_PROJECT
git pull origin audit-2026-10-03
Expand-Archive "$env:USERPROFILE\Downloads\road_shield_run1_results.zip" -DestinationPath . -Force
python -m scripts.segmenter_deployment_check
python -m scripts.verify_rdd_detector --clean-roads
python scripts/build_claims.py
python scripts/build_readme.py
python scripts/build_model_card.py
python -m unittest discover -s tests -t . 2>&1 | Out-File -Encoding utf8 logs/tests_after.txt
Select-String -Path logs/tests_after.txt -Pattern "^Ran |^OK|^FAILED"
```

Then commit and push as usual.

## Rules that keep the result honest

- **Run the end-to-end check once, on the final model.** If you train five variants and keep whichever passes
  `segmenter_deployment_check`, the check has become part of training and no longer measures anything. Pick
  the run you will submit *before* running the check, and report it whatever the result.
- **Thresholds come from calibration photographs only.** Your photographs join them as a road-scene source.
- **The DNIT split, the selection rule and the end-to-end gate are unchanged.** Earlier runs stay on record in
  `segmenter_selection.json` (`previous_run`).
- **Report the score on your own photographs with its size.** It comes from about 15% of your photographs,
  so 300 photographs give a test set of about 45.
