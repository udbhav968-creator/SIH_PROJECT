# Training & tests — what was actually run

This records real results, and is honest about what this environment could and
could not do. No GPU was available, `torch` was not installed, and the large
labelled detection datasets (RDD2022-India) could not be downloaded here, so the
deep detectors were **not** retrained; only models that train on CPU from in-repo
data were trained and measured.

## Real CPU training that was run here

**IMU shock classifier** (`training/train_imu.py`, `StandardScaler → RandomForest`).
Trained on the repository's real 100 Hz tri-axial accelerometer windows
(`datasets/04_mobile_imu_telemetry_100hz/`), evaluated on the held-out split that
is separated from training by a temporal block boundary.

| Metric | Result |
|---|---|
| Train / held-out windows | 688 / 164 |
| Held-out accuracy | **86.6%** (4 classes; 25% random baseline) |
| Pothole Impact | P 0.80 · R 1.00 · F1 0.89 |
| Smooth Asphalt | P 0.94 · R 0.90 · F1 0.92 |
| Unmarked Speed Breaker | P 0.90 · R 0.81 · F1 0.85 |
| Marked Speed Breaker | P 0.38 · R 0.55 · F1 0.44 *(few samples — honestly weak)* |

Trained in ~1.6 s on CPU; reproduces the repository's reported 86.6%. The shipped
checkpoint was left untouched (the run saved to a temp path). Full numbers in
`imu_retrain_report.json`.

## Real testing over the datasets

The detect-everything pipeline was run over **136 sampled images** across all 13
dataset folders. Because the detection sets have no bounding-box ground truth in
the repo, this measures **coverage** (what the detectors fire on), not accuracy:
13 object classes detected; 206 road-damage, 118 car, 44 person, 24 crack boxes,
plus zebra crossings, traffic lights, buses, trucks. It also surfaces honest
failure modes — the candidate damage detector over-fires on waterlogged scenes.
Full breakdown in `dataset_eval.json`.

## Automated test suite

`python -m unittest discover -s tests` → **404 tests pass** (1 skipped), including
7 new tests for the detect-everything pipeline (candidate tagging, counts match
detections, video de-duplication, and that the served gate stays closed).

## What is NOT trained here, and where it lives

The deep models (MobileNetV3 classifier, U-Net, the YOLOv8 road-damage detector)
need a GPU and the full datasets. Their training is in `notebooks/` (Colab) and
`training/` and `colab/`. To train and then verify the road-damage detector so its
boxes graduate from *candidate* to *verified*, follow `docs/DETECTION_PIPELINE.md`.

## Deep-learning extensions you asked about (GAN / adversarial)

Not added. Training a GAN or adversarial model to any real standard needs a GPU and
large data that are unavailable in this environment; shipping an untrained toy and
calling it a model would be exactly the kind of fabricated result this project's
integrity (see `checkpoints/claims.json`) was built to avoid. If you want to pursue
it, the honest home for it is a Colab notebook under `notebooks/`, scored by a rule
fixed before its test set and recorded in the claims registry — never served until
measured.
