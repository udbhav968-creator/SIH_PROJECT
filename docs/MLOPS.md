# MLOps: the models after training

ROAD-SHIELD serves about ten trained models. This document covers what happens to them after a
training script finishes: how a version is recorded, judged, deployed, watched in production, and
improved with the photographs the models found hardest.

```
             ┌───────────── training (laptop CPU / Colab GPU) ─────────────┐
             │ training/*.py ──► checkpoints/*  +  run in mlops/tracking  │
             └──────────────────────────────┬──────────────────────────────┘
                                            ▼
   python -m mlops retrain|register ──► registry (mlops_store/): version, SHA-256 files, metrics, run, commit
                                            │ gate (floor, max drop vs production, guard metrics)
                               candidate ──►│──► staging ──(shadow on live traffic)──► promote ──► checkpoints/
                                            │                                                  │ reload in place
                                            ▼                                                  ▼
                                    rollback ◄── events log                       api/server.py serves
                                                                                               │
   ┌───────────────────────── production ───────────────────────────────────────────────────┐ │
   │ every photo: input guard (OOD) ► models ► monitor (drift PSI + χ², OOD rate, latency)    │◄┘
   │                                         ► active-learning queue (uncertain, redacted)    │
   │ bus "trf" events + video ► traffic estimate ► Priority Index                              │
   │ /metrics (Prometheus) · /mlops page · alerts on drift                                     │
   └──────────────────────────────────────────────────────────────────────────────────────────┘
                     labelled queue ──► export zip ──► next training run
```

## 1. Experiment tracking (`mlops/tracking.py`)

Every run records parameters, metrics (with steps, for curves), tags, the git commit (`+dirty` if
the tree had changes), and the SHA-256 and size of each file it wrote. A run that raises is kept
as FAILED with the error. The store is SQLite (`mlops_store/runs.db`) and needs nothing installed.
With `pip install mlflow` every run is also mirrored to MLflow (`MLFLOW_TRACKING_URI`, default
`./mlruns`; `mlflow ui` to browse; `docker compose --profile mlflow up` starts a server).

`training/train_ood_guard.py` and the retraining pipeline log through it; `python -m mlops runs`
lists runs.

## 2. Model registry (`mlops/registry.py`, `mlops/specs.py`)

A **version** is the set of files the model's spec names, by SHA-256, copied into a
content-addressed blob store (`mlops_store/blobs/`). Registering identical bytes twice returns the
same version. Each version carries the metrics read from its own report files, the run that made it,
and the commit.

| Stage | Meaning |
|---|---|
| candidate | registered, not judged |
| staging | passed its gate; runs in shadow beside production |
| production | the files in `checkpoints/` are this version (one per model) |
| archived | was production; `rollback` returns to it |

**Gate** (per model, in `specs.py`): the primary metric must reach its floor and may not fall more
than `max_drop` below production; guard metrics have their own limits. Examples: the classifier's
held-out macro-F1 ≥ 0.75 and no more than 0.01 below production, with Indian-roads F1 and CPU latency
as guards; the input guard's AUROC ≥ 0.9 with at most 3% of genuine road photos refused.

**Promotion** writes each file to a temporary name and renames it over the old one, removes files
the new version does not have, archives the previous version, logs the event (with the gate
result, and `forced` if an operator overrode a failed gate), and the server reloads its models in
place. If `checkpoints/` holds files that are not the production version (copied in by hand),
promotion refuses until they are registered, so no unregistered model is ever overwritten.

On first start the server registers whatever is in `checkpoints/` as version 1 of each model.

```
python -m mlops status                       # every model: version, metrics, files match?
python -m mlops register vision_classifier   # after copying Colab outputs into checkpoints/
python -m mlops gate vision_classifier 2
python -m mlops promote vision_classifier 2
python -m mlops rollback vision_classifier
python -m mlops events
```

## 3. Retraining pipeline (`mlops/retrain.py`)

`python -m mlops retrain <model> [--promote]` for models that train on a CPU (IMU classifier, PCI,
depth, deterioration, input guard):

1. **validate data**: files per folder and a digest of which files; empty folders fail the run
2. **snapshot** every checkpoint file into the blob store
3. **train**: the model's command; output saved to the run log
4. **register** the files of this model that changed as a candidate
5. **restore** every changed file, including other models' files the script touched
   (`train_civil_models.py` also rewrites the segmenter), so production keeps serving
6. **gate**: pass → staging (→ production with `--promote`); fail → stays a candidate, with reasons

GPU models (CNN, U-Net, YOLO) train in Colab; copy the outputs back and `register`, then `gate` and
`promote`.

## 4. Input guard: the out-of-distribution model (`models/ood_guard.py`)

Runs before every analysis. Three parts:

- **novelty**: Mahalanobis distance of the MobileNetV2 embedding (PCA to 64 dims, Ledoit–Wolf
  covariance) from the training road photographs. Needs no examples of "not a road".
- **not-road classifier**: logistic regression on the same 64 dims, trained with road photographs
  against the ImageNet sample set (one photo per class, ~58 road-like classes removed).
- **quality**: brightness, contrast, sharpness (Laplacian variance), clipped highlights, with limits
  from the 0.5% tails of the training photographs.

Measured on held-out photographs (`checkpoints/ood_guard_report.json`):

| | |
|---|---|
| AUROC, road vs everyday photographs | 0.9945 |
| novelty alone (never saw a non-road photo) | 0.984 AUROC |
| everyday photographs refused | 94.2% |
| genuine road photographs refused | 1.25% (4 of 321) |
| genuine road photographs warned "unusual" | 3.7% |
| synthetic darkening / blur / overexposure caught | 100% / 100% / 100% |
| crack close-ups (20 cm texture patches) flagged | 49% |

The citizen Report page refuses "not a road" and "too dark/blurred" (HTTP 422, with the reason);
operator uploads get the verdict as a warning. The guard is stored as plain NumPy arrays, so it
loads under any scikit-learn version.

Retrain: `git clone --depth 1 https://github.com/EliSchwartz/imagenet-sample-images ../imagenet-sample-images`
then `python -m mlops retrain ood_guard --ood-dir ../imagenet-sample-images`. The same script writes
`checkpoints/monitoring_reference.json`.

## 5. Production monitoring (`mlops/monitor.py`)

Each analysed photograph adds a row: source (api / citizen / video / fleet), the model versions
that served it, the classifier's top class and confidence, the guard's verdict and novelty score,
four image-quality measures and latency. No image, no location. Bus detections add class and
confidence.

**Drift** = PSI of each measure in a window against the reference histogram. PSI < 0.10 stable,
0.10–0.25 watch, > 0.25 drift, but only when a chi-square test also says the shift is unlikely to
be chance (p < 0.01). On small windows PSI is mostly noise (about (bins−1)/n), so the test matters.
Class-mix drift uses the same rule. A window under 30 photographs is "not enough data".

**Reference**: built from held-out training photographs. A city's own cameras will differ from
public datasets on day one, so after the first weeks of fleet data an operator sets the reference
from production ("use the last 7 days as the reference" on the MLOps page, or
`POST /api/v1/mlops/rebaseline`); `{"reset": true}` goes back to the training reference.

**Alerts**: when the 24-hour status turns to drift, a `model_drift` event goes to the live feed and,
with `ROAD_SHIELD_ALERT_WEBHOOK`, to Slack/Teams/ticketing.

**Prometheus** (`/metrics`): `road_shield_model_psi{measure=…}`, `road_shield_model_drift`,
`road_shield_input_ood_ratio`, `road_shield_model_latency_ms{quantile=…}`,
`road_shield_model_predictions`.

**Shadow**: a classifier version in staging is loaded beside production and run on the same
photographs in a background thread (a bounded queue; when it is full, the shadow skips rather than
slowing a request). Agreement, the commonest disagreements and its p95 latency show on the MLOps
page before anyone promotes it.

## 6. Active learning (`mlops/active_learning.py`)

Each analysed photograph is scored:

```
score = 0.5·entropy + 0.2·(1 − top-two margin) + 0.3·disagreement + 0.2·unusual
```

**Disagreement** means the classifier and the measured regions disagree: the classifier is confident
of a pothole/crack and no region was measured, a region was measured where the classifier is confident
the road is normal, or the YOLO damage detector boxed something the pipeline did not report.

Above 0.35 the photograph is kept: people and number plates blurred first (no redaction, no storage),
near-duplicates dropped by cosine similarity of the guard's 64-d embedding, at most 500, lowest score
evicted. **Citizen photographs are never kept** (the Report page promises that).

Operators label on the MLOps page; "Download labelled set" gives a zip, one folder per class plus a
manifest, to add to the training corpus. The page also shows how often the classifier agreed with
the labellers, a running estimate of accuracy on hard cases.

## 7. Traffic estimate (`pipeline/traffic.py`)

The Priority Index's traffic term used to be the number of buses that reported a defect. Now each
bus counts vehicles with the COCO detector, averages them per 100 m road cell, and sends a sealed
`trf` event when it leaves the cell:

```
density k = PCU in view / 40 m      flow q = k · bus speed      daily = q / share of the day's traffic in that hour
```

Frames with the bus below 8 km/h (stops, signals) count for density, not flow. The hour profile is
an assumed Indian urban arterial curve. A cell feeds the Priority Index (`traffic_basis:
measured_pcu_per_day`) once it has 6 observations in 3 different hours; until then the bus-count
basis stays. Uploaded video with a GPS track adds observations too. The road map has a "Traffic from
bus cameras" layer.

**Before relying on it**: run classified volume counts (IRC:SP:41) on a few corridors, compare, and
set the view length and hour profile from them.

## 8. CI/CD

| Workflow | What it does |
|---|---|
| `ci.yml` | full test suite on 3.11 and 3.12 |
| `models.yml` | `python -m mlops.ci_check`: every committed model meets its floor; MLOps tests |
| `docker.yml` | builds the image, starts it, waits for `/api/v1/ready`, analyses a photo, checks `/api/v1/mlops/overview`; on master pushes `ghcr.io/<owner>/road-shield:latest` and `:<commit>` |

## 9. API

`GET /api/v1/mlops/overview` · `drift?hours=&source=` · `versions?model=` · `gate?model=&version=` ·
`runs` · `run?id=` · `al/queue` 🔑 · `al/image?id=` 🔑 · `/api/v1/traffic/cells` · `traffic/estimate?lat=&lon=`

`POST /api/v1/mlops/promote` 🔑 · `rollback` 🔑 · `stage` 🔑 · `register` 🔑 · `reload` 🔑 ·
`rebaseline` 🔑 · `al/label` 🔑 · `al/export` 🔑

🔑 = needs the operator key when `ROAD_SHIELD_API_KEY` is set.

## What is not done

- No GPU retraining from the pipeline: Colab runs stay manual, then `register`.
- The traffic estimate is uncalibrated until compared with manual counts.
- Drift reference is public-dataset photos until a city rebaselines on its own fleet.
- Shadow testing covers the classifier only; the segmenter and detectors are judged offline.
- No canary split (a share of live traffic served by the new version): with one server, shadow +
  gate + instant rollback is the safer equivalent.
