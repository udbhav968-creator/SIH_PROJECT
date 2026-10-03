# ROAD-SHIELD

**AI-assisted road-defect assessment using the public bus fleet as a mobile sensing network.**  
Smart India Hackathon 2026 · Problem statement **SIH26124** · Bharat Electronics Limited

**Live site:** [road-shield-ai-engine.vercel.app](https://road-shield-ai-engine.vercel.app) — every measured result and the claims registry. Photograph and video analysis runs on the engine (`python -m api.server` or Docker), because the model stack is larger than a serverless function allows.

A road photograph goes in. A classified, outlined, measured and costed repair order comes out, sealed so it cannot be quietly edited. Every number in this file is read from the report that measured it (`python -m scripts.build_readme`), and every report is produced by code in this repository.

---

## Headline results

| What | Result | Measured on |
|---|---|---|
| Road-condition classifier, 7 classes | **93.3%** accuracy, macro-F1 **0.828** | 630 held-out images from 567 photographs never seen in training, scored once |
| Same classifier on **Indian roads** | **92.2%** accuracy, macro-F1 **0.922** (was 33.4% before Indian data was added) | 1,200 crops from held-out RDD2022 India photographs (normal / crack / pothole), never trained on |
| Hand-crafted baseline (HOG + LBP + colour → SVM) | 84.9%, macro-F1 0.558 | the identical split — the number the deep model has to beat |
| Defect segmentation (pixel classifier) | crack IoU **0.231**, pothole IoU **0.144** | 500 held-out DNIT photographs |
| CNN semantic gate on the segmenter | pothole IoU 0.102 → 0.271; clean roads with a false blob 8/50 → 2/50 | photographs the classifier never saw |
| IMU shock classifier (RandomForest) | **87.2%** accuracy, macro-F1 0.729 | 164 held-out windows of real Indian-road drive logs, split by time block |

## What it does

```
photo ─► classify ─► segment ─► project to ground ─► depth interval ─► MoRTH cost ─► SHA-256 work order
          (CNN)    (mask, not     (per-vehicle          (estimate,       (range)         (tamper-evident)
                     a box)        calibration)          never a measurement)
   │
   ├─► people & vehicles (YOLOv8 COCO) ─► privacy blur of heads and number plates before sharing
   └─► fleet ledger: reports of the same defect class within 8 m merge into one defect (SQLite)
```

| Stage | How | Status |
|---|---|---|
| Classification | mobilenet_v3_large (ImageNet-pretrained), fine-tuned end to end on a Colab T4 GPU, ONNX Runtime on CPU, flip test-time augmentation | **measured**, 93.3% |
| Segmentation | Pixel classifier (gradient boosting on 11 per-pixel features), trained on the DNIT hand-drawn polygons; per-class thresholds tuned on a calibration split | **measured**, IoU above |
| Area | each mask pixel's own ground footprint, summed (inverse perspective mapping) | **measured** *if* the camera is calibrated — 30 cm of mount height moves area ~46% |
| Depth | IRC band placed by measured extent / cavity contrast | **estimate**, always an interval |
| Cost | MoRTH Section 500 bitumen tonnage (compaction factor 1.15) × rate | **range**, because depth is a range |
| PCI | ASTM D6433 deduct-value procedure | formula; reproduces the deduct curves, not field-validated |
| Deterioration | HDM-4-style growth model, 30/60/90/180 days | formula, not field-validated |
| Sensor fusion | Bayesian log-odds gate over vision + IMU | reports *unavailable* when no IMU window is sent |
| Work orders | SHA-256 seal over the order fields; no GPS → `HELD_NO_GPS`, never an invented location | verified by tests |
| Repair verification | SSIM + sharpness + perceptual hash | catches a resubmitted old photograph |
| Privacy | heads (from person boxes), number plates and faces blurred | implemented; **recall not measured** |
| GIS | Google Maps if a key is set, else OpenStreetMap / OSRM / Open-Meteo | reports UNAVAILABLE rather than inventing data |

Measured on one photograph through the API: with an assumed camera mount the defect is 0.007 m² and ₹4.5; with the vehicle's calibrated profile it is 0.049 m² and ₹34.5 — seven times. That is why every response carries the provenance of each number in it.

## The models

| Model | Kind | Trained here? | Result | Serving |
|---|---|---|---|---|
| efficientnet_b0 | CNN, ImageNet-pretrained, fine-tuned end to end | yes (GPU) | 91.3%, F1 0.832; India 90.9% | no (lost on validation) |
| efficientnet_b2 | CNN, ImageNet-pretrained, fine-tuned end to end | yes (GPU) | 91.4%, F1 0.828; India 91.7% | no (lost on validation) |
| mobilenet_v3_large | CNN, ImageNet-pretrained, fine-tuned end to end | yes (GPU) | 91.4%, F1 0.840; India 90.8% | **yes** |
| resnet50 | CNN, ImageNet-pretrained, fine-tuned end to end | yes (GPU) | 89.5%, F1 0.857; India 91.9% | no (lost on validation) |
| MobileNetV2 + ensemble_soft | frozen CNN features + trained head | head only | 87.9%, F1 0.745 | fallback |
| HOG/LBP + PCA + SVM | classical features | yes | 84.9% | fallback |
| Pixel segmenter | gradient boosting on 11 features | yes | crack 0.231, pothole 0.144 | **yes** |
| YOLOv8n (COCO) | pretrained object detector | **no** — used as published | people, vehicles, signs | yes |
| IMU 1-D CNN | deep, from scratch | yes (GPU) | 78.0% | no (lost in cross-validation) |
| IMU RandomForest | classical, from scratch | yes | 87.2% | **yes** |

**How a model gets served.** Each deep model replaces its classical counterpart only by a rule written before its test set is scored: the fine-tuned CNN must beat the frozen head on validation accuracy *and* macro-F1; the U-Net must beat the pixel classifier on crack *and* pothole IoU on calibration photographs without more false blobs on clean roads; the IMU CNN must win 5-fold cross-validation on accuracy *and* macro-F1. Losers are reported, not hidden. Pretrained ImageNet/COCO weights are the starting point (transfer learning); training then updates every layer on this project's data, except the frozen-head baseline.

### Per class (served classifier)

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Normal Road / Sound Pavement | 0.97 | 0.97 | 0.97 | 148 |
| Crack (Longitudinal / Transverse / Alligator) | 0.92 | 0.94 | 0.93 | 218 |
| Pothole Cavity | 0.94 | 0.94 | 0.94 | 234 |
| Waterlogging / Flooding Hazard | 0.80 | 0.57 | 0.67 | 7 |
| Missing Zebra Crossing | 0.67 | 0.75 | 0.71 | 8 |
| Missing Road Divider | 0.71 | 0.62 | 0.67 | 8 |
| Damaged Traffic Sign | 1.00 | 0.86 | 0.92 | 7 |

Normal road, crack and pothole carry almost all test images. The four rare classes have seven or eight test images each, so one photograph moves their F1 by more than 0.1. That is a data-volume problem and it is shown, not averaged away.

## Data

| Source | Classes | Kept | Note |
|---|---|---|---|
| DNIT *Cracks and Potholes in Road Images* (Brazil) | crack, pothole, normal | 1,667 crops | 2,235 photographs, 4,720 hand-drawn polygons — the only source with outlines, so the segmenter trains on it |
| RDD2022 India (CRDDC 2022, smartphone) | crack, pothole, normal | 3,000 crops | from 1,785 training photographs; the Indian test set is 1,200 crops from 781 held-out photographs (split by photograph — the official test split has no public labels) |
| Kaggle pothole sets (3) | pothole, normal | 1,287 | one set was 94% a re-upload: 700 of 739 rejected as perceptual duplicates |
| Kaggle surface cracks (concrete walls) | — | 0 of 2,376 | **excluded**: close-ups of plaster with no road and no horizon (`pipeline/corpus_policy.py`); kept on disk, reproducible with `ROAD_SHIELD_NO_CORPUS_FILTER=1` |
| Wikimedia Commons / Geograph + field photographs | waterlogging, zebra, divider, sign | 207 | ~50 photographs per rare class — still the binding constraint |
| IMU drive logs (`VishalSingh25/Pothole-Project`) | 4 shock classes | 852 windows | 10 real drives on Indian roads, 205,491 samples at 100 Hz, from a car — not a bus |

## How the numbers are kept honest

- **Split by source photograph**, never by file: an augmented copy can never sit on the other side of the split from its original.
- **Model choice before the test set**: heads, architectures and thresholds are chosen on training or validation data; each test set is scored once.
- **Near-duplicate rejection**: every image is perceptually hashed; within 8 bits of an existing image is a re-upload (threshold measured over 60 photographs, not guessed).
- **Label-conflict audit**: 20 photographs were filed under three contradictory labels at once — the bug that held accuracy at 36.6%.
- **Domain policy**: data that is not a road scene is excluded from road-scene training and measurement, and the exclusion travels with every number it changes.
- **Claims registry**: `checkpoints/claims.json` lists 19 subsystems with their evidence files and 15 earlier claims that were withdrawn or corrected; the site's Architecture page shows both.
- **Nothing invented at runtime**: no default GPS, no default PCI, no confidence where there is no probability; missing inputs produce a 400 or an explicit `unavailable`.

## System design and platform

The full design — requirements, capacity maths for a 5,000-bus fleet, architecture from the bus edge to the ledger, data model, ML lifecycle, deployment, scaling, security and privacy — is in [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) and on the site's `/design` page, with an interactive capacity calculator. Every component there is marked *implemented* or *designed*.

| Platform feature | Where |
|---|---|
| OpenAPI 3 contract for the client-facing endpoints | `GET /api/v1/openapi.json`, rendered at `/api-docs` |
| Model registry: every model's artefact SHA-256, held-out metrics, serving status and deciding rule | `GET /api/v1/models/served`, shown on `/design` |
| Request IDs on every response; one JSON access-log line per request | `X-Request-ID`; `ROAD_SHIELD_ACCESS_LOG=1` |
| API key on state-changing endpoints (constant-time compare), off unless configured | `ROAD_SHIELD_API_KEY` |
| Container with a health check that fails when models did not load, plus a persistent ledger volume | `Dockerfile`, `docker-compose.yml` |
| CI on every push (Python 3.11 and 3.12, module imports, test suite) | `.github/workflows/ci.yml` |
| Claims registry and this README generated from the measured reports | `scripts/build_claims.py`, `scripts/build_readme.py` |

## Run it

```bash
pip install -r requirements.txt
python -m api.server                      # http://127.0.0.1:8000/
python -m unittest discover -s tests -t . # the test suite
```

Docker:

```bash
docker compose up --build                 # engine + persistent ledger volume, http://localhost:8000
```

No GPU and no PyTorch at inference — every network runs on ONNX Runtime on the CPU. scikit-learn is pinned (`>=1.8,<1.9`) because pickled estimators are not portable across minor versions.

## Train it

Deep training runs on a free Google Colab T4 GPU and resumes after a disconnect when Google Drive is mounted:

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone --depth 1 -b audit-2026-10-03 https://github.com/udbhav968-creator/SIH_PROJECT.git
%cd SIH_PROJECT
!mkdir -p logs && bash scripts/colab_train_all.sh 2>&1 | tee -a logs/colab_run.txt     # main run
!bash scripts/colab_train_extra.sh 2>&1 | tee -a logs/colab_extra.txt                 # U-Net + YOLOv8
```

The main run fetches DNIT and RDD2022 India, retrains the frozen head, fine-tunes EfficientNet-B0/B2, MobileNetV3-Large and ResNet-50, compares the IMU 1-D CNN with the RandomForest, runs the benchmark and the tests, and rebuilds the claims and the report. The extra run trains the U-Net segmenter and the YOLOv8 road-damage detector (it also runs on Kaggle). `scripts/apply_colab_outputs.ps1` copies the results back into the repository; `python -m scripts.build_readme` then refreshes this file.

Individual trainers: `training/train_cnn_head.py`, `training/train_finetune_cnn.py`, `training/train_unet_segmenter.py`, `training/train_rdd_detector.py`, `training/train_imu_deep.py`, `training/train_segmenter.py`.

## The site

| Page | For |
|---|---|
| `/` | overview, headline numbers and the honest limits |
| `/inspect` | analyse a photograph: class, mask, area, depth interval, cost range, privacy blur |
| `/video` | dashcam ingest, frames sampled by ground distance |
| `/corridor` | fleet map and the deduplication ledger |
| `/works` | costing and the SHA-256 tamper demonstration |
| `/models` | model card, per class, deep-vs-classical comparisons, including what scores badly |
| `/data` | dataset lineage, duplicate rejection, what the data does not cover |
| `/system` | components, calibration, storage, live endpoint self-test |
| `/architecture` | the claims registry and every withdrawn claim |
| `/design` | system design: architecture, capacity calculator, model registry, data model |
| `/api-docs` | the API reference, rendered from the OpenAPI document |

Main API endpoints (full reference at `/api-docs`): `POST /api/v1/pipeline/deep-audit` (full analysis of a photograph), `POST /api/v1/dispatch/work-order` and `/api/v1/dispatch/verify-seal`, `POST /api/v1/fleet/report-defect`, `POST /api/v1/privacy/redact`, `POST /api/v1/video/ingest`, `GET /api/v1/training/metrics`, `GET /api/v1/models/served`, `GET /api/v1/claims`, `GET /api/v1/health`.

## Repository layout

| Folder | Contents |
|---|---|
| `api/` | REST server (`server.py`) and the static Vercel entry point (`vercel_app.py`) |
| `models/` | classifiers, segmenters, detectors, geometry, depth, PCI, costing, sealing, privacy |
| `pipeline/` | the end-to-end audit pipeline, fleet deduplication, video ingest, corpus policy |
| `training/` | every trainer; each writes its model and a JSON report to `checkpoints/` |
| `scripts/` | data fetchers, Colab runners, claims/README/report builders |
| `checkpoints/` | trained models (ONNX / joblib) and the reports every number comes from |
| `web/` | the site |
| `docs/` | system design and the claims-vs-repository audit |
| `tests/` | 201 tests: models, pipeline, REST API, integrity fixes, Vercel entry point |

## Limitations

- **No bus data yet.** Photographs are DNIT (Brazil), Kaggle and RDD2022 India smartphone images; IMU logs are from a car. A bus-mounted pilot is the next step.
- **Four rare classes** (waterlogging, missing zebra crossing, missing divider, damaged sign) have ~50 photographs each, and their scores are unstable.
- **The Indian test covers three classes** (normal, crack, pothole), not seven.
- **Segmentation** (crack IoU 0.231, pothole IoU 0.144) is measured on DNIT photographs, not Indian or bus-camera frames; area and cost inherit its error.
- **Depth is an estimate**; PCI and deterioration reproduce engineering formulas and are not validated against field surveys.
- **Privacy blur recall is not measured** — there is no annotated face/plate set.
- **The live site serves results, not inference**: photograph and video analysis needs the engine.

## How accuracy got here

| Stage | Held-out accuracy | What changed |
|---|---|---|
| Inherited code | 36.6% | models untrained; some outputs fabricated |
| Label conflicts fixed | 79.4% | 20 photographs under three contradictory labels |
| Real dataset (DNIT) | 83.4% | 133 → 1,800 distinct photographs |
| ImageNet CNN embeddings | 88.5% | a pretrained CNN replaced hand-written features |
| Kaggle ingest + a labelling fix | 89.2% | removing 299 mislabelled sign images *raised* accuracy |
| …but on Indian roads | 33.4% | the same model on RDD2022 India crops |
| Concrete patches excluded, RDD2022 India added, fine-tuned CNN | **93.3%** (India **92.2%**) | larger, harder test set with Indian photographs in it |

An earlier README described models this code did not contain ("90.36% validation accuracy", "R² = 0.9908", a "10/10 PASS guaranteed" harness, simulated IMU data presented as real). It was removed in the October 2026 audit; see `CHANGES.md` and the corrections on the Architecture page.

---

**Team** Road Shield AI · **Developer** Udbhav Yadav · SIH26124 (BEL) · Theme: Smart Automation  
Developed for Smart India Hackathon 2026, academic and research use.

<sub>Generated by `scripts/build_readme.py` from the files in `checkpoints/`.</sub>
