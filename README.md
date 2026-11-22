# ROAD-SHIELD

**AI-assisted road-defect assessment using the public bus fleet as a mobile sensing network.**  
Smart India Hackathon 2026 · Problem statement **SIH26124** · Bharat Electronics Limited

**Live site:** [road-shield-ai-engine.vercel.app](https://road-shield-ai-engine.vercel.app) — every measured result and the claims registry. Photograph and video analysis runs on the engine (`python -m api.server` or Docker), because the model stack is larger than a serverless function allows; `docs/DEPLOY_LIVE_ENGINE.md` connects a hosted engine to the site.

A road photograph goes in. A classified, outlined, measured and costed repair order comes out, sealed so it cannot be quietly edited. Every number in this file is read from the report that measured it (`python -m scripts.build_readme`), and every report is produced by code in this repository.

---

## Headline results

| What | Result | Measured on |
|---|---|---|
| Road-condition classifier, 7 classes | **94.4%** accuracy, macro-F1 **0.879** | 531 held-out images from 512 photographs never seen in training, scored once |
| Same classifier on **Indian roads** | **92.3%** accuracy, macro-F1 **0.922** (was 33.4% before Indian data was added) | 1,200 crops from held-out RDD2022 India photographs (normal / crack / pothole), never trained on |
| Hand-crafted baseline (HOG + LBP + colour → SVM) | 78.0%, macro-F1 0.651 | the identical split — the number the deep model has to beat |
| Defect segmentation (pixel classifier) | crack IoU **0.208**, pothole IoU **0.119** | 125 held-out DNIT photographs |
| CNN semantic gate on the segmenter | pothole IoU 0.102 → 0.271; clean roads with a false blob 8/50 → 2/50 | photographs the classifier never saw |
| Road-damage detector (YOLOv8s, boxes), retrained | mAP@0.5 **0.400**, mAP@0.5:0.95 0.186 | 780 held-out photographs, duplicate-safe split. **Boxes hidden:** the exported ONNX file failed the artefact check |
| IMU shock classifier (RandomForest) | **86.6%** accuracy, macro-F1 0.776 | 164 held-out windows of real Indian-road drive logs, split by time block |

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
| Classification | mobilenet_v3_large (ImageNet-pretrained), fine-tuned end to end on a Colab T4 GPU, ONNX Runtime on CPU, flip test-time augmentation | **measured**, 94.4% |
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
| MobileNetV2 + mlp_512_256 | frozen CNN features + trained head | head only | 86.4%, F1 0.787 | fallback |
| HOG/LBP + PCA + SVM | classical features | yes | 78.0% | fallback |
| U-Net (ResNet-18 encoder) | deep segmenter, ImageNet encoder, all layers trained | yes (GPU) | crack 0.303, pothole 0.466 | no (won on masks, lost the end-to-end check; 18/24 end to end) |
| Pixel segmenter | gradient boosting on 11 features | yes | crack 0.208, pothole 0.119 | **yes** |
| YOLOv8s road-damage detector | yolov8s.pt (COCO-pretrained) on RDD2022 India + Kaggle pothole | yes (GPU) | mAP@0.5 0.400 | **no** — the exported ONNX file failed the artefact check; boxes hidden |
| YOLOv8n (COCO) | pretrained object detector | **no** — used as published | people, vehicles, signs | yes |
| IMU 1-D CNN | deep, from scratch | yes (GPU) | 79.9% | no (lost in cross-validation) |
| IMU RandomForest | classical, from scratch | yes | 86.6% | **yes** |


### Removed modules (formerly listed as the five advanced neural suites)

SurfaceTypeNet, Metric3D, V-IMU, PINN-HDM4 and the Forensic VLM were removed from the decision path of both engines. Their training labels or targets were produced by the same heuristics or formulas they were scored against, so the accuracies once reported for them (including the 92.4% fusion agreement) were circular and are withdrawn (see `checkpoints/claims.json`, corrections). The API keeps their keys with `status: not_modelled` and no invented numbers, and surface type is reported as `paved_unverified`. Pothole boxes still come from heuristic proposers plus the RDD2022 detector, whose pothole recall on 40 Kaggle photos was 0 at confidence 0.25; a detector retrained on real labelled photographs (RDD2022 India + Kaggle pothole) is installed, but its boxes are not shown: its exported file did not pass the pre-registered artefact check (see Model card).

### Field-Inspection Grounding & Integrity Gates

- **High-Confidence Crack Fissure Extraction**: Refined pixel segmentation proposal floor (`mean_p >= 0.65` and `area >= 18 px`) captures thin longitudinal and alligator cracking networks in the road body (e.g., CRACK500 fatigue cracking) On 120 clean-road photographs the pixel segmenter flagged a blob on 23 (19.2%), so it is not false-alarm free.
- **Carriageway-Bound Zebra Crossing Detection**: MoRTH Class 4 fallbacks bind to the carriageway bounding box detected by the geometric crosswalk analyzer (`detect_zebra`), eliminating background overlap on buildings or sky.
- **Geotechnical Survey & Pricing Guard**: Uncalibrated macro close-ups covering > 35% of the frame or > 2.0 m² are flagged for manual field survey (IRC:37/82) with ₹0 committed, preventing runaway patch price estimates on cropped photos.
- **Serverless and local paths**: `pipeline/deep_inference_pipeline.py` and `api/vercel_inference.py` are covered by the same test suite; this is a test result, not a proof that the two always agree.

**How a model gets served.** Each deep model replaces its classical counterpart only by a rule written before its test set is scored: the fine-tuned CNN must beat the frozen head on validation accuracy *and* macro-F1; the U-Net must beat the pixel classifier on crack *and* pothole IoU on calibration photographs without more false blobs on clean roads, and then hold up end to end - through the full pipeline on photographs from other datasets it must find at least as many defects with no more false alarms (`scripts/segmenter_deployment_check.py`; it did not: U-Net 18/24 defects found vs 20/24); the IMU CNN must win 5-fold cross-validation on accuracy *and* macro-F1. Losers are reported, not hidden. Pretrained ImageNet/COCO weights are the starting point (transfer learning); training then updates every layer on this project's data, except the frozen-head baseline.

### Per class (served classifier)

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Normal Road / Sound Pavement | 0.89 | 0.99 | 0.94 | 77 |
| Crack (Longitudinal / Transverse / Alligator) | 0.95 | 0.97 | 0.96 | 184 |
| Pothole Cavity | 0.96 | 0.96 | 0.96 | 211 |
| Waterlogging / Flooding Hazard | 0.88 | 0.50 | 0.64 | 14 |
| Missing Zebra Crossing | 1.00 | 0.80 | 0.89 | 15 |
| Missing Road Divider | 0.81 | 0.87 | 0.84 | 15 |
| Damaged Traffic Sign | 1.00 | 0.87 | 0.93 | 15 |

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
| RDD2022 India + Kaggle pothole boxes (detector) | D00 / D10 / D20 / D40 | 6,248 / 781 / 780 photographs (train / val / test) | duplicate-clustered split; 0 test photographs have a near-duplicate in train |

**Retrained detector: why its boxes are hidden.** The exported ONNX file scores mAP@0.5 0.422 on the test split against 0.401 for the trained network, a gap of 0.0211; the rule, fixed before the run, allows 0.02. The serving decoder itself passed its part of the check (94.4% recall and 98.7% precision against the trained network on 100 photographs). Scoring the trained network under the same settings gave 0.400, so the settings do not explain the gap. The cause has not been found and the rule was not loosened. The clean-road check has not been run (it runs only after the artefact check passes).

## Detect everything (image & video)

One entry point takes an **image or a video** and returns every road object detected, boxed and counted: potholes, cracks, zebra crossings, road arrows, people, cars, buses, trucks, motorcycles, bicycles, traffic lights and stop signs. It adds no new model — it fuses three pretrained detectors already in the repository (road damage, COCO scene/traffic, crosswalk) through `pipeline/detect_everything.py`, and for video a lightweight IoU tracker counts each object once across frames.

```python
from pipeline.detect_everything import analyze_image, analyze_video
analyze_image('road.jpg')['counts']   # {'potholes': 1, 'cracks': 1, 'persons': 4, 'buses': 1, 'total': 7}
analyze_video('drive.mp4')['unique_counts']
```

| Where | How |
|---|---|
| API | `POST /api/v1/detect/everything` (engine and Vercel serverless) → counts, boxes, annotated PNG |
| Web UI | `/detect` — drag-and-drop an image, live boxes and counts (served by the engine or Vercel) |
| Honesty | road-damage boxes are shown only as an **unverified candidate** (drawn dashed, `verification_status`); the served gate stays closed until `scripts/verify_rdd_detector.py` passes |

The COCO and crosswalk detectors are pretrained and used as published. The road-damage detector has **not** passed its verification gate, so by default its boxes stay hidden; `candidate_damage=True` shows them, always labelled unverified, never claimed as validated. See `docs/DETECTION_PIPELINE.md` for the full contract and the steps to graduate the detector to `verified`.

## How the numbers are kept honest

- **Split by source photograph**, never by file: an augmented copy can never sit on the other side of the split from its original.
- **Model choice before the test set**: heads, architectures and thresholds are chosen on training or validation data; each test set is scored once.
- **Near-duplicate rejection**: every image is perceptually hashed; within 8 bits of an existing image is a re-upload (threshold measured over 60 photographs, not guessed).
- **Label-conflict audit**: 20 photographs were filed under three contradictory labels at once — the bug that held accuracy at 36.6%.
- **Domain policy**: data that is not a road scene is excluded from road-scene training and measurement, and the exclusion travels with every number it changes.
- **Claims registry**: `checkpoints/claims.json` lists 26 subsystems with their evidence files and 20 earlier claims that were withdrawn or corrected; the site's Architecture page shows both.
- **Nothing invented at runtime**: no default GPS, no default PCI, no confidence where there is no probability; missing inputs produce a 400 or an explicit `unavailable`.

## System design and platform

The full design — requirements, capacity maths for a 5,000-bus fleet, architecture from the bus edge to the ledger, data model, ML lifecycle, deployment, scaling, security and privacy — is in [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) and on the site's `/design` page, with an interactive capacity calculator. Every component there is marked *implemented* or *designed*. Impact, cost per km, the Responsible AI mapping (Microsoft's six principles), the 90-day pilot and the Azure mapping are in [`docs/IMPACT_AND_RESPONSIBLE_AI.md`](docs/IMPACT_AND_RESPONSIBLE_AI.md) and on `/impact`.

| Platform feature | Where |
|---|---|
| OpenAPI 3 contract for the client-facing endpoints | `GET /api/v1/openapi.json`, rendered at `/api-docs` |
| Model registry: every model's artefact SHA-256, held-out metrics, serving status and deciding rule | `GET /api/v1/models/served`, shown on `/design` |
| Request IDs on every response; one JSON access-log line per request | `X-Request-ID`; `ROAD_SHIELD_ACCESS_LOG=1` |
| API key on state-changing endpoints (constant-time compare), off unless configured | `ROAD_SHIELD_API_KEY` |
| Container with a health check that fails when models did not load, plus a persistent ledger volume | `Dockerfile`, `docker-compose.yml` |
| CI on every push (Python 3.11 and 3.12, module imports, test suite) | `.github/workflows/ci.yml` |
| Prometheus metrics: requests by route and status, latency histogram, model readiness | `GET /metrics` |
| Readiness probe: 200 only when the classifier and segmenter are loaded (the container health check uses it) | `GET /api/v1/ready` |
| Request-size limit (413) and per-client rate limit on model endpoints (429 + Retry-After) | `ROAD_SHIELD_MAX_BODY_MB`, `ROAD_SHIELD_RATE_LIMIT` (`api/ops.py`) |
| Model card for every served model: intended use, data, held-out metrics, limits, ethics, corrections | [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), generated by `scripts/build_model_card.py` |
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
# multi-dataset U-Net and YOLOv8 in one cell: paste scripts/colab_train_all_datasets.py
```

The main run fetches DNIT and RDD2022 India, retrains the frozen head, fine-tunes EfficientNet-B0/B2, MobileNetV3-Large and ResNet-50, compares the IMU 1-D CNN with the RandomForest, runs the benchmark and the tests, and rebuilds the claims and the report. The extra run trains the U-Net segmenter and the YOLOv8 road-damage detector (it also runs on Kaggle). `scripts/colab_train_all_datasets.py` trains the U-Net on DNIT plus public outline datasets (CrackSeg9k, Kaggle pothole segmentation, Pothole Mix) after dropping any image that is a near-copy of a measurement photograph (`scripts/fetch_seg_datasets.py`). `scripts/apply_colab_outputs.ps1` copies the results back into the repository; `python -m scripts.build_readme` then refreshes this file.

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
| `/impact` | sourced road-safety statistics, cost per km, Responsible AI mapping, pilot plan, Azure mapping |

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
| `web/` | the site (`web/config.js` names the live engine it sends photographs to) |
| `deploy/` | the Hugging Face Space that hosts the public live engine |
| `docs/` | system design and the claims-vs-repository audit |
| `docs/project/` | the 10-page project report and the 10-slide deck (PDF and editable source) |
| `tests/` | 404 tests: models, pipeline, REST API, integrity fixes, Vercel entry point |

## Visual Evaluation Suite

| Test Image | Defect / Hazard | Domain & Standard | Result & Metric | Visual Evidence Report |
|---|---|---|---|---|
| 01. Waterlogged Earthen Road | 4 Cavities, Severe Waterlogging | earlier run only: classified Unpaved Rural Earthen by the removed SurfaceTypeNet; the pipeline now reports `paved_unverified` | earlier-run figures (PCI 35.9, camber grading) are not reproduced by the current pipeline | `reports/visual_verification/01_user_upload_multipothole_earthen.png` |
| 02. Real Asphalt Pothole Cavity | 2 Deep Potholes (Avg 84%) | Flexible Bituminous (IRC:37 / MoRTH Cl. 500) | PCI 53.8 (POOR), Bitumen patching applicable | `reports/visual_verification/02_kaggle_pothole_cavity.png` |
| 03. CRACK500 Fatigue Cracking | Center & Wheelpath Fatigue Cracks | Flexible Bituminous (ASTM D6433) | PCI 72.7 (SATISFACTORY), 44.9 deducts | `reports/visual_verification/03_crack500_fatigue_crack.png` |
| 04. Missing Zebra Crossing | Missing Pedestrian Markings (87%) | Carriageway Bound (IRC:35-2015 Cl. 7.2) | Carriageway box `[148, 343, 492, 137]` | `reports/visual_verification/04_missing_zebra_crossing.png` |
| 05. Damaged Traffic Sign | Regulatory Sign Deformation (85%) | Road Furniture (IRC:67-2012) | Microprismatic retroreflective retrofit | `reports/visual_verification/05_damaged_traffic_sign.png` |

## Test Suite & Verification Gates

- **Unit tests**: 375 pass and 1 is skipped (full suite, Python 3.13, run before the last merge).
- **False alarms**: the retrained detector's clean-road check has not been run, so its false-alarm rate on sound pavement is not measured; its boxes are hidden.
- **Serverless and local paths**: see the test suite; no claim of identical outputs is made here.

## Limitations

- **No bus data yet.** Photographs are DNIT (Brazil), Kaggle and RDD2022 India smartphone images; IMU logs are from a car. A bus-mounted pilot is the next step.
- **Four rare classes** (waterlogging, missing zebra crossing, missing divider, damaged sign) have ~50 photographs each, and their scores are unstable.
- **The Indian test covers three classes** (normal, crack, pothole), not seven.
- **Segmentation** (crack IoU 0.208, pothole IoU 0.119) is measured on DNIT photographs, not Indian or bus-camera frames; area and cost inherit its error.
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
| Concrete patches excluded, RDD2022 India added, fine-tuned CNN | **94.4%** (India **92.3%**) | larger, harder test set with Indian photographs in it |

An earlier README described models this code did not contain ("90.36% validation accuracy", "R² = 0.9908", a "10/10 PASS guaranteed" harness, simulated IMU data presented as real). It was removed in the October 2026 audit; see `CHANGES.md` and the corrections on the Architecture page.

---

**Team** Road Shield AI · **Developer** Udbhav Yadav · SIH26124 (BEL) · Theme: Smart Automation  
Developed for Smart India Hackathon 2026, academic and research use.

<sub>Generated by `scripts/build_readme.py` from the files in `checkpoints/`.</sub>
