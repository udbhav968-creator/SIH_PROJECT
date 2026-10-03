# ROAD-SHIELD AI Engine

**Smart India Hackathon 2026 · Problem statement SIH26124 · Bharat Electronics Limited**
AI-assisted road quality assessment using the public bus fleet as a mobile sensing network.

A road photograph goes in; a classified, measured and costed repair record comes
out, sealed so it cannot be quietly altered. Every figure below was measured by
code in this repository, and the code that measures it is included.

---

## What it actually does

| Capability | How it works | Status |
|---|---|---|
| Road distress classification, 7 classes | MobileNetV2 ImageNet embeddings (ONNX Runtime), flip test-time augmentation → soft-voting ensemble (SVC + logistic + MLP), chosen by grouped cross-validation | **88.8%**, macro-F1 0.747, on 502 held-out images from 483 unseen photographs, scored once |
| — same task, ResNet-50 path | ResNet-50 embeddings | not shipped: fetch the 98 MB backbone, then `python -m training.train_cnn_head --backbone resnet50 --compare` |
| — same task, fallback path | HOG + LBP + colour features → PCA → class-balanced RBF SVM | 86.2% on its own 807-photo split, 79.9% on the CNN head's split; serves when no CNN backbone is on disk |
| Object detection, 80 classes | YOLOv8n trained on COCO, served through ONNX Runtime | Working: people, bicycles, cars, buses, trucks, traffic lights, signs |
| IMU shock classification | 100 Hz tri-axial accelerometer windows → RandomForest | **87.2%** on 164 held-out windows of real Indian-road drive logs, time-block split |
| Defect **segmentation** | Pixel classifier on 11 features, trained on 4,720 hand-drawn polygons | **crack IoU 0.231, pothole IoU 0.144** on 500 unseen photographs; 23.3% of clean photographs still show a false blob |
| Defect **area** | Each mask pixel's own ground footprint, summed | Measured — a bounding box overstates a diagonal crack ~13× |
| Camera **calibration** | Per-device profile from a checkerboard or published FOV | Per vehicle; 30 cm of mount height moves area ~46% |
| Defect **depth** | IRC band placed by measured extent / cavity contrast | **Estimate with an interval**, never a measurement |
| Video ingest | cv2 decode, frames sampled by ground distance, pHash suppression | Real decoding; refuses to invent GPS |
| Fleet ledger | SQLite, raw sightings kept as the dedup audit trail | Survives restart |
| Repair costing | MoRTH Section 500 bitumen tonnage and rates | Deterministic |
| Pavement condition index | ASTM D6433 deduct-value procedure | Deterministic |
| Tamper-proof work orders | SHA-256 seal over the order fields | Verified by tests |
| Fleet deduplication | Haversine distance clustering of reports | Verified by tests |
| Repair verification | SSIM + Laplacian variance + perceptual hash | Catches resubmitted photographs |
| Sensor fusion | Bayesian gate over vision + IMU evidence | Reports "unavailable" when no IMU window exists |
| GIS services | Google Maps if a key is set, else OpenStreetMap Nominatim / OSRM / Overpass / Open-Meteo | Reports UNAVAILABLE rather than inventing data |

## The measurement chain

The classifier's 89.2% is measured. Everything *after* it decides the rupee
figure, and those stages are now labelled individually — because half are
measurements and half are estimates:

| Stage | What it is | Provenance |
|---|---|---|
| Classification | ResNet-50 embeddings → logistic head | **measured**, 89.2% |
| Segmentation | pixel classifier on 4,720 real polygons | **measured**, crack IoU 0.231 |
| Area | each mask pixel's ground footprint, summed | **measured** *if* the camera is calibrated |
| Camera geometry | per-device profile, rescaled per request | **measured** with a profile, **estimate** without |
| Depth | IRC band placed by extent or cavity contrast | **estimate**, always with an interval |
| Cost | MoRTH Section 500 × area × depth | **range**, because depth is a range |

Measured on one photograph through the API: with the assumed mount the defect
comes out at 0.007 m² and ₹4.5; with a calibrated profile for the same vehicle,
0.049 m² and ₹34.5. Seven times. That is why calibration is not a detail, and
why every response carries the provenance of the numbers in it.

## Measured performance

| Test | Result |
|---|---|
| Held-out accuracy | **88.8%**, macro-F1 0.747, on 502 images from 483 unseen photographs (random guess 14.3%) |
| Same split, hand-crafted features | 79.9%, macro-F1 0.520 |
| Model selection | 5-fold cross-validation on the training split only, grouped by source photograph; the test set is scored once, by the selected model |
| Leakage audit | 0 cross-class duplicates, 0 photograph groups spanning splits (`checkpoints/validation_report.json`) |
| Latency | ~30 ms for the two MobileNetV2 embeddings (image + mirror); full pipeline 1.7–2.4 s on a single-core VM, dominated by pixel segmentation |

Reproduce: `python -m training.train_cnn_head --backbone mobilenetv2 --compare`.

### How the head was chosen

Thirteen candidates — three head families at several settings, on plain and on
flip-averaged embeddings, plus a soft-voting ensemble — were scored by grouped
cross-validation inside the training split. The ensemble on flip-averaged
embeddings scored best and is the only model that saw the test set. CV figures
are lower than the test figure because the validation folds also contain the
harder region-scale sub-crops.

| Head | Features | CV accuracy | CV macro-F1 |
|---|---|---|---|
| ensemble_soft | flip_tta | 84.9% ± 1.8 | 0.769 ± 0.033 |
| svc_C10 | flip_tta | 84.3% ± 2.6 | 0.765 ± 0.036 |
| mlp_512_256 | flip_tta | 84.2% ± 1.8 | 0.764 ± 0.029 |
| svc_C30 | flip_tta | 83.9% ± 2.4 | 0.764 ± 0.033 |
| svc_C3 | flip_tta | 83.6% ± 2.8 | 0.762 ± 0.037 |
| svc_C10 | plain | 83.6% ± 2.3 | 0.758 ± 0.033 |
| svc_C30 | plain | 83.5% ± 2.9 | 0.758 ± 0.035 |
| svc_C3 | plain | 83.2% ± 2.5 | 0.759 ± 0.034 |
| mlp_512_256 | plain | 82.9% ± 2.6 | 0.749 ± 0.037 |
| logistic_C0.3 | flip_tta | 82.3% ± 1.8 | 0.732 ± 0.031 |
| logistic_C1 | flip_tta | 81.7% ± 1.5 | 0.720 ± 0.031 |
| logistic_C0.3 | plain | 81.4% ± 0.9 | 0.715 ± 0.020 |
| logistic_C1 | plain | 81.3% ± 1.4 | 0.714 ± 0.028 |

The previous version picked its head by test-set score, which inflates the
reported figure. This one did not, so its 88.8% is directly
comparable to unseen data; the 0.4-point difference from the earlier 89.2% is
two images and within noise.

### Per class

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Normal Road / Sound Pavement | 0.88 | 0.86 | 0.87 | 77 |
| Crack (Longitudinal / Transverse / Alligator) | 0.90 | 0.93 | 0.91 | 184 |
| Pothole Cavity | 0.93 | 0.90 | 0.91 | 211 |
| Waterlogging / Flooding Hazard | 0.67 | 0.86 | 0.75 | 7 |
| Missing Zebra Crossing | 0.43 | 0.38 | 0.40 | 8 |
| Missing Road Divider | 0.45 | 0.62 | 0.53 | 8 |
| Damaged Traffic Sign | 0.86 | 0.86 | 0.86 | 7 |

Normal road, crack and pothole carry 472 of the 502 test images. The other four
classes have seven or eight test images each, so one image moves their F1 by
more than 0.1. That is a data-volume problem and it is shown, not averaged away.

## Data

| # | Class | Images | Distinct photographs |
|---|---|---|---|
| 0 | Normal Road / Sound Pavement | 1807 | ~1580 |
| 1 | Crack (Longitudinal / Transverse / Alligator) | 2413 | ~2400 |
| 2 | Pothole Cavity | 1404 | ~1400 |
| 3 | Waterlogging / Flooding Hazard | 14 | 14 |
| 4 | Missing Zebra Crossing | 19 | 19 |
| 5 | Missing Road Divider | 20 | 20 |
| 6 | Damaged Traffic Sign | 14 | 14 |

2,373 distinct photographs in total after the Kaggle ingest. The imbalance is
the point: classes 3-6 are the ones holding macro-F1 down, and no amount of
modelling substitutes for photographs of them.

**Primary source:** *Cracks and Potholes in Road Images* — 2,235 photographs
collected by DNIT, the Brazilian federal highway department, with 1,921 crack
and 564 pothole polygon annotations. Each annotated defect is cropped into a
training example by `scripts/fetch_cracks_potholes_dataset.py`.
Original: github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset ·
COCO conversion: github.com/andrijdavid/Cracks-and-Potholes-in-Road-Images-Dataset

**Kaggle:** four datasets pulled through the official API — surface cracks
(concrete, close range: real crack texture but not road scenes), and three
pothole/plain-road sets. `virenbr11/pothole-and-plain-rode-images` turned out to
be 94% a re-upload of `atulyakumar98/pothole-detection-dataset`; 700 of its 739
images were rejected as perceptual duplicates. That check is the difference
between an honest score and an inflated one.

A fifth, `andrewmvd/road-sign-detection`, was removed after it went in: it is a
dataset of road signs, not damaged road signs, so the model learned "a sign is
present" under a label claiming "this sign is damaged". Removing its 299 images
raised accuracy from 88.5% to 89.2%.

**Smaller classes:** Wikimedia Commons and Geograph photographs.

**Object detection:** COCO, 330,000 images, through the published YOLOv8n weights.

**IMU:** 15,000 windows shipped with this repository. These are **simulated**,
not recorded from a vehicle, and the model's 100% score should be read in that
light.

**Not Indian road data.** The photographs are Brazilian. RDD2022 provides
47,000 annotated images including India and is the obvious next step; the
downloader is written and waiting on a GPU.

### Segmentation — measured

| Class | IoU | Dice | Pixel precision | Pixel recall |
|---|---|---|---|---|
| Crack | 0.231 | 0.376 | 0.373 | 0.379 |
| Pothole | 0.144 | 0.251 | 0.242 | 0.262 |

500 unseen photographs, split by photograph, scored only on pixels inside the
lane polygon. Trained on 5.6 million labelled pixels from 4,720 hand-drawn
polygons. The decision is a per-class threshold tuned for IoU on a separate
calibration split rather than argmax — with ~97% of pixels being sound road,
argmax floods the mask with false positives, and fixing that alone took crack
IoU from 0.154 to 0.187 before more data took it to 0.232.

These are working numbers, not solved ones. What they replace is a bounding box
that had no measured accuracy at all.

## Deployment

```bash
docker build -t road-shield .
docker run -p 8000:8000 -v "$PWD/checkpoints:/app/checkpoints" road-shield
```

The checkpoints volume carries the trained models and the SQLite ledger; without
it the container starts with neither, and the `/system` page says so. No CUDA and
no PyTorch — the CNN runs on ONNX Runtime on the CPU.

GitHub Actions runs the suite on 3.11 and 3.12, checks every module imports, and
starts the server to confirm all eight pages serve.

## The site

Eight pages, not one dashboard file:

| Page | What it is for |
|---|---|
| `/` | overview and the honest limits |
| `/inspect` | analyse a photograph; mask, area, depth interval, cost range |
| `/video` | dashcam ingest, sampled by ground distance |
| `/corridor` | fleet map and the deduplication ledger |
| `/works` | costing and the SHA-256 tamper demonstration |
| `/models` | model card, per class, including what scores badly |
| `/data` | dataset lineage, duplicate rejection, what the data misses |
| `/system` | components, calibration, storage, live endpoint self-test |

Measured and estimated numbers are visually distinguished on every page. On this
system that is a correctness requirement, not decoration.

## Running it

```bash
pip install -r requirements.txt
python -m scripts.run_full_pipeline                            # everything below, in one command
```

Or stage by stage:

```bash
python -m scripts.fetch_cracks_potholes_dataset --limit 2235   # ~70 s, real data
python -m training.train_mega_suite                            # ~3 min, baseline models
python -m scripts.fetch_cnn_backbone                           # 98 MB ResNet-50, once
python -m training.train_cnn_head --compare                    # ~4 min, the 89.2% model
python -m api.server                                           # http://127.0.0.1:8000/dashboard
```

On startup the server prints which classifier it loaded. `deep CNN embeddings
(cnn:resnet50+logistic)` is the 89.2% path; `HOG/LBP + SVM baseline` means the
backbone is missing and it fell back. On a slow machine,
`python -m scripts.fetch_cnn_backbone --model mobilenetv2` is 14 MB and 10 ms
per image instead of 33, at 83.5% accuracy. A head only ever runs with the
backbone it was trained on — the loader pairs them and skips mismatches.

Optional extras:

```bash
python -m scripts.fetch_detector            # COCO object detector (needs ultralytics once)
python -m scripts.validate_models           # leakage, cross-validation, calibration, latency
python -m scripts.model_selection           # compare six classifiers on identical folds
python -m unittest discover -s tests       # 140 tests: models, pipeline, REST API, Vercel entrypoint
python -m training.train_segmenter         # pixel segmentation on the DNIT polygons
python -m scripts.calibrate_camera --list  # camera calibration profiles
python -m training.train_deep_vision        # fine-tune a CNN (needs PyTorch)
```

### Kaggle datasets

```bash
pip install kaggle                                   # once
# kaggle.com -> Settings -> API -> Create New Token, save to ~/.kaggle/kaggle.json
python -m scripts.fetch_kaggle_datasets --verify     # what's reachable, downloads nothing
python -m scripts.fetch_kaggle_datasets --plan       # downloads, shows the mapping, copies nothing
python -m scripts.fetch_kaggle_datasets              # ingest
python -m scripts.fetch_kaggle_datasets --undo       # remove every image it added
```

Images are filed by the folder names the dataset actually uses - `potholes/`,
`Positive/`, `plain road/` - not by a layout assumed in advance, and anything
unrecognised is skipped rather than guessed at. Every candidate is perceptually
hashed and dropped if it is a near-duplicate of an image already present:
several Kaggle pothole datasets are re-uploads of each other, and without this
the same photograph lands in both training and test.

The duplicate threshold is measured, not guessed. Over 60 photographs, 64-bit
pHash distance between an image and a copy of it was at most 2 bits after a
JPEG re-encode and at most 8 after a half-size re-upload, while genuinely
different photographs sat at 18 bits and above.

Surface-crack datasets are close-range concrete, not road scenes. They are
real crack images and they help, but the domain gap is real; the catalogue
marks them `domain="concrete"` and the ingest manifest records how many images
came from each domain.

A Google Maps key is optional and not needed for anything in the demo; without
one the system uses OpenStreetMap (Nominatim, OSRM, Overpass) and Open-Meteo,
and reports `UNAVAILABLE` rather than inventing a result.

## How accuracy got here

| Stage | Held-out accuracy | What changed |
|---|---|---|
| Inherited code | 36.6% | Models were untrained; some outputs were fabricated |
| Real training | 36.6% | Genuine classifiers trained on the 133 photographs present |
| Label conflicts fixed | 79.4% | 20 photographs were filed under three contradictory labels at once |
| Real dataset added | 83.4% | 133 → 1,800 distinct photographs |
| ImageNet CNN embeddings | 88.5% | ResNet-50 replaces hand-written features |
| Kaggle ingest + a labelling fix | **89.2%** | 2,373 distinct photographs; removing 299 mislabelled sign images *raised* accuracy |

`scripts/validate_models.py` is what found the label conflicts, by hashing
every image and looking for the same photograph under different labels.

## Honest limitations

- The classifier is a frozen ImageNet CNN with a trained head, not a fine-tuned
  network. Fine-tuning (`training/train_deep_vision.py`) needs PyTorch: `pip install -r requirements-train.txt`.
- The photographs are Brazilian, not Indian.
- The IMU data is real but small: 10 drive logs from one project, not a fleet.
- A semantic gate (the CNN classifier filtering the segmenter) raised pothole IoU from 0.102 to 0.271 and cut clean-road false alarms from 16% to 4% on photographs the classifier never saw; water-filled cavities are still under-detected.
- Before the gate, the segmenter drew a false blob on 23.3% of clean road photographs (the regression gate is 25%). The classifier in front of it limits the damage; painted markings remain the hardest case.
- PCI, deterioration and depth models are fitted to engineering formulas (ASTM D6433, HDM-4, an IRC depth band). Their R² measures fidelity to the formula, not field accuracy.
- There is no dashcam video in this repository; the system analyses photographs.
- Four classes have too few examples to work well.
- No demographic inference is performed on people in frame, by design.


---

## Older sections removed

This README used to continue with an earlier architecture write-up (a "9-class" VisionDistressNet, an UrbanTrafficNet
with "90.36% validation accuracy", a PCI regressor with "R² = 0.9908", a "10/10 PASS guaranteed" test harness, a
CRACK500 dataset and 15,000 simulated IMU windows). None of that describes the code in this repository, so it was
deleted in the October 2026 audit rather than left beside the measured numbers above. See `CHANGES.md`.

---

## 👥 Team

**Team:** Road Shield AI  
**SIH Problem:** SIH26124 — Bharat Electronics Limited (BEL)  
**Theme:** Smart Automation  
**Developer:** Udbhav Yadav  

---

## 📄 License

This project is developed for **Smart India Hackathon 2026** under academic/research use.  
Inference uses NumPy, scikit-learn, OpenCV and ONNX Runtime; training of the fine-tuned networks uses PyTorch (`requirements-train.txt`).

---

<p align="center">
  <b>🛡️ ROAD-SHIELD | Protecting Indian Roads, One Bus at a Time</b><br>
  <sub>Built with ❤️ for Bharat Electronics Limited · MoRTH/NHAI · Smart India Hackathon 2026</sub>
</p>
