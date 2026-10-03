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
| Road distress classification, 7 classes | Frozen MobileNetV2 ImageNet embeddings (ONNX Runtime), flip test-time augmentation → RBF SVM head, chosen by grouped 5-fold cross-validation | **87.0%**, macro-F1 0.700, on 625 held-out images from 589 unseen photographs, scored once |
| — same task, end-to-end fine-tuned CNN | EfficientNet-B0/B2, MobileNetV3-L, ResNet-50 fine-tuned on a Colab GPU (`training/train_finetune_cnn.py`), exported to ONNX | Served **only** if it beats the head on validation accuracy *and* macro-F1 (`scripts/select_vision_model.py`); the site shows whichever is served |
| — on Indian roads | Same classifier on crops from the official RDD2022 India **test** split, never used in training | **80.4%**, macro-F1 0.781, on 997 crops from 778 photographs (normal / crack / pothole) — 33.4% before Indian data was added |
| — fallback path | HOG + LBP + colour features → PCA → class-balanced RBF SVM | 79.0%, macro-F1 0.522, on the same split; serves when no CNN backbone is on disk |
| Object detection, 80 classes | YOLOv8n trained on COCO, served through ONNX Runtime | Pretrained; people, vehicles, traffic lights, signs. Not re-trained or re-measured here |
| IMU shock classification | 100 Hz tri-axial accelerometer windows → RandomForest; a 1-D CNN is compared by 5-fold CV and served only if better | **87.2%** on 164 held-out windows of real Indian-road drive logs, time-block split |
| Defect **segmentation** | Pixel classifier on 11 features, trained on the DNIT polygons | **crack IoU 0.231, pothole IoU 0.144** on 500 unseen photographs; 23.3% of clean photographs show a false blob before the gate |
| Semantic gate | CNN window heat map filters segmenter pothole pixels | pothole IoU 0.102 → 0.271, clean-road false blobs 8/50 → 2/50 |
| Defect **area** | Each mask pixel's own ground footprint, summed | Measured if the camera is calibrated — a bounding box overstates a diagonal crack ~13× |
| Camera **calibration** | Per-device profile from a checkerboard or published FOV | Per vehicle; 30 cm of mount height moves area ~46% |
| Defect **depth** | IRC band placed by measured extent / cavity contrast | **Estimate with an interval**, never a measurement |
| Privacy redaction | People (head region), number plates and faces blurred before an image is shared | Implemented; **recall not measured** (no annotated set) |
| Video ingest | cv2 decode, frames sampled by ground distance, pHash suppression | Real decoding; refuses to invent GPS |
| Fleet ledger + deduplication | SQLite; haversine 8 m, same defect class only; raw sightings kept as the audit trail | Survives restart; verified by tests |
| Repair costing | MoRTH Section 500 bitumen tonnage and rates (compaction factor 1.15) | Deterministic; rate basis shown in every output |
| Pavement condition index | ASTM D6433 deduct-value procedure | Reproduces the deduct curves; not validated against field surveys |
| Tamper-proof work orders | SHA-256 seal over the order fields; no GPS → `HELD_NO_GPS`, never a made-up location | Verified by tests |
| Repair verification | SSIM + Laplacian variance + perceptual hash | Catches resubmitted photographs |
| Sensor fusion | Bayesian gate over vision + IMU evidence | Reports "unavailable" when no IMU window exists |
| GIS services | Google Maps if a key is set, else OpenStreetMap Nominatim / OSRM / Overpass / Open-Meteo | Reports UNAVAILABLE rather than inventing data |

The live numbers are in `checkpoints/claims.json`, rebuilt from the reports by
`python -m scripts.build_claims`, and the site reads the checkpoints directly —
if a retraining run changes a number, the site changes with it.

## The measurement chain

The classifier's accuracy is measured. Everything *after* it decides the rupee
figure, and those stages are labelled individually — because half are
measurements and half are estimates:

| Stage | What it is | Provenance |
|---|---|---|
| Classification | MobileNetV2 embeddings → SVM head (or the fine-tuned CNN, if selected) | **measured**, 87.0% |
| Segmentation | pixel classifier on real polygons | **measured**, crack IoU 0.231 |
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
| Held-out accuracy | **87.0%**, macro-F1 0.700, on 625 images from 589 unseen photographs (random guess 14.3%) |
| Indian roads (RDD2022 test split) | **80.4%**, macro-F1 0.781, 997 crops / 778 photographs, 3 classes |
| Same split, hand-crafted features | 79.0%, macro-F1 0.522 |
| Model selection | 5-fold cross-validation on the training split only, grouped by source photograph; the test set is scored once, by the selected model |
| Leakage audit | 0 cross-class duplicates, 0 photograph groups spanning splits (`checkpoints/validation_report.json`) |
| Latency | ~30 ms for the two MobileNetV2 embeddings (image + mirror); full pipeline 1.7–2.4 s on a single-core VM, dominated by pixel segmentation |

Reproduce: `python -m training.train_cnn_head --backbone mobilenetv2 --compare`
and `python -m scripts.eval_indian_roads`. The full GPU run (fine-tuning, IMU
CNN, benchmark, claims, report) is `scripts/colab_train_all.sh`.

### How the head was chosen

Thirteen candidates — three head families at several settings, on plain and on
flip-averaged embeddings, plus a soft-voting ensemble — were scored by grouped
cross-validation inside the training split (score = mean of accuracy and
macro-F1). The best is the only model that saw the test set.

| Head | Features | CV accuracy | CV macro-F1 |
|---|---|---|---|
| **svc_C3** | flip_tta | 80.4% ± 1.4 | 0.751 ± 0.031 |
| svc_C10 | flip_tta | 80.5% ± 1.7 | 0.750 ± 0.029 |
| svc_C30 | flip_tta | 80.3% ± 1.7 | 0.750 ± 0.029 |
| ensemble_soft | flip_tta | 80.4% ± 1.8 | 0.747 ± 0.038 |
| svc_C3 | plain | 80.1% ± 1.3 | 0.743 ± 0.028 |
| svc_C10 | plain | 79.7% ± 1.8 | 0.741 ± 0.033 |
| svc_C30 | plain | 79.0% ± 1.8 | 0.738 ± 0.034 |
| mlp_512_256 | flip_tta | 78.6% ± 2.0 | 0.728 ± 0.033 |
| mlp_512_256 | plain | 79.0% ± 2.5 | 0.724 ± 0.036 |
| logistic_C0.3 | flip_tta | 77.2% ± 1.3 | 0.703 ± 0.021 |
| logistic_C1 | flip_tta | 76.9% ± 1.4 | 0.697 ± 0.022 |
| logistic_C0.3 | plain | 76.2% ± 1.2 | 0.682 ± 0.020 |
| logistic_C1 | plain | 75.4% ± 1.5 | 0.677 ± 0.022 |

CV figures are lower than the test figure because the validation folds also
contain the harder region-scale sub-crops and the Indian training crops.

### Per class

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Normal Road / Sound Pavement | 0.85 | 0.94 | 0.89 | 148 |
| Crack (Longitudinal / Transverse / Alligator) | 0.84 | 0.92 | 0.88 | 225 |
| Pothole Cavity | 0.92 | 0.82 | 0.87 | 222 |
| Waterlogging / Flooding Hazard | 1.00 | 0.14 | 0.25 | 7 |
| Missing Zebra Crossing | 0.71 | 0.62 | 0.67 | 8 |
| Missing Road Divider | 0.75 | 0.38 | 0.50 | 8 |
| Damaged Traffic Sign | 1.00 | 0.71 | 0.83 | 7 |

Normal road, crack and pothole carry 595 of the 625 test images. The other four
classes have seven or eight test images each, so one image moves their F1 by
more than 0.1. That is a data-volume problem and it is shown, not averaged away.

## Data

| Source | Classes | Kept | Note |
|---|---|---|---|
| DNIT *Cracks and Potholes in Road Images* (Brazil) | crack, pothole, normal | 1,667 crops | 2,235 photographs, 4,720 polygons; the only source with polygons, so the segmenter trains on it |
| RDD2022 India (CRDDC 2022, smartphone) | crack, pothole, normal | 2,947 crops | 2,173 training photographs; a separate 997-crop set from the official test split is never trained on |
| Kaggle pothole sets (3) | pothole, normal | 1,287 | `virenbr11/...` was 94% a re-upload: 700 of 739 rejected as perceptual duplicates |
| Kaggle surface cracks (concrete walls) | — | 0 of 622 | **Excluded**: close-ups of concrete and plaster, no road, no horizon (`pipeline/corpus_policy.py`); kept on disk, reproducible with `ROAD_SHIELD_NO_CORPUS_FILTER=1` |
| Wikimedia Commons / Geograph + field photographs | waterlogging, zebra, divider, sign | 207 | 67 field + 140 Wikimedia (35 per class); still ~50 photographs per class |
| IMU drive logs (`VishalSingh25/Pothole-Project`) | 4 shock classes | 852 windows | 10 real drives on Indian roads, 205,491 samples at 100 Hz, from a car — not a bus |

Original DNIT data: github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset ·
RDD2022: Arya et al., *RDD2022: A multi-national image dataset for automatic road damage detection* (figshare).

`andrewmvd/road-sign-detection` was removed after it went in: it is a dataset of
road signs, not *damaged* road signs, so the model learned "a sign is present"
under a label claiming "this sign is damaged". Removing its 299 images raised
accuracy (88.5% → 89.2% on the corpus of that time).

**Object detection:** COCO, 330,000 images, through the published YOLOv8n weights.

### Segmentation — measured

| Class | IoU | Dice | Pixel precision | Pixel recall |
|---|---|---|---|---|
| Crack | 0.231 | 0.376 | 0.373 | 0.379 |
| Pothole | 0.144 | 0.251 | 0.242 | 0.262 |

500 unseen photographs, split by photograph, scored only on pixels inside the
lane polygon. Trained on 3.7 million labelled pixels from the DNIT hand-drawn
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
starts the server to confirm every page serves.

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
python -m scripts.fetch_cnn_backbone --model mobilenetv2       # 14 MB backbone, once
python -m training.train_cnn_head --backbone mobilenetv2 --compare   # the 87.0% model
python -m api.server                                           # http://127.0.0.1:8000/
```

On startup the server prints which classifier it loaded: the fine-tuned CNN if
`checkpoints/vision_model_selection.json` selected it, otherwise the MobileNetV2
head; `HOG/LBP + SVM baseline` means no backbone is on disk and it fell back. A
head only ever runs with the backbone it was trained on — the loader pairs them
and skips mismatches.

Deep training (GPU) runs end to end in Colab — open
`notebooks/ROAD_SHIELD_colab_training.ipynb`, or:

```bash
bash scripts/colab_train_all.sh 2>&1 | tee logs/colab_run.txt   # resumable if Google Drive is mounted
```

Optional extras:

```bash
python -m scripts.fetch_detector            # COCO object detector (needs ultralytics once)
python -m scripts.validate_models           # leakage, cross-validation, calibration, latency
python -m scripts.model_selection           # compare six classifiers on identical folds
python -m unittest discover -s tests -t .  # 178 tests: models, pipeline, REST API, integrity fixes, Vercel entrypoint
python -m training.train_segmenter         # pixel segmentation on the DNIT polygons
python -m scripts.calibrate_camera --list  # camera calibration profiles
python -m training.train_finetune_cnn       # fine-tune CNNs (needs PyTorch + GPU)
python -m training.train_imu_deep           # IMU 1-D CNN vs RandomForest
python -m scripts.fetch_rdd2022_india       # RDD2022 India (official release)
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
excluded from road-scene training and measurement by `pipeline/corpus_policy.py`;
the files stay on disk and the exclusion is listed in every report that depends on it.

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
| Kaggle ingest + a labelling fix | 89.2% | 2,373 distinct photographs; removing 299 mislabelled sign images *raised* accuracy |
| Concrete patches excluded, RDD2022 India added, larger test set | **87.0%** | 589-photograph test set (was 356) with Indian crops in it; Indian-roads test 33.4% → 80.4% |

`scripts/validate_models.py` is what found the label conflicts, by hashing
every image and looking for the same photograph under different labels.

## Honest limitations

- The served classifier is a frozen ImageNet CNN with a trained head unless a fine-tuned network beats it on validation; the rule is fixed in advance and the choice is recorded in `checkpoints/vision_model_selection.json`.
- Indian photographs come from RDD2022 (smartphone); none were taken from a bus, and the Indian test covers three classes only.
- The IMU data is real but small: 10 drive logs from a car, not a bus fleet.
- Privacy redaction is implemented but its recall has not been measured.
- A semantic gate (the CNN classifier filtering the segmenter) raised pothole IoU from 0.102 to 0.271 and cut clean-road false alarms from 16% to 4% on photographs the classifier never saw; water-filled cavities are still under-detected.
- Before the gate, the segmenter drew a false blob on 23.3% of clean road photographs (the regression gate is 25%). The classifier in front of it limits the damage; painted markings remain the hardest case.
- PCI, deterioration and depth models are fitted to engineering formulas (ASTM D6433, HDM-4, an IRC depth band). Their R² measures fidelity to the formula, not field accuracy.
- There is no dashcam video in this repository; the video page decodes a clip you upload. The rest of the system analyses photographs.
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
