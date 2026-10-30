# ROAD-SHIELD AI Engine — rewrite notes

## Latest: 30 Oct (3) - flooded potholes, duplicate boxes, labels, survey wording

- **No zebra crossing inside a pothole** (`models/marking_detector.discard_bars_in_defects`,
  applied where the main engine assembles its result). Water in a pothole reflects the sky
  as bright bars, and the main engine reported "Zebra Crossing 95% - 3 bars" on flooded
  potholes. Bars whose centre lies inside a pothole or waterlogging detection are dropped.
  Measured through `audit_image()`: on 40 pothole photographs where the detector fires,
  false crossings 40 -> 29 (potholes found unchanged, 30/40); flooded streets 2/14 -> 1/14;
  real crossings unchanged at 4/19. A first version tested against the hole-filled pothole
  mask removed none (reflections sit outside it) and was replaced.
- **One cavity, one detection** (`_merge_overlapping_potholes`). The mask breaks around the
  bright water in a pothole, and each piece was reported as a pothole ("100%", "96%", "87%"
  on one cavity). Pothole proposals whose regions overlap are merged; separate potholes stay
  separate; area is still counted from the mask's pothole pixels. On 30 pothole
  photographs: 56 -> 45 pothole boxes, photographs with 2+ pothole boxes 15 -> 11. Detection
  quality unchanged (36 clean / 24 defect: 3/36 false positives, 22/24 found).
- **Labels** (`web/inspect.html`): vehicle labels no longer cover defect labels (a crack box
  showed only "99%" under "car 82%"); a probability of 0.995+ shows as ">=99%", not "100%".
- **Survey panel** states the actual reason (area above the single-patch bound, or the share
  of the frame) and no longer says the defect "is priced above" when no cost is quoted.
- **Same photograph, same answer, whatever the input type** (`models/cv_cavity_detector.decode_image`).
  Files, uploads (base64) and bytes were resized to the working resolution, but numpy
  arrays - the camera frames `edge/bus_agent.py` passes, and video frames - were analysed at
  their own resolution. On the 36 clean / 24 defect check photographs: 6/36 false positives
  as arrays vs 3/36 as files, so the published 3/36 did not describe the bus deployment.
  Arrays are now resized the same way: 3/36 and 22/24 for every input type.

## Previous: 30 Oct (2) - the website runs the served segmenter and the main engine's rules

Website engine (`api/vercel_inference.py`, Vercel build), 36 clean / 24 defect check
photographs, measured in an environment with only the Vercel packages (numpy, Pillow,
onnxruntime):

| Stage | Clean roads flagged | Defects found |
|---|---|---|
| Before 30 Oct | 32/36 | 21/24 |
| + box-level semantic gate, no invented hazard boxes | 11/36 | 19/24 |
| + served pixel segmenter (this change) | 12/36 | 22/24 |
| + main engine's region rules | 11/36 | 22/24 |
| + semantic gate per pixel, as in the main engine | **8/36** | **22/24** |
| Main engine, for reference | 3/36 | 22/24 |

- **Served segmenter on Vercel.** `scripts/export_pixel_segmenter_onnx.py` builds
  `checkpoints/defect_segmenter_pixel.onnx` (1.5 MB) from the fitted trees; probabilities
  match scikit-learn to 8e-7. `models/pixel_segmenter_lite.py` ports the 11 features to
  numpy; on 42 photographs 99.7% of pixel labels agree with the OpenCV original and all 42
  get the same defect / no-defect verdict (`--verify`). The website follows
  `segmenter_selection.json`; it previously always ran the U-Net.
- **Region rules match the main engine.** A crop the classifier calls sound pavement, or
  zebra / divider / sign, is dropped; reports need confidence >= 0.45. Removed: keeping
  "sound" regions with a small mask, relabelling zebra/divider/sign crops as damage at
  p >= 0.20, a "sky reflection" pothole rule, and a 0.65 confidence floor.
- **Semantic gate per pixel** before blobs are formed (as `models/semantic_gate.apply`),
  heat map computed once and reused for the box check.
- **Health page** reports the segmenter actually running, not "U-Net ResNet-34" (the
  encoder is ResNet-18, and the U-Net is not served).
- Remaining gap to the main engine: crack regions and the salient-cavity proposals, which
  the main engine also filters with rim verification.

## Previous: 30 Oct - invented values removed from serving paths, CI gates the served model

- **Work-order seal key** (`api/vercel_inference.py`). The serverless path sealed work
  orders with HMAC key `ROAD-SHIELD-MORTH-SEAL` written in the public repository, so
  anyone could forge a "tamper-proof" seal. It now follows `models/morth_dispatch_agent`:
  HMAC-SHA256 only with `ROAD_SHIELD_SEAL_KEY` from the host environment, otherwise a
  plain SHA-256 labelled as such (`seal_algorithm`). **Set `ROAD_SHIELD_SEAL_KEY` in the
  Vercel project settings.**
- **Rutting and IRI are no longer invented.** Both pipelines filled them from the PCI or
  the pothole count (main: 3.0 + 3.5 mm per pothole and 1.8 + 0.9 m/km per pothole;
  serverless: straight lines from the PCI) and the main pipeline fed them into the PCI,
  so one pothole deducted three times. A photograph measures neither; they are now
  `None` with a `not_measured` reason and add no deduct.
- **Serverless PCI follows its own engine.** Removed a hidden floor that forced the
  corrected deduct to >= 65 (PCI <= 35) for large or deep potholes under a mis-cited
  "ASTM D6433" comment, an invented 14.5 "safety_corridor" deduct, and a PCI floor of 5.
  The large-cavity rule survives as an explicit operational `priority_repair` flag.
- **Serverless hazards no longer get an invented box.** A whole-frame classifier opinion
  of waterlogging / zebra / divider / sign was turned into a detection with a fixed
  rectangle and a distance (and for waterlogging an area) computed from it. It is now
  `frame_level_hazard`, which sets no area, depth or cost - the main engine's rule.
- **Serverless semantic gate.** The main engine's measured gate (CNN heat map, threshold
  0.5) now filters serverless pothole boxes. On the 36 clean / 24 defect check
  photographs: false positives 32/36 -> 11/36, defects found 21/24 -> 19/24. The main
  engine is 3/36 and 22/24; the gap is the pixel classifier, which the serverless build
  does not load.
- **CI gates the served segmenter** (`mlops/specs.py`). The gate checked the U-Net's IoU
  while the pixel classifier was served (its pothole IoU, 0.144, would fail that floor).
  It now gates `deployment_check.served_end_to_end` (detection 0.917, false positives
  0.083), recorded by `scripts/segmenter_deployment_check.py` for whichever is served.
- **Stress suite** (`scripts/deep_continuous_eval.py`) adds sound-pavement photographs,
  reports defects found and clean false alarms, scores each image once (the same photo
  sits in more than one dataset folder), says its gates are safety checks rather than
  accuracy, and exits non-zero on failure. It replaces "ALL MLOPS QUALITY GATES PASSED
  PERFECTLY", which printed while 4 of 5 pothole photographs returned no detection.

## Previous: 29 Oct audit - full-history review, hard-coded values removed, every claim re-measured

Reviewed all 213 commits and every serving, training and evaluation script.
Test suite: 352/356 -> 356/356 (1 skip by design).

**Re-measured in this audit (same splits, same code):**

| Claim | Re-measured | Verdict |
|---|---|---|
| Pixel segmenter, DNIT test (500 photos): crack 0.2314, pothole 0.1437 | 0.2314, 0.1437 | reproduces exactly |
| U-Net 30-epoch weights (92bf3f0): crack 0.2842, pothole 0.605 | 0.2842, 0.605 | reproduces exactly |
| U-Net weights now committed (0766e83) | crack 0.3035, pothole 0.4661 | previously reported with the old weights' numbers; now its own |
| IMU RandomForest: 86.6% on 164 held-out windows | 86.6% | reproduces exactly |
| DNIT corpus: 1,200 crack + 465 pothole crops | 1,200 + 465 | reproduces exactly |
| Classifier 94.9%, India 92.3% | not reproducible in the audit sandbox (RDD2022 India is on S3) | India figure belongs to the previous weights - see below |

- **Hard-coded IoU removed** (`training/train_unet_segmenter.py`). Since 3ff6c3c
  a run with fewer than 100 test photographs replaced a measured pothole IoU
  below 0.40 with 0.6050 before writing `segmenter_selection.json`. Such a run
  now leaves the full-dataset reports untouched and writes no number it did not
  measure. No other literal metric exists in training, evaluation or serving code.
- **Segmenter served by the deployment check.** The U-Net weights were replaced
  five times on 29 Oct without `scripts/segmenter_deployment_check.py`. End to
  end on the 36 clean / 24 defect check photographs (seed 7): 92bf3f0 6/36 false
  positives, 17/24 found; aff80cc and fa3140c 0/24 found; 6ff2ad3 33/36 false
  positives; c0f430f 19/36; 0766e83 6/36, 18/24; pixel classifier 3/36, 22/24.
  The pixel classifier wins on both counts and is served; the U-Net stays on disk.
  **Rule going forward: no weight file is committed without that check.**
- **Classifier India figure.** 87000a9 replaced the served classifier's weights
  ("rare-class head") with no training script and no validation set (the model is
  refit on train+val), and test macro-F1 rose 0.879 -> 0.921. On 400
  normal/crack/pothole photographs the old and new weights disagree on 12, yet the
  India result (92.25% on 1,200 crops) is identical to four decimals - it was not
  re-measured. Annotated in the report; re-measure with `scripts/eval_indian_roads.py`.
- **CAN-bus demo labelled** (`/api/v1/automotive/can-stream`): the response now
  declares `data_source: simulated_ui_fixture_not_vehicle_data` and the assumed
  vision probability (0.85), which were previously indistinguishable from vehicle data.
- **Zebra detector** (`models/marking_detector.py`). 0450d5e raised bar solidity
  0.35 -> 0.55, halving recall (4/19 -> 2/19) and doubling every line of the file.
  Repaired; 0.35 restored as `MIN_SOLIDITY` with the measured trade-off.
- **Pothole depth interval** (`models/depth_estimator.py`) keeps its width inside
  the band, so a weak cue never reports a tighter interval than a strong one.
- **Pitch deck and SIH submission** corrected to the repository's numbers
  (79.87%/9 classes, PCI R2 0.99, 10/10 tests, pure NumPy, 207 ms on ARM and
  94% fusion false-positive removal had no evidence here).
- **Reproducing the data.** DNIT crops (`cpr_*`) and RDD2022 India crops
  (`rddin_*`) are git-ignored by design. A fresh clone trains on about half the
  corpus unless `scripts/fetch_cracks_potholes_dataset.py --limit 2235` and
  `scripts/fetch_rdd2022_india.py` are run first.

## Previous: RDD2022 India - training data and an Indian-roads test

`scripts/ingest_rdd2022_india.py` reads the YOLO-format RDD2022 India set
(images/labels split into train/valid/test). Crack (D00/D10/D20) and pothole
(D40) boxes from train+valid become `rddin_*` training crops; photographs with
no label supply capped normal-road crops. The RDD **test** split goes only to
`datasets/_eval_rdd2022_india/`, which no training run reads, and
`scripts/eval_indian_roads.py` scores the served classifier on it - the first
accuracy figure measured on Indian roads. Crops from one photograph share a
group key, so they never straddle a split. Tested on a synthetic RDD layout,
including the guarantee that no test-split photograph reaches a training folder.

## Previous: semantic gate - pothole precision 0.12 -> 0.53

The pixel segmenter judges pixels from 11 local features and marks rough
foreground asphalt and dark crack lines as "pothole". The CNN classifier now
scores overlapping windows, and segmenter pothole pixels it does not see as
pothole are dropped (`models/semantic_gate.py`). On 110 annotated photographs
with no crop in the classifier's training data and 50 clean roads from its test
split: pothole IoU 0.102 -> 0.271, precision 0.12 -> 0.53, recall 0.39 -> 0.36,
crack IoU unchanged, clean roads falsely flagged 8/50 -> 2/50
(`checkpoints/semantic_gate_report.json`). It cannot add pixels the segmenter
missed, so water-filled cavities remain under-detected; that needs a stronger
segmenter or pixel labels for such cavities. Cost: about 1.5 s per image.

## Previous: website brought in line with the models

- Homepage, architecture, data-lineage and model-card pages described the
  previous configuration (ResNet-50 + logistic head, simulated IMU, 2-3 rare-class
  test images). They now describe what is served: MobileNetV2 with flip
  averaging and a cross-validated soft-voting ensemble, real IMU drive logs,
  7-8 rare-class test images, and the segmenter's clean-road false-alarm rate.
- The stale ResNet-50 head and its report are removed. They came from an older
  data snapshot and a scikit-learn 1.9 pickle, and because neither head report
  is flagged active, the homepage could headline the old 89.2% instead of the
  served 88.8%. Retrain it after fetching its backbone if you want that path.
- `.vercelignore` excludes model binaries and local run outputs; the Vercel
  handler reads only the JSON reports. `.gitignore` excludes local datasets,
  YOLO run folders and weights, and `.env*.local`.

## Previous: reproducible retrain, thirteen fixes, honest model selection, API under test

Every shipped checkpoint was retrained under scikit-learn 1.8.0 from a clean
data fetch, the test suite grew from 78 to 140 tests, and the REST API went
from 0% to tested. Numbers below are what the training scripts printed.

### Models retrained

| Model | Held-out result | Notes |
|---|---|---|
| MobileNetV2 + flip TTA -> soft-voting ensemble | **88.8% accuracy, macro-F1 0.747** | chosen by grouped CV over 13 candidates; test scored once |
| same split, hand-crafted baseline | 79.9%, macro-F1 0.520 | |
| HOG/LBP/colour -> SVM (own split) | 86.2% | 807 held-out photographs |
| IMU shock classifier | 87.2% | real Indian-road drive logs, time-block split |
| Defect segmenter | crack IoU 0.231, pothole IoU 0.144; clean-road false blobs 23.3% | shipped: 0.228 / 0.126; pothole precision 0.186 -> 0.242 |
| PCI / deterioration / depth | R² 0.94 / 0.98 / 1.00 | **surrogate fits to engineering formulas, not field accuracy** |

**Classifier: selected honestly, not a measured gain.** The trainer used to fit
three heads and keep whichever scored best on the test set. It now selects by
5-fold cross-validation grouped by photograph inside the training split. The
winner - a soft-voting ensemble of SVC, logistic regression and an MLP on
mirror-averaged embeddings (CV 84.9%, macro-F1 0.769) - beat every single head
in CV and was then scored on the test set once. Its 88.8% sits within two
images of the earlier 89.2%, so this is a more trustworthy number rather than a
higher one. The previous committed head (90.1% on a different split) cannot be
compared fairly: scored on this split its rare-class recall of 96.9% shows most
of those test photographs were in its training data. Flip TTA doubles the
embedding cost to about 30 ms.

**Segmenter: two calibration bugs fixed, false alarms back under the gate.**
The first retrain failed two regression tests (28.3% of clean photographs with a
false blob against a 25% gate; 3 of 5 zebra crossings reported). Swapping models
one at a time showed neither new model failed alone - only the pair. The cause
was in threshold calibration (fixes 11 and 12 below). After both fixes the
pothole threshold is 0.30 rather than 0.20, false blobs fall to
23.3%, pothole IoU and precision both rise, and every guard passes.

The ResNet-50 head was not retrained (its backbone was unreachable from the
build machine) and still carries a scikit-learn 1.9 pickle; it stays dormant
unless the backbone is fetched, in which case retrain it first. DAN-DAG was not
retrained (needs PyTorch, now in `requirements-train.txt`); it stores plain
NumPy arrays and is unaffected by the version issue.

**Surrogate models.** The depth regressor's targets are computed from its own
inputs, so R² = 1.00 means it reproduced a formula. PCI and deterioration are
likewise fitted to ASTM D6433 / HDM-4 curves. Never quote these R² as accuracy.

### Bugs fixed

1. **`train_cnn_head --compare` always crashed** before saving. Training sub-crops
   made the label vector longer than the baseline's feature matrix (4,006 vs
   2,953). The documented reproduce command and stage 5 of `run_full_pipeline`
   could never produce a model. Labels for the baseline now come from the items.
2. **Work-order IDs collided** — five orders on one corridor in one second all
   got the same ID. IDs now carry a random suffix.
3. **`/dispatch/verify-seal` crashed** on a non-object `work_order`; now 400.
4. **Incident reports invented a location** (Delhi) when no GPS was supplied;
   now `gps_coordinates: null`.
5. **`/incidents/alpr` fabricated an incident** from an empty request (demo
   track + Bengaluru GPS, then sealed it). A real track is now required.
6. **The video tracker counted every non-pothole hazard as a crack.** Waterlogging,
   markings and signs now go to `total_unique_other_hazards_counted`.
7. **A speed reading was labelled "reckless lane cutting"**; it is now
   `OVERSPEEDING`. Lane cutting is only inferred from a track.
8. **Fresh edge export crashed**: the exporter read `PCI.SEVERITY_CAP`, removed
   when the PCI engine moved to deduct curves. It now exports the curves.
   `/models/registry` hid this by reusing old files already on disk.

9. **The claims registry described a model that was not running.** M1 was always
   filled from the ResNet-50 report, but the server serves MobileNetV2 whenever
   the ResNet-50 backbone is absent - which it is in this repository. M1 now
   follows the head the loader would actually pick. Its PCA-variance figure was
   also hand-carried (45.1%) while the retrained model keeps 41.5%; it is now
   measured from the fitted model on every refresh.
10. **The robustness check's level was misleading.** It samples up to 12 images
    per class, so the rare classes are half the set and the figure (47.6%) is
    class-balanced accuracy, not comparable to headline accuracy. The output
    and report now say so; read the deltas under degradation.

11. **Segmenter calibration never saw a clean road.** It was passed 240 defect
    photographs followed by the clean ones and truncated to the first 60, so
    the clean-road photographs that exist to stop zebra-crossing false alarms
    never influenced the operating point. It now samples 60 defect and 30 clean.
12. **Threshold near-ties went to the loosest option.** The pothole IoU curve is
    nearly flat (0.070 at both 0.20 and 0.30), and the first maximum on an
    ascending grid always won. Within 3% of the peak, the strictest threshold
    now wins: when the objective cannot tell them apart, fewer false defects is
    the better answer.
13. **The classifier head was chosen on the test set.** Now chosen by grouped
    cross-validation on the training split; the test set is scored once.

### Dependencies

- `scikit-learn>=1.8.0,<1.9` — the checkpoints had been pickled under 1.9.0
  while this file capped the version below 1.9, so every load crossed a
  version. All retrained checkpoints now load with zero version warnings.
- `onnxruntime` added. The CNN head was trained on ONNX Runtime embeddings;
  without it the embedder silently falls back to cv2.dnn.
- `requirements-train.txt` for the optional PyTorch trainers.

### Tests

| File | Tests | Covers |
|---|---|---|
| `tests/test_api_server.py` (new) | 34 | every page and frontend endpoint, tamper seal, path traversal, CORS, malformed input, video probe + ingest, edge export, Vercel entrypoint |
| `tests/test_untested_modules.py` (new) | 28 | ALPR, video tracker, augmentation, late fusion, ADAS policy, CAN encoding, edge exporter |
| existing three files | 78 | unchanged |

The API tests run the real server on an ephemeral port against a temporary
ledger. `ROAD_SHIELD_WRITABLE_DIR` (new) points the server's writable state at
any directory; the tests use it so they never touch `checkpoints/road_shield.db`.

---

## Previous: deep CNN embeddings replace hand-engineered features

The classifier's input used to be HOG gradients, local binary patterns and
colour histograms — 4,419 numbers written by hand. It is now the output of a
ResNet-50 trained on ImageNet's 1.28 million images, run frozen through ONNX
Runtime, with a class-balanced logistic head learning the mapping onto the seven
road classes. This is ordinary transfer learning; the only unusual choice is
ONNX rather than PyTorch, made because PyTorch's DLLs will not load on the
demo laptop and ONNX Runtime installs everywhere.

Measured on the identical grouped split (243 images from 205 photographs the
model never saw, scored once):

| Features | Accuracy | macro-F1 |
|---|---|---|
| ResNet-50 embeddings → logistic | **88.5%** | 0.846 |
| ResNet-50 embeddings → SVC-RBF | 90.5% | 0.813 |
| MobileNetV2 embeddings → logistic | 83.5% | 0.868 |
| HOG + LBP + colour → SVM (previous) | 87.7% | 0.840 |

Selection is by the mean of accuracy and macro-F1. SVC-RBF has the highest raw
accuracy but loses two of the rare classes entirely; macro-F1 alone swings by
0.1 on a single image when four classes have two or three test examples. The
mean is the compromise, and it is stated here rather than buried, because either
metric alone could have been quoted to flatter a different model.

Per class, the change fixed exactly what was broken: Damaged Traffic Sign F1
0.00 → 1.00, Missing Road Divider 0.44 → 0.80, Normal Road 0.80 → 0.98.

New files: `models/cnn_embedder.py`, `training/train_cnn_head.py`,
`scripts/fetch_cnn_backbone.py`. `models/deep_vision_net.py` gained
`CNNHeadClassifier`, and `load_best_vision_model` now prefers it over the
baseline when a backbone is on disk.

A head is only ever paired with the backbone it was trained on. ResNet-50 and
MobileNetV2 embeddings are unrelated vector spaces, so feeding one head the
other's vectors would produce confident nonsense with no error; the loader skips
any head whose backbone file is absent rather than substituting what it finds.

The ResNet-50 file is 98 MB and is not in version control — GitHub rejects files
that size, and weights do not belong in git. `python -m scripts.fetch_cnn_backbone`
downloads it from the official ONNX Model Zoo. MobileNetV2 (14 MB) is committed,
so a fresh clone has a working CNN path with no download at all.

The rewrite covers `models/`, `pipeline/`, `api/`, `services/`,
`training/`, `data/`, plus the trained models in `checkpoints/`,
`requirements.txt` and the Vercel config.
`tests/`, `datasets/*.py` scripts and `README.md` were intentionally left
untouched. The frontend was only changed to stop hard-coding the local
API address (see "Vercel deployment").

## What changed and why

The previous version of this codebase had several places where the
"AI models" were either untrained NumPy arrays of random weights, or
where a "sensor fusion" step was silently synthesizing one sensor's
reading from another model's own output, or where a demo path
returned data made up on the spot instead of the model's real
output. This pass replaces every one of those with something that
actually computes what it claims to compute, and — where a claimed
capability genuinely doesn't exist yet (e.g. a real dashcam video
dataset) — reports that honestly instead of faking it.

**models/**
- `forensic_audit_engine.py` — the old "deep metric embedder" duplicate
  detector (random untrained weights) is replaced by a real DCT-based
  perceptual hash (`ForensicDuplicateHasher`). SSIM/Laplacian-variance
  texture checks now prefer `skimage`/`opencv`'s real implementations,
  with a manual fallback if those aren't installed.
- `multimodal_transformer_fusion.py` — the old "5-modality cross-attention
  transformer" fabricated 3 of its 5 claimed sensor inputs (no LiDAR,
  CAN-bus, or environment-vector data exists in this project). Replaced
  with an honest weighted late-fusion of the two modalities this project
  actually has trained models for: vision + IMU.
- `edge_model_exporter.py` — used to dump the untrained fake network's
  random weight arrays as a "neural spec". Now introspects the real
  fitted scikit-learn pipelines (PCA/SVC/RandomForest) and only
  hand-ports the genuinely deterministic formula engines (PCI,
  deterioration) to C — it does not pretend to export an SVM to C.
- `alpr_incident_tracker.py` — license-plate text used to be randomly
  generated. Now does real classical-CV plate localization (Canny +
  contour filtering) and real Tesseract OCR, honestly reporting
  `detected: False` when there's no image or no plate found.
- `morth_dispatch_agent.py`, `realworld_video_tracker.py` — logic was
  already real; only docstrings were cleaned up.

**pipeline/**
- `deep_inference_pipeline.py` — rewritten end-to-end. The most
  important fix: when no real IMU telemetry is supplied for a frame,
  the pipeline now honestly reports `available: False` and falls back
  to a neutral prior, instead of the old behavior of fabricating an
  IMU reading from the vision model's own prediction (which made the
  "dual-sensor Bayesian fusion" claim circular and meaningless).
- `fleet_deduplication_engine.py` — fabricated fallback address/
  elevation values replaced with honest `None` + a `geocode_source`
  field; added a real `get_deduplication_stats()` computed from actual
  ingested/registered counts instead of a hardcoded percentage.
- Removed `pipeline/google_maps_service.py`, a byte-identical unused
  duplicate of `services/google_maps_service.py`.

**data/**
- `dataset_generator.py`, `benchmark_dataset_hub.py`,
  `realworld_media_engine.py` — these used to fabricate entire
  datasets (Gaussian noise dressed up as "RDD2022", "Kaggle
  Pothole-600", etc., with invented sample counts) and a curated video
  catalog that doesn't exist on disk. Now they report real counts read
  straight from `datasets/`, and honestly report that no dashcam video
  files are present (`datasets/08_dashcam_video_streams` is empty)
  instead of returning fabricated clip metadata.

**training/**
- `mega_pipeline.py` — used to run a fake NumPy training loop over the
  fabricated datasets above and write out an invented "model zoo"
  with made-up accuracy numbers. Now it calls this project's two real
  training scripts (`train_vision.py`, `train_imu.py`) and reports
  their genuine held-out accuracy, confusion matrix, and per-class
  precision/recall/F1.
- Removed five training scripts that were fully superseded and, after
  the `data/` rewrite, broken (`ImportError` on load): `train_all.py`,
  `train_deep_suite.py`, `train_deep_vision_suite.py`,
  `train_forensic_embedder.py`, `train_realworld_vision.py`. They
  claimed to train on "100,000+ samples" across benchmarks that were
  never actually in this repo.
- `train_mega_suite.py` — rewritten as a small, working CLI entrypoint
  (`python3 -m training.train_mega_suite`) that runs the real training
  suite and prints/saves the genuine results.

**api/ and services/**
- `server.py` — rewritten endpoint-by-endpoint. Removed the
  `preferred_class` parameter that let a caller force the vision
  classifier's answer (the "classifier-rigging" cheat), random-noise
  image generation standing in for real forensic photos, a fake IMU
  generator, and several hardcoded fake payloads (ledger list, fleet
  stats, model registry). All endpoints now call into the real,
  rewritten models above. Every endpoint path is unchanged, for
  frontend and `tests/test_api_server.py` compatibility.
- `google_maps_service.py` — rewritten. An earlier version of these
  notes called this file's fallbacks genuine; that was wrong. When no
  Google key was set it invented elevations from a sine/cosine formula,
  returned the same four made-up "nearby facilities" (hospital, depot,
  police post) at fixed offsets from any point, answered unknown searches
  with Silk Board, drew fake curved routes with invented turn
  instructions, and labelled every address outside its table as
  "Bengaluru". Now each lookup uses a real source (Google → OpenStreetMap
  Nominatim / OSRM / Overpass → Open-Meteo elevation) and reports
  `UNAVAILABLE`, or a clearly labelled straight-line / city-level
  estimate, when none answers. Pothole avoidance now picks, among the
  router's real alternative routes, the one passing the fewest known
  defects instead of nudging line coordinates off the road. Drainage risk
  is a documented local-relief heuristic (point vs. ring 250 m away).

## Vercel deployment

- `vercel.json`: removed the 15 MB `maxLambdaSize` cap, which the real
  scikit-learn / SciPy / scikit-image / OpenCV stack cannot fit under.
  Python functions on Vercel may be up to 500 MB uncompressed; this stack
  is roughly 400 MB, so it should fit, but it is close to the limit.
- `.vercelignore`: training datasets are still excluded, except the small
  IMU validation split so `/api/v1/telemetry/imu` can serve real windows.
- On Vercel the deployed files are read-only, so the server writes runtime
  files (feedback log, exported specs) to `/tmp`, `/api/v1/training/launch`
  returns a clear 503 (train locally, commit `checkpoints/`, redeploy), and
  endpoints that need the photo datasets return 503 instead of crashing.
- Startup no longer calls external geocoding services (demo defects are
  geocoded lazily on first map load), so cold starts don't wait on them.
- Frontend: 9 API calls that were hard-coded to `http://127.0.0.1:8000`
  now use the page's own origin, so the dashboard talks to the Vercel
  backend when served from Vercel (and still to the local server locally).

## Honest limitations, stated plainly

- The vision classifier's held-out accuracy is currently ~37% on 7
  classes (vs. a 14% random-guess baseline). That's real, and it's
  low mainly because several classes have very few distinct source
  photos (see the `dataset_inventory` returned by
  `/api/v1/datasets/benchmarks`) — a data problem, not a hidden one.
  More labeled photos per class is the fix, not more code.
- The IMU shock classifier scores 100% on its held-out split, but its
  data is the simulated 100Hz accelerometer windows shipped in this
  repo, not field-collected vehicle logs — treat it as validated
  against the simulator until retrained on real sensor data.
- The ASTM D6433 PCI "deduct value" curve is a documented analytic
  stand-in shaped like the real published curves, not a pixel-accurate
  digitization of them (the real curves are charts, not equations).
  The surrounding iterative-correction algorithm is the real ASTM
  procedure.

## What will change in `tests/`

`tests/` was left untouched per scope, but several test files
exercise the exact fabricated behaviors this pass removed (e.g. the
old `preferred_class` cheat, `generate_forensic_triplets`,
`data.massive_dataset_generator`, `PCIRegressorNet`'s old fake-weight
interface). Those specific tests will now fail or error on import —
that's expected, since they were testing behavior that no longer
exists on purpose. Everything under `tests/test_api_server.py` that
hits real endpoint paths with real payloads should still pass, since
every endpoint path was preserved.
