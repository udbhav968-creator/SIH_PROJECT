# ROAD-SHIELD model card

Generated 2026-10-04 from `checkpoints/claims.json` by `scripts/build_model_card.py`. Regenerate after any retraining; do not edit by hand.

## System at a glance

| | |
|---|---|
| Purpose | Road-defect records from bus-camera frames: class, outline, ground area, depth interval, cost range, sealed work order |
| Intended users | Road authorities and their contractors; inspectors reviewing system reports |
| Decision role | **Recommends; people decide.** No repair is dispatched without a human authority |
| Out of scope | Any decision about a person; legal evidence of fault; roads or cameras unlike the evaluation data without re-validation; depth as a measurement |
| Served segmenter | pixel_classifier (rule: U-Net served only if, on the calibration photographs, its crack IoU AND pothole IoU both exceed the pixel classifier's, and its clean-road false-blob rate is no) |
| Served vibration model | random_forest |

## Vision distress classifier

**Intended use.** Classify a road photograph or region into seven road-condition classes, as the first step of a defect record.

**Model.** mobilenet_v3_large (ImageNet-pretrained, fine-tuned end to end on the road corpus, refit on train+val), ONNX Runtime, flip TTA

**Why this design.** Chosen over the frozen MobileNetV2 + scikit-learn head by a rule fixed before the test set was read: it had to beat the head on both accuracy and macro-F1 on non-test data. fine-tuned mobilenet_v3_large: validation accuracy 0.9182, macro-F1 0.8807; frozen head: grouped-CV accuracy 0.8234, macro-F1 0.7520

**Measured (held-out data).**

| metric | value |
|---|---|
| served model | mobilenet_v3_large_refit |
| held out accuracy | 0.9333 |
| held out macro f1 | 0.8283 |
| test images | 630 |
| test photographs | 567 |
| indian roads | images 1200, accuracy 0.9225, macro_f1 0.9224 |
| onnx cpu ms per image | 7.2 |
| onnx size mb | 16.8 |

**Limits - do not rely on it for this.** Latency is 7.2 ms per image (single image, ONNX Runtime on the 2-thread training machine's CPU, before flip TTA doubles it); it was not measured on a Raspberry Pi or Jetson. The rare municipal classes have 7-8 test images each.

**Alternatives tried.** Frozen-embedding head (kept as the fallback); other fine-tuned architectures lost on validation - see finetune_summary.json.

**Evidence.** `checkpoints/finetune_mobilenet_v3_large_refit_report.json, checkpoints/finetune_summary.json, checkpoints/vision_model_selection.json` · reproduce: `python -m training.train_finetune_cnn && python -m scripts.select_vision_model`

## Defect segmentation

**Intended use.** Outline cracks and potholes pixel by pixel, so area (and from it material and cost) is measured from the defect's own footprint rather than a box.

**Model.** HistGradientBoosting pixel classifier on 11 vectorised features, trained on defect polygons AND clean-road photographs, with hard-negative mining

**Why this design.** A bounding box around a diagonal crack overstates its area by about 13x, and area drives tonnage which drives cost. A per-pixel mask gives the defect's own footprint. It also decides where a defect IS, which is the proposal stage for the whole pipeline.

**Measured (held-out data).**

| metric | value |
|---|---|
| crack iou | 0.2314 |
| pothole iou | 0.1437 |
| crack dice | 0.3758 |
| pothole dice | 0.2513 |
| clean road false blob rate | 0.2333 |
| clean photographs scored | 120 |
| test photographs | 500 |
| training pixels | 3692729 |

**Limits - do not rely on it for this.** IoU of 0.2314 (crack) and 0.1437 (pothole) is a working model, not a solved problem. On held-out clean roads the segmenter alone draws a false blob on 23.3% of 120 photographs. The CNN semantic gate in front of it reduced clean roads with a false blob from 8/50 to 2/50 and raised pothole IoU from 0.1024 to 0.2706 (checkpoints/semantic_gate_report.json, measured with the frozen-embedding classifier). Water-filled potholes are still under-detected.

**Alternatives tried.** A U-Net (ResNet-18 encoder) was trained on DNIT plus public outline datasets (crackseg9k, kaggle_pothole, pothole_mix) and did NOT replace this model: calibration IoU crack 0.255 vs 0.214, pothole 0.640 vs 0.119; clean false-blob rate 0.0241 vs 0.1807; NOT served after the end-to-end check: end to end, U-Net found 18/24 defects with 3/36 false positives; the pixel classifier found 24/24 with 3/36 Held-out test (same 500 photographs): U-Net crack IoU 0.2712, pothole IoU 0.6317; pixel classifier crack 0.2314, pothole 0.1437.

**Evidence.** `checkpoints/defect_segmenter_report.json, checkpoints/segmenter_selection.json` · reproduce: `python -m training.train_segmenter --images 2000`

## IMU shock classifier

**Intended use.** Classify a 1-second accelerometer window (smooth road, speed breakers, pothole impact) to corroborate a visual detection.

**Model.** RandomForest, 300 trees, on 62 statistical and FFT band-energy features per 1-second 100 Hz window

**Why this design.** Suspension vibration is periodic and noisy. RMS, peak-to-peak, crest factor, kurtosis and spectral power separate a sharp impact from a repetitive wave. Tree ensembles are immune to the noise and need no sequence buffer.

**Measured (held-out data).**

| metric | value |
|---|---|
| held out accuracy | 0.872 |
| held out macro f1 | 0.7292 |
| held out windows | 164 |
| cnn compared | served random_forest |

**Limits - do not rely on it for this.** The windows are REAL accelerometer recordings, but from 10 car drive logs published by one GitHub project (VishalSingh25/Pothole-Project), not from a bus fleet. 164 held-out windows; the speed-breaker classes have 11-32 of them, so their scores move with single windows.

**Alternatives tried.** A 1-D residual CNN is trained and compared on the same windows (training/train_imu_deep.py); it is served only if it beats the forest on cross-validation over the training windows.

**Evidence.** `checkpoints/imu_shock_report.json, checkpoints/imu_model_selection.json` · reproduce: `python -m training.train_imu_deep`

## Scene object detector

**Intended use.** Find people, vehicles and traffic furniture in a frame, for privacy blurring and to keep them out of the defect search.

**Model.** YOLOv8n, COCO 80 classes, ONNX Runtime

**Why this design.** Single-stage, 12.8 MB, runs on CPU. Detects that a person is PRESENT and how far away, without storing any identity; the same boxes drive the privacy redactor that blurs heads and plates before sharing.

**Measured (held-out data).**

| metric | value |
|---|---|
| weights size mb | 12.8 |
| classes | 80 |
| trained on | COCO, ~330,000 images (pretrained) |

**Limits - do not rely on it for this.** This is a COCO detector: it has NO pavement class. It does not find potholes. Defect localisation is the segmenter's job.

**Alternatives tried.** Faster R-CNN - two-stage, far too slow on an edge CPU. Facial recognition - rejected outright as invasive.

## Privacy redaction

**Intended use.** Blur heads and number plates before an image is shared.

**Model.** COCO person / vehicle boxes -> head-region and contour-localised plate Gaussian blur (+ Haar frontal-face cascade when installed)

**Why this design.** The report promised it; it did not exist. Uses detections the pipeline already computes.

**Measured (held-out data).**

| metric | value |
|---|---|
| recall measured | False |

**Limits - do not rely on it for this.** No recall figure: there is no annotated face/plate set in this project. A missed face or plate is possible; the API reports which detectors ran.

**Evidence.** `models/privacy_redactor.py, tests/test_integrity_fixes.py PrivacyRedaction`

## Region proposals and false-positive control

**Intended use.** Decide which mask regions become reported defects, trading false alarms against missed defects.

**Model.** candidate regions from connected components of the segmentation mask, filtered by blob size and mean predicted probability; the brightness grid may not raise a crack or pothole when a segmenter is loaded

**Why this design.** The original proposal stage was a 16x24 brightness and gradient grid. On a normally textured road most cells fire, merge, and produce ONE box over the whole carriageway - the giant box users saw. A real pothole occupies under 1% of that box, so a gate expressed as a fraction of the box discarded real potholes while passing zebra crossings. A connected component of the mask is a defect-shaped region by construction: tight box, meaningful area, and the classifier sees the defect rather than fifty square metres of road.

**Measured (held-out data).**

| metric | value |
|---|---|
| false positive rate before | 0.444 |
| detection rate before | 1 |
| before note | the pre-fix build, measured through audit_image() |
| max heuristic box fraction | 0.25 |
| min component fraction | 0.004 |
| min component confidence | 0.45 |
| false positive rate | 0.1111 |
| detection rate | 0.75 |
| clean photographs | 36 |
| defect photographs | 24 |
| measured through | pipeline.deep_inference_pipeline.DeepInferencePipeline.audit_image |

**Limits - do not rely on it for this.** 11.1% false positives is better, not solved - 4 of 36 clean photographs are still reported as defective. Detection is 75.0% of 24 defect photographs. Measured through audit_image() by scripts/measure_detection_quality.py.

**Alternatives tried.** Raising the confidence floor. Measured: floors of 0.45 and 0.60 gave identical results. The classifier is confidently wrong, not hesitantly wrong, because it has seven defect classes and no way to answer 'none of these'. No threshold on a wrong question fixes the answer.

**Evidence.** `checkpoints/proposal_tuning_report.json, checkpoints/detection_quality_report.json` · reproduce: `python -m scripts.tune_proposals && python -m scripts.measure_detection_quality`

## How models are chosen

Every deep model replaces its simpler counterpart only by a rule written before its test set is scored, and the segmenter must also hold up end to end through the full pipeline on photographs from other datasets. Losers stay on disk and are reported. Training data is checked for near-copies of every measurement photograph before training.

- Vision classifier: serve the fine-tuned network only if it beats the frozen head on BOTH accuracy and macro-F1, measured on non-test data (head: grouped 5-fold CV on the fit split; fine-tuned: validation split). Test numbers are recorded, not used.
- Segmenter: U-Net served only if, on the calibration photographs, its crack IoU AND pothole IoU both exceed the pixel classifier's, and its clean-road false-blob rate is no more than 5 points worse. Decided before the test set was scored.
- Vibration: serve the CNN only if its mean 5-fold CV accuracy AND macro-F1 on the training windows beat the RandomForest's; folds are contiguous time blocks of 25 windows (no neighbouring window on both sides); fixed before the held-out windows were scored

## Ethical considerations

- **People.** No identity is stored; heads and plates are blurred before sharing. Blur recall has not been measured, so a missed face or plate is possible.
- **Fairness is geographic.** Roads without bus routes are not inspected, and a model trained in one country degrades in another (33% on Indian roads before Indian data was added). Coverage and per-city accuracy must be reported before any allocation of repair budgets relies on the system.
- **Money.** Costs are ranges from an estimated depth. A work order is a recommendation for a human authority, sealed so it cannot be altered after issue.

## Corrections on record (17)

Claims this project made and later withdrew or corrected, kept visible on the Architecture page:

- **WITHDRAWN** - +/-3.2% metric error on pothole area
- **CORRECTED** - 15.6 ms CPU latency for the ResNet-50 classifier
- **CORRECTED to 41.5%** - PCA retains 56% of variance
- **QUALIFIED** - ASTM D6433 implementation is legally certified
- **QUALIFIED** - Geometry basis h=1.45 m, theta=18.4 deg
- **FIXED (partially), after a wrong fix** - Reported defects are trustworthy without a gate
- **WITHDRAWN** - 27.8% false positives after the segmentation gate
- **CORRECTED (protocol fixed)** - The IMU 1-D CNN beats the RandomForest (5-fold CV)
- **FIXED (implemented)** - Faces and number plates are blurred on the bus before transmission
- **FIXED** - A request without GPS is located in Delhi / Bengaluru
- **FIXED** - A work order is sealed even when its measurements are missing
- **CORRECTED** - Detections carry a model confidence
- **QUALIFIED** - The ledger shows the city's defects
- **CORRECTED** - IMU classes include expansion joints and rumble strips
- **WITHDRAWN** - Laplacian-variance gate (42.5), 4th-order Butterworth filter, CLAHE, 2.45 m camera
- **REVERSED** - The U-Net segmenter serves (pothole IoU 0.641 vs 0.144)
- **REVERSED** - The U-Net segmenter serves (pothole IoU 0.632 vs 0.144) - multi-dataset run
