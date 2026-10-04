# ROAD-SHIELD — Milestone 2 claims vs. the repository (audit, 3 Oct 2026)

Every row was checked against the code, the checkpoints and the reports in this
repository. "Fixed" means changed in this audit; the report
(`CSET485_ROAD_SHIELD_Milestone2_Report_Corrected_TrackedChanges.docx`) carries
the matching tracked changes under the author "Claude (audit 2026-10-03)".

## Completed and working

| Area | Status | Evidence |
|---|---|---|
| 7-class road-condition classifier — **served: fine-tuned MobileNetV3-Large, 93.3% test, 92.3% on held-out Indian roads, 7.2 ms/image on CPU** | Done; chosen on validation over the frozen head (87.9%) and three other fine-tuned networks | `checkpoints/finetune_summary.json`, `checkpoints/vision_model_selection.json` |
| End-to-end fine-tuned CNNs (EfficientNet-B0 91.3%, B2 91.4%, MobileNetV3-L 91.4%, ResNet-50 89.5% before refit) | Done in this audit (Colab T4), same grouped split | `training/train_finetune_cnn.py`, `checkpoints/finetune_*_report.json` |
| Served-model choice by a rule fixed before reading test scores | Done in this audit | `scripts/select_vision_model.py`, `checkpoints/vision_model_selection.json` |
| RDD2022 India: training crops + Indian-roads test split | Done (official CRDDC archive converter added) | `scripts/prepare_rdd2022_voc.py`, `scripts/ingest_rdd2022_india.py`, `checkpoints/indian_roads_eval_report.json` |
| Pixel defect segmenter (11 features, gradient boosting) + CNN semantic gate | Done | `checkpoints/defect_segmenter_report.json`, `checkpoints/semantic_gate_report.json` |
| Ground-plane area (pinhole camera, per-vehicle calibration profiles) | Done | `models/ipm_homography_engine.py`, `models/camera_calibration.py` |
| Depth as an interval estimate (never called a measurement) | Done | `models/depth_estimator.py` |
| IMU shock classifier on real drive logs — **served: RandomForest, 87.2% held-out** | Done; the 1-D CNN lost a time-blocked CV comparison (the first, shuffled-window CV leaked and was corrected) | `checkpoints/imu_deep_report.json`, `checkpoints/imu_model_selection.json` |
| Bayesian vision + IMU fusion gate | Done (only fires when a real IMU window is supplied) | `models/bayesian_fusion_gate.py` |
| ASTM D6433 PCI engine | Done - reproduces the deduct curves (R² 0.941), not validated against field surveys | `checkpoints/pci_model_report.json` |
| Deterioration forecast 30/60/90/180 days | Done - fitted to an HDM-4-style formula, not to field data | `checkpoints/deterioration_model_report.json` |
| MoRTH Section 500 tonnage and cost | Done; one material table shared by audit and work order (fixed) | `models/ipm_homography_engine.py`, `models/morth_dispatch_agent.py` |
| SHA-256 sealed work orders + tamper check | Done | `/api/v1/dispatch/verify-seal` |
| Haversine fleet deduplication, 8 m | Done; now merges only the same defect class (fixed) | `pipeline/fleet_deduplication_engine.py` |
| Privacy redaction of people and number plates | Added in this audit (recall not measured) | `models/privacy_redactor.py`, `/api/v1/privacy/redact` |
| REST API + site (11 pages incl. Design and API reference) + Vercel static deployment | Done | `api/server.py`, `web/`, `api/vercel_app.py` |
| System design, OpenAPI contract, model registry, request IDs, opt-in API key, docker compose | Done in this audit | `docs/SYSTEM_DESIGN.md`, `api/openapi.py`, `/api/v1/models/served` |
| U-Net segmenter and YOLOv8 RDD2022 detector | Trainers, ONNX serving and selection rules done; served only if trained and the rule picks them | `training/train_unet_segmenter.py`, `training/train_rdd_detector.py` |
| Automated tests | 200+; pipeline suite OK on a fresh clone with scikit-learn 1.8 and the fine-tuned CNN served | `tests/`, `logs/tests_after.txt` |

## Left to do (not claimed as done)

| Item | Why it is not done |
|---|---|
| Hardware on a real bus (dashcam + MPU-6050) | No vehicle trial yet; all images are public datasets, IMU data is from car drive logs |
| Detection recall for the privacy redactor | No annotated face / plate set in the project |
| Priority Index `PI = w1(100-PCI) + w2 Vol + w3 Traffic` | Proposed only; weights not set, not implemented |
| Encrypted transmission | Work orders are tamper-evident (SHA-256), not encrypted; TLS belongs to a deployment |
| INT8 benchmark on Raspberry Pi 5 / Jetson | INT8 ONNX files exist; never timed on that hardware |
| Field validation of PCI / deterioration / depth | Needs field surveys or depth ground truth |
| More photographs for the four municipal classes | ~50 each; test scores rest on 7-8 images |
| Pixel labels for water-filled potholes | Segmenter under-detects them; no labels exist |
| Live GIS feed from buses | No fleet; the ledger fills only from API calls (demo rows are opt-in) |

## False or hard-coded items found and what was done

| Found | Fix |
|---|---|
| Report: "We mounted a dashcam and MPU-6050 on city buses" | Corrected - designed for buses, not yet installed |
| Report: ResNet-50 backbone, 9 classes | Corrected - 7 classes; served model named from the selection file |
| Report: Laplacian-variance gate at 42.5 | Corrected - the code uses luminance std < 6.5 |
| Report: camera height 2.45 m, 3x3 bird's-eye warp | Corrected - 1.45 m default / 1.52 m bus profile; per-pixel ground projection |
| Report: 4th-order Butterworth 0.5-25 Hz | Corrected - no filter exists; per-window mean removal + FFT band energies |
| Report: faces/plates blurred, encrypted packets | Redactor implemented; encryption claim corrected |
| Report: CLAHE on CRACK500, concrete set "down-weighted" | Corrected - no CRACK500 data; concrete patches excluded |
| Report: urban hazard set 67 images | Corrected - 207 (67 field + 140 Wikimedia) |
| Report: "5,000 deaths every year" | Corrected - 5,626 over 2018-2020 |
| Report: 4 literature citations with wrong venue / author / figure | Corrected (Koch & Brilakis, Mednis, Bhatia, Depth Anything V2); two unverifiable rows replaced with verifiable papers |
| Report: 30-image benchmark "1.492 t bitumen, 100% seal verification" | Re-measured; see the report's Section 10 |
| `datasets/README.md`: 73,060 samples, 15,000 IMU sequences, files that do not exist | Rewritten from the folders on disk |
| README legacy half: 90.36% traffic net, R² 0.9908, "10/10 PASS guaranteed" | Removed |
| API: missing GPS became Delhi (28.7041, 77.1025) or Bengaluru (12.9716, 77.5946) | Now null / 400 |
| API: work order sealed for an invented 2.2 m², PCI 42 pothole when fields were missing | Now 400 with the missing field list |
| `web/works.html` sent `defect_class` / `pci`, which the server ignored, and displayed form values instead of the sealed order | Server accepts the names; page renders the sealed fields |
| Deep-audit with no image silently analysed a bundled pothole photo | Now 400; `image_path` restricted to `datasets/` |
| Five invented bus reports seeded into the ledger on every fresh start | Opt-in via `ROAD_SHIELD_SEED_DEMO=1` |
| Rule decisions reported `confidence: 0.99` | Now `null` with a `decision_basis` |
| DAN-DAG agreement raised confidence by `max(p, 0.55p + 0.45q + 0.03)` | Removed; agreement reported separately |
| ALPR kinematics "confidence" = made-up formula with a 0.75 floor | Replaced by `threshold_ratio` |
| Video tracker defaulted missing confidences to 0.9 | Now 0 |
| `process_batch` invented chainage 100 km + 250 m per file | Removed |
| Dedup merged a sign and a pothole 5 m apart | Same-class only |
| Work order and pipeline kept separate copies of densities / rates | One table, with a rate-basis note |
| IMU classes named "Expansion Joint" / "Rumble Strip" | Renamed to what the logs contain: unmarked / marked speed breakers |
| IMU CNN chosen by cross-validation over shuffled 1-second windows (neighbours leaked across folds) | Time-blocked folds; the fair comparison serves the RandomForest |
| Semantic gate silently disabled when the fine-tuned CNN was served (zebra crossings reported as potholes) | Gate runs with either CNN classifier |
| With the segmenter unloadable, pothole photos read as "Normal Road" with no explanation | Whole-frame classifier opinion shown beside the result; the note names the cause |
