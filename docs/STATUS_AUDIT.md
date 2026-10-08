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
| U-Net segmenter (ResNet-18 encoder) | Trained three times; not served. Run 1 (DNIT only): 9/24 defects end to end. Run 2 (+ CrackSeg9k, Kaggle outlines): Kaggle pothole IoU 0.679, 18/24 end to end, lost the IoU rule on DNIT cracks. Run 3 (+ Pothole Mix, 905 copies of DNIT check/test photographs removed after our overlap check found them; 7,967 training images from 5 sources): passed the IoU rule (DNIT calibration crack 0.255 vs 0.214, pothole 0.640 vs 0.119), test DNIT crack 0.271 / pothole 0.632, Kaggle pothole 0.617, CrackSeg9k crack 0.431; end to end 18/24 defects with 3/36 false positives vs the pixel classifier's 24/24 and 3/36, so the pixel classifier stays. Pothole Mix's own test still contains some DNIT copies, so its 0.709 is an upper bound | `checkpoints/segmenter_selection.json`, `training/train_unet_multi.py`, `scripts/segmenter_deployment_check.py` |
| YOLOv8 RDD2022 detector | Trainer, ONNX serving and selection rule done; not yet trained | `training/train_rdd_detector.py` |
| Repair Priority Index `PI = w1(100-PCI) + w2 Vol + w3 Traffic` (Milestone 2, section 3.5) | Done (8 Oct): each term on 0-100, weights 0.5/0.2/0.3 adjustable and validated, missing terms re-weighted rather than assumed, traffic basis labelled (measured PCU/day or the fleet's bus-count proxy), Kendall-tau check of how much the order depends on the weights; the corridor page ranks the ledger by it | `models/priority_index.py`, `/api/v1/priority/ranking`, `/api/v1/priority/score` |
| Keyed work-order seal (the SIH submission said HMAC) | Done (8 Oct): HMAC-SHA256 when `ROAD_SHIELD_SEAL_KEY` is set; the algorithm is inside the sealed fields, and a verifier with a key refuses unkeyed seals (no downgrade) | `models/morth_dispatch_agent.py`, `/api/v1/dispatch/verify-seal` |
| Bus agent: camera + MPU-6050 + GPS loop, frame-quality gate, IMU-only fallback when the camera cannot see | Done in software (8 Oct), tested by replay against the real server; not yet run on a Pi with the parts | `edge/bus_agent.py`, `edge/sensors.py`, `edge/frame_quality.py`, `docs/EDGE_AGENT.md` |
| Encrypted packets under 1 KB and offline store-and-forward (SIH: "encrypted SQLite store-and-forward queue") | Done (8 Oct): AES-256-GCM with bus id and sequence authenticated, replay refused by the server, queue encrypted at rest, bounded, in-order delivery with back-off | `edge/crypto.py`, `edge/store_forward.py`, `pipeline/edge_ingest.py` |
| Live command centre (Milestone 3 plan, item 4) | Done (8 Oct): server-sent events for bus positions, defects, IMU-only shocks and work orders; the corridor map moves buses and adds defects as they arrive, with a polling fallback | `/api/v1/live/stream`, `/api/v1/fleet/live`, `web/corridor.html` |
| Street-level road map | Done (8 Oct): streets, satellite with road names, and dark base maps to zoom 22; potholes and cracks as markers coloured by repair priority and sized by area, clustered when zoomed out, a priority heatmap, filters by type and priority band, my-location, full screen, and a details card per defect (PCI, area, depth, reports, address, Google Maps / Street View) | `web/corridor.html` |
| Detect a pothole in a photo, put it on the map | Done (8 Oct): the Inspection page reads the photo's own GPS (EXIF) in the browser, or uses the phone's location, or a spot picked on a map, and adds the detected pothole to the ledger; a public demo asks for the operator key that the tunnel window prints | `web/inspect.html`, `scripts/go_live_tunnel.ps1` |
| Repair lifecycle | Done (8 Oct): an order is issued from a ledger defect with its measured quantities, sealed, stored, and moved through assigned (contractor), in progress and repaired; SLA due time and overdue flag; every step appended to a hash-chained history, so an edited or backdated step is detected. Before this, sealed orders were never saved | `pipeline/works.py`, `/api/v1/works/*`, `web/works.html` |
| Repair verified by the fleet | Done (8 Oct): after "repaired", buses whose track passes within 15 m count as passes; 3 passes over 24 h (configurable) with no new sighting, and a 60 s settle window after the last pass, verify the repair; a new sighting reopens it. Shown end to end by `scripts/demo_city.py` (3 replayed buses: an order went issued → repaired → verified by two bus passes) | `pipeline/works.py`, `tests/test_repair_lifecycle.py` |
| GIS export and alerts | Done (8 Oct): the ledger as GeoJSON or CSV (with priority and repair status; spreadsheet formulas neutralised); alerts for new P1 defects and overdue orders on the live feed and, with `ROAD_SHIELD_ALERT_WEBHOOK`, to Slack / Teams / a ticketing system | `pipeline/export.py`, `pipeline/alerts.py` |
| Fusion-gate jolt thresholds recalibrated | Fixed (8 Oct): the gate treated a 3.5-6 m/s² peak-to-peak jolt as pothole evidence and "under 1 m/s²" as no jolt; on the project's own drive logs plain road exceeds 6 m/s² in 78-96% of 1-second windows, so the optical-false-alarm rule could never fire on real data. The severe-jolt threshold is now 28 m/s² (plain-road 99th percentile 27.7; pothole drives 40.9 at the 90th); below it the IMU classifier's probability decides | `models/bayesian_fusion_gate.py`, `tests/test_edge_and_priority.py` |
| INT8 benchmark tooling (Milestone 3 plan, item 3) | Script done (8 Oct): times every ONNX model on the machine it runs on and builds a calibrated INT8 copy of the served classifier with an agreement check. The Pi 5 / Jetson numbers still need the board | `scripts/benchmark_edge.py` |
| Privacy-redactor recall | Measurement done in code (8 Oct) on WIDER FACE validation and a public licence-plate set; it runs in Colab stage 5 (the sets cannot be downloaded from the build machine). The figure appears in `checkpoints/privacy_redaction_report.json` once that stage has run | `scripts/measure_redactor_recall.py` |
| Classifier confidence calibration | Measurement done in code (8 Oct): ECE, NLL, Brier and a reliability table on the Indian held-out crops, with a temperature fitted on one half and reported on the other; served only if both NLL and ECE improve. Runs in Colab stage 5 | `scripts/measure_calibration.py` |
| Citizen reports | Done (8 Oct): `/report` takes a phone photograph and a location (photo GPS first, else the phone's position); the engine must find a pothole or crack, then the report is pinned on the map as pending. It joins the ledger only when a bus reports the same class within 15 m or an operator promotes it, so one person cannot fill the repair list. The photograph is not stored; senders are a salted hash, 10 reports an hour each | `pipeline/citizen.py`, `web/report.html`, `tests/test_citizen_reports.py` |
| Printable work order | Done (8 Oct): `/order?id=` renders the sealed order on A4 with the live seal check, the hash-chained history and signature lines; the browser's print dialog saves it as PDF | `web/order.html` |
| MLOps: registry, tracking, retraining | Done (8 Oct): every model version is its files by SHA-256 in a blob store, with metrics from its own reports, the training run and the commit; promotion only through a per-model gate (floor + max drop vs production + guard metrics), atomic deploy into checkpoints/, in-place reload, rollback, event log; hand-copied files are never overwritten. `python -m mlops retrain` validates data, trains, registers the candidate, restores production and gates it. Runs tracked in SQLite, mirrored to MLflow when installed | `mlops/`, `docs/MLOPS.md`, `tests/test_mlops.py` |
| Input guard (out-of-distribution model) | Done (8 Oct), trained here: Mahalanobis novelty + not-road classifier on MobileNetV2 embeddings + quality limits. Held out: AUROC 0.9945; 94.2% of everyday photographs refused, 1.25% of road photographs refused (4/321); dark, blurred and overexposed frames all caught. The citizen endpoint refuses not-a-road and unusable photographs | `models/ood_guard.py`, `training/train_ood_guard.py`, `checkpoints/ood_guard_report.json` |
| Production monitoring | Done (8 Oct): per-photograph record (no image, no location); PSI drift with a chi-square check against the training reference, class-mix drift, OOD rate, latency percentiles, daily series; alert on drift; Prometheus gauges; rebaseline from the city's own fleet; shadow test of a staging classifier off the request path | `mlops/monitor.py`, `/mlops` |
| Active learning | Done (8 Oct): uncertain and disagreeing photographs queued after people and plates are blurred (citizen photos never), near-duplicates dropped, labelled on the MLOps page, exported as a class-folder zip with a manifest | `mlops/active_learning.py` |
| Traffic from bus cameras | Done (8 Oct), not calibrated: COCO vehicle counts per 100 m cell, density × bus speed, expanded by an assumed hour profile; feeds the Priority Index once a cell has 6 observations in 3 hours. Needs comparison with manual counts before it is relied on | `pipeline/traffic.py`, `edge/bus_agent.py` |
| CI/CD for models | Done (8 Oct): model-quality gate on committed checkpoints, Docker build + container smoke test, image published to GHCR from master | `.github/workflows/models.yml`, `docker.yml`, `mlops/ci_check.py` |
| Automated tests | 350+; pipeline suite OK on a fresh clone with scikit-learn 1.8 and the fine-tuned CNN served | `tests/`, `logs/tests_after.txt` |

## Left to do (not claimed as done)

| Item | Why it is not done |
|---|---|
| Hardware on a real bus (dashcam + MPU-6050) | No vehicle trial yet; all images are public datasets, IMU data is from car drive logs |
| Redactor recall and calibration numbers | The code is done; the numbers come from Colab stage 5 |
| Priority Index weights chosen by a municipality | The defaults are a stated policy choice, not a consultation result |
| INT8 benchmark on Raspberry Pi 5 / Jetson | `scripts/benchmark_edge.py` is ready; needs the board |
| Bus agent on real hardware | Tested by replay only; MPU-6050 and GPS readers not yet run on a Pi |
| Per-bus keys | One fleet key today; per-bus keys need a key-distribution process |
| Permanent hosting | Laptop tunnel for demos; options in `docs/HOSTING.md` |
| Field validation of PCI / deterioration / depth | Needs field surveys or depth ground truth |
| More photographs for the four municipal classes | ~50 each; test scores rest on 7-8 images |
| Pixel labels for water-filled potholes | Segmenter under-detects them; no labels exist |
| Traffic estimate calibration | Needs classified manual counts (IRC:SP:41) on a few corridors to set the view length and hour profile |
| Live GIS feed from real buses | The feed and the bus agent exist; there is no fleet yet, so the map shows replayed or demo buses |

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
