# Milestone 2 report → implementation

Every technical claim in *CSET485 ROAD-SHIELD Milestone 2 Full Report*, checked
against this repository. Each row names the code that implements it, the test
that pins it, and the number measured on held-out data. Where the report and
the measurement disagree, the measurement wins and the correction is listed at
the end.

Status: **Done** = implemented, tested, measured. **Partial** = implemented
with a stated gap. **Not yet** = still future work.

## The eleven-stage pipeline (report §8)

| # | Report stage | Status | Implementation | Evidence |
|---|---|---|---|---|
| 1 | Frame decoding | Done | `api/request_images.py` (`load_rgb`: native resolution, EXIF orientation, size and pixel caps) | `tests/test_api_security_and_perception.py` |
| 2 | Asphalt-texture gatekeeper | Done, re-calibrated | `models/frame_gate.py`; `scripts/calibrate_frame_gate.py` | Keeps 100% of 450 real frames; catches 100% covered-lens / severe defocus, 98% whiteout, 89% moderate defocus. See correction C4. |
| 3 | Defect proposal regions | Done | Road-damage YOLO (`configs/detectors/road_damage.yaml`) plus the existing segmenter | Model card `checkpoints/detectors/road_damage.json` |
| 4 | Distress classification | Done | `models/deep_vision_net.py` (CNN embeddings + head) | 7 classes, see C1 |
| 5 | IPM metric sizing | Done | `models/ipm_homography_engine.py`, `models/camera_calibration.py` | `tests/test_geometry_and_storage.py` |
| 6 | 100 Hz IMU window | Done | `models/imu_shock_classifier.py` | `tests/test_road_shield.py` |
| 7 | Bayesian dual-sensor gate | Done | `models/bayesian_fusion_gate.py` | `tests/test_road_shield.py` |
| 8 | ASTM D6433 PCI | Done | `models/pci_regressor_net.py` | `tests/test_road_shield.py` |
| 9 | 180-day deterioration forecast | Done | `models/pavement_deterioration_forecaster.py` | existing tests |
| 10 | MoRTH Section 500 costing | Done | `models/ipm_homography_engine.py` (`estimate_repair_materials`) | existing tests |
| 11 | SHA-256 work-order seal | Done | `models/morth_dispatch_agent.py` | `tests/test_road_shield.py` (WorkOrderSealing) |

## Other claims

| Report claim | Status | Implementation | Measured |
|---|---|---|---|
| Pothole / crack **detection** on Indian roads (RDD2022 India, Milestone 3 item 1) | Done, weak | `scripts/fetch_rdd2022.py`, `configs/detectors/road_damage.yaml` | Test mAP50 0.245 (alligator 0.543, pothole 0.244, longitudinal 0.192, transverse 0.000). Pothole at image level: precision 0.43, recall 0.50; 16% alarms on clean frames. CPU-trained, 16 epochs at 512 px; validation mAP50 still rising, so under-trained. |
| Zebra crossing detection | Done | `configs/detectors/crosswalk.yaml`, `scripts/prepare_crosswalk_dataset.py` | Test mAP50 0.869. Finds a crossing in 75.4% of crossing frames with 0 false alarms on 263 crossing-free frames, vs 23.5% recall and 6 false alarms for the old geometric detector. |
| People, cars, buses, trucks, two-wheelers, signals | Done | COCO YOLO via `models/road_scene_perception.py` | Published COCO weights; no project test split |
| Faces **and plates** blurred before transmission (DPDP Act 2023) | Partial | `redact()` in `models/road_scene_perception.py`; plate detector `configs/detectors/license_plate.yaml` | People: always blurred. Plates: blurred once the plate model is trained (Colab notebook §7b); until then the output states that plates were *not* redacted. Blurs whole person boxes, not faces specifically, and only those the detector finds (one of five pedestrians was missed in a demo frame). |
| Edge transmits a JSON packet < 1 KB, never video | Done | `pipeline/edge_event.py`, `/api/v1/edge/encode`, `/api/v1/edge/verify` | ≤ 1,024 bytes by construction; HMAC-SHA256 per device; people sent as counts only. `tests/test_edge_event.py` |
| 4th-order Butterworth 0.5–25 Hz IMU filter | Done | `bandpass_windows` step in the saved IMU pipeline | Held-out: 80.5% clean, 78.7% with 35 Hz engine vibration (60.1% without the filter). See C3. |
| Haversine deduplication within 8 m | Done | `pipeline/fleet_deduplication_engine.py` | The server used 10 m; now 8 m, with a 7 m / 9 m boundary test |
| Priority Index PI = w1(100−PCI) + w2·Vol + w3·Traffic (report §3.5) | Done | `models/priority_index.py`; ledger ordered by it | Fairness test: locality metadata cannot change a score |
| Traffic counting / congestion (UrbanTrafficNet) | Done, redefined | `counts_from_detections` + IRC:106 PCU formula | Counts come from the detector or the caller; see C5 |
| SSIM repair verification | Done | `models/forensic_audit_engine.py` | existing tests |
| INT8 quantization (Milestone 3 item 3) | Done on CPU | `scripts/quantize_models.py` | Crossing detector: mAP50 0.864 → 0.858, 61 → 48 ms, 10.0 → 4.1 MB (detection head kept FP32; quantizing it collapsed mAP50 to 0.100). Damage detector: 0.234 → 0.222, 86 → 75 ms, 10.1 → 4.2 MB. Classifier backbone: 1.9× faster, 3.6× smaller, −2.6 accuracy points (FP32 vs INT8 on identical images; the image set differs from the published held-out split after the DNIT crops were regenerated, so only the difference is comparable). Raspberry Pi / Jetson benchmarks not done. |
| Real bus sensor mounting (Milestone 3 item 2) | Not yet | — | Needs hardware |
| Live command-centre dashboard (Milestone 3 item 4) | Partial | `web/corridor.html`, `web/detect.html`, ranked `/api/v1/ledger/defects` | No live bus telemetry feed |

## Corrections to the report

| # | Report says | Measured / actual | Why |
|---|---|---|---|
| C1 | "spot 9 classes of road defects" | The classifier has **7** classes (`VisionDistressNet.CLASS_NAMES`); the new detectors add 4 damage classes, 2 marking classes and COCO traffic classes | Count from code |
| C2 | "all 10 internal pipeline tests passed"; "10-Subsystem Integration Harness 10/10 PASS" | The suite has **178 unittest cases** run by CI on Python 3.11 and 3.12 | The 10/10 harness no longer exists; cite the CI suite |
| C3 | IMU "simulated using spring-mass-damper dynamics", 15,000 windows (the README added "100%") | **852 real field-recorded windows** (VishalSingh25/Pothole-Project, Indian roads), 80.5% held-out | The data was replaced with real logs; the old figure no longer applies |
| C4 | Gatekeeper threshold "Laplacian variance below 42.5" | A fixed 42.5 would **drop 10.4% of real road frames**; the shipped gate uses 5.0 plus luminance and glare checks, and keeps 100% | Texture varies ten-fold between cameras (median 1,192 on CDSet, 127 on project photos) |
| C5 | "UrbanTrafficNet scored 90.36% validation accuracy across 7 classes" | No such trained network exists in the code; traffic counts now come from the COCO detector | The figure cannot be reproduced; drop it |
| C6 | "Faces and civilian license plates are blurred" (§1, §6.4, §7.5) | People are blurred; plates only once the plate detector is trained; no separate face detector | See the privacy row above |
| C7 | "completing an entire inference cycle in 76 ms" | Measured on an i5-1145G7 laptop CPU on AC power, 1280×720 frame, median of 15, quality gate included: COCO traffic 203 ms, road damage 115 ms, crossings 76 ms, **all three 391 ms**; CNN classifier backbone 21 ms. Only the crossing detector alone is about 76 ms | `scripts/measure_latency.py` → `checkpoints/perception_latency.json` |
| C8 | "camera height (2.45 m)" | Calibration profiles use 1.45–1.52 m | `models/camera_calibration.py`, `checkpoints/calibration/` |
| C9 | "ResNet-50 feature backbone" as the served model | A fresh clone serves **MobileNetV2** (90.1% on its held-out split); ResNet-50 (89.2%) needs `scripts/fetch_cnn_backbone` | ResNet-50 weights are not in git |
| C10 | RDD2022 India "1,000+ frames", "47,000 photos" to ingest | India subset: **7,706 labelled images** used (5,368 / 1,172 / 1,166); RDD2022 has 47,420 in all six countries | `scripts/fetch_rdd2022.py` |
| C11 | Deduplication "within 8 meters" | Was 10 m in the running server; now 8 m | Fixed in code |

## Reproducing every number

```bash
pip install -r requirements.txt -r requirements-train.txt
python -m unittest discover -s tests                 # the full suite
python -m training.train_imu                         # IMU, incl. the vibration test
python -m scripts.calibrate_frame_gate               # gate thresholds
python -m scripts.benchmark_crosswalk                # learned vs geometric crossings
python -m scripts.quantize_models                    # INT8 cost
# detectors: notebooks/train_detectors_colab.ipynb, or training/train_detector.py locally
```
