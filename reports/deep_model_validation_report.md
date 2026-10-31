# ROAD-SHIELD Validation Report
**Timestamp UTC**: 2026-10-31T06:32:43.469669+00:00  
**Status**: **CERTIFIED**  

## 1. Checkpoint Integrity
| Model Name | Checkpoint File | Size (MB) | Status |
| :--- | :--- | :---: | :---: |
| Vision CNN MobileNetV3 (Refit) | `deep_vision_mobilenet_v3_large_refit.onnx` | 16.83 MB | **VERIFIED** |
| Dense Pixel Segmenter (Pixel ONNX) | `defect_segmenter_pixel.onnx` | 1.5 MB | **VERIFIED** |
| DAN-DAG Model Weights | `dan_dag_model.joblib` | 4.69 MB | **VERIFIED** |
| ASTM D6433 PCI Regressor | `pci_model.joblib` | 0.54 MB | **VERIFIED** |
| HDM-4 Deterioration Forecaster | `deterioration_model.joblib` | 1.12 MB | **VERIFIED** |
| Photometric Depth Regressor | `depth_estimator_model.joblib` | 0.91 MB | **VERIFIED** |
| 100 Hz IMU Shock CNN | `imu_shock_model.joblib` | 11.37 MB | **VERIFIED** |
| Out-of-Distribution Input Guard | `ood_guard.npz` | 0.56 MB | **VERIFIED** |
| Claims & Certified Metrics | `claims.json` | 0.03 MB | **VERIFIED** |

## 2. Statistical Quality Gates
| Model Architecture | Evaluated Metric | Quality Gate Floor | Measured Result | Status |
| :--- | :--- | :---: | :---: | :---: |
| Vision CNN (MobileNetV3 Refit) | Test Accuracy | >= 0.85 | **0.9435** | PASSED |
| Vision CNN (MobileNetV3 Refit) | Test Macro-F1 | >= 0.75 | **0.8786** | PASSED |
| Vision CNN (Indian Roads Bench) | Indian Roads Accuracy | >= 0.85 | **0.9225** | PASSED |
| DAN-DAG Dual Attention Network | Mean 21-Node DAG Accuracy | >= 0.90 | **0.9352** | PASSED |
| DAN-DAG Dual Attention Network | Held-Out Test Accuracy | >= 0.80 | **0.8663** | PASSED |
| ASTM D6433 PCI Regressor | Held-Out R2 | >= 0.85 | **0.8839** | PASSED |
| HDM-4 Deterioration Forecaster | Area Expansion R2 | >= 0.80 | **0.9879** | PASSED |
| Photometric Depth Regressor | Pothole Depth R2 | >= 0.50 | **0.6211** | PASSED |
| 100 Hz IMU Shock 1-D CNN | Validation Accuracy | >= 0.70 | **0.8659** | PASSED |
| Out-of-Distribution Input Guard | AUROC Combined | >= 0.90 | **0.9945** | PASSED |

## 3. End-to-End Pavement Scene Tests
| Test Scene | Class | Confidence | Area | Repair Cost | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Scene C: Foreground Crater | Pothole Cavity | 88.3% | 0.7277 m2 | Rs 888.75 | **PASS** |
| Rural Dirt Road | Pothole Cavity | 89.1% | 0.105 m2 | Rs 105.75 | **PASS** |

## 4. Civil Engineering Standards
| Standard | Clause / Scope | Status |
| :--- | :--- | :---: |
| MoRTH Section 500 | Mass & Cost Calculation | **PASSED** |
| IRC:SP:83 | Depth Interval Monotonicity | **PASSED** |
| ASTM D6433 | PCI Deduction Monotonicity | **PASSED** |

## 5. Latency Profile
- **Vision Classifier Latency**: 9.8 ms (102.0 FPS on 2 CPU threads)
- **Pipeline Latency**: 4312.0 ms
- **Latency Budget**: <= 35.0 ms
- **Status**: **PASSED**