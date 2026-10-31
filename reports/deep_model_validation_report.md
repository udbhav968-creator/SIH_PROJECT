# ROAD-SHIELD: Deep Model Architecture & Forensic Validation Audit
**Audit Timestamp UTC**: 2026-10-31T05:25:05.009085+00:00  
**Overall Compliance Status**: **CERTIFIED**  

---
## 1. Checkpoint Integrity & Parameter Health
| Model Name | Checkpoint File | Size (MB) | Status |
| :--- | :--- | :---: | :---: |
| Vision CNN MobileNetV3 (Refit) | `deep_vision_mobilenet_v3_large_refit.onnx` | 16.83 MB | **VERIFIED** |
| Dense Pixel Segmenter (Pixel ONNX) | `defect_segmenter_pixel.onnx` | 1.5 MB | **VERIFIED** |
| Flagship DAN-DAG Weights | `dan_dag_model.joblib` | 4.69 MB | **VERIFIED** |
| ASTM D6433 PCI Regressor | `pci_model.joblib` | 0.54 MB | **VERIFIED** |
| HDM-4 Deterioration Forecaster | `deterioration_model.joblib` | 1.12 MB | **VERIFIED** |
| Photometric Depth Regressor | `depth_estimator_model.joblib` | 0.91 MB | **VERIFIED** |
| 100 Hz IMU Shock Vibration CNN | `imu_shock_model.joblib` | 11.37 MB | **VERIFIED** |
| Out-of-Distribution Input Guard | `ood_guard.npz` | 0.56 MB | **VERIFIED** |
| Claims & Certified Metrics | `claims.json` | 0.03 MB | **VERIFIED** |

## 2. Statistical Quality Gates (Held-Out Test Sets)
| Model Architecture | Evaluated Metric | Quality Gate Floor | Measured Result | Status |
| :--- | :--- | :---: | :---: | :---: |
| Vision CNN (MobileNetV3 Refit) | Test Accuracy | >= 0.85 | **0.9435** | PASSED |
| Vision CNN (MobileNetV3 Refit) | Test Macro-F1 | >= 0.75 | **0.8786** | PASSED |
| Vision CNN (Indian Roads Bench) | Indian Roads Accuracy | >= 0.85 | **0.9225** | PASSED |
| DAN-DAG Dual Attention Network | Mean 21-Node DAG Accuracy | >= 0.90 | **0.9352** | PASSED |
| DAN-DAG Dual Attention Network | Held-Out Test Accuracy | >= 0.80 | **0.8663** | PASSED |
| ASTM D6433 PCI Regressor | Held-Out R² | >= 0.85 | **0.8839** | PASSED |
| HDM-4 Deterioration Forecaster | Area Expansion R² | >= 0.80 | **0.9879** | PASSED |
| Photometric Depth Regressor | Pothole Depth R² | >= 0.50 | **0.6211** | PASSED |
| 100 Hz IMU Shock 1-D CNN | Validation Accuracy | >= 0.70 | **0.8659** | PASSED |
| Out-of-Distribution Input Guard | AUROC Combined | >= 0.90 | **0.9945** | PASSED |

## 3. End-to-End Pavement Scene Stress Tests
| Test Scene Description | Primary Finding | Confidence | Surface Area | Repair Cost | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Scene C: Foreground Crater with Reflection | Pothole Cavity | 93.0% | 1.8121 m² | Rs 2175.75 | **PASS** |
| Rural Dirt Road with Puddles | Pothole Cavity | 89.1% | 0.105 m² | Rs 105.75 | **PASS** |

## 4. Civil Engineering Standards Compliance
| Governing Standard | Specification Clause | Verification Scope | Status |
| :--- | :--- | :--- | :---: |
| MoRTH Section 500 | IRC/MoRTH Spec | Mass & Cost Calculation | **PASSED** |
| IRC:SP:83 | IRC/MoRTH Spec | Depth Interval Monotonicity | **PASSED** |
| ASTM D6433 | IRC/MoRTH Spec | PCI Deduction Monotonicity | **PASSED** |

## 5. Inference Latency & Performance Profile
- **Core ONNX Vision Classifier Latency**: 11.6 ms (86.2 FPS on 2 CPU threads)
- **Full Multi-Stage Pipeline Latency**: 5125.4 ms (Complete 8-Model Cascade on CPU)
- **Edge Neural Latency Budget**: <= 35.0 ms
- **Status**: **PASSED**

---
*Certified by ROAD-SHIELD Deep Verification Engine | SIH 2026 Problem Statement SIH26124*