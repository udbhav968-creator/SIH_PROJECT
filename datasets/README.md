# ROAD-SHIELD AI: Master Benchmark & Real GitHub Datasets Vault

**Authority & Compliance:** Ministry of Road Transport & Highways (MoRTH Section 500) | National Highways Authority of India (NHAI) | SIH2026-MORTH-TRANS-018

---

## 1. Executive Summary & Inventory Matrix

This directory (`road_shield_ai_engine/datasets/`) consolidates all **7 canonical multi-modal benchmark datasets** plus **real raw road photographs directly downloaded from GitHub repositories and Kaggle mirrors**:
- **Vision Distress Net (Model M1)**
- **100 Hz IMU Shock Net (Model M4)**
- **ASTM D6433 Continuous PCI Regressor (Model M_PCI)**
- **180-Day Monsoon Degradation Forecaster (Model M_DEGRADE)**
- **Dual-Sensor Bayesian Fusion Gate (Model M5)**

| Sub-Directory | Benchmark Origin & Real Source | Total Samples | Primary Formats | Target Classes / Prediction Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **`01_rdd2022_india/`** | IEEE BigData Crowdsensing 2022 + Real Photos | 20,000 + JPEGs | `.npz`, `.json`, `.csv`, `.jpg` | `D00` (Longitudinal), `D10` (Transverse), `D20` (Alligator), `D40` (Pothole) |
| **`02_kaggle_pothole_600/`** | Kaggle Pothole Open Benchmark + Real JPEGs | 10,000 + JPEGs | `.npz`, `.json`, `.csv`, `.jpg` | High-contrast cavity discrimination (`Normal` vs `Pothole`) |
| **`03_crack500_fatigue/`** | CRACK500 Pavement Fatigue Study + Real JPEGs | 8,000 + JPEGs | `.npz`, `.json`, `.csv`, `.jpg` | Structural fatigue cracking (`D00`, `D10`, `D20`) |
| **`04_mobile_imu_telemetry_100hz/`** | MoRTH Patrol Vehicle Telemetry | 15,000 | `.npz`, `.json`, `.csv` | 100 Hz tri-axial accelerometer ($a_x, a_y, a_z$) dynamic shock waves |
| **`05_morth_civil_hard_negatives/`** | ROAD-SHIELD Civil Hard Negatives | 5,000 | `.npz`, `.json`, `.csv` | Utility Manholes, Tar Snakes, Expansion Joints, Tree Shadows |
| **`06_astm_d6433_pci_benchmark/`** | ASTM D6433 Standard Practice | 10,000 | `.npz`, `.json`, `.csv` | 20 standard deduct metrics $\to$ Continuous PCI Score ($0 - 100$) |
| **`07_monsoon_pavement_deterioration/`**| NHAI Monsoon Degradation Logs | 5,000 | `.npz`, `.json`, `.csv` | 180-day degradation vectors under heavy monsoon axle loading |
| **`real_images/` Folders** | GitHub Raw (`andrijdavid/Cracks-and-Potholes`) | Real Photos | `.jpg` JPEGs | Real 1024x640 highway photos with COCO bounding boxes |
| **TOTAL REPOSITORY** | **7 Benchmarks + Real GitHub Data** | **73,060+** | **Mixed Raw & Tensors** | **Full Multi-Modal Deep Pipeline Ready** |

---

## 2. Directory Hierarchy

```text
road_shield_ai_engine/datasets/
├── README.md                                    # This master documentation file
├── dataset_manifest.json                        # Master machine-readable JSON inventory
├── real_github_images_manifest.json             # GitHub repo citation & COCO metadata
├── real_github_images_features.npz              # 64-dim neural feature vectors extracted from real pixels
├── master_pipeline.py                           # Single unified loader & training pipeline
│
├── 01_rdd2022_india/                            # IEEE RDD2022 India Highway Subset
│   ├── real_images/                             # Real raw verified road photos from GitHub (.jpg)
│   │   ├── index.json                           # SHA-256 hashes and file sizes
│   │   └── *.jpg
│   ├── rdd2022_train.npz                        # Training tensors (X: 16000x64, y_cls, y_geo)
│   ├── rdd2022_val.npz                          # Validation tensors (X: 4000x64, y_cls, y_geo)
│   ├── rdd2022_metadata.json                    # Highway names, weather profiles, bounding formats
│   └── rdd2022_sample_annotations.csv           # 500 tabular records with GPS, highway, and defect codes
│
├── 02_kaggle_pothole_600/                       # Kaggle Pothole-600 Optical Benchmark
│   ├── real_images/                             # Real raw pothole photos from GitHub (.jpg)
│   │   ├── index.json                           # Dimensions, bounding boxes, file sizes
│   │   └── *.jpg
│   ├── kaggle_potholes.npz                      # 10,000 cavity optical tensors (64-dim)
│   ├── kaggle_pothole_metadata.json             # Camera perspectives & optical parameters
│   └── kaggle_annotations.csv                   # Depths, estimated area (m²), and MoRTH repair costs
│
├── 03_crack500_fatigue/                         # CRACK500 Structural Fatigue Cracking
│   ├── real_images/                             # Real raw cracking photos from GitHub (.jpg)
│   │   ├── index.json                           # Bounding boxes, crack types
│   │   └── *.jpg
│   ├── crack500_train.npz                       # 8,000 high-res cracking representations
│   ├── crack500_metadata.json                   # Crack classification schema (D00, D10, D20)
│   └── crack500_measurements.csv                # Crack widths (mm), lengths (m), severity ratings
│
├── 04_mobile_imu_telemetry_100hz/               # Mobile IMU 100 Hz Accelerometer Telemetry
│   ├── imu_shock_100hz_train.npz                # 12,000 sequences (100 timesteps × 3 axes)
│   ├── imu_shock_100hz_val.npz                  # 3,000 validation sequences
│   ├── imu_metadata.json                        # 100 Hz sampling spec, peak shock criteria
│   └── imu_telemetry_sample.csv                 # Raw 10ms-step tri-axial vibration readings
│
├── 05_morth_civil_hard_negatives/               # False-Alarm Suppression & OHEM Mining
│   ├── hard_negatives.npz                       # 5,000 hard negative feature vectors
│   ├── hard_negatives_metadata.json             # Manhole, tar sealant, and tree shadow signatures
│   └── rejection_rules.csv                      # Optical appearance vs IMU sensor response rules
│
├── 06_astm_d6433_pci_benchmark/                 # ASTM D6433 Pavement Condition Index
│   ├── pci_dataset.npz                          # 10,000 road sections with 20 distress deducts & true PCI
│   ├── pci_metadata.json                        # Standard ASTM condition bands (Good -> Failed)
│   └── pci_survey_records.csv                   # Sample highway survey sections & treatment codes
│
└── 07_monsoon_pavement_deterioration/           # Monsoon 180-Day Lifecycle Trajectories
    ├── deterioration_trajectories.npz           # Initial states (5-dim) & 4-quarter decay projections
    ├── deterioration_metadata.json              # Physical governing differential equations
    └── 180day_forecast_sample.csv               # 250 sample trajectory records with urgent patch flags
```

---

## 3. Real GitHub & Kaggle Dataset Sources

1. **`andrijdavid/Cracks-and-Potholes-in-Road-Images-Dataset`**:
   - Repository: `https://github.com/andrijdavid/Cracks-and-Potholes-in-Road-Images-Dataset`
   - Real JPEGs downloaded directly into `02_kaggle_pothole_600/real_images/` and `03_crack500_fatigue/real_images/`.
   - Resolution: $1024 \times 640$ high-resolution road photography with COCO bounding boxes.
2. **Kaggle Pothole-600 Benchmark**:
   - Open source dataset mapping optical cavity depth and edge spalling across diverse daylight and moisture conditions.
3. **IEEE RDD2022**:
   - Sekimoto Lab / Crowdsensing Road Damage Detection benchmark capturing highway conditions in India.

---

## 4. How to Execute the Master Pipeline

```bash
# 1. Verify all 7 benchmark datasets + Real GitHub images:
python datasets/master_pipeline.py

# 2. Run deep training on Model M1 across the unified vision corpus (34,448 samples):
python datasets/master_pipeline.py --train
```
