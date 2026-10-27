#!/usr/bin/env bash
# ==============================================================================
# ROAD-SHIELD: 2-3 HOUR FULL DEEP LEARNING GPU TRAINING PIPELINE
# SIH 2026 Problem Statement SIH26124 | Bharat Electronics Limited (BEL)
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs checkpoints

echo "================================================================================"
echo "🚀 ROAD-SHIELD: DEEP GPU MULTI-MODAL TRAINING PIPELINE"
echo "================================================================================"
date -u

# Check GPU
python -c '
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA GPU not detected! Training will run very slowly on CPU.")
else:
    print(f"CUDA GPU Active: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Allocated: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
'

# 1. Dataset Expansion for Scarce Classes (Waterlogging, Zebra, Divider, Signs)
echo -e "\n--------------------------------------------------------------------------------"
echo ">>> [Stage 1/6] Ingesting & Expanding Rare Classes with Real Photographs..."
echo "--------------------------------------------------------------------------------"
python -m scripts.expand_rare_classes --target 80 || true
python -m scripts.fetch_cracks_potholes_dataset --limit 2235 --workers 8 || true

# 2. Civil Engineering Models (ASTM D6433 PCI, HDM-4 Deterioration, IRC Depth)
echo -e "\n--------------------------------------------------------------------------------"
echo ">>> [Stage 2/6] Training Photometric Civil Engineering Models..."
echo "--------------------------------------------------------------------------------"
python -m training.train_civil_real

# 3. Deep 1-D CNN IMU Shock Vibration Classifier (~10 min)
echo -e "\n--------------------------------------------------------------------------------"
echo ">>> [Stage 3/6] Training Deep 1-D CNN on 100 Hz IMU Accelerometer Signals..."
echo "--------------------------------------------------------------------------------"
python -m training.train_imu_deep

# 4. Deep U-Net Pixel Defect Segmenter on GPU (~45-60 min)
echo -e "\n--------------------------------------------------------------------------------"
echo ">>> [Stage 4/6] Training Deep U-Net (ResNet-34 Encoder) Pixel Segmenter on GPU..."
echo "--------------------------------------------------------------------------------"
python -m training.train_unet_segmenter

# 5. Deep End-to-End CNN Fine-Tuning with Unfrozen Convolutions (~60-75 min)
echo -e "\n--------------------------------------------------------------------------------"
echo ">>> [Stage 5/6] End-to-End Fine-Tuning of ConvNeXt / EfficientNet / ResNet-50..."
echo "--------------------------------------------------------------------------------"
python -m training.train_finetune_cnn --archs efficientnet_b0,efficientnet_b2,mobilenet_v3_large,resnet50 --epochs 40 --batch 48 --workers 2

# 6. Flagship Dual Attention Network (DAN) + 21-Node Decision DAG (~15 min)
echo -e "\n--------------------------------------------------------------------------------"
echo ">>> [Stage 6/6] Training Flagship DAN (PAM/CAM) + 21-Node Pairwise Decision DAG..."
echo "--------------------------------------------------------------------------------"
python -m training.train_dan_dag --backbone mobilenetv2 --epochs 65

# 7. Model Selection & Full Verification Suite
echo -e "\n--------------------------------------------------------------------------------"
echo ">>> Final Evaluation & Verification Test Suite..."
echo "--------------------------------------------------------------------------------"
python -m scripts.select_vision_model || true
python -m unittest tests.test_road_shield

# 8. Package all outputs
echo -e "\n--------------------------------------------------------------------------------"
echo ">>> Packaging All Trained Checkpoints and ONNX Models..."
echo "--------------------------------------------------------------------------------"
zip -q -r road_shield_trained_checkpoints.zip checkpoints/ logs/
echo "Done! Generated 'road_shield_trained_checkpoints.zip' ($(du -sh road_shield_trained_checkpoints.zip | cut -f1))"
date -u
