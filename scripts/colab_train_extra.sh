#!/usr/bin/env bash
# Extra deep models, run AFTER scripts/colab_train_all.sh in the SAME Colab session
# (it reuses the RDD2022 India and DNIT data already on the runtime's disk):
#
#   !bash scripts/colab_train_extra.sh 2>&1 | tee -a logs/colab_extra.txt
#
#   E1  U-Net deep segmenter on the DNIT polygons      (~20-35 min on a T4)
#   E1c SegFormer-B1 candidate (transformer segmenter) on all pixel-labelled sets; replaces the U-Net only
#       if better end to end (scripts/select_deep_segmenter.py)  (~40-60 min on a T4)
#   E1b IMU 1-D CNN vs RandomForest, re-run with time-blocked CV folds (~5 min)
#   E2  YOLOv8 road-damage detector on RDD2022 India   (~45-70 min on a T4)
#
# Each model is trained, scored once on held-out photographs, exported to ONNX,
# and served only by the selection rule written in its trainer. A quick smoke run
# precedes each full run so a bug fails in one minute, not after an hour. If a
# step fails, the next one still runs, and the site keeps the models it had.
# Resumable through Google Drive exactly like colab_train_all.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
step() { echo; echo "=================== $* ==================="; date -u; }
if [ -z "${STATE_DIR+x}" ]; then
  if [ -d /content/drive/MyDrive ]; then STATE_DIR=/content/drive/MyDrive/road_shield_state; else STATE_DIR=""; fi
fi
persist() {
  [ -n "$STATE_DIR" ] || return 0
  mkdir -p "$STATE_DIR"
  tar -czf "$STATE_DIR/extra.tmp.tgz" checkpoints logs 2>/dev/null && mv "$STATE_DIR/extra.tmp.tgz" "$STATE_DIR/extra.tgz"
  echo "  [saved progress to $STATE_DIR/extra.tgz]"
}
skip() { if [ -f "logs/.extra_done_$1" ]; then echo "  (already done - skipped)"; return 0; fi; return 1; }
done_() { touch "logs/.extra_done_$1"; persist; }

# A fresh session: bring back the main run's results first, then this script's own.
if [ ! -f checkpoints/finetune_summary.json ] && [ -n "$STATE_DIR" ] && [ -f "$STATE_DIR/state.tgz" ]; then
  echo "restoring the main run from $STATE_DIR/state.tgz"; tar -xzf "$STATE_DIR/state.tgz"
fi
if [ -n "$STATE_DIR" ] && [ -f "$STATE_DIR/extra.tgz" ]; then
  echo "restoring this script's progress from $STATE_DIR/extra.tgz"; tar -xzf "$STATE_DIR/extra.tgz"
fi
[ -f logs/.extra_start ] || touch logs/.extra_start

step "E0. environment"
pip install -q "scikit-learn>=1.8.0,<1.9" "opencv-python-headless>=4.8,<5" onnxruntime onnx onnxscript ultralytics transformers scipy 2>&1 | tail -2
python -c "import torch, ultralytics; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

step "E1. U-Net deep segmenter (DNIT polygons, same 500 test photographs as the pixel classifier)"
if ! skip E1; then
  if [ ! -f datasets/incoming/cracks_potholes_dnit/coco.json ]; then
    python -m scripts.fetch_cracks_potholes_dataset --limit 2235 --workers 16 2>&1 | tail -3
  fi
  echo "--- smoke run (2 epochs, 60 photographs, written to /tmp)"
  if python -m training.train_unet_segmenter --smoke --workers 0 --out /tmp/unet_smoke 2>&1 | tail -6; then
    echo "--- full run"
    python -m training.train_unet_segmenter --epochs "${UNET_EPOCHS:-45}" 2>&1 | tee logs/unet_segmenter.txt \
      | grep -E "device|loaded|epoch +(1|5|10|15|20|25|30|35|40|45)/|early stop|thresholds|SELECTION|TEST|exported|Error|error"
    [ -f checkpoints/segmenter_selection.json ] && done_ E1
  else
    echo "U-Net smoke run FAILED - skipping the full run; the pixel classifier stays in service"
  fi
fi

step "E1c. SegFormer candidate (transformer segmenter; CrackSeg9k, DeepCrack, CrackForest, Kaggle + DNIT)"
if ! skip E1c; then
  python -m scripts.fetch_seg_datasets --only crackseg9k deepcrack crackforest kaggle_pothole 2>&1 | tail -4
  echo "--- smoke run (1 epoch, untrained weights, written to /tmp)"
  if python -m training.train_unet_multi --arch "${SEGFORMER_ARCH:-segformer-b1}" --smoke --workers 0 \
       --out /tmp/segformer_smoke 2>&1 | tail -4; then
    echo "--- full run"
    python -m training.train_unet_multi --arch "${SEGFORMER_ARCH:-segformer-b1}" --epochs "${SEGFORMER_EPOCHS:-30}" \
        --out checkpoints/segformer_candidate ${STATE_DIR:+--ckpt-dir "$STATE_DIR/segformer"} --ckpt-every-min 30 \
        2>&1 | tee logs/segformer.txt | grep -E "device|SegFormer|segformer|epoch +(1|5|10|15|20|25|30)/|early stop|thresholds|SELECTION|TEST|exported|Error|error"
    if [ -f checkpoints/segformer_candidate/defect_segmenter_unet.onnx ]; then
      # replaces the U-Net only if it is better end to end; then the usual check against the pixel classifier
      python -m scripts.select_deep_segmenter --candidate checkpoints/segformer_candidate 2>&1 | tail -8 && done_ E1c
    fi
  else
    echo "SegFormer smoke run FAILED - the served segmenter is unchanged"
  fi
fi

step "E1b. IMU CNN vs RandomForest again, with time-blocked CV folds (~5 min)"
if ! skip E1b; then
  python -m training.train_imu_deep 2>&1 | tail -12
  [ -f checkpoints/imu_model_selection.json ] && done_ E1b
fi

step "E2. YOLOv8 road-damage detector (RDD2022 India boxes)"
if ! skip E2; then
  if [ ! -d datasets/rdd2022_india/images/train ]; then
    echo "RDD2022 YOLO data not on this runtime (new session) - fetching again (~15 min)"
    python -m scripts.fetch_rdd2022_india ${RDD_ZIP:+--local-zip "$RDD_ZIP"}
    INDIA=$(python -c "from scripts.fetch_rdd2022_india import india_root; print(india_root() or '')")
    [ -n "$INDIA" ] && python -m scripts.prepare_rdd2022_voc --src "$INDIA" --out datasets/rdd2022_india | tail -3
    rm -rf datasets/_downloads/rdd2022
  fi
  echo "--- smoke run (1 epoch, 5% of train, yolov8n, written to /tmp)"
  if python -m training.train_rdd_detector --model yolov8n.pt --epochs 1 --fraction 0.05 --out /tmp/rdd_smoke 2>&1 | tail -4; then
    echo "--- full run"
    python -m training.train_rdd_detector --epochs "${RDD_EPOCHS:-40}" 2>&1 | tee logs/rdd_detector.txt \
      | grep -E "rdd-detector|VAL|TEST|exported|Error|error"
    [ -f checkpoints/road_damage_detector_report.json ] && done_ E2
  else
    echo "detector smoke run FAILED - skipping the full run"
  fi
fi

step "E3. claims registry + tests"
python -m scripts.build_claims 2>&1 | tail -3
python -m unittest discover -s tests -t . > logs/tests_extra.txt 2>&1
grep -E "^(Ran|OK|FAILED)" logs/tests_extra.txt

step "E4. package"
find checkpoints logs -type f -newer logs/.extra_start ! -name "*.db*" ! -name ".extra_*" > logs/extra_files.txt
rm -f road_shield_extra_outputs.zip
zip -q -r road_shield_extra_outputs.zip -@ < logs/extra_files.txt
persist
[ -n "$STATE_DIR" ] && cp road_shield_extra_outputs.zip "$STATE_DIR/" && echo "  copy saved to $STATE_DIR/road_shield_extra_outputs.zip"
ls -la road_shield_extra_outputs.zip; cat logs/extra_files.txt
echo "EXTRA_STEPS_DONE"
