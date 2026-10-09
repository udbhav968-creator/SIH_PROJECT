#!/usr/bin/env bash
# Transformers, an ensemble served as a cascade, and an IMU transformer - on a Colab GPU, AFTER
# scripts/colab_train_all.sh in the SAME session (it needs that run's corpus on disk):
#
#   !bash scripts/colab_train_deep.sh 2>&1 | tee -a logs/colab_deep.txt
#   !RTDETR=1 bash scripts/colab_train_deep.sh ...      # also train RT-DETR as a road-damage detector candidate
#
#   D1  ensemble members, train-only, one per call (resumable): two CNNs and two vision transformers
#       (DeiT-Small, LeViT-256 from timm); each leaves its ONNX and validation/test logits in checkpoints/ensemble_runs
#                                                                              (~15-30 min each on a T4)
#   D2  training/select_ensemble.py: members, temperatures and the cascade threshold chosen on validation only;
#       served only if it beats the best single network by the rule's margin; test scored once with a bootstrap
#   D3  IMU: 1-D CNN (two widths) vs transformer vs RandomForest, time-blocked CV decides  (~10 min)
#   D4  (RTDETR=1) RT-DETR detector candidate; scripts/select_rdd_detector.py decides on India validation
#   D5  tests + package road_shield_deep_outputs.zip
#
# Every decision is made by the script that measured it, on validation; the site keeps its current models when a
# new one does not win. Resumable through Google Drive like the other Colab scripts.
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
  tar -czf "$STATE_DIR/deep.tmp.tgz" checkpoints logs 2>/dev/null && mv "$STATE_DIR/deep.tmp.tgz" "$STATE_DIR/deep.tgz"
  echo "  [saved progress to $STATE_DIR/deep.tgz]"
}
skip() { if [ -f "logs/.deep_done_$1" ]; then echo "  (already done - skipped)"; return 0; fi; return 1; }
done_() { touch "logs/.deep_done_$1"; persist; }

if [ ! -f checkpoints/finetune_summary.json ] && [ -n "$STATE_DIR" ] && [ -f "$STATE_DIR/state.tgz" ]; then
  echo "restoring the main run from $STATE_DIR/state.tgz"; tar -xzf "$STATE_DIR/state.tgz"
fi
if [ -n "$STATE_DIR" ] && [ -f "$STATE_DIR/deep.tgz" ]; then
  echo "restoring this script's progress from $STATE_DIR/deep.tgz"; tar -xzf "$STATE_DIR/deep.tgz"
fi
[ -f logs/.deep_start ] || touch logs/.deep_start

step "D0. environment"
pip install -q "scikit-learn>=1.8.0,<1.9" "opencv-python-headless>=4.8,<5" onnxruntime onnx onnxscript timm 2>&1 | tail -2
python -c "import torch, timm; print('torch', torch.__version__, '| timm', timm.__version__, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
# The members must be trained and scored on the corpus the main run used; refuse a bare checkout.
python - <<'EOF' || { echo "The image corpus is not on this runtime: run scripts/colab_train_all.sh (steps 1-3) in this session first."; exit 1; }
import sys
from training.train_cnn_head import capped_grouped_split
_, (tr, va, te) = capped_grouped_split(42, 1500)
print(f"corpus: train {len(tr)} | val {len(va)} | test {len(te)}")
sys.exit(0 if len(te) >= 400 else 1)
EOF

MEMBERS="${MEMBERS:-mobilenet_v3_large,efficientnet_b0,deit_small,levit_256}"
# Train-only networks live under their own folder, so they never overwrite the main run's reports or leave stray
# deep_vision_*.onnx files next to the served one.
RUNS=checkpoints/ensemble_runs
step "D1. ensemble members (train-only): $MEMBERS"
if ! skip D1; then
  for A in ${MEMBERS//,/ }; do
    if [ -f "$RUNS/ensemble/${A}_logits.npz" ]; then echo "--- $A: already trained (logits on disk)"; continue; fi
    echo "--- $A: smoke run first"
    if ! python -m training.train_finetune_cnn --archs "$A" --smoke --train-only --out /tmp/deep_smoke 2>&1 | tail -3; then
      echo "--- $A smoke run FAILED - left out of the ensemble"; continue
    fi
    python -m training.train_finetune_cnn --archs "$A" --epochs "${EPOCHS:-40}" --batch 48 --workers 2 --train-only \
        --out "$RUNS" \
        2>&1 | grep -v "^  cached" | tee -a logs/finetune_deep.txt | grep -E "VAL|early stop|epoch +(1|10|20|30|40)/|Error|error"
    persist
  done
  N=$(ls "$RUNS"/ensemble/*_logits.npz 2>/dev/null | wc -l)
  echo "--- $N member(s) trained"
  [ "$N" -ge 2 ] && done_ D1
fi

step "D2. ensemble + cascade, chosen on validation"
if ! skip D2; then
  python -m training.select_ensemble --dir "$RUNS/ensemble" --out checkpoints 2>&1 | tee logs/ensemble.txt | tail -40 \
    && done_ D2
fi

step "D3. IMU: CNN vs transformer vs RandomForest"
if ! skip D3; then
  python -m training.train_imu_deep 2>&1 | tail -30 && done_ D3
fi

if [ "${RTDETR:-0}" = "1" ]; then
  step "D4. RT-DETR road-damage detector candidate"
  if ! skip D4; then
    pip install -q ultralytics 2>&1 | tail -1
    if [ -d datasets/rdd2022_india/images/train ]; then
      if python -m training.train_rdd_detector --model rtdetr-l.pt --tag rtdetr --run-name rdd_rtdetr_smoke \
           --epochs 1 --fraction 0.05 --batch 4 --out /tmp/rtdetr_smoke 2>&1 | tail -4; then
        python -m training.train_rdd_detector --model rtdetr-l.pt --tag rtdetr --run-name rdd_rtdetr \
            --epochs "${RDD_EPOCHS:-40}" --batch 8 2>&1 | tee logs/rdd_rtdetr.txt | grep -E "VAL|TEST|exported|Error|error"
        python -m scripts.select_rdd_detector --tag rtdetr && done_ D4
      else
        echo "RT-DETR smoke run FAILED - the served detector is unchanged"
      fi
    else
      echo "RDD2022 YOLO data not on this runtime: run scripts/colab_train_extra.sh step E2 first"
    fi
  fi
fi

step "D5. tests + package"
python -m unittest discover -s tests -t . > logs/tests_deep.txt 2>&1
grep -E "^(Ran|OK|FAILED)" logs/tests_deep.txt
# train-only networks' ONNX files are large and not served; only the chosen copies (deep_vision_ens_*) are packaged
find checkpoints logs -type f -newer logs/.deep_start ! -name "*.db*" ! -name ".deep_*" ! -path "checkpoints/ensemble_runs/*.onnx" \
    ! -path "checkpoints/ensemble_runs/*/*.onnx" > logs/deep_files.txt
rm -f road_shield_deep_outputs.zip
zip -q -r road_shield_deep_outputs.zip -@ < logs/deep_files.txt
persist
[ -n "$STATE_DIR" ] && cp road_shield_deep_outputs.zip "$STATE_DIR/" && echo "  copy saved to $STATE_DIR/road_shield_deep_outputs.zip"
ls -la road_shield_deep_outputs.zip
echo "DEEP_STEPS_DONE"
