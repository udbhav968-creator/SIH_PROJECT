#!/usr/bin/env bash
# End-to-end retraining on a Google Colab GPU runtime (or any Linux box with a GPU).
#
#   !git clone https://github.com/udbhav968-creator/SIH_PROJECT.git && cd SIH_PROJECT
#   !bash scripts/colab_train_all.sh 2>&1 | tee logs/colab_run.txt
#
# Every number this produces is written by the script that measured it; this
# file only runs them in order. Outputs land in checkpoints/ and logs/, and
# road_shield_colab_outputs.zip collects everything that changed.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
mkdir -p logs
touch logs/.start_marker
step() { echo; echo "=================== $* ==================="; date -u; }

step "0. environment"
# scikit-learn is pinned: the pickled checkpoints are not portable across minor versions.
pip install -q "scikit-learn>=1.8.0,<1.9" "opencv-python-headless>=4.8,<5" onnxruntime onnx onnxscript \
    scikit-image joblib pillow 2>&1 | tail -2
python - <<'EOF'
import sklearn, cv2, torch, torchvision, onnxruntime, platform
print("python", platform.python_version(), "| sklearn", sklearn.__version__, "| cv2", cv2.__version__,
      "| torch", torch.__version__, "| torchvision", torchvision.__version__, "| ort", onnxruntime.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
EOF
nproc; free -g | head -2

step "1. data: DNIT cracks & potholes crops (regenerated, not in git)"
python -m scripts.fetch_cracks_potholes_dataset --limit 2235 --workers 16 2>&1 | tail -15

step "2. data: RDD2022 India (official CRDDC 2022 release)"
# The per-country S3 archive now returns 403; fetch_rdd2022_india.py falls back to
# the official figshare release (13 GB, India part extracted) and verifies every
# archive. Upload your own copy and set RDD_ZIP=/content/RDD2022_India.zip to skip
# the download. SKIP_RDD=1 trains without Indian data (not recommended).
if [ "${SKIP_RDD:-0}" != "1" ]; then
  python -m scripts.fetch_rdd2022_india ${RDD_ZIP:+--local-zip "$RDD_ZIP"}
  INDIA=$(python -c "from scripts.fetch_rdd2022_india import india_root; print(india_root() or '')")
  if [ -z "$INDIA" ] || [ ! -d "$INDIA/train" ]; then
    echo "STOP: RDD2022 India is not available (see the message above)."; exit 1
  fi
  python -m scripts.prepare_rdd2022_voc --src "$INDIA" --out datasets/rdd2022_india | tee logs/rdd_prepare.json
  python -m scripts.ingest_rdd2022_india 2>&1 | tail -8
fi

step "3. corpus inventory"
python - <<'EOF' | tee logs/corpus_inventory.json
import json, os, collections, re
from data.image_dataset import CLASS_FOLDERS
from pipeline.corpus_policy import is_road_scene
out = {}
for cid, (folder, name) in sorted(CLASS_FOLDERS.items()):
    d = os.path.join("datasets", folder, "real_images")
    fs = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.isdir(d) else []
    pref = collections.Counter(re.sub(r"^(cpr|rddin|aug|WM|real|kag_[a-z-]+_\d).*$", r"\1", f) if re.match(r"^(cpr|rddin|aug|WM|real|kag_)", f) else "dnit_raw" for f in fs)
    out[name] = {"files": len(fs), "road_scene_files": sum(is_road_scene(os.path.join(d, f)) for f in fs), "by_source_prefix": dict(pref)}
ev = "datasets/_eval_rdd2022_india"
out["_indian_eval"] = {c: len(os.listdir(os.path.join(ev, c))) for c in ("normal", "crack", "pothole") if os.path.isdir(os.path.join(ev, c))}
print(json.dumps(out, indent=1))
EOF

step "4. test suite BEFORE retraining (pinned environment)"
python -m unittest discover -s tests -t . > logs/tests_before.txt 2>&1; tail -4 logs/tests_before.txt

step "5. frozen MobileNetV2 + head, retrained on this corpus (the comparison baseline)"
python -m training.train_cnn_head --backbone mobilenetv2 --compare 2>&1 | grep -v "^\s*embedding\|progress" | tail -40
python -m scripts.eval_indian_roads --tag frozen_head

step "6. end-to-end fine-tuning (GPU)"
python -m training.train_finetune_cnn --archs "${ARCHS:-efficientnet_b0,efficientnet_b2,mobilenet_v3_large,resnet50}" \
    --epochs "${EPOCHS:-40}" --batch 48 --workers 2 2>&1 | grep -v "^  cached" | tee logs/finetune.txt | grep -E "VAL|early|chosen|served|epoch +(1|10|20|30|40)/"

step "7. choose the served image classifier (rule fixed in advance)"
python -m scripts.select_vision_model
python -m scripts.eval_indian_roads --tag served

step "8. IMU: deep 1-D CNN vs RandomForest"
python -m training.train_imu_deep 2>&1 | tail -30

step "9. end-to-end pipeline benchmark on distinct real photographs"
python tests/test_deep_pipeline_real_images.py 2>&1 | tail -25

step "10. claims registry rebuilt from the reports"
python -m scripts.build_claims 2>&1 | tail -5

step "11. test suite AFTER retraining"
python -m unittest discover -s tests -t . > logs/tests_after.txt 2>&1; tail -4 logs/tests_after.txt

step "12. report rebuilt from the new checkpoints (tracked changes)"
python scripts/report_audit/build_report.py --tests-log logs/tests_after.txt

step "13. package outputs"
find checkpoints logs datasets/_eval_rdd2022_india/manifest.json CSET485_ROAD_SHIELD_Milestone2_Report_Audited_TrackedChanges.docx \
    -type f -newer logs/.start_marker ! -name "*.db*" ! -path "*/detectors/*" > logs/changed_files.txt
zip -q -r road_shield_colab_outputs.zip -@ < logs/changed_files.txt
ls -la road_shield_colab_outputs.zip; wc -l logs/changed_files.txt
echo "ALL_STEPS_DONE"
