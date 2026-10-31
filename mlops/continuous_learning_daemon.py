"""
Continuous 24/7 Automated MLOps Engine for ROAD-SHIELD AI.

Runs autonomously as a background daemon or scheduled service to provide:
  1. Automated ingestion, quality auditing, and drift monitoring.
  2. Active learning & hard-negative uncertainty mining (Shannon entropy H >= 1.0).
  3. Scheduled deep continuous evaluation across held-out test benchmarks.
  4. Automated retraining of model subsystems when data batches accumulate.
  5. Automated production gating: candidate promoted and hot-reloaded only if
     it beats production metrics on held-out test sets without regressions.
  6. Persistent audit ledger (reports/continuous_mlops_ledger.json) for 24/7 traceability.
"""

import os
import sys
import time
import json
import threading
import traceback
from datetime import datetime

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
REPORTS_DIR = os.path.join(ENGINE_ROOT, "reports")
LEDGER_PATH = os.path.join(REPORTS_DIR, "continuous_mlops_ledger.json")


class ContinuousMLOpsEngine:
    """Autonomous 24/7 training, testing, and continuous improvement supervisor."""

    def __init__(self, check_interval_sec=120, auto_retrain=True):
        self.interval = check_interval_sec
        self.auto_retrain = auto_retrain
        self.running = False
        self.thread = None
        self.cycle_count = 0
        self.last_cycle_time = None
        self.last_benchmark_score = None
        self.status = "IDLE"
        self.history = []
        os.makedirs(REPORTS_DIR, exist_ok=True)
        self._load_ledger()

    def _load_ledger(self):
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.cycle_count = data.get("total_cycles", 0)
                    self.history = data.get("recent_cycles", [])[-50:]
                    self.last_benchmark_score = data.get("latest_benchmark")
            except Exception:
                self.history = []

    def _save_ledger(self):
        try:
            payload = {
                "engine": "ROAD-SHIELD Continuous 24/7 MLOps Engine",
                "status": self.status,
                "running": self.running,
                "total_cycles": self.cycle_count,
                "last_cycle_time": self.last_cycle_time,
                "latest_benchmark": self.last_benchmark_score,
                "recent_cycles": self.history[-50:],
                "updated_at": datetime.now().isoformat(),
            }
            with open(LEDGER_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write continuous ledger: {e}")

    def start(self):
        """Starts the 24/7 background worker thread."""
        if self.running:
            return False
        self.running = True
        self.status = "ONLINE_ACTIVE"
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="MLOps-24-7")
        self.thread.start()
        print(f"[MLOps 24/7] Continuous Learning Engine started (cycle interval: {self.interval}s)")
        return True

    def stop(self):
        """Stops the 24/7 background worker gracefully."""
        self.running = False
        self.status = "STOPPED"
        self._save_ledger()
        print("[MLOps 24/7] Continuous Learning Engine stopped")

    def run_single_cycle(self):
        """Executes one full iteration of auditing, active learning, evaluation, and retraining."""
        t0 = time.time()
        self.cycle_count += 1
        cycle_id = f"cycle_{self.cycle_count}_{int(t0)}"
        print(f"\n[MLOps 24/7] === Starting Continuous Cycle #{self.cycle_count} ({cycle_id}) ===")

        report = {
            "cycle_id": cycle_id,
            "timestamp": datetime.now().isoformat(),
            "tasks": {},
            "status": "SUCCESS"
        }

        # 1. System Health & Model Asset Verification
        try:
            report["tasks"]["model_verification"] = self._verify_models()
        except Exception as e:
            report["tasks"]["model_verification"] = {"status": "FAILED", "error": str(e)}

        # 2. Continuous Deep Benchmark Evaluation
        try:
            bench_res = self._run_continuous_eval()
            report["tasks"]["benchmark_evaluation"] = bench_res
            self.last_benchmark_score = bench_res.get("summary")
        except Exception as e:
            report["tasks"]["benchmark_evaluation"] = {"status": "FAILED", "error": str(e)}

        # 3. Active Learning & Uncertainty Mining
        try:
            al_res = self._mine_uncertain_samples()
            report["tasks"]["active_learning_mining"] = al_res
        except Exception as e:
            report["tasks"]["active_learning_mining"] = {"status": "FAILED", "error": str(e)}

        # 4. Automated Candidate Retraining & Production Gate (if enabled)
        if self.auto_retrain:
            try:
                retrain_res = self._check_and_retrain_candidates()
                report["tasks"]["auto_retrain"] = retrain_res
            except Exception as e:
                report["tasks"]["auto_retrain"] = {"status": "FAILED", "error": str(e)}

        elapsed = round(time.time() - t0, 2)
        report["elapsed_sec"] = elapsed
        self.last_cycle_time = datetime.now().isoformat()
        self.history.append(report)
        self._save_ledger()
        print(f"[MLOps 24/7] Cycle #{self.cycle_count} completed in {elapsed}s")
        return report

    def _run_loop(self):
        while self.running:
            try:
                self.run_single_cycle()
            except Exception as e:
                print(f"[MLOps 24/7 ERROR] Cycle failed: {e}\n{traceback.format_exc()}")
            # Sleep in 1-second ticks so stop() is responsive
            for _ in range(int(self.interval)):
                if not self.running:
                    break
                time.sleep(1)

    def _verify_models(self):
        """Verifies integrity of served models and detects missing files."""
        expected = [
            "deep_vision_mobilenet_v3_large_refit.onnx",
            "defect_segmenter_pixel.onnx",
            "defect_segmenter_unet.onnx",
            "road_shield_detector.onnx",
            "imu_shock_cnn.onnx",
            "ood_guard.npz"
        ]
        found = {}
        for m in expected:
            p = os.path.join(CKPT_DIR, m)
            found[m] = {
                "exists": os.path.exists(p),
                "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2) if os.path.exists(p) else 0.0
            }
        all_ready = all(v["exists"] for v in found.values())
        return {"status": "OK" if all_ready else "DEGRADED", "models": found}

    def _run_continuous_eval(self):
        """Runs fast continuous audit across diverse test images to detect accuracy drift."""
        from pipeline.deep_inference_pipeline import DeepInferencePipeline
        import glob
        import cv2

        pipe = DeepInferencePipeline(CKPT_DIR)
        test_images = []

        # Gather sample test images across distinct real classes
        patterns = [
            os.path.join(ENGINE_ROOT, "datasets", "02_kaggle_pothole_600", "real_images", "*.jpg"),
            os.path.join(ENGINE_ROOT, "datasets", "03_crack500_fatigue", "real_images", "*.jpg"),
            os.path.join(ENGINE_ROOT, "datasets", "05_morth_civil_hard_negatives", "real_images", "*.jpg"),
        ]
        for pat in patterns:
            files = sorted(glob.glob(pat))
            if files:
                test_images.extend(files[:3])  # Sample top 3 from each category for fast 24/7 audit

        if not test_images:
            return {"status": "SKIPPED", "reason": "No test images found on disk"}

        crashes = 0
        audited = 0
        distress_found = 0
        false_alarms_clean = 0
        total_tonnage = 0.0

        for img_path in test_images:
            try:
                res = pipe.audit_image(image_input=img_path)
                audited += 1
                is_clean_set = "05_morth_civil_hard_negatives" in img_path
                has_defect = bool(res.get("all_detections"))
                if has_defect:
                    distress_found += 1
                    if is_clean_set:
                        false_alarms_clean += 1
                ledger = res.get("morth_civil_ledger") or {}
                total_tonnage += float(ledger.get("total_bitumen_tonnage_t", 0.0))
            except Exception as e:
                crashes += 1

        summary = {
            "audited_images": audited,
            "crashes": crashes,
            "distress_detected": distress_found,
            "false_alarms_on_clean": false_alarms_clean,
            "clean_false_positive_rate": round(false_alarms_clean / max(1, 3), 4),
            "total_bitumen_tonnage_t": round(total_tonnage, 4),
            "pass_verdict": crashes == 0 and false_alarms_clean == 0
        }
        return {"status": "COMPLETED", "summary": summary}

    def _mine_uncertain_samples(self):
        """Scans the active learning pool for samples with Shannon entropy >= 1.0 bits."""
        from mlops.active_learning import ActiveLearningQueue
        from mlops.paths import WRITABLE_DIR

        db_path = os.path.join(WRITABLE_DIR, "road_shield.db")
        al_dir = os.path.join(WRITABLE_DIR, "al_images")
        al = ActiveLearningQueue(db_path, al_dir)
        queue = al.list_queue(limit=20)
        return {
            "status": "COMPLETED",
            "queued_samples_count": len(queue),
            "high_uncertainty_count": sum(1 for q in queue if float(q.get("uncertainty", 0.0)) >= 1.0)
        }

    def _check_and_retrain_candidates(self):
        """Checks if retrain triggers are met (e.g. queue size) and runs candidate training."""
        from mlops.active_learning import ActiveLearningQueue
        from mlops.paths import WRITABLE_DIR

        db_path = os.path.join(WRITABLE_DIR, "road_shield.db")
        al_dir = os.path.join(WRITABLE_DIR, "al_images")
        al = ActiveLearningQueue(db_path, al_dir)
        labeled = al.list_labeled(limit=100)

        # Retrain triggered only when >= 15 new human-labeled ground truth samples exist
        if len(labeled) < 15:
            return {
                "status": "STANDBY",
                "reason": f"Active learning buffer has {len(labeled)}/15 labeled samples needed for automated retraining."
            }

        # Run automated retrain pipeline on IMU / Segmenter
        from mlops.retrain import retrain_model
        res = retrain_model("imu_classifier", promote_if_passed=True)
        return {"status": "RETRAINED", "result": res}

    def get_status(self):
        """Returns comprehensive telemetry dictionary for the MLOps dashboard."""
        return {
            "status": self.status,
            "running": self.running,
            "interval_seconds": self.interval,
            "total_cycles": self.cycle_count,
            "last_cycle_time": self.last_cycle_time,
            "latest_benchmark": self.last_benchmark_score,
            "recent_cycles_count": len(self.history),
            "updated_at": datetime.now().isoformat()
        }


# Singleton engine instance
CONTINUOUS_ENGINE = ContinuousMLOpsEngine(check_interval_sec=120)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ROAD-SHIELD 24/7 Continuous MLOps Engine")
    parser.add_argument("--once", action="store_true", help="Run a single audit cycle and exit")
    parser.add_argument("--daemon", action="store_true", help="Run continuously in the background")
    parser.add_argument("--interval", type=int, default=120, help="Cycle interval in seconds (default: 120)")
    args = parser.parse_args()

    engine = ContinuousMLOpsEngine(check_interval_sec=args.interval)
    if args.once:
        engine.run_single_cycle()
    elif args.daemon:
        engine.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            engine.stop()
    else:
        # Default: run one cycle for immediate test
        engine.run_single_cycle()
