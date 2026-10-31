"""Pipeline coordinator for model training workflows."""

import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "2")

import time
import threading

from training import train_vision, train_imu


class TrainingTelemetryStreamer:
    """Thread-safe status for a real, stage-based (not epoch-based) training run."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TrainingTelemetryStreamer, cls).__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.is_training = False
        self.run_id = "IDLE"
        self.status = "IDLE_READY"
        self.stage = "IDLE"
        self.results = {}
        self.start_time = 0.0

    def get_status(self):
        with self._lock:
            elapsed = (time.time() - self.start_time) if self.is_training else 0.0
            return {
                "is_training": self.is_training,
                "run_id": self.run_id,
                "status": self.status,
                "stage": self.stage,
                "elapsed_seconds": round(elapsed, 1),
                "results": self.results,
            }

    def _set_stage(self, stage):
        with self._lock:
            self.stage = stage

    def _complete(self, results, final_msg):
        with self._lock:
            self.is_training = False
            self.status = final_msg
            self.results = results


telemetry_streamer = TrainingTelemetryStreamer()


def run_training_suite(async_mode=False):
    """
    Runs the project's two real training scripts (vision + IMU) against
    their real on-disk datasets and reports genuine held-out results.
    Unlike the old fabricated version there's no "dataset_name" or "epochs"
    request parameter to accept - each real model trains on a fixed real
    dataset in one call, so there's nothing meaningful to parameterize
    per-request.
    """
    streamer = telemetry_streamer
    if streamer.is_training:
        return {"status": "ERROR_ALREADY_TRAINING", "message": "A training run is already in progress."}

    run_id = f"RUN-{int(time.time())}"

    def _worker():
        streamer.is_training = True
        streamer.run_id = run_id
        streamer.status = "RUNNING"
        streamer.start_time = time.time()
        results = {}

        streamer._set_stage("TRAINING_VISION_DISTRESS_NET")
        try:
            results["vision"] = train_vision.run_training()
        except Exception as e:
            results["vision"] = {"error": str(e)}

        streamer._set_stage("TRAINING_IMU_SHOCK_CLASSIFIER")
        try:
            results["imu"] = train_imu.run_training()
        except Exception as e:
            results["imu"] = {"error": str(e)}

        streamer._complete(results, "ALL_MODELS_TRAINED")

    if async_mode:
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return {"status": "ASYNC_TRAINING_STARTED", "run_id": run_id}
    else:
        _worker()
        return streamer.get_status()


run_mega_training_suite = run_training_suite
