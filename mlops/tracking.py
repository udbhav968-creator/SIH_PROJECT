"""
Experiment tracking: every training run leaves a record of what it was given and what it measured.

    from mlops.tracking import start_run
    with start_run("ood_guard", params={"pca_dims": 64}) as run:
        ...
        run.log_metrics({"auroc": 0.97})
        run.log_artifact("checkpoints/ood_guard.npz")

A run stores its parameters, metrics (with optional steps, for curves), tags, the git commit it ran
from, and the SHA-256 and size of each file it produced, so a model in the registry can be traced
back to the exact run and code that made it. Runs that raise are kept and marked FAILED.

The store is SQLite under mlops_store/ and needs nothing installed. If MLflow is installed the same
run is mirrored to it (MLFLOW_TRACKING_URI, default ./mlruns), so `mlflow ui` shows every run;
set ROAD_SHIELD_MLFLOW=0 to turn the mirror off.
"""
import json
import os
import platform
import threading
import time
import uuid

from mlops.paths import connect, git_commit, sha256_file, store_dir, ENGINE_ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    experiment  TEXT NOT NULL,
    status      TEXT NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    git_commit  TEXT,
    params      TEXT NOT NULL DEFAULT '{}',
    tags        TEXT NOT NULL DEFAULT '{}',
    artifacts   TEXT NOT NULL DEFAULT '[]',
    error       TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    key    TEXT NOT NULL,
    value  REAL NOT NULL,
    step   INTEGER,
    at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, key);
CREATE INDEX IF NOT EXISTS idx_runs_exp ON runs(experiment, started_at);
"""

_LOCK = threading.RLock()


def _db(path=None):
    db = connect(path or os.path.join(store_dir(), "runs.db"))
    db.executescript(SCHEMA)
    return db


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


class _MLflowMirror:
    def __init__(self, experiment, params, tags):
        self.active = False
        if os.environ.get("ROAD_SHIELD_MLFLOW", "1") == "0":
            return
        try:
            import mlflow  # noqa: F401
        except Exception:
            return
        try:
            import mlflow
            mlflow.set_experiment(f"road-shield/{experiment}")
            mlflow.start_run(tags={k: str(v) for k, v in (tags or {}).items()})
            if params:
                mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            self.mlflow = mlflow
            self.active = True
        except Exception as e:
            print(f"[tracking] MLflow mirror off: {e}")

    def call(self, name, *a, **kw):
        if self.active:
            try:
                getattr(self.mlflow, name)(*a, **kw)
            except Exception as e:
                print(f"[tracking] MLflow {name} failed: {e}")


class Run:
    def __init__(self, experiment, params=None, tags=None, db_path=None):
        self.run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.experiment = str(experiment)
        self.params = {k: _jsonable(v) for k, v in (params or {}).items()}
        self.tags = {"host": platform.node(), "python": platform.python_version(), **(tags or {})}
        self.metrics = {}
        self.artifacts = []
        self._db_path = db_path
        self._started = time.time()
        with _LOCK:
            db = _db(db_path)
            db.execute("INSERT INTO runs(run_id, experiment, status, started_at, git_commit, params, tags) "
                       "VALUES(?,?,?,?,?,?,?)",
                       (self.run_id, self.experiment, "RUNNING", self._started, git_commit(),
                        json.dumps(self.params), json.dumps(self.tags)))
            db.close()
        self._mlflow = _MLflowMirror(self.experiment, self.params, self.tags)

    def log_params(self, params):
        self.params.update({k: _jsonable(v) for k, v in params.items()})
        self._save("params", json.dumps(self.params))
        self._mlflow.call("log_params", {k: str(v)[:500] for k, v in params.items()})

    def set_tag(self, key, value):
        self.tags[key] = value
        self._save("tags", json.dumps(self.tags))
        self._mlflow.call("set_tag", key, str(value))

    def log_metric(self, key, value, step=None):
        if value is None:
            return
        value = float(value)
        self.metrics[key] = value
        with _LOCK:
            db = _db(self._db_path)
            db.execute("INSERT INTO metrics(run_id, key, value, step, at) VALUES(?,?,?,?,?)",
                       (self.run_id, key, value, step, time.time()))
            db.close()
        self._mlflow.call("log_metric", key, value, step=step)

    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.log_metric(k, v, step)

    def log_artifact(self, path):
        if not os.path.exists(path):
            return
        rel = os.path.relpath(os.path.abspath(path), ENGINE_ROOT).replace("\\", "/")
        rec = {"path": rel, "sha256": sha256_file(path), "bytes": os.path.getsize(path)}
        self.artifacts = [a for a in self.artifacts if a["path"] != rel] + [rec]
        self._save("artifacts", json.dumps(self.artifacts))
        if rec["bytes"] < 5 * 1024 * 1024:
            self._mlflow.call("log_artifact", path)

    def _save(self, col, value):
        with _LOCK:
            db = _db(self._db_path)
            db.execute(f"UPDATE runs SET {col}=? WHERE run_id=?", (value, self.run_id))
            db.close()

    def end(self, status="FINISHED", error=None):
        with _LOCK:
            db = _db(self._db_path)
            db.execute("UPDATE runs SET status=?, ended_at=?, error=? WHERE run_id=?",
                       (status, time.time(), error, self.run_id))
            db.close()
        self._mlflow.call("end_run", status="FINISHED" if status == "FINISHED" else "FAILED")

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        self.end("FINISHED" if et is None else "FAILED", None if et is None else f"{et.__name__}: {ev}")
        return False


def start_run(experiment, params=None, tags=None, db_path=None):
    return Run(experiment, params, tags, db_path)


def _row(db, r):
    out = dict(r)
    for k in ("params", "tags", "artifacts"):
        out[k] = json.loads(out[k] or ("[]" if k == "artifacts" else "{}"))
    last = {}
    for m in db.execute("SELECT key, value, step FROM metrics WHERE run_id=? ORDER BY at", (r["run_id"],)):
        last[m["key"]] = m["value"]
    out["metrics"] = last
    out["duration_s"] = round(out["ended_at"] - out["started_at"], 1) if out.get("ended_at") else None
    return out


def list_runs(experiment=None, limit=100, db_path=None):
    with _LOCK:
        db = _db(db_path)
        if experiment:
            rows = db.execute("SELECT * FROM runs WHERE experiment=? ORDER BY started_at DESC LIMIT ?",
                              (experiment, int(limit))).fetchall()
        else:
            rows = db.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (int(limit),)).fetchall()
        out = [_row(db, r) for r in rows]
        db.close()
    return out


def get_run(run_id, db_path=None):
    with _LOCK:
        db = _db(db_path)
        r = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        out = _row(db, r) if r else None
        if out:
            out["history"] = [dict(m) for m in db.execute(
                "SELECT key, value, step, at FROM metrics WHERE run_id=? ORDER BY at", (run_id,))]
        db.close()
    return out
