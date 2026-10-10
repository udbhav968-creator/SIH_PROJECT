"""
Model registry: every version of every model, what it scored, where it came from, and which one
is serving.

A version is the set of files the model's spec names (mlops/specs.py), each recorded by SHA-256
and kept in a content-addressed blob store (mlops_store/blobs/), so any version can be put back
byte for byte. Each version carries the metrics read from its own report files, the training run
that produced it and the git commit.

Stages:  candidate -> staging -> production -> archived
    candidate   registered, not yet judged
    staging     passed its gate; may run in shadow next to production
    production  the files in checkpoints/ are this version (exactly one per model)
    archived    was production; `rollback` brings the last one back

Promotion copies the version's files into checkpoints/ (each written to a temporary name, then
renamed over the old one), after a gate: the primary metric must reach the spec's floor and may
not fall more than max_drop below production, and every guard metric has its own limit. A failed
gate blocks promotion unless forced, and a forced promotion is recorded as such. If the files in
checkpoints/ no longer match the production version (someone copied a model in by hand), promotion
refuses to overwrite them until they are registered, so no unregistered model is ever lost.

Every register, stage change, promotion and rollback is written to an event log.
"""
import fnmatch
import glob
import json
import os
import shutil
import threading
import time

from mlops.paths import CKPT_DIR, connect, dig, git_commit, sha256_file, store_dir
from mlops.specs import SPECS, spec

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    model       TEXT NOT NULL,
    version     INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    stage       TEXT NOT NULL,
    files       TEXT NOT NULL,
    fileset     TEXT NOT NULL,
    metrics     TEXT NOT NULL DEFAULT '{}',
    run_id      TEXT,
    git_commit  TEXT,
    source      TEXT,
    note        TEXT,
    PRIMARY KEY (model, version)
);
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    at      REAL NOT NULL,
    model   TEXT NOT NULL,
    version INTEGER,
    action  TEXT NOT NULL,
    actor   TEXT,
    detail  TEXT NOT NULL DEFAULT '{}'
);
"""
STAGES = ("candidate", "staging", "production", "archived")


class RegistryError(ValueError):
    pass


class Registry:
    def __init__(self, store=None, ckpt_dir=None):
        self.store = store or store_dir()
        self.ckpt_dir = ckpt_dir or CKPT_DIR
        self.blobs = os.path.join(self.store, "blobs")
        os.makedirs(self.blobs, exist_ok=True)
        self._lock = threading.RLock()
        self._db = connect(os.path.join(self.store, "registry.db"))
        self._db.executescript(SCHEMA)
        self.listeners = []

    def close(self):
        with self._lock:
            self._db.close()

    # -- files ---------------------------------------------------------------------------------
    def _rel(self, path):
        return os.path.relpath(path, self.ckpt_dir).replace("\\", "/")

    def files_on_disk(self, model, root=None):
        root = root or self.ckpt_dir
        out = {}
        for pat in spec(model)["files"]:
            for p in glob.glob(os.path.join(root, pat)):
                if os.path.isfile(p):
                    out[os.path.relpath(p, root).replace("\\", "/")] = p
        return dict(sorted(out.items()))

    def _belongs(self, model, rel):
        return any(fnmatch.fnmatch(rel, pat) for pat in spec(model)["files"])

    def _blob_path(self, sha):
        return os.path.join(self.blobs, sha[:2], sha)

    def _put_blob(self, path, sha):
        dst = self._blob_path(sha)
        if os.path.exists(dst):
            return dst
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        # a copy, never a hard link: training scripts rewrite checkpoints in place, which would change the
        # stored version through the shared inode
        shutil.copyfile(path, tmp)
        if sha256_file(tmp) != sha:
            os.remove(tmp)
            raise RegistryError(f"{path} changed while it was being stored")
        os.replace(tmp, dst)
        return dst

    def snapshot(self, model, root=None):
        files = []
        for rel, p in self.files_on_disk(model, root).items():
            files.append({"path": rel, "sha256": sha256_file(p), "bytes": os.path.getsize(p)})
        return files

    @staticmethod
    def _fileset(files):
        return "|".join(f"{f['path']}={f['sha256']}" for f in sorted(files, key=lambda f: f["path"]))

    def read_metrics(self, model, root=None):
        root = root or self.ckpt_dir
        out = {}
        from mlops.specs import metric_sources
        for fname, keys in metric_sources(model, root).items():
            p = os.path.join(root, fname)
            if not os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as fh:
                    rep = json.load(fh)
            except Exception:
                continue
            for k, path in keys.items():
                v = dig(rep, path)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[k] = float(v)
        return out

    # -- records -------------------------------------------------------------------------------
    def _event(self, model, version, action, actor=None, **detail):
        self._db.execute("INSERT INTO events(at, model, version, action, actor, detail) VALUES(?,?,?,?,?,?)",
                         (time.time(), model, version, action, actor, json.dumps(detail)))
        for fn in self.listeners:
            try:
                fn({"model": model, "version": version, "action": action, "actor": actor, **detail})
            except Exception:
                pass

    def _row(self, r):
        if r is None:
            return None
        out = dict(r)
        out["files"] = json.loads(out["files"])
        out["metrics"] = json.loads(out["metrics"] or "{}")
        out.pop("fileset", None)
        out["bytes"] = sum(f["bytes"] for f in out["files"])
        return out

    def get(self, model, version):
        with self._lock:
            return self._row(self._db.execute("SELECT * FROM versions WHERE model=? AND version=?",
                                              (model, int(version))).fetchone())

    def versions(self, model):
        with self._lock:
            return [self._row(r) for r in self._db.execute(
                "SELECT * FROM versions WHERE model=? ORDER BY version DESC", (model,))]

    def in_stage(self, model, stage):
        with self._lock:
            return self._row(self._db.execute(
                "SELECT * FROM versions WHERE model=? AND stage=? ORDER BY version DESC LIMIT 1",
                (model, stage)).fetchone())

    def production(self, model):
        return self.in_stage(model, "production")

    def events(self, model=None, limit=100):
        with self._lock:
            q = "SELECT * FROM events" + (" WHERE model=?" if model else "") + " ORDER BY id DESC LIMIT ?"
            rows = self._db.execute(q, ((model, int(limit)) if model else (int(limit),))).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"] or "{}")} for r in rows]

    # -- register ------------------------------------------------------------------------------
    def register(self, model, root=None, stage="candidate", run_id=None, source="register", note=None,
                 actor="cli", metrics=None):
        """Record the model's files under `root` (default checkpoints/) as a version.
        Registering the same bytes twice returns the existing version."""
        spec(model)
        if stage not in STAGES or stage == "archived":
            raise RegistryError(f"cannot register straight into {stage}")
        files = self.snapshot(model, root)
        if not files:
            raise RegistryError(f"no files for {model} under {root or self.ckpt_dir}")
        fileset = self._fileset(files)
        with self._lock:
            same = self._db.execute("SELECT * FROM versions WHERE model=? AND fileset=?", (model, fileset)).fetchone()
            if same is not None:
                return self._row(same)
            src_root = root or self.ckpt_dir
            for f in files:
                self._put_blob(os.path.join(src_root, f["path"]), f["sha256"])
            m = self.read_metrics(model, root)
            m.update(metrics or {})
            n = (self._db.execute("SELECT MAX(version) v FROM versions WHERE model=?", (model,)).fetchone()["v"] or 0) + 1
            if stage == "production" and self.production(model) is not None:
                raise RegistryError("there is already a production version; register as candidate and promote it")
            self._db.execute("""INSERT INTO versions(model, version, created_at, stage, files, fileset, metrics, run_id,
                                git_commit, source, note) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                             (model, n, time.time(), stage, json.dumps(files), fileset, json.dumps(m), run_id,
                              git_commit(), source, note))
            self._event(model, n, "register", actor, stage=stage, source=source, run_id=run_id)
            return self.get(model, n)

    def bootstrap(self, actor="bootstrap"):
        """First run: whatever is in checkpoints/ now becomes version 1 in production."""
        done = {}
        for name in SPECS:
            if self.production(name) is not None or not self.files_on_disk(name):
                continue
            v = self.register(name, stage="production", source="bootstrap", actor=actor,
                              note="the files in checkpoints/ when the registry was created")
            done[name] = v["version"]
        return done

    # -- judging and moving ---------------------------------------------------------------------
    def gate(self, model, version):
        s = spec(model)
        g = s.get("gate") or {}
        cand = self.get(model, version)
        if cand is None:
            raise RegistryError(f"{model} v{version} does not exist")
        prod = self.production(model)
        checks = []
        if not g.get("primary"):
            return {"passed": False, "manual": True, "checks": [], "reason": g.get("manual", "no automatic gate")}
        cm, pm = cand["metrics"], (prod or {}).get("metrics", {})
        key, direction = g["primary"], g.get("direction", "max")
        v = cm.get(key)
        if v is None:
            checks.append({"metric": key, "passed": False, "why": "the candidate has no value for its primary metric"})
        else:
            if g.get("floor") is not None:
                ok = v >= g["floor"] if direction == "max" else v <= g["floor"]
                checks.append({"metric": key, "value": v, "limit": g["floor"], "rule": "floor", "passed": ok})
            if key in pm and prod and prod["version"] != cand["version"]:
                lim = pm[key] - g.get("max_drop", 0) if direction == "max" else pm[key] + g.get("max_drop", 0)
                ok = v >= lim if direction == "max" else v <= lim
                checks.append({"metric": key, "value": v, "production": pm[key], "limit": round(lim, 6),
                               "rule": "no worse than production", "passed": ok})
        for gk, rule in (g.get("guards") or {}).items():
            cv, pv = cm.get(gk), pm.get(gk)
            if cv is None:
                if rule.get("ceiling") is not None or (pv is not None and prod and prod["version"] != cand["version"]):
                    checks.append({"metric": gk, "value": None, "rule": "guard", "passed": False,
                                   "why": "the candidate's report does not have this metric"})
                continue
            if rule.get("ceiling") is not None:
                checks.append({"metric": gk, "value": cv, "limit": rule["ceiling"], "rule": "ceiling",
                               "passed": cv <= rule["ceiling"]})
            if pv is None or not prod or prod["version"] == cand["version"]:
                continue
            if rule.get("direction", "max") == "max":
                lim = pv - rule.get("max_drop", 0)
                ok = cv >= lim
            else:
                lim = pv + rule.get("max_rise", 0)
                ok = cv <= lim
            checks.append({"metric": gk, "value": cv, "production": pv, "limit": round(lim, 6),
                           "rule": "guard", "passed": ok})
        passed = bool(checks) and all(c["passed"] for c in checks)
        return {"passed": passed, "manual": False, "checks": checks,
                "against": prod["version"] if prod else None}

    def set_stage(self, model, version, stage, actor="cli"):
        if stage not in ("candidate", "staging", "archived"):
            raise RegistryError("use promote() to make a version production")
        with self._lock:
            v = self.get(model, version)
            if v is None:
                raise RegistryError(f"{model} v{version} does not exist")
            if v["stage"] == "production":
                raise RegistryError("the production version changes stage only when another is promoted")
            if stage == "staging":
                for r in self._db.execute("SELECT version FROM versions WHERE model=? AND stage='staging'", (model,)):
                    self._db.execute("UPDATE versions SET stage='candidate' WHERE model=? AND version=?",
                                     (model, r["version"]))
            self._db.execute("UPDATE versions SET stage=? WHERE model=? AND version=?", (stage, model, int(version)))
            self._event(model, int(version), f"stage:{stage}", actor)
            return self.get(model, version)

    def version_on_disk(self, model):
        """The registered version whose files are exactly what is in checkpoints/ now, or None."""
        files = self.snapshot(model)
        if not files:
            return None
        with self._lock:
            r = self._db.execute("SELECT version FROM versions WHERE model=? AND fileset=?",
                                 (model, self._fileset(files))).fetchone()
        return r["version"] if r else None

    def served_state(self, model):
        """Do the files in checkpoints/ match the production version?"""
        prod = self.production(model)
        disk = {f["path"]: f["sha256"] for f in self.snapshot(model)}
        if prod is None:
            return {"registered": False, "matches": False, "files_on_disk": len(disk)}
        want = {f["path"]: f["sha256"] for f in prod["files"]}
        changed = sorted(p for p in set(want) | set(disk) if want.get(p) != disk.get(p))
        return {"registered": True, "matches": not changed, "changed_files": changed, "version": prod["version"]}

    def promote(self, model, version, actor="cli", force=False, reason=None, action="promote",
                overwrite_unregistered=False):
        """Deploy a version. `force` overrides a failed gate only; files in checkpoints/ that no version
        holds are never overwritten unless `overwrite_unregistered` (and even then they are kept in the
        blob store first, so nothing is lost)."""
        with self._lock:
            cand = self.get(model, version)
            if cand is None:
                raise RegistryError(f"{model} v{version} does not exist")
            if cand["stage"] == "production":
                raise RegistryError(f"{model} v{version} is already in production")
            gate = self.gate(model, version)
            if not gate["passed"] and not force:
                raise RegistryError(f"gate failed for {model} v{version}: " + json.dumps(gate["checks"] or gate.get("reason")))
            known = {f["sha256"] for r in self.versions(model) for f in r["files"]}
            on_disk = {rel: sha256_file(p) for rel, p in self.files_on_disk(model).items()}
            unregistered = sorted(rel for rel, sha in on_disk.items() if sha not in known)
            if unregistered and not overwrite_unregistered:
                raise RegistryError("checkpoints/ holds files no registered version has ("
                                    + ", ".join(unregistered[:5])
                                    + f"); register them first: python -m mlops register {model}")
            for f in cand["files"]:
                if not os.path.exists(self._blob_path(f["sha256"])):
                    raise RegistryError(f"blob for {f['path']} is missing from the store")
            # every file about to be overwritten or removed is in the blob store before anything changes
            for rel, sha in on_disk.items():
                self._put_blob(os.path.join(self.ckpt_dir, rel), sha)
            prev = self.production(model)
            for f in cand["files"]:
                dst = os.path.join(self.ckpt_dir, f["path"])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                tmp = dst + ".deploying"
                shutil.copyfile(self._blob_path(f["sha256"]), tmp)
                os.replace(tmp, dst)
            keep = {f["path"] for f in cand["files"]}
            for rel, p in self.files_on_disk(model).items():
                if rel not in keep:
                    os.remove(p)
            if prev is not None:
                self._db.execute("UPDATE versions SET stage='archived' WHERE model=? AND version=?",
                                 (model, prev["version"]))
            self._db.execute("UPDATE versions SET stage='production' WHERE model=? AND version=?", (model, int(version)))
            self._event(model, int(version), action, actor, previous=prev["version"] if prev else None,
                        forced=bool(force and not gate["passed"]), gate=gate, reason=reason,
                        overwrote_unregistered=unregistered if unregistered else None)
            return {"model": model, "version": int(version), "previous": prev["version"] if prev else None,
                    "gate": gate, "forced": bool(force and not gate["passed"])}

    def production_history(self, model):
        """Versions in the order they became production, with rollbacks popped off: the last entry is
        production, the one before it is where a rollback goes."""
        stack = []
        for e in reversed(self.events(model, limit=100000)):
            if e["action"] == "register" and e["detail"].get("stage") == "production":
                stack.append(e["version"])
            elif e["action"] == "promote":
                stack.append(e["version"])
            elif e["action"] == "rollback":
                if stack:
                    stack.pop()
                if not stack or stack[-1] != e["version"]:
                    stack.append(e["version"])
        return stack

    def rollback(self, model, actor="cli", reason=None):
        """Bring back the version that was in production before the current one. A second rollback goes one
        further back, never forward to the version just rolled back."""
        prod = self.production(model)
        if prod is None:
            raise RegistryError(f"{model} has no production version")
        stack = self.production_history(model)
        while stack and stack[-1] == prod["version"]:
            stack.pop()
        if not stack:
            raise RegistryError(f"{model} v{prod['version']} has no earlier production version to return to")
        return self.promote(model, stack[-1], actor=actor, force=True, reason=reason or "rollback", action="rollback")

    def materialize(self, model, version, dest):
        """Write a version's files into `dest` (same layout as checkpoints/), e.g. to run it in shadow."""
        v = self.get(model, version)
        if v is None:
            raise RegistryError(f"{model} v{version} does not exist")
        for f in v["files"]:
            p = os.path.join(dest, f["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p) or sha256_file(p) != f["sha256"]:
                shutil.copyfile(self._blob_path(f["sha256"]), p)
        return dest

    def restore(self, files, root=None):
        """Put files (as recorded by snapshot) back from the blob store."""
        root = root or self.ckpt_dir
        for f in files:
            dst = os.path.join(root, f["path"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".restoring"
            shutil.copyfile(self._blob_path(f["sha256"]), tmp)
            os.replace(tmp, dst)

    def keep(self, files, root=None):
        """Store the blobs of a snapshot without registering it (so restore() can use them)."""
        root = root or self.ckpt_dir
        for f in files:
            self._put_blob(os.path.join(root, f["path"]), f["sha256"])

    # -- overview ---------------------------------------------------------------------------------
    def overview(self):
        out = []
        for name, s in SPECS.items():
            vs = self.versions(name)
            prod = next((v for v in vs if v["stage"] == "production"), None)
            stag = next((v for v in vs if v["stage"] == "staging"), None)
            state = self.served_state(name) if (prod or self.files_on_disk(name)) else {"registered": False, "matches": False}
            out.append({
                "model": name, "title": s["title"], "task": s["task"],
                "train": s["train"] if isinstance(s["train"], str) else " ".join(s["train"]),
                "production": prod and {k: prod[k] for k in ("version", "created_at", "metrics", "run_id",
                                                             "git_commit", "source", "bytes")},
                "staging": stag and {k: stag[k] for k in ("version", "created_at", "metrics", "run_id")},
                "versions": len(vs),
                "served_files_match": state.get("matches"),
                "changed_files": state.get("changed_files", []),
                "on_disk": bool(self.files_on_disk(name)),
                "gate": {k: v for k, v in (s.get("gate") or {}).items()},
            })
        return out
