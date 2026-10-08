import hashlib
import os
import sqlite3
import subprocess

ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")


def store_dir():
    """Where runs, the registry and model blobs live. Not in git: it holds every version's weights."""
    d = os.environ.get("ROAD_SHIELD_MLOPS_DIR") or os.path.join(ENGINE_ROOT, "mlops_store")
    os.makedirs(d, exist_ok=True)
    return d


def connect(path):
    db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=15)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    return db


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_GIT = {}


def git_commit():
    if "c" not in _GIT:
        try:
            out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ENGINE_ROOT,
                                 capture_output=True, text=True, timeout=5)
            sha = out.stdout.strip() or None
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ENGINE_ROOT,
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            _GIT["c"] = f"{sha}+dirty" if sha and dirty else sha
        except Exception:
            _GIT["c"] = None
    return _GIT["c"]


def dig(obj, path, default=None):
    """dig(report, "test.combined.auroc") -> value or default."""
    cur = obj
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur
