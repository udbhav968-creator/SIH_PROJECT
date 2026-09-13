"""
A fingerprint of the data and the model a measurement was taken with.

Why this file exists
--------------------
A test asserted `found >= 5` out of 8 photographs. The 5 was measured here,
on a machine holding 489 photographs in datasets/02_kaggle_pothole_600 and a
segmenter calibrated to {crack 0.55, pothole 0.35}. It then ran on a machine
holding 1,404 photographs in that folder and a segmenter calibrated to
{crack 0.5, pothole 0.2}. The seeded sample drew eight photographs with ZERO
overlap with the eight the number came from, and the failure message blamed
the proposal filter.

Nothing was wrong with the build. The constant was a measurement of one
corpus and one model, written down as if it were a property of the code.

So a recorded figure now carries a fingerprint of what produced it, and a
test that finds a fingerprint it does not match refuses to judge rather than
reporting a failure it cannot support. Two machines legitimately disagree;
the mistake is letting one of them assert a number about the other.
"""

import glob
import hashlib
import os

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _sha(parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def corpus_fingerprint(folders):
    """
    Per folder: how many photographs, and a digest of WHICH ones.

    The count alone is not enough - two folders of 489 different photographs
    sample differently - and the digest alone hides the more common case,
    which is simply a bigger download.
    """
    out = {}
    for f in folders:
        files = sorted(
            os.path.relpath(p, ENGINE_ROOT).replace("\\", "/")
            for p in glob.glob(os.path.join(ENGINE_ROOT, "datasets", f, "**", "*.jpg"),
                               recursive=True)
            if "_label_conflicts" not in p)
        out[f] = {"count": len(files), "digest": _sha(files)}
    return out


def model_fingerprint(segmenter):
    """
    Taken from the LOADED model, never from its sidecar report.

    The sidecar is a description that can be replaced independently of what it
    describes - a shipped zip overwrote one - so a fingerprint read from it
    would certify the wrong thing. The digest is of the estimator file itself.
    """
    path = getattr(segmenter, "model_path", None) or os.path.join(
        ENGINE_ROOT, "checkpoints", "defect_segmenter.joblib")
    digest = None
    if os.path.exists(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()[:16]
    return {
        "segmenter_digest": digest,
        "thresholds": dict(getattr(segmenter, "thresholds", {}) or {}),
        "ready": bool(getattr(segmenter, "is_ready", False)),
    }


def fingerprint(segmenter, folders):
    return {"corpus": corpus_fingerprint(folders), "model": model_fingerprint(segmenter)}


def describe_mismatch(recorded, current):
    """A one-line reason, or None when the two agree."""
    if not recorded:
        return "no baseline recorded for this machine"
    rc, cc = (recorded.get("corpus") or {}), (current.get("corpus") or {})
    for folder, cur in cc.items():
        old = rc.get(folder)
        if old is None:
            return f"{folder}: not in the baseline"
        if old.get("digest") != cur.get("digest"):
            return (f"{folder}: {old.get('count')} photographs when the baseline was "
                    f"recorded, {cur.get('count')} here")
    rm, cm = (recorded.get("model") or {}), (current.get("model") or {})
    if rm.get("segmenter_digest") != cm.get("segmenter_digest"):
        return (f"segmenter differs: baseline {rm.get('thresholds')} "
                f"vs {cm.get('thresholds')} here")
    return None
