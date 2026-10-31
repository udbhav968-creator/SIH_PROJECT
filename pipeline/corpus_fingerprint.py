"""Image hash and perceptual fingerprinting utilities."""

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
    from pipeline.corpus_policy import filter_paths
    out = {}
    for f in folders:
        files = sorted(
            os.path.relpath(p, ENGINE_ROOT).replace("\\", "/")
            for p in glob.glob(os.path.join(ENGINE_ROOT, "datasets", f, "**", "*.jpg"),
                               recursive=True)
            if "_label_conflicts" not in p)

        files, _ = filter_paths(files)
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
