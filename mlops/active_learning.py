"""
Active learning: keep the photographs the models are least sure about, so that the next round of
labelling is spent where it changes the model most.

Each analysed photograph is scored on four signals:

    entropy        how evenly the classifier spreads its probability over the 7 classes (0-1)
    margin         how close the top two classes are (1 - (p1 - p2))
    disagreement   the classifier and the region analysis (segmenter + measurement) disagree: the
                   classifier is confident of a pothole or crack and no region was found, or a region
                   was measured where the classifier is confident the road is normal; or the YOLO
                   damage detector boxed something the pipeline did not report
    unusual        the input guard found the photograph unlike the training set without calling it
                   "not a road" (the near-distribution cases a model learns most from)

    score = 0.5 entropy + 0.2 margin + 0.3 disagreement + 0.2 unusual        (kept above MIN_SCORE)

Near-duplicates are dropped by cosine similarity of the guard's 64-d embedding (a bus passing the
same spot every 20 minutes would otherwise fill the queue with one pothole); the higher-scoring
copy is kept. The queue holds CAPACITY photographs and drops the lowest-scoring one when full.

Privacy: citizen photographs and anonymous uploads to a public deployment are never kept, and neither
are the project's own dataset photographs (they would leak test images into training). Everything else
is stored only when the COCO detector ran on it, after people and number plates are blurred
(models/privacy_redactor.py).

Operators label photographs on the MLOps page; `export` writes the labelled ones as one folder per
class with a manifest, ready to add to the training corpus for the next run.
"""
import io
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
import zipfile

CAPACITY = 500
MIN_SCORE = 0.35
DUP_COSINE = 0.985
SKIP_SOURCES = {"citizen", "public", "dataset"}   # citizens and anonymous visitors are promised nothing is kept
SKIP_VERDICTS = {"not_road", "poor_quality"}
CLASSES = ["Normal Road / Sound Pavement", "Crack (Longitudinal / Transverse / Alligator)", "Pothole Cavity",
           "Waterlogging / Flooding Hazard", "Missing Zebra Crossing", "Missing Road Divider", "Damaged Traffic Sign"]
EXTRA_LABELS = ["not a road", "unusable"]
CONFIDENT = 0.6

SCHEMA = """
CREATE TABLE IF NOT EXISTS al_queue (
    item_id     TEXT PRIMARY KEY,
    at          REAL NOT NULL,
    source      TEXT NOT NULL,
    score       REAL NOT NULL,
    reasons     TEXT NOT NULL,
    predicted   TEXT,
    confidence  REAL,
    probabilities TEXT,
    projection  TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    label       TEXT,
    labelled_by TEXT,
    labelled_at REAL,
    exported_in TEXT,
    width       INTEGER,
    height      INTEGER,
    redaction   TEXT
);
CREATE INDEX IF NOT EXISTS idx_al_status ON al_queue(status, score);
"""


def acquisition(result, input_check=None):
    """(score 0-1, reasons) for one pipeline result."""
    fc = (result or {}).get("frame_classification") or {}
    probs = [float(v) for v in (fc.get("probabilities") or {}).values()]
    reasons, score = [], 0.0
    if len(probs) >= 2:
        k = len(probs)
        h = -sum(p * math.log(p) for p in probs if p > 0) / math.log(k)
        top = sorted(probs, reverse=True)
        margin = 1.0 - (top[0] - top[1])
        score += 0.5 * h + 0.2 * margin
        if h > 0.5:
            reasons.append(f"classifier unsure (entropy {h:.2f})")
        elif margin > 0.7:
            reasons.append(f"top two classes close ({top[0]:.2f} vs {top[1]:.2f})")
    distress = bool((result or {}).get("is_distress"))
    pd = (result or {}).get("primary_distress") or {}
    top_cls, top_conf = fc.get("class_name") or "", float(fc.get("confidence") or 0.0)
    disagree = None
    if top_conf >= CONFIDENT and ("Pothole" in top_cls or "Crack" in top_cls) and not distress:
        disagree = f"classifier says {top_cls.split(' (')[0]} ({top_conf:.2f}), no region measured"
    elif top_conf >= CONFIDENT and top_cls.startswith("Normal") and distress and pd.get("class_id") in (1, 2, 3):
        disagree = f"region measured as {pd.get('class_name')}, classifier says normal ({top_conf:.2f})"
    elif ((result or {}).get("road_damage_summary") or {}).get("boxes") and not distress:
        disagree = "damage detector boxed damage the pipeline did not report"
    if disagree:
        score += 0.3
        reasons.append(disagree)
    ic = input_check or (result or {}).get("input_check") or {}
    if ic.get("verdict") == "unusual":
        score += 0.2
        reasons.append("unlike the training photographs")
    return round(min(1.0, score), 4), reasons


def _cos(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


class ActiveLearningQueue:
    def __init__(self, db_path, image_dir, events=None, clock=time.time, capacity=CAPACITY):
        self.image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)
        self.events = events
        self.clock = clock
        self.capacity = capacity
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def _path(self, item_id):
        return os.path.join(self.image_dir, f"{item_id}.jpg")

    def _admission(self, score, projection):
        """(admit?, item ids to remove to make room). Call with the lock held."""
        drop = []
        pending = self._db.execute("SELECT item_id, score, projection FROM al_queue WHERE status='pending'").fetchall()
        if projection:
            for r in pending:
                if r["projection"] and _cos(projection, json.loads(r["projection"])) >= DUP_COSINE:
                    if score <= r["score"]:
                        return False, []
                    drop.append(r["item_id"])
                    break
        if len(pending) - len(drop) >= self.capacity:
            low = next((r for r in sorted(pending, key=lambda r: r["score"]) if r["item_id"] not in drop), None)
            if low is None or score <= low["score"]:
                return False, []
            drop.append(low["item_id"])
        return True, drop

    def consider(self, img, result, source, input_check=None, projection=None):
        """Queue the photograph if it is informative enough. Returns the item id or None."""
        if source in SKIP_SOURCES or img is None:
            return None
        ic = input_check or (result or {}).get("input_check") or {}
        if ic.get("verdict") in SKIP_VERDICTS:
            return None
        # Blurring people needs the COCO detector's boxes. If it did not run, an empty list would read as
        # "nobody in frame" and the photograph would be stored with faces and bodies visible: never.
        if not ((result or {}).get("scene_summary") or {}).get("available"):
            return None
        score, reasons = acquisition(result, ic)
        if score < MIN_SCORE:
            return None
        projection = projection or ic.get("projection")
        with self._lock:
            ok, _ = self._admission(score, projection)
        if not ok:
            return None
        from PIL import Image
        import numpy as np
        try:
            from models.privacy_redactor import redact
            red, rep = redact(img, detections=(result or {}).get("scene_objects") or [])
            redaction = {k: rep.get(k) for k in ("people_blurred", "faces_blurred", "plates_blurred")}
        except Exception as e:
            print(f"[active-learning] not kept, redaction failed: {e}")
            return None            # an unredacted photograph is never stored
        im = Image.fromarray(np.asarray(red, dtype=np.uint8))
        if max(im.size) > 1280:
            s = 1280 / max(im.size)
            im = im.resize((int(im.width * s), int(im.height * s)))
        item_id = "AL-" + uuid.uuid4().hex[:10].upper()
        im.save(self._path(item_id), quality=88)
        fc = (result or {}).get("frame_classification") or {}
        with self._lock:
            ok, drop = self._admission(score, projection)     # again: another request may have filled the slot
            if not ok:
                os.remove(self._path(item_id))
                return None
            for d in drop:
                self._remove(d)
            self._db.execute("""INSERT INTO al_queue(item_id, at, source, score, reasons, predicted, confidence,
                                probabilities, projection, width, height, redaction) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                             (item_id, self.clock(), str(source)[:20], score, json.dumps(reasons), fc.get("class_name"),
                              fc.get("confidence"), json.dumps(fc.get("probabilities") or {}),
                              json.dumps(projection) if projection else None, im.width, im.height, json.dumps(redaction)))
        if self.events is not None:
            try:
                self.events.publish("al_item", {"item_id": item_id, "score": score, "reasons": reasons})
            except Exception:
                pass
        return item_id

    def _remove(self, item_id):
        self._db.execute("DELETE FROM al_queue WHERE item_id=?", (item_id,))
        try:
            os.remove(self._path(item_id))
        except OSError:
            pass

    def _row(self, r):
        out = {k: r[k] for k in r.keys() if k != "projection"}
        for k in ("reasons", "probabilities", "redaction"):
            out[k] = json.loads(out[k]) if out.get(k) else ([] if k == "reasons" else {})
        return out

    def list(self, status="pending", limit=100):
        order = "score DESC" if status == "pending" else "labelled_at DESC"
        with self._lock:
            rows = self._db.execute(f"SELECT * FROM al_queue WHERE status=? ORDER BY {order} LIMIT ?",
                                    (status, int(limit))).fetchall()
        return [self._row(r) for r in rows]

    def get(self, item_id):
        with self._lock:
            r = self._db.execute("SELECT * FROM al_queue WHERE item_id=?", (item_id,)).fetchone()
        return self._row(r) if r else None

    def image(self, item_id):
        if not re.fullmatch(r"AL-[0-9A-F]{10}", str(item_id or "")):
            return None
        p = self._path(item_id)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as fh:
            return fh.read()

    def label(self, item_id, label, actor="operator"):
        if label not in CLASSES + EXTRA_LABELS:
            raise ValueError("label must be one of: " + "; ".join(CLASSES + EXTRA_LABELS))
        with self._lock:
            r = self._db.execute("SELECT status FROM al_queue WHERE item_id=?", (item_id,)).fetchone()
            if r is None:
                raise KeyError(item_id)
            if r["status"] == "exported":
                raise ValueError(f"{item_id} was already exported")
            self._db.execute("UPDATE al_queue SET status='labelled', label=?, labelled_by=?, labelled_at=? WHERE item_id=?",
                             (label, str(actor)[:40], self.clock(), item_id))
        return self.get(item_id)

    def stats(self):
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) c FROM al_queue GROUP BY status").fetchall()
            labels = self._db.execute("SELECT label, COUNT(*) c FROM al_queue WHERE label IS NOT NULL GROUP BY label").fetchall()
            agree = self._db.execute("SELECT COUNT(*) n, SUM(label=predicted) a FROM al_queue "
                                     "WHERE label IS NOT NULL AND label NOT IN ('not a road','unusable')").fetchone()
        out = {"pending": 0, "labelled": 0, "exported": 0}
        out.update({r["status"]: r["c"] for r in rows})
        out["labels"] = {r["label"]: r["c"] for r in labels}
        out["capacity"] = self.capacity
        out["classifier_agreed_with_labellers"] = round(agree["a"] / agree["n"], 4) if agree["n"] else None
        out["labelled_for_agreement"] = agree["n"]
        return out

    def export(self, include_exported=False):
        """Zip of the labelled photographs, one folder per class, with a manifest. Marks them exported."""
        with self._lock:
            q = "SELECT * FROM al_queue WHERE status IN ('labelled'" + (",'exported'" if include_exported else "") + ")"
            rows = [r for r in self._db.execute(q).fetchall() if r["label"] != "unusable"]
            batch = time.strftime("al_%Y%m%d_%H%M%S")
            buf = io.BytesIO()
            manifest = {"batch": batch, "created_unix": int(self.clock()), "items": [], "classes": CLASSES,
                        "note": "Photographs the models were least sure about, labelled by operators. People and "
                                "number plates were blurred before storage. Add each class folder to the matching "
                                "training class; 'not_a_road' is for the input guard."}
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for r in rows:
                    p = self._path(r["item_id"])
                    if not os.path.exists(p):
                        continue
                    arc = f"{batch}/{_slug(r['label'])}/{r['item_id']}.jpg"
                    z.write(p, arc)
                    manifest["items"].append({"item_id": r["item_id"], "file": arc, "label": r["label"],
                                              "predicted": r["predicted"], "score": r["score"], "source": r["source"],
                                              "labelled_by": r["labelled_by"]})
                counts = {}
                for it in manifest["items"]:
                    counts[it["label"]] = counts.get(it["label"], 0) + 1
                manifest["counts"] = counts
                z.writestr(f"{batch}/manifest.json", json.dumps(manifest, indent=2))
            for it in manifest["items"]:
                self._db.execute("UPDATE al_queue SET status='exported', exported_in=? WHERE item_id=?",
                                 (batch, it["item_id"]))
        return batch, buf.getvalue(), manifest
