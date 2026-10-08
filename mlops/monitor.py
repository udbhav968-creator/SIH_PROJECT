"""
What the served models see in production, compared with what they were trained on.

Every analysed photograph (uploads, citizen reports, video frames, bus frames) adds one row:
source, the versions that served it, the classifier's top class and confidence, the input guard's
verdict and novelty score, four image-quality measures and the latency. Bus events add the class
and confidence the bus reported. Nothing identifying is stored; no image, no location.

Drift is the Population Stability Index of each measure in a recent window against the histogram
in checkpoints/monitoring_reference.json (written by training/train_ood_guard.py from held-out road
photographs):

    PSI = sum over bins of (p_now - p_ref) * ln(p_now / p_ref)
    < 0.10 stable     0.10-0.25 watch     > 0.25 drift (the usual credit-scoring convention)

On a small window PSI is mostly sampling noise, so a shift is only called watch or drift when a
two-sample chi-square test (the reference is itself a sample; sparse bins merged) also says it is
unlikely to be chance, at 0.01 split across the measures tested (Bonferroni).

and of the class mix against the reference class mix. A window with fewer than MIN_SAMPLES rows
is reported as "not enough data" rather than judged. The out-of-distribution rate, latency
percentiles and a per-day series are reported beside it. When the overall status changes to drift
an alert goes out on the live feed and, if configured, to the webhook (pipeline/alerts.py).

Shadow testing: when a model has a version in "staging" (mlops/registry.py) the shadow runner loads
it next to production and runs it on a sample of the same photographs in a background thread,
recording whether it agrees with production. The request never waits for the shadow.
"""
import json
import math
import os
import queue
import sqlite3
import statistics
import threading
import time

from mlops.paths import CKPT_DIR

MAX_ROWS = 50000
MIN_SAMPLES = 50
MIN_REBASELINE = 200
PSI_WATCH, PSI_DRIFT = 0.10, 0.25
P_VALUE = 0.01
QUALITY = ("brightness", "contrast", "clipped_fraction")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          REAL NOT NULL,
    source      TEXT NOT NULL,
    versions    TEXT,
    top_class   TEXT,
    confidence  REAL,
    distress    TEXT,
    verdict     TEXT,
    novelty     REAL,
    brightness  REAL,
    contrast    REAL,
    sharpness_log10 REAL,
    clipped_fraction REAL,
    latency_ms  REAL
);
CREATE INDEX IF NOT EXISTS idx_pred_at ON predictions(at);
CREATE TABLE IF NOT EXISTS shadow (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          REAL NOT NULL,
    model       TEXT NOT NULL,
    production  INTEGER,
    candidate   INTEGER,
    prod_class  TEXT,
    cand_class  TEXT,
    agree       INTEGER NOT NULL,
    prod_conf   REAL,
    cand_conf   REAL,
    cand_ms     REAL
);
CREATE INDEX IF NOT EXISTS idx_shadow ON shadow(model, candidate, at);
"""


def _num(v, lo=None, hi=None):
    """A finite number in range, or None (a NaN from a bus must not reach the page's JSON)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return None
    return float(v)


def psi(ref_probs, now_probs, eps=1e-4):
    total = 0.0
    for e, a in zip(ref_probs, now_probs):
        e, a = max(e, eps), max(a, eps)
        total += (a - e) * math.log(a / e)
    return total


def bin_probs(values, edges):
    """Share of values per bin, with numpy.histogram's convention (a value equal to an edge goes to the bin
    above it), which is how the reference histograms were built."""
    import bisect
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[bisect.bisect_right(edges, v)] += 1
    n = max(1, len(values))
    return [c / n for c in counts]


def _status(p, p_value=None, alpha=None):
    # PSI on a small window is mostly sampling noise (about (bins-1)/n when nothing has changed), so a shift
    # that the two-sample test cannot tell from chance is reported as stable whatever its PSI
    if p_value is not None and p_value >= (alpha if alpha is not None else P_VALUE):
        return "stable"
    return "stable" if p < PSI_WATCH else "watch" if p < PSI_DRIFT else "drift"


def two_sample_p(ref_probs, ref_n, now_counts, min_expected=5.0):
    """p-value that production counts and the reference histogram (itself a sample of ref_n) come from the same
    distribution: Pearson chi-square on the 2 x k table, with neighbouring bins merged until every expected
    count is at least `min_expected`. None when SciPy is missing."""
    try:
        from scipy.stats import chi2
    except Exception:
        return None
    ref_n = max(1.0, float(ref_n or 500))
    ref_counts = [max(0.0, p) * ref_n for p in ref_probs]
    n_now = float(sum(now_counts))
    total = ref_n + n_now
    if n_now <= 0:
        return None
    small = min(ref_n, n_now)
    cols, acc_r, acc_n = [], 0.0, 0.0
    for r, c in zip(ref_counts, now_counts):
        acc_r, acc_n = acc_r + r, acc_n + c
        if (acc_r + acc_n) * small / total >= min_expected:
            cols.append([acc_r, acc_n])
            acc_r = acc_n = 0.0
    if acc_r + acc_n > 0:
        if cols:
            cols[-1][0] += acc_r
            cols[-1][1] += acc_n
        else:
            cols.append([acc_r, acc_n])
    if len(cols) < 2:
        return 1.0
    stat = 0.0
    for r, c in cols:
        col = r + c
        for obs, row_total in ((r, ref_n), (c, n_now)):
            exp = col * row_total / total
            if exp > 0:
                stat += (obs - exp) ** 2 / exp
    return float(chi2.sf(stat, len(cols) - 1))


def _pct(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


class Monitor:
    def __init__(self, db_path, reference_path=None, events=None, alerts=None, clock=time.time, override_path=None):
        self.events = events
        self.alerts = alerts
        self.clock = clock
        self.reference_path = reference_path or os.path.join(CKPT_DIR, "monitoring_reference.json")
        # a reference rebuilt from production (rebaseline) takes precedence over the training one
        self.override_path = override_path or os.path.join(os.path.dirname(os.path.abspath(db_path)),
                                                           "monitoring_reference_production.json")
        self.reference = self._load_reference()
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._last_status = None
        self._since_check = 0
        self._since_prune = 0
        self._checking = threading.Lock()

    def _load_reference(self):
        for path, src in ((self.override_path, "production"), (self.reference_path, "training")):
            try:
                with open(path, encoding="utf-8") as fh:
                    ref = json.load(fh)
                ref["source"] = src
                return ref
            except Exception:
                continue
        return None

    def rebaseline(self, hours=168, actor="operator"):
        """Make the last `hours` of production the reference that later windows are compared with."""
        import numpy as np
        rows = [r for r in self._rows(self.clock() - hours * 3600) if r["source"] != "fleet_event" and r["brightness"] is not None]
        if len(rows) < MIN_REBASELINE:
            raise ValueError(f"only {len(rows)} analysed photographs in the last {hours:g} h; "
                             f"a reference needs at least {MIN_REBASELINE}")

        def hist(vals, bins=10):
            v = np.asarray([x for x in vals if x is not None], float)
            edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1))[1:-1])
            counts = np.histogram(v, bins=np.concatenate([[-np.inf], edges, [np.inf]]))[0]
            return {"edges": [round(float(e), 6) for e in edges], "probs": [round(float(c) / len(v), 6) for c in counts],
                    "n": int(len(v)), "mean": round(float(v.mean()), 4)}
        cols = {"brightness": "brightness", "contrast": "contrast", "clipped_fraction": "clipped_fraction",
                "sharpness_log10": "sharpness_log10", "novelty_score": "novelty", "top_confidence": "confidence"}
        feats = {}
        for name, col in cols.items():
            vals = [r[col] for r in rows if r[col] is not None]
            if len(vals) >= MIN_REBASELINE // 2:
                feats[name] = hist(vals)
        cls = [r["top_class"] for r in rows if r["top_class"]]
        ref = {"built_unix": int(self.clock()),
               "built_from": f"{len(rows)} production photographs over {hours:g} h, set by {actor}",
               "features": feats, "class_mix": {c: round(cls.count(c) / len(cls), 4) for c in sorted(set(cls))} if cls else {},
               "class_mix_n": len(cls)}
        tmp = self.override_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(ref, fh, indent=2)
        os.replace(tmp, self.override_path)
        self.reference = self._load_reference()
        self._last_status = None
        return ref

    def reset_reference(self):
        """Go back to the training reference."""
        try:
            os.remove(self.override_path)
        except OSError:
            pass
        self.reference = self._load_reference()

    def reload_reference(self):
        self.reference = self._load_reference()

    # -- recording -------------------------------------------------------------------------------
    def record(self, source, result, latency_ms=None, input_check=None, versions=None, at=None):
        fc = (result or {}).get("frame_classification") or {}
        pd = (result or {}).get("primary_distress") or {}
        ic = input_check or (result or {}).get("input_check") or {}
        q = ic.get("quality") or {}
        sharp = q.get("sharpness")
        sharp = _num(sharp)
        row = (at or self.clock(), str(source)[:20], json.dumps(versions or {}),
               fc.get("class_name"), _num(fc.get("confidence")),
               pd.get("class_name") if (result or {}).get("is_distress") else None,
               ic.get("verdict"), _num(ic.get("novelty_score")),
               _num(q.get("brightness")), _num(q.get("contrast")),
               math.log10(1 + sharp) if sharp is not None and sharp >= 0 else None,
               _num(q.get("clipped_fraction")),
               _num(latency_ms if latency_ms is not None else (result or {}).get("latency_ms")))
        check = False
        with self._lock:
            self._db.execute("""INSERT INTO predictions(at, source, versions, top_class, confidence, distress, verdict,
                                novelty, brightness, contrast, sharpness_log10, clipped_fraction, latency_ms)
                                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
            self._since_prune += 1
            if self._since_prune >= 500:
                self._since_prune = 0
                self._db.execute("DELETE FROM predictions WHERE id <= (SELECT MAX(id) FROM predictions) - ?", (MAX_ROWS,))
            self._since_check += 1
            if self._since_check >= 50:
                self._since_check, check = 0, True
        if check:
            # off the request thread: a drift check reads up to a day of rows
            threading.Thread(target=self._safe_check, daemon=True).start()

    def _safe_check(self):
        if not self._checking.acquire(blocking=False):
            return
        try:
            self.check_and_alert()
        except Exception as e:
            print(f"[monitor] drift check failed: {e}")
        finally:
            self._checking.release()

    def record_fleet_event(self, bus_id, kind, event, at=None):
        """A bus's own detection: class and confidence only."""
        if kind not in ("defect", "cam"):
            return
        with self._lock:
            self._db.execute("""INSERT INTO predictions(at, source, distress, confidence, verdict)
                                VALUES(?,?,?,?,?)""",
                             (at or self.clock(), "fleet_event", str(event.get("cls"))[:80],
                              _num(event.get("conf"), 0, 1),
                              "camera_only" if kind == "cam" else "felt"))

    # -- reading ---------------------------------------------------------------------------------
    def _rows(self, since, source=None):
        with self._lock:
            if source:
                return self._db.execute("SELECT * FROM predictions WHERE at>=? AND source=? ORDER BY at",
                                        (since, source)).fetchall()
            return self._db.execute("SELECT * FROM predictions WHERE at>=? ORDER BY at", (since,)).fetchall()

    def drift(self, hours=168, source=None):
        now = self.clock()
        rows = [r for r in self._rows(now - hours * 3600, source) if r["source"] != "fleet_event"]
        ref = self.reference
        out = {"window_hours": hours, "samples": len(rows), "reference": bool(ref),
               "reference_built_from": (ref or {}).get("built_from"), "reference_source": (ref or {}).get("source"),
               "features": {}, "class_mix": None}
        image_rows = [r for r in rows if r["brightness"] is not None]
        enough = len(image_rows) >= MIN_SAMPLES
        if ref:
            feats = ref.get("features", {})
            cols = {"brightness": "brightness", "contrast": "contrast", "clipped_fraction": "clipped_fraction",
                    "sharpness_log10": "sharpness_log10", "novelty_score": "novelty", "top_confidence": "confidence"}
            for name, col in cols.items():
                if name not in feats:
                    continue
                vals = [r[col] for r in image_rows if r[col] is not None]
                f = {"samples": len(vals), "reference_mean": feats[name].get("mean"),
                     "mean": round(statistics.fmean(vals), 4) if vals else None}
                if len(vals) >= MIN_SAMPLES:
                    now_p = bin_probs(vals, feats[name]["edges"])
                    p = psi(feats[name]["probs"], now_p)
                    pv = two_sample_p(feats[name]["probs"], feats[name].get("n"), [x * len(vals) for x in now_p])
                    f.update(psi=round(p, 4), p_value=None if pv is None else round(pv, 6))
                else:
                    f.update(psi=None, status="not enough data")
                out["features"][name] = f
            mix_ref = ref.get("class_mix") or {}
            cls = [r["top_class"] for r in image_rows if r["top_class"]]
            if mix_ref and len(cls) >= MIN_SAMPLES:
                names = sorted(set(mix_ref) | set(cls))
                now_p = [cls.count(n) / len(cls) for n in names]
                p = psi([mix_ref.get(n, 0.0) for n in names], now_p)
                pv = two_sample_p([mix_ref.get(n, 0.0) for n in names], ref.get("class_mix_n"),
                                  [x * len(cls) for x in now_p])
                out["class_mix"] = {"psi": round(p, 4), "p_value": None if pv is None else round(pv, 6),
                                    "now": {n: round(v, 4) for n, v in zip(names, now_p)},
                                    "reference": {n: mix_ref.get(n, 0.0) for n in names}}
            elif mix_ref:
                out["class_mix"] = {"psi": None, "status": "not enough data", "reference": mix_ref}
        verdicts = [r["verdict"] for r in image_rows if r["verdict"]]
        out["input_guard"] = {
            "checked": len(verdicts),
            "ood_rate": round(sum(v != "ok" for v in verdicts) / len(verdicts), 4) if verdicts else None,
            "by_verdict": {v: verdicts.count(v) for v in sorted(set(verdicts))},
            "reference_rate": None,
        }
        lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        out["latency_ms"] = {"p50": _pct(lat, .5), "p95": _pct(lat, .95), "p99": _pct(lat, .99), "samples": len(lat)}
        # one test per measure: Bonferroni, so that seven measures do not give seven chances of a false alarm
        tested = [f for f in out["features"].values() if f.get("psi") is not None]
        if out["class_mix"] and out["class_mix"].get("psi") is not None:
            tested.append(out["class_mix"])
        alpha = P_VALUE / max(1, len(tested))
        for f in tested:
            f["status"] = _status(f["psi"], f.get("p_value"), alpha)
        out["significance"] = {"alpha_per_measure": round(alpha, 5), "family_alpha": P_VALUE,
                               "test": "two-sample chi-square, bins merged to expected >= 5, Bonferroni"}
        statuses = [f["status"] for f in out["features"].values() if f.get("psi") is not None]
        out["drifted"] = [k for k, f in out["features"].items() if f.get("status") == "drift"]
        if out["class_mix"] and out["class_mix"].get("psi") is not None:
            statuses.append(out["class_mix"]["status"])
        if not ref:
            overall = "no reference"
        elif not enough or not statuses:
            overall = "not enough data"
        else:
            overall = "drift" if "drift" in statuses else "watch" if "watch" in statuses else "stable"
        out["status"] = overall
        out["by_source"] = {}
        for r in rows:
            out["by_source"][r["source"]] = out["by_source"].get(r["source"], 0) + 1
        fleet = [r for r in self._rows(now - hours * 3600, "fleet_event")]
        out["fleet_events"] = {"count": len(fleet),
                               "mean_confidence": round(statistics.fmean([r["confidence"] for r in fleet if r["confidence"] is not None]), 4)
                               if any(r["confidence"] is not None for r in fleet) else None,
                               "camera_only_share": round(sum(r["verdict"] == "camera_only" for r in fleet) / len(fleet), 4)
                               if fleet else None}
        out["daily"] = self.daily(days=max(1, int(math.ceil(hours / 24))))
        return out

    def prometheus(self, hours=24):
        """Drift, OOD rate and model latency for the last day, in Prometheus text format (appended to /metrics)."""
        d = self.drift(hours=hours)
        lines = ["# HELP road_shield_model_psi Population Stability Index of a model input or output against the training reference (24 h)",
                 "# TYPE road_shield_model_psi gauge"]
        for k, f in d["features"].items():
            if f.get("psi") is not None:
                lines.append(f'road_shield_model_psi{{measure="{k}"}} {f["psi"]}')
        if (d.get("class_mix") or {}).get("psi") is not None:
            lines.append(f'road_shield_model_psi{{measure="class_mix"}} {d["class_mix"]["psi"]}')
        lines += ["# HELP road_shield_model_drift 1 when any measure is in drift (PSI > 0.25 and significant)",
                  "# TYPE road_shield_model_drift gauge", f"road_shield_model_drift {int(d['status'] == 'drift')}",
                  "# HELP road_shield_input_ood_ratio Share of analysed photographs the input guard did not pass (24 h)",
                  "# TYPE road_shield_input_ood_ratio gauge",
                  f"road_shield_input_ood_ratio {d['input_guard']['ood_rate'] if d['input_guard']['ood_rate'] is not None else 'NaN'}",
                  "# HELP road_shield_model_latency_ms Whole-pipeline latency per photograph (24 h)",
                  "# TYPE road_shield_model_latency_ms gauge"]
        for q in ("p50", "p95", "p99"):
            v = d["latency_ms"].get(q)
            lines.append(f'road_shield_model_latency_ms{{quantile="{q}"}} {v if v is not None else "NaN"}')
        lines += ["# HELP road_shield_model_predictions Photographs analysed (24 h)", "# TYPE road_shield_model_predictions gauge",
                  f"road_shield_model_predictions {d['samples']}"]
        return "\n".join(lines) + "\n"

    def daily(self, days=14):
        now = self.clock()
        start = now - days * 86400
        buckets = {}
        for r in self._rows(start):
            if r["source"] == "fleet_event":
                continue
            d = time.strftime("%Y-%m-%d", time.localtime(r["at"]))
            b = buckets.setdefault(d, {"n": 0, "conf": [], "ood": 0, "checked": 0, "lat": []})
            b["n"] += 1
            if r["confidence"] is not None:
                b["conf"].append(r["confidence"])
            if r["verdict"]:
                b["checked"] += 1
                b["ood"] += r["verdict"] != "ok"
            if r["latency_ms"] is not None:
                b["lat"].append(r["latency_ms"])
        return [{"day": d, "predictions": b["n"],
                 "mean_confidence": round(statistics.fmean(b["conf"]), 4) if b["conf"] else None,
                 "ood_rate": round(b["ood"] / b["checked"], 4) if b["checked"] else None,
                 "p95_latency_ms": _pct(b["lat"], .95)} for d, b in sorted(buckets.items())]

    def check_and_alert(self):
        d = self.drift(hours=24)
        st = d["status"]
        if st == "drift" and self._last_status != "drift":
            bad = [k for k, f in d["features"].items() if f.get("status") == "drift"]
            if (d.get("class_mix") or {}).get("status") == "drift":
                bad.append("class mix")
            msg = {"kind": "model_drift", "severity": "warning", "features": bad, "samples": d["samples"],
                   "title": "Model input drift: " + ", ".join(bad),
                   "text": "Production photographs differ from the training reference on " + ", ".join(bad)
                           + ". Check the MLOps page; consider labelling recent frames and retraining."}
            if self.events is not None:
                try:
                    self.events.publish("model_drift", msg)
                except Exception:
                    pass
            if self.alerts is not None:
                try:
                    self.alerts.notify("model_drift", time.strftime("%Y-%m-%d %H"), msg["text"], msg)
                except Exception:
                    pass
        self._last_status = st
        return st

    # -- shadow ----------------------------------------------------------------------------------
    def record_shadow(self, model, production, candidate, prod_class, cand_class, prod_conf, cand_conf, cand_ms):
        with self._lock:
            self._db.execute("""INSERT INTO shadow(at, model, production, candidate, prod_class, cand_class, agree,
                                prod_conf, cand_conf, cand_ms) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                             (self.clock(), model, production, candidate, prod_class, cand_class,
                              int(prod_class == cand_class), prod_conf, cand_conf, cand_ms))

    def shadow_summary(self, model=None, candidate=None):
        q, args = "SELECT * FROM shadow WHERE 1=1", []
        if model:
            q += " AND model=?"
            args.append(model)
        if candidate is not None:
            q += " AND candidate=?"
            args.append(int(candidate))
        with self._lock:
            rows = self._db.execute(q + " ORDER BY at DESC LIMIT 5000", args).fetchall()
        if not rows:
            return {"compared": 0}
        pairs = {}
        for r in rows:
            if not r["agree"]:
                k = f"{r['prod_class']} -> {r['cand_class']}"
                pairs[k] = pairs.get(k, 0) + 1
        return {"compared": len(rows), "agreement": round(sum(r["agree"] for r in rows) / len(rows), 4),
                "production": rows[0]["production"], "candidate": rows[0]["candidate"],
                "candidate_p95_ms": _pct([r["cand_ms"] for r in rows if r["cand_ms"] is not None], .95),
                "disagreements": dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:10])}


class ShadowRunner:
    """Runs the staging version of the classifier beside production, off the request path."""

    def __init__(self, registry, monitor, workdir, model="vision_classifier", sample_rate=1.0):
        self.registry, self.monitor, self.model = registry, monitor, model
        self.workdir = workdir
        self.sample_rate = float(sample_rate)
        self.candidate = None
        self.version = None
        self.error = None
        self._q = queue.Queue(maxsize=8)
        self._n = 0
        threading.Thread(target=self._worker, daemon=True).start()

    def refresh(self):
        """Load the model's staging version, if any (call after stage changes)."""
        try:
            v = self.registry.in_stage(self.model, "staging")
        except Exception as e:
            self.error = str(e)
            return None
        if v is None:
            self.candidate, self.version = None, None
            return None
        if v["version"] == self.version:
            return self.version
        try:
            d = self.registry.materialize(self.model, v["version"], os.path.join(self.workdir, f"v{v['version']}"))
            from models.deep_vision_net import load_best_vision_model
            model, backend = load_best_vision_model(d, verbose=False)
            if backend not in ("deep_cnn", "cnn_embeddings"):
                raise RuntimeError(f"staging version loads as {backend}, not a CNN")
            self.candidate, self.version, self.error = model, v["version"], None
        except Exception as e:
            self.candidate, self.version, self.error = None, None, f"could not load staging v{v['version']}: {e}"
        return self.version

    @property
    def active(self):
        return self.candidate is not None

    def submit(self, img, frame_cls):
        if not self.active or not frame_cls or "class_name" not in frame_cls:
            return False
        self._n += 1
        if self.sample_rate < 1 and (self._n % max(1, int(round(1 / max(self.sample_rate, 1e-3))))) != 0:
            return False
        try:
            self._q.put_nowait((img, frame_cls, self.candidate, self.version))
            return True
        except queue.Full:
            return False

    def _worker(self):
        while True:
            img, fc, cand, ver = self._q.get()
            try:
                t0 = time.time()
                import numpy as np
                pr = np.asarray(cand.predict_probabilities(img)).ravel()
                ms = (time.time() - t0) * 1000
                names = list(getattr(cand, "class_names", None) or fc.get("probabilities", {}).keys())
                k = int(pr.argmax())
                prod = self.registry.production(self.model)
                self.monitor.record_shadow(self.model, prod["version"] if prod else None, ver,
                                           fc["class_name"], names[k] if k < len(names) else str(k),
                                           fc.get("confidence"), float(pr[k]), round(ms, 1))
            except Exception as e:
                self.error = f"shadow inference failed: {e}"

    def describe(self):
        return {"model": self.model, "active": self.active, "candidate_version": self.version,
                "error": self.error, "sample_rate": self.sample_rate,
                **({"results": self.monitor.shadow_summary(self.model, self.version)} if self.version else {})}
