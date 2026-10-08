"""
The MLOps side of the API server: model registry, monitoring, shadow, active learning, traffic.

server.py calls setup() once and then offers each GET/POST to handle_get()/handle_post(), which
return True when they answered. Reads are open (they show model metadata and aggregate numbers),
except the active-learning photographs; everything that changes a model, a label or a stage
needs the operator key, like the other write endpoints.
"""
import os
import threading
import time
import urllib.parse

PROTECTED_POST = {
    "/api/v1/mlops/promote", "/api/v1/mlops/rollback", "/api/v1/mlops/stage", "/api/v1/mlops/register",
    "/api/v1/mlops/reload", "/api/v1/mlops/al/label", "/api/v1/mlops/al/export", "/api/v1/mlops/rebaseline",
}
PROTECTED_GET = {"/api/v1/mlops/al/image", "/api/v1/mlops/al/queue"}


class MLOps:
    def __init__(self, writable_dir, ckpt_dir, pipeline, edge_ingest=None, events=None, alerts=None,
                 reload_pipeline=None):
        from mlops.monitor import Monitor, ShadowRunner
        from mlops.active_learning import ActiveLearningQueue
        from pipeline.traffic import TrafficEstimator
        self.ckpt_dir = ckpt_dir
        self.events = events
        self.reload_pipeline = reload_pipeline
        db = os.path.join(writable_dir, "road_shield.db")
        self.monitor = Monitor(db, os.path.join(ckpt_dir, "monitoring_reference.json"), events=events, alerts=alerts)
        self.traffic = TrafficEstimator(db, events=events)
        self.al = ActiveLearningQueue(db, os.path.join(writable_dir, "al_images"), events=events)
        self.registry, self.registry_error = None, None
        self.reload_state = {"state": "idle"}
        self._reload_lock = threading.Lock()
        self._reload_again = False
        self._loaded_versions = {}
        try:
            from mlops.registry import Registry
            # the same store as the command line (ROAD_SHIELD_MLOPS_DIR, else the repository's mlops_store/), so a
            # promotion from either side is one history over one checkpoints/
            self.registry = Registry(ckpt_dir=ckpt_dir)
            self.registry.listeners.append(self._on_registry_event)
        except Exception as e:
            self.registry_error = f"registry unavailable: {e}"
        self.shadow = None
        if self.registry is not None:
            self.shadow = ShadowRunner(self.registry, self.monitor, os.path.join(writable_dir, "shadow"))
        self.pipeline = pipeline
        self.attach(pipeline)
        if edge_ingest is not None:
            edge_ingest.event_listeners.append(self._on_edge_event)
        threading.Thread(target=self._bootstrap, daemon=True).start()

    # -- wiring ---------------------------------------------------------------------------------
    def attach(self, pipeline):
        self.pipeline = pipeline
        if self._after_audit not in pipeline.audit_hooks:
            pipeline.audit_hooks.append(self._after_audit)
        self._loaded_versions = self._versions_on_disk()
        pipeline.model_versions = dict(self._loaded_versions)

    def _bootstrap(self):
        if self.registry is None:
            return
        try:
            self.registry.bootstrap(actor="server")
            if self.pipeline is not None:
                self._loaded_versions = self._versions_on_disk()
                self.pipeline.model_versions = dict(self._loaded_versions)
            if self.shadow is not None:
                self.shadow.refresh()
        except Exception as e:
            self.registry_error = f"bootstrap failed: {e}"

    SERVED = ("vision_classifier", "segmenter", "road_damage_detector", "coco_detector", "ood_guard",
              "imu_classifier", "pci_regressor", "depth_estimator", "deterioration_forecaster")

    def _versions_on_disk(self):
        """Which registered version each model's files are, read when the models are loaded: that is what is
        in memory, even if checkpoints/ changes later (a promotion from the command line needs a reload)."""
        if self.registry is None:
            return {}
        out = {}
        for name in self.SERVED:
            try:
                v = self.registry.version_on_disk(name)
            except Exception:
                v = None
            if v is not None:
                out[name] = v
            elif self.registry.files_on_disk(name):
                out[name] = "unregistered"
        return out

    def versions(self):
        return dict(self._loaded_versions)

    def _after_audit(self, img, result, source, projection, latency_ms):
        if source == "dataset":
            return              # the project's own photographs say nothing about production and must not be relabelled
        versions = getattr(self.pipeline, "model_versions", None)
        self.monitor.record(source, result, latency_ms=latency_ms, versions=versions)
        if source != "citizen":
            self.al.consider(img, result, source, projection=projection)
        if self.shadow is not None and self.shadow.active:
            self.shadow.submit(img, result.get("frame_classification"))

    def _on_edge_event(self, bus, kind, event, at):
        if kind == "trf":
            self.traffic.observe(event["lat"], event["lon"], event.get("veh") or {}, speed_kmh=event.get("spd"),
                                 at=at, frames=int(event.get("n") or 1), bus_id=bus, source="fleet")
        else:
            self.monitor.record_fleet_event(bus, kind, event, at)

    def _on_registry_event(self, ev):
        if self.events is not None:
            try:
                self.events.publish("model_registry", ev)
            except Exception:
                pass

    def _reload(self, reason):
        """Load the models again from checkpoints/ in the background and swap them in."""
        if self.reload_pipeline is None:
            self.reload_state = {"state": "manual", "note": "restart the server to load the new files"}
            return
        self.reload_state = {"state": "reloading", "since": time.time(), "reason": reason}

        def work():
            # one reload at a time; a promotion during a reload triggers one more pass, so the models loaded
            # last are always the files deployed last
            while True:
                self._reload_again = False
                try:
                    with self.registry._lock if self.registry is not None else threading.Lock():
                        new = self.reload_pipeline()      # files cannot change while they are read
                    self.attach(new)
                    self.monitor.reload_reference()
                    if self.shadow is not None:
                        self.shadow.refresh()
                    self.reload_state = {"state": "done", "at": time.time(), "reason": reason,
                                         "versions": self.versions()}
                except Exception as e:
                    self.reload_state = {"state": "failed", "at": time.time(), "error": str(e)}
                if not self._reload_again:
                    break
            self._reload_lock.release()

        if not self._reload_lock.acquire(blocking=False):
            self._reload_again = True
            return
        threading.Thread(target=work, daemon=True).start()

    # -- views ----------------------------------------------------------------------------------
    def overview(self):
        from mlops import tracking
        guard = getattr(self.pipeline, "input_guard", None)
        reg = None
        if self.registry is not None:
            try:
                reg = self.registry.overview()
            except Exception as e:
                self.registry_error = str(e)
        try:
            runs = tracking.list_runs(limit=15, db_path=os.path.join(self.registry.store, "runs.db")
                                      if self.registry is not None else None)
        except Exception:
            runs = []
        return {
            "registry": reg, "registry_error": self.registry_error,
            "serving": self.versions(), "reload": self.reload_state,
            "input_guard": guard.describe() if guard is not None else {"ready": False},
            "monitoring": self.monitor.drift(hours=168),
            "shadow": self.shadow.describe() if self.shadow is not None else None,
            "active_learning": self.al.stats(),
            "traffic": self.traffic.stats(),
            "runs": runs,
        }


def handle_get(h, m, path, full_path, key_ok):
    q = urllib.parse.parse_qs(urllib.parse.urlparse(full_path).query)
    one = lambda k, d=None: (q.get(k) or [d])[0]  # noqa: E731
    if not (path.startswith("/api/v1/mlops") or path.startswith("/api/v1/traffic")):
        return False
    if m is None:
        h._send_json(503, {"error": "MLOps is not running on this server"})
        return True
    if path in PROTECTED_GET and not key_ok(h.headers):
        h._send_json(401, {"error": "this endpoint requires an API key (X-API-Key header or Authorization: Bearer <key>)"})
        return True
    if path == "/api/v1/mlops/overview":
        h._send_json(200, m.overview())
    elif path == "/api/v1/mlops/drift":
        try:
            hours = max(1.0, min(24 * 90.0, float(one("hours", 168))))
        except ValueError:
            h._send_json(400, {"error": "hours must be a number"})
            return True
        h._send_json(200, m.monitor.drift(hours=hours, source=one("source")))
    elif path == "/api/v1/mlops/versions":
        if m.registry is None:
            h._send_json(503, {"error": m.registry_error})
            return True
        try:
            name = one("model", "")
            h._send_json(200, {"model": name, "versions": m.registry.versions(name),
                               "events": m.registry.events(name, limit=50)})
        except KeyError as e:
            h._send_json(404, {"error": str(e)})
    elif path in ("/api/v1/mlops/gate", "/api/v1/mlops/runs", "/api/v1/mlops/run") and m.registry is None:
        h._send_json(503, {"error": m.registry_error or "registry unavailable"})
    elif path == "/api/v1/mlops/gate":
        try:
            h._send_json(200, m.registry.gate(one("model", ""), int(one("version", "0"))))
        except (KeyError, ValueError) as e:
            h._send_json(404, {"error": str(e)})
    elif path == "/api/v1/mlops/runs":
        from mlops import tracking
        h._send_json(200, {"runs": tracking.list_runs(one("experiment"), limit=100,
                                                      db_path=os.path.join(m.registry.store, "runs.db"))})
    elif path == "/api/v1/mlops/run":
        from mlops import tracking
        r = tracking.get_run(one("id", ""), db_path=os.path.join(m.registry.store, "runs.db"))
        h._send_json(200 if r else 404, r or {"error": "no such run"})
    elif path == "/api/v1/mlops/al/queue":
        status = one("status", "pending")
        if status not in ("pending", "labelled", "exported"):
            h._send_json(400, {"error": "status must be pending, labelled or exported"})
            return True
        from mlops.active_learning import CLASSES, EXTRA_LABELS
        try:
            limit = max(1, min(500, int(one("limit", "60"))))
        except ValueError:
            h._send_json(400, {"error": "limit must be a whole number"})
            return True
        h._send_json(200, {"items": m.al.list(status, limit=limit), "stats": m.al.stats(),
                           "labels": CLASSES + EXTRA_LABELS})
    elif path == "/api/v1/mlops/al/image":
        data = m.al.image(one("id", ""))
        if data is None:
            h._send_json(404, {"error": "no such photograph"})
            return True
        h.send_response(200)
        h.send_header("Content-Type", "image/jpeg")
        h.send_header("Content-Length", str(len(data)))
        h.send_header("Cache-Control", "private, max-age=600")
        h._send_cors_headers()
        h.end_headers()
        h.wfile.write(data)
    elif path == "/api/v1/traffic/cells":
        h._send_json(200, {"cells": m.traffic.cells(), **m.traffic.stats()})
    elif path == "/api/v1/traffic/estimate":
        try:
            lat, lon = float(one("lat")), float(one("lon"))
        except (TypeError, ValueError):
            h._send_json(400, {"error": "lat and lon are required"})
            return True
        h._send_json(200, {"lat": lat, "lon": lon, "estimate": m.traffic.estimate(lat, lon)})
    else:
        return False
    return True


def handle_post(h, m, path, body, actor):
    if not path.startswith("/api/v1/mlops"):
        return False
    if m is None or (m.registry is None and not path.startswith(("/api/v1/mlops/al", "/api/v1/mlops/rebaseline"))):
        h._send_json(503, {"error": (m.registry_error if m else None) or "MLOps is not running on this server"})
        return True
    from mlops.registry import RegistryError
    body = body if isinstance(body, dict) else {}
    try:
        if path == "/api/v1/mlops/promote":
            res = m.registry.promote(str(body.get("model")), int(body.get("version")), actor=actor,
                                     force=bool(body.get("force")), reason=str(body.get("reason") or "")[:200] or None)
            m._reload(f"promoted {res['model']} v{res['version']}")
            h._send_json(200, {**res, "reload": m.reload_state})
        elif path == "/api/v1/mlops/rollback":
            res = m.registry.rollback(str(body.get("model")), actor=actor,
                                      reason=str(body.get("reason") or "")[:200] or None)
            m._reload(f"rolled back {res['model']} to v{res['version']}")
            h._send_json(200, {**res, "reload": m.reload_state})
        elif path == "/api/v1/mlops/stage":
            res = m.registry.set_stage(str(body.get("model")), int(body.get("version")), str(body.get("stage")),
                                       actor=actor)
            if m.shadow is not None:
                m.shadow.refresh()
            h._send_json(200, res)
        elif path == "/api/v1/mlops/register":
            res = m.registry.register(str(body.get("model")), stage="candidate", actor=actor, source="api",
                                      note=str(body.get("note") or "")[:200] or None)
            h._send_json(200, {**res, "gate": m.registry.gate(res["model"], res["version"])})
        elif path == "/api/v1/mlops/rebaseline":
            if body.get("reset"):
                m.monitor.reset_reference()
                h._send_json(200, {"reference": "training"})
            else:
                hours = max(24.0, min(24 * 90.0, float(body.get("hours") or 168)))
                ref = m.monitor.rebaseline(hours=hours, actor=actor)
                h._send_json(200, {"reference": "production", "built_from": ref["built_from"]})
        elif path == "/api/v1/mlops/reload":
            m._reload("requested")
            h._send_json(202, m.reload_state)
        elif path == "/api/v1/mlops/al/label":
            try:
                h._send_json(200, m.al.label(str(body.get("item_id")), str(body.get("label")), actor=actor))
            except KeyError:
                h._send_json(404, {"error": "no such item"})
        elif path == "/api/v1/mlops/al/export":
            batch, data, manifest = m.al.export(include_exported=bool(body.get("include_exported")))
            if not manifest["items"]:
                h._send_json(409, {"error": "nothing labelled to export"})
                return True
            h.send_response(200)
            h.send_header("Content-Type", "application/zip")
            h.send_header("Content-Disposition", f'attachment; filename="{batch}.zip"')
            h.send_header("Content-Length", str(len(data)))
            h._send_cors_headers()
            h.end_headers()
            h.wfile.write(data)
        else:
            return False
    except (RegistryError, ValueError, TypeError) as e:
        h._send_json(409 if isinstance(e, RegistryError) else 400, {"error": str(e)})
    except KeyError as e:
        h._send_json(404, {"error": str(e)})
    return True
