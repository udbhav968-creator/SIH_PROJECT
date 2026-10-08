"""
Operational guard rails for the engine: what a deployment needs that a demo does not.

  body limit     requests larger than ROAD_SHIELD_MAX_BODY_MB (default 25) are refused
                 with 413 before the body is read, so one oversized upload cannot
                 exhaust the server's memory
  rate limit     the model endpoints (image analysis, video, redaction) allow
                 ROAD_SHIELD_RATE_LIMIT requests per minute per client address
                 (default 0 = off, so a demo is never throttled); over the limit
                 the answer is 429 with Retry-After
  metrics        GET /metrics in the Prometheus text format: requests by route,
                 method and status, latency histogram, model readiness
  readiness      GET /api/v1/ready is 200 only when the models a road frame
                 needs are loaded, else 503 naming what is missing - the probe a
                 load balancer or Kubernetes should route on (/api/v1/health only
                 says the process is up)

Everything here is standard library and thread-safe; the server is threaded.
"""

import os
import threading
import time

MAX_BODY_BYTES = int(float(os.environ.get("ROAD_SHIELD_MAX_BODY_MB", "25")) * 1024 * 1024)

# Endpoints that run a model on the request. Cheap reads and the static site are never limited.
RATE_LIMITED = {
    "/api/v1/pipeline/deep-audit", "/api/v1/vision/analyze-photo", "/api/v1/vision/analyze-custom-photo",
    "/api/v1/video/ingest", "/api/v1/detect/objects", "/api/v1/pedestrian/detect",
    "/api/v1/privacy/redact", "/api/v1/telemetry/imu", "/api/v1/pipeline/batch", "/api/v1/citizen/report",
}

LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class RateLimiter:
    """Token bucket per client: `per_minute` requests, refilled continuously. 0 disables it."""

    def __init__(self, per_minute=None):
        self.per_minute = int(os.environ.get("ROAD_SHIELD_RATE_LIMIT", "0")) if per_minute is None else per_minute
        self._buckets = {}
        self._lock = threading.Lock()

    def check(self, client, now=None):
        """(allowed, retry_after_seconds)."""
        if self.per_minute <= 0:
            return True, 0
        now = time.monotonic() if now is None else now
        rate = self.per_minute / 60.0
        with self._lock:
            tokens, last = self._buckets.get(client, (float(self.per_minute), now))
            tokens = min(float(self.per_minute), tokens + (now - last) * rate)
            if tokens >= 1.0:
                self._buckets[client] = (tokens - 1.0, now)
                return True, 0
            self._buckets[client] = (tokens, now)
            return False, max(1, int((1.0 - tokens) / rate + 0.999))


class Metrics:
    """Counters and a latency histogram, exported in the Prometheus text format."""

    def __init__(self, known_routes=()):
        self.known = set(known_routes)
        self._lock = threading.Lock()
        self.requests = {}            # (route, method, status) -> count
        self.latency = {}             # route -> [bucket counts..., +Inf count, sum]
        self.started = time.time()

    def route_of(self, path):
        # Unknown paths share one label, so a scanner probing random URLs cannot blow up the series count.
        if path in self.known or path.startswith("/api/v1/"):
            return path if path in self.known else "/api/v1/other"
        if path.startswith("/web/") or path in ("/", "/inspect", "/models", "/data", "/corridor", "/works",
                                                 "/video", "/system", "/architecture", "/design", "/impact",
                                                 "/api-docs"):
            return "site"
        return "other"

    def observe(self, path, method, status, seconds):
        route = self.route_of(path)
        with self._lock:
            k = (route, method or "-", str(status))
            self.requests[k] = self.requests.get(k, 0) + 1
            h = self.latency.setdefault(route, [0] * (len(LATENCY_BUCKETS) + 1) + [0.0])
            for i, b in enumerate(LATENCY_BUCKETS):
                if seconds <= b:
                    h[i] += 1
            h[len(LATENCY_BUCKETS)] += 1
            h[-1] += seconds

    def render(self, readiness=None):
        lines = ["# HELP road_shield_requests_total HTTP requests by route, method and status.",
                 "# TYPE road_shield_requests_total counter"]
        with self._lock:
            for (route, method, status), n in sorted(self.requests.items()):
                lines.append(f'road_shield_requests_total{{route="{route}",method="{method}",status="{status}"}} {n}')
            lines += ["# HELP road_shield_request_duration_seconds Request latency.",
                      "# TYPE road_shield_request_duration_seconds histogram"]
            for route, h in sorted(self.latency.items()):
                for i, b in enumerate(LATENCY_BUCKETS):
                    lines.append(f'road_shield_request_duration_seconds_bucket{{route="{route}",le="{b}"}} {h[i]}')
                lines.append(f'road_shield_request_duration_seconds_bucket{{route="{route}",le="+Inf"}} '
                             f'{h[len(LATENCY_BUCKETS)]}')
                lines.append(f'road_shield_request_duration_seconds_sum{{route="{route}"}} {round(h[-1], 6)}')
                lines.append(f'road_shield_request_duration_seconds_count{{route="{route}"}} {h[len(LATENCY_BUCKETS)]}')
        lines += ["# HELP road_shield_uptime_seconds Seconds since the engine started.",
                  "# TYPE road_shield_uptime_seconds gauge",
                  f"road_shield_uptime_seconds {round(time.time() - self.started, 1)}"]
        if readiness is not None:
            lines += ["# HELP road_shield_model_ready 1 if the model is loaded.",
                      "# TYPE road_shield_model_ready gauge"]
            for name, ok in sorted(readiness.get("models", {}).items()):
                lines.append(f'road_shield_model_ready{{model="{name}"}} {1 if ok else 0}')
            lines.append(f"road_shield_ready {1 if readiness.get('ready') else 0}")
        return "\n".join(lines) + "\n"


def readiness(vision_ready, segmenter_ready, imu_ready, ledger_ready):
    """What a road frame needs: the classifier and the segmenter. IMU and ledger are reported, not required."""
    models = {"vision_classifier": bool(vision_ready), "defect_segmenter": bool(segmenter_ready),
              "imu_classifier": bool(imu_ready), "ledger": bool(ledger_ready)}
    missing = [k for k in ("vision_classifier", "defect_segmenter") if not models[k]]
    return {"ready": not missing, "models": models, "missing": missing,
            "note": ("all models a road frame needs are loaded" if not missing else
                     "not ready: " + ", ".join(missing) + " not loaded (see /system for the reason)")}
