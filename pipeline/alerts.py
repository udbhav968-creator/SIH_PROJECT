"""
Alerts for the people who act on them: a P1 defect appears, or a work order passes its SLA.

Each alert is published to the live map and kept in a short list (/api/v1/alerts). When
ROAD_SHIELD_ALERT_WEBHOOK is set, it is also POSTed there as JSON in the background, which is how it reaches
a team chat or a ticketing system: Slack and Microsoft Teams incoming webhooks, and most ticketing tools,
accept a JSON POST with a "text" field, which every alert carries.

An alert is sent once per subject (one P1 defect, one overdue order), not on every request that notices
it. Only https URLs are accepted, plus http to this machine for testing.
"""
import json
import os
import threading
import time
import urllib.parse
import urllib.request

KEEP = 200


def webhook_allowed(url):
    try:
        u = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if u.scheme == "https" and u.hostname:
        return True
    return u.scheme == "http" and u.hostname in ("127.0.0.1", "localhost", "::1")


class Alerts:
    def __init__(self, events=None, webhook=None, timeout=5.0):
        self.events = events
        url = webhook if webhook is not None else os.environ.get("ROAD_SHIELD_ALERT_WEBHOOK", "")
        self.webhook = url if url and webhook_allowed(url) else None
        self.webhook_rejected = bool(url) and self.webhook is None
        self.timeout = timeout
        self._sent = set()
        self._recent = []
        self._lock = threading.Lock()
        self.delivered = 0
        self.failed = 0
        self._count_lock = threading.Lock()

    def notify(self, kind, key, text, data=None):
        """Send one alert unless this (kind, key) was already sent. Returns the alert or None."""
        with self._lock:
            if (kind, key) in self._sent:
                return None
            if len(self._sent) > 20000:          # bounded memory on a long-running server
                self._sent.clear()
            self._sent.add((kind, key))
            alert = {"kind": kind, "key": key, "text": text, "data": data or {}, "at": round(time.time(), 3)}
            self._recent.append(alert)
            del self._recent[:-KEEP]
        if self.events is not None:
            try:
                self.events.publish("alert", alert)
            except Exception:
                pass
        if self.webhook:
            threading.Thread(target=self._post, args=(alert,), daemon=True).start()
        return alert

    def _post(self, alert):
        body = json.dumps({"text": f"ROAD-SHIELD: {alert['text']}", "alert": alert}, default=str).encode("utf-8")
        req = urllib.request.Request(self.webhook, data=body, headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                r.read(1024)
            with self._count_lock:
                self.delivered += 1
        except Exception:
            with self._count_lock:
                self.failed += 1

    def recent(self, n=50):
        with self._lock:
            return list(self._recent[-n:])

    def status(self):
        return {"webhook": "configured" if self.webhook else ("rejected (must be https)" if self.webhook_rejected
                                                              else "not set (ROAD_SHIELD_ALERT_WEBHOOK)"),
                "delivered": self.delivered, "failed": self.failed}

    # ------------------------------------------------------------- checks
    def check_defect(self, defect, priority):
        if priority and priority.get("band") == "P1":
            return self.notify("p1_defect", defect["defect_id"],
                               f"P1 {defect.get('defect_class')} at {defect['lat']:.5f}, {defect['lon']:.5f} "
                               f"(priority {priority['priority_index']}, PCI {defect.get('severity_pci')})",
                               {"defect_id": defect["defect_id"], "lat": defect["lat"], "lon": defect["lon"],
                                "priority": priority})
        return None

    def check_overdue(self, orders):
        out = []
        for o in orders:
            if o.get("overdue"):
                a = self.notify("order_overdue", o["work_order_id"],
                                f"work order {o['work_order_id']} ({o.get('defect_class')}) is past its "
                                f"{o.get('sla_hours')} h SLA and still {o.get('status')}",
                                {"work_order_id": o["work_order_id"], "contractor": o.get("contractor")})
                if a:
                    out.append(a)
        return out
