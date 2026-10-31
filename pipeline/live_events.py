"""Server-sent events emitter for real-time frontend updates."""
import itertools
import json
import queue
import threading
import time

RECENT = 200
PER_SUBSCRIBER = 500
MAX_SUBSCRIBERS = 64


class LiveEvents:
    def __init__(self, recent=RECENT, per_subscriber=PER_SUBSCRIBER, max_subscribers=MAX_SUBSCRIBERS):
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._subs = set()
        self._recent = []
        self.recent_size = recent
        self.per_subscriber = per_subscriber
        self.max_subscribers = max_subscribers

    def publish(self, kind, data):
        ev = {"id": next(self._ids), "type": str(kind), "ts": round(time.time(), 3), "data": data}
        with self._lock:
            self._recent.append(ev)
            del self._recent[:-self.recent_size]
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(ev)
                except (queue.Empty, queue.Full):
                    pass
        return ev

    def subscribe(self, last_event_id=None):
        """A queue of events, pre-filled with anything after last_event_id; None when at capacity."""
        q = queue.Queue(maxsize=self.per_subscriber)
        with self._lock:
            if len(self._subs) >= self.max_subscribers:
                return None
            if last_event_id is not None:
                for ev in self._recent:
                    if ev["id"] > last_event_id:
                        q.put_nowait(ev)
            self._subs.add(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs.discard(q)

    def recent(self, since_id=0):
        with self._lock:
            return [ev for ev in self._recent if ev["id"] > since_id]

    @property
    def subscribers(self):
        with self._lock:
            return len(self._subs)


def sse_frame(ev):
    return (f"id: {ev['id']}\nevent: {ev['type']}\n"
            f"data: {json.dumps(ev, separators=(',', ':'), default=str)}\n\n").encode("utf-8")


live_events = LiveEvents()
