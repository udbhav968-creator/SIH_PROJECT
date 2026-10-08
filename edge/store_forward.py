"""
Store-and-forward queue for a bus that is often out of mobile coverage.

Every event is sealed (edge/crypto.py) the moment it is queued and only the sealed envelope is written to
SQLite, so the queue on the SD card is encrypted at rest. flush() sends the oldest envelopes first and
stops at the first failure, which keeps sequence numbers arriving in order: the server refuses a number at
or below the last one it accepted, so sending out of order would lose packets.

The sequence counter lives in the same database and is advanced in the same transaction as the insert, so
a power cut can never hand out one number twice. Each database also gets a random epoch when it is created;
a new SD card is a new epoch, so the server does not mistake its packets for replays of the old card's.

A packet the server answers but refuses (it cannot be opened: a key change, a corrupted row) is moved to
the dead_letter table after MAX_REFUSALS answers, so one bad row cannot stop a bus's uploads for good.
A network failure is not a refusal and never dead-letters anything, however long the outage lasts.

Disk is bounded: past max_rows the oldest undelivered position heartbeats are dropped first, then the
oldest events, and every drop is counted (stats()["dropped"]) rather than silent.
"""
import json
import os
import sqlite3
import threading
import time

from edge import crypto

SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    seq       INTEGER NOT NULL UNIQUE,
    kind      TEXT    NOT NULL,
    envelope  TEXT    NOT NULL,
    queued_at REAL    NOT NULL,
    attempts  INTEGER NOT NULL DEFAULT 0,
    refusals  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS dead_letter (
    seq       INTEGER PRIMARY KEY,
    kind      TEXT    NOT NULL,
    envelope  TEXT    NOT NULL,
    queued_at REAL    NOT NULL,
    moved_at  REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""
MAX_REFUSALS = 3


class SendError(Exception):
    """Raised by a sender when the server could not be reached (as opposed to answering and refusing)."""


class StoreAndForward:
    def __init__(self, path, bus_id, key, max_rows=50000):
        if key is None:
            raise ValueError("a fleet key is required (ROAD_SHIELD_FLEET_KEY)")
        self.bus_id = str(bus_id)
        self.key = key
        self.max_rows = int(max_rows)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        row = self._db.execute("SELECT v FROM meta WHERE k='epoch'").fetchone()
        if row is None:
            self._db.execute("INSERT INTO meta(k, v) VALUES('epoch', ?)", (os.urandom(8).hex(),))
            row = self._db.execute("SELECT v FROM meta WHERE k='epoch'").fetchone()
        self.epoch = row[0]
        self._backoff_until = 0.0
        self._failures = 0

    def close(self):
        with self._lock:
            self._db.close()

    def _meta(self, k, default):
        row = self._db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return int(row[0]) if row else default

    def _set_meta(self, k, v):
        self._db.execute("INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))

    def put(self, event):
        """Seal and queue one event; returns its sequence number."""
        kind = str(event.get("t", "event"))
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                seq = self._meta("next_seq", 1)
                env = crypto.pack(event, self.bus_id, seq, self.key, epoch=self.epoch)
                self._db.execute("INSERT INTO outbox(seq, kind, envelope, queued_at) VALUES(?,?,?,?)",
                                 (seq, kind, json.dumps(env, separators=(",", ":")), time.time()))
                self._set_meta("next_seq", seq + 1)
                self._trim()
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        return seq

    def _trim(self):
        n = self._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        over = n - self.max_rows
        if over <= 0:
            return
        dropped = 0
        for kind_clause in ("kind = 'pos'", "1=1"):
            if over <= 0:
                break
            ids = [r[0] for r in self._db.execute(
                f"SELECT id FROM outbox WHERE {kind_clause} ORDER BY id LIMIT ?", (over,))]
            if ids:
                self._db.execute(f"DELETE FROM outbox WHERE id IN ({','.join('?' * len(ids))})", ids)
                over -= len(ids)
                dropped += len(ids)
        self._set_meta("dropped", self._meta("dropped", 0) + dropped)

    def pending(self, limit=100):
        with self._lock:
            rows = self._db.execute("SELECT id, envelope FROM outbox ORDER BY seq LIMIT ?", (limit,)).fetchall()
        return [(r[0], json.loads(r[1])) for r in rows]

    def flush(self, send_batch, batch_size=50, now=None):
        """Send queued envelopes in order with send_batch(list_of_envelopes) -> number accepted from the
        front of the list; send_batch raises when the server cannot be reached. Returns how many were
        delivered. After a failure, waits with exponential backoff (5 s doubling to 5 min)."""
        now = time.time() if now is None else now
        if now < self._backoff_until:
            return 0
        delivered = 0
        while True:
            batch = self.pending(batch_size)
            if not batch:
                break
            refused = False
            try:
                accepted = int(send_batch([env for _, env in batch]))
                refused = accepted < len(batch)
            except Exception:
                accepted = 0
            accepted = max(0, min(accepted, len(batch)))
            ids = [i for i, _ in batch]
            moved = False
            with self._lock:
                if accepted:
                    done = ids[:accepted]
                    self._db.execute(f"DELETE FROM outbox WHERE id IN ({','.join('?' * len(done))})", done)
                if accepted < len(batch):
                    head = ids[accepted]
                    self._db.execute("UPDATE outbox SET attempts = attempts + 1, refusals = refusals + ? "
                                     "WHERE id = ?", (1 if refused else 0, head))
                    r = self._db.execute("SELECT refusals FROM outbox WHERE id = ?", (head,)).fetchone()
                    if r and r[0] >= MAX_REFUSALS:
                        self._db.execute("INSERT OR REPLACE INTO dead_letter(seq, kind, envelope, queued_at, moved_at) "
                                         "SELECT seq, kind, envelope, queued_at, ? FROM outbox WHERE id = ?",
                                         (now, head))
                        self._db.execute("DELETE FROM outbox WHERE id = ?", (head,))
                        moved = True
            delivered += accepted
            if moved:
                continue
            if accepted < len(batch):
                self._failures += 1
                self._backoff_until = now + min(300.0, 5.0 * 2 ** (self._failures - 1))
                break
            self._failures = 0
        return delivered

    def stats(self):
        with self._lock:
            n = self._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
            oldest = self._db.execute("SELECT MIN(queued_at) FROM outbox").fetchone()[0]
            dead = self._db.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0]
            return {"queued": n, "next_seq": self._meta("next_seq", 1), "dropped": self._meta("dropped", 0),
                    "dead_letter": dead, "epoch": self.epoch,
                    "oldest_queued_age_s": round(time.time() - oldest, 1) if oldest else None,
                    "consecutive_failures": self._failures}
