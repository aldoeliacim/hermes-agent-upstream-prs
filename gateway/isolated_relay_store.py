"""SQLite queue and bounded authority ledger for the managed native Relay.

One connector/event loop owns a store. Transactions serialize admission and ACK
updates; no in-memory queue copy can get ahead of a durable commit. Binding
identity tombstones are never pruned or reassigned (credentials may rotate).
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from gateway.isolated_relay import RelayBinding


class DuplicateDelivery(ValueError):
    """Already durably pending or in the bounded ACK history."""


class RelayStore:
    def __init__(self, path: str | Path | None, bindings: Sequence[RelayBinding],
                 max_pending: int, history_size: int):
        self.max_pending = max_pending
        self.history_size = history_size
        # No implicit profile path: the parent supplies its isolated state path.
        self.db = sqlite3.connect(str(path) if path is not None else ":memory:")
        try:
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS bindings (
                    gateway_id TEXT PRIMARY KEY, identity TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pending (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    gateway_id TEXT NOT NULL REFERENCES bindings(gateway_id),
                    delivery_id TEXT NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(gateway_id, delivery_id));
                CREATE TABLE IF NOT EXISTS acknowledgements (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    gateway_id TEXT NOT NULL REFERENCES bindings(gateway_id),
                    delivery_id TEXT NOT NULL, UNIQUE(gateway_id, delivery_id));
                CREATE TABLE IF NOT EXISTS targets (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    gateway_id TEXT NOT NULL REFERENCES bindings(gateway_id),
                    message_id TEXT NOT NULL, own INTEGER NOT NULL CHECK(own IN (0,1)),
                    UNIQUE(gateway_id, message_id));
            """)
            with self.transaction():
                for b in bindings:
                    identity = json.dumps([b.platform.value, b.chat_id, b.thread_id,
                                           b.scope_id, b.bot_id], ensure_ascii=False)
                    row = self.db.execute("SELECT identity FROM bindings WHERE gateway_id=?",
                                          (b.gateway_id,)).fetchone()
                    if row is not None and row[0] != identity:
                        raise ValueError("stored binding mismatch")
                    self.db.execute("INSERT OR IGNORE INTO bindings VALUES (?, ?)",
                                    (b.gateway_id, identity))
                    count = self.db.execute("SELECT COUNT(*) FROM pending WHERE gateway_id=?",
                                            (b.gateway_id,)).fetchone()[0]
                    if count > max_pending:
                        raise ValueError("stored queue exceeds max_pending")
                    self._prune("acknowledgements", b.gateway_id)
                    self._prune("targets", b.gateway_id)
        except BaseException:
            self.db.close()
            raise

    @contextmanager
    def transaction(self):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            yield

    def _prune(self, table: str, gateway_id: str):
        # table names are internal constants, never worker input.
        self.db.execute(f"DELETE FROM {table} WHERE gateway_id=? AND seq NOT IN "
                        f"(SELECT seq FROM {table} WHERE gateway_id=? ORDER BY seq DESC LIMIT ?)",
                        (gateway_id, gateway_id, self.history_size))

    def _remember_target(self, gateway_id: str, message_id: str, own: bool):
        # An inbound observation must never revoke ownership of our own send.
        previous = self.target(gateway_id, message_id)
        self.db.execute("DELETE FROM targets WHERE gateway_id=? AND message_id=?",
                        (gateway_id, message_id))
        self.db.execute("INSERT INTO targets(gateway_id,message_id,own) VALUES (?,?,?)",
                        (gateway_id, message_id, bool(own or previous)))
        self._prune("targets", gateway_id)

    def enqueue(self, gateway_id: str, delivery_id: str, payload: str, message_id: str):
        with self.transaction():
            for table in ("pending", "acknowledgements"):
                if self.db.execute(f"SELECT 1 FROM {table} WHERE gateway_id=? AND delivery_id=?",
                                   (gateway_id, delivery_id)).fetchone():
                    raise DuplicateDelivery("duplicate delivery")
            count = self.db.execute("SELECT COUNT(*) FROM pending WHERE gateway_id=?",
                                    (gateway_id,)).fetchone()[0]
            if count >= self.max_pending:
                raise asyncio.QueueFull
            self.db.execute("INSERT INTO pending(gateway_id,delivery_id,payload) VALUES (?,?,?)",
                            (gateway_id, delivery_id, payload))
            self._remember_target(gateway_id, message_id, False)

    def peek(self, gateway_id: str):
        return self.db.execute("SELECT delivery_id,payload FROM pending WHERE gateway_id=? "
                               "ORDER BY seq LIMIT 1", (gateway_id,)).fetchone()

    def ack(self, gateway_id: str, delivery_id: str):
        with self.transaction():
            first = self.peek(gateway_id)
            if first is None or first[0] != delivery_id:
                raise ValueError("ACK rejected")
            self.db.execute("DELETE FROM pending WHERE gateway_id=? AND delivery_id=?",
                            (gateway_id, delivery_id))
            self.db.execute("INSERT INTO acknowledgements(gateway_id,delivery_id) VALUES (?,?)",
                            (gateway_id, delivery_id))
            self._prune("acknowledgements", gateway_id)

    def target(self, gateway_id: str, message_id: str) -> bool | None:
        row = self.db.execute("SELECT own FROM targets WHERE gateway_id=? AND message_id=?",
                              (gateway_id, message_id)).fetchone()
        return bool(row[0]) if row is not None else None

    def remember_sent(self, gateway_id: str, message_id: str):
        with self.transaction():
            self._remember_target(gateway_id, message_id, True)

    def close(self):
        self.db.close()
