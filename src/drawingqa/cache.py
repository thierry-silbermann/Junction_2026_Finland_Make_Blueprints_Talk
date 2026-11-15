"""Persistent SQLite cache; validated extraction only, never answer labels."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from .schemas import Extraction


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


class ExtractionCache:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "extractions.sqlite3"
        self._locks = {}
        self._guard = threading.Lock()
        with sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    @contextmanager
    def locked(self, key):
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            yield

    def get(self, key, schema=Extraction):
        with sqlite3.connect(self.path, timeout=30) as conn:
            row = conn.execute("SELECT payload FROM cache WHERE key=?", (key,)).fetchone()
        if row:
            try:
                return schema.model_validate_json(row[0])
            except ValueError:
                self.delete(key)
        return None

    def put(self, key, extraction):
        with sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute("INSERT OR REPLACE INTO cache(key,payload) VALUES (?,?)", (key, extraction.model_dump_json()))

    def delete(self, key):
        with sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute("DELETE FROM cache WHERE key=?", (key,))
