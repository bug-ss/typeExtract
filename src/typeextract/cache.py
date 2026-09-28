"""Content-addressed answer caches: identical (model, state, question) is never paid for twice."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

DEFAULT_CACHE_PATH = ".typeextract-cache.sqlite"


@runtime_checkable
class Cache(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...

    def set(self, key: str, value: dict[str, Any]) -> None: ...


class MemoryCache:
    """In-process LRU cache."""

    def __init__(self, max_items: int = 200_000):
        self.max_items = max_items
        self._data: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class SQLiteCache:
    """Persistent cache shared by threads and processes on one machine (WAL mode)."""

    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL)"
            )
            self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM answers WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO answers (key, value, created) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def open_cache(spec: Cache | str | Path | bool | None) -> Cache | None:
    """``None``/``False``: no cache; ``True``: SQLite at the default path; ``":memory:"``: LRU;
    a path: SQLite at that path; a ``Cache``: itself."""
    if spec is None or spec is False:
        return None
    if spec is True:
        return SQLiteCache(DEFAULT_CACHE_PATH)
    if isinstance(spec, (str, Path)):
        return MemoryCache() if str(spec) == ":memory:" else SQLiteCache(spec)
    if isinstance(spec, Cache):
        return spec
    raise TypeError(f"unsupported cache spec: {spec!r}")
