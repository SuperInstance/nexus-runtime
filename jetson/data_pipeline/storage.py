"""In-memory storage engine — StorageBackend and TimeSeriesStore."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from collections import deque


@dataclass
class StorageEntry:
    """Single entry in the storage backend."""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StorageBackend:
    """Generic in-memory key-value store with TTL and predicate queries."""

    def __init__(self, default_ttl: Optional[float] = None) -> None:
        self._store: Dict[str, StorageEntry] = {}
        self._default_ttl = default_ttl
        self._ops_count = 0

    # ── CRUD ───────────────────────────────────────────────────

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value.  TTL overrides default."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        self._store[key] = StorageEntry(
            key=key, value=value,
            timestamp=time.time(), ttl=effective_ttl,
        )
        self._ops_count += 1

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value or None (also if expired)."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            del self._store[key]
            return None
        self._ops_count += 1
        return entry.value

    def delete(self, key: str) -> bool:
        """Remove a key.  Returns True if it existed."""
        if key in self._store:
            del self._store[key]
            self._ops_count += 1
            return True
        return False

    def update(self, key: str, value: Any) -> bool:
        """Update an existing key.  Returns False if key doesn't exist."""
        if key not in self._store:
            return False
        entry = self._store[key]
        self._store[key] = StorageEntry(
            key=key, value=value,
            timestamp=time.time(), ttl=entry.ttl,
            metadata=dict(entry.metadata),
        )
        self._ops_count += 1
        return True

    def expire(self, key: str) -> bool:
        """Immediately expire a key (remove it).  Returns True if existed."""
        return self.delete(key)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()
        self._ops_count += 1

    # ── queries ────────────────────────────────────────────────

    def keys(self) -> List[str]:
        """Return all non-expired keys."""
        self._purge_expired()
        return list(self._store.keys())

    def values(self) -> List[Any]:
        """Return all non-expired values."""
        self._purge_expired()
        return [e.value for e in self._store.values()]

    def size(self) -> int:
        self._purge_expired()
        return len(self._store)

    def contains(self, key: str) -> bool:
        if key not in self._store:
            return False
        if self._is_expired(self._store[key]):
            del self._store[key]
            return False
        return True

    def query(self, predicate: Callable[[StorageEntry], bool]) -> List[StorageEntry]:
        """Return all entries matching *predicate*."""
        self._purge_expired()
        return [e for e in self._store.values() if predicate(e)]

    def stats(self) -> Dict[str, Any]:
        self._purge_expired()
        return {
            "total_keys": len(self._store),
            "total_ops": self._ops_count,
            "entries": {
                k: {"age": time.time() - e.timestamp, "ttl": e.ttl}
                for k, e in self._store.items()
            },
        }

    # ── internals ──────────────────────────────────────────────

    def _is_expired(self, entry: StorageEntry) -> bool:
        if entry.ttl is None:
            return False
        return (time.time() - entry.timestamp) > entry.ttl

    def _purge_expired(self) -> int:
        now = time.time()
        expired = [k for k, e in self._store.items()
                   if e.ttl is not None and (now - e.timestamp) > e.ttl]
        for k in expired:
            del self._store[k]
        return len(expired)


class TimeSeriesStore(StorageBackend):
    """Specialised storage for named time-series data."""

    def __init__(self) -> None:
        super().__init__()
        self._series: Dict[str, Deque[Tuple[float, Any]]] = {}

    # ── series operations ──────────────────────────────────────

    def append(self, series_name: str, point: Tuple[float, Any]) -> None:
        """Append (timestamp, value) to a named series."""
        if series_name not in self._series:
            self._series[series_name] = deque()
        self._series[series_name].append(point)

    def query(self, series_name: str,
              start: Optional[float] = None,
              end: Optional[float] = None) -> List[Tuple[float, Any]]:
        """Query a series with optional time range [start, end]."""
        data = self._series.get(series_name)
        if data is None:
            return []
        result: List[Tuple[float, Any]] = []
        for ts, val in data:
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                continue
            result.append((ts, val))
        return result

    def latest(self, series_name: str, n: int = 1) -> List[Tuple[float, Any]]:
        """Return the last *n* points of a series."""
        data = self._series.get(series_name)
        if data is None:
            return []
        pts = list(data)
        return pts[-n:]

    def delete_series(self, series_name: str) -> bool:
        """Delete an entire series.  Returns True if it existed."""
        if series_name in self._series:
            del self._series[series_name]
            return True
        return False

    def series_names(self) -> List[str]:
        return list(self._series.keys())

    def count(self, series_name: str) -> int:
        data = self._series.get(series_name)
        return len(data) if data is not None else 0
