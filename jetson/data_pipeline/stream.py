"""Data stream processing — DataPoint and DataStream classes."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional


@dataclass
class DataPoint:
    """Single data point in a stream."""
    timestamp: float
    source: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataStream:
    """In-memory data stream with functional-style operations.

    Maintains insertion order and provides push/pop semantics along
    with lazy filter, map, and window operations that return new streams.
    """

    def __init__(self, max_size: int = 0) -> None:
        self._buffer: Deque[DataPoint] = deque()
        self._max_size = max_size

    # ── mutation ───────────────────────────────────────────────

    def push(self, point: DataPoint) -> None:
        """Append a point.  If max_size > 0, oldest is evicted."""
        self._buffer.append(point)
        if self._max_size > 0 and len(self._buffer) > self._max_size:
            self._buffer.popleft()

    # ── accessors ──────────────────────────────────────────────

    def pop(self) -> Optional[DataPoint]:
        """Remove and return the oldest point, or None."""
        if not self._buffer:
            return None
        return self._buffer.popleft()

    def peek(self) -> Optional[DataPoint]:
        """Return the oldest point without removing it, or None."""
        if not self._buffer:
            return None
        return self._buffer[0]

    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def size(self) -> int:
        return len(self._buffer)

    # ── lazy functional ops ────────────────────────────────────

    def filter(self, predicate_fn: Callable[[DataPoint], bool]) -> "DataStream":
        """Return a new stream containing only points matching *predicate_fn*."""
        result = DataStream()
        for point in self._buffer:
            if predicate_fn(point):
                result.push(point)
        return result

    def map(self, transform_fn: Callable[[DataPoint], DataPoint]) -> "DataStream":
        """Return a new stream with each point transformed."""
        result = DataStream()
        for point in self._buffer:
            result.push(transform_fn(point))
        return result

    # ── batch / window ─────────────────────────────────────────

    def batch(self, size: int) -> List[List[DataPoint]]:
        """Split stream into batches of *size*.  Last batch may be smaller."""
        points = list(self._buffer)
        batches: List[List[DataPoint]] = []
        for i in range(0, len(points), size):
            batches.append(points[i : i + size])
        return batches

    def window(self, duration_seconds: float) -> "DataStream":
        """Return a new stream with points from the last *duration_seconds*."""
        cutoff = time.time() - duration_seconds
        result = DataStream()
        for point in self._buffer:
            if point.timestamp >= cutoff:
                result.push(point)
        return result

    def merge(self, other_stream: "DataStream") -> "DataStream":
        """Merge another stream into this one (sorted by timestamp)."""
        combined = list(self._buffer) + list(other_stream._buffer)
        combined.sort(key=lambda p: p.timestamp)
        merged = DataStream()
        for p in combined:
            merged.push(p)
        return merged

    # ── dunder helpers ─────────────────────────────────────────

    def __iter__(self):
        return iter(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"DataStream(size={self.size()})"


def make_point(source: str, value: Any, timestamp: float | None = None,
               metadata: Dict[str, Any] | None = None) -> DataPoint:
    """Convenience factory for DataPoint."""
    return DataPoint(
        timestamp=timestamp if timestamp is not None else time.time(),
        source=source,
        value=value,
        metadata=metadata or {},
    )
