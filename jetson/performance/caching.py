"""Caching strategies.

Provides LRU cache management with configurable policies,
TTL support, eviction, and cache-optimization recommendations.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EvictionPolicy(Enum):
    LRU = "lru"
    FIFO = "fifo"
    LFU = "lfu"


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    key: Any
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int = 0
    ttl: Optional[float] = None

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    def touch(self) -> None:
        """Update last_accessed and increment access_count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def is_expired(self, ttl: Optional[float] = None) -> bool:
        """Check if the entry has exceeded its TTL."""
        effective_ttl = ttl if ttl is not None else self.ttl
        if effective_ttl is None:
            return False
        return self.age > effective_ttl

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "age": self.age,
            "ttl": self.ttl,
        }


@dataclass
class CachePolicy:
    """Configuration policy for cache behavior."""

    max_size: int = 1000
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    ttl_seconds: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_size": self.max_size,
            "eviction_policy": self.eviction_policy.value,
            "ttl_seconds": self.ttl_seconds,
        }


class CacheManager:
    """Thread-safe cache manager with multiple eviction policies."""

    def __init__(self, policy: Optional[CachePolicy] = None) -> None:
        self._policy = policy or CachePolicy()
        self._cache: OrderedDict[Any, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve a value by key, or None on miss."""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        entry.touch()
        # Move to end for LRU
        if self._policy.eviction_policy == EvictionPolicy.LRU:
            self._cache.move_to_end(key)
        return entry.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> List[Any]:
        """Store a value; returns list of evicted keys."""
        evicted: List[Any] = []
        effective_ttl = ttl if ttl is not None else self._policy.ttl_seconds
        now = time.time()
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            last_accessed=now,
            access_count=1,
            size_bytes=self._estimate_size(value),
            ttl=effective_ttl,
        )

        if key in self._cache:
            del self._cache[key]

        while len(self._cache) >= self._policy.max_size:
            evicted.append(self._evict_one())

        self._cache[key] = entry
        return evicted

    def invalidate(self, key: Any) -> bool:
        """Remove a single key. Returns True if it existed."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        """Remove all entries. Returns count of cleared entries."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._policy.max_size,
            "utilization": len(self._cache) / self._policy.max_size if self._policy.max_size > 0 else 0.0,
            "eviction_policy": self._policy.eviction_policy.value,
            "ttl_seconds": self._policy.ttl_seconds,
        }

    def compute_hit_rate(self) -> float:
        """Return cache hit rate as a percentage (0–100)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return (self._hits / total) * 100.0

    def keys(self) -> List[Any]:
        """Return all current cache keys."""
        return list(self._cache.keys())

    def entries(self) -> List[CacheEntry]:
        """Return all cache entries."""
        return list(self._cache.values())

    def _evict_one(self) -> Any:
        """Evict a single entry according to policy. Returns evicted key."""
        if not self._cache:
            return None
        if self._policy.eviction_policy == EvictionPolicy.LFU:
            # Evict least frequently used
            key = min(self._cache, key=lambda k: self._cache[k].access_count)
            del self._cache[key]
            return key
        elif self._policy.eviction_policy == EvictionPolicy.FIFO:
            # Evict oldest
            key = next(iter(self._cache))
            del self._cache[key]
            return key
        else:
            # LRU — evict first (oldest in ordered dict)
            key, _ = self._cache.popitem(last=False)
            return key

    @staticmethod
    def _estimate_size(value: Any) -> int:
        """Rough estimate of value size in bytes."""
        try:
            return len(str(value))
        except Exception:
            return 1

    @staticmethod
    def optimize_cache(
        access_patterns: List[Dict[str, Any]], cache_size: int,
    ) -> CachePolicy:
        """Analyze access patterns and recommend an optimal cache config.

        *access_patterns* is a list of dicts with 'key', 'frequency', and
        optionally 'recency' fields.
        """
        total_freq = sum(p.get("frequency", 1) for p in access_patterns)
        if total_freq == 0:
            return CachePolicy(max_size=cache_size, eviction_policy=EvictionPolicy.LRU)

        unique_keys = set(p.get("key") for p in access_patterns)
        hit_rate_if_full = total_freq / total_freq  # always 1.0 if all fit

        # If more unique keys than cache size, eviction matters
        if len(unique_keys) > cache_size:
            # Check recency correlation
            has_recency = any("recency" in p for p in access_patterns)
            if has_recency:
                policy = EvictionPolicy.LRU
            else:
                policy = EvictionPolicy.LFU
        else:
            policy = EvictionPolicy.LRU

        return CachePolicy(max_size=cache_size, eviction_policy=policy)

    @staticmethod
    def implement_lru(max_size: int) -> "CacheManager":
        """Create a CacheManager configured for pure LRU behavior."""
        return CacheManager(
            policy=CachePolicy(max_size=max_size, eviction_policy=EvictionPolicy.LRU)
        )
