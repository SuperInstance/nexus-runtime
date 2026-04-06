"""Tests for jetson.performance.caching — CacheEntry, CachePolicy, CacheManager."""

import time
import pytest
from jetson.performance.caching import CacheEntry, CachePolicy, CacheManager, EvictionPolicy

class TestCacheEntry:
    def test_construction(self):
        e = CacheEntry(key="k", value="v", created_at=100.0, last_accessed=100.0, access_count=1)
        assert e.key == "k"
        assert e.value == "v"
        assert e.access_count == 1
        assert e.size_bytes == 0

    def test_age(self):
        e = CacheEntry(key="k", value="v", created_at=time.time() - 1.0, last_accessed=time.time(), access_count=1)
        assert e.age >= 0.9

    def test_touch(self):
        e = CacheEntry(key="k", value="v", created_at=100.0, last_accessed=100.0, access_count=1)
        e.touch()
        assert e.access_count == 2
        assert e.last_accessed >= 100.0

    def test_is_expired_no_ttl(self):
        e = CacheEntry(key="k", value="v", created_at=0.0, last_accessed=0.0, access_count=1)
        assert e.is_expired(None) is False

    def test_is_expired_valid(self):
        e = CacheEntry(key="k", value="v", created_at=time.time(), last_accessed=time.time(), access_count=1)
        assert e.is_expired(10.0) is False

    def test_is_expired_past_ttl(self):
        e = CacheEntry(key="k", value="v", created_at=time.time() - 20.0, last_accessed=time.time() - 20.0, access_count=1)
        assert e.is_expired(10.0) is True

    def test_as_dict(self):
        e = CacheEntry(key="k", value=42, created_at=100.0, last_accessed=101.0, access_count=5, size_bytes=10)
        d = e.as_dict()
        assert d["key"] == "k"
        assert d["value"] == 42
        assert d["access_count"] == 5
        assert d["size_bytes"] == 10

class TestCachePolicy:
    def test_defaults(self):
        p = CachePolicy()
        assert p.max_size == 1000
        assert p.eviction_policy == EvictionPolicy.LRU
        assert p.ttl_seconds is None

    def test_custom(self):
        p = CachePolicy(max_size=100, eviction_policy=EvictionPolicy.FIFO, ttl_seconds=60.0)
        assert p.max_size == 100
        assert p.eviction_policy == EvictionPolicy.FIFO
        assert p.ttl_seconds == 60.0

    def test_as_dict(self):
        p = CachePolicy(max_size=50, eviction_policy=EvictionPolicy.LFU, ttl_seconds=30.0)
        d = p.as_dict()
        assert d["max_size"] == 50
        assert d["eviction_policy"] == "lfu"
        assert d["ttl_seconds"] == 30.0

class TestCacheManager:
    def test_get_miss(self):
        cm = CacheManager()
        assert cm.get("missing") is None

    def test_put_and_get(self):
        cm = CacheManager()
        cm.put("key", "value")
        assert cm.get("key") == "value"

    def test_put_overwrite(self):
        cm = CacheManager()
        cm.put("k", "v1")
        cm.put("k", "v2")
        assert cm.get("k") == "v2"

    def test_put_returns_evicted(self):
        cm = CacheManager(policy=CachePolicy(max_size=3))
        evicted = cm.put("a", 1)
        assert evicted == []
        evicted = cm.put("b", 2)
        assert evicted == []
        evicted = cm.put("c", 3)
        assert evicted == []
        evicted = cm.put("d", 4)
        assert len(evicted) == 1

    def test_invalidate_existing(self):
        cm = CacheManager()
        cm.put("k", "v")
        assert cm.invalidate("k") is True
        assert cm.get("k") is None

    def test_invalidate_missing(self):
        cm = CacheManager()
        assert cm.invalidate("nope") is False

    def test_clear(self):
        cm = CacheManager()
        cm.put("a", 1)
        cm.put("b", 2)
        count = cm.clear()
        assert count == 2
        assert cm.get("a") is None

    def test_clear_empty(self):
        cm = CacheManager()
        assert cm.clear() == 0

    def test_stats_initial(self):
        cm = CacheManager()
        s = cm.stats()
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["size"] == 0

    def test_stats_after_operations(self):
        cm = CacheManager()
        cm.put("k", "v")
        cm.get("k")   # hit
        cm.get("x")   # miss
        s = cm.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["size"] == 1

    def test_compute_hit_rate(self):
        cm = CacheManager()
        assert cm.compute_hit_rate() == 0.0
        cm.put("k", "v")
        cm.get("k")
        cm.get("k")
        cm.get("x")
        rate = cm.compute_hit_rate()
        assert rate == pytest.approx(66.666, abs=0.01)

    def test_lru_eviction_order(self):
        cm = CacheManager(policy=CachePolicy(max_size=3, eviction_policy=EvictionPolicy.LRU))
        cm.put("a", 1)
        cm.put("b", 2)
        cm.put("c", 3)
        cm.get("a")  # Touch a
        cm.put("d", 4)  # Should evict b (LRU)
        assert cm.get("a") == 1
        assert cm.get("b") is None
        assert cm.get("d") == 4

    def test_fifo_eviction_order(self):
        cm = CacheManager(policy=CachePolicy(max_size=3, eviction_policy=EvictionPolicy.FIFO))
        cm.put("a", 1)
        cm.put("b", 2)
        cm.put("c", 3)
        cm.put("d", 4)  # Should evict a (first in)
        assert cm.get("a") is None
        assert cm.get("d") == 4

    def test_lfu_eviction_order(self):
        cm = CacheManager(policy=CachePolicy(max_size=3, eviction_policy=EvictionPolicy.LFU))
        cm.put("a", 1)
        cm.put("b", 2)
        cm.put("c", 3)
        cm.get("a")
        cm.get("a")
        cm.get("b")
        cm.put("d", 4)  # c has lowest access count (1), should evict c
        assert cm.get("c") is None
        assert cm.get("a") == 1

    def test_ttl_expiration(self):
        cm = CacheManager(policy=CachePolicy(ttl_seconds=0.01))
        cm.put("k", "v")
        time.sleep(0.02)
        assert cm.get("k") is None

    def test_ttl_override_per_put(self):
        cm = CacheManager(policy=CachePolicy(ttl_seconds=100.0))
        cm.put("k", "v", ttl=0.01)
        time.sleep(0.02)
        assert cm.get("k") is None

    def test_keys(self):
        cm = CacheManager()
        cm.put("a", 1)
        cm.put("b", 2)
        assert set(cm.keys()) == {"a", "b"}

    def test_entries(self):
        cm = CacheManager()
        cm.put("x", 99)
        entries = cm.entries()
        assert len(entries) == 1
        assert entries[0].key == "x"
        assert entries[0].value == 99

    def test_implement_lru(self):
        cm = CacheManager.implement_lru(50)
        assert cm._policy.max_size == 50
        assert cm._policy.eviction_policy == EvictionPolicy.LRU

    def test_optimize_cache_lru(self):
        patterns = [{"key": "a", "frequency": 10, "recency": 1}, {"key": "b", "frequency": 5, "recency": 2}]
        policy = CacheManager.optimize_cache(patterns, 100)
        assert policy.max_size == 100
        assert policy.eviction_policy == EvictionPolicy.LRU

    def test_optimize_cache_small_cache(self):
        patterns = [{"key": str(i), "frequency": 1, "recency": i} for i in range(200)]
        policy = CacheManager.optimize_cache(patterns, 50)
        assert policy.eviction_policy in (EvictionPolicy.LRU, EvictionPolicy.LFU)

    def test_optimize_cache_empty_patterns(self):
        policy = CacheManager.optimize_cache([], 100)
        assert policy.max_size == 100

    def test_utilization_in_stats(self):
        cm = CacheManager(policy=CachePolicy(max_size=10))
        for i in range(5):
            cm.put(i, i)
        s = cm.stats()
        assert s["utilization"] == pytest.approx(0.5)

    def test_max_size_fill(self):
        cm = CacheManager(policy=CachePolicy(max_size=5))
        for i in range(10):
            cm.put(i, i)
        assert len(cm.keys()) == 5

    def test_multiple_clears(self):
        cm = CacheManager()
        cm.put("a", 1)
        cm.clear()
        cm.put("b", 2)
        assert cm.clear() == 1

    def test_expired_entry_counts_as_miss(self):
        cm = CacheManager(policy=CachePolicy(ttl_seconds=0.01))
        cm.put("k", "v")
        time.sleep(0.02)
        cm.get("k")  # expired miss
        s = cm.stats()
        assert s["misses"] == 1
