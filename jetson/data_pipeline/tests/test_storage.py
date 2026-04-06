"""Tests for storage.py — StorageEntry, StorageBackend, TimeSeriesStore."""

import time

import pytest

from jetson.data_pipeline.storage import StorageBackend, StorageEntry, TimeSeriesStore


# ── StorageEntry dataclass ─────────────────────────────────────

class TestStorageEntry:

    def test_create(self):
        e = StorageEntry(key="k1", value=42, timestamp=1.0)
        assert e.key == "k1"
        assert e.value == 42
        assert e.ttl is None
        assert e.metadata == {}

    def test_with_ttl(self):
        e = StorageEntry(key="k1", value=1, timestamp=1.0, ttl=60.0)
        assert e.ttl == 60.0

    def test_with_metadata(self):
        e = StorageEntry(key="k", value="v", timestamp=0, metadata={"x": 1})
        assert e.metadata["x"] == 1


# ── StorageBackend CRUD ────────────────────────────────────────

class TestStorageBackendCRUD:

    def setup_method(self):
        self.store = StorageBackend()

    def test_put_and_get(self):
        self.store.put("a", 1)
        assert self.store.get("a") == 1

    def test_get_nonexistent(self):
        assert self.store.get("nope") is None

    def test_put_overwrites(self):
        self.store.put("a", 1)
        self.store.put("a", 2)
        assert self.store.get("a") == 2

    def test_delete(self):
        self.store.put("a", 1)
        assert self.store.delete("a") is True
        assert self.store.get("a") is None

    def test_delete_nonexistent(self):
        assert self.store.delete("nope") is False

    def test_update_existing(self):
        self.store.put("a", 1)
        assert self.store.update("a", 99) is True
        assert self.store.get("a") == 99

    def test_update_nonexistent(self):
        assert self.store.update("z", 1) is False

    def test_clear(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        self.store.clear()
        assert self.store.size() == 0


# ── StorageBackend queries ─────────────────────────────────────

class TestStorageBackendQueries:

    def setup_method(self):
        self.store = StorageBackend()

    def test_keys(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        assert set(self.store.keys()) == {"a", "b"}

    def test_values(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        assert set(self.store.values()) == {1, 2}

    def test_size(self):
        self.store.put("a", 1)
        assert self.store.size() == 1
        self.store.put("b", 2)
        assert self.store.size() == 2

    def test_contains(self):
        self.store.put("a", 1)
        assert self.store.contains("a") is True
        assert self.store.contains("b") is False

    def test_query_by_value(self):
        self.store.put("a", 10)
        self.store.put("b", 20)
        self.store.put("c", 30)
        result = self.store.query(lambda e: e.value >= 20)
        assert len(result) == 2

    def test_query_by_key_prefix(self):
        self.store.put("sensor_1", 1)
        self.store.put("sensor_2", 2)
        self.store.put("other", 3)
        result = self.store.query(lambda e: e.key.startswith("sensor_"))
        assert len(result) == 2

    def test_query_empty(self):
        assert self.store.query(lambda e: True) == []

    def test_query_no_match(self):
        self.store.put("a", 1)
        assert self.store.query(lambda e: e.value > 100) == []


# ── StorageBackend stats ───────────────────────────────────────

class TestStorageBackendStats:

    def test_stats_keys(self):
        store = StorageBackend()
        store.put("a", 1)
        stats = store.stats()
        assert stats["total_keys"] == 1
        assert stats["total_ops"] >= 1

    def test_stats_has_entries(self):
        store = StorageBackend()
        store.put("x", 42)
        stats = store.stats()
        assert "x" in stats["entries"]


# ── StorageBackend TTL ─────────────────────────────────────────

class TestStorageBackendTTL:

    def test_expired_key_returns_none(self):
        store = StorageBackend()
        store.put("tmp", 1, ttl=0.05)  # 50ms
        time.sleep(0.06)
        assert store.get("tmp") is None

    def test_default_ttl(self):
        store = StorageBackend(default_ttl=0.05)
        store.put("x", 1)
        time.sleep(0.06)
        assert store.get("x") is None

    def test_override_default_ttl(self):
        store = StorageBackend(default_ttl=0.05)
        store.put("x", 1, ttl=60.0)  # explicit longer TTL
        time.sleep(0.06)
        assert store.get("x") == 1

    def test_no_ttl_persists(self):
        store = StorageBackend()
        store.put("permanent", 1)
        time.sleep(0.01)
        assert store.get("permanent") == 1

    def test_expire_method(self):
        store = StorageBackend()
        store.put("a", 1)
        assert store.expire("a") is True
        assert store.get("a") is None

    def test_expire_nonexistent(self):
        store = StorageBackend()
        assert store.expire("nope") is False

    def test_size_excludes_expired(self):
        store = StorageBackend()
        store.put("a", 1, ttl=0.05)
        time.sleep(0.06)
        assert store.size() == 0

    def test_contains_expired(self):
        store = StorageBackend()
        store.put("a", 1, ttl=0.05)
        time.sleep(0.06)
        assert store.contains("a") is False


# ── TimeSeriesStore ────────────────────────────────────────────

class TestTimeSeriesStore:

    def setup_method(self):
        self.ts = TimeSeriesStore()

    def test_append_and_count(self):
        for i in range(10):
            self.ts.append("temp", (float(i), float(i)))
        assert self.ts.count("temp") == 10

    def test_append_creates_series(self):
        self.ts.append("new", (1.0, 42.0))
        assert "new" in self.ts.series_names()

    def test_count_nonexistent(self):
        assert self.ts.count("nope") == 0

    def test_query_full(self):
        for i in range(5):
            self.ts.append("x", (float(i), float(i)))
        result = self.ts.query("x")
        assert len(result) == 5

    def test_query_with_range(self):
        for i in range(10):
            self.ts.append("x", (float(i), float(i)))
        result = self.ts.query("x", start=3.0, end=6.0)
        assert len(result) == 4  # 3, 4, 5, 6

    def test_query_start_only(self):
        for i in range(10):
            self.ts.append("x", (float(i), float(i)))
        result = self.ts.query("x", start=8.0)
        assert len(result) == 2

    def test_query_end_only(self):
        for i in range(10):
            self.ts.append("x", (float(i), float(i)))
        result = self.ts.query("x", end=2.0)
        assert len(result) == 3  # 0, 1, 2

    def test_query_nonexistent(self):
        assert self.ts.query("nope") == []

    def test_latest(self):
        for i in range(5):
            self.ts.append("x", (float(i), float(i)))
        result = self.ts.latest("x", n=2)
        assert len(result) == 2
        assert result[0][1] == 3.0
        assert result[1][1] == 4.0

    def test_latest_one(self):
        self.ts.append("x", (1.0, 10.0))
        result = self.ts.latest("x")
        assert len(result) == 1
        assert result[0][1] == 10.0

    def test_latest_nonexistent(self):
        assert self.ts.latest("nope") == []

    def test_delete_series(self):
        self.ts.append("x", (1.0, 1.0))
        assert self.ts.delete_series("x") is True
        assert self.ts.count("x") == 0

    def test_delete_nonexistent_series(self):
        assert self.ts.delete_series("nope") is False

    def test_series_names(self):
        self.ts.append("a", (1.0, 1.0))
        self.ts.append("b", (1.0, 1.0))
        names = self.ts.series_names()
        assert "a" in names
        assert "b" in names

    def test_multiple_series(self):
        self.ts.append("temp", (1.0, 20.0))
        self.ts.append("pressure", (1.0, 1013.0))
        assert self.ts.count("temp") == 1
        assert self.ts.count("pressure") == 1

    def test_inherits_storage_backend(self):
        self.ts.put("meta", "info")
        assert self.ts.get("meta") == "info"
