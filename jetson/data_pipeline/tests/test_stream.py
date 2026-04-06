"""Tests for stream.py — DataPoint and DataStream."""

import time

import pytest

from jetson.data_pipeline.stream import DataPoint, DataStream, make_point


# ── DataPoint creation ─────────────────────────────────────────

class TestDataPoint:

    def test_create_minimal(self):
        dp = DataPoint(timestamp=1.0, source="s1", value=42)
        assert dp.timestamp == 1.0
        assert dp.source == "s1"
        assert dp.value == 42
        assert dp.metadata == {}

    def test_create_with_metadata(self):
        dp = DataPoint(timestamp=2.0, source="gps", value=3.14,
                       metadata={"quality": "high"})
        assert dp.metadata["quality"] == "high"

    def test_value_can_be_any_type(self):
        dp = DataPoint(timestamp=0.0, source="x", value=[1, 2, 3])
        assert dp.value == [1, 2, 3]

    def test_mutable_metadata(self):
        dp = DataPoint(timestamp=0, source="a", value=1, metadata={"k": 1})
        dp.metadata["k"] = 2
        assert dp.metadata["k"] == 2


# ── make_point helper ──────────────────────────────────────────

class TestMakePoint:

    def test_auto_timestamp(self):
        before = time.time()
        dp = make_point("s", 1)
        after = time.time()
        assert before <= dp.timestamp <= after

    def test_explicit_timestamp(self):
        dp = make_point("s", 1, timestamp=100.0)
        assert dp.timestamp == 100.0

    def test_with_metadata(self):
        dp = make_point("s", 1, metadata={"unit": "m"})
        assert dp.metadata["unit"] == "m"


# ── DataStream basics ──────────────────────────────────────────

class TestDataStreamBasics:

    def test_empty_stream(self):
        ds = DataStream()
        assert ds.is_empty()
        assert ds.size() == 0
        assert ds.peek() is None

    def test_push_and_size(self):
        ds = DataStream()
        ds.push(DataPoint(1, "s", 10))
        assert ds.size() == 1
        assert not ds.is_empty()

    def test_push_multiple(self):
        ds = DataStream()
        for i in range(5):
            ds.push(DataPoint(float(i), "s", i))
        assert ds.size() == 5

    def test_pop_empty(self):
        ds = DataStream()
        assert ds.pop() is None

    def test_pop_returns_oldest(self):
        ds = DataStream()
        ds.push(DataPoint(1.0, "a", "first"))
        ds.push(DataPoint(2.0, "b", "second"))
        assert ds.pop().value == "first"
        assert ds.pop().value == "second"
        assert ds.is_empty()

    def test_peek_does_not_remove(self):
        ds = DataStream()
        ds.push(DataPoint(1, "s", 99))
        p1 = ds.peek()
        p2 = ds.peek()
        assert p1 is p2
        assert ds.size() == 1

    def test_peek_empty(self):
        assert DataStream().peek() is None

    def test_max_size_eviction(self):
        ds = DataStream(max_size=2)
        ds.push(DataPoint(1, "s", 1))
        ds.push(DataPoint(2, "s", 2))
        ds.push(DataPoint(3, "s", 3))
        assert ds.size() == 2
        assert ds.pop().value == 2  # oldest evicted

    def test_len_dunder(self):
        ds = DataStream()
        ds.push(DataPoint(1, "s", 1))
        assert len(ds) == 1

    def test_iter(self):
        ds = DataStream()
        points = [DataPoint(float(i), "s", i) for i in range(3)]
        for p in points:
            ds.push(p)
        collected = list(ds)
        assert collected == points

    def test_repr(self):
        ds = DataStream()
        assert "size=0" in repr(ds)
        ds.push(DataPoint(1, "s", 1))
        assert "size=1" in repr(ds)


# ── DataStream functional ops ──────────────────────────────────

class TestDataStreamFilter:

    def test_filter_by_value(self):
        ds = DataStream()
        for i in range(10):
            ds.push(DataPoint(float(i), "s", i))
        evens = ds.filter(lambda p: p.value % 2 == 0)
        assert evens.size() == 5

    def test_filter_by_source(self):
        ds = DataStream()
        ds.push(DataPoint(1, "a", 1))
        ds.push(DataPoint(2, "b", 2))
        ds.push(DataPoint(3, "a", 3))
        a_only = ds.filter(lambda p: p.source == "a")
        assert a_only.size() == 2

    def test_filter_empty_stream(self):
        ds = DataStream()
        assert ds.filter(lambda p: True).is_empty()

    def test_filter_no_match(self):
        ds = DataStream()
        ds.push(DataPoint(1, "s", 1))
        assert ds.filter(lambda p: False).is_empty()


class TestDataStreamMap:

    def test_map_doubles_value(self):
        ds = DataStream()
        for i in range(4):
            ds.push(DataPoint(float(i), "s", i))
        doubled = ds.map(lambda p: DataPoint(p.timestamp, p.source, p.value * 2))
        assert doubled.size() == 4
        assert doubled.peek().value == 0
        last = list(doubled)[-1]
        assert last.value == 6

    def test_map_changes_source(self):
        ds = DataStream()
        ds.push(DataPoint(1, "old", 1))
        mapped = ds.map(lambda p: DataPoint(p.timestamp, "new", p.value))
        assert mapped.peek().source == "new"

    def test_map_empty(self):
        assert DataStream().map(lambda p: p).is_empty()


# ── batch and window ───────────────────────────────────────────

class TestDataStreamBatch:

    def test_batch_exact(self):
        ds = DataStream()
        for i in range(6):
            ds.push(DataPoint(float(i), "s", i))
        batches = ds.batch(3)
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3

    def test_batch_with_remainder(self):
        ds = DataStream()
        for i in range(7):
            ds.push(DataPoint(float(i), "s", i))
        batches = ds.batch(3)
        assert len(batches) == 3
        assert len(batches[-1]) == 1

    def test_batch_larger_than_stream(self):
        ds = DataStream()
        ds.push(DataPoint(1, "s", 1))
        batches = ds.batch(10)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_batch_empty(self):
        assert DataStream().batch(5) == []


class TestDataStreamWindow:

    def test_window_recent_points(self):
        ds = DataStream()
        now = time.time()
        ds.push(DataPoint(now - 100, "s", "old"))
        ds.push(DataPoint(now - 1, "s", "recent"))
        ds.push(DataPoint(now, "s", "now"))
        win = ds.window(10)
        assert win.size() == 2  # -1 and now

    def test_window_empty(self):
        ds = DataStream()
        ds.push(DataPoint(time.time() - 100, "s", "x"))
        win = ds.window(10)
        assert win.is_empty()


# ── merge ──────────────────────────────────────────────────────

class TestDataStreamMerge:

    def test_merge_two_streams(self):
        a = DataStream()
        b = DataStream()
        a.push(DataPoint(2.0, "a", 2))
        a.push(DataPoint(5.0, "a", 5))
        b.push(DataPoint(1.0, "b", 1))
        b.push(DataPoint(3.0, "b", 3))
        merged = a.merge(b)
        vals = [p.value for p in merged]
        assert vals == [1, 2, 3, 5]

    def test_merge_empty(self):
        a = DataStream()
        b = DataStream()
        merged = a.merge(b)
        assert merged.is_empty()

    def test_merge_one_empty(self):
        a = DataStream()
        a.push(DataPoint(1, "a", 1))
        merged = a.merge(DataStream())
        assert merged.size() == 1
