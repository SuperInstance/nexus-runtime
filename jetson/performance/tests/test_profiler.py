"""Tests for jetson.performance.profiler — ProfileEntry, ProfileReport, Profiler."""

import time
import pytest
from jetson.performance.profiler import ProfileEntry, ProfileReport, Profiler

class TestProfileEntry:
    def test_construction(self):
        e = ProfileEntry("foo", 10, 1.0, 0.1, 0.05, 0.2)
        assert e.function_name == "foo"
        assert e.call_count == 10
        assert e.total_time == pytest.approx(1.0)

    def test_percentage_property(self):
        e = ProfileEntry("bar", 1, 0.5, 0.5, 0.5, 0.5)
        assert e.percentage == 0.0

    def test_as_dict(self):
        e = ProfileEntry("baz", 3, 0.3, 0.1, 0.08, 0.12)
        d = e.as_dict()
        assert d["function_name"] == "baz"
        assert d["call_count"] == 3

    def test_as_dict_keys_match_fields(self):
        e = ProfileEntry("fn", 1, 1.0, 1.0, 1.0, 1.0)
        d = e.as_dict()
        assert set(d.keys()) == {"function_name","call_count","total_time","avg_time","min_time","max_time"}

    def test_single_call_entry(self):
        e = ProfileEntry("solo", 1, 0.5, 0.5, 0.5, 0.5)
        assert e.min_time == e.max_time

class TestProfileReport:
    def test_construction(self):
        entries = [ProfileEntry("a", 1, 0.1, 0.1, 0.1, 0.1)]
        r = ProfileReport(entries=entries, total_time=1.0, overhead=0.01)
        assert len(r.entries) == 1
        assert r.total_time == 1.0

    def test_default_hotspots(self):
        r = ProfileReport(entries=[], total_time=0.0, overhead=0.0)
        assert r.hotspots == []

    def test_as_dict(self):
        entries = [ProfileEntry("x", 2, 0.2, 0.1, 0.08, 0.12)]
        r = ProfileReport(entries=entries, total_time=2.0, overhead=0.05, hotspots=entries)
        d = r.as_dict()
        assert d["total_time"] == 2.0
        assert len(d["entries"]) == 1
        assert len(d["hotspots"]) == 1

    def test_empty_report(self):
        r = ProfileReport(entries=[], total_time=0.0, overhead=0.0)
        d = r.as_dict()
        assert d["entries"] == []

class TestProfiler:
    def test_start_stop(self):
        p = Profiler()
        p.start()
        p.stop()

    def test_start_sets_running(self):
        p = Profiler()
        p.start()
        assert p._running

    def test_stop_clears_running(self):
        p = Profiler()
        p.start()
        p.stop()
        assert not p._running

    def test_stop_without_start_raises(self):
        p = Profiler()
        with pytest.raises(RuntimeError, match="not started"):
            p.stop()

    def test_measure_returns_tuple(self):
        p = Profiler()
        p.start()
        result, elapsed = p.measure(lambda: sum(range(100)))
        assert result == 4950
        assert elapsed >= 0
        p.stop()

    def test_measure_records_entry(self):
        p = Profiler()
        p.start()
        p.measure(lambda: 42)
        p.stop()
        assert "<lambda>" in p.get_entries()

    def test_measure_multiple_calls(self):
        p = Profiler()
        p.start()
        for _ in range(5):
            p.measure(lambda: 1)
        p.stop()
        assert len(p.get_entries()["<lambda>"]) == 5

    def test_measure_with_args(self):
        p = Profiler()
        p.start()
        result, _ = p.measure(lambda x, y: x + y, 3, 4)
        assert result == 7
        p.stop()

    def test_measure_with_kwargs(self):
        p = Profiler()
        p.start()
        result, _ = p.measure(lambda a=0, b=0: a * b, b=5, a=6)
        assert result == 30
        p.stop()

    def test_measure_elapsed_positive(self):
        p = Profiler()
        p.start()
        _, elapsed = p.measure(lambda: time.sleep(0.001))
        assert elapsed > 0
        p.stop()

    def test_add_marker(self):
        p = Profiler()
        p.start()
        offset = p.add_marker("m1")
        assert offset >= 0
        p.stop()

    def test_add_multiple_markers(self):
        p = Profiler()
        p.start()
        p.add_marker("a")
        p.add_marker("b")
        markers = p.get_markers()
        assert "a" in markers and "b" in markers
        p.stop()

    def test_marker_order_preserved(self):
        p = Profiler()
        p.start()
        p.add_marker("first")
        p.add_marker("second")
        assert p.get_marker_order() == ["first", "second"]
        p.stop()

    def test_marker_offsets_increasing(self):
        p = Profiler()
        p.start()
        o1 = p.add_marker("m1")
        time.sleep(0.005)
        o2 = p.add_marker("m2")
        assert o2 > o1
        p.stop()

    def test_record_call_manual(self):
        p = Profiler()
        p.start()
        p.record_call("manual_fn", 0.05)
        p.stop()
        assert "manual_fn" in p.get_entries()

    def test_report_has_entries_after_measure(self):
        p = Profiler()
        p.start()
        p.measure(lambda: 1)
        report = p.stop()
        assert len(report.entries) >= 1

    def test_report_total_time_positive(self):
        p = Profiler()
        p.start()
        time.sleep(0.001)
        report = p.stop()
        assert report.total_time > 0

    def test_report_entries_sorted(self):
        p = Profiler()
        p.start()
        for _ in range(10): p.record_call("fast", 0.001)
        for _ in range(5): p.record_call("slow", 0.01)
        report = p.stop()
        if len(report.entries) >= 2:
            assert report.entries[0].total_time >= report.entries[1].total_time

    def test_identify_hotspots_above(self):
        e = ProfileEntry("hot", 1, 0.5, 0.5, 0.5, 0.5)
        r = ProfileReport([e], 1.0, 0.0)
        assert len(Profiler.identify_hotspots(r, 0.3)) == 1

    def test_identify_hotspots_below(self):
        e = ProfileEntry("cool", 1, 0.05, 0.05, 0.05, 0.05)
        r = ProfileReport([e], 1.0, 0.0)
        assert Profiler.identify_hotspots(r, 0.3) == []

    def test_identify_hotspots_zero_total(self):
        e = ProfileEntry("x", 1, 0.0, 0.0, 0.0, 0.0)
        r = ProfileReport([e], 0.0, 0.0)
        assert Profiler.identify_hotspots(r) == []

    def test_identify_hotspots_custom_threshold(self):
        entries = [ProfileEntry("a", 1, 0.15, 0.15, 0.15, 0.15), ProfileEntry("b", 1, 0.85, 0.85, 0.85, 0.85)]
        r = ProfileReport(entries, 1.0, 0.0)
        spots = Profiler.identify_hotspots(r, 0.5)
        assert len(spots) == 1 and spots[0].function_name == "b"

    def test_compute_call_graph(self):
        e = ProfileEntry("a", 1, 0.1, 0.1, 0.1, 0.1)
        g = Profiler.compute_call_graph([e])
        assert "a" in g and g["a"] == []

    def test_compute_call_graph_multiple(self):
        entries = [ProfileEntry("a", 1, 0.1, 0.1, 0.1, 0.1), ProfileEntry("b", 2, 0.2, 0.1, 0.1, 0.1)]
        g = Profiler.compute_call_graph(entries)
        assert len(g) == 2

    def test_compute_call_graph_empty(self):
        assert Profiler.compute_call_graph([]) == {}

    def test_compare_reports_same(self):
        e = ProfileEntry("fn", 5, 0.5, 0.1, 0.05, 0.2)
        r1 = ProfileReport([e], 1.0, 0.0)
        r2 = ProfileReport([ProfileEntry("fn", 5, 0.5, 0.1, 0.05, 0.2)], 1.0, 0.0)
        cmp = Profiler.compare_reports(r1, r2)
        assert cmp["total_time_change"] == 0.0

    def test_compare_reports_different(self):
        e1 = ProfileEntry("fn", 5, 0.5, 0.1, 0.05, 0.2)
        e2 = ProfileEntry("fn", 10, 1.0, 0.1, 0.05, 0.2)
        r1 = ProfileReport([e1], 1.0, 0.0)
        r2 = ProfileReport([e2], 2.0, 0.01)
        cmp = Profiler.compare_reports(r1, r2)
        assert cmp["functions"]["fn"]["call_count_change"] == 5

    def test_compare_reports_new_function(self):
        r1 = ProfileReport([ProfileEntry("old", 1, 0.1, 0.1, 0.1, 0.1)], 0.5, 0.0)
        r2 = ProfileReport([ProfileEntry("new", 1, 0.1, 0.1, 0.1, 0.1)], 0.5, 0.0)
        cmp = Profiler.compare_reports(r1, r2)
        assert cmp["functions"]["old"]["present_in_b"] is False
        assert cmp["functions"]["new"]["present_in_a"] is False

    def test_compare_reports_pct_change(self):
        r1 = ProfileReport([ProfileEntry("fn", 1, 0.1, 0.1, 0.1, 0.1)], 0.5, 0.0)
        r2 = ProfileReport([ProfileEntry("fn", 1, 0.2, 0.2, 0.2, 0.2)], 0.5, 0.0)
        cmp = Profiler.compare_reports(r1, r2)
        assert cmp["functions"]["fn"]["avg_time_pct_change"] == pytest.approx(100.0)

    def test_get_entries_empty(self):
        assert Profiler().get_entries() == {}

    def test_get_markers_empty(self):
        assert Profiler().get_markers() == {}

    def test_multiple_start_stop_cycles(self):
        p = Profiler()
        for _ in range(3):
            p.start()
            p.measure(lambda: 1)
            p.stop()
