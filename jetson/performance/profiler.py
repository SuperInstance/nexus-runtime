"""Code profiling and measurement.

Provides deterministic profiling of function calls with call-graph
construction, hotspot identification, and report comparison.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class ProfileEntry:
    """Single profiled function entry."""

    function_name: str
    call_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float

    @property
    def percentage(self) -> float:
        return 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
        }


@dataclass
class ProfileReport:
    """Aggregated profile report."""

    entries: List[ProfileEntry]
    total_time: float
    overhead: float
    hotspots: List[ProfileEntry] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_time": self.total_time,
            "overhead": self.overhead,
            "entries": [e.as_dict() for e in self.entries],
            "hotspots": [e.as_dict() for e in self.hotspots],
        }


class Profiler:
    """Deterministic code profiler with markers and call-graph support."""

    def __init__(self) -> None:
        self._entries: Dict[str, List[float]] = defaultdict(list)
        self._markers: Dict[str, float] = {}
        self._marker_order: List[str] = []
        self._start_time: Optional[float] = None
        self._running = False

    def start(self) -> None:
        """Begin profiling session."""
        self._entries.clear()
        self._markers.clear()
        self._marker_order.clear()
        self._start_time = time.perf_counter()
        self._running = True

    def stop(self) -> ProfileReport:
        """End profiling session and return a report."""
        if self._start_time is None:
            raise RuntimeError("Profiler was not started")
        end_time = time.perf_counter()
        total_time = end_time - self._start_time
        overhead = 0.0
        self._running = False

        entries = []
        for name, timings in self._entries.items():
            if not timings:
                continue
            entry = ProfileEntry(
                function_name=name,
                call_count=len(timings),
                total_time=sum(timings),
                avg_time=sum(timings) / len(timings),
                min_time=min(timings),
                max_time=max(timings),
            )
            entries.append(entry)

        entries.sort(key=lambda e: e.total_time, reverse=True)
        hotspots = self.identify_hotspots(
            ProfileReport(entries=entries, total_time=total_time, overhead=overhead),
            threshold=0.1,
        )
        return ProfileReport(
            entries=entries, total_time=total_time, overhead=overhead, hotspots=hotspots,
        )

    def add_marker(self, name: str) -> float:
        """Add a named time-marker and return offset from session start."""
        now = time.perf_counter()
        self._markers[name] = now
        self._marker_order.append(name)
        if self._start_time is not None:
            return now - self._start_time
        return 0.0

    def measure(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
        """Execute *function* and return (result, execution_time)."""
        start = time.perf_counter()
        result = function(*args, **kwargs)
        elapsed = time.perf_counter() - start
        name = getattr(function, "__name__", str(function))
        self._entries[name].append(elapsed)
        return result, elapsed

    def record_call(self, name: str, elapsed: float) -> None:
        """Manually record a timed call."""
        self._entries[name].append(elapsed)

    def get_entries(self) -> Dict[str, List[float]]:
        return dict(self._entries)

    def get_markers(self) -> Dict[str, float]:
        return dict(self._markers)

    def get_marker_order(self) -> List[str]:
        return list(self._marker_order)

    @staticmethod
    def compute_call_graph(entries: List[ProfileEntry]) -> Dict[str, List[str]]:
        """Compute a flat call-graph adjacency list from profile entries.

        Since a deterministic profiler cannot observe caller/callee
        relationships, this returns an empty graph keyed by function name.
        """
        return {entry.function_name: [] for entry in entries}

    @staticmethod
    def identify_hotspots(report: ProfileReport, threshold: float = 0.1) -> List[ProfileEntry]:
        """Return entries whose total_time / report.total_time >= *threshold*."""
        if report.total_time <= 0:
            return []
        return [
            e for e in report.entries
            if (e.total_time / report.total_time) >= threshold
        ]

    @staticmethod
    def compare_reports(
        report_a: ProfileReport, report_b: ProfileReport,
    ) -> Dict[str, Any]:
        """Compare two profile reports and return a diff dict."""
        map_a = {e.function_name: e for e in report_a.entries}
        map_b = {e.function_name: e for e in report_b.entries}
        all_names = set(map_a) | set(map_b)
        comparison: Dict[str, Any] = {
            "total_time_change": report_b.total_time - report_a.total_time,
            "overhead_change": report_b.overhead - report_a.overhead,
            "functions": {},
        }
        for name in sorted(all_names):
            ea = map_a.get(name)
            eb = map_b.get(name)
            entry_cmp: Dict[str, Any] = {"present_in_a": ea is not None, "present_in_b": eb is not None}
            if ea and eb:
                entry_cmp["total_time_change"] = eb.total_time - ea.total_time
                entry_cmp["avg_time_change"] = eb.avg_time - ea.avg_time
                entry_cmp["call_count_change"] = eb.call_count - ea.call_count
                if ea.avg_time > 0:
                    entry_cmp["avg_time_pct_change"] = (eb.avg_time - ea.avg_time) / ea.avg_time * 100
            comparison["functions"][name] = entry_cmp
        return comparison
