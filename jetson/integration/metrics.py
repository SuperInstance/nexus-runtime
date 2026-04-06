"""
System-wide metrics collection — gauges, counters, histograms,
aggregation, statistics, and export.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Metric:
    name: str
    value: float
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricCollector:
    """Record, aggregate, and export system metrics."""

    def __init__(self) -> None:
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, metric: Metric) -> None:
        self._metrics[metric.name].append(metric)

    def record_gauge(self, name: str, value: float, unit: str = "",
                     tags: Optional[Dict[str, str]] = None) -> None:
        m = Metric(name=name, value=value, unit=unit,
                   tags=tags or {})
        self._metrics[name].append(m)

    def record_counter(self, name: str, increment: float = 1.0) -> None:
        self._counters[name] += increment
        self._metrics[name].append(
            Metric(name=name, value=self._counters[name], unit="count"))

    def record_histogram(self, name: str, value: float,
                         unit: str = "",
                         tags: Optional[Dict[str, str]] = None) -> None:
        m = Metric(name=name, value=value, unit=unit,
                   tags=tags or {})
        self._metrics[name].append(m)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_metric(self, name: str) -> List[Metric]:
        return list(self._metrics.get(name, []))

    def get_all_metrics(self) -> Dict[str, List[Metric]]:
        return {k: list(v) for k, v in self._metrics.items()}

    def get_metric_names(self) -> List[str]:
        return list(self._metrics.keys())

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self, metric_name: str, agg_type: str = "mean",
                  window: Optional[float] = None) -> Optional[float]:
        """Aggregate metric values. agg_type: mean|sum|min|max|count|last."""
        values = self._metrics.get(metric_name, [])
        if window is not None:
            cutoff = time.time() - window
            values = [m for m in values if m.timestamp >= cutoff]
        nums = [m.value for m in values]
        if not nums:
            return None
        if agg_type == "mean":
            return sum(nums) / len(nums)
        if agg_type == "sum":
            return sum(nums)
        if agg_type == "min":
            return min(nums)
        if agg_type == "max":
            return max(nums)
        if agg_type == "count":
            return float(len(nums))
        if agg_type == "last":
            return nums[-1]
        return None

    def compute_statistics(self, metric_name: str) -> Dict[str, Optional[float]]:
        values = self._metrics.get(metric_name, [])
        nums = [m.value for m in values]
        if not nums:
            return {
                "mean": None, "std": None, "min": None, "max": None,
                "p50": None, "p95": None, "p99": None, "count": 0,
            }
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        mean = sum(nums) / n
        variance = sum((x - mean) ** 2 for x in nums) / n if n > 0 else 0
        std = math.sqrt(variance)

        def _percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            d0 = data[int(f)] * (c - k)
            d1 = data[int(c)] * (k - f)
            return d0 + d1

        return {
            "mean": mean,
            "std": std,
            "min": sorted_nums[0],
            "max": sorted_nums[-1],
            "p50": _percentile(sorted_nums, 50),
            "p95": _percentile(sorted_nums, 95),
            "p99": _percentile(sorted_nums, 99),
            "count": float(n),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_metrics(self, name: Optional[str] = None) -> None:
        if name is None:
            self._metrics.clear()
            self._counters.clear()
        else:
            self._metrics.pop(name, None)
            self._counters.pop(name, None)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_metrics(self, fmt: str = "text") -> str:
        if fmt == "json":
            return self._export_json()
        return self._export_text()

    def _export_text(self) -> str:
        lines: List[str] = ["NEXUS Metrics Export"]
        for name, metrics in self._metrics.items():
            lines.append(f"\n[{name}]")
            for m in metrics:
                tag_str = ""
                if m.tags:
                    tag_str = " " + " ".join(f"{k}={v}" for k, v in m.tags.items())
                lines.append(f"  {m.value} {m.unit}{tag_str} @ {m.timestamp:.3f}")
        return "\n".join(lines)

    def _export_json(self) -> str:
        import json
        payload: Dict[str, Any] = {}
        for name, metrics in self._metrics.items():
            payload[name] = [
                {
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp,
                    "tags": m.tags,
                }
                for m in metrics
            ]
        return json.dumps(payload, indent=2)
