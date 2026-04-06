"""Performance optimization analysis.

Provides complexity estimation, optimization suggestions,
memory-leak detection, cache-miss analysis, and algorithmic
bottleneck identification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ComplexityClass(Enum):
    O_1 = "O(1)"
    O_LOG_N = "O(log n)"
    O_N = "O(n)"
    O_N_LOG_N = "O(n log n)"
    O_N2 = "O(n²)"
    O_N3 = "O(n³)"
    O_2N = "O(2ⁿ)"
    UNKNOWN = "Unknown"


@dataclass
class OptimizationSuggestion:
    """A single optimization suggestion."""

    type: str
    location: str
    description: str
    estimated_improvement: str
    complexity: str = "low"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "location": self.location,
            "description": self.description,
            "estimated_improvement": self.estimated_improvement,
            "complexity": self.complexity,
        }


@dataclass
class ComplexityEstimate:
    time_complexity: str
    space_complexity: str


class CodeAnalyzer:
    """Static / empirical analysis of code complexity and performance."""

    def analyze_complexity(
        self, function: Callable[..., Any], inputs: List[Any],
    ) -> Tuple[str, str]:
        """Empirically estimate time and space complexity.

        Runs *function* on progressively larger inputs and fits a
        power-law to estimate the Big-O class.
        """
        times: List[float] = []
        sizes: List[int] = []
        for inp in inputs:
            start = time.perf_counter()
            function(inp)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            sizes.append(len(inp) if hasattr(inp, "__len__") else 0)

        time_c = self.compute_big_o_sequence(sizes, times)
        # Space is hard to measure without custom allocators; estimate from input
        space_c = "O(n)"
        return time_c, space_c

    def suggest_optimizations(
        self, profile_data: Dict[str, Any],
    ) -> List[OptimizationSuggestion]:
        """Analyze profile data and produce optimization suggestions."""
        suggestions: List[OptimizationSuggestion] = []
        entries = profile_data.get("entries", [])
        for entry in entries:
            total = entry.get("total_time", 0)
            name = entry.get("function_name", "unknown")
            if total > 1.0:
                suggestions.append(OptimizationSuggestion(
                    type="hotspot",
                    location=name,
                    description=f"Function '{name}' takes {total:.4f}s — consider caching or algorithmic improvement.",
                    estimated_improvement="50-80%",
                    complexity="medium",
                ))
            call_count = entry.get("call_count", 0)
            if call_count > 1000 and total < 0.01:
                suggestions.append(OptimizationSuggestion(
                    type="micro_optimization",
                    location=name,
                    description=f"Function '{name}' called {call_count} times with low per-call cost — consider batching.",
                    estimated_improvement="10-30%",
                    complexity="low",
                ))
            if entry.get("max_time", 0) / max(entry.get("avg_time", 1), 1e-12) > 10:
                suggestions.append(OptimizationSuggestion(
                    type="variance",
                    location=name,
                    description=f"Function '{name}' has high timing variance — investigate contention or GC pressure.",
                    estimated_improvement="20-50%",
                    complexity="medium",
                ))
        if not suggestions:
            suggestions.append(OptimizationSuggestion(
                type="info",
                location="system",
                description="No major optimization opportunities detected.",
                estimated_improvement="N/A",
                complexity="none",
            ))
        return suggestions

    @staticmethod
    def compute_big_o_sequence(sizes: List[int], times: List[float]) -> str:
        """Estimate the Big-O class from a sequence of (size, time) pairs.

        Compares the growth ratio to known complexity curves.
        """
        if len(sizes) < 2 or len(times) < 2:
            return "Unknown"

        # Filter out zero-size entries
        pairs = [(s, t) for s, t in zip(sizes, times) if s > 0 and t > 0]
        if len(pairs) < 2:
            return "Unknown"

        # Use ratio between last two entries
        s1, t1 = pairs[-2]
        s2, t2 = pairs[-1]
        if t1 <= 0:
            return "Unknown"
        size_ratio = s2 / s1
        time_ratio = t2 / t1

        if size_ratio <= 0:
            return "Unknown"

        if time_ratio < 1.1:
            return "O(1)"
        # Compare time_ratio to size_ratio^k for various k
        if abs(time_ratio - 1.0) < 0.3:
            return "O(1)"
        if abs(time_ratio - math_log_safe(size_ratio)) < 0.5:
            return "O(log n)"
        if abs(time_ratio - size_ratio) < 0.5 * size_ratio:
            return "O(n)"
        if abs(time_ratio - size_ratio * math_log_safe(size_ratio)) < 0.5 * size_ratio * math_log_safe(size_ratio):
            return "O(n log n)"
        if abs(time_ratio - size_ratio ** 2) < 0.5 * (size_ratio ** 2):
            return "O(n²)"
        if abs(time_ratio - size_ratio ** 3) < 0.5 * (size_ratio ** 3):
            return "O(n³)"
        if time_ratio > size_ratio ** 3:
            return "O(2ⁿ)"
        return "O(n)"  # Default fallback

    @staticmethod
    def detect_memory_leaks(resource_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect potential memory leaks from a time-series of resource snapshots."""
        if len(resource_history) < 2:
            return {"leak_detected": False, "details": "Insufficient data"}

        memory_values = [
            h.get("memory_used", h.get("memory_mb", 0)) for h in resource_history
        ]
        first = memory_values[0]
        last = memory_values[-1]
        growth = last - first
        growth_rate = growth / len(memory_values) if memory_values else 0

        leak_detected = False
        severity = "none"
        details = ""

        # Simple linear trend
        n = len(memory_values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(memory_values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(memory_values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0

        if slope > 0.1 and growth > 1.0:
            leak_detected = True
            severity = "high" if slope > 1.0 else "medium"
        elif slope > 0:
            leak_detected = True
            severity = "low"

        details = f"Memory grew {growth:.2f} units (slope={slope:.4f})"

        return {
            "leak_detected": leak_detected,
            "severity": severity,
            "growth": growth,
            "slope": slope,
            "details": details,
        }

    @staticmethod
    def compute_cache_miss_rate(cache_stats: Dict[str, Any]) -> float:
        """Compute cache miss rate from hit/miss statistics."""
        hits = cache_stats.get("hits", 0)
        misses = cache_stats.get("misses", 0)
        total = hits + misses
        if total == 0:
            return 0.0
        return misses / total

    @staticmethod
    def analyze_algorithmic_bottleneck(profile: Dict[str, Any]) -> Dict[str, Any]:
        """Identify the primary algorithmic bottleneck from profile data."""
        entries = profile.get("entries", [])
        if not entries:
            return {"bottleneck": None, "message": "No profile data available"}

        # Find entry with highest total_time
        bottleneck_entry = max(entries, key=lambda e: e.get("total_time", 0))
        total_time_all = sum(e.get("total_time", 0) for e in entries)

        pct = (
            bottleneck_entry.get("total_time", 0) / total_time_all * 100
            if total_time_all > 0
            else 0
        )

        return {
            "bottleneck": bottleneck_entry.get("function_name"),
            "total_time": bottleneck_entry.get("total_time", 0),
            "percentage": pct,
            "message": f"Bottleneck: {bottleneck_entry.get('function_name', 'unknown')} ({pct:.1f}%)",
        }


def math_log_safe(x: float) -> float:
    """Safe log that returns 0 for x <= 0."""
    import math
    if x <= 0:
        return 0.0
    return math.log(x)
