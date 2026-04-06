"""System resilience metrics module — measure and report on system resilience.

Pure Python, zero external dependencies.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ── Data classes & enums ─────────────────────────────────────────────────

class MetricTrend(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


@dataclass
class ResilienceMetric:
    """A single resilience measurement."""
    name: str
    value: float
    target: float = 100.0
    trend: MetricTrend = MetricTrend.UNKNOWN
    timestamp: float = field(default_factory=time.time)
    unit: str = ""


@dataclass
class ResilienceReport:
    """Comprehensive resilience assessment report."""
    overall_score: float  # 0 – 100
    mttr: float  # mean time to recovery (seconds)
    mtbf: float  # mean time between failures (seconds)
    availability: float  # percentage 0 – 100
    fault_tolerance: float  # 0 – 1
    redundancy_level: float  # 0 – 1
    metrics: List[ResilienceMetric] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# ── SystemResilience ─────────────────────────────────────────────────────

class SystemResilience:
    """Computes resilience metrics and generates reports."""

    def __init__(self) -> None:
        self._metric_history: Dict[str, List[ResilienceMetric]] = {}
        self._failure_events: List[Dict[str, Any]] = []
        self._recovery_events: List[Dict[str, Any]] = []

    # ── core MTTR / MTBF calculations ──

    def compute_mean_time_to_recovery(self, history: List[Dict[str, Any]]) -> float:
        """Compute MTTR (seconds) from a list of recovery events.

        Each entry must have 'time_to_recover' (float) key.
        """
        if not history:
            return 0.0
        values = [h["time_to_recover"] for h in history if "time_to_recover" in h]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def compute_mean_time_between_failures(self, history: List[Dict[str, Any]]) -> float:
        """Compute MTBF (seconds) from a list of failure events.

        Each entry must have 'timestamp' (float) key, sorted ascending.
        """
        if len(history) < 2:
            return 0.0
        timestamps = [h["timestamp"] for h in history if "timestamp" in h]
        timestamps.sort()
        if len(timestamps) < 2:
            return 0.0
        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        return sum(intervals) / len(intervals)

    def compute_availability(self, mtbf: float, mttr: float) -> float:
        """Compute availability as a percentage (0 – 100).

        Formula: MTBF / (MTBF + MTTR) * 100
        """
        total = mtbf + mttr
        if total <= 0:
            return 100.0
        return (mtbf / total) * 100.0

    # ── fault tolerance & redundancy ──

    def compute_fault_tolerance(self, failed_components: int,
                                 total_components: int) -> float:
        """Compute fault tolerance as a ratio 0.0 – 1.0.

        Returns 1.0 if no failures; degrades as more components fail.
        """
        if total_components <= 0:
            return 1.0
        if failed_components >= total_components:
            return 0.0
        # Exponential degradation: tolerance = (1 - f/t)^2
        ratio = failed_components / total_components
        return (1.0 - ratio) ** 2

    def compute_redundancy_level(self, components: List[Dict[str, Any]]) -> float:
        """Compute redundancy score (0.0 – 1.0) from component info.

        Each component dict may have 'redundancy_count' (int) and 'critical' (bool).
        """
        if not components:
            return 0.0

        total = len(components)
        redundant = sum(1 for c in components if c.get("redundancy_count", 0) > 0)
        critical_redundant = sum(
            1 for c in components
            if c.get("critical", False) and c.get("redundancy_count", 0) > 0
        )
        critical_total = sum(1 for c in components if c.get("critical", False))

        base_score = redundant / total
        critical_bonus = 0.0
        if critical_total > 0:
            critical_bonus = (critical_redundant / critical_total) * 0.3

        return min(1.0, base_score + critical_bonus)

    # ── resilience index ──

    def compute_resilience_index(self, metrics: List[ResilienceMetric]) -> float:
        """Compute an overall resilience index (0 – 100) from a list of metrics.

        Weights each metric by proximity to its target.
        """
        if not metrics:
            return 0.0

        weighted_scores: List[float] = []
        for m in metrics:
            if m.target == 0:
                score = 100.0 if m.value == 0 else 0.0
            else:
                score = (m.value / m.target) * 100.0
            score = max(0.0, min(100.0, score))
            # Trend bonus/penalty
            if m.trend == MetricTrend.IMPROVING:
                score = min(100.0, score + 5.0)
            elif m.trend == MetricTrend.DEGRADING:
                score = max(0.0, score - 5.0)
            weighted_scores.append(score)

        return sum(weighted_scores) / len(weighted_scores)

    # ── report generation ──

    def generate_resilience_report(self, metrics: List[ResilienceMetric],
                                   failure_history: Optional[List[Dict[str, Any]]] = None,
                                   recovery_history: Optional[List[Dict[str, Any]]] = None,
                                   components: Optional[List[Dict[str, Any]]] = None) -> ResilienceReport:
        """Generate a comprehensive resilience report."""
        failure_history = failure_history or self._failure_events
        recovery_history = recovery_history or self._recovery_events
        components = components or []

        mttr = self.compute_mean_time_to_recovery(recovery_history)
        mtbf = self.compute_mean_time_between_failures(failure_history)
        availability = self.compute_availability(mtbf, mttr)

        failed = sum(1 for c in components if c.get("failed", False))
        total = len(components) if components else 1
        fault_tolerance = self.compute_fault_tolerance(failed, total)
        redundancy = self.compute_redundancy_level(components)

        resilience_index = self.compute_resilience_index(metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            availability, mttr, fault_tolerance, redundancy, resilience_index
        )

        return ResilienceReport(
            overall_score=resilience_index,
            mttr=mttr,
            mtbf=mtbf,
            availability=availability,
            fault_tolerance=fault_tolerance,
            redundancy_level=redundancy,
            metrics=metrics,
            recommendations=recommendations,
        )

    # ── metric tracking ──

    def record_metric(self, metric: ResilienceMetric) -> None:
        self._metric_history.setdefault(metric.name, []).append(metric)
        # Keep last 500
        if len(self._metric_history[metric.name]) > 500:
            self._metric_history[metric.name] = self._metric_history[metric.name][-500:]

    def record_failure(self, event: Dict[str, Any]) -> None:
        self._failure_events.append(event)

    def record_recovery(self, event: Dict[str, Any]) -> None:
        self._recovery_events.append(event)

    def get_metric_trend(self, name: str, window: int = 10) -> MetricTrend:
        """Determine trend direction for a named metric."""
        history = self._metric_history.get(name, [])
        if len(history) < 2:
            return MetricTrend.UNKNOWN
        recent = history[-window:]
        values = [m.value for m in recent]
        half = len(values) // 2
        first_avg = sum(values[:half]) / half if half > 0 else values[0]
        second_avg = sum(values[half:]) / (len(values) - half) if len(values) > half else values[-1]
        diff = second_avg - first_avg
        threshold = first_avg * 0.02 if first_avg != 0 else 0.01
        if diff > threshold:
            return MetricTrend.IMPROVING
        elif diff < -threshold:
            return MetricTrend.DEGRADING
        return MetricTrend.STABLE

    # ── internal ──

    def _generate_recommendations(self, availability: float, mttr: float,
                                  fault_tolerance: float, redundancy: float,
                                  resilience_index: float) -> List[str]:
        recs: List[str] = []
        if availability < 99.0:
            recs.append(f"Availability ({availability:.1f}%) below 99% target. Investigate MTTR reduction.")
        if mttr > 60.0:
            recs.append(f"MTTR ({mttr:.1f}s) is high. Consider automating recovery procedures.")
        if fault_tolerance < 0.7:
            recs.append(f"Fault tolerance ({fault_tolerance:.2f}) is low. Add redundancy to critical components.")
        if redundancy < 0.5:
            recs.append("Redundancy level is insufficient. Deploy backup instances for critical services.")
        if resilience_index < 70.0:
            recs.append(f"Overall resilience index ({resilience_index:.1f}) below 70. Review all subsystems.")
        if not recs:
            recs.append("All resilience metrics within acceptable ranges.")
        return recs
