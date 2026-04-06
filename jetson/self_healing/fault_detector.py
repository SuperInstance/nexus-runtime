"""Fault detection module — automatic health monitoring and fault identification.

Pure Python, zero external dependencies.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ── Severity & category enums ─────────────────────────────────────────────

class FaultSeverity(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __ge__(self, other):
        if isinstance(other, FaultSeverity):
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, FaultSeverity):
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, FaultSeverity):
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, FaultSeverity):
            return self.value < other.value
        return NotImplemented


class FaultCategory(Enum):
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    SENSOR = "sensor"
    POWER = "power"
    THERMAL = "thermal"
    MEMORY = "memory"
    UNKNOWN = "unknown"


class IndicatorStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class HealthIndicator:
    """Represents a single monitored health indicator for a component."""
    component: str
    metric_name: str
    value: float
    normal_range: Tuple[float, float] = (0.0, 100.0)
    status: IndicatorStatus = IndicatorStatus.UNKNOWN
    timestamp: float = field(default_factory=time.time)

    def evaluate(self) -> IndicatorStatus:
        """Re-evaluate status based on current value vs normal range."""
        lo, hi = self.normal_range
        margin = (hi - lo) * 0.1  # 10 % margin band
        if lo - margin <= self.value <= hi + margin:
            return IndicatorStatus.HEALTHY
        elif lo - margin * 2 <= self.value <= hi + margin * 2:
            return IndicatorStatus.WARNING
        else:
            return IndicatorStatus.CRITICAL


@dataclass
class FaultEvent:
    """Represents a detected fault."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    component: str = ""
    fault_type: str = ""
    severity: FaultSeverity = FaultSeverity.NONE
    timestamp: float = field(default_factory=time.time)
    symptoms: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DegradationReport:
    """Summary of degradation detected over time."""
    component: str
    metric_name: str
    start_value: float
    end_value: float
    degradation_pct: float
    trend: str  # "improving" | "stable" | "degrading"
    sample_count: int
    time_span_seconds: float


# ── FaultDetector ─────────────────────────────────────────────────────────

class FaultDetector:
    """Monitors registered health indicators and detects faults."""

    def __init__(self) -> None:
        self._indicators: Dict[str, HealthIndicator] = {}
        self._fault_history: List[FaultEvent] = []
        self._indicator_history: Dict[str, List[HealthIndicator]] = {}
        self._anomaly_threshold: float = 2.0  # std deviations

    # ── indicator management ──

    def register_health_indicator(self, indicator: HealthIndicator) -> None:
        """Register (or update) a health indicator."""
        key = f"{indicator.component}:{indicator.metric_name}"
        indicator.status = indicator.evaluate()
        self._indicators[key] = indicator
        self._indicator_history.setdefault(key, []).append(indicator)
        # keep last 1000 samples
        if len(self._indicator_history[key]) > 1000:
            self._indicator_history[key] = self._indicator_history[key][-1000:]

    def check_indicators(self) -> List[HealthIndicator]:
        """Return a snapshot of all registered indicators (re-evaluated)."""
        result: List[HealthIndicator] = []
        for key, ind in self._indicators.items():
            ind.status = ind.evaluate()
            result.append(ind)
        return result

    # ── fault detection ──

    def detect_fault(self, indicators: List[HealthIndicator]) -> Optional[FaultEvent]:
        """Scan a list of indicators and return a FaultEvent if any is unhealthy."""
        worst_severity = FaultSeverity.NONE
        worst_indicator: Optional[HealthIndicator] = None

        for ind in indicators:
            ind.status = ind.evaluate()
            if ind.status == IndicatorStatus.CRITICAL:
                severity = FaultSeverity.HIGH
            elif ind.status == IndicatorStatus.WARNING:
                severity = FaultSeverity.MEDIUM
            else:
                continue
            if severity > worst_severity:
                worst_severity = severity
                worst_indicator = ind

        if worst_indicator is None:
            return None

        fault = FaultEvent(
            component=worst_indicator.component,
            fault_type=f"{worst_indicator.metric_name}_anomaly",
            severity=worst_severity,
            symptoms=[f"{worst_indicator.metric_name}={worst_indicator.value:.4f} "
                       f"(normal {worst_indicator.normal_range})"],
            context={"metric_name": worst_indicator.metric_name,
                      "value": worst_indicator.value,
                      "normal_range": worst_indicator.normal_range},
        )
        self._fault_history.append(fault)
        return fault

    def classify_fault(self, fault: FaultEvent) -> FaultCategory:
        """Classify a fault into a category based on its type string."""
        ft = fault.fault_type.lower()
        component = fault.component.lower()

        rules: List[Tuple[str, FaultCategory]] = [
            ("temp", FaultCategory.THERMAL),
            ("thermal", FaultCategory.THERMAL),
            ("cpu_temp", FaultCategory.THERMAL),
            ("memory", FaultCategory.MEMORY),
            ("mem_", FaultCategory.MEMORY),
            ("ram", FaultCategory.MEMORY),
            ("network", FaultCategory.NETWORK),
            ("latency", FaultCategory.NETWORK),
            ("packet", FaultCategory.NETWORK),
            ("sensor", FaultCategory.SENSOR),
            ("gps", FaultCategory.SENSOR),
            ("imu", FaultCategory.SENSOR),
            ("lidar", FaultCategory.SENSOR),
            ("power", FaultCategory.POWER),
            ("voltage", FaultCategory.POWER),
            ("current", FaultCategory.POWER),
            ("battery", FaultCategory.POWER),
            ("disk", FaultCategory.HARDWARE),
            ("io", FaultCategory.HARDWARE),
            ("motor", FaultCategory.HARDWARE),
            ("actuator", FaultCategory.HARDWARE),
            ("firmware", FaultCategory.SOFTWARE),
            ("crash", FaultCategory.SOFTWARE),
            ("exception", FaultCategory.SOFTWARE),
            ("timeout", FaultCategory.SOFTWARE),
        ]

        for pattern, cat in rules:
            if pattern in ft or pattern in component:
                return cat
        return FaultCategory.UNKNOWN

    def compute_severity(self, fault: FaultEvent) -> FaultSeverity:
        """Re-compute severity based on symptoms and context."""
        symptoms = fault.symptoms
        ctx = fault.context or {}

        # Count critical indicators in context
        indicator_count = ctx.get("indicator_count", len(symptoms))
        value = ctx.get("value", 0.0)
        normal_range = ctx.get("normal_range", (0, 100))

        if indicator_count >= 3:
            return FaultSeverity.CRITICAL

        # How far outside normal range?
        lo, hi = normal_range
        span = hi - lo if hi > lo else 1.0
        if value < lo:
            deviation = (lo - value) / span
        elif value > hi:
            deviation = (value - hi) / span
        else:
            deviation = 0.0

        if deviation > 0.5:
            return FaultSeverity.CRITICAL
        elif deviation > 0.2:
            return FaultSeverity.HIGH
        elif deviation > 0.05:
            return FaultSeverity.MEDIUM
        else:
            return FaultSeverity.LOW

    def detect_degradation(self, indicator_history: List[HealthIndicator]) -> Optional[DegradationReport]:
        """Analyse a time-series of indicators for gradual degradation."""
        if len(indicator_history) < 3:
            return None

        component = indicator_history[0].component
        metric_name = indicator_history[0].metric_name
        values = [h.value for h in indicator_history]
        timestamps = [h.timestamp for h in indicator_history]

        start_val = values[0]
        end_val = values[-1]
        change = end_val - start_val
        span = start_val if start_val != 0 else 1.0
        degradation_pct = abs(change) / span * 100.0

        # Simple linear-trend sign
        mid_idx = len(values) // 2
        first_half_avg = sum(values[:mid_idx]) / mid_idx if mid_idx else start_val
        second_half_avg = sum(values[mid_idx:]) / (len(values) - mid_idx) if len(values) > mid_idx else end_val

        if second_half_avg < first_half_avg - 0.01:
            trend = "degrading"
        elif second_half_avg > first_half_avg + 0.01:
            trend = "improving"
        else:
            trend = "stable"

        time_span = timestamps[-1] - timestamps[0]

        return DegradationReport(
            component=component,
            metric_name=metric_name,
            start_value=start_val,
            end_value=end_val,
            degradation_pct=degradation_pct,
            trend=trend,
            sample_count=len(values),
            time_span_seconds=time_span,
        )

    def get_fault_history(self, component: Optional[str] = None, limit: int = 50) -> List[FaultEvent]:
        """Return fault history, optionally filtered by component."""
        if component is None:
            return list(self._fault_history[-limit:])
        filtered = [f for f in self._fault_history if f.component == component]
        return filtered[-limit:]

    # ── helpers ──

    @property
    def anomaly_threshold(self) -> float:
        return self._anomaly_threshold

    @anomaly_threshold.setter
    def anomaly_threshold(self, val: float) -> None:
        self._anomaly_threshold = max(0.1, val)

    def get_indicator_history(self, component: str, metric_name: str) -> List[HealthIndicator]:
        key = f"{component}:{metric_name}"
        return list(self._indicator_history.get(key, []))

    def clear_fault_history(self) -> None:
        self._fault_history.clear()
