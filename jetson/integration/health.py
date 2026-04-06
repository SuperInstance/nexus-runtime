"""
System health monitoring — tracks subsystem health, detects degradation,
computes uptime, and produces health reports.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SubsystemHealth:
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: float = 0.0
    details: str = ""
    response_time_ms: float = 0.0
    history: List[HealthStatus] = field(default_factory=list)
    check_timestamps: List[float] = field(default_factory=list)


class SystemHealthMonitor:
    """Monitors health of registered subsystems over time."""

    def __init__(self) -> None:
        self._health: Dict[str, SubsystemHealth] = {}
        self._history_limit: int = 1000

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_subsystem(self, name: str) -> None:
        if name not in self._health:
            self._health[name] = SubsystemHealth(name=name)

    def unregister_subsystem(self, name: str) -> bool:
        return self._health.pop(name, None) is not None

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_health(self, name: str, status: HealthStatus,
                      details: str = "", response_time_ms: float = 0.0) -> None:
        if name not in self._health:
            self.register_subsystem(name)
        h = self._health[name]
        h.status = status
        h.last_check = time.time()
        h.details = details
        h.response_time_ms = response_time_ms
        h.history.append(status)
        h.check_timestamps.append(h.last_check)
        if len(h.history) > self._history_limit:
            h.history = h.history[-self._history_limit:]
            h.check_timestamps = h.check_timestamps[-self._history_limit:]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_health(self, name: str) -> Optional[SubsystemHealth]:
        return self._health.get(name)

    def get_overall_health(self) -> HealthStatus:
        if not self._health:
            return HealthStatus.UNKNOWN
        statuses = [h.status for h in self._health.values()]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        return HealthStatus.UNKNOWN

    def compute_uptime(self, subsystem: str, period: float = 3600.0) -> float:
        """Return fraction of checks in the last *period* seconds that were HEALTHY."""
        h = self._health.get(subsystem)
        if h is None or not h.check_timestamps:
            return 0.0
        cutoff = time.time() - period
        healthy_count = 0
        total = 0
        for ts, st in zip(h.check_timestamps, h.history):
            if ts >= cutoff:
                total += 1
                if st == HealthStatus.HEALTHY:
                    healthy_count += 1
        return healthy_count / total if total > 0 else 0.0

    def detect_degradation(self, subsystem: str,
                           history: Optional[List[HealthStatus]] = None) -> Tuple[bool, str]:
        """Return (is_degraded, reason)."""
        if history is None:
            h = self._health.get(subsystem)
            if h is None:
                return (False, "subsystem not found")
            history = h.history
        if len(history) < 3:
            return (False, "insufficient data")
        recent = history[-5:]
        degraded_count = sum(1 for s in recent if s == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for s in recent if s == HealthStatus.UNHEALTHY)
        if unhealthy_count >= 3:
            return (True, "multiple unhealthy checks")
        if degraded_count >= 2:
            return (True, "repeated degraded checks")
        return (False, "ok")

    def get_health_report(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "overall": self.get_overall_health().value,
            "subsystems": {},
            "total": len(self._health),
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "unknown": 0,
        }
        for name, h in self._health.items():
            report["subsystems"][name] = {
                "status": h.status.value,
                "last_check": h.last_check,
                "details": h.details,
                "response_time_ms": h.response_time_ms,
                "check_count": len(h.history),
            }
            key = h.status.value
            if key in report:
                report[key] += 1
        return report

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def set_history_limit(self, limit: int) -> None:
        self._history_limit = limit

    def get_registered_subsystems(self) -> List[str]:
        return list(self._health.keys())
