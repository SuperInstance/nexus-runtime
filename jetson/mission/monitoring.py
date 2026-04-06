"""Mission progress monitoring for NEXUS marine robotics platform.

Tracks mission progress, detects deviations, monitors objectives,
and generates status reports.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

from jetson.mission.execution import ExecutionState
from jetson.mission.planner import MissionPlan, MissionPhase


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TrendDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    UNKNOWN = "unknown"


@dataclass
class ProgressMetric:
    """A single progress metric."""
    name: str = ""
    value: float = 0.0
    target: float = 100.0
    unit: str = "%"
    trend: TrendDirection = TrendDirection.UNKNOWN
    history: List[float] = field(default_factory=list)

    def update(self, value: float):
        """Update metric value and compute trend."""
        if self.history:
            diff = value - self.history[-1]
            if diff > 0.001:
                self.trend = TrendDirection.IMPROVING
            elif diff < -0.001:
                self.trend = TrendDirection.DECLINING
            else:
                self.trend = TrendDirection.STABLE
        self.history.append(value)
        self.value = value

    def get_progress_percentage(self) -> float:
        """Get progress as percentage towards target."""
        if self.target == 0:
            return 100.0 if self.value == 0 else 0.0
        return min((self.value / self.target) * 100.0, 100.0)


@dataclass
class MissionAlert:
    """An alert about mission status."""
    level: AlertLevel
    message: str
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


@dataclass
class MissionStatus:
    """Complete mission status snapshot."""
    mission_id: str = ""
    state: str = "idle"
    progress: float = 0.0
    current_phase: str = ""
    metrics: Dict[str, ProgressMetric] = field(default_factory=dict)
    alerts: List[MissionAlert] = field(default_factory=list)
    estimated_completion: Optional[float] = None
    started_at: Optional[float] = None
    updated_at: float = field(default_factory=time.time)


@dataclass
class DeviationReport:
    """Report of plan deviations."""
    phase: str = ""
    deviation_type: str = ""
    expected: Any = None
    actual: Any = None
    magnitude: float = 0.0
    severity: float = 0.0  # 0.0 - 1.0


@dataclass
class StatusReport:
    """Comprehensive status report."""
    mission_id: str = ""
    state: str = ""
    progress: float = 0.0
    current_phase: str = ""
    phases_completed: int = 0
    phases_total: int = 0
    objectives_met: int = 0
    objectives_total: int = 0
    efficiency_score: float = 0.0
    active_alerts: int = 0
    estimated_completion: Optional[float] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)


@dataclass
class ResourceWarning:
    """Warning about resource usage."""
    resource: str = ""
    current_usage: float = 0.0
    limit: float = 0.0
    percent_used: float = 0.0
    severity: AlertLevel = AlertLevel.INFO


class MissionMonitor:
    """Monitors mission progress, detects deviations, and generates reports."""

    def __init__(self):
        self._missions: Dict[str, MissionStatus] = {}
        self._metrics_history: Dict[str, Dict[str, List[float]]] = {}
        self._alerts: Dict[str, List[MissionAlert]] = {}
        self._resource_limits: Dict[str, float] = {}

    def register_mission(self, mission_id: str, plan: Optional[MissionPlan] = None) -> MissionStatus:
        """Register a mission for monitoring."""
        status = MissionStatus(
            mission_id=mission_id,
            started_at=time.time(),
        )
        if plan:
            status.state = plan.name
            status.phases_total = len(plan.phases)  # type: ignore
        self._missions[mission_id] = status
        self._metrics_history[mission_id] = {}
        self._alerts[mission_id] = []
        return status

    def update_progress(self, mission_id: str, metric: ProgressMetric) -> MissionStatus:
        """Update progress metric for a mission. Returns updated status."""
        if mission_id not in self._missions:
            self.register_mission(mission_id)

        status = self._missions[mission_id]
        status.metrics[metric.name] = metric
        status.updated_at = time.time()

        # Track history
        if mission_id not in self._metrics_history:
            self._metrics_history[mission_id] = {}
        if metric.name not in self._metrics_history[mission_id]:
            self._metrics_history[mission_id][metric.name] = []
        self._metrics_history[mission_id][metric.name].append(metric.value)

        # Update overall progress
        total = sum(m.get_progress_percentage() for m in status.metrics.values())
        count = max(len(status.metrics), 1)
        status.progress = total / count

        return status

    def check_objectives(self, plan: MissionPlan,
                         progress: float) -> List[Dict[str, Any]]:
        """Check objective completion status based on progress."""
        objective_statuses = []
        for i, obj in enumerate(plan.objectives):
            # Each objective maps to a phase roughly
            phase_progress = min(100.0, progress * (len(plan.objectives) / max(len(plan.phases), 1)))
            phase_idx = i
            obj_progress = max(0.0, min(100.0, phase_progress - (phase_idx * (100.0 / max(len(plan.objectives), 1)))))

            objective_statuses.append({
                "objective_id": obj.id,
                "name": obj.name,
                "type": obj.type,
                "progress": round(obj_progress, 1),
                "estimated_complete": obj_progress >= 100.0,
                "on_track": obj_progress >= progress * 0.8,
            })
        return objective_statuses

    def detect_deviations(self, plan: MissionPlan,
                          actual_durations: Dict[str, float]) -> List[DeviationReport]:
        """Detect deviations between planned and actual execution."""
        deviations = []
        for phase in plan.phases:
            actual = actual_durations.get(phase.name)
            if actual is not None:
                expected = phase.duration
                if expected > 0:
                    magnitude = abs(actual - expected) / expected
                    if magnitude > 0.1:  # > 10% deviation
                        severity = min(magnitude, 1.0)
                        dev_type = "overrun" if actual > expected else "underrun"
                        deviations.append(DeviationReport(
                            phase=phase.name,
                            deviation_type=dev_type,
                            expected=expected,
                            actual=actual,
                            magnitude=round(magnitude, 3),
                            severity=round(severity, 3),
                        ))
        return deviations

    def estimate_completion(self, mission_id: str) -> Optional[float]:
        """Estimate completion time (ETA) for a mission."""
        status = self._missions.get(mission_id)
        if not status or not status.started_at:
            return None

        elapsed = time.time() - status.started_at
        if status.progress <= 0:
            return None

        # Linear extrapolation
        rate = status.progress / elapsed
        if rate <= 0:
            return None

        remaining = (100.0 - status.progress) / rate
        return time.time() + remaining

    def generate_status_report(self, mission_id: str) -> StatusReport:
        """Generate a comprehensive status report for a mission."""
        status = self._missions.get(mission_id)
        if not status:
            return StatusReport(mission_id=mission_id, state="unknown")

        report = StatusReport(
            mission_id=mission_id,
            state=status.state,
            progress=round(status.progress, 1),
            current_phase=status.current_phase,
            active_alerts=len([a for a in status.alerts if not a.acknowledged]),
            estimated_completion=self.estimate_completion(mission_id),
        )

        # Generate recommendations
        recommendations = []
        if status.progress < 25 and status.started_at:
            recommendations.append("Mission is in early stages; monitor closely")
        elif 25 <= status.progress < 75:
            recommendations.append("Mission progressing normally")
        elif status.progress >= 75:
            recommendations.append("Mission approaching completion")
        if report.active_alerts > 3:
            recommendations.append("High number of active alerts - review recommended")
        report.recommendations = recommendations

        return report

    def set_resource_limit(self, resource: str, limit: float):
        """Set a resource usage limit for monitoring."""
        self._resource_limits[resource] = limit

    def check_resource_status(self, mission_id: str,
                              current_usage: Dict[str, float]) -> List[ResourceWarning]:
        """Check resource usage against limits. Returns warnings."""
        warnings = []
        for resource, usage in current_usage.items():
            limit = self._resource_limits.get(resource)
            if limit and limit > 0:
                pct = (usage / limit) * 100.0
                warning = ResourceWarning(
                    resource=resource,
                    current_usage=usage,
                    limit=limit,
                    percent_used=round(pct, 1),
                )
                if pct > 90:
                    warning.severity = AlertLevel.CRITICAL
                elif pct > 70:
                    warning.severity = AlertLevel.WARNING
                else:
                    warning.severity = AlertLevel.INFO
                if pct > 70:
                    warnings.append(warning)

        # Store alerts
        if mission_id in self._alerts:
            for w in warnings:
                if w.severity in (AlertLevel.WARNING, AlertLevel.CRITICAL):
                    self._alerts[mission_id].append(MissionAlert(
                        level=w.severity,
                        message=f"{w.resource} at {w.percent_used}% of limit",
                        source="resource_monitor",
                    ))

        return warnings

    def compute_mission_efficiency(self, mission_id: str) -> float:
        """Compute mission efficiency score (0.0 - 1.0)."""
        status = self._missions.get(mission_id)
        if not status:
            return 0.0

        # Factor in progress rate, alert count, and metric trends
        progress_score = status.progress / 100.0
        alert_penalty = min(len(status.alerts) * 0.05, 0.3)

        # Metric trend score
        trend_score = 1.0
        for metric in status.metrics.values():
            if metric.trend == TrendDirection.DECLINING:
                trend_score *= 0.9
            elif metric.trend == TrendDirection.IMPROVING:
                trend_score *= 1.05

        trend_score = min(trend_score, 1.0)
        efficiency = (progress_score + trend_score) / 2.0 - alert_penalty
        return round(max(0.0, min(1.0, efficiency)), 3)

    def add_alert(self, mission_id: str, alert: MissionAlert):
        """Add an alert for a mission."""
        if mission_id not in self._alerts:
            self._alerts[mission_id] = []
        self._alerts[mission_id].append(alert)
        if mission_id in self._missions:
            self._missions[mission_id].alerts.append(alert)

    def get_alerts(self, mission_id: str,
                   level: Optional[AlertLevel] = None) -> List[MissionAlert]:
        """Get alerts for a mission, optionally filtered by level."""
        alerts = self._alerts.get(mission_id, [])
        if level:
            alerts = [a for a in alerts if a.level == level]
        return alerts

    def get_status(self, mission_id: str) -> Optional[MissionStatus]:
        """Get current mission status."""
        return self._missions.get(mission_id)

    def get_metric_history(self, mission_id: str,
                           metric_name: str) -> List[float]:
        """Get history of a specific metric."""
        return self._metrics_history.get(mission_id, {}).get(metric_name, [])

    def acknowledge_alert(self, mission_id: str, index: int) -> bool:
        """Acknowledge an alert."""
        alerts = self._alerts.get(mission_id, [])
        if 0 <= index < len(alerts):
            alerts[index].acknowledged = True
            return True
        return False

    def unregister_mission(self, mission_id: str) -> bool:
        """Remove a mission from monitoring."""
        if mission_id in self._missions:
            del self._missions[mission_id]
            del self._metrics_history[mission_id]
            del self._alerts[mission_id]
            return True
        return False

    def get_all_missions(self) -> List[str]:
        """Get all monitored mission ids."""
        return list(self._missions.keys())
