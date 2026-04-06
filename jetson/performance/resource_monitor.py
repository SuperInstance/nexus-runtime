"""Resource usage monitoring.

Provides snapshot-based resource tracking, trend computation,
anomaly detection, exhaustion prediction, alerting, and
efficiency scoring.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResourceSnapshot:
    """Point-in-time snapshot of system resources."""

    timestamp: float
    cpu_percent: float
    memory_used: float
    memory_total: float
    disk_io: float = 0.0
    network_io: float = 0.0
    open_files: int = 0
    thread_count: int = 0

    @property
    def memory_percent(self) -> float:
        if self.memory_total <= 0:
            return 0.0
        return (self.memory_used / self.memory_total) * 100.0

    @property
    def age(self) -> float:
        return time.time() - self.timestamp

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_used": self.memory_used,
            "memory_total": self.memory_total,
            "memory_percent": self.memory_percent,
            "disk_io": self.disk_io,
            "network_io": self.network_io,
            "open_files": self.open_files,
            "thread_count": self.thread_count,
        }


@dataclass
class ResourceAlert:
    """Alert generated when a resource crosses a threshold."""

    resource_type: str
    current_value: float
    threshold: float
    severity: str
    recommendation: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "resource_type": self.resource_type,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "severity": self.severity,
            "recommendation": self.recommendation,
        }


class ResourceMonitor:
    """Monitor system resources with trend analysis and alerting."""

    def __init__(self) -> None:
        self._snapshots: List[ResourceSnapshot] = []
        self._callbacks: List[Callable[[ResourceSnapshot], None]] = []

    def take_snapshot(self) -> ResourceSnapshot:
        """Capture a resource snapshot.

        Uses synthetic data suitable for testing / simulation.
        Replace with OS-specific calls in production.
        """
        snap = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=0.0,
            memory_used=0.0,
            memory_total=100.0,
            disk_io=0.0,
            network_io=0.0,
            open_files=0,
            thread_count=1,
        )
        self._snapshots.append(snap)
        for cb in self._callbacks:
            cb(snap)
        return snap

    def take_custom_snapshot(self, **kwargs: Any) -> ResourceSnapshot:
        """Create a snapshot with custom values for testing."""
        snap = ResourceSnapshot(
            timestamp=kwargs.get("timestamp", time.time()),
            cpu_percent=kwargs.get("cpu_percent", 0.0),
            memory_used=kwargs.get("memory_used", 0.0),
            memory_total=kwargs.get("memory_total", 100.0),
            disk_io=kwargs.get("disk_io", 0.0),
            network_io=kwargs.get("network_io", 0.0),
            open_files=kwargs.get("open_files", 0),
            thread_count=kwargs.get("thread_count", 1),
        )
        self._snapshots.append(snap)
        return snap

    def add_snapshot(self, snapshot: ResourceSnapshot) -> None:
        """Manually add a snapshot."""
        self._snapshots.append(snapshot)

    def get_snapshots(self) -> List[ResourceSnapshot]:
        return list(self._snapshots)

    def clear_snapshots(self) -> None:
        self._snapshots.clear()

    def register_callback(self, callback: Callable[[ResourceSnapshot], None]) -> None:
        self._callbacks.append(callback)

    def monitor_interval(
        self, seconds: float, callback: Callable[[ResourceSnapshot], None],
    ) -> List[ResourceSnapshot]:
        """Monitor at a fixed interval for a given duration.

        Returns collected snapshots.
        """
        collected: List[ResourceSnapshot] = []
        elapsed = 0.0
        step = min(seconds, 0.05)  # 50ms minimum step
        while elapsed < seconds:
            snap = self.take_snapshot()
            callback(snap)
            collected.append(snap)
            time.sleep(step)
            elapsed += step
        return collected

    @staticmethod
    def compute_trend(
        snapshots: List[ResourceSnapshot], metric: str,
    ) -> Dict[str, Any]:
        """Compute a linear trend for a metric across snapshots.

        *metric* is one of 'cpu_percent', 'memory_used', 'memory_percent',
        'disk_io', 'network_io', 'open_files', 'thread_count'.
        """
        if not snapshots:
            return {"metric": metric, "trend": "no_data", "slope": 0.0, "direction": "flat"}

        values = []
        for s in snapshots:
            if metric == "memory_percent":
                values.append(s.memory_percent)
            else:
                values.append(getattr(s, metric, 0.0))

        n = len(values)
        if n < 2:
            return {"metric": metric, "trend": "insufficient", "slope": 0.0, "direction": "flat"}

        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0.0

        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "flat"

        return {"metric": metric, "trend": "linear", "slope": slope, "direction": direction}

    @staticmethod
    def detect_anomalies(
        snapshots: List[ResourceSnapshot],
    ) -> List[Dict[str, Any]]:
        """Detect anomalous snapshots using simple z-score analysis."""
        if len(snapshots) < 3:
            return []

        metrics = ["cpu_percent", "memory_percent", "disk_io", "network_io"]
        anomalies: List[Dict[str, Any]] = []

        for metric in metrics:
            values: List[float] = []
            for s in snapshots:
                if metric == "memory_percent":
                    values.append(s.memory_percent)
                else:
                    values.append(getattr(s, metric, 0.0))

            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(variance) if variance > 0 else 0.0

            if std == 0:
                continue

            for i, val in enumerate(values):
                z = abs(val - mean) / std
                if z > 2.0:
                    anomalies.append({
                        "index": i,
                        "metric": metric,
                        "value": val,
                        "z_score": z,
                        "timestamp": snapshots[i].timestamp,
                    })

        return anomalies

    @staticmethod
    def predict_exhaustion(
        snapshots: List[ResourceSnapshot], resource_type: str,
    ) -> Dict[str, Any]:
        """Predict when a resource will be exhausted based on trend.

        *resource_type* is one of 'memory', 'disk', or 'cpu' (cpu saturates at 100%).
        """
        if len(snapshots) < 2:
            return {"resource_type": resource_type, "time_until_exhaustion": None, "message": "Insufficient data"}

        # Determine values and ceiling
        if resource_type == "memory":
            values = [s.memory_used for s in snapshots]
            ceiling = snapshots[0].memory_total if snapshots[0].memory_total > 0 else 100.0
        elif resource_type == "cpu":
            values = [s.cpu_percent for s in snapshots]
            ceiling = 100.0
        else:
            values = [s.disk_io for s in snapshots]
            ceiling = 100.0  # Assume 100 as generic ceiling

        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0.0

        remaining = ceiling - values[-1]
        if remaining <= 0:
            return {
                "resource_type": resource_type,
                "time_until_exhaustion": 0.0,
                "message": "Resource already exhausted",
                "slope": slope,
            }

        if slope <= 0:
            return {
                "resource_type": resource_type,
                "time_until_exhaustion": None,
                "message": "Resource usage is stable or decreasing",
                "slope": slope,
            }

        # Time between snapshots
        if n >= 2:
            dt = snapshots[-1].timestamp - snapshots[0].timestamp
            interval = dt / (n - 1)
        else:
            interval = 1.0

        ticks = remaining / slope
        seconds_left = ticks * interval

        return {
            "resource_type": resource_type,
            "time_until_exhaustion": seconds_left,
            "message": f"Estimated exhaustion in {seconds_left:.1f}s",
            "slope": slope,
            "current_value": values[-1],
            "ceiling": ceiling,
        }

    @staticmethod
    def generate_alerts(
        snapshots: List[ResourceSnapshot],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[ResourceAlert]:
        """Generate alerts for the latest snapshot based on thresholds."""
        if not snapshots:
            return []

        if thresholds is None:
            thresholds = {
                "cpu_percent": 90.0,
                "memory_percent": 85.0,
                "disk_io": 80.0,
                "network_io": 80.0,
            }

        latest = snapshots[-1]
        alerts: List[ResourceAlert] = []

        cpu_val = latest.cpu_percent
        cpu_thresh = thresholds.get("cpu_percent", 90.0)
        if cpu_val >= cpu_thresh:
            severity = "critical" if cpu_val >= 99 else "high"
            alerts.append(ResourceAlert(
                resource_type="cpu_percent",
                current_value=cpu_val,
                threshold=cpu_thresh,
                severity=severity,
                recommendation="Reduce CPU load or scale horizontally.",
            ))

        mem_val = latest.memory_percent
        mem_thresh = thresholds.get("memory_percent", 85.0)
        if mem_val >= mem_thresh:
            severity = "critical" if mem_val >= 95 else "high"
            alerts.append(ResourceAlert(
                resource_type="memory_percent",
                current_value=mem_val,
                threshold=mem_thresh,
                severity=severity,
                recommendation="Free memory or increase cache eviction.",
            ))

        disk_val = latest.disk_io
        disk_thresh = thresholds.get("disk_io", 80.0)
        if disk_val >= disk_thresh:
            alerts.append(ResourceAlert(
                resource_type="disk_io",
                current_value=disk_val,
                threshold=disk_thresh,
                severity="medium",
                recommendation="Reduce disk I/O or add faster storage.",
            ))

        net_val = latest.network_io
        net_thresh = thresholds.get("network_io", 80.0)
        if net_val >= net_thresh:
            alerts.append(ResourceAlert(
                resource_type="network_io",
                current_value=net_val,
                threshold=net_thresh,
                severity="medium",
                recommendation="Optimize network traffic or increase bandwidth.",
            ))

        return alerts

    @staticmethod
    def compute_resource_efficiency(
        snapshots: List[ResourceSnapshot],
    ) -> Dict[str, float]:
        """Compute an efficiency score for each resource (0–100).

        100 = perfectly utilized, 0 = wasted or overloaded.
        Uses a parabolic model peaking at 60% utilization.
        """
        if not snapshots:
            return {}

        result: Dict[str, float] = {}
        metrics = {
            "cpu_percent": "cpu",
            "memory_percent": "memory",
            "disk_io": "disk",
            "network_io": "network",
        }

        for snap_attr, name in metrics.items():
            values: List[float] = []
            for s in snapshots:
                if snap_attr == "memory_percent":
                    values.append(s.memory_percent)
                else:
                    values.append(getattr(s, snap_attr, 0.0))

            if not values:
                result[name] = 0.0
                continue

            avg = sum(values) / len(values)
            # Parabolic efficiency: peak at 60%, drops on both sides
            # f(x) = 100 * (1 - ((x - 60) / 60)^2)
            normalized = (avg - 60.0) / 60.0
            efficiency = max(0.0, 100.0 * (1.0 - normalized ** 2))
            result[name] = round(efficiency, 2)

        return result
