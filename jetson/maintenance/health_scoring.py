"""
Equipment health scoring for marine robotics predictive maintenance.

Provides sensor reading models, health aggregation, trend detection,
and structured reporting. Pure Python – no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SensorReading:
    """Single sensor measurement."""
    timestamp: float
    sensor_id: str
    value: float
    unit: str = ""
    quality: float = 1.0  # 0-1 signal quality


@dataclass
class EquipmentHealth:
    """Aggregated equipment health snapshot."""
    equipment_id: str
    overall_score: float = 1.0
    subsystem_scores: Dict[str, float] = field(default_factory=dict)
    trend: str = "stable"  # improving, degrading, stable
    last_updated: float = 0.0


class HealthScorer:
    """Compute health scores from sensor readings using deviation-based scoring."""

    # ── public API ──────────────────────────────────────────────

    def compute_health_score(
        self,
        readings: List[SensorReading],
        baseline: float,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> float:
        """Return a health score in [0, 1].

        Parameters
        ----------
        readings : list[SensorReading]
            Recent sensor readings (ideally sorted by timestamp).
        baseline : float
            Nominal / expected value for the sensor.
        thresholds : dict, optional
            Keys: ``warning`` and ``critical`` (absolute deviation from baseline).
            Defaults to ``warning=10%``, ``critical=25%`` of baseline.
        """
        if not readings:
            return 1.0

        if thresholds is None:
            abs_bl = abs(baseline) if baseline != 0 else 1.0
            thresholds = {"warning": 0.10 * abs_bl, "critical": 0.25 * abs_bl}

        warn = thresholds.get("warning", 0.10 * abs(baseline))
        crit = thresholds.get("critical", 0.25 * abs(baseline))

        total_score = 0.0
        for r in readings:
            dev = abs(r.value - baseline)
            if dev <= warn:
                s = 1.0
            elif dev <= crit:
                s = 1.0 - 0.5 * ((dev - warn) / (crit - warn))
            else:
                s = max(0.0, 0.5 - 0.5 * ((dev - crit) / (crit + 1e-9)))
            s *= r.quality
            total_score += s
        return max(0.0, min(1.0, total_score / len(readings)))

    def compute_subsystem_health(
        self, subsystem_readings: Dict[str, List[SensorReading]]
    ) -> Dict[str, float]:
        """Compute a per-subsystem health score.

        Each subsystem maps to a list of readings; we use the latest value
        as the baseline for that subsystem and score deviations from it
        across the remaining readings.
        """
        scores: Dict[str, float] = {}
        for sub_id, readings in subsystem_readings.items():
            if not readings:
                scores[sub_id] = 1.0
                continue
            # baseline = mean of all readings for simplicity
            baseline = sum(r.value for r in readings) / len(readings)
            scores[sub_id] = self.compute_health_score(readings, baseline)
        return scores

    def detect_degradation_trend(
        self, readings: List[SensorReading], window: int = 5
    ) -> Tuple[str, float]:
        """Detect degradation trend using a sliding-window linear fit.

        Returns
        -------
        (direction, slope) : (str, float)
            direction is ``"improving"``, ``"degrading"``, or ``"stable"``.
            slope is the fitted slope of value over normalised time.
        """
        if len(readings) < 2:
            return "stable", 0.0

        values = [r.value for r in readings[-window:]]
        times = list(range(len(values)))

        n = len(values)
        sx = sum(times)
        sy = sum(values)
        sxx = sum(t * t for t in times)
        sxy = sum(t * v for t, v in zip(times, values))
        denom = n * sxx - sx * sx
        if denom == 0:
            return "stable", 0.0

        slope = (n * sxy - sx * sy) / denom

        # Sensitivity: 0.1 % of mean value
        mean_v = abs(sy / n) if n else 1.0
        threshold = 0.001 * mean_v if mean_v > 0 else 0.01

        if slope < -threshold:
            direction = "degrading"
        elif slope > threshold:
            direction = "improving"
        else:
            direction = "stable"
        return direction, slope

    def aggregate_subsystems(
        self,
        subsystem_scores: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Weighted average of subsystem scores.

        If *weights* is ``None`` every subsystem contributes equally.
        """
        if not subsystem_scores:
            return 1.0

        if weights is None:
            return sum(subsystem_scores.values()) / len(subsystem_scores)

        total_w = 0.0
        weighted_sum = 0.0
        for sub_id, score in subsystem_scores.items():
            w = weights.get(sub_id, 0.0)
            weighted_sum += w * score
            total_w += w
        if total_w == 0:
            return sum(subsystem_scores.values()) / len(subsystem_scores)
        return weighted_sum / total_w

    def generate_health_report(self, equipment_health: EquipmentHealth) -> Dict:
        """Return a structured health report dictionary."""
        status = "healthy"
        if equipment_health.overall_score < 0.5:
            status = "critical"
        elif equipment_health.overall_score < 0.75:
            status = "warning"

        return {
            "equipment_id": equipment_health.equipment_id,
            "overall_score": round(equipment_health.overall_score, 4),
            "status": status,
            "subsystem_scores": {
                k: round(v, 4) for k, v in equipment_health.subsystem_scores.items()
            },
            "trend": equipment_health.trend,
            "last_updated": equipment_health.last_updated,
            "recommendations": self._recommendations(equipment_health),
        }

    # ── private helpers ─────────────────────────────────────────

    @staticmethod
    def _recommendations(eh: EquipmentHealth) -> List[str]:
        recs: List[str] = []
        if eh.overall_score < 0.5:
            recs.append("Immediate maintenance required.")
        elif eh.overall_score < 0.75:
            recs.append("Schedule preventive maintenance within 48 hours.")
        if eh.trend == "degrading":
            recs.append("Equipment health is declining — investigate root cause.")
        for sub, score in eh.subsystem_scores.items():
            if score < 0.6:
                recs.append(f"Subsystem '{sub}' requires urgent attention.")
        return recs
