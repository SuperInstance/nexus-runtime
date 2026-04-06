"""
Failure prediction via time-series analysis.

Linear regression, moving-average smoothing, anomaly detection,
multi-sensor fusion, and confidence-interval estimation.
Pure Python – no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TimeSeriesPoint:
    """Single time-series observation."""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of a failure-prediction analysis."""
    predicted_failure_time: Optional[float] = None
    confidence: float = 0.0
    contributing_factors: List[str] = field(default_factory=list)


class FailurePredictor:
    """Analyse sensor time-series to predict equipment failures."""

    # ── public API ──────────────────────────────────────────────

    def analyze_trend(
        self,
        readings: List[TimeSeriesPoint],
        failure_threshold: float = 0.0,
    ) -> PredictionResult:
        """Fit a linear model and predict when the value crosses *failure_threshold*.

        Uses ordinary least-squares regression on the readings.
        """
        if len(readings) < 2:
            return PredictionResult(confidence=0.0)

        n = len(readings)
        times = [r.timestamp for r in readings]
        values = [r.value for r in readings]
        t0 = times[0]

        sx = sum(times)
        sy = sum(values)
        sxx = sum(t * t for t in times)
        sxy = sum(t * v for t, v in zip(times, values))
        denom = n * sxx - sx * sx
        if denom == 0:
            return PredictionResult(confidence=0.0)

        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n

        # Extrapolate to failure threshold
        predicted_time: Optional[float] = None
        if slope != 0:
            predicted_time = (failure_threshold - intercept) / slope
            if predicted_time < t0:
                predicted_time = None  # failure already occurred or never

        # Confidence based on R²
        mean_y = sy / n
        ss_tot = sum((v - mean_y) ** 2 for v in values) or 1e-12
        ss_res = sum(
            (v - (slope * t + intercept)) ** 2 for t, v in zip(times, values)
        )
        r2 = 1.0 - ss_res / ss_tot
        confidence = max(0.0, min(1.0, r2))

        factors: List[str] = []
        if slope < 0:
            factors.append("declining_trend")
        if confidence > 0.8:
            factors.append("high_confidence_fit")

        return PredictionResult(
            predicted_failure_time=predicted_time,
            confidence=confidence,
            contributing_factors=factors,
        )

    def detect_anomalies(
        self,
        readings: List[TimeSeriesPoint],
        sensitivity: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Flag readings whose z-score exceeds *sensitivity* standard deviations.

        Returns a list of dicts with ``index``, ``timestamp``, ``value``,
        ``z_score``, ``mean``, ``std``.
        """
        if len(readings) < 2:
            return []

        values = [r.value for r in readings]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1e-9

        anomalies: List[Dict[str, Any]] = []
        for i, r in enumerate(readings):
            z = (r.value - mean) / std
            if abs(z) >= sensitivity:
                anomalies.append({
                    "index": i,
                    "timestamp": r.timestamp,
                    "value": r.value,
                    "z_score": z,
                    "mean": mean,
                    "std": std,
                })
        return anomalies

    def compute_failure_probability(
        self,
        readings: List[TimeSeriesPoint],
        model: str = "linear",
    ) -> float:
        """Return estimated probability of failure in the next observation window.

        Supported models: ``"linear"`` (trend-based), ``"threshold"`` (proximity).
        """
        if not readings:
            return 0.0

        if model == "threshold":
            # Simple proximity model: probability increases as values move
            # away from the historical mean by more than 2 standard deviations.
            values = [r.value for r in readings]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1e-9
            latest = readings[-1].value
            z = abs(latest - mean) / std
            # sigmoid mapping z -> (0,1)
            prob = 1.0 / (1.0 + math.exp(-z + 2.0))
            return max(0.0, min(1.0, prob))

        # Default: linear trend probability
        if len(readings) < 2:
            return 0.0
        res = self.analyze_trend(readings)
        # Map confidence to probability
        return res.confidence

    def multi_sensor_fusion(
        self,
        sensor_predictions: Dict[str, PredictionResult],
        weights: Optional[Dict[str, float]] = None,
    ) -> PredictionResult:
        """Combine predictions from multiple sensors using weighted fusion."""
        if not sensor_predictions:
            return PredictionResult(confidence=0.0)

        if weights is None:
            w_total = len(sensor_predictions)
            weights = {k: 1.0 for k in sensor_predictions}

        w_sum = sum(weights.get(s, 0.0) for s in sensor_predictions)
        if w_sum == 0:
            return PredictionResult(confidence=0.0)

        # Weighted average failure time
        ft_sum = 0.0
        ft_count = 0
        for sid, pred in sensor_predictions.items():
            if pred.predicted_failure_time is not None:
                ft_sum += weights.get(sid, 1.0) * pred.predicted_failure_time
                ft_count += weights.get(sid, 1.0)

        avg_ft = ft_sum / ft_count if ft_count > 0 else None

        # Weighted confidence
        avg_conf = sum(
            weights.get(s, 1.0) * p.confidence
            for s, p in sensor_predictions.items()
        ) / w_sum

        all_factors: List[str] = []
        for p in sensor_predictions.values():
            all_factors.extend(p.contributing_factors)

        return PredictionResult(
            predicted_failure_time=avg_ft,
            confidence=max(0.0, min(1.0, avg_conf)),
            contributing_factors=list(set(all_factors)),
        )

    def compute_confidence_interval(
        self,
        prediction: float,
        uncertainty: float,
        z_multiplier: float = 1.96,
    ) -> Tuple[float, float]:
        """Return ``(lower_bound, upper_bound)`` for a prediction."""
        return (
            prediction - z_multiplier * uncertainty,
            prediction + z_multiplier * uncertainty,
        )
