"""
Degradation curve modeling for marine equipment.

Fit linear, exponential, and Weibull degradation curves;
select the best model; predict future health; compute residuals.
Pure Python – no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DegradationModel:
    """Fitted degradation model descriptor."""
    model_type: str = ""
    parameters: Dict[str, float] = field(default_factory=dict)
    fit_quality: float = 0.0  # R²


@dataclass
class DegradationCurve:
    """Full degradation curve with time points, values, and fitted model."""
    time_points: List[float] = field(default_factory=list)
    health_values: List[float] = field(default_factory=list)
    model: Optional[DegradationModel] = None


class DegradationModeler:
    """Fit, select, and extrapolate degradation models."""

    # ── public API ──────────────────────────────────────────────

    def fit_linear(
        self,
        time_points: List[float],
        health_values: List[float],
    ) -> DegradationCurve:
        """Fit a linear degradation model  h(t) = a + b·t."""
        model = self._ols(time_points, health_values)
        model.model_type = "linear"
        return DegradationCurve(
            time_points=list(time_points),
            health_values=list(health_values),
            model=model,
        )

    def fit_exponential(
        self,
        time_points: List[float],
        health_values: List[float],
    ) -> DegradationCurve:
        """Fit  h(t) = a · exp(b · t)  via log-transform."""
        if not health_values or any(v <= 0 for v in health_values):
            return DegradationCurve(
                time_points=list(time_points),
                health_values=list(health_values),
                model=DegradationModel(model_type="exponential", fit_quality=0.0),
            )

        log_vals = [math.log(v) for v in health_values]
        model = self._ols(time_points, log_vals)

        # Transform back
        a = math.exp(model.parameters.get("intercept", 0))
        b = model.parameters.get("slope", 0)
        model.parameters = {"a": a, "b": b}
        model.model_type = "exponential"

        # Re-compute R² in original space
        predicted = [a * math.exp(b * t) for t in time_points]
        model.fit_quality = self._r_squared(health_values, predicted)

        return DegradationCurve(
            time_points=list(time_points),
            health_values=list(health_values),
            model=model,
        )

    def fit_weibull(
        self,
        time_points: List[float],
        health_values: List[float],
    ) -> DegradationCurve:
        """Approximate Weibull-CDF degradation  h(t) = exp(-(t/λ)^k).

        Uses a grid search over shape *k* and computes optimal *λ* analytically.
        """
        if len(time_points) < 2:
            return DegradationCurve(
                time_points=list(time_points),
                health_values=list(health_values),
                model=DegradationModel(model_type="weibull", fit_quality=0.0),
            )

        # Health values should be in (0, 1]; normalise if needed
        h_max = max(health_values)
        h_min = min(health_values)
        rng = h_max - h_min if h_max != h_min else 1.0
        normalised = [(v - h_min) / rng for v in health_values]

        best_k = 1.0
        best_r2 = -1e9
        best_lam = 1.0

        for k in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
            # Solve for lambda analytically: -ln(h) = (t/λ)^k
            # => t / λ = (-ln h)^(1/k) => λ = t / (-ln h)^(1/k)
            lam_sum = 0.0
            lam_count = 0
            for t, h in zip(time_points, normalised):
                if h <= 0 or h >= 1:
                    continue
                neg_ln_h = -math.log(h)
                if neg_ln_h <= 0:
                    continue
                val = neg_ln_h ** (1.0 / k)
                if val > 0:
                    lam_sum += t / val
                    lam_count += 1

            if lam_count == 0:
                continue
            lam = lam_sum / lam_count
            if lam <= 0:
                continue

            pred = [math.exp(-((t / lam) ** k)) for t in time_points]
            # Scale prediction to match range of original values
            pred_scaled = [p * rng + h_min for p in pred]
            r2 = self._r_squared(health_values, pred_scaled)

            if r2 > best_r2:
                best_r2 = r2
                best_k = k
                best_lam = lam

        return DegradationCurve(
            time_points=list(time_points),
            health_values=list(health_values),
            model=DegradationModel(
                model_type="weibull",
                parameters={"k": best_k, "lambda": best_lam},
                fit_quality=best_r2,
            ),
        )

    def select_best_model(self, curves: List[DegradationCurve]) -> DegradationCurve:
        """Return the curve with the highest R² fit quality."""
        if not curves:
            return DegradationCurve()
        return max(
            curves,
            key=lambda c: c.model.fit_quality if c.model else 0.0,
        )

    def predict_future(
        self,
        curve: DegradationCurve,
        future_points: List[float],
    ) -> List[float]:
        """Extrapolate health values at future time points."""
        if curve.model is None:
            return [0.0] * len(future_points)

        m = curve.model
        if m.model_type == "linear":
            a = m.parameters.get("intercept", 0)
            b = m.parameters.get("slope", 0)
            return [a + b * t for t in future_points]

        if m.model_type == "exponential":
            a = m.parameters.get("a", 1)
            b = m.parameters.get("b", 0)
            return [a * math.exp(b * t) for t in future_points]

        if m.model_type == "weibull":
            k = m.parameters.get("k", 1)
            lam = m.parameters.get("lambda", 1)
            if lam <= 0:
                return [0.0] * len(future_points)
            # We need to un-normalise: get the original range
            if curve.health_values:
                h_min = min(curve.health_values)
                h_max = max(curve.health_values)
                rng = h_max - h_min if h_max != h_min else 1.0
            else:
                h_min, rng = 0.0, 1.0
            raw = [math.exp(-((t / lam) ** k)) for t in future_points]
            return [r * rng + h_min for r in raw]

        return [0.0] * len(future_points)

    def compute_residuals(
        self,
        observed: List[float],
        predicted: List[float],
    ) -> List[float]:
        """Return per-point residuals (observed − predicted)."""
        if len(observed) != len(predicted):
            raise ValueError("observed and predicted must have the same length")
        return [o - p for o, p in zip(observed, predicted)]

    # ── private helpers ─────────────────────────────────────────

    @staticmethod
    def _ols(
        x: List[float], y: List[float]
    ) -> DegradationModel:
        """Ordinary least-squares linear fit. Returns model with slope, intercept, R²."""
        n = len(x)
        if n < 2:
            return DegradationModel(fit_quality=0.0)

        sx = sum(x)
        sy = sum(y)
        sxx = sum(xi * xi for xi in x)
        sxy = sum(xi * yi for xi, yi in zip(x, y))
        denom = n * sxx - sx * sx
        if denom == 0:
            return DegradationModel(fit_quality=0.0)

        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        predicted = [slope * xi + intercept for xi in x]
        r2 = DegradationModeler._r_squared(y, predicted)

        return DegradationModel(
            parameters={"slope": slope, "intercept": intercept},
            fit_quality=r2,
        )

    @staticmethod
    def _r_squared(observed: List[float], predicted: List[float]) -> float:
        """Coefficient of determination R²."""
        n = len(observed)
        if n == 0:
            return 0.0
        mean = sum(observed) / n
        ss_tot = sum((o - mean) ** 2 for o in observed)
        ss_res = sum((o - p) ** 2 for o, p in zip(observed, predicted))
        if ss_tot == 0:
            return 1.0
        return max(0.0, 1.0 - ss_res / ss_tot)
