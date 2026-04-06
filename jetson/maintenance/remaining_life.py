"""
Remaining Useful Life (RUL) estimation.

Linear, exponential, Weibull, and ensemble estimation methods.
Pure Python – no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RULEstimate:
    """Remaining Useful Life estimate."""
    median: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    confidence: float = 0.0  # 0-1
    method: str = ""


class RemainingLifeEstimator:
    """Estimate how long a component will function before failure."""

    # ── public API ──────────────────────────────────────────────

    def estimate_linear(
        self,
        current_health: float,
        degradation_rate: float,
        min_threshold: float,
    ) -> RULEstimate:
        """RUL assuming constant (linear) degradation.

        Parameters
        ----------
        current_health : float
            Current health value (higher is better).
        degradation_rate : float
            Health units lost per time unit (must be > 0).
        min_threshold : float
            Health value at which failure occurs.
        """
        if degradation_rate <= 0:
            return RULEstimate(
                median=float("inf"), lower_bound=float("inf"),
                upper_bound=float("inf"), confidence=1.0, method="linear",
            )
        if current_health <= min_threshold:
            return RULEstimate(
                median=0.0, lower_bound=0.0, upper_bound=0.0,
                confidence=1.0, method="linear",
            )
        rul = (current_health - min_threshold) / degradation_rate
        spread = 0.2 * rul  # ±20 %
        return RULEstimate(
            median=rul,
            lower_bound=max(0.0, rul - spread),
            upper_bound=rul + spread,
            confidence=0.7,
            method="linear",
        )

    def estimate_exponential(
        self,
        current_health: float,
        degradation_rate: float,
        threshold: float,
    ) -> RULEstimate:
        """RUL assuming exponential degradation.

        Health follows  h(t) = current_health * exp(-degradation_rate * t).
        Solve for t when h(t) = threshold.
        """
        if degradation_rate <= 0:
            return RULEstimate(
                median=float("inf"), lower_bound=float("inf"),
                upper_bound=float("inf"), confidence=1.0, method="exponential",
            )
        if current_health <= threshold or threshold <= 0:
            return RULEstimate(
                median=0.0, lower_bound=0.0, upper_bound=0.0,
                confidence=1.0, method="exponential",
            )
        ratio = current_health / threshold
        if ratio <= 1.0:
            return RULEstimate(
                median=0.0, lower_bound=0.0, upper_bound=0.0,
                confidence=1.0, method="exponential",
            )
        rul = math.log(ratio) / degradation_rate
        spread = 0.25 * rul
        return RULEstimate(
            median=rul,
            lower_bound=max(0.0, rul - spread),
            upper_bound=rul + spread,
            confidence=0.75,
            method="exponential",
        )

    def estimate_weibull(
        self,
        shape: float,
        scale: float,
        current_age: float,
    ) -> RULEstimate:
        """RUL using a Weibull distribution model.

        Uses the conditional survival:  RUL = scale * ((current_age/scale)^shape + 1)^(1/shape) - current_age

        Simplified: median remaining life assuming Weibull CDF.
        """
        if shape <= 0 or scale <= 0:
            return RULEstimate(
                median=0.0, lower_bound=0.0, upper_bound=0.0,
                confidence=0.0, method="weibull",
            )

        # Conditional mean residual life (approximate)
        # E[T - t | T > t] for Weibull
        gamma = math.gamma(1.0 + 1.0 / shape)
        mean_life = scale * gamma
        rul = max(0.0, mean_life - current_age)
        spread = scale * 0.3

        return RULEstimate(
            median=rul,
            lower_bound=max(0.0, rul - spread),
            upper_bound=rul + spread,
            confidence=0.65,
            method="weibull",
        )

    def ensemble_estimate(
        self,
        estimates: List[RULEstimate],
        weights: Optional[List[float]] = None,
    ) -> RULEstimate:
        """Weighted combination of multiple RUL estimates."""
        if not estimates:
            return RULEstimate(median=0.0, confidence=0.0, method="ensemble")

        if weights is None:
            weights = [1.0] * len(estimates)

        total_w = sum(weights)
        if total_w == 0:
            return RULEstimate(median=0.0, confidence=0.0, method="ensemble")

        median = sum(w * e.median for w, e in zip(weights, estimates)) / total_w
        lower = sum(w * e.lower_bound for w, e in zip(weights, estimates)) / total_w
        upper = sum(w * e.upper_bound for w, e in zip(weights, estimates)) / total_w
        conf = sum(w * e.confidence for w, e in zip(weights, estimates)) / total_w

        methods = ", ".join(e.method for e in estimates if e.method)

        return RULEstimate(
            median=median,
            lower_bound=max(0.0, lower),
            upper_bound=upper,
            confidence=min(1.0, conf),
            method=f"ensemble({methods})",
        )

    def update_with_reading(
        self,
        rul_estimate: RULEstimate,
        new_reading: float,
        expected_reading: float,
    ) -> RULEstimate:
        """Update a RUL estimate based on a new sensor reading.

        If the reading is worse than expected, RUL is reduced proportionally.
        """
        if expected_reading == 0:
            return rul_estimate

        ratio = new_reading / expected_reading
        adjustment = min(1.0, max(0.0, ratio))
        new_median = rul_estimate.median * adjustment
        spread = (rul_estimate.upper_bound - rul_estimate.lower_bound) / 2
        return RULEstimate(
            median=max(0.0, new_median),
            lower_bound=max(0.0, new_median - spread),
            upper_bound=new_median + spread,
            confidence=min(1.0, rul_estimate.confidence * 1.02),
            method=rul_estimate.method,
        )
