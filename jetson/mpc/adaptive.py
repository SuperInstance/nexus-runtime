"""
Adaptive horizon tuning and parameter adaptation for MPC.

Pure Python — math, dataclasses, random, collections.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HorizonConfig:
    min_horizon: int = 5
    max_horizon: int = 50
    adaptation_rate: float = 0.1
    performance_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    tracking_error: float = 0.0
    constraint_violations: float = 0.0
    computation_time: float = 0.0
    cost: float = 0.0


@dataclass
class AdaptedParameters:
    horizon: int = 10
    dt: float = 0.1
    weights: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AdaptiveController
# ---------------------------------------------------------------------------

class AdaptiveController:
    """Adapt MPC parameters based on online performance metrics."""

    def __init__(self):
        self._history: List[PerformanceMetrics] = []
        self._current_horizon: int = 10
        self._current_weights: List[float] = []
        self._current_dt: float = 0.1

    @property
    def history(self) -> List[PerformanceMetrics]:
        return list(self._history)

    @property
    def current_horizon(self) -> int:
        return self._current_horizon

    @property
    def current_dt(self) -> float:
        return self._current_dt

    @property
    def current_weights(self) -> List[float]:
        return list(self._current_weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_horizon(
        self,
        performance: PerformanceMetrics,
        config: HorizonConfig,
    ) -> int:
        """Select prediction horizon based on current performance."""
        self._history.append(performance)

        error = performance.tracking_error
        violations = performance.constraint_violations
        comp_time = performance.computation_time

        thresholds = config.performance_thresholds
        err_thresh = thresholds.get("tracking_error", 1.0)
        viol_thresh = thresholds.get("constraint_violations", 0.1)
        time_thresh = thresholds.get("computation_time", 0.05)

        # Increase horizon if tracking is poor or constraints are violated
        if error > err_thresh or violations > viol_thresh:
            new_horizon = self._current_horizon + max(1, int(config.adaptation_rate * 10))
        elif comp_time > time_thresh:
            # Decrease if computation is too slow
            new_horizon = self._current_horizon - max(1, int(config.adaptation_rate * 5))
        else:
            # Slight increase for improving accuracy
            new_horizon = self._current_horizon + 1

        new_horizon = max(config.min_horizon, min(config.max_horizon, new_horizon))
        self._current_horizon = new_horizon
        return new_horizon

    def adapt_weights(
        self,
        performance: PerformanceMetrics,
        current_weights: List[float],
    ) -> List[float]:
        """Adjust weights based on performance — increase weight on high-error states."""
        if not current_weights:
            return current_weights
        weights = list(current_weights)
        n = len(weights)
        # Simple adaptation: scale weights by tracking error
        error_scale = 1.0 + performance.tracking_error
        violation_scale = 1.0 + performance.constraint_violations
        for i in range(n):
            # Apply more weight to earlier states (position vs velocity)
            factor = error_scale if i < n // 2 else violation_scale
            weights[i] *= factor
        # Normalise so average weight stays ~1
        avg = sum(weights) / n if n > 0 else 1.0
        if avg > 0:
            weights = [w / avg for w in weights]
        self._current_weights = weights
        return weights

    def should_replan(
        self,
        performance: PerformanceMetrics,
        thresholds: Dict[str, float],
    ) -> bool:
        """Decide whether the MPC should re-plan based on performance."""
        t = thresholds
        if performance.tracking_error > t.get("tracking_error", 1.0):
            return True
        if performance.constraint_violations > t.get("constraint_violations", 0.1):
            return True
        if performance.cost > t.get("cost", 100.0):
            return True
        return False

    def compute_optimal_dt(
        self,
        state_complexity: float,
        horizon: int,
    ) -> float:
        """Compute time step given state complexity and horizon."""
        # Higher complexity → smaller dt for accuracy
        # Larger horizon → larger dt for computation budget
        base_dt = 0.1
        complexity_factor = max(0.01, 1.0 / (1.0 + state_complexity))
        horizon_factor = min(1.0, horizon / 20.0)
        dt = base_dt * complexity_factor * (0.5 + 0.5 * horizon_factor)
        dt = max(0.01, min(1.0, dt))
        self._current_dt = dt
        return dt

    def tune_parameters(
        self,
        performance_history: List[PerformanceMetrics],
        parameter_bounds: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        """
        Simple random-search parameter tuning over history.
        Returns a dict of tuned parameter values.
        """
        if not performance_history:
            return {k: (lo + hi) / 2 for k, (lo, hi) in parameter_bounds.items()}

        avg_error = sum(p.tracking_error for p in performance_history) / len(performance_history)
        avg_viol = sum(p.constraint_violations for p in performance_history) / len(performance_history)
        avg_time = sum(p.computation_time for p in performance_history) / len(performance_history)

        tuned: Dict[str, float] = {}
        for name, (lo, hi) in parameter_bounds.items():
            if name == "horizon":
                # Larger horizon if error is high
                val = lo + (hi - lo) * min(1.0, avg_error / 2.0)
                tuned[name] = round(val)
            elif name == "dt":
                # Smaller dt if violation is high
                val = lo + (hi - lo) * max(0.0, 1.0 - avg_viol)
                tuned[name] = round(val, 4)
            elif name == "weight_scale":
                val = lo + (hi - lo) * min(1.0, avg_error)
                tuned[name] = round(val, 4)
            else:
                tuned[name] = (lo + hi) / 2.0

        # Clamp to bounds
        for name in tuned:
            lo, hi = parameter_bounds[name]
            tuned[name] = max(lo, min(hi, tuned[name]))

        return tuned
