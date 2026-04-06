"""
Multi-objective optimization — weighted sum, epsilon-constraint, Pareto front.

Pure Python — math, dataclasses, enum, collections.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ObjectivePriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class Objective:
    name: str = ""
    weight: float = 1.0
    function: Optional[Callable[[List[float]], float]] = None
    priority: ObjectivePriority = ObjectivePriority.MEDIUM


@dataclass
class ParetoPoint:
    objectives: List[float] = field(default_factory=list)
    decision_variables: List[float] = field(default_factory=list)
    dominated: bool = False


# ---------------------------------------------------------------------------
# MultiObjectiveOptimizer
# ---------------------------------------------------------------------------

class MultiObjectiveOptimizer:
    """Multi-objective optimisation utilities."""

    def __init__(self):
        self._objectives: List[Objective] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_objective(
        self,
        name: str,
        fn: Callable[[List[float]], float],
        weight: float = 1.0,
        priority: ObjectivePriority = ObjectivePriority.MEDIUM,
    ) -> None:
        self._objectives.append(Objective(name=name, weight=weight,
                                          function=fn, priority=priority))

    @property
    def objectives(self) -> List[Objective]:
        return list(self._objectives)

    def scalarize_weighted_sum(
        self,
        objectives: List[float],
        weights: List[float],
    ) -> float:
        """Weighted-sum scalarisation of objective values."""
        n = min(len(objectives), len(weights))
        return sum(weights[i] * objectives[i] for i in range(n))

    def scalarize_epsilon_constraint(
        self,
        primary: float,
        primary_weight: float,
        others: List[float],
        epsilon: List[float],
        penalty: float = 1e6,
    ) -> float:
        """
        Epsilon-constraint scalarisation.
        Penalise violation of secondary objectives beyond their epsilon limits.
        """
        cost = primary_weight * primary
        for i, val in enumerate(others):
            eps = epsilon[i] if i < len(epsilon) else float("inf")
            if val > eps:
                cost += penalty * (val - eps)
        return cost

    def find_pareto_front(
        self, solutions: List[ParetoPoint]
    ) -> List[ParetoPoint]:
        """Return the non-dominated subset of *solutions*."""
        n = len(solutions)
        for i in range(n):
            solutions[i].dominated = False
            for j in range(n):
                if i == j:
                    continue
                if self.dominates(solutions[j], solutions[i]):
                    solutions[i].dominated = True
                    break
        return [s for s in solutions if not s.dominated]

    def dominates(self, a: ParetoPoint, b: ParetoPoint) -> bool:
        """True if *a* weakly dominates *b* (and is not identical)."""
        if len(a.objectives) != len(b.objectives):
            return False
        at_least_one_better = False
        for oa, ob in zip(a.objectives, b.objectives):
            if oa > ob:  # minimisation: smaller is better
                return False
            if oa < ob:
                at_least_one_better = True
        return at_least_one_better

    def select_knee_point(
        self, pareto_front: List[ParetoPoint]
    ) -> Optional[ParetoPoint]:
        """Select the knee / compromise point from the Pareto front."""
        if len(pareto_front) < 2:
            return pareto_front[0] if pareto_front else None

        n_obj = len(pareto_front[0].objectives)
        # Normalise objectives to [0, 1]
        mins = [float("inf")] * n_obj
        maxs = [float("-inf")] * n_obj
        for pt in pareto_front:
            for k in range(n_obj):
                mins[k] = min(mins[k], pt.objectives[k])
                maxs[k] = max(maxs[k], pt.objectives[k])

        def _norm(objs: List[float]) -> List[float]:
            return [
                (objs[k] - mins[k]) / (maxs[k] - mins[k] + 1e-12)
                for k in range(n_obj)
            ]

        best = None
        best_dist = -1.0
        for pt in pareto_front:
            nrm = _norm(pt.objectives)
            # Distance from ideal (0,...,0)
            dist = math.sqrt(sum(v * v for v in nrm))
            # Distance from line connecting extremes
            avg = sum(nrm) / len(nrm)
            perp_dist = abs(sum((v - avg) ** 2 for v in nrm))
            if perp_dist > best_dist:
                best_dist = perp_dist
                best = pt
        return best

    def adaptive_weights(
        self,
        performance_history: List[List[float]],
    ) -> List[float]:
        """Adapt weights based on recent performance (increase weight on worst)."""
        if not performance_history:
            return [1.0] * len(self._objectives)

        n_obj = len(performance_history[0])
        n = len(performance_history)
        avg_errors = [0.0] * n_obj
        for hist in performance_history:
            for k in range(min(n_obj, len(hist))):
                avg_errors[k] += hist[k]
        for k in range(n_obj):
            avg_errors[k] /= max(n, 1)

        # Normalise
        total = sum(avg_errors) + 1e-12
        new_weights = [e / total for e in avg_errors]
        # Scale so they sum to n_obj (keep average ~ 1)
        scale = n_obj / (sum(new_weights) + 1e-12)
        return [w * scale for w in new_weights]
