"""Multi-objective optimization: Pareto fronts, scalarization, NSGA-II selection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Objective:
    """A single optimization objective."""
    name: str
    optimize: str  # "min" or "max"
    weight: float = 1.0
    function: Optional[Callable[[Any], float]] = None


@dataclass
class ParetoFront:
    """Represents the set of non-dominated solutions."""
    solutions: List[Any] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)
    dominated_count: int = 0


class MultiObjectiveOptimizer:
    """Core multi-objective optimization engine."""

    def evaluate(
        self,
        solution: Any,
        objectives: List[Objective],
    ) -> List[float]:
        """Evaluate a solution against all objectives, returning raw values."""
        values = []
        for obj in objectives:
            if obj.function is not None:
                val = obj.function(solution)
            else:
                val = 0.0
            values.append(float(val))
        return values

    def dominates(
        self,
        solution_a: List[float],
        solution_b: List[float],
        directions: Optional[List[str]] = None,
    ) -> bool:
        """Check if solution_a Pareto-dominates solution_b.

        For "min" directions, lower is better.  For "max", higher is better.
        If *directions* is ``None`` all objectives are minimised.
        """
        if len(solution_a) != len(solution_b):
            return False
        if directions is None:
            directions = ["min"] * len(solution_a)
        if len(directions) != len(solution_a):
            return False

        at_least_as_good = True
        strictly_better = False
        for av, bv, d in zip(solution_a, solution_b, directions):
            if d == "min":
                if av > bv:
                    at_least_as_good = False
                    break
                if av < bv:
                    strictly_better = True
            else:  # max
                if av < bv:
                    at_least_as_good = False
                    break
                if av > bv:
                    strictly_better = True
        return at_least_as_good and strictly_better

    def compute_pareto_front(
        self,
        solutions: List[Any],
        objectives: List[Objective],
    ) -> ParetoFront:
        """Compute the Pareto-optimal front from a set of solutions."""
        # Evaluate all solutions
        all_values: List[List[float]] = []
        for sol in solutions:
            vals = self.evaluate(sol, objectives)
            all_values.append(vals)

        directions = [obj.optimize for obj in objectives]
        obj_names = [obj.name for obj in objectives]

        # Find non-dominated solutions
        non_dominated_indices: List[int] = []
        dominated_count = 0
        for i, vals_i in enumerate(all_values):
            is_dominated = False
            for j, vals_j in enumerate(all_values):
                if i == j:
                    continue
                if self.dominates(vals_j, vals_i, directions):
                    is_dominated = True
                    break
            if is_dominated:
                dominated_count += 1
            else:
                non_dominated_indices.append(i)

        pareto_solutions = [solutions[i] for i in non_dominated_indices]
        return ParetoFront(
            solutions=pareto_solutions,
            objectives=obj_names,
            dominated_count=dominated_count,
        )

    def find_knee_point(
        self,
        pareto_front: ParetoFront,
        objectives: List[Objective],
        solutions: List[Any],
    ) -> Any:
        """Find the knee point (best compromise) on the Pareto front.

        Uses the perpendicular-distance method: for each Pareto point,
        compute the normalised distance to the line connecting the two
        extreme points.  The point with maximum distance is the knee.
        """
        if not pareto_front.solutions:
            return None

        # Evaluate all pareto solutions
        pareto_values: List[List[float]] = []
        for sol in pareto_front.solutions:
            pareto_values.append(self.evaluate(sol, objectives))

        # Normalize each objective to [0, 1]
        n_obj = len(objectives)
        if n_obj == 0:
            return pareto_front.solutions[0]

        normalized: List[List[float]] = []
        for obj_idx in range(n_obj):
            col = [pv[obj_idx] for pv in pareto_values]
            min_val = min(col)
            max_val = max(col)
            rng = max_val - min_val if max_val != min_val else 1.0
            for row_idx, pv in enumerate(pareto_values):
                nv = (pv[obj_idx] - min_val) / rng
                if row_idx >= len(normalized):
                    normalized.append([])
                normalized[row_idx].append(nv)

        # For two objectives use perpendicular-distance method
        if n_obj == 2:
            # Sort by first objective (assuming first is min)
            pairs = list(enumerate(normalized))
            pairs.sort(key=lambda x: x[1][0])
            first = pairs[0][1]
            last = pairs[-1][1]
            dx = last[0] - first[0]
            dy = last[1] - first[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length == 0:
                return pareto_front.solutions[pairs[0][0]]

            best_dist = -1.0
            best_idx = pairs[0][0]
            for orig_idx, norm_vals in pairs:
                # Distance from point to line
                dist = abs(dy * norm_vals[0] - dx * norm_vals[1]
                           + last[0] * first[1] - last[1] * first[0]) / length
                if dist > best_dist:
                    best_dist = dist
                    best_idx = orig_idx
            return pareto_front.solutions[best_idx]

        # For more objectives, use minimum-angle heuristic:
        # pick the solution closest to the ideal point
        ideal = [min(nv[i] for nv in normalized) for i in range(n_obj)]
        best_dist = float("inf")
        best_idx = 0
        for idx, nv in enumerate(normalized):
            d = sum((nv[j] - ideal[j]) ** 2 for j in range(n_obj))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return pareto_front.solutions[best_idx]

    def scalarize(
        self,
        objective_values: List[float],
        weights: Optional[List[float]] = None,
        method: str = "weighted_sum",
    ) -> float:
        """Convert multi-objective values to a single score.

        Methods:
          - ``weighted_sum``:  Σ w_i * v_i
          - ``weighted_chebyshev``:  max_i (w_i * v_i)  (after normalisation)
        """
        if weights is None:
            weights = [1.0 / len(objective_values)] * len(objective_values)
        if len(weights) != len(objective_values):
            raise ValueError("Weights length must match objectives length")

        if method == "weighted_sum":
            return sum(w * v for w, v in zip(weights, objective_values))

        if method == "weighted_chebyshev":
            # Normalize values for Chebyshev
            min_vals = [float(v) for v in objective_values]
            max_vals = [float(v) for v in objective_values]
            rng = [
                max_vals[i] - min_vals[i] if max_vals[i] != min_vals[i] else 1.0
                for i in range(len(objective_values))
            ]
            normed = [(objective_values[i] - min_vals[i]) / rng[i]
                      for i in range(len(objective_values))]
            return max(w * v for w, v in zip(weights, normed))

        raise ValueError(f"Unknown scalarization method: {method}")

    # ------------------------------------------------------------------
    # Simplified NSGA-II non-dominated sorting & crowding selection
    # ------------------------------------------------------------------

    def nsga_ii_select(
        self,
        population: List[Any],
        objectives: List[Objective],
        population_size: int,
    ) -> List[Any]:
        """Simplified NSGA-II: non-dominated sorting + crowding distance.

        Returns *population_size* solutions from *population* using
        fast-non-dominated-sort and crowding-distance tie-breaking.
        """
        directions = [obj.optimize for obj in objectives]
        all_values: List[List[float]] = [
            self.evaluate(sol, objectives) for sol in population
        ]

        # Fast non-dominated sort
        fronts: List[List[int]] = self._fast_non_dominated_sort(
            all_values, directions
        )

        selected_indices: List[int] = []
        for front in fronts:
            if len(selected_indices) + len(front) <= population_size:
                selected_indices.extend(front)
            else:
                remaining = population_size - len(selected_indices)
                if remaining > 0:
                    crowding = self._crowding_distance(front, all_values)
                    # Sort by crowding descending
                    front_sorted = sorted(
                        front, key=lambda i: crowding.get(i, 0.0), reverse=True
                    )
                    selected_indices.extend(front_sorted[:remaining])
                break

        return [population[i] for i in selected_indices[:population_size]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_non_dominated_sort(
        values: List[List[float]],
        directions: List[str],
    ) -> List[List[int]]:
        """Fast non-dominated sorting returning list of fronts."""
        n = len(values)
        domination_count: List[int] = [0] * n
        dominated_set: List[List[int]] = [[] for _ in range(n)]
        rank: List[int] = [0] * n

        for i in range(n):
            for j in range(i + 1, n):
                if MultiObjectiveOptimizer._dominates_static(
                    values[i], values[j], directions
                ):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif MultiObjectiveOptimizer._dominates_static(
                    values[j], values[i], directions
                ):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        fronts: List[List[int]] = []
        current_front: List[int] = [
            i for i in range(n) if domination_count[i] == 0
        ]
        front_idx = 0
        while current_front:
            fronts.append(current_front)
            next_front: List[int] = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
                        rank[j] = front_idx + 1
            front_idx += 1
            current_front = next_front
        return fronts

    @staticmethod
    def _dominates_static(
        a: List[float], b: List[float], directions: List[str],
    ) -> bool:
        """Static helper for domination check."""
        at_least_as_good = True
        strictly_better = False
        for av, bv, d in zip(a, b, directions):
            if d == "min":
                if av > bv:
                    at_least_as_good = False
                    break
                if av < bv:
                    strictly_better = True
            else:
                if av < bv:
                    at_least_as_good = False
                    break
                if av > bv:
                    strictly_better = True
        return at_least_as_good and strictly_better

    @staticmethod
    def _crowding_distance(
        front: List[int],
        all_values: List[List[float]],
    ) -> Dict[int, float]:
        """Compute crowding distance for solutions in *front*."""
        n_obj = len(all_values[0]) if all_values else 0
        distances: Dict[int, float] = {i: 0.0 for i in front}
        if len(front) <= 2 or n_obj == 0:
            # Boundary points get infinite distance
            for i in front:
                distances[i] = float("inf")
            return distances

        for m in range(n_obj):
            sorted_front = sorted(front, key=lambda i: all_values[i][m])
            distances[sorted_front[0]] = float("inf")
            distances[sorted_front[-1]] = float("inf")
            min_val = all_values[sorted_front[0]][m]
            max_val = all_values[sorted_front[-1]][m]
            rng = max_val - min_val if max_val != min_val else 1.0
            for k in range(1, len(sorted_front) - 1):
                distances[sorted_front[k]] += (
                    all_values[sorted_front[k + 1]][m]
                    - all_values[sorted_front[k - 1]][m]
                ) / rng
        return distances
