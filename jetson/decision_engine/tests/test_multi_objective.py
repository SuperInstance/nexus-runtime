"""Tests for multi_objective.py — MultiObjectiveOptimizer, Objective, ParetoFront."""

import math
import pytest

from jetson.decision_engine.multi_objective import (
    Objective, ParetoFront, MultiObjectiveOptimizer,
)


# ============================================================
# Objective dataclass
# ============================================================

class TestObjective:
    def test_objective_defaults(self):
        obj = Objective(name="cost", optimize="min")
        assert obj.name == "cost"
        assert obj.optimize == "min"
        assert obj.weight == 1.0
        assert obj.function is None

    def test_objective_with_function(self):
        fn = lambda x: x["cost"] * 2
        obj = Objective(name="cost", optimize="min", function=fn)
        assert obj.function is not None
        assert obj.function({"cost": 5}) == 10

    def test_objective_weight(self):
        obj = Objective(name="quality", optimize="max", weight=2.5)
        assert obj.weight == pytest.approx(2.5)


# ============================================================
# ParetoFront dataclass
# ============================================================

class TestParetoFront:
    def test_pareto_front_defaults(self):
        pf = ParetoFront()
        assert pf.solutions == []
        assert pf.objectives == []
        assert pf.dominated_count == 0

    def test_pareto_front_with_data(self):
        pf = ParetoFront(
            solutions=["a", "b"],
            objectives=["cost", "quality"],
            dominated_count=3,
        )
        assert len(pf.solutions) == 2
        assert pf.dominated_count == 3


# ============================================================
# MultiObjectiveOptimizer.evaluate
# ============================================================

class TestEvaluate:
    def setup_method(self):
        self.opt = MultiObjectiveOptimizer()
        self.objs = [
            Objective(name="cost", optimize="min", function=lambda s: s["cost"]),
            Objective(name="quality", optimize="max", function=lambda s: s["quality"]),
        ]

    def test_evaluate_basic(self):
        sol = {"cost": 10, "quality": 0.8}
        vals = self.opt.evaluate(sol, self.objs)
        assert len(vals) == 2
        assert vals[0] == 10.0
        assert vals[1] == 0.8

    def test_evaluate_no_function(self):
        obj_no_fn = Objective(name="dummy", optimize="min")
        vals = self.opt.evaluate({"x": 1}, [obj_no_fn])
        assert vals == [0.0]

    def test_evaluate_mixed(self):
        objs = [
            Objective(name="a", optimize="min", function=lambda s: s["a"]),
            Objective(name="b", optimize="min"),  # no function
            Objective(name="c", optimize="max", function=lambda s: s["c"]),
        ]
        sol = {"a": 5, "c": 3}
        vals = self.opt.evaluate(sol, objs)
        assert vals == [5.0, 0.0, 3.0]

    def test_evaluate_empty_objectives(self):
        vals = self.opt.evaluate({"x": 1}, [])
        assert vals == []

    def test_evaluate_single_objective(self):
        objs = [Objective(name="y", optimize="min", function=lambda s: s["y"])]
        vals = self.opt.evaluate({"y": 42}, objs)
        assert vals == [42.0]


# ============================================================
# MultiObjectiveOptimizer.dominates
# ============================================================

class TestDominates:
    def setup_method(self):
        self.opt = MultiObjectiveOptimizer()

    def test_dominates_true_min(self):
        a = [1.0, 3.0]
        b = [2.0, 4.0]
        assert self.opt.dominates(a, b) is True

    def test_dominates_false(self):
        a = [2.0, 3.0]
        b = [1.0, 4.0]
        assert self.opt.dominates(a, b) is False

    def test_dominates_equal(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0]
        assert self.opt.dominates(a, b) is False

    def test_dominates_max_direction(self):
        a = [5.0, 5.0]
        b = [4.0, 4.0]
        assert self.opt.dominates(a, b, ["max", "max"]) is True

    def test_dominates_mixed_directions(self):
        # a is better on cost (min), b is better on quality (max)
        a = [1.0, 2.0]
        b = [2.0, 3.0]
        assert self.opt.dominates(a, b, ["min", "max"]) is False
        assert self.opt.dominates(a, b, ["min", "min"]) is True

    def test_dominates_length_mismatch(self):
        assert self.opt.dominates([1, 2], [1]) is False

    def test_dominates_directions_length_mismatch(self):
        assert self.opt.dominates([1, 2], [1, 2], ["min"]) is False

    def test_dominates_partial_better(self):
        a = [1.0, 5.0]
        b = [2.0, 5.0]
        assert self.opt.dominates(a, b) is True


# ============================================================
# MultiObjectiveOptimizer.compute_pareto_front
# ============================================================

class TestComputeParetoFront:
    def setup_method(self):
        self.opt = MultiObjectiveOptimizer()

    def test_pareto_two_objectives(self):
        objs = [
            Objective(name="cost", optimize="min", function=lambda s: s[0]),
            Objective(name="quality", optimize="max", function=lambda s: s[1]),
        ]
        solutions = [
            (1, 5), (2, 8), (3, 10), (5, 3),
        ]
        pf = self.opt.compute_pareto_front(solutions, objs)
        assert (1, 5) in pf.solutions
        assert (2, 8) in pf.solutions
        assert (3, 10) in pf.solutions
        assert (5, 3) not in pf.solutions  # dominated by (1, 5)

    def test_pareto_all_nondominated(self):
        objs = [
            Objective(name="x", optimize="min", function=lambda s: s[0]),
            Objective(name="y", optimize="max", function=lambda s: s[1]),
        ]
        # Each solution is best on one objective and worst on the other
        solutions = [(1, 1), (2, 2), (3, 3)]
        pf = self.opt.compute_pareto_front(solutions, objs)
        assert len(pf.solutions) == 3

    def test_pareto_empty_solutions(self):
        objs = [Objective(name="x", optimize="min", function=lambda s: s)]
        pf = self.opt.compute_pareto_front([], objs)
        assert pf.solutions == []
        assert pf.dominated_count == 0

    def test_pareto_single_solution(self):
        objs = [Objective(name="x", optimize="min", function=lambda s: s)]
        pf = self.opt.compute_pareto_front([5], objs)
        assert len(pf.solutions) == 1

    def test_pareto_dominated_count(self):
        objs = [
            Objective(name="x", optimize="min", function=lambda s: s[0]),
            Objective(name="y", optimize="min", function=lambda s: s[1]),
        ]
        solutions = [(1, 1), (2, 2), (3, 3)]
        pf = self.opt.compute_pareto_front(solutions, objs)
        assert pf.dominated_count == 2

    def test_pareto_objective_names(self):
        objs = [Objective(name="alpha", optimize="min", function=lambda s: s)]
        pf = self.opt.compute_pareto_front([1, 2], objs)
        assert pf.objectives == ["alpha"]


# ============================================================
# MultiObjectiveOptimizer.find_knee_point
# ============================================================

class TestFindKneePoint:
    def setup_method(self):
        self.opt = MultiObjectiveOptimizer()

    def test_knee_point_two_objectives(self):
        objs = [
            Objective(name="cost", optimize="min", function=lambda s: s[0]),
            Objective(name="quality", optimize="max", function=lambda s: s[1]),
        ]
        solutions = [
            (1, 10), (2, 8), (3, 7), (4, 6.5), (5, 6),
        ]
        pf = self.opt.compute_pareto_front(solutions, objs)
        knee = self.opt.find_knee_point(pf, objs, solutions)
        assert knee is not None
        assert knee in pf.solutions

    def test_knee_point_empty_front(self):
        objs = [Objective(name="x", optimize="min", function=lambda s: s)]
        pf = ParetoFront()
        knee = self.opt.find_knee_point(pf, objs, [])
        assert knee is None

    def test_knee_point_single_solution(self):
        objs = [Objective(name="x", optimize="min", function=lambda s: s)]
        pf = ParetoFront(solutions=[42])
        knee = self.opt.find_knee_point(pf, objs, [42])
        assert knee == 42

    def test_knee_point_three_objectives(self):
        objs = [
            Objective(name="x", optimize="min", function=lambda s: s[0]),
            Objective(name="y", optimize="min", function=lambda s: s[1]),
            Objective(name="z", optimize="max", function=lambda s: s[2]),
        ]
        solutions = [(1, 1, 1), (2, 2, 5), (3, 3, 10)]
        pf = self.opt.compute_pareto_front(solutions, objs)
        knee = self.opt.find_knee_point(pf, objs, solutions)
        assert knee is not None

    def test_knee_point_no_objectives(self):
        pf = ParetoFront(solutions=[1])
        knee = self.opt.find_knee_point(pf, [], [1])
        assert knee == 1


# ============================================================
# MultiObjectiveOptimizer.scalarize
# ============================================================

class TestScalarize:
    def setup_method(self):
        self.opt = MultiObjectiveOptimizer()

    def test_weighted_sum_basic(self):
        vals = [10.0, 5.0]
        weights = [0.6, 0.4]
        score = self.opt.scalarize(vals, weights, "weighted_sum")
        assert score == pytest.approx(8.0)

    def test_weighted_sum_default_weights(self):
        vals = [3.0, 7.0]
        score = self.opt.scalarize(vals, method="weighted_sum")
        assert score == pytest.approx(5.0)

    def test_weighted_sum_equal_weights(self):
        vals = [4.0, 6.0]
        weights = [0.5, 0.5]
        score = self.opt.scalarize(vals, weights, "weighted_sum")
        assert score == pytest.approx(5.0)

    def test_weighted_chebyshev(self):
        vals = [3.0, 7.0]
        weights = [0.5, 0.5]
        score = self.opt.scalarize(vals, weights, "weighted_chebyshev")
        assert score >= 0.0

    def test_scalarize_weight_mismatch_raises(self):
        with pytest.raises(ValueError):
            self.opt.scalarize([1, 2], [0.5], "weighted_sum")

    def test_scalarize_unknown_method_raises(self):
        with pytest.raises(ValueError):
            self.opt.scalarize([1, 2], [0.5, 0.5], "unknown")

    def test_weighted_sum_single_value(self):
        score = self.opt.scalarize([42.0], [1.0], "weighted_sum")
        assert score == pytest.approx(42.0)

    def test_weighted_chebyshev_equal_values(self):
        vals = [5.0, 5.0]
        weights = [0.5, 0.5]
        score = self.opt.scalarize(vals, weights, "weighted_chebyshev")
        assert score == pytest.approx(0.0)


# ============================================================
# MultiObjectiveOptimizer.nsga_ii_select
# ============================================================

class TestNSGAII:
    def setup_method(self):
        self.opt = MultiObjectiveOptimizer()

    def test_nsga_basic(self):
        objs = [
            Objective(name="cost", optimize="min", function=lambda s: s[0]),
            Objective(name="quality", optimize="max", function=lambda s: s[1]),
        ]
        pop = [(1, 10), (2, 8), (3, 6), (5, 5), (4, 7)]
        selected = self.opt.nsga_ii_select(pop, objs, 3)
        assert len(selected) == 3
        assert all(s in pop for s in selected)

    def test_nsga_larger_than_population(self):
        objs = [
            Objective(name="x", optimize="min", function=lambda s: s[0]),
            Objective(name="y", optimize="max", function=lambda s: s[1]),
        ]
        pop = [(1, 10), (2, 8)]
        selected = self.opt.nsga_ii_select(pop, objs, 5)
        assert len(selected) <= 2

    def test_nsga_single_objective(self):
        objs = [
            Objective(name="x", optimize="min", function=lambda s: s),
        ]
        pop = [1, 2, 3, 4, 5]
        selected = self.opt.nsga_ii_select(pop, objs, 3)
        assert len(selected) == 3
        # Should select the smallest values
        assert 1 in selected

    def test_nsga_empty_population(self):
        objs = [Objective(name="x", optimize="min", function=lambda s: s)]
        selected = self.opt.nsga_ii_select([], objs, 3)
        assert selected == []

    def test_nsga_all_same(self):
        objs = [
            Objective(name="x", optimize="min", function=lambda s: s[0]),
            Objective(name="y", optimize="max", function=lambda s: s[1]),
        ]
        pop = [(5, 5), (5, 5), (5, 5)]
        selected = self.opt.nsga_ii_select(pop, objs, 2)
        assert len(selected) == 2

    def test_nsga_selects_from_first_front(self):
        objs = [
            Objective(name="x", optimize="min", function=lambda s: s[0]),
            Objective(name="y", optimize="max", function=lambda s: s[1]),
        ]
        pop = [(1, 10), (10, 1), (5, 5)]
        selected = self.opt.nsga_ii_select(pop, objs, 1)
        assert len(selected) == 1
        # Either (1,10) or (10,1) should be on the Pareto front
        assert selected[0] in [(1, 10), (10, 1)]
