"""Tests for jetson.mpc.multi_objective — 31 tests."""
import math
import pytest
from jetson.mpc.multi_objective import (
    Objective,
    ObjectivePriority,
    ParetoPoint,
    MultiObjectiveOptimizer,
)


# ---- Data classes ----

class TestObjectiveDataClass:
    def test_default(self):
        o = Objective()
        assert o.name == ""
        assert o.weight == 1.0
        assert o.function is None
        assert o.priority == ObjectivePriority.MEDIUM

    def test_custom(self):
        o = Objective(name="cost", weight=2.0,
                     function=lambda x: sum(x),
                     priority=ObjectivePriority.CRITICAL)
        assert o.name == "cost"
        assert o.weight == 2.0
        assert o.function is not None
        assert o.priority == ObjectivePriority.CRITICAL

    def test_function_callable(self):
        o = Objective(function=lambda x: x[0] ** 2)
        assert o.function([3]) == 9.0


class TestParetoPoint:
    def test_default(self):
        p = ParetoPoint()
        assert p.objectives == []
        assert p.decision_variables == []
        assert p.dominated is False

    def test_with_data(self):
        p = ParetoPoint(objectives=[1.0, 2.0], decision_variables=[0.5, 1.5])
        assert p.objectives == [1.0, 2.0]
        assert p.dominated is False

    def test_dominated_flag(self):
        p = ParetoPoint(objectives=[1, 2], dominated=True)
        assert p.dominated is True


# ---- MultiObjectiveOptimizer ----

class TestMultiObjectiveOptimizer:
    def setup_method(self):
        self.opt = MultiObjectiveOptimizer()

    def test_add_objective(self):
        self.opt.add_objective("cost", lambda x: x[0] ** 2, weight=1.0)
        assert len(self.opt.objectives) == 1
        assert self.opt.objectives[0].name == "cost"

    def test_add_multiple_objectives(self):
        self.opt.add_objective("a", lambda x: x[0], 1.0)
        self.opt.add_objective("b", lambda x: x[1], 2.0)
        assert len(self.opt.objectives) == 2

    def test_add_objective_with_priority(self):
        self.opt.add_objective("safety", lambda x: 0, 10.0,
                               ObjectivePriority.CRITICAL)
        assert self.opt.objectives[0].priority == ObjectivePriority.CRITICAL

    def test_objectives_copy(self):
        self.opt.add_objective("a", lambda x: 0)
        obj1 = self.opt.objectives
        obj2 = self.opt.objectives
        assert obj1 is not obj2

    def test_scalarize_weighted_sum_equal_weights(self):
        cost = self.opt.scalarize_weighted_sum([1, 2, 3], [1, 1, 1])
        assert cost == pytest.approx(6.0)

    def test_scalarize_weighted_sum_custom_weights(self):
        cost = self.opt.scalarize_weighted_sum([1, 2], [3, 4])
        assert cost == pytest.approx(11.0)

    def test_scalarize_weighted_sum_mismatched_lengths(self):
        cost = self.opt.scalarize_weighted_sum([1, 2, 3, 4], [1, 1])
        assert cost == pytest.approx(3.0)

    def test_scalarize_weighted_sum_zeros(self):
        cost = self.opt.scalarize_weighted_sum([5, 5], [0, 0])
        assert cost == pytest.approx(0.0)

    def test_scalarize_epsilon_no_violation(self):
        cost = self.opt.scalarize_epsilon_constraint(
            primary=1.0, primary_weight=1.0,
            others=[0.5, 0.3], epsilon=[1.0, 1.0],
        )
        assert cost == pytest.approx(1.0)

    def test_scalarize_epsilon_with_violation(self):
        cost = self.opt.scalarize_epsilon_constraint(
            primary=1.0, primary_weight=1.0,
            others=[2.0, 0.3], epsilon=[1.0, 1.0],
        )
        assert cost > 1.0

    def test_scalarize_epsilon_penalty(self):
        cost = self.opt.scalarize_epsilon_constraint(
            primary=0.0, primary_weight=1.0,
            others=[5.0], epsilon=[1.0],
            penalty=100.0,
        )
        assert cost == pytest.approx(400.0)

    def test_scalarize_epsilon_custom_primary_weight(self):
        cost = self.opt.scalarize_epsilon_constraint(
            primary=2.0, primary_weight=5.0,
            others=[], epsilon=[],
        )
        assert cost == pytest.approx(10.0)

    def test_find_pareto_front_single(self):
        p = ParetoPoint(objectives=[1.0, 2.0])
        front = self.opt.find_pareto_front([p])
        assert len(front) == 1
        assert front[0].dominated is False

    def test_find_pareto_front_dominated(self):
        p1 = ParetoPoint(objectives=[1.0, 1.0])
        p2 = ParetoPoint(objectives=[2.0, 2.0])
        front = self.opt.find_pareto_front([p1, p2])
        assert len(front) == 1
        assert front[0].objectives == [1.0, 1.0]

    def test_find_pareto_front_no_dominance(self):
        p1 = ParetoPoint(objectives=[1.0, 5.0])
        p2 = ParetoPoint(objectives=[5.0, 1.0])
        front = self.opt.find_pareto_front([p1, p2])
        assert len(front) == 2

    def test_find_pareto_front_equal(self):
        p1 = ParetoPoint(objectives=[1.0, 1.0])
        p2 = ParetoPoint(objectives=[1.0, 1.0])
        front = self.opt.find_pareto_front([p1, p2])
        # Neither strictly dominates the other (no strictly better in any obj)
        assert len(front) == 2

    def test_dominates_true(self):
        a = ParetoPoint(objectives=[1.0, 1.0])
        b = ParetoPoint(objectives=[2.0, 2.0])
        assert self.opt.dominates(a, b) is True

    def test_dominates_false(self):
        a = ParetoPoint(objectives=[2.0, 2.0])
        b = ParetoPoint(objectives=[1.0, 1.0])
        assert self.opt.dominates(a, b) is False

    def test_dominates_not_strict(self):
        a = ParetoPoint(objectives=[1.0, 2.0])
        b = ParetoPoint(objectives=[1.0, 1.0])
        # a[0] == b[0], a[1] > b[1] → does not dominate
        assert self.opt.dominates(a, b) is False

    def test_dominates_mismatched_len(self):
        a = ParetoPoint(objectives=[1.0])
        b = ParetoPoint(objectives=[1.0, 2.0])
        assert self.opt.dominates(a, b) is False

    def test_select_knee_empty(self):
        assert self.opt.select_knee_point([]) is None

    def test_select_knee_single(self):
        p = ParetoPoint(objectives=[1.0, 2.0])
        result = self.opt.select_knee_point([p])
        assert result is not None
        assert result.objectives == [1.0, 2.0]

    def test_select_knee_two_points(self):
        p1 = ParetoPoint(objectives=[1.0, 5.0])
        p2 = ParetoPoint(objectives=[5.0, 1.0])
        result = self.opt.select_knee_point([p1, p2])
        assert result is not None

    def test_select_knee_returns_pareto_point(self):
        p1 = ParetoPoint(objectives=[0.0, 10.0])
        p2 = ParetoPoint(objectives=[10.0, 0.0])
        p3 = ParetoPoint(objectives=[3.0, 3.0])
        front = self.opt.find_pareto_front([p1, p2, p3])
        knee = self.opt.select_knee_point(front)
        assert knee is not None
        # Knee should be one of the three
        assert knee in [p1, p2, p3]

    def test_adaptive_weights_empty(self):
        self.opt.add_objective("a", lambda x: 0)
        self.opt.add_objective("b", lambda x: 0)
        w = self.opt.adaptive_weights([])
        assert len(w) == 2
        assert all(wi == 1.0 for wi in w)

    def test_adaptive_weights_balanced(self):
        self.opt.add_objective("a", lambda x: 0)
        self.opt.add_objective("b", lambda x: 0)
        history = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        w = self.opt.adaptive_weights(history)
        assert len(w) == 2
        # Equal errors → equal weights
        assert abs(w[0] - w[1]) < 1e-6

    def test_adaptive_weights_unequal(self):
        self.opt.add_objective("a", lambda x: 0)
        self.opt.add_objective("b", lambda x: 0)
        history = [[10.0, 1.0]]
        w = self.opt.adaptive_weights(history)
        assert w[0] > w[1]  # higher error gets higher weight
