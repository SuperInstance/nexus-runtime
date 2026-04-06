"""Tests for jetson.mpc.constraints — 32 tests."""
import math
import pytest
from jetson.mpc.constraints import (
    Constraint,
    ConstraintKind,
    ConstraintSet,
    ConstraintHandler,
    MarineConstraints,
    VesselState,
    _dist2,
    _point_in_polygon,
)


# ---- helpers ----

class TestConstraintHelpers:
    def test_dist2_origin(self):
        assert _dist2(0, 0, 3, 4) == pytest.approx(5.0)

    def test_dist2_same(self):
        assert _dist2(1, 1, 1, 1) == pytest.approx(0.0)

    def test_point_in_polygon_inside(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_in_polygon(5, 5, poly) is True

    def test_point_in_polygon_outside(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_in_polygon(15, 5, poly) is False

    def test_point_in_polygon_corner(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_in_polygon(0, 0, poly) is True

    def test_point_in_polygon_triangle(self):
        poly = [(0, 0), (10, 0), (5, 10)]
        assert _point_in_polygon(5, 3, poly) is True

    def test_point_in_polygon_empty(self):
        assert _point_in_polygon(5, 5, []) is False


# ---- Constraint data class ----

class TestConstraint:
    def test_default(self):
        c = Constraint()
        assert c.name == ""
        assert c.constraint_type == ConstraintKind.INEQUALITY
        assert c.bounds == (0.0, float("inf"))
        assert c.active is True

    def test_custom(self):
        c = Constraint(name="speed", constraint_type=ConstraintKind.BOUND_UPPER,
                       bounds=(0.0, 5.0), active=False, index=2)
        assert c.name == "speed"
        assert c.active is False
        assert c.index == 2

    def test_bounds_access(self):
        c = Constraint(bounds=(-10, 10))
        assert c.bounds == (-10, 10)


class TestConstraintSet:
    def test_default(self):
        cs = ConstraintSet()
        assert cs.constraints == []
        assert cs.dimension == 0

    def test_with_constraints(self):
        cs = ConstraintSet(constraints=[Constraint(name="a"), Constraint(name="b")],
                           dimension=2)
        assert len(cs.constraints) == 2
        assert cs.dimension == 2


# ---- ConstraintHandler ----

class TestConstraintHandler:
    def setup_method(self):
        self.handler = ConstraintHandler(dimension=3)

    def test_dimension(self):
        assert self.handler.dimension == 3

    def test_add_constraint(self):
        cid = self.handler.add_constraint("x", ConstraintKind.BOUND_LOWER, (0, 10))
        assert cid == 0
        assert len(self.handler.constraints) == 1

    def test_add_multiple_constraints(self):
        self.handler.add_constraint("a", ConstraintKind.BOUND_LOWER, (0, 10))
        self.handler.add_constraint("b", ConstraintKind.BOUND_UPPER, (0, 5))
        assert len(self.handler.constraints) == 2

    def test_constraints_copy(self):
        self.handler.add_constraint("a", ConstraintKind.BOUND_LOWER, (0, 1))
        c1 = self.handler.constraints
        c2 = self.handler.constraints
        assert c1 is not c2

    def test_check_constraints_no_violations(self):
        self.handler.add_constraint("x", ConstraintKind.BOUND_LOWER, (0, 10))
        # Constraint index 0 → variable 0
        violations = self.handler.check_constraints([5.0, 1.0, 2.0])
        assert len(violations) == 0

    def test_check_constraints_below_lower(self):
        self.handler.add_constraint("x", ConstraintKind.BOUND_LOWER, (0, 10))
        violations = self.handler.check_constraints([-1.0, 1.0])
        assert len(violations) == 1
        assert violations[0]["type"] == "below_lower"

    def test_check_constraints_above_upper(self):
        self.handler.add_constraint("x", ConstraintKind.BOUND_LOWER, (0, 5))
        violations = self.handler.check_constraints([10.0, 1.0])
        assert len(violations) == 1
        assert violations[0]["type"] == "above_upper"

    def test_check_constraints_inactive(self):
        c = Constraint(name="x", constraint_type=ConstraintKind.BOUND_LOWER,
                       bounds=(0, 10), active=False, index=0)
        self.handler._constraints.append(c)
        violations = self.handler.check_constraints([-5.0])
        assert len(violations) == 0

    def test_check_constraints_out_of_range(self):
        self.handler.add_constraint("x", ConstraintKind.BOUND_LOWER, (0, 10))
        violations = self.handler.check_constraints([5.0])  # only 1 variable
        assert len(violations) == 0

    def test_project_to_feasible(self):
        self.handler.add_constraint("x", ConstraintKind.BOUND_LOWER, (0, 10))
        projected = self.handler.project_to_feasible([-5.0])
        assert projected[0] == pytest.approx(0.0)

    def test_project_to_feasible_upper(self):
        self.handler.add_constraint("x", ConstraintKind.BOUND_LOWER, (0, 5))
        projected = self.handler.project_to_feasible([10.0])
        assert projected[0] == pytest.approx(5.0)

    def test_project_to_feasible_no_change(self):
        self.handler.add_constraint("x", ConstraintKind.BOUND_LOWER, (0, 10))
        projected = self.handler.project_to_feasible([5.0])
        assert projected[0] == pytest.approx(5.0)

    def test_project_to_feasible_custom_constraints(self):
        cs = [Constraint(name="c", bounds=(1, 3), index=0)]
        projected = self.handler.project_to_feasible([5.0], constraints=cs)
        assert projected[0] == pytest.approx(3.0)

    def test_compute_constraint_margin_inside(self):
        c = Constraint(bounds=(0, 10), index=0)
        margin = self.handler.compute_constraint_margin([5.0], c)
        assert margin == pytest.approx(5.0)

    def test_compute_constraint_margin_at_lower(self):
        c = Constraint(bounds=(0, 10), index=0)
        margin = self.handler.compute_constraint_margin([0.0], c)
        assert margin == pytest.approx(0.0)

    def test_compute_constraint_margin_at_upper(self):
        c = Constraint(bounds=(0, 10), index=0)
        margin = self.handler.compute_constraint_margin([10.0], c)
        assert margin == pytest.approx(0.0)

    def test_compute_constraint_margin_out_of_range(self):
        c = Constraint(bounds=(0, 10), index=5)
        margin = self.handler.compute_constraint_margin([5.0], c)
        assert margin == float("inf")

    def test_soft_constraint_penalty_no_violation(self):
        c = Constraint(bounds=(0, 10), index=0)
        penalty = self.handler.soft_constraint_penalty([5.0], c, 10.0)
        assert penalty == pytest.approx(0.0)

    def test_soft_constraint_penalty_lower(self):
        c = Constraint(bounds=(0, 10), index=0)
        penalty = self.handler.soft_constraint_penalty([-2.0], c, 10.0)
        assert penalty == pytest.approx(40.0)

    def test_soft_constraint_penalty_upper(self):
        c = Constraint(bounds=(0, 10), index=0)
        penalty = self.handler.soft_constraint_penalty([12.0], c, 5.0)
        assert penalty == pytest.approx(20.0)

    def test_soft_constraint_penalty_out_of_range(self):
        c = Constraint(bounds=(0, 10), index=5)
        penalty = self.handler.soft_constraint_penalty([5.0], c, 10.0)
        assert penalty == pytest.approx(0.0)

    def test_generate_constraints(self):
        cs = self.handler.generate_constraints({
            "dimension": 3,
            "bounds": [(0, 10), (-5, 5), (0, 100)],
        })
        assert cs.dimension == 3
        assert len(cs.constraints) == 3

    def test_generate_constraints_empty(self):
        cs = self.handler.generate_constraints({"dimension": 0})
        assert len(cs.constraints) == 0


# ---- MarineConstraints ----

class TestMarineConstraints:
    def setup_method(self):
        self.mc = MarineConstraints()

    def test_actuator_limits(self):
        cs = self.mc.actuator_limits((-100, 100), (-30, 30))
        assert cs.dimension == 2
        assert len(cs.constraints) == 2

    def test_safety_zone_far(self):
        cs = self.mc.safety_zone((0, 0), [(100, 100)], 10.0)
        # Obstacle is far → not active
        assert cs.constraints[0].active is False

    def test_safety_zone_close(self):
        cs = self.mc.safety_zone((0, 0), [(5, 0)], 20.0)
        # Obstacle 5m away, min_distance 20 → active
        assert cs.constraints[0].active is True

    def test_safety_zone_empty(self):
        cs = self.mc.safety_zone((0, 0), [], 10.0)
        assert len(cs.constraints) == 0

    def test_geofence(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        cs = self.mc.geofence(poly)
        assert len(cs.constraints) == 1

    def test_speed_limit(self):
        cs = self.mc.speed_limit(5.0)
        assert cs.dimension == 1
        assert cs.constraints[0].bounds == (0.0, 5.0)

    def test_colregs_safe(self):
        own = VesselState(x=0, y=0)
        others = [VesselState(x=100, y=100)]
        cs = self.mc.colregs_rules(own, others)
        assert len(cs.constraints) == 1
        assert cs.constraints[0].active is False

    def test_colregs_close(self):
        own = VesselState(x=0, y=0)
        others = [VesselState(x=10, y=10)]
        cs = self.mc.colregs_rules(own, others)
        assert cs.constraints[0].active is True

    def test_colregs_multiple(self):
        own = VesselState(x=0, y=0)
        others = [VesselState(x=10, y=10), VesselState(x=200, y=200)]
        cs = self.mc.colregs_rules(own, others)
        assert len(cs.constraints) == 2

    def test_vessel_state(self):
        v = VesselState(x=1, y=2, heading=1.57, speed=3.0)
        assert v.speed == 3.0
        assert abs(v.heading - 1.57) < 1e-6
