"""Tests for reward_shaping.py — 38 tests."""

import math
import pytest
from jetson.rl.reward_shaping import (
    NavigationRewardShaper, PatrolRewardShaper, MultiObjectiveReward,
)


class TestNavigationRewardShaper:
    def setup_method(self):
        self.shaper = NavigationRewardShaper()

    def test_create_defaults(self):
        assert self.shaper.goal_weight == 10.0
        assert self.shaper.step_penalty == -0.1

    def test_distance_reward_at_goal(self):
        r = self.shaper.distance_reward((2.0, 2.0), (2.0, 2.0))
        assert r == 10.0  # goal_weight

    def test_distance_reward_far(self):
        r = self.shaper.distance_reward((0.0, 0.0), (10.0, 10.0))
        assert r < 0  # negative for distance

    def test_distance_reward_closer_higher(self):
        r1 = self.shaper.distance_reward((0, 0), (10, 10))
        r2 = self.shaper.distance_reward((5, 5), (10, 10))
        assert r2 > r1  # closer is better (less negative)

    def test_heading_reward_aligned(self):
        r = self.shaper.heading_reward(0.0, 0.0)
        assert r == 2.0  # heading_weight * 1.0

    def test_heading_reward_opposite(self):
        r = self.shaper.heading_reward(0.0, math.pi)
        assert r == 0.0  # heading_weight * (1 - 1)

    def test_heading_reward_partial(self):
        r = self.shaper.heading_reward(0.0, math.pi / 2)
        assert 0 < r < 2.0

    def test_heading_reward_wraps_pi(self):
        r1 = self.shaper.heading_reward(0.1, 2 * math.pi - 0.1)
        r2 = self.shaper.heading_reward(0.1, -0.1)
        assert abs(r1 - r2) < 1e-6

    def test_speed_reward_optimal(self):
        r = self.shaper.speed_reward(5.0, 5.0)
        assert r == 1.0  # speed_weight * 1.0

    def test_speed_reward_below(self):
        r = self.shaper.speed_reward(2.5, 5.0)
        assert r == 0.5

    def test_speed_reward_above(self):
        r = self.shaper.speed_reward(10.0, 5.0)
        assert r == 0.5  # penalized

    def test_speed_reward_zero_optimal(self):
        r = self.shaper.speed_reward(5.0, 0.0)
        assert r == 0.0

    def test_collision_reward_safe(self):
        r = self.shaper.collision_reward(False, 10.0)
        assert r == 0.0

    def test_collision_reward_hit(self):
        r = self.shaper.collision_reward(True, 0.0)
        assert r == -50.0

    def test_collision_reward_proximity(self):
        r = self.shaper.collision_reward(False, 1.5)
        assert r < 0  # proximity penalty

    def test_collision_reward_no_proximity(self):
        r = self.shaper.collision_reward(False, 5.0)
        assert r == 0.0

    def test_boundary_reward_safe(self):
        r = self.shaper.boundary_reward((5.0, 5.0), 10.0)
        assert r == 0.0

    def test_boundary_reward_edge(self):
        r = self.shaper.boundary_reward((0.0, 5.0), 10.0)
        assert r == -10.0

    def test_boundary_reward_corner(self):
        r = self.shaper.boundary_reward((0.0, 0.0), 10.0)
        assert r == -10.0

    def test_compute_returns_dict(self):
        r = self.shaper.compute((0, 0), (5, 5))
        assert isinstance(r, dict)
        assert "total" in r
        assert "distance" in r
        assert "heading" in r
        assert "speed" in r
        assert "collision" in r
        assert "boundary" in r
        assert "step" in r

    def test_compute_total_is_sum(self):
        r = self.shaper.compute((5, 5), (5, 5))
        expected = sum(v for k, v in r.items() if k != "total")
        assert abs(r["total"] - expected) < 1e-10

    def test_custom_weights(self):
        s = NavigationRewardShaper(collision_penalty=-100.0)
        r = s.collision_reward(True, 0.0)
        assert r == -100.0


class TestPatrolRewardShaper:
    def setup_method(self):
        self.shaper = PatrolRewardShaper()

    def test_create_defaults(self):
        assert self.shaper.waypoint_reward == 10.0
        assert self.shaper.completion_bonus == 50.0

    def test_coverage_reward_empty(self):
        r = self.shaper.coverage_reward(set(), 10)
        assert r == 0.0

    def test_coverage_reward_full(self):
        visited = {(x, y) for x in range(5) for y in range(5)}
        r = self.shaper.coverage_reward(visited, 5)
        assert r == 10.0

    def test_coverage_reward_partial(self):
        visited = {(0, 0), (1, 1)}
        r = self.shaper.coverage_reward(visited, 10)
        assert r > 0.0
        assert r < 10.0

    def test_efficiency_reward_no_waypoints(self):
        r = self.shaper.efficiency_reward(10, 0, 5)
        assert r == 0.0

    def test_efficiency_reward_positive(self):
        r = self.shaper.efficiency_reward(10, 5, 10)
        assert r > 0.0

    def test_fuel_reward_full(self):
        r = self.shaper.fuel_reward(100.0, 100.0)
        assert r == 0.5

    def test_fuel_reward_empty(self):
        r = self.shaper.fuel_reward(0.0, 100.0)
        assert r == 0.0

    def test_waypoint_reward_fn_none(self):
        r = self.shaper.waypoint_reward_fn(False, False)
        assert r == 0.0

    def test_waypoint_reward_fn_visited(self):
        r = self.shaper.waypoint_reward_fn(True, False)
        assert r == 10.0

    def test_waypoint_reward_fn_all(self):
        r = self.shaper.waypoint_reward_fn(True, True)
        assert r == 50.0

    def test_idle_penalty_active(self):
        r = self.shaper.idle_penalty_fn(True)
        assert r == -0.5

    def test_idle_penalty_not_idle(self):
        r = self.shaper.idle_penalty_fn(False)
        assert r == 0.0

    def test_compute_returns_dict(self):
        r = self.shaper.compute(set(), 10, 5, 2, 5, 80.0, 100.0)
        assert isinstance(r, dict)
        assert "total" in r
        assert "coverage" in r
        assert "efficiency" in r
        assert "fuel" in r
        assert "waypoint" in r

    def test_compute_total_is_sum(self):
        r = self.shaper.compute(set(), 10, 5, 2, 5, 80.0, 100.0, just_visited_wp=True)
        expected = sum(v for k, v in r.items() if k != "total")
        assert abs(r["total"] - expected) < 1e-10

    def test_compute_with_all_flags(self):
        r = self.shaper.compute(
            {(0, 0)}, 5, 10, 1, 3, 50.0, 100.0,
            just_visited_wp=True, all_visited=True, is_idle=True
        )
        assert r["waypoint"] == 50.0
        assert r["idle"] == -0.5


class TestMultiObjectiveReward:
    def setup_method(self):
        self.mor = MultiObjectiveReward()

    def test_create_empty(self):
        assert self.mor.num_objectives() == 0

    def test_add_objective(self):
        self.mor.add_objective("safety", weight=2.0, value=0.5)
        assert self.mor.num_objectives() == 1

    def test_remove_objective(self):
        self.mor.add_objective("temp", weight=1.0)
        self.mor.remove_objective("temp")
        assert self.mor.num_objectives() == 0

    def test_set_weight(self):
        self.mor.add_objective("speed", weight=1.0, value=5.0)
        self.mor.set_weight("speed", 3.0)
        obj = self.mor.get_objectives()
        assert obj["speed"] == (3.0, 5.0)

    def test_set_value(self):
        self.mor.add_objective("speed", weight=1.0, value=5.0)
        self.mor.set_value("speed", 10.0)
        obj = self.mor.get_objectives()
        assert obj["speed"] == (1.0, 10.0)

    def test_compute_weighted_empty(self):
        assert self.mor.compute_weighted() == 0.0

    def test_compute_weighted_single(self):
        self.mor.add_objective("a", weight=2.0, value=3.0)
        assert self.mor.compute_weighted() == 6.0

    def test_compute_weighted_multiple(self):
        self.mor.add_objective("a", weight=1.0, value=2.0)
        self.mor.add_objective("b", weight=3.0, value=4.0)
        assert self.mor.compute_weighted() == 14.0  # 1*2 + 3*4

    def test_compute_objective_vector(self):
        self.mor.add_objective("a", weight=2.0, value=3.0)
        vec = self.mor.compute_objective_vector()
        assert vec == {"a": 6.0}

    def test_record_and_history(self):
        self.mor.add_objective("x", weight=1.0, value=5.0)
        self.mor.record()
        self.mor.set_value("x", 7.0)
        self.mor.record()
        history = self.mor.get_history()
        assert len(history) == 2
        assert history[0]["x"] == 5.0
        assert history[1]["x"] == 7.0

    def test_pareto_filter_empty(self):
        assert self.mor.pareto_filter() == []

    def test_pareto_filter_single(self):
        self.mor.add_objective("x", weight=1.0, value=5.0)
        self.mor.record()
        pareto = self.mor.pareto_filter()
        assert len(pareto) == 1

    def test_pareto_filter_dominated(self):
        self.mor.add_objective("x", weight=1.0)
        self.mor.set_value("x", 10.0)
        self.mor.record()
        self.mor.set_value("x", 5.0)
        self.mor.record()
        pareto = self.mor.pareto_filter()
        # First solution dominates second
        assert len(pareto) == 1
        assert pareto[0]["x"] == 10.0

    def test_pareto_filter_nondominated(self):
        self.mor.add_objective("a", weight=1.0)
        self.mor.set_value("a", 10.0)
        self.mor.record()
        self.mor.set_value("a", 5.0)
        self.mor.record()
        # With a single objective, higher is always better
        # The (10,0) dominates (5,0), so only 1 pareto solution
        pareto = self.mor.pareto_filter()
        assert len(pareto) == 1

    def test_clear_history(self):
        self.mor.add_objective("x", weight=1.0, value=1.0)
        self.mor.record()
        self.mor.clear_history()
        assert self.mor.get_history() == []

    def test_get_objectives(self):
        self.mor.add_objective("a", weight=1.0, value=2.0)
        obj = self.mor.get_objectives()
        assert isinstance(obj, dict)
        assert "a" in obj
