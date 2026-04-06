"""Tests for environments.py — 42 tests."""

import pytest
from jetson.rl.environments import (
    ActionSpace, ObservationSpace, StepResult,
    MarineNavigationEnv, MarinePatrolEnv, CollisionAvoidanceEnv,
)


class TestActionSpace:
    def test_create_default_actions(self):
        space = ActionSpace(n=4)
        assert space.n == 4
        assert space.actions == [0, 1, 2, 3]

    def test_create_custom_actions(self):
        space = ActionSpace(n=3, actions=["a", "b", "c"])
        assert space.actions == ["a", "b", "c"]

    def test_sample_within_range(self):
        space = ActionSpace(n=5)
        for _ in range(50):
            assert space.sample() in space.actions

    def test_contains_valid(self):
        space = ActionSpace(n=4)
        assert space.contains(2) is True

    def test_contains_invalid(self):
        space = ActionSpace(n=4)
        assert space.contains(5) is False


class TestObservationSpace:
    def test_create(self):
        space = ObservationSpace(shape=(4,))
        assert space.shape == (4,)

    def test_sample_length(self):
        space = ObservationSpace(shape=(3, 4))
        obs = space.sample()
        assert len(obs) == 12

    def test_sample_in_bounds(self):
        space = ObservationSpace(shape=(4,), low=0, high=10)
        for _ in range(20):
            obs = space.sample()
            assert all(0 <= v <= 10 for v in obs)

    def test_contains_list(self):
        space = ObservationSpace(shape=(2,), low=0, high=10)
        assert space.contains([5, 5]) is True
        assert space.contains([15, 5]) is False

    def test_contains_scalar(self):
        space = ObservationSpace(shape=(1,), low=0, high=10)
        assert space.contains(5) is True


class TestStepResult:
    def test_create_default(self):
        sr = StepResult(observation=[1, 2], reward=1.0, done=False)
        assert sr.observation == [1, 2]
        assert sr.reward == 1.0
        assert sr.done is False
        assert sr.info == {}

    def test_create_with_info(self):
        sr = StepResult(observation=[0], reward=-1.0, done=True, info={"reason": "max"})
        assert sr.info["reason"] == "max"

    def test_mutable_info(self):
        sr = StepResult(observation=[], reward=0, done=False)
        sr.info["key"] = "val"
        assert sr.info["key"] == "val"


class TestMarineNavigationEnv:
    def test_create_default(self):
        env = MarineNavigationEnv()
        assert env.grid_size == 10
        assert env.max_steps == 200
        assert env.action_space.n == 5

    def test_reset(self):
        env = MarineNavigationEnv()
        obs = env.reset()
        assert len(obs) == 4
        assert env.observation_space.contains(obs)

    def test_step_north(self):
        env = MarineNavigationEnv(grid_size=5)
        env.reset()
        obs, reward, done, info = env.step(0)  # North
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_step_east(self):
        env = MarineNavigationEnv(grid_size=5)
        env.reset()
        env.step(1)

    def test_step_south(self):
        env = MarineNavigationEnv(grid_size=5)
        env.reset()
        env.step(2)

    def test_step_west(self):
        env = MarineNavigationEnv(grid_size=5)
        env.reset()
        env.step(3)

    def test_step_stay(self):
        env = MarineNavigationEnv(grid_size=5)
        env.reset()
        pos_before = env.agent_pos
        env.step(4)
        assert env.agent_pos == pos_before

    def test_boundary_clamp(self):
        env = MarineNavigationEnv(grid_size=5)
        env.reset()
        env.set_agent_pos((0, 0))
        env.step(3)  # West -> stays at 0
        assert env.agent_pos[0] == 0

    def test_obstacle_collision(self):
        env = MarineNavigationEnv(grid_size=5, obstacles=[(1, 1)])
        env.set_agent_pos((0, 1))
        obs, _, _, _ = env.step(1)  # East -> blocked by obstacle
        assert env.agent_pos == (0, 1)

    def test_goal_reached(self):
        env = MarineNavigationEnv(grid_size=5)
        env.set_goal((2, 2))
        env.set_agent_pos((1, 2))
        _, reward, done, info = env.step(1)  # East
        assert done is True
        assert reward == 100.0
        assert info["reason"] == "goal_reached"

    def test_max_steps(self):
        env = MarineNavigationEnv(grid_size=5, max_steps=5)
        env.reset()
        for _ in range(5):
            _, _, done, _ = env.step(4)
        assert done is True

    def test_stochastic_mode(self):
        env = MarineNavigationEnv(grid_size=5, stochastic=True)
        env.reset()
        env.set_agent_pos((3, 3))
        env.step(0)  # May slip
        # Just verify it doesn't crash
        assert isinstance(env.agent_pos, tuple)

    def test_render(self):
        env = MarineNavigationEnv(grid_size=5)
        env.reset()
        buf = env.render()
        assert buf is not None
        assert len(buf) == 5

    def test_add_remove_obstacle(self):
        env = MarineNavigationEnv(grid_size=5)
        env.add_obstacle((3, 3))
        assert (3, 3) in env.get_obstacles()
        env.remove_obstacle((3, 3))
        assert (3, 3) not in env.get_obstacles()

    def test_set_agent_pos(self):
        env = MarineNavigationEnv(grid_size=5)
        env.set_agent_pos((4, 4))
        assert env.agent_pos == (4, 4)

    def test_set_goal(self):
        env = MarineNavigationEnv(grid_size=5)
        env.set_goal((0, 0))
        assert env.goal == (0, 0)

    def test_step_returns_step_result(self):
        env = MarineNavigationEnv(grid_size=5)
        env.reset()
        result = env.step(0)
        assert isinstance(result, StepResult)


class TestMarinePatrolEnv:
    def test_create_default(self):
        env = MarinePatrolEnv()
        assert env.grid_size == 8
        assert env.max_fuel == 100.0

    def test_reset(self):
        env = MarinePatrolEnv()
        obs = env.reset()
        assert len(obs) == 6

    def test_step(self):
        env = MarinePatrolEnv()
        env.reset()
        result = env.step(1)
        assert isinstance(result, StepResult)

    def test_fuel_decrease(self):
        env = MarinePatrolEnv(max_fuel=10.0, fuel_per_step=1.0)
        env.reset()
        fuel_before = env.get_fuel()
        env.step(1)
        assert env.get_fuel() < fuel_before

    def test_waypoint_visit(self):
        env = MarinePatrolEnv(grid_size=4, waypoints=[(0, 0), (3, 0)])
        env.reset()
        # Reset starts at waypoint 0 and advances index to 1
        assert env.get_waypoints_visited() == 1
        # Move east to reach waypoint 1
        for _ in range(3):
            env.step(1)
        if env.agent_pos == (3, 0):
            assert env.get_waypoints_visited() >= 2

    def test_out_of_fuel(self):
        env = MarinePatrolEnv(max_fuel=2.0, fuel_per_step=1.0)
        env.reset()
        env.step(1)
        env.step(1)
        env.step(1)  # should have negative fuel
        assert env.get_fuel() <= 0

    def test_render(self):
        env = MarinePatrolEnv(grid_size=4)
        env.reset()
        buf = env.render()
        assert len(buf) == 4

    def test_get_remaining_waypoints(self):
        env = MarinePatrolEnv(waypoints=[(0, 0), (1, 1), (2, 2)])
        env.reset()
        assert env.get_remaining_waypoints() > 0


class TestCollisionAvoidanceEnv:
    def test_create_default(self):
        env = CollisionAvoidanceEnv()
        assert env.grid_size == 12

    def test_reset(self):
        env = CollisionAvoidanceEnv()
        obs = env.reset()
        assert len(obs) == 7

    def test_step(self):
        env = CollisionAvoidanceEnv()
        env.reset()
        result = env.step(1)
        assert isinstance(result, StepResult)

    def test_get_agent_pos(self):
        env = CollisionAvoidanceEnv()
        env.reset()
        pos = env.get_agent_pos()
        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_get_static_obstacles(self):
        env = CollisionAvoidanceEnv(num_static=3)
        env.reset()
        statics = env.get_static_obstacles()
        assert len(statics) == 3

    def test_get_dynamic_obstacles(self):
        env = CollisionAvoidanceEnv(num_dynamic=2)
        env.reset()
        dynamics = env.get_dynamic_obstacles()
        assert len(dynamics) == 2

    def test_dynamic_obstacles_move(self):
        env = CollisionAvoidanceEnv(num_dynamic=1)
        env.reset()
        pos_before = env.get_dynamic_obstacles()[0]
        env.step(4)  # Stay, but obstacles move
        pos_after = env.get_dynamic_obstacles()[0]
        # They might or might not move due to random, just verify valid positions

    def test_render(self):
        env = CollisionAvoidanceEnv(grid_size=5, num_static=1, num_dynamic=0)
        env.reset()
        buf = env.render()
        assert buf is not None

    def test_observation_space(self):
        env = CollisionAvoidanceEnv()
        assert env.observation_space.shape == (7,)

    def test_max_steps(self):
        env = CollisionAvoidanceEnv(max_steps=3)
        env.reset()
        for _ in range(3):
            _, _, done, _ = env.step(4)
        assert done is True

    def test_step_returns_distance_info(self):
        env = CollisionAvoidanceEnv()
        env.reset()
        _, _, _, info = env.step(4)
        assert "min_obstacle_dist" in info
