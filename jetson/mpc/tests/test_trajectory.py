"""Tests for jetson.mpc.trajectory — 34 tests."""
import math
import pytest
from jetson.mpc.trajectory import (
    Waypoint,
    TrajectoryPoint,
    Trajectory,
    Obstacle,
    Pose2D,
    TrajectoryPlanner,
    _dist3,
    _dist2,
    _angle_wrap,
    _heading,
)


# ---- helpers ----

class TestTrajectoryHelpers:
    def test_dist3_same_point(self):
        assert _dist3((1, 2, 3), (1, 2, 3)) == pytest.approx(0.0)

    def test_dist3_1d(self):
        assert _dist3((0, 0, 0), (3, 4, 0)) == pytest.approx(5.0)

    def test_dist2(self):
        assert _dist2((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_angle_wrap_zero(self):
        assert _angle_wrap(0.0) == pytest.approx(0.0)

    def test_angle_wrap_positive(self):
        assert abs(_angle_wrap(2 * math.pi)) < 1e-9

    def test_angle_wrap_negative(self):
        val = _angle_wrap(-math.pi)
        assert val >= -math.pi and val <= math.pi

    def test_heading_east(self):
        assert _heading(0, 0, 1, 0) == pytest.approx(0.0)

    def test_heading_north(self):
        assert _heading(0, 0, 0, 1) == pytest.approx(math.pi / 2)


# ---- data classes ----

class TestTrajectoryDataClasses:
    def test_waypoint_defaults(self):
        w = Waypoint()
        assert w.x == 0.0
        assert w.speed_limit == 1.0

    def test_waypoint_custom(self):
        w = Waypoint(x=1, y=2, z=3, speed_limit=5, heading_tolerance=0.5, arrival_time=10)
        assert w.x == 1 and w.z == 3
        assert w.arrival_time == 10

    def test_trajectory_point(self):
        tp = TrajectoryPoint(x=1, y=2, z=3, vx=0.5, vy=0, vz=-0.1,
                             heading=1.57, time=2.0)
        assert tp.vx == 0.5
        assert tp.time == 2.0

    def test_trajectory(self):
        t = Trajectory(
            points=[TrajectoryPoint(x=0, y=0), TrajectoryPoint(x=1, y=1)],
            total_distance=math.sqrt(2),
            total_time=1.0,
        )
        assert len(t.points) == 2
        assert t.total_distance == pytest.approx(math.sqrt(2))

    def test_obstacle(self):
        o = Obstacle(x=5, y=5, z=0, radius=2)
        assert o.radius == 2

    def test_pose2d(self):
        p = Pose2D(x=1, y=2, theta=3.14)
        assert p.theta == pytest.approx(3.14)


# ---- TrajectoryPlanner ----

class TestTrajectoryPlanner:
    def setup_method(self):
        self.planner = TrajectoryPlanner()

    def test_plan_straight_basic(self):
        traj = self.planner.plan_straight((0, 0, 0), (10, 0, 0), 2.0, 0.1)
        assert len(traj.points) >= 2
        assert abs(traj.total_distance - 10.0) < 1e-3
        assert traj.points[0].x == pytest.approx(0.0)
        assert traj.points[-1].x == pytest.approx(10.0)

    def test_plan_straight_3d(self):
        traj = self.planner.plan_straight((0, 0, 0), (0, 0, 5), 1.0, 0.1)
        assert abs(traj.total_distance - 5.0) < 1e-3
        assert traj.points[-1].z == pytest.approx(5.0)

    def test_plan_straight_velocities(self):
        traj = self.planner.plan_straight((0, 0, 0), (10, 0, 0), 2.0, 0.1)
        for p in traj.points:
            assert p.vy == pytest.approx(0.0, abs=1e-6)

    def test_plan_straight_heading(self):
        traj = self.planner.plan_straight((0, 0, 0), (0, 10, 0), 1.0, 0.1)
        assert abs(traj.points[0].heading - math.pi / 2) < 1e-6

    def test_plan_straight_zero_speed(self):
        traj = self.planner.plan_straight((0, 0, 0), (0, 0, 0), 0.0, 0.1)
        assert len(traj.points) >= 1
        assert traj.total_distance == pytest.approx(0.0)

    def test_plan_with_waypoints_two(self):
        wps = [Waypoint(0, 0, 0), Waypoint(10, 0, 0)]
        traj = self.planner.plan_with_waypoints(wps, 2.0, 0.1)
        assert len(traj.points) >= 2
        assert abs(traj.total_distance - 10.0) < 1e-3

    def test_plan_with_waypoints_three(self):
        wps = [Waypoint(0, 0, 0), Waypoint(5, 0, 0), Waypoint(5, 5, 0)]
        traj = self.planner.plan_with_waypoints(wps, 2.0, 0.1)
        assert abs(traj.total_distance - 10.0) < 1e-2

    def test_plan_with_waypoints_single(self):
        wps = [Waypoint(1, 2, 3)]
        traj = self.planner.plan_with_waypoints(wps, 1.0, 0.1)
        assert len(traj.points) >= 1
        assert traj.points[0].x == pytest.approx(1.0)

    def test_plan_with_waypoints_speed_limit(self):
        wps = [Waypoint(0, 0, 0, speed_limit=0.5), Waypoint(10, 0, 0, speed_limit=0.5)]
        traj = self.planner.plan_with_waypoints(wps, 2.0, 0.1)
        # Speed should be capped to 0.5
        assert traj.total_time > 10.0 / 0.5 * 0.5  # at least some time

    def test_avoid_obstacle_no_intersection(self):
        obs = Obstacle(x=100, y=100, radius=5)
        traj = self.planner.avoid_obstacle((0, 0, 0), (10, 0, 0), obs, 2.0)
        assert abs(traj.total_distance - 10.0) < 1e-2

    def test_avoid_obstacle_on_path(self):
        obs = Obstacle(x=5, y=0, radius=1)
        traj = self.planner.avoid_obstacle((0, 0, 0), (10, 0, 0), obs, 2.0)
        assert len(traj.points) >= 2
        # Should be longer than straight line
        assert traj.total_distance >= 10.0

    def test_avoid_obstacle_start_inside(self):
        obs = Obstacle(x=0.5, y=0, radius=1)
        traj = self.planner.avoid_obstacle((0, 0, 0), (10, 0, 0), obs, 2.0)
        assert len(traj.points) >= 2

    def test_avoid_obstacle_end_inside(self):
        obs = Obstacle(x=9.5, y=0, radius=1)
        traj = self.planner.avoid_obstacle((0, 0, 0), (10, 0, 0), obs, 2.0)
        assert len(traj.points) >= 2

    def test_dubins_basic(self):
        start = Pose2D(0, 0, 0)
        end = Pose2D(10, 0, 0)
        traj = self.planner.dubins_path(start, end, turning_radius=1.0)
        assert len(traj.points) >= 2
        assert traj.total_distance > 0

    def test_dubins_same_point(self):
        p = Pose2D(5, 5, 1.0)
        traj = self.planner.dubins_path(p, p, turning_radius=1.0)
        assert len(traj.points) == 1
        assert traj.total_distance == pytest.approx(0.0)

    def test_dubins_turn(self):
        start = Pose2D(0, 0, 0)
        end = Pose2D(0, 10, math.pi / 2)
        traj = self.planner.dubins_path(start, end, turning_radius=2.0)
        assert traj.total_distance > 0
        assert len(traj.points) >= 2

    def test_smooth_trajectory_basic(self):
        pts = [TrajectoryPoint(x=i, y=0, z=0, time=i * 0.1) for i in range(10)]
        traj = Trajectory(points=pts, total_distance=9, total_time=0.9)
        smoothed = self.planner.smooth_trajectory(traj, 0.5)
        assert len(smoothed.points) == 10
        assert smoothed.total_distance == traj.total_distance

    def test_smooth_trajectory_empty(self):
        traj = Trajectory(points=[])
        smoothed = self.planner.smooth_trajectory(traj, 0.5)
        assert len(smoothed.points) == 0

    def test_smooth_trajectory_zero_factor(self):
        pts = [TrajectoryPoint(x=0, y=0), TrajectoryPoint(x=10, y=10)]
        traj = Trajectory(points=pts)
        smoothed = self.planner.smooth_trajectory(traj, 0.0)
        assert smoothed.points[-1].x == pytest.approx(10.0)

    def test_smooth_trajectory_unit_factor(self):
        pts = [TrajectoryPoint(x=0, y=0), TrajectoryPoint(x=10, y=10)]
        traj = Trajectory(points=pts)
        smoothed = self.planner.smooth_trajectory(traj, 1.0)
        # With alpha=1.0, second point should be same as first
        assert smoothed.points[1].x == pytest.approx(0.0)

    def test_time_parameterize_basic(self):
        pts = [TrajectoryPoint(x=0, y=0, time=0), TrajectoryPoint(x=10, y=0, time=1)]
        traj = Trajectory(points=pts, total_distance=10, total_time=1)
        param = self.planner.time_parameterize(traj, [5.0])
        assert param.total_time > 0
        assert param.points[-1].time > 0

    def test_time_parameterize_empty(self):
        traj = Trajectory(points=[])
        param = self.planner.time_parameterize(traj, [1.0])
        assert len(param.points) == 0

    def test_time_parameterize_single_point(self):
        traj = Trajectory(points=[TrajectoryPoint(x=1, y=2, time=0)])
        param = self.planner.time_parameterize(traj, [1.0])
        assert len(param.points) == 1

    def test_time_parameterize_multiple_speeds(self):
        pts = [TrajectoryPoint(x=i, y=0, time=0) for i in range(5)]
        traj = Trajectory(points=pts, total_distance=4)
        param = self.planner.time_parameterize(traj, [2.0, 1.0, 1.0, 2.0])
        # Lower speed limits should increase time
        assert param.total_time > 0
