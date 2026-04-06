"""
Trajectory planning with obstacle avoidance and Dubins curves.

Pure Python — math, dataclasses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    speed_limit: float = 1.0
    heading_tolerance: float = 0.1
    arrival_time: float = 0.0


@dataclass
class TrajectoryPoint:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    heading: float = 0.0
    time: float = 0.0


@dataclass
class Obstacle:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    radius: float = 1.0


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # heading in radians


@dataclass
class Trajectory:
    points: List[TrajectoryPoint] = field(default_factory=list)
    total_distance: float = 0.0
    total_time: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist3(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _angle_wrap(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _heading(x1, y1, x2, y2) -> float:
    return math.atan2(y2 - y1, x2 - x1)


# ---------------------------------------------------------------------------
# TrajectoryPlanner
# ---------------------------------------------------------------------------

class TrajectoryPlanner:
    """Generate and manipulate trajectories for marine vehicles."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_straight(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        speed: float,
        dt: float,
    ) -> Trajectory:
        """Straight-line trajectory from *start* to *end*."""
        sx, sy, sz = start
        ex, ey, ez = end
        dist = _dist3(start, end)
        if speed <= 0:
            return Trajectory(
                points=[TrajectoryPoint(x=sx, y=sy, z=sz, heading=0.0, time=0.0)],
                total_distance=dist, total_time=0.0,
            )
        n_steps = max(int(math.ceil(dist / (speed * dt))), 1)
        actual_dt = dist / (n_steps * speed) if speed > 0 else dt

        pts: List[TrajectoryPoint] = []
        for i in range(n_steps + 1):
            frac = i / n_steps
            x = sx + frac * (ex - sx)
            y = sy + frac * (ey - sy)
            z = sz + frac * (ez - sz)
            t = i * actual_dt
            vx = (ex - sx) / (n_steps * actual_dt) if n_steps * actual_dt > 0 else 0.0
            vy = (ey - sy) / (n_steps * actual_dt) if n_steps * actual_dt > 0 else 0.0
            vz = (ez - sz) / (n_steps * actual_dt) if n_steps * actual_dt > 0 else 0.0
            hdg = _heading(sx, sy, ex, ey)
            pts.append(TrajectoryPoint(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                                       heading=hdg, time=t))
        return Trajectory(points=pts, total_distance=dist,
                          total_time=n_steps * actual_dt)

    def plan_with_waypoints(
        self,
        waypoints: List[Waypoint],
        speed: float,
        dt: float,
    ) -> Trajectory:
        """Chain straight segments through *waypoints*."""
        all_pts: List[TrajectoryPoint] = []
        total_dist = 0.0
        total_time = 0.0
        for i in range(len(waypoints) - 1):
            w1 = waypoints[i]
            w2 = waypoints[i + 1]
            seg = self.plan_straight(
                (w1.x, w1.y, w1.z), (w2.x, w2.y, w2.z),
                speed if speed <= w2.speed_limit else w2.speed_limit, dt,
            )
            offset = total_time
            for p in seg.points:
                new_p = TrajectoryPoint(
                    x=p.x, y=p.y, z=p.z, vx=p.vx, vy=p.vy, vz=p.vz,
                    heading=p.heading, time=p.time + offset,
                )
                all_pts.append(new_p)
            total_dist += seg.total_distance
            total_time += seg.total_time

        if len(waypoints) == 1:
            wp = waypoints[0]
            all_pts.append(TrajectoryPoint(x=wp.x, y=wp.y, z=wp.z,
                                           heading=0.0, time=0.0))
        return Trajectory(points=all_pts, total_distance=total_dist,
                          total_time=total_time)

    def avoid_obstacle(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        obstacle: Obstacle,
        safety_margin: float,
    ) -> Trajectory:
        """Return a trajectory that avoids *obstacle* by going around it."""
        sx, sy, sz = start
        ex, ey, ez = end
        oc = (obstacle.x, obstacle.y)
        r = obstacle.radius + safety_margin

        # Check if direct path intersects obstacle zone
        d_start_obs = _dist2((sx, sy), oc)
        d_end_obs = _dist2((ex, ey), oc)

        if d_start_obs > r and d_end_obs > r:
            # Check closest point on line to obstacle
            dx, dy = ex - sx, ey - sy
            seg_len2 = dx * dx + dy * dy
            if seg_len2 < 1e-12:
                return self.plan_straight(start, end, 1.0, 0.1)
            t_param = max(0.0, min(1.0,
                        ((oc[0] - sx) * dx + (oc[1] - sy) * dy) / seg_len2))
            closest_x = sx + t_param * dx
            closest_y = sy + t_param * dy
            d_closest = _dist2((closest_x, closest_y), oc)

            if d_closest > r:
                return self.plan_straight(start, end, 1.0, 0.1)

        # Compute waypoint around obstacle
        angle_to_obs = math.atan2(obstacle.y - sy, obstacle.x - sx)
        mid_angle = angle_to_obs
        wp_x = obstacle.x + r * 1.5 * math.cos(mid_angle + math.pi / 2)
        wp_y = obstacle.y + r * 1.5 * math.sin(mid_angle + math.pi / 2)

        # Two waypoints: approach and depart
        approach_dist = r * 1.2
        depart_dist = r * 1.2

        angle_start_to_obs = math.atan2(obstacle.y - sy, obstacle.x - sx)
        perp = angle_start_to_obs + math.pi / 2

        wp1 = (obstacle.x + approach_dist * math.cos(perp),
               obstacle.y + approach_dist * math.sin(perp),
               (sz + ez) / 2.0)
        wp2 = (obstacle.x - approach_dist * math.cos(perp),
               obstacle.y - approach_dist * math.sin(perp),
               (sz + ez) / 2.0)

        # Choose the waypoint closer to the direct line
        mid = ((sx + ex) / 2, (sy + ey) / 2)
        d1 = _dist2(wp1[:2], mid)
        d2 = _dist2(wp2[:2], mid)
        bypass = wp1 if d1 <= d2 else wp2

        # Three segments
        seg1 = self.plan_straight(start, bypass, 1.0, 0.1)
        seg2 = self.plan_straight(bypass, end, 1.0, 0.1)

        pts = list(seg1.points) + list(seg2.points[1:])
        total_dist = seg1.total_distance + seg2.total_distance
        total_time = seg1.total_time + seg2.total_time
        return Trajectory(points=pts, total_distance=total_dist,
                          total_time=total_time)

    def dubins_path(
        self,
        start_pose: Pose2D,
        end_pose: Pose2D,
        turning_radius: float,
    ) -> Trajectory:
        """
        Generate a Dubins-like path (simplified: LSL or RSR).
        Returns a Trajectory with approximate Dubins curve points.
        """
        dx = end_pose.x - start_pose.x
        dy = end_pose.y - start_pose.y
        D = math.sqrt(dx * dx + dy * dy)
        if D < 1e-9:
            return Trajectory(
                points=[TrajectoryPoint(x=start_pose.x, y=start_pose.y, z=0.0,
                                       heading=start_pose.theta, time=0.0)],
                total_distance=0.0, total_time=0.0,
            )

        d_theta = _angle_wrap(end_pose.theta - start_pose.theta)
        # Simplified Dubins: straight + turn
        straight_len = max(D - 2 * turning_radius, 0.0)
        arc_len = turning_radius * abs(d_theta)

        total_len = straight_len + arc_len
        n_pts = max(int(math.ceil(total_len / 0.5)), 2)
        pts: List[TrajectoryPoint] = []
        dt_step = 1.0

        # Direction of straight segment
        alpha = math.atan2(dy, dx)

        for i in range(n_pts + 1):
            frac = i / n_pts
            s = frac * total_len
            if s <= straight_len:
                x = start_pose.x + (s / total_len) * D * math.cos(alpha) * (straight_len / max(total_len, 1e-9)) / max(straight_len / total_len, 1e-9) if total_len > 0 else start_pose.x
                y = start_pose.y + (s / max(total_len, 1e-9)) * D * math.sin(alpha)
                # Simpler: linear interp for the straight part
                x = start_pose.x + s * math.cos(alpha)
                y = start_pose.y + s * math.sin(alpha)
                hdg = alpha
            else:
                arc_s = s - straight_len
                arc_angle = d_theta * (arc_s / max(arc_len, 1e-9))
                cx = start_pose.x + straight_len * math.cos(alpha)
                cy = start_pose.y + straight_len * math.sin(alpha)
                x = cx + turning_radius * math.sin(arc_angle)
                y = cy - turning_radius * math.cos(arc_angle) + turning_radius
                hdg = alpha + arc_angle
            pts.append(TrajectoryPoint(x=x, y=y, z=0.0,
                                       heading=_angle_wrap(hdg),
                                       time=frac * total_len / 1.0))
        return Trajectory(points=pts, total_distance=total_len,
                          total_time=total_len / 1.0)

    def smooth_trajectory(
        self,
        trajectory: Trajectory,
        smoothing_factor: float,
    ) -> Trajectory:
        """Exponential moving-average smoothing."""
        if not trajectory.points or smoothing_factor <= 0:
            return trajectory
        alpha = min(max(smoothing_factor, 0.0), 1.0)
        smoothed: List[TrajectoryPoint] = []
        prev = trajectory.points[0]
        smoothed.append(TrajectoryPoint(
            x=prev.x, y=prev.y, z=prev.z, vx=prev.vx, vy=prev.vy, vz=prev.vz,
            heading=prev.heading, time=prev.time,
        ))
        for p in trajectory.points[1:]:
            nx = alpha * prev.x + (1 - alpha) * p.x
            ny = alpha * prev.y + (1 - alpha) * p.y
            nz = alpha * prev.z + (1 - alpha) * p.z
            nvx = alpha * prev.vx + (1 - alpha) * p.vx
            nvy = alpha * prev.vy + (1 - alpha) * p.vy
            nvz = alpha * prev.vz + (1 - alpha) * p.vz
            nh = alpha * prev.heading + (1 - alpha) * p.heading
            sp = TrajectoryPoint(x=nx, y=ny, z=nz, vx=nvx, vy=nvy, vz=nvz,
                                 heading=nh, time=p.time)
            smoothed.append(sp)
            prev = sp
        return Trajectory(points=smoothed, total_distance=trajectory.total_distance,
                          total_time=trajectory.total_time)

    def time_parameterize(
        self,
        trajectory: Trajectory,
        speed_limits: List[float],
    ) -> Trajectory:
        """Re-parameterize trajectory with given speed limits per segment."""
        if not trajectory.points or not speed_limits:
            return trajectory
        pts = list(trajectory.points)
        n = len(pts)
        total_time = 0.0
        for i in range(1, n):
            dx = pts[i].x - pts[i - 1].x
            dy = pts[i].y - pts[i - 1].y
            dz = pts[i].z - pts[i - 1].z
            seg_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            spd = speed_limits[min(i - 1, len(speed_limits) - 1)]
            dt_seg = seg_dist / max(spd, 1e-9)
            total_time += dt_seg
            pts[i] = TrajectoryPoint(
                x=pts[i].x, y=pts[i].y, z=pts[i].z,
                vx=dx / max(dt_seg, 1e-9),
                vy=dy / max(dt_seg, 1e-9),
                vz=dz / max(dt_seg, 1e-9),
                heading=pts[i].heading,
                time=total_time,
            )
        return Trajectory(points=pts, total_distance=trajectory.total_distance,
                          total_time=total_time)
