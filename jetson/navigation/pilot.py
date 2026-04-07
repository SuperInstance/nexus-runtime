"""Autopilot / Pilot system for autonomous marine navigation.

Implements pilot modes, control computation, waypoint following,
station keeping, emergency handling, and control smoothing.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from math import cos, sin
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

from .geospatial import Coordinate, GeoCalculator
from .collision import CollisionAvoidance, CollisionThreat, VesselState
from .path_follower import PathFollower


class PilotMode(Enum):
    """Autopilot operating modes."""
    MANUAL = 0
    AUTOPILOT = 1
    WAYPOINT_FOLLOWING = 2
    STATION_KEEPING = 3
    EMERGENCY = 4


@dataclass
class PilotCommand:
    """Command output from the autopilot."""
    thrust: float         # -1.0 to 1.0
    rudder: float         # -1.0 to 1.0 (negative = port, positive = starboard)
    target_speed: float   # m/s
    target_heading: float # degrees (0-360)
    mode: PilotMode = PilotMode.AUTOPILOT


@dataclass
class VesselStateInternal:
    """Internal vessel state representation."""
    position: Coordinate
    speed: float
    heading: float
    drift_speed: float = 0.0
    drift_heading: float = 0.0


class Autopilot:
    """Autopilot system for marine vehicles."""

    def __init__(self):
        self.mode = PilotMode.AUTOPILOT
        self._prev_command: Optional[PilotCommand] = None
        self._collision_avoidance = CollisionAvoidance()
        self._max_thrust_rate = 0.1     # Max thrust change per update
        self._max_rudder_rate = 0.15    # Max rudder change per update
        self._max_heading_rate = 5.0    # Max heading change per update (degrees)
        self._max_speed_rate = 0.2      # Max speed change per update (m/s)
        self._stationkeeping_gain = 0.5  # PD gain for station keeping
        self._waypoint_manager = None

    def set_mode(self, mode: PilotMode) -> None:
        """Set the autopilot operating mode."""
        logger.info("Autopilot mode changed: %s -> %s", self.mode.name, mode.name)
        self.mode = mode

    def get_mode(self) -> PilotMode:
        """Get current autopilot mode."""
        return self.mode

    def compute_control(
        self, current_state: VesselStateInternal, target_state: VesselStateInternal
    ) -> PilotCommand:
        """Compute control commands to reach target state.

        Uses proportional control for heading and speed.

        Returns a PilotCommand with thrust and rudder values.
        """
        # Heading error
        heading_error = target_state.heading - current_state.heading
        # Normalize to [-180, 180]
        heading_error = (heading_error + 180) % 360 - 180

        # Speed error
        speed_error = target_state.speed - current_state.speed

        if abs(heading_error) > 30.0:
            logger.warning(
                "Large heading error: %.1f degrees (current=%.1f, target=%.1f)",
                heading_error, current_state.heading, target_state.heading,
            )

        # PD control for heading
        rudder = max(-1.0, min(1.0, heading_error / 45.0))

        # P control for thrust
        thrust = max(-1.0, min(1.0, speed_error / 3.0))

        command = PilotCommand(
            thrust=thrust,
            rudder=rudder,
            target_speed=target_state.speed,
            target_heading=target_state.heading,
            mode=self.mode,
        )
        return self.smooth_control(self._prev_command, command)

    def hold_position(
        self, position: Coordinate, drift: Tuple[float, float]
    ) -> PilotCommand:
        """Compute control to maintain position against drift.

        Args:
            position: Desired station position.
            drift: (drift_speed m/s, drift_heading degrees).

        Returns:
            PilotCommand to counteract drift.
        """
        drift_speed, drift_heading = drift

        if drift_speed < 0.01:
            return PilotCommand(
                thrust=0.0,
                rudder=0.0,
                target_speed=0.0,
                target_heading=0.0,
                mode=PilotMode.STATION_KEEPING,
            )

        # Counter heading: opposite of drift direction
        counter_heading = (drift_heading + 180) % 360

        # Thrust proportional to drift speed
        thrust = min(1.0, (drift_speed / 2.0) * self._stationkeeping_gain)

        return PilotCommand(
            thrust=thrust,
            rudder=0.0,
            target_speed=drift_speed,
            target_heading=counter_heading,
            mode=PilotMode.STATION_KEEPING,
        )

    def follow_waypoints(
        self, state: VesselStateInternal, waypoint_manager
    ) -> PilotCommand:
        """Compute control to follow a waypoint sequence.

        Args:
            state: Current vessel state.
            waypoint_manager: WaypointManager with loaded waypoints.

        Returns:
            PilotCommand for waypoint following.
        """
        waypoints = waypoint_manager.get_all_waypoints()
        if not waypoints:
            return PilotCommand(
                thrust=0.0, rudder=0.0,
                target_speed=0.0, target_heading=state.heading,
                mode=self.mode,
            )

        # Find current target waypoint
        current_wp = waypoint_manager.get_current_target(state.position, waypoints)
        if current_wp is None:
            # All waypoints reached
            return PilotCommand(
                thrust=0.0, rudder=0.0,
                target_speed=0.0, target_heading=state.heading,
                mode=self.mode,
            )

        # Compute path points from current position through remaining waypoints
        path_points = [state.position] + [
            wp.to_coordinate() for wp in waypoints
            if not waypoint_manager.is_waypoint_reached(state.position, wp)
        ]

        if len(path_points) < 2:
            desired_heading = GeoCalculator.bearing(
                state.position, current_wp.to_coordinate()
            )
        else:
            desired_heading = PathFollower.pure_pursuit(
                state.position, path_points, lookahead=50.0
            )

        # Compute cross-track error for speed adjustment
        if len(path_points) >= 2:
            cte = PathFollower.compute_cross_track_error(
                state.position, path_points[0], path_points[1]
            )
            adjusted_speed = PathFollower.compute_speed_adjustment(
                cte.magnitude, current_wp.speed
            )
        else:
            adjusted_speed = current_wp.speed

        target = VesselStateInternal(
            position=current_wp.to_coordinate(),
            speed=adjusted_speed,
            heading=desired_heading,
        )
        return self.compute_control(state, target)

    def emergency_stop(self) -> PilotCommand:
        """Generate emergency stop command.

        Returns a PilotCommand with zero thrust and rudder centered.
        """
        logger.error("Emergency stop activated")
        return PilotCommand(
            thrust=0.0,
            rudder=0.0,
            target_speed=0.0,
            target_heading=0.0,
            mode=PilotMode.EMERGENCY,
        )

    def smooth_control(
        self, prev_cmd: Optional[PilotCommand], new_cmd: PilotCommand
    ) -> PilotCommand:
        """Smooth control transitions to prevent sudden actuator changes.

        Applies rate limiting to thrust and rudder commands.

        Args:
            prev_cmd: Previous pilot command (None for first call).
            new_cmd: Desired new pilot command.

        Returns:
            Smoothed PilotCommand.
        """
        if prev_cmd is None:
            self._prev_command = new_cmd
            return new_cmd

        # Smooth thrust
        thrust_delta = new_cmd.thrust - prev_cmd.thrust
        if abs(thrust_delta) > self._max_thrust_rate:
            thrust_delta = self._max_thrust_rate * (1 if thrust_delta > 0 else -1)
        smooth_thrust = prev_cmd.thrust + thrust_delta

        # Smooth rudder
        rudder_delta = new_cmd.rudder - prev_cmd.rudder
        if abs(rudder_delta) > self._max_rudder_rate:
            rudder_delta = self._max_rudder_rate * (1 if rudder_delta > 0 else -1)
        smooth_rudder = prev_cmd.rudder + rudder_delta

        # Smooth heading
        heading_delta = new_cmd.target_heading - prev_cmd.target_heading
        heading_delta = (heading_delta + 180) % 360 - 180
        if abs(heading_delta) > self._max_heading_rate:
            heading_delta = self._max_heading_rate * (1 if heading_delta > 0 else -1)
        smooth_heading = (prev_cmd.target_heading + heading_delta) % 360

        # Smooth speed
        speed_delta = new_cmd.target_speed - prev_cmd.target_speed
        if abs(speed_delta) > self._max_speed_rate:
            speed_delta = self._max_speed_rate * (1 if speed_delta > 0 else -1)
        smooth_speed = max(0.0, prev_cmd.target_speed + speed_delta)

        smoothed = PilotCommand(
            thrust=max(-1.0, min(1.0, smooth_thrust)),
            rudder=max(-1.0, min(1.0, smooth_rudder)),
            target_speed=smooth_speed,
            target_heading=smooth_heading,
            mode=new_cmd.mode,
        )
        self._prev_command = smoothed
        return smoothed

    def set_max_rates(
        self,
        thrust_rate: Optional[float] = None,
        rudder_rate: Optional[float] = None,
        heading_rate: Optional[float] = None,
        speed_rate: Optional[float] = None,
    ) -> None:
        """Set maximum rate-of-change limits for control smoothing."""
        if thrust_rate is not None:
            self._max_thrust_rate = thrust_rate
        if rudder_rate is not None:
            self._max_rudder_rate = rudder_rate
        if heading_rate is not None:
            self._max_heading_rate = heading_rate
        if speed_rate is not None:
            self._max_speed_rate = speed_rate

    def avoid_collision(
        self, state: VesselStateInternal, threats: List[CollisionThreat]
    ) -> Optional[PilotCommand]:
        """Generate collision avoidance control commands.

        Returns evasive maneuver command if threats are present, None otherwise.
        """
        if not threats:
            return None

        most_severe = max(threats, key=lambda t: t.severity.value)
        own_vessel = VesselState(
            position=state.position,
            speed=state.speed,
            heading=state.heading,
        )
        heading_change, speed_mult = self._collision_avoidance.generate_evasive_maneuver(
            most_severe, own_vessel
        )

        new_heading = (state.heading + heading_change) % 360
        new_speed = state.speed * speed_mult

        return PilotCommand(
            thrust=0.5,
            rudder=max(-1.0, min(1.0, heading_change / 60.0)),
            target_speed=new_speed,
            target_heading=new_heading,
            mode=self.mode,
        )
