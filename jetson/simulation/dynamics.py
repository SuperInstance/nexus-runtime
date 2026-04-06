"""Multi-body dynamics engine for NEXUS marine robotics simulation.

Implements rigid body physics with semi-implicit Euler integration,
force accumulation, impulse application, and energy/momentum queries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .world import Vector3, WorldObject


@dataclass
class RigidBody:
    """A rigid body with physical properties."""

    mass: float = 1.0
    inertia: float = 1.0
    damping: float = 0.1
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    acceleration: Vector3 = field(default_factory=Vector3)
    force_accumulator: Vector3 = field(default_factory=Vector3)
    angular_velocity: float = 0.0
    torque: float = 0.0


class DynamicsEngine:
    """Physics engine for multi-body simulation using semi-implicit Euler."""

    def __init__(self) -> None:
        self._bodies: Dict[str, RigidBody] = {}
        self._gravity: Vector3 = Vector3(0.0, 0.0, -9.81)
        self._gravity_enabled: bool = True

    def add_body(self, body: RigidBody, body_id: str) -> None:
        self._bodies[body_id] = body

    def remove_body(self, body_id: str) -> bool:
        if body_id in self._bodies:
            del self._bodies[body_id]
            return True
        return False

    def get_body(self, body_id: str) -> Optional[RigidBody]:
        return self._bodies.get(body_id)

    def apply_force(self, body_id: str, force_vector: Vector3, dt: float = 0.0) -> None:
        body = self._bodies.get(body_id)
        if body is not None:
            body.force_accumulator = body.force_accumulator.add(force_vector)

    def apply_gravity(self, body_id: str) -> None:
        if not self._gravity_enabled:
            return
        body = self._bodies.get(body_id)
        if body is not None:
            gravity_force = self._gravity.scale(body.mass)
            body.force_accumulator = body.force_accumulator.add(gravity_force)

    def step(self, dt: float) -> None:
        """Advance simulation by dt using semi-implicit Euler integration.

        Semi-implicit Euler:
            v(t+dt) = v(t) + a * dt
            x(t+dt) = x(t) + v(t+dt) * dt
        """
        for body in self._bodies.values():
            if self._gravity_enabled:
                self.apply_gravity_via_body(body)

            # Compute acceleration from accumulated forces
            if body.mass > 0.0:
                body.acceleration = body.force_accumulator.scale(1.0 / body.mass)
            else:
                body.acceleration = Vector3(0.0, 0.0, 0.0)

            # Semi-implicit Euler: update velocity first
            body.velocity = body.velocity.add(body.acceleration.scale(dt))

            # Apply damping
            damp_factor = max(0.0, 1.0 - body.damping * dt)
            body.velocity = body.velocity.scale(damp_factor)

            # Then update position using new velocity
            body.position = body.position.add(body.velocity.scale(dt))

            # Angular dynamics
            if body.inertia > 0.0:
                angular_accel = body.torque / body.inertia
                body.angular_velocity += angular_accel * dt
                body.angular_velocity *= damp_factor

            # Reset accumulators
            body.force_accumulator = Vector3(0.0, 0.0, 0.0)
            body.torque = 0.0

    def apply_gravity_via_body(self, body: RigidBody) -> None:
        gravity_force = self._gravity.scale(body.mass)
        body.force_accumulator = body.force_accumulator.add(gravity_force)

    def compute_kinetic_energy(self, body_id: str) -> float:
        body = self._bodies.get(body_id)
        if body is None:
            return 0.0
        # Translational KE: 0.5 * m * v^2
        v_sq = body.velocity.dot(body.velocity)
        ke_trans = 0.5 * body.mass * v_sq
        # Rotational KE: 0.5 * I * omega^2
        ke_rot = 0.5 * body.inertia * body.angular_velocity ** 2
        return ke_trans + ke_rot

    def compute_momentum(self, body_id: str) -> Vector3:
        body = self._bodies.get(body_id)
        if body is None:
            return Vector3(0.0, 0.0, 0.0)
        return body.velocity.scale(body.mass)

    def apply_impulse(self, body_id: str, impulse_vector: Vector3) -> None:
        body = self._bodies.get(body_id)
        if body is not None and body.mass > 0.0:
            delta_v = impulse_vector.scale(1.0 / body.mass)
            body.velocity = body.velocity.add(delta_v)

    def apply_torque(self, body_id: str, torque: float) -> None:
        body = self._bodies.get(body_id)
        if body is not None:
            body.torque += torque

    def set_gravity(self, gravity: Vector3) -> None:
        self._gravity = gravity

    def enable_gravity(self, enabled: bool) -> None:
        self._gravity_enabled = enabled

    @property
    def bodies(self) -> Dict[str, RigidBody]:
        return dict(self._bodies)

    @property
    def body_count(self) -> int:
        return len(self._bodies)
