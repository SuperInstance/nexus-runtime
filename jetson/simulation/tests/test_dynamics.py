"""Tests for dynamics.py — RigidBody, DynamicsEngine."""

import math
import pytest

from jetson.simulation.world import Vector3
from jetson.simulation.dynamics import RigidBody, DynamicsEngine


class TestRigidBody:
    def test_default_creation(self):
        body = RigidBody()
        assert body.mass == 1.0
        assert body.inertia == 1.0
        assert body.damping == 0.1
        assert body.position.magnitude() == 0.0
        assert body.velocity.magnitude() == 0.0

    def test_custom_creation(self):
        body = RigidBody(mass=10.0, inertia=5.0, damping=0.2)
        assert body.mass == 10.0
        assert body.inertia == 5.0
        assert body.damping == 0.2

    def test_initial_accumulators_zero(self):
        body = RigidBody()
        assert body.force_accumulator.magnitude() == 0.0
        assert body.torque == 0.0


class TestDynamicsEngineCreation:
    def test_default_creation(self):
        engine = DynamicsEngine()
        assert engine.body_count == 0

    def test_add_body(self):
        engine = DynamicsEngine()
        body = RigidBody()
        engine.add_body(body, "b1")
        assert engine.body_count == 1

    def test_add_multiple_bodies(self):
        engine = DynamicsEngine()
        engine.add_body(RigidBody(), "a")
        engine.add_body(RigidBody(), "b")
        assert engine.body_count == 2

    def test_remove_body(self):
        engine = DynamicsEngine()
        engine.add_body(RigidBody(), "b1")
        assert engine.remove_body("b1") is True
        assert engine.body_count == 0

    def test_remove_nonexistent(self):
        engine = DynamicsEngine()
        assert engine.remove_body("nope") is False

    def test_get_body(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=5.0)
        engine.add_body(body, "b1")
        found = engine.get_body("b1")
        assert found is not None
        assert found.mass == 5.0

    def test_get_nonexistent_body(self):
        engine = DynamicsEngine()
        assert engine.get_body("nope") is None


class TestApplyForce:
    def test_apply_force_accumulates(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0)
        engine.add_body(body, "b1")
        engine.apply_force("b1", Vector3(10, 0, 0))
        assert body.force_accumulator.x == pytest.approx(10.0)

    def test_apply_multiple_forces(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0)
        engine.add_body(body, "b1")
        engine.apply_force("b1", Vector3(10, 0, 0))
        engine.apply_force("b1", Vector3(0, 5, 0))
        assert body.force_accumulator.x == pytest.approx(10.0)
        assert body.force_accumulator.y == pytest.approx(5.0)

    def test_apply_force_nonexistent(self):
        engine = DynamicsEngine()
        # Should not raise
        engine.apply_force("nope", Vector3(1, 0, 0))


class TestStep:
    def test_step_advances_position(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0)
        engine.add_body(body, "b1")
        engine.enable_gravity(False)
        engine.apply_force("b1", Vector3(10, 0, 0))
        engine.step(1.0)
        # Semi-implicit: v = F/m * dt = 10, x = v * dt = 10
        assert body.position.x == pytest.approx(9.0, abs=1.0)

    def test_step_resets_accumulator(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0)
        engine.add_body(body, "b1")
        engine.enable_gravity(False)
        engine.apply_force("b1", Vector3(10, 0, 0))
        engine.step(0.1)
        assert body.force_accumulator.magnitude() == pytest.approx(0.0)

    def test_step_zero_dt(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0, position=Vector3(5, 0, 0))
        engine.add_body(body, "b1")
        engine.enable_gravity(False)
        engine.step(0.0)
        assert body.position.x == pytest.approx(5.0)

    def test_step_with_gravity(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0, position=Vector3(0, 0, 100))
        engine.add_body(body, "b1")
        initial_z = body.position.z
        engine.step(0.1)
        # Should fall due to gravity
        assert body.position.z < initial_z


class TestKineticEnergy:
    def test_ke_zero_at_rest(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0)
        engine.add_body(body, "b1")
        ke = engine.compute_kinetic_energy("b1")
        assert ke == pytest.approx(0.0)

    def test_ke_translational(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=2.0, velocity=Vector3(3, 4, 0))
        engine.add_body(body, "b1")
        # KE = 0.5 * 2 * (9+16) = 25
        ke = engine.compute_kinetic_energy("b1")
        assert ke == pytest.approx(25.0)

    def test_ke_nonexistent_body(self):
        engine = DynamicsEngine()
        ke = engine.compute_kinetic_energy("nope")
        assert ke == 0.0

    def test_ke_with_rotation(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0, inertia=2.0, angular_velocity=3.0)
        engine.add_body(body, "b1")
        # KE_rot = 0.5 * 2 * 9 = 9
        ke = engine.compute_kinetic_energy("b1")
        assert ke == pytest.approx(9.0)


class TestMomentum:
    def test_momentum_at_rest(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0)
        engine.add_body(body, "b1")
        p = engine.compute_momentum("b1")
        assert p.magnitude() == pytest.approx(0.0)

    def test_momentum_moving(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=5.0, velocity=Vector3(3, 0, 0))
        engine.add_body(body, "b1")
        p = engine.compute_momentum("b1")
        assert p.x == pytest.approx(15.0)

    def test_momentum_nonexistent(self):
        engine = DynamicsEngine()
        p = engine.compute_momentum("nope")
        assert p.magnitude() == pytest.approx(0.0)


class TestImpulse:
    def test_impulse_changes_velocity(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=10.0)
        engine.add_body(body, "b1")
        engine.apply_impulse("b1", Vector3(100, 0, 0))
        # dv = impulse / mass = 10
        assert body.velocity.x == pytest.approx(10.0)

    def test_impulse_nonexistent(self):
        engine = DynamicsEngine()
        engine.apply_impulse("nope", Vector3(100, 0, 0))

    def test_impulse_zero_mass(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=0.0)
        engine.add_body(body, "b1")
        engine.apply_impulse("b1", Vector3(100, 0, 0))
        # Should not change velocity (mass=0 is infinite inertia)
        assert body.velocity.magnitude() == pytest.approx(0.0)


class TestGravity:
    def test_set_gravity(self):
        engine = DynamicsEngine()
        engine.set_gravity(Vector3(0, 0, -5.0))
        engine.enable_gravity(False)
        engine.enable_gravity(True)
        body = RigidBody(mass=1.0, position=Vector3(0, 0, 0))
        engine.add_body(body, "b1")
        engine.step(0.1)
        # F = m*g = -5, a = -5, v = -0.5, x = -0.05
        assert body.velocity.z < 0

    def test_disable_gravity(self):
        engine = DynamicsEngine()
        engine.enable_gravity(False)
        body = RigidBody(mass=1.0, position=Vector3(0, 0, 100))
        engine.add_body(body, "b1")
        engine.step(1.0)
        assert body.position.z == pytest.approx(100.0)

    def test_default_gravity_enabled(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0, position=Vector3(0, 0, 0))
        engine.add_body(body, "b1")
        engine.step(0.1)
        assert body.velocity.z < 0


class TestTorque:
    def test_apply_torque(self):
        engine = DynamicsEngine()
        body = RigidBody(inertia=2.0)
        engine.add_body(body, "b1")
        engine.apply_torque("b1", 10.0)
        assert body.torque == pytest.approx(10.0)

    def test_torque_nonexistent(self):
        engine = DynamicsEngine()
        engine.apply_torque("nope", 10.0)

    def test_angular_velocity_changes(self):
        engine = DynamicsEngine()
        body = RigidBody(inertia=2.0)
        engine.add_body(body, "b1")
        engine.enable_gravity(False)
        engine.apply_torque("b1", 10.0)
        engine.step(1.0)
        # alpha = 10/2 = 5, omega = 5
        assert body.angular_velocity == pytest.approx(5.0, abs=0.5)


class TestDamping:
    def test_damping_reduces_velocity(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0, damping=0.5, velocity=Vector3(10, 0, 0))
        engine.add_body(body, "b1")
        engine.enable_gravity(False)
        engine.step(0.1)
        assert body.velocity.x < 10.0

    def test_zero_damping(self):
        engine = DynamicsEngine()
        body = RigidBody(mass=1.0, damping=0.0, velocity=Vector3(10, 0, 0))
        engine.add_body(body, "b1")
        engine.enable_gravity(False)
        engine.step(0.1)
        # No damping, no forces, velocity stays same (semi-implicit)
        assert body.velocity.x > 9.0


class TestBodiesProperty:
    def test_bodies_dict(self):
        engine = DynamicsEngine()
        engine.add_body(RigidBody(mass=2.0), "a")
        bodies = engine.bodies
        assert "a" in bodies
        assert bodies["a"].mass == 2.0
