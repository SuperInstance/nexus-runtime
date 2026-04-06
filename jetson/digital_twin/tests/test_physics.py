"""Tests for 6-DOF vessel physics simulation."""

import math
import pytest
from jetson.digital_twin.physics import (
    VesselState, Force, DragCoefficients, VesselProperties, VesselPhysics
)


class TestVesselState:
    def test_default_state(self):
        s = VesselState()
        assert s.x == 0.0 and s.y == 0.0 and s.z == 0.0
        assert s.vx == 0.0 and s.vy == 0.0 and s.vz == 0.0
        assert s.roll == 0.0 and s.pitch == 0.0 and s.yaw == 0.0

    def test_copy(self):
        s = VesselState(x=1, y=2, z=3, vx=4, vy=5, vz=6, roll=0.1, pitch=0.2, yaw=0.3)
        c = s.copy()
        assert c.x == 1.0 and c.y == 2.0 and c.z == 3.0
        c.x = 99
        assert s.x == 1.0  # original unchanged

    def test_distance_to(self):
        s1 = VesselState(x=0, y=0, z=0)
        s2 = VesselState(x=3, y=4, z=0)
        assert abs(s1.distance_to(s2) - 5.0) < 1e-9

    def test_distance_to_3d(self):
        s1 = VesselState()
        s2 = VesselState(x=1, y=2, z=2)
        assert abs(s1.distance_to(s2) - 3.0) < 1e-9

    def test_speed_zero(self):
        s = VesselState()
        assert s.speed() == 0.0

    def test_speed_nonzero(self):
        s = VesselState(vx=3, vy=4, vz=0)
        assert abs(s.speed() - 5.0) < 1e-9

    def test_speed_3d(self):
        s = VesselState(vx=1, vy=2, vz=2)
        assert abs(s.speed() - 3.0) < 1e-9

    def test_angular_speed_zero(self):
        s = VesselState()
        assert s.angular_speed() == 0.0

    def test_angular_speed(self):
        s = VesselState(wx=1, wy=2, wz=2)
        assert abs(s.angular_speed() - 3.0) < 1e-9

    def test_custom_values(self):
        s = VesselState(x=10, y=-5, z=2.5, vx=1.5, vy=-0.5, vz=0.1,
                        roll=0.1, pitch=-0.2, yaw=1.57,
                        wx=0.01, wy=-0.02, wz=0.03)
        assert s.x == 10.0 and s.y == -5.0 and s.z == 2.5


class TestForce:
    def test_default_force(self):
        f = Force()
        assert f.fx == 0.0 and f.fy == 0.0 and f.fz == 0.0

    def test_magnitude_zero(self):
        f = Force()
        assert f.magnitude() == 0.0

    def test_magnitude(self):
        f = Force(fx=3, fy=4, fz=0)
        assert abs(f.magnitude() - 5.0) < 1e-9

    def test_magnitude_3d(self):
        f = Force(fx=1, fy=2, fz=2)
        assert abs(f.magnitude() - 3.0) < 1e-9

    def test_torque_magnitude_zero(self):
        f = Force()
        assert f.torque_magnitude() == 0.0

    def test_torque_magnitude(self):
        f = Force(torque_x=3, torque_y=4, torque_z=0)
        assert abs(f.torque_magnitude() - 5.0) < 1e-9

    def test_addition(self):
        f1 = Force(fx=1, fy=2, fz=3, torque_x=4, torque_y=5, torque_z=6)
        f2 = Force(fx=10, fy=20, fz=30, torque_x=40, torque_y=50, torque_z=60)
        result = f1 + f2
        assert result.fx == 11 and result.fy == 22 and result.fz == 33
        assert result.torque_x == 44 and result.torque_y == 55 and result.torque_z == 66

    def test_addition_identity(self):
        f = Force(fx=5, fy=3, fz=1)
        zero = Force()
        result = f + zero
        assert result.fx == 5 and result.fy == 3 and result.fz == 1

    def test_multiplication(self):
        f = Force(fx=1, fy=2, fz=3, torque_x=4, torque_y=5, torque_z=6)
        result = f * 2.0
        assert result.fx == 2.0 and result.fy == 4.0 and result.fz == 6.0
        assert result.torque_x == 8.0 and result.torque_y == 10.0 and result.torque_z == 12.0

    def test_multiplication_zero(self):
        f = Force(fx=10, fy=20)
        result = f * 0.0
        assert result.fx == 0.0 and result.fy == 0.0

    def test_zero_check(self):
        assert Force().zero() is True
        assert Force(fx=0.1).zero() is False

    def test_multiple_additions(self):
        forces = [Force(fx=i) for i in range(5)]
        total = Force()
        for f in forces:
            total = total + f
        assert total.fx == 10.0


class TestDragCoefficients:
    def test_defaults(self):
        dc = DragCoefficients()
        assert dc.linear_x == 50.0 and dc.linear_y == 100.0
        assert dc.angular_x == 10.0

    def test_custom(self):
        dc = DragCoefficients(linear_x=10, quadratic_x=5)
        assert dc.linear_x == 10.0 and dc.quadratic_x == 5.0


class TestVesselProperties:
    def test_defaults(self):
        p = VesselProperties()
        assert p.mass == 100.0 and p.inertia_x == 50.0

    def test_custom(self):
        p = VesselProperties(mass=200, inertia_z=100)
        assert p.mass == 200.0 and p.inertia_z == 100.0


class TestVesselPhysics:
    def setup_method(self):
        self.physics = VesselPhysics()

    def test_compute_drag_zero_velocity(self):
        drag = self.physics.compute_drag((0, 0, 0), (0, 0, 0))
        assert drag.fx == 0.0 and drag.fy == 0.0 and drag.fz == 0.0

    def test_compute_drag_linear(self):
        drag = self.physics.compute_drag((1, 0, 0), (0, 0, 0))
        assert drag.fx < 0  # opposing motion

    def test_compute_drag_opposes_motion(self):
        drag = self.physics.compute_drag((2, 0, 0), (0, 0, 0))
        assert drag.fx < 0
        drag2 = self.physics.compute_drag((-2, 0, 0), (0, 0, 0))
        assert drag2.fx > 0  # opposes negative motion

    def test_compute_drag_3d(self):
        drag = self.physics.compute_drag((1, 2, 0), (0, 0, 0))
        assert drag.fx != 0 and drag.fy != 0

    def test_compute_drag_angular(self):
        drag = self.physics.compute_drag((0, 0, 0), (1, 0, 0))
        assert drag.torque_x != 0

    def test_compute_thrust_zero(self):
        thrust = self.physics.compute_thrust(0, 0)
        assert thrust.fx == 0.0 and thrust.fy == 0.0

    def test_compute_thrust_full(self):
        thrust = self.physics.compute_thrust(1.0, 0.0, 500.0)
        assert abs(thrust.fx - 500.0) < 1e-9
        assert abs(thrust.fy) < 1e-9

    def test_compute_thrust_heading(self):
        thrust = self.physics.compute_thrust(0.5, math.pi / 2, 100.0)
        assert abs(thrust.fx) < 1e-9
        assert abs(thrust.fy - 50.0) < 1e-9

    def test_compute_thrust_with_depth(self):
        thrust = self.physics.compute_thrust_with_depth(1.0, 0, math.pi/4, 100)
        assert thrust.fz < 0  # downward force with positive pitch angle
        assert thrust.fx > 0

    def test_compute_buoyancy_neutral(self):
        buoyancy = self.physics.compute_buoyancy(1.0, 1000.0)
        # mass=100, displacement=1000*9.81, weight=100*9.81
        # net = 1000*9.81 - 100*9.81 = 900*9.81 > 0
        assert buoyancy.fz > 0

    def test_compute_buoyancy_zero_submersion(self):
        buoyancy = self.physics.compute_buoyancy(0.0)
        assert buoyancy.fz < 0  # gravity only

    def test_compute_gravity(self):
        gravity = self.physics.compute_gravity()
        assert abs(gravity.fz + 100.0 * 9.81) < 1e-9
        assert gravity.fx == 0 and gravity.fy == 0

    def test_apply_force_no_change_zero(self):
        state = VesselState()
        new_state = self.physics.apply_force(state, Force(), 0.1)
        assert abs(new_state.x - 0.0) < 1e-9

    def test_apply_force_accelerates(self):
        state = VesselState()
        force = Force(fx=100.0)
        new_state = self.physics.apply_force(state, force, 1.0)
        assert new_state.vx > 0
        # Euler: position uses old velocity (0), so x stays 0 in single step
        new_state2 = self.physics.apply_force(new_state, force, 1.0)
        assert new_state2.x > 0

    def test_update_state_single_force(self):
        state = VesselState()
        force = Force(fx=100.0)
        new_state = self.physics.update_state(state, [force], 1.0)
        assert new_state.vx > 0
        new_state2 = self.physics.update_state(new_state, [force], 1.0)
        assert new_state2.x > 0

    def test_update_state_multiple_forces(self):
        state = VesselState()
        forces = [Force(fx=100), Force(fy=100)]
        new_state = self.physics.update_state(state, forces, 1.0)
        assert new_state.vx > 0 and new_state.vy > 0
        new_state2 = self.physics.update_state(new_state, forces, 1.0)
        assert new_state2.x > 0 and new_state2.y > 0

    def test_compute_derivatives_zero(self):
        state = VesselState()
        derivs = self.physics.compute_derivatives(state, [Force()])
        assert derivs['dvx'] == 0.0 and derivs['dvy'] == 0.0

    def test_compute_derivatives_nonzero(self):
        state = VesselState()
        derivs = self.physics.compute_derivatives(state, [Force(fx=100)])
        assert derivs['dvx'] == 1.0  # 100/100 = 1 m/s^2

    def test_rk4_step_matches_euler_small_dt(self):
        state = VesselState()
        force = Force(fx=100)
        euler = self.physics.update_state(state, [force], 0.001)
        rk4 = self.physics.rk4_step(state, [force], 0.001)
        assert abs(euler.x - rk4.x) < 1e-6

    def test_rk4_step_larger_dt(self):
        state = VesselState()
        force = Force(fx=100)
        result = self.physics.rk4_step(state, [force], 0.1)
        assert result.x > 0

    def test_rk4_step_zero_force(self):
        state = VesselState(x=1, y=2, z=3, vx=0.5, vy=0.3)
        result = self.physics.rk4_step(state, [], 0.1)
        assert abs(result.vx - 0.5) < 1e-9
        assert result.x > 1.0  # moves due to velocity

    def test_simulate_euler(self):
        state = VesselState()
        force = Force(fx=100)
        traj = self.physics.simulate(state, [force], 0.1, 10, 'euler')
        assert len(traj) == 11
        assert traj[-1].x > traj[0].x

    def test_simulate_rk4(self):
        state = VesselState()
        force = Force(fx=100)
        traj = self.physics.simulate(state, [force], 0.1, 10, 'rk4')
        assert len(traj) == 11
        assert traj[-1].x > traj[0].x

    def test_kinetic_energy_zero(self):
        state = VesselState()
        assert self.physics.kinetic_energy(state) == 0.0

    def test_kinetic_energy(self):
        state = VesselState(vx=2, vy=0, vz=0)
        # KE = 0.5 * 100 * 4 = 200
        assert abs(self.physics.kinetic_energy(state) - 200.0) < 1e-9

    def test_rotational_energy_zero(self):
        state = VesselState()
        assert self.physics.rotational_energy(state) == 0.0

    def test_rotational_energy(self):
        state = VesselState(wx=2, wy=0, wz=0)
        expected = 0.5 * 50.0 * 4.0  # Ix=50
        assert abs(self.physics.rotational_energy(state) - expected) < 1e-9

    def test_momentum(self):
        state = VesselState(vx=3, vy=4, vz=0)
        mx, my, mz = self.physics.momentum(state)
        assert mx == 300.0 and my == 400.0 and mz == 0.0

    def test_total_force(self):
        forces = [Force(fx=1), Force(fx=2), Force(fx=3)]
        total = self.physics.total_force(forces)
        assert total.fx == 6.0

    def test_total_force_empty(self):
        total = self.physics.total_force([])
        assert total.zero() is True

    def test_custom_properties(self):
        props = VesselProperties(mass=200, inertia_x=100)
        phys = VesselPhysics(properties=props)
        derivs = phys.compute_derivatives(VesselState(), [Force(fx=200)])
        assert derivs['dvx'] == 1.0  # 200/200

    def test_custom_drag(self):
        dc = DragCoefficients(linear_x=10, quadratic_x=5)
        phys = VesselPhysics(drag=dc)
        drag = phys.compute_drag((1, 0, 0), (0, 0, 0))
        # drag = -(10*1 + 5*1*|1|) = -15
        assert drag.fx == -15.0

    def test_simulate_increasing_velocity(self):
        state = VesselState()
        force = Force(fx=100)
        traj = self.physics.simulate(state, [force], 0.1, 20)
        for i in range(1, len(traj)):
            assert traj[i].vx >= traj[i-1].vx

    def test_equilibrium_with_drag(self):
        """Vessel should reach terminal velocity with constant thrust + drag."""
        state = VesselState()
        thrust = Force(fx=500.0)
        traj = self.physics.simulate(state, [thrust], 0.1, 500)
        # Check velocity converges
        v_start = traj[100].vx
        v_end = traj[-1].vx
        assert v_start > 0 and v_end > 0
        # Should be near equilibrium (velocity changes slowly or stays constant)
        accel_late = abs(traj[-1].vx - traj[-2].vx)
        accel_early = abs(traj[10].vx - traj[9].vx)
        assert accel_late <= accel_early
