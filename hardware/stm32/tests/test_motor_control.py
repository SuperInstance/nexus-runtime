"""
Tests for NEXUS Motor Control Configuration Module.

Covers: ESCConfig, PIDParams, ThrusterConfig, ThrusterArrayConfig,
        duty cycle conversion, thrust mapping, PID computation, failsafes.
"""

import math
import pytest
from hardware.stm32.motor_control import (
    CommutationMode, PWMChannel, ThrusterAxis,
    FailsafeAction, CurrentLimitMode,
    ESCConfig,
    PIDParams,
    ThrusterConfig,
    ThrusterArrayConfig,
)


class TestESCConfig:
    def test_default_pwm_freq(self):
        esc = ESCConfig()
        assert esc.pwm_freq_hz == 16_000

    def test_timer_period_auto(self):
        esc = ESCConfig(pwm_freq_hz=16_000)
        # 168MHz / 16000 - 1 = 10499
        assert esc.timer_period == 10499

    def test_dead_time_ticks(self):
        esc = ESCConfig(dead_time_ns=500, pwm_freq_hz=16_000)
        ticks = esc.dead_time_ticks
        assert ticks > 0

    def test_is_3_phase(self):
        esc = ESCConfig()
        assert esc.is_3_phase is True

    def test_duty_to_ticks(self):
        esc = ESCConfig(pwm_freq_hz=16_000)
        ticks_50 = esc.duty_to_ticks(50.0)
        ticks_100 = esc.duty_to_ticks(100.0)
        assert ticks_50 > 0
        assert ticks_100 >= ticks_50

    def test_duty_clamped(self):
        esc = ESCConfig(min_duty_pct=0.0, max_duty_pct=95.0)
        ticks = esc.duty_to_ticks(110.0)
        assert esc.duty_to_ticks(95.0) == ticks

    def test_validate_ok(self):
        esc = ESCConfig()
        assert esc.validate() == []

    def test_validate_bad_freq(self):
        esc = ESCConfig(pwm_freq_hz=100)
        errors = esc.validate()
        assert any("PWM freq" in e for e in errors)

    def test_validate_bad_duty(self):
        esc = ESCConfig(min_duty_pct=50.0, max_duty_pct=30.0)
        errors = esc.validate()
        assert any("Min duty" in e for e in errors)

    def test_validate_current_inversion(self):
        esc = ESCConfig(continuous_current_limit_a=30.0, peak_current_limit_a=15.0)
        errors = esc.validate()
        assert any("Continuous" in e for e in errors)

    def test_validate_thermal(self):
        esc = ESCConfig(thermal_derating_c=90.0, thermal_shutdown_c=70.0)
        errors = esc.validate()
        assert any("thermal" in e.lower() for e in errors)

    def test_clone(self):
        esc = ESCConfig(pwm_freq_hz=20_000)
        clone = esc.clone()
        clone.pwm_freq_hz = 30_000
        assert esc.pwm_freq_hz == 20_000


class TestPIDParams:
    def test_default_params(self):
        pid = PIDParams()
        assert pid.kp == 1.0
        assert pid.ki == 0.1
        assert pid.kd == 0.01

    def test_integral_limit_auto(self):
        pid = PIDParams(output_limit=1.0)
        assert pid.integral_limit == 0.5

    def test_compute_positive(self):
        pid = PIDParams(kp=1.0, ki=0.0, kd=0.0, output_limit=1.0)
        output, _ = pid.compute(setpoint=1000, measurement=0, dt=0.01)
        assert output > 0

    def test_compute_clamps(self):
        pid = PIDParams(kp=1000.0, ki=0.0, kd=0.0, output_limit=1.0)
        output, _ = pid.compute(setpoint=100, measurement=0, dt=0.01)
        assert abs(output) <= 1.0

    def test_deadband(self):
        pid = PIDParams(kp=0.1, ki=0.0, kd=0.0, output_deadband=0.5)
        output, _ = pid.compute(setpoint=1.0, measurement=0.9, dt=0.01)
        assert output == 0.0

    def test_validate_ok(self):
        pid = PIDParams()
        assert pid.validate() == []

    def test_validate_negative_kp(self):
        pid = PIDParams(kp=-1.0)
        errors = pid.validate()
        assert any("Kp" in e for e in errors)

    def test_validate_bad_deadband(self):
        pid = PIDParams(output_deadband=2.0, output_limit=1.0)
        errors = pid.validate()
        assert any("Deadband" in e for e in errors)


class TestThrusterConfig:
    def test_thrust_to_command_forward(self):
        t = ThrusterConfig(max_thrust_N=100.0, max_reverse_thrust_N=80.0)
        assert t.thrust_to_command(50.0) == pytest.approx(0.5)

    def test_thrust_to_command_reverse(self):
        t = ThrusterConfig(max_thrust_N=100.0, max_reverse_thrust_N=80.0)
        assert t.thrust_to_command(-80.0) == pytest.approx(-1.0)

    def test_command_to_thrust(self):
        t = ThrusterConfig(max_thrust_N=120.0, max_reverse_thrust_N=80.0)
        assert t.command_to_thrust(0.5) == pytest.approx(60.0)

    def test_command_to_thrust_clamp(self):
        t = ThrusterConfig(max_thrust_N=120.0)
        assert t.command_to_thrust(2.0) == pytest.approx(120.0)

    def test_force_vector_z_axis(self):
        t = ThrusterConfig(axis=ThrusterAxis.Z)
        fx, fy, fz = t.force_vector(100.0)
        assert fx == 0.0
        assert fy == 0.0
        assert fz == 100.0

    def test_force_vector_x_axis(self):
        t = ThrusterConfig(axis=ThrusterAxis.X, thrust_direction=0.0)
        fx, fy, fz = t.force_vector(50.0)
        assert fx == pytest.approx(50.0)
        assert fz == 0.0

    def test_validate_ok(self):
        t = ThrusterConfig()
        assert t.validate() == []

    def test_validate_bad_efficiency(self):
        t = ThrusterConfig(efficiency=1.5)
        errors = t.validate()
        assert any("Efficiency" in e for e in errors)


class TestThrusterArrayConfig:
    def test_default_6_thrusters(self):
        arr = ThrusterArrayConfig()
        assert arr.num_thrusters == 6

    def test_total_max_thrust(self):
        arr = ThrusterArrayConfig()
        assert arr.total_max_thrust_N == 6 * 120.0

    def test_validate_ok(self):
        arr = ThrusterArrayConfig()
        assert arr.validate() == []

    def test_validate_too_few_thrusters(self):
        arr = ThrusterArrayConfig(thrusters=[
            ThrusterConfig(thruster_id=0),
            ThrusterConfig(thruster_id=1),
        ])
        errors = arr.validate()
        assert any("Minimum" in e for e in errors)

    def test_validate_duplicate_id(self):
        arr = ThrusterArrayConfig(thrusters=[
            ThrusterConfig(thruster_id=0),
            ThrusterConfig(thruster_id=0),
            ThrusterConfig(thruster_id=2),
        ])
        errors = arr.validate()
        assert any("Duplicate" in e for e in errors)

    def test_clone_independence(self):
        arr = ThrusterArrayConfig()
        clone = arr.clone()
        clone.thrusters[0].max_thrust_N = 999
        assert arr.thrusters[0].max_thrust_N != 999
