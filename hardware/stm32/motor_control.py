"""
NEXUS Motor Control Configuration Module

Brushless DC (BLDC) motor controller configuration for marine thrusters.
Includes ESC interface config, PID closed-loop tuning, thruster geometry,
and failsafe parameters for the NEXUS distributed intelligence platform.

Supports:
    - FOC (Field-Oriented Control) and trapezoidal commutation
    - Multi-channel thruster arrays for 6-DOF vehicle control
    - Hardware watchdog, thermal derating, and loss-of-signal failsafes
    - PWM generation with dead-time insertion
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CommutationMode(Enum):
    TRAPEZOIDAL = "trapezoidal"       # Six-step commutation (sensorless)
    FOC = "foc"                        # Field-Oriented Control
    FOC_SENSORLESS = "foc_sensorless"  # FOC without position sensor


class PWMChannel(Enum):
    CH1 = "channel_1"
    CH2 = "channel_2"
    CH3 = "channel_3"
    CH4 = "channel_4"


class ThrusterAxis(Enum):
    X = "X"       # Surge (forward/back)
    Y = "Y"       # Sway (port/starboard)
    Z = "Z"       # Heave (up/down)
    RX = "RX"     # Roll
    RY = "RY"     # Pitch
    RZ = "RZ"     # Yaw


class FailsafeAction(Enum):
    NEUTRAL = "neutral"        # Stop thrust (coast)
    BRAKE = "brake"            # Active braking
    LAST_COMMAND = "last"      # Hold last valid command
    SURFACE = "surface"        # Execute surface routine


class CurrentLimitMode(Enum):
    CONTINUOUS = "continuous"
    PEAK = "peak"
    BOTH = "both"


# ---------------------------------------------------------------------------
# ESC Configuration
# ---------------------------------------------------------------------------

@dataclass
class ESCConfig:
    """
    Electronic Speed Controller configuration.

    Configures PWM generation, dead-time insertion, current limits, and
    commutation mode for BLDC motor driving.
    """
    # PWM timing
    pwm_freq_hz: int = 16_000      # Switching frequency
    dead_time_ns: int = 500         # Dead-time between high/low side (nanoseconds)
    min_duty_pct: float = 0.0       # Minimum duty cycle (%)
    max_duty_pct: float = 95.0      # Maximum duty cycle (%)

    # Timer config
    timer_instance: str = "TIM1"
    timer_prescaler: int = 0
    timer_period: int = 0           # Auto-calculated from freq

    # Commutation
    commutation: CommutationMode = CommutationMode.FOC
    pole_pairs: int = 7             # Motor pole pairs
    max_rpm: int = 5000             # Maximum mechanical RPM

    # Current sensing
    current_sense_adc_channel: int = 0
    current_sense_opamp_gain: float = 20.0
    current_sense_resistor_mohm: float = 1.0
    continuous_current_limit_a: float = 15.0
    peak_current_limit_a: float = 25.0
    peak_current_duration_ms: int = 5000
    current_limit_mode: CurrentLimitMode = CurrentLimitMode.BOTH

    # Voltage
    bus_voltage_nominal_v: float = 24.0
    bus_voltage_min_v: float = 18.0
    bus_voltage_max_v: float = 30.0
    input_voltage_adc_channel: int = 1

    # Failsafe
    watchdog_timeout_ms: int = 100
    failsafe_action: FailsafeAction = FailsafeAction.NEUTRAL
    thermal_shutdown_c: float = 85.0
    thermal_derating_c: float = 70.0

    # PWM channels
    pwm_channels: List[PWMChannel] = field(default_factory=lambda: [
        PWMChannel.CH1, PWMChannel.CH2, PWMChannel.CH3
    ])

    def __post_init__(self):
        if self.timer_period == 0 and self.pwm_freq_hz > 0:
            # Assuming 168 MHz timer clock (STM32F4)
            self.timer_period = 168_000_000 // self.pwm_freq_hz - 1

    @property
    def timer_clock_hz(self) -> int:
        return (self.timer_period + 1) * self.pwm_freq_hz

    @property
    def dead_time_ticks(self) -> int:
        """Dead-time in timer ticks."""
        return max(0, int(self.dead_time_ns * self.timer_clock_hz / 1e9))

    @property
    def is_3_phase(self) -> bool:
        return len(self.pwm_channels) >= 3

    def duty_to_ticks(self, duty_pct: float) -> int:
        """Convert duty cycle percentage to timer compare register value."""
        clamped = max(self.min_duty_pct, min(self.max_duty_pct, duty_pct))
        return int((clamped / 100.0) * self.timer_period)

    def validate(self) -> List[str]:
        errors = []
        if self.pwm_freq_hz < 4000 or self.pwm_freq_hz > 100_000:
            errors.append(f"PWM freq {self.pwm_freq_hz} out of [4 kHz, 100 kHz]")
        if self.min_duty_pct < 0 or self.min_duty_pct >= self.max_duty_pct:
            errors.append(f"Min duty {self.min_duty_pct}% >= max duty {self.max_duty_pct}%")
        if self.max_duty_pct > 100:
            errors.append(f"Max duty {self.max_duty_pct}% > 100%")
        if self.continuous_current_limit_a > self.peak_current_limit_a:
            errors.append(
                f"Continuous current {self.continuous_current_limit_a}A > "
                f"peak {self.peak_current_limit_a}A"
            )
        if self.bus_voltage_min_v >= self.bus_voltage_max_v:
            errors.append(f"Bus voltage min {self.bus_voltage_min_v}V >= max {self.bus_voltage_max_v}V")
        if self.thermal_derating_c >= self.thermal_shutdown_c:
            errors.append(
                f"Thermal derating {self.thermal_derating_c}°C >= shutdown {self.thermal_shutdown_c}°C"
            )
        if self.pole_pairs < 1 or self.pole_pairs > 50:
            errors.append(f"Pole pairs {self.pole_pairs} out of [1, 50]")
        return errors

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def clone(self) -> "ESCConfig":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# PID Controller Parameters
# ---------------------------------------------------------------------------

@dataclass
class PIDParams:
    """
    PID controller tuning parameters for closed-loop motor control.

    Includes anti-windup, derivative filtering, and output limiting.
    """
    kp: float = 1.0               # Proportional gain
    ki: float = 0.1               # Integral gain
    kd: float = 0.01              # Derivative gain
    output_limit: float = 1.0     # Output clamp (e.g., -1.0 to 1.0 for normalised thrust)
    integral_limit: Optional[float] = None  # Anti-windup: max integral accumulator
    derivative_filter_hz: float = 100.0     # Low-pass filter on derivative term
    output_deadband: float = 0.0            # Ignore commands below this threshold
    feedforward_gain: float = 0.0           # Feed-forward term

    # Source and target units
    input_unit: str = "rpm"
    output_unit: str = "normalised"

    def __post_init__(self):
        if self.integral_limit is None:
            self.integral_limit = self.output_limit * 0.5

    def compute(self, setpoint: float, measurement: float,
                dt: float, integral_state: float = 0.0) -> Tuple[float, float]:
        """
        Compute PID output. Returns (output, new_integral_state).

        This is a reference implementation; actual PID runs on the MCU.
        """
        error = setpoint - measurement

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup
        integral_state = max(-self.integral_limit,
                             min(self.integral_limit, integral_state + error * dt))
        i_term = self.ki * integral_state

        # Derivative (on error, filtered)
        # Note: simplified; real impl uses previous error
        d_term = self.kd * error  # Placeholder

        # Feed-forward
        ff = self.feedforward_gain * setpoint

        output = p_term + i_term + d_term + ff

        # Clamp output
        output = max(-self.output_limit, min(self.output_limit, output))

        # Apply deadband
        if abs(output) < self.output_deadband:
            output = 0.0

        return output, integral_state

    def validate(self) -> List[str]:
        errors = []
        if self.kp < 0:
            errors.append(f"Kp {self.kp} must be non-negative")
        if self.ki < 0:
            errors.append(f"Ki {self.ki} must be non-negative")
        if self.kd < 0:
            errors.append(f"Kd {self.kd} must be non-negative")
        if self.output_limit <= 0:
            errors.append(f"Output limit {self.output_limit} must be positive")
        if self.output_deadband < 0:
            errors.append(f"Deadband {self.output_deadband} must be non-negative")
        if self.output_deadband >= self.output_limit:
            errors.append(f"Deadband {self.output_deadband} >= output limit {self.output_limit}")
        return errors

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def clone(self) -> "PIDParams":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Thruster Configuration
# ---------------------------------------------------------------------------

@dataclass
class ThrusterConfig:
    """
    Complete thruster configuration combining ESC, PID, and mechanical params.

    Each thruster has a position/orientation on the vehicle and is controlled
    through an ESC with PID closed-loop feedback.
    """
    name: str = "thruster_0"
    thruster_id: int = 0
    axis: ThrusterAxis = ThrusterAxis.X

    # ESC and control
    esc: ESCConfig = field(default_factory=ESCConfig)
    pid: PIDParams = field(default_factory=PIDParams)

    # Mechanical
    max_thrust_N: float = 120.0    # Maximum forward thrust in Newtons
    max_reverse_thrust_N: float = 80.0  # Reverse thrust (typically < forward)
    thrust_direction: float = 0.0  # Direction angle in degrees (0 = forward)
    efficiency: float = 0.70       # Propeller efficiency

    # Position on vehicle (meters from center of gravity)
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0

    # Mapping: thrust (N) -> normalised ESC command (-1.0 to 1.0)
    def thrust_to_command(self, thrust_N: float) -> float:
        """Convert desired thrust in Newtons to normalised ESC command."""
        if thrust_N >= 0:
            return min(1.0, thrust_N / self.max_thrust_N)
        else:
            return max(-1.0, thrust_N / self.max_reverse_thrust_N)

    def command_to_thrust(self, command: float) -> float:
        """Convert normalised ESC command to thrust in Newtons."""
        command = max(-1.0, min(1.0, command))
        if command >= 0:
            return command * self.max_thrust_N
        else:
            return command * self.max_reverse_thrust_N

    def force_vector(self, thrust_N: float) -> Tuple[float, float, float]:
        """Calculate 3D force vector from thrust magnitude and direction."""
        angle_rad = math.radians(self.thrust_direction)
        fx = thrust_N * math.cos(angle_rad)
        fy = thrust_N * math.sin(angle_rad)
        fz = 0.0
        if self.axis in (ThrusterAxis.Z, ThrusterAxis.RX, ThrusterAxis.RY, ThrusterAxis.RZ):
            fx, fy, fz = 0.0, 0.0, thrust_N
        return (fx, fy, fz)

    def validate(self) -> List[str]:
        errors = []
        errors.extend(self.esc.validate())
        errors.extend(self.pid.validate())
        if self.max_thrust_N <= 0:
            errors.append(f"Max thrust {self.max_thrust_N}N must be positive")
        if self.max_reverse_thrust_N <= 0:
            errors.append(f"Max reverse thrust {self.max_reverse_thrust_N}N must be positive")
        if self.efficiency <= 0 or self.efficiency > 1.0:
            errors.append(f"Efficiency {self.efficiency} out of (0, 1.0]")
        return errors

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def clone(self) -> "ThrusterConfig":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Thruster Array
# ---------------------------------------------------------------------------

@dataclass
class ThrusterArrayConfig:
    """
    Multi-thruster array configuration for full 6-DOF vehicle control.

    Defines the thruster layout, allocation matrix, and overall limits
    for an underwater vehicle.
    """
    vehicle_name: str = "NEXUS_ROV"
    thrusters: List[ThrusterConfig] = field(default_factory=list)
    max_bus_current_a: float = 100.0
    battery_capacity_ah: float = 30.0
    operating_depth_m: float = 500.0
    max_depth_m: float = 1000.0

    def __post_init__(self):
        if not self.thrusters:
            # Default 6-thruster ROV configuration
            self.thrusters = [
                ThrusterConfig(name="port_fwd", thruster_id=0, axis=ThrusterAxis.Y,
                               position_x=0.3, position_y=0.2, position_z=0.0),
                ThrusterConfig(name="stbd_fwd", thruster_id=1, axis=ThrusterAxis.Y,
                               position_x=0.3, position_y=-0.2, position_z=0.0),
                ThrusterConfig(name="port_aft", thruster_id=2, axis=ThrusterAxis.Y,
                               position_x=-0.3, position_y=0.2, position_z=0.0),
                ThrusterConfig(name="stbd_aft", thruster_id=3, axis=ThrusterAxis.Y,
                               position_x=-0.3, position_y=-0.2, position_z=0.0),
                ThrusterConfig(name="vert_port", thruster_id=4, axis=ThrusterAxis.Z,
                               position_x=0.0, position_y=0.2, position_z=0.0),
                ThrusterConfig(name="vert_stbd", thruster_id=5, axis=ThrusterAxis.Z,
                               position_x=0.0, position_y=-0.2, position_z=0.0),
            ]

    @property
    def num_thrusters(self) -> int:
        return len(self.thrusters)

    @property
    def total_max_thrust_N(self) -> float:
        return sum(t.max_thrust_N for t in self.thrusters)

    def validate(self) -> List[str]:
        errors = []
        ids = set()
        for t in self.thrusters:
            errors.extend(t.validate())
            if t.thruster_id in ids:
                errors.append(f"Duplicate thruster ID: {t.thruster_id}")
            ids.add(t.thruster_id)
        if self.num_thrusters < 3:
            errors.append(f"Minimum 3 thrusters for controlled vehicle, got {self.num_thrusters}")
        if self.max_bus_current_a <= 0:
            errors.append(f"Max bus current {self.max_bus_current_a}A must be positive")
        if self.max_depth_m <= 0:
            errors.append(f"Max depth {self.max_depth_m}m must be positive")
        return errors

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def clone(self) -> "ThrusterArrayConfig":
        return copy.deepcopy(self)
