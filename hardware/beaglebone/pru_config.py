"""
PRU (Programmable Real-time Unit) Configuration for NEXUS Marine Robotics.

Provides configuration dataclasses for PRU-ICSS real-time motor control,
sensor polling, and deterministic I/O on BeagleBone platforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Tuple


class PRUMode(str, Enum):
    """Operational modes for PRU cores."""
    IDLE = "idle"
    MOTOR_CONTROL = "motor_control"
    SENSOR_POLLING = "sensor_polling"
    PWM_GENERATION = "pwm_generation"
    CUSTOM_FIRMWARE = "custom_firmware"


@dataclass(frozen=True)
class PRUCoreConfig:
    """Configuration for a single PRU core."""
    core_id: int = 0
    mode: PRUMode = PRUMode.IDLE
    clock_hz: int = 200_000_000
    firmware_path: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class PRUSharedMemory:
    """Shared memory region between PRU and ARM host."""
    base_address: int = 0x4A300000
    size_kb: int = 12
    ddr_offset: int = 0x82000000
    ddr_size_kb: int = 256


@dataclass(frozen=True)
class MotorChannelConfig:
    """Configuration for a single motor/actuator channel via PRU PWM."""
    channel_id: int = 0
    pwm_pin: str = "P9.14"
    pwm_frequency_hz: int = 50
    min_pulse_us: int = 1000
    max_pulse_us: int = 2000
    dead_zone_percent: float = 5.0
    direction_pin: str = ""
    encoder_pin: str = ""
    feedback_enabled: bool = False


@dataclass
class PRUControllerConfig:
    """
    Top-level PRU controller configuration.

    Manages PRU cores, shared memory, and motor control channels
    for real-time marine actuator control.
    """
    pru_cores: List[PRUCoreConfig] = field(
        default_factory=lambda: [
            PRUCoreConfig(core_id=0, mode=PRUMode.MOTOR_CONTROL),
            PRUCoreConfig(core_id=1, mode=PRUMode.SENSOR_POLLING),
        ]
    )
    shared_memory: PRUSharedMemory = field(default_factory=PRUSharedMemory)
    motor_channels: List[MotorChannelConfig] = field(
        default_factory=lambda: [
            MotorChannelConfig(channel_id=0, pwm_pin="P9.14"),
            MotorChannelConfig(channel_id=1, pwm_pin="P9.16"),
            MotorChannelConfig(channel_id=2, pwm_pin="P8.19"),
            MotorChannelConfig(channel_id=3, pwm_pin="P8.13"),
        ]
    )
    control_loop_hz: int = 200
    watchdog_timeout_ms: int = 100
    safety_clamp_enabled: bool = True
    emergency_stop_pin: str = "P8.12"


def create_pru_controller_config(**overrides: Any) -> PRUControllerConfig:
    """
    Factory function that creates a PRUControllerConfig with optional overrides.

    Supported override keys match PRUControllerConfig fields, including nested
    dicts for sub-configs.

    Returns:
        PRUControllerConfig instance with defaults replaced by any provided overrides.
    """
    defaults: Dict[str, Any] = {
        "pru_cores": [
            PRUCoreConfig(core_id=0, mode=PRUMode.MOTOR_CONTROL),
            PRUCoreConfig(core_id=1, mode=PRUMode.SENSOR_POLLING),
        ],
        "shared_memory": PRUSharedMemory(),
        "motor_channels": [
            MotorChannelConfig(channel_id=0, pwm_pin="P9.14"),
            MotorChannelConfig(channel_id=1, pwm_pin="P9.16"),
            MotorChannelConfig(channel_id=2, pwm_pin="P8.19"),
            MotorChannelConfig(channel_id=3, pwm_pin="P8.13"),
        ],
        "control_loop_hz": 200,
        "watchdog_timeout_ms": 100,
        "safety_clamp_enabled": True,
        "emergency_stop_pin": "P8.12",
    }

    for key, value in overrides.items():
        defaults[key] = value

    return PRUControllerConfig(**defaults)
