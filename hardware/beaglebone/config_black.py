"""
BeagleBone Black Hardware Configuration for NEXUS Marine Robotics.

Defines hardware dataclasses for the BeagleBone Black board with AM3358
ARM Cortex-A8, 512MB DDR3, and PRU (Programmable Real-time Unit) for
deterministic motor control and sensor interfacing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict


@dataclass(frozen=True)
class BlackCPUConfig:
    """CPU configuration for BeagleBone Black — AM3358 ARM Cortex-A8."""
    arch: str = "ARM"
    cores: int = 1
    core_type: str = "Cortex-A8"
    clock_ghz: float = 1.0


@dataclass(frozen=True)
class BlackMemoryConfig:
    """Memory configuration for BeagleBone Black."""
    ram_gb: float = 0.5
    bandwidth_gbps: float = 6.4
    type: str = "DDR3"


@dataclass(frozen=True)
class BlackStorageConfig:
    """Storage configuration for BeagleBone Black."""
    boot: str = "microSD / eMMC 4GB"
    usb2_count: int = 1
    usb3_count: int = 0


@dataclass(frozen=True)
class BlackPowerConfig:
    """Power management configuration for BeagleBone Black."""
    max_watts: int = 5
    thermal_throttle_c: int = 85
    input_voltage: str = "5V DC"
    power_rail: str = "TPS65217C PMIC"


@dataclass(frozen=True)
class BlackGPIOConfig:
    """GPIO / bus pin mapping for NEXUS marine peripherals on BeagleBone Black."""
    GPS_TX: int = "P9.21"
    GPS_RX: int = "P9.22"
    IMU_SDA: int = "P9.20"
    IMU_SCL: int = "P9.19"
    SONAR_TRIG: int = "P8.7"
    SONAR_ECHO: int = "P8.8"
    THRUSTER_PWM_0: int = "P9.14"
    THRUSTER_PWM_1: int = "P9.16"
    THRUSTER_PWM_2: int = "P8.19"
    THRUSTER_PWM_3: int = "P8.13"
    LED_USR0: int = "P9.23"
    LED_USR1: int = "P9.24"
    LED_USR2: int = "P9.26"
    LED_USR3: int = "P8.7"


@dataclass
class BeagleBoneBlackConfig:
    """
    Top-level BeagleBone Black board configuration.

    Composes all sub-configurations and holds board-level metadata.
    """
    board_name: str = "BeagleBone Black"
    cpu: BlackCPUConfig = field(default_factory=BlackCPUConfig)
    memory: BlackMemoryConfig = field(default_factory=BlackMemoryConfig)
    storage: BlackStorageConfig = field(default_factory=BlackStorageConfig)
    power: BlackPowerConfig = field(default_factory=BlackPowerConfig)
    gpio: BlackGPIOConfig = field(default_factory=BlackGPIOConfig)
    pru_count: int = 2


def create_beaglebone_black_config(**overrides: Any) -> BeagleBoneBlackConfig:
    """
    Factory function that creates a BeagleBoneBlackConfig with optional overrides.

    Supported override keys match BeagleBoneBlackConfig fields, including nested
    dicts for sub-configs (e.g. cpu={'clock_ghz': 0.8}).

    Returns:
        BeagleBoneBlackConfig instance with defaults replaced by any provided overrides.
    """
    defaults: Dict[str, Any] = {
        "board_name": "BeagleBone Black",
        "cpu": BlackCPUConfig(),
        "memory": BlackMemoryConfig(),
        "storage": BlackStorageConfig(),
        "power": BlackPowerConfig(),
        "gpio": BlackGPIOConfig(),
        "pru_count": 2,
    }

    for key, value in overrides.items():
        if key in defaults and isinstance(value, dict):
            current = defaults[key]
            defaults[key] = replace(current, **value)
        else:
            defaults[key] = value

    return BeagleBoneBlackConfig(**defaults)
