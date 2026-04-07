"""
BeagleBone AI-64 Hardware Configuration for NEXUS Marine Robotics.

Defines hardware dataclasses for the BeagleBone AI-64 board with TI AM5729
dual Cortex-A15 CPUs, C66x DSPs, EVE vision accelerators, and 2GB DDR3,
targeting advanced on-vehicle signal processing and computer vision.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List


@dataclass(frozen=True)
class AI64CPUConfig:
    """CPU configuration for BeagleBone AI-64 — TI AM5729 dual Cortex-A15."""
    arch: str = "ARM"
    cores: int = 2
    core_type: str = "Cortex-A15"
    clock_ghz: float = 1.5


@dataclass(frozen=True)
class AI64DSPConfig:
    """DSP configuration for BeagleBone AI-64 — dual C66x floating-point DSPs."""
    dsp_type: str = "C66x"
    count: int = 2
    clock_ghz: float = 1.0
    eve_count: int = 4
    eve_type: str = "EVE (Vision Accelerator)"


@dataclass(frozen=True)
class AI64MemoryConfig:
    """Memory configuration for BeagleBone AI-64."""
    ram_gb: int = 2
    bandwidth_gbps: float = 12.8
    type: str = "DDR3"


@dataclass(frozen=True)
class AI64StorageConfig:
    """Storage configuration for BeagleBone AI-64."""
    boot: str = "microSD / eMMC 8GB"
    usb2_count: int = 1
    usb3_count: int = 1


@dataclass(frozen=True)
class AI64PowerConfig:
    """Power management configuration for BeagleBone AI-64."""
    max_watts: int = 10
    thermal_throttle_c: int = 90
    input_voltage: str = "12V DC / 5V USB-C"
    power_rail: str = "TPS659162 PMIC"


@dataclass(frozen=True)
class AI64GPIOConfig:
    """GPIO / bus pin mapping for NEXUS marine peripherals on BeagleBone AI-64."""
    GPS_TX: int = "P9.21"
    GPS_RX: int = "P9.22"
    IMU_SDA: int = "P9.20"
    IMU_SCL: int = "P9.19"
    SONAR_TRIG: int = "P8.7"
    SONAR_ECHO: int = "P8.8"
    CAM0_I2C: int = "P9.19"
    CAM0_DATA: int = "P9.27"
    THRUSTER_PWM_0: int = "P9.14"
    THRUSTER_PWM_1: int = "P9.16"
    THRUSTER_PWM_2: int = "P8.19"
    THRUSTER_PWM_3: int = "P8.13"
    LED_USR0: int = "P9.23"
    LED_USR1: int = "P9.24"


@dataclass
class BeagleBoneAI64Config:
    """
    Top-level BeagleBone AI-64 board configuration.

    Composes all sub-configurations and holds board-level metadata.
    Optimized for DSP-accelerated signal processing and vision.
    """
    board_name: str = "BeagleBone AI-64"
    cpu: AI64CPUConfig = field(default_factory=AI64CPUConfig)
    dsp: AI64DSPConfig = field(default_factory=AI64DSPConfig)
    memory: AI64MemoryConfig = field(default_factory=AI64MemoryConfig)
    storage: AI64StorageConfig = field(default_factory=AI64StorageConfig)
    power: AI64PowerConfig = field(default_factory=AI64PowerConfig)
    gpio: AI64GPIOConfig = field(default_factory=AI64GPIOConfig)
    pru_count: int = 4


def create_beaglebone_ai64_config(**overrides: Any) -> BeagleBoneAI64Config:
    """
    Factory function that creates a BeagleBoneAI64Config with optional overrides.

    Supported override keys match BeagleBoneAI64Config fields, including nested
    dicts for sub-configs (e.g. cpu={'cores': 1}).

    Returns:
        BeagleBoneAI64Config instance with defaults replaced by any provided overrides.
    """
    defaults: Dict[str, Any] = {
        "board_name": "BeagleBone AI-64",
        "cpu": AI64CPUConfig(),
        "dsp": AI64DSPConfig(),
        "memory": AI64MemoryConfig(),
        "storage": AI64StorageConfig(),
        "power": AI64PowerConfig(),
        "gpio": AI64GPIOConfig(),
        "pru_count": 4,
    }

    for key, value in overrides.items():
        if key in defaults and isinstance(value, dict):
            current = defaults[key]
            defaults[key] = replace(current, **value)
        else:
            defaults[key] = value

    return BeagleBoneAI64Config(**defaults)
