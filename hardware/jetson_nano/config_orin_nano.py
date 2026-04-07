"""
Jetson Orin Nano 8GB Hardware Configuration for NEXUS Marine Robotics.

Defines hardware dataclasses for the Jetson Orin Nano 8GB module
with Ampere GPU and upgraded ARM cores.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict


@dataclass(frozen=True)
class OrinNanoCPUConfig:
    """CPU configuration for Jetson Orin Nano 8GB."""
    arch: str = "ARM"
    cores: int = 6
    core_type: str = "Cortex-A78AE"
    clock_ghz: float = 1.5


@dataclass(frozen=True)
class OrinNanoGPUConfig:
    """GPU configuration for Jetson Orin Nano 8GB."""
    name: str = "Ampere"
    cuda_cores: int = 1024
    tflops: float = 2.5
    tensor_cores: int = 32
    tops: int = 40


@dataclass(frozen=True)
class OrinNanoMemoryConfig:
    """Memory configuration for Jetson Orin Nano 8GB."""
    ram_gb: int = 8
    bandwidth_gbps: float = 68.0
    type: str = "LPDDR5"


@dataclass(frozen=True)
class OrinNanoStorageConfig:
    """Storage configuration for Jetson Orin Nano 8GB."""
    boot: str = "NVMe SSD"
    usb3_count: int = 4
    pcie_gen: int = 4


@dataclass(frozen=True)
class OrinNanoPowerConfig:
    """Power management configuration for Jetson Orin Nano 8GB."""
    max_watts: int = 25
    thermal_throttle_c: int = 90
    fan_curve: Dict[int, int] = field(
        default_factory=lambda: {40: 0, 55: 30, 70: 60, 85: 100}
    )
    power_modes: Dict[str, int] = field(
        default_factory=lambda: {"15W": 15, "25W": 25}
    )


@dataclass(frozen=True)
class OrinNanoAIConfig:
    """AI / inference configuration for Jetson Orin Nano 8GB."""
    tensorrt_precision: str = "FP16"
    max_batch: int = 16
    inference_threads: int = 6
    max_resolution: int = 3840
    dlas_supported: bool = True


@dataclass(frozen=True)
class OrinNanoPinMapping:
    """GPIO / bus pin mapping for NEXUS marine peripherals on Orin Nano."""
    GPS_TX: int = 14
    GPS_RX: int = 15
    IMU_SDA: int = 1
    IMU_SCL: int = 0
    SONAR_TRIG: int = 18
    SONAR_ECHO: int = 17
    CAM0_CLK: int = 28
    CAM0_DATA0: int = 56
    THRUSTER_I2C: int = 1
    LED: int = 33


@dataclass
class JetsonOrinNanoConfig:
    """
    Top-level Jetson Orin Nano 8GB board configuration.

    Composes all sub-configurations and holds board-level metadata.
    """
    board_name: str = "Jetson Orin Nano 8GB"
    cpu: OrinNanoCPUConfig = field(default_factory=OrinNanoCPUConfig)
    gpu: OrinNanoGPUConfig = field(default_factory=OrinNanoGPUConfig)
    memory: OrinNanoMemoryConfig = field(default_factory=OrinNanoMemoryConfig)
    storage: OrinNanoStorageConfig = field(default_factory=OrinNanoStorageConfig)
    power: OrinNanoPowerConfig = field(default_factory=OrinNanoPowerConfig)
    ai: OrinNanoAIConfig = field(default_factory=OrinNanoAIConfig)
    pin_map: OrinNanoPinMapping = field(default_factory=OrinNanoPinMapping)
    jetpack_version: str = "5.1"


def create_jetson_orin_nano_config(**overrides: Any) -> JetsonOrinNanoConfig:
    """
    Factory function that creates a JetsonOrinNanoConfig with optional overrides.

    Supported override keys match JetsonOrinNanoConfig fields, including nested
    dicts for sub-configs (e.g. cpu={'cores': 4}).

    Returns:
        JetsonOrinNanoConfig instance with defaults replaced by any provided overrides.
    """
    defaults: Dict[str, Any] = {
        "board_name": "Jetson Orin Nano 8GB",
        "cpu": OrinNanoCPUConfig(),
        "gpu": OrinNanoGPUConfig(),
        "memory": OrinNanoMemoryConfig(),
        "storage": OrinNanoStorageConfig(),
        "power": OrinNanoPowerConfig(),
        "ai": OrinNanoAIConfig(),
        "pin_map": OrinNanoPinMapping(),
        "jetpack_version": "5.1",
    }

    for key, value in overrides.items():
        if key in defaults and isinstance(value, dict):
            current = defaults[key]
            defaults[key] = replace(current, **value)
        else:
            defaults[key] = value

    return JetsonOrinNanoConfig(**defaults)
