"""
Jetson Nano 4GB Hardware Configuration for NEXUS Marine Robotics.

Defines dataclasses for CPU, GPU, Memory, Storage, Power, AI, and Pin Mapping
specific to the Jetson Nano 4GB module running JetPack 4.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict


@dataclass(frozen=True)
class CPUConfig:
    """CPU configuration for Jetson Nano 4GB."""
    arch: str = "ARM"
    cores: int = 4
    core_type: str = "Cortex-A57"
    clock_ghz: float = 1.43


@dataclass(frozen=True)
class GPUConfig:
    """GPU configuration for Jetson Nano 4GB."""
    name: str = "Maxwell"
    cuda_cores: int = 128
    tflops: float = 0.47
    tensor_cores: int = 0


@dataclass(frozen=True)
class MemoryConfig:
    """Memory configuration for Jetson Nano 4GB."""
    ram_gb: int = 4
    bandwidth_gbps: float = 25.6
    type: str = "LPDDR4"


@dataclass(frozen=True)
class StorageConfig:
    """Storage configuration for Jetson Nano 4GB."""
    boot: str = "microSD"
    usb3_count: int = 4


@dataclass(frozen=True)
class PowerConfig:
    """Power management configuration for Jetson Nano 4GB."""
    max_watts: int = 20
    thermal_throttle_c: int = 85
    fan_curve: Dict[int, int] = field(
        default_factory=lambda: {40: 0, 60: 50, 80: 100}
    )


@dataclass(frozen=True)
class AIConfig:
    """AI / inference configuration for Jetson Nano 4GB."""
    tensorrt_precision: str = "FP16"
    max_batch: int = 8
    inference_threads: int = 4
    max_resolution: int = 1920


@dataclass(frozen=True)
class PinMapping:
    """GPIO / bus pin mapping for NEXUS marine peripherals on Jetson Nano."""
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
class JetsonConfig:
    """
    Top-level Jetson Nano 4GB board configuration.

    Composes all sub-configurations and holds board-level metadata.
    """
    board_name: str = "Jetson Nano 4GB"
    cpu: CPUConfig = field(default_factory=CPUConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    pin_map: PinMapping = field(default_factory=PinMapping)
    jetpack_version: str = "4.6"


def create_jetson_nano_config(**overrides: Any) -> JetsonConfig:
    """
    Factory function that creates a JetsonConfig with optional overrides.

    Supported override keys match JetsonConfig fields, including nested
    dicts for sub-configs (e.g. cpu={'cores': 2}).

    Returns:
        JetsonConfig instance with defaults replaced by any provided overrides.
    """
    defaults: Dict[str, Any] = {
        "board_name": "Jetson Nano 4GB",
        "cpu": CPUConfig(),
        "gpu": GPUConfig(),
        "memory": MemoryConfig(),
        "storage": StorageConfig(),
        "power": PowerConfig(),
        "ai": AIConfig(),
        "pin_map": PinMapping(),
        "jetpack_version": "4.6",
    }

    for key, value in overrides.items():
        if key in defaults and isinstance(value, dict):
            current = defaults[key]
            defaults[key] = replace(current, **value)
        else:
            defaults[key] = value

    return JetsonConfig(**defaults)
