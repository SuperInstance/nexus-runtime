"""
Jetson AGX Orin 64GB Hardware Configuration for NEXUS Marine Robotics.

Defines hardware dataclasses for the flagship Jetson AGX Orin 64GB module
with maximum Ampere GPU compute and large memory pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict


@dataclass(frozen=True)
class AGXCPUConfig:
    """CPU configuration for Jetson AGX Orin 64GB."""
    arch: str = "ARM"
    cores: int = 12
    core_type: str = "Cortex-A78AE"
    clock_ghz: float = 2.2


@dataclass(frozen=True)
class AGXGPUConfig:
    """GPU configuration for Jetson AGX Orin 64GB."""
    name: str = "Ampere"
    cuda_cores: int = 2048
    tflops: float = 9.7
    tensor_cores: int = 64
    tops: int = 275


@dataclass(frozen=True)
class AGXMemoryConfig:
    """Memory configuration for Jetson AGX Orin 64GB."""
    ram_gb: int = 64
    bandwidth_gbps: float = 204.8
    type: str = "LPDDR5"


@dataclass(frozen=True)
class AGXStorageConfig:
    """Storage configuration for Jetson AGX Orin 64GB."""
    boot: str = "NVMe SSD"
    usb3_count: int = 5
    pcie_gen: int = 4
    sata_ports: int = 0


@dataclass(frozen=True)
class AGXPowerConfig:
    """Power management configuration for Jetson AGX Orin 64GB."""
    max_watts: int = 60
    thermal_throttle_c: int = 100
    fan_curve: Dict[int, int] = field(
        default_factory=lambda: {40: 0, 55: 20, 70: 50, 85: 80, 95: 100}
    )
    power_modes: Dict[str, int] = field(
        default_factory=lambda: {"15W": 15, "30W": 30, "50W": 50, "60W": 60}
    )


@dataclass(frozen=True)
class AGXAIConfig:
    """AI / inference configuration for Jetson AGX Orin 64GB."""
    tensorrt_precision: str = "FP16"
    max_batch: int = 32
    inference_threads: int = 12
    max_resolution: int = 7680
    dlas_supported: bool = True
    triton_supported: bool = True


@dataclass(frozen=True)
class AGXPinMapping:
    """GPIO / bus pin mapping for NEXUS marine peripherals on AGX Orin."""
    GPS_TX: int = 14
    GPS_RX: int = 15
    IMU_SDA: int = 1
    IMU_SCL: int = 0
    SONAR_TRIG: int = 18
    SONAR_ECHO: int = 17
    CAM0_CLK: int = 28
    CAM0_DATA0: int = 56
    CAM1_CLK: int = 29
    CAM1_DATA0: int = 57
    THRUSTER_I2C: int = 1
    LED: int = 33


@dataclass
class JetsonAGXOrinConfig:
    """
    Top-level Jetson AGX Orin 64GB board configuration.

    Composes all sub-configurations and holds board-level metadata.
    """
    board_name: str = "Jetson AGX Orin 64GB"
    cpu: AGXCPUConfig = field(default_factory=AGXCPUConfig)
    gpu: AGXGPUConfig = field(default_factory=AGXGPUConfig)
    memory: AGXMemoryConfig = field(default_factory=AGXMemoryConfig)
    storage: AGXStorageConfig = field(default_factory=AGXStorageConfig)
    power: AGXPowerConfig = field(default_factory=AGXPowerConfig)
    ai: AGXAIConfig = field(default_factory=AGXAIConfig)
    pin_map: AGXPinMapping = field(default_factory=AGXPinMapping)
    jetpack_version: str = "5.1"


def create_jetson_agx_orin_config(**overrides: Any) -> JetsonAGXOrinConfig:
    """
    Factory function that creates a JetsonAGXOrinConfig with optional overrides.

    Supported override keys match JetsonAGXOrinConfig fields, including nested
    dicts for sub-configs (e.g. cpu={'cores': 8}).

    Returns:
        JetsonAGXOrinConfig instance with defaults replaced by any provided overrides.
    """
    defaults: Dict[str, Any] = {
        "board_name": "Jetson AGX Orin 64GB",
        "cpu": AGXCPUConfig(),
        "gpu": AGXGPUConfig(),
        "memory": AGXMemoryConfig(),
        "storage": AGXStorageConfig(),
        "power": AGXPowerConfig(),
        "ai": AGXAIConfig(),
        "pin_map": AGXPinMapping(),
        "jetpack_version": "5.1",
    }

    for key, value in overrides.items():
        if key in defaults and isinstance(value, dict):
            current = defaults[key]
            defaults[key] = replace(current, **value)
        else:
            defaults[key] = value

    return JetsonAGXOrinConfig(**defaults)
