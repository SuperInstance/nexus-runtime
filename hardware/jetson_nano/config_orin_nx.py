"""
Jetson Orin NX Hardware Configuration for NEXUS Marine Robotics.

Defines hardware dataclasses for the Jetson Orin NX 16GB module with Ampere GPU
(1024 CUDA cores, 32 Tensor cores), 8-core Carmel CPU, targeting high-performance
NEXUS fleet commander workloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict


@dataclass(frozen=True)
class OrinNXCPUConfig:
    """CPU configuration for Jetson Orin NX — 8-core Carmel."""
    arch: str = "ARM"
    cores: int = 8
    core_type: str = "Carmel"
    clock_ghz: float = 2.0


@dataclass(frozen=True)
class OrinNXGPUConfig:
    """GPU configuration for Jetson Orin NX — NVIDIA Ampere."""
    name: str = "Ampere"
    cuda_cores: int = 1024
    tflops: float = 5.0
    tensor_cores: int = 32
    tops: int = 100


@dataclass(frozen=True)
class OrinNXMemoryConfig:
    """Memory configuration for Jetson Orin NX."""
    ram_gb: int = 16
    bandwidth_gbps: float = 102.4
    type: str = "LPDDR5"


@dataclass(frozen=True)
class OrinNXStorageConfig:
    """Storage configuration for Jetson Orin NX."""
    boot: str = "NVMe SSD"
    usb3_count: int = 4
    pcie_gen: int = 4


@dataclass(frozen=True)
class OrinNXPowerConfig:
    """Power management configuration for Jetson Orin NX."""
    max_watts: int = 25
    thermal_throttle_c: int = 100
    fan_curve: Dict[int, int] = field(
        default_factory=lambda: {35: 0, 50: 15, 65: 40, 80: 70, 95: 100}
    )
    power_modes: Dict[str, int] = field(
        default_factory=lambda: {
            "10W": 10,
            "15W": 15,
            "25W": 25,
        }
    )


@dataclass(frozen=True)
class OrinNXAIConfig:
    """AI / inference configuration for Jetson Orin NX — fleet commander."""
    tensorrt_precision: str = "FP16"
    max_batch: int = 32
    inference_threads: int = 8
    max_resolution: int = 7680
    dlas_supported: bool = True
    triton_supported: bool = True
    dl_model_cache_mb: int = 2048
    fleet_commander_mode: bool = True


@dataclass(frozen=True)
class OrinNXPinMapping:
    """GPIO / bus pin mapping for NEXUS marine peripherals on Orin NX."""
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
    CAM2_CLK: int = 30
    CAM2_DATA0: int = 58
    THRUSTER_I2C: int = 1
    LED: int = 33


@dataclass
class JetsonOrinNXConfig:
    """
    Top-level Jetson Orin NX 16GB board configuration.

    Composes all sub-configurations and holds board-level metadata.
    Optimized for NEXUS fleet commander high-performance workloads.
    """
    board_name: str = "Jetson Orin NX 16GB"
    cpu: OrinNXCPUConfig = field(default_factory=OrinNXCPUConfig)
    gpu: OrinNXGPUConfig = field(default_factory=OrinNXGPUConfig)
    memory: OrinNXMemoryConfig = field(default_factory=OrinNXMemoryConfig)
    storage: OrinNXStorageConfig = field(default_factory=OrinNXStorageConfig)
    power: OrinNXPowerConfig = field(default_factory=OrinNXPowerConfig)
    ai: OrinNXAIConfig = field(default_factory=OrinNXAIConfig)
    pin_map: OrinNXPinMapping = field(default_factory=OrinNXPinMapping)
    jetpack_version: str = "5.1"


def create_jetson_orin_nx_config(**overrides: Any) -> JetsonOrinNXConfig:
    """
    Factory function that creates a JetsonOrinNXConfig with optional overrides.

    Supported override keys match JetsonOrinNXConfig fields, including nested
    dicts for sub-configs (e.g. cpu={'cores': 6}).

    Returns:
        JetsonOrinNXConfig instance with defaults replaced by any provided overrides.
    """
    defaults: Dict[str, Any] = {
        "board_name": "Jetson Orin NX 16GB",
        "cpu": OrinNXCPUConfig(),
        "gpu": OrinNXGPUConfig(),
        "memory": OrinNXMemoryConfig(),
        "storage": OrinNXStorageConfig(),
        "power": OrinNXPowerConfig(),
        "ai": OrinNXAIConfig(),
        "pin_map": OrinNXPinMapping(),
        "jetpack_version": "5.1",
    }

    for key, value in overrides.items():
        if key in defaults and isinstance(value, dict):
            current = defaults[key]
            defaults[key] = replace(current, **value)
        else:
            defaults[key] = value

    return JetsonOrinNXConfig(**defaults)
