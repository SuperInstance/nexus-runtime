"""
Jetson Xavier NX Hardware Configuration for NEXUS Marine Robotics.

Defines hardware dataclasses for the Jetson Xavier NX module with Volta GPU
and 6-core Carmel CPU, targeting deep learning inference workloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict


@dataclass(frozen=True)
class XavierNXCPUConfig:
    """CPU configuration for Jetson Xavier NX — 6-core NVIDIA Carmel."""
    arch: str = "ARM"
    cores: int = 6
    core_type: str = "Carmel"
    clock_ghz: float = 1.9


@dataclass(frozen=True)
class XavierNXGPUConfig:
    """GPU configuration for Jetson Xavier NX — NVIDIA Volta."""
    name: str = "Volta"
    cuda_cores: int = 384
    tflops: float = 1.93
    tensor_cores: int = 48
    tops: int = 21


@dataclass(frozen=True)
class XavierNXMemoryConfig:
    """Memory configuration for Jetson Xavier NX."""
    ram_gb: int = 8
    bandwidth_gbps: float = 59.7
    type: str = "LPDDR4x"


@dataclass(frozen=True)
class XavierNXStorageConfig:
    """Storage configuration for Jetson Xavier NX."""
    boot: str = "microSD / NVMe"
    usb3_count: int = 4
    pcie_gen: int = 3


@dataclass(frozen=True)
class XavierNXPowerConfig:
    """Power management configuration for Jetson Xavier NX."""
    max_watts: int = 20
    thermal_throttle_c: int = 95
    fan_curve: Dict[int, int] = field(
        default_factory=lambda: {40: 0, 55: 25, 70: 55, 85: 85, 95: 100}
    )
    power_modes: Dict[str, int] = field(
        default_factory=lambda: {
            "10W": 10,
            "15W": 15,
            "20W": 20,
        }
    )


@dataclass(frozen=True)
class XavierNXAIConfig:
    """AI / inference configuration for Jetson Xavier NX — deep learning focus."""
    tensorrt_precision: str = "FP16"
    max_batch: int = 16
    inference_threads: int = 6
    max_resolution: int = 3840
    dlas_supported: bool = True
    triton_supported: bool = False
    dl_model_cache_mb: int = 512


@dataclass(frozen=True)
class XavierNXPinMapping:
    """GPIO / bus pin mapping for NEXUS marine peripherals on Xavier NX."""
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
class JetsonXavierNXConfig:
    """
    Top-level Jetson Xavier NX board configuration.

    Composes all sub-configurations and holds board-level metadata.
    Optimized for deep learning inference at the edge.
    """
    board_name: str = "Jetson Xavier NX"
    cpu: XavierNXCPUConfig = field(default_factory=XavierNXCPUConfig)
    gpu: XavierNXGPUConfig = field(default_factory=XavierNXGPUConfig)
    memory: XavierNXMemoryConfig = field(default_factory=XavierNXMemoryConfig)
    storage: XavierNXStorageConfig = field(default_factory=XavierNXStorageConfig)
    power: XavierNXPowerConfig = field(default_factory=XavierNXPowerConfig)
    ai: XavierNXAIConfig = field(default_factory=XavierNXAIConfig)
    pin_map: XavierNXPinMapping = field(default_factory=XavierNXPinMapping)
    jetpack_version: str = "5.0"


def create_jetson_xavier_nx_config(**overrides: Any) -> JetsonXavierNXConfig:
    """
    Factory function that creates a JetsonXavierNXConfig with optional overrides.

    Supported override keys match JetsonXavierNXConfig fields, including nested
    dicts for sub-configs (e.g. cpu={'cores': 4}).

    Returns:
        JetsonXavierNXConfig instance with defaults replaced by any provided overrides.
    """
    defaults: Dict[str, Any] = {
        "board_name": "Jetson Xavier NX",
        "cpu": XavierNXCPUConfig(),
        "gpu": XavierNXGPUConfig(),
        "memory": XavierNXMemoryConfig(),
        "storage": XavierNXStorageConfig(),
        "power": XavierNXPowerConfig(),
        "ai": XavierNXAIConfig(),
        "pin_map": XavierNXPinMapping(),
        "jetpack_version": "5.0",
    }

    for key, value in overrides.items():
        if key in defaults and isinstance(value, dict):
            current = defaults[key]
            defaults[key] = replace(current, **value)
        else:
            defaults[key] = value

    return JetsonXavierNXConfig(**defaults)
