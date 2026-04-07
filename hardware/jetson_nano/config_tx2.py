"""
Jetson TX2 Hardware Configuration for NEXUS Marine Robotics.

Defines hardware dataclasses for the Jetson TX2 module with NVIDIA Pascal GPU,
targeting marine robotics real-time computer vision workloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict


@dataclass(frozen=True)
class TX2CPUConfig:
    """CPU configuration for Jetson TX2."""
    arch: str = "ARM"
    cores: int = 6
    core_type: str = "Cortex-A57"
    clock_ghz: float = 2.0


@dataclass(frozen=True)
class TX2GPUConfig:
    """GPU configuration for Jetson TX2 — NVIDIA Pascal."""
    name: str = "Pascal"
    cuda_cores: int = 256
    tflops: float = 1.3
    tensor_cores: int = 0


@dataclass(frozen=True)
class TX2MemoryConfig:
    """Memory configuration for Jetson TX2."""
    ram_gb: int = 8
    bandwidth_gbps: float = 51.2
    type: str = "LPDDR4"


@dataclass(frozen=True)
class TX2StorageConfig:
    """Storage configuration for Jetson TX2."""
    boot: str = "eMMC 32GB"
    usb3_count: int = 3
    sata_support: bool = True


@dataclass(frozen=True)
class TX2PowerConfig:
    """Power management configuration for Jetson TX2."""
    max_watts: int = 15
    thermal_throttle_c: int = 90
    fan_curve: Dict[int, int] = field(
        default_factory=lambda: {40: 0, 55: 30, 75: 70, 85: 100}
    )
    power_modes: Dict[str, int] = field(
        default_factory=lambda: {
            "MAXN": 15,
            "MAXQ": 7,
            "MAXQ_CORE_ALL_OFF": 5,
            "MAXP_CORE_ALL_OFF": 6,
        }
    )


@dataclass(frozen=True)
class TX2AIConfig:
    """AI / inference configuration for Jetson TX2 — real-time CV focus."""
    tensorrt_precision: str = "FP16"
    max_batch: int = 8
    inference_threads: int = 6
    max_resolution: int = 2560
    marine_cv_optimized: bool = True
    target_fps: int = 30


@dataclass(frozen=True)
class TX2PinMapping:
    """GPIO / bus pin mapping for NEXUS marine peripherals on Jetson TX2."""
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
    WATER_TEMP: int = 19


@dataclass
class JetsonTX2Config:
    """
    Top-level Jetson TX2 board configuration.

    Composes all sub-configurations and holds board-level metadata.
    Optimized for marine robotics real-time computer vision.
    """
    board_name: str = "Jetson TX2"
    cpu: TX2CPUConfig = field(default_factory=TX2CPUConfig)
    gpu: TX2GPUConfig = field(default_factory=TX2GPUConfig)
    memory: TX2MemoryConfig = field(default_factory=TX2MemoryConfig)
    storage: TX2StorageConfig = field(default_factory=TX2StorageConfig)
    power: TX2PowerConfig = field(default_factory=TX2PowerConfig)
    ai: TX2AIConfig = field(default_factory=TX2AIConfig)
    pin_map: TX2PinMapping = field(default_factory=TX2PinMapping)
    jetpack_version: str = "4.6"


def create_jetson_tx2_config(**overrides: Any) -> JetsonTX2Config:
    """
    Factory function that creates a JetsonTX2Config with optional overrides.

    Supported override keys match JetsonTX2Config fields, including nested
    dicts for sub-configs (e.g. cpu={'cores': 4}).

    Returns:
        JetsonTX2Config instance with defaults replaced by any provided overrides.
    """
    defaults: Dict[str, Any] = {
        "board_name": "Jetson TX2",
        "cpu": TX2CPUConfig(),
        "gpu": TX2GPUConfig(),
        "memory": TX2MemoryConfig(),
        "storage": TX2StorageConfig(),
        "power": TX2PowerConfig(),
        "ai": TX2AIConfig(),
        "pin_map": TX2PinMapping(),
        "jetpack_version": "4.6",
    }

    for key, value in overrides.items():
        if key in defaults and isinstance(value, dict):
            current = defaults[key]
            defaults[key] = replace(current, **value)
        else:
            defaults[key] = value

    return JetsonTX2Config(**defaults)
