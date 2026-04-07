"""
NEXUS Marine Robotics - Jetson Hardware Configuration Package

Provides hardware configuration dataclasses and factory functions
for Jetson Nano, Orin Nano, and AGX Orin platforms.
"""

from .config_nano import (
    CPUConfig,
    GPUConfig,
    MemoryConfig,
    StorageConfig,
    PowerConfig,
    AIConfig,
    PinMapping,
    JetsonConfig,
    create_jetson_nano_config,
)
from .config_orin_nano import (
    OrinNanoCPUConfig,
    OrinNanoGPUConfig,
    OrinNanoMemoryConfig,
    OrinNanoStorageConfig,
    OrinNanoPowerConfig,
    OrinNanoAIConfig,
    OrinNanoPinMapping,
    JetsonOrinNanoConfig,
    create_jetson_orin_nano_config,
)
from .config_agx_orin import (
    AGXCPUConfig,
    AGXGPUConfig,
    AGXMemoryConfig,
    AGXStorageConfig,
    AGXPowerConfig,
    AGXAIConfig,
    AGXPinMapping,
    JetsonAGXOrinConfig,
    create_jetson_agx_orin_config,
)
from .ai_pipeline import (
    ObjectDetector,
    CameraConfig,
    ModelConfig,
    PerceptionPipeline,
    get_pipeline_profile,
)

__all__ = [
    "CPUConfig", "GPUConfig", "MemoryConfig", "StorageConfig",
    "PowerConfig", "AIConfig", "PinMapping", "JetsonConfig",
    "create_jetson_nano_config",
    "OrinNanoCPUConfig", "OrinNanoGPUConfig", "OrinNanoMemoryConfig",
    "OrinNanoStorageConfig", "OrinNanoPowerConfig", "OrinNanoAIConfig",
    "OrinNanoPinMapping", "JetsonOrinNanoConfig",
    "create_jetson_orin_nano_config",
    "AGXCPUConfig", "AGXGPUConfig", "AGXMemoryConfig", "AGXStorageConfig",
    "AGXPowerConfig", "AGXAIConfig", "AGXPinMapping", "JetsonAGXOrinConfig",
    "create_jetson_agx_orin_config",
    "ObjectDetector", "CameraConfig", "ModelConfig",
    "PerceptionPipeline", "get_pipeline_profile",
]
