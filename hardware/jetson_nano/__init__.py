"""
NEXUS Marine Robotics - Jetson Hardware Configuration Package

Provides hardware configuration dataclasses and factory functions
for all NVIDIA Jetson platforms: Nano, TX2, Xavier NX, Orin Nano,
Orin NX, and AGX Orin.
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
from .config_tx2 import (
    TX2CPUConfig,
    TX2GPUConfig,
    TX2MemoryConfig,
    TX2StorageConfig,
    TX2PowerConfig,
    TX2AIConfig,
    TX2PinMapping,
    JetsonTX2Config,
    create_jetson_tx2_config,
)
from .config_xavier_nx import (
    XavierNXCPUConfig,
    XavierNXGPUConfig,
    XavierNXMemoryConfig,
    XavierNXStorageConfig,
    XavierNXPowerConfig,
    XavierNXAIConfig,
    XavierNXPinMapping,
    JetsonXavierNXConfig,
    create_jetson_xavier_nx_config,
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
from .config_orin_nx import (
    OrinNXCPUConfig,
    OrinNXGPUConfig,
    OrinNXMemoryConfig,
    OrinNXStorageConfig,
    OrinNXPowerConfig,
    OrinNXAIConfig,
    OrinNXPinMapping,
    JetsonOrinNXConfig,
    create_jetson_orin_nx_config,
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

# Board registry mapping board names to their config classes and factory functions
BOARD_REGISTRY: dict = {
    "jetson-nano": {
        "config_class": JetsonConfig,
        "factory": create_jetson_nano_config,
        "description": "Jetson Nano 4GB — Maxwell 128 CUDA, 4GB LPDDR4",
    },
    "jetson-tx2": {
        "config_class": JetsonTX2Config,
        "factory": create_jetson_tx2_config,
        "description": "Jetson TX2 — Pascal 256 CUDA, 8GB LPDDR4, marine CV",
    },
    "jetson-xavier-nx": {
        "config_class": JetsonXavierNXConfig,
        "factory": create_jetson_xavier_nx_config,
        "description": "Jetson Xavier NX — Volta 384 CUDA, 8GB LPDDR4x, DL inference",
    },
    "jetson-orin-nano": {
        "config_class": JetsonOrinNanoConfig,
        "factory": create_jetson_orin_nano_config,
        "description": "Jetson Orin Nano 8GB — Ampere 1024 CUDA, 8GB LPDDR5",
    },
    "jetson-orin-nx": {
        "config_class": JetsonOrinNXConfig,
        "factory": create_jetson_orin_nx_config,
        "description": "Jetson Orin NX 16GB — Ampere 1024 CUDA, 16GB LPDDR5, fleet commander",
    },
    "jetson-agx-orin": {
        "config_class": JetsonAGXOrinConfig,
        "factory": create_jetson_agx_orin_config,
        "description": "Jetson AGX Orin 64GB — Ampere 2048 CUDA, 64GB LPDDR5, flagship",
    },
}


def list_supported_boards() -> list:
    """Return a list of all supported Jetson board names."""
    return list(BOARD_REGISTRY.keys())


def get_board_info(board_name: str) -> dict:
    """
    Return registry info for a given Jetson board name.

    Raises:
        ValueError: If board_name is not in the registry.
    """
    if board_name not in BOARD_REGISTRY:
        raise ValueError(
            f"Unknown board '{board_name}'. "
            f"Supported: {', '.join(sorted(BOARD_REGISTRY.keys()))}"
        )
    return BOARD_REGISTRY[board_name]


__all__ = [
    "BOARD_REGISTRY",
    "list_supported_boards",
    "get_board_info",
    # Jetson Nano
    "CPUConfig", "GPUConfig", "MemoryConfig", "StorageConfig",
    "PowerConfig", "AIConfig", "PinMapping", "JetsonConfig",
    "create_jetson_nano_config",
    # Jetson TX2
    "TX2CPUConfig", "TX2GPUConfig", "TX2MemoryConfig",
    "TX2StorageConfig", "TX2PowerConfig", "TX2AIConfig",
    "TX2PinMapping", "JetsonTX2Config", "create_jetson_tx2_config",
    # Jetson Xavier NX
    "XavierNXCPUConfig", "XavierNXGPUConfig", "XavierNXMemoryConfig",
    "XavierNXStorageConfig", "XavierNXPowerConfig", "XavierNXAIConfig",
    "XavierNXPinMapping", "JetsonXavierNXConfig",
    "create_jetson_xavier_nx_config",
    # Jetson Orin Nano
    "OrinNanoCPUConfig", "OrinNanoGPUConfig", "OrinNanoMemoryConfig",
    "OrinNanoStorageConfig", "OrinNanoPowerConfig", "OrinNanoAIConfig",
    "OrinNanoPinMapping", "JetsonOrinNanoConfig",
    "create_jetson_orin_nano_config",
    # Jetson Orin NX
    "OrinNXCPUConfig", "OrinNXGPUConfig", "OrinNXMemoryConfig",
    "OrinNXStorageConfig", "OrinNXPowerConfig", "OrinNXAIConfig",
    "OrinNXPinMapping", "JetsonOrinNXConfig",
    "create_jetson_orin_nx_config",
    # Jetson AGX Orin
    "AGXCPUConfig", "AGXGPUConfig", "AGXMemoryConfig", "AGXStorageConfig",
    "AGXPowerConfig", "AGXAIConfig", "AGXPinMapping", "JetsonAGXOrinConfig",
    "create_jetson_agx_orin_config",
    # AI Pipeline
    "ObjectDetector", "CameraConfig", "ModelConfig",
    "PerceptionPipeline", "get_pipeline_profile",
]
