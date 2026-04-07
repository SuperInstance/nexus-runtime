"""
NEXUS Marine Robotics - BeagleBone Hardware Configuration Package

Provides hardware configuration dataclasses and factory functions
for BeagleBone Black, BeagleBone AI-64, and PRU cape management.
"""

from .config_black import (
    BlackCPUConfig,
    BlackMemoryConfig,
    BlackStorageConfig,
    BlackPowerConfig,
    BlackGPIOConfig,
    BeagleBoneBlackConfig,
    create_beaglebone_black_config,
)
from .config_ai64 import (
    AI64CPUConfig,
    AI64DSPConfig,
    AI64MemoryConfig,
    AI64StorageConfig,
    AI64PowerConfig,
    AI64GPIOConfig,
    BeagleBoneAI64Config,
    create_beaglebone_ai64_config,
)
from .pru_config import (
    PRUMode,
    PRUCoreConfig,
    PRUSharedMemory,
    PRUControllerConfig,
    MotorChannelConfig,
    create_pru_controller_config,
)
from .cape_manager import (
    CapeInfo,
    CapeManager,
    CapeSlot,
    create_cape_manager,
)

# Board registry mapping board names to their config classes and factory functions
BOARD_REGISTRY: dict = {
    "beaglebone-black": {
        "config_class": "BeagleBoneBlackConfig",
        "factory": "create_beaglebone_black_config",
        "description": "BeagleBone Black — AM3358 Cortex-A8, 512MB DDR3, PRU real-time",
    },
    "beaglebone-ai64": {
        "config_class": "BeagleBoneAI64Config",
        "factory": "create_beaglebone_ai64_config",
        "description": "BeagleBone AI-64 — AM5729 dual A15 + C66x DSPs, 2GB DDR3",
    },
}


def list_supported_boards() -> list:
    """Return a list of all supported BeagleBone board names."""
    return list(BOARD_REGISTRY.keys())


def get_board_info(board_name: str) -> dict:
    """
    Return registry info for a given board name.

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
    # BeagleBone Black
    "BlackCPUConfig", "BlackMemoryConfig", "BlackStorageConfig",
    "BlackPowerConfig", "BlackGPIOConfig", "BeagleBoneBlackConfig",
    "create_beaglebone_black_config",
    # BeagleBone AI-64
    "AI64CPUConfig", "AI64DSPConfig", "AI64MemoryConfig",
    "AI64StorageConfig", "AI64PowerConfig", "AI64GPIOConfig",
    "BeagleBoneAI64Config", "create_beaglebone_ai64_config",
    # PRU
    "PRUMode", "PRUCoreConfig", "PRUSharedMemory",
    "PRUControllerConfig", "MotorChannelConfig",
    "create_pru_controller_config",
    # Cape Manager
    "CapeInfo", "CapeManager", "CapeSlot", "create_cape_manager",
]
