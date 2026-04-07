"""
NEXUS Distributed Intelligence Platform — Raspberry Pi Hardware Configuration Layer.

Provides board descriptions, GPIO pin mappings, and marine sensor interfaces
for Raspberry Pi 4B, Pi 5, Pi Zero 2W, and Compute Module 4 deployed in
marine robotics environments (ROV, AUV, ASV, sensor buoys).
"""

from __future__ import annotations

import os
import platform
import re
from enum import Enum
from typing import Optional

__version__ = "1.0.0"
__platform__ = "raspberry_pi"


class BoardModel(Enum):
    """Supported Raspberry Pi board models."""
    PI_4B = "pi4b"
    PI_5 = "pi5"
    PI_ZERO_2W = "pizerow"
    CM4 = "cm4"


class PeripheralType(Enum):
    """Peripheral interface types."""
    I2C = "i2c"
    SPI = "spi"
    UART = "uart"
    PWM = "pwm"
    GPIO = "gpio"
    CSI = "csi"
    DSI = "dsi"
    PCIe = "pcie"
    USB3 = "usb3"
    ETHERNET = "ethernet"
    WIFI = "wifi"
    BLE = "ble"


# Lazy imports to avoid circular dependencies and heavy module loads
def __getattr__(name: str):
    """Lazy-load board configs on first access."""
    _lazy_modules = {
        "config_pi4": "hardware.raspberry_pi.config_pi4",
        "config_pi5": "hardware.raspberry_pi.config_pi5",
        "config_pizerow": "hardware.raspberry_pi.config_pizerow",
        "sensor_hat": "hardware.raspberry_pi.sensor_hat",
    }
    if name in _lazy_modules:
        import importlib
        return importlib.import_module(_lazy_modules[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def detect_board() -> Optional[BoardModel]:
    """
    Attempt to auto-detect the current Raspberry Pi model.

    Checks:
    1. /proc/device-tree/model (on real hardware)
    2. Environment variable NEXUS_BOARD (for testing / override)
    3. Returns None if detection fails.

    Returns:
        BoardModel or None if undetectable.
    """
    # Allow override for testing in non-Pi environments
    env_board = os.environ.get("NEXUS_BOARD", "").strip().lower()
    if env_board:
        try:
            return BoardModel(env_board)
        except ValueError:
            pass

    # Check device tree on real Raspberry Pi hardware
    model_path = "/proc/device-tree/model"
    if os.path.exists(model_path):
        try:
            with open(model_path, "r", encoding="utf-8", errors="replace") as f:
                model_str = f.read().strip().rstrip("\x00")
        except OSError:
            return None

        model_lower = model_str.lower()

        if "raspberry pi 5" in model_lower:
            return BoardModel.PI_5
        elif "raspberry pi 4" in model_lower:
            return BoardModel.PI_4B
        elif "zero 2" in model_lower:
            return BoardModel.PI_ZERO_2W
        elif "compute module 4" in model_lower:
            return BoardModel.CM4

    # Fallback: check for Raspberry Pi architecture
    if platform.machine() in ("armv7l", "aarch64"):
        # Heuristic: aarch64 on ARM is likely Pi 4B or Pi 5
        if os.path.exists("/sys/firmware/devicetree/base/model"):
            pass  # Already checked above

    return None


def get_config(board: BoardModel):
    """
    Get the hardware configuration object for the specified board.

    Args:
        board: BoardModel enum value.

    Returns:
        Board configuration instance (Pi4Config, Pi5Config, PiZero2WConfig, etc.)

    Raises:
        ValueError: If board model is not supported.
    """
    if board == BoardModel.PI_4B:
        from hardware.raspberry_pi.config_pi4 import Pi4Config
        return Pi4Config()
    elif board == BoardModel.PI_5:
        from hardware.raspberry_pi.config_pi5 import Pi5Config
        return Pi5Config()
    elif board == BoardModel.PI_ZERO_2W:
        from hardware.raspberry_pi.config_pizerow import PiZero2WConfig
        return PiZero2WConfig()
    elif board == BoardModel.CM4:
        # CM4 shares BCM2711 with Pi 4B but has different form factor
        from hardware.raspberry_pi.config_pi4 import Pi4Config, CM4Variant
        return Pi4Config(variant=CM4Variant())
    else:
        raise ValueError(f"Unsupported board model: {board}")


def list_boards() -> list[dict]:
    """
    List all supported board models with summary info.

    Returns:
        List of dicts with board metadata.
    """
    return [
        {"model": "pi4b", "name": "Raspberry Pi 4 Model B", "soc": "BCM2711",
         "cpu": "4x Cortex-A72 @ 1.5GHz", "ram_options": [1, 2, 4, 8],
         "use_case": "Primary compute, vision processing"},
        {"model": "pi5", "name": "Raspberry Pi 5", "soc": "BCM2712",
         "cpu": "4x Cortex-A76 @ 2.4GHz", "ram_options": [4, 8],
         "use_case": "High-performance autonomy, 4K visualization"},
        {"model": "pizerow", "name": "Raspberry Pi Zero 2 W", "soc": "BCM2710",
         "cpu": "4x Cortex-A53 @ 1.0GHz", "ram_options": [512],
         "use_case": "Sensor node, BLE telemetry"},
        {"model": "cm4", "name": "Raspberry Pi Compute Module 4", "soc": "BCM2711",
         "cpu": "4x Cortex-A72 @ 1.5GHz", "ram_options": [1, 2, 4, 8],
         "use_case": "Custom carrier, embedded hull mount"},
    ]


__all__ = [
    "BoardModel",
    "PeripheralType",
    "detect_board",
    "get_config",
    "list_boards",
]
