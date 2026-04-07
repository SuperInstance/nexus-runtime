"""Teensy board configurations for NEXUS marine robotics.

Supported boards: Teensy 3.6, Teensy 4.0, Teensy 4.1
MCU family: NXP i.MX RT series (Cortex-M7)
"""

from hardware.teensy.config_teensy40 import Teensy40Config, create_teensy40
from hardware.teensy.config_teensy41 import Teensy41Config, create_teensy41
from hardware.teensy.config_teensy36 import Teensy36Config, create_teensy36

__all__ = [
    "Teensy40Config", "create_teensy40",
    "Teensy41Config", "create_teensy41",
    "Teensy36Config", "create_teensy36",
]

BOARD_REGISTRY = {
    "teensy36": {"class": Teensy36Config, "display": "Teensy 3.6", "mcu": "MK66FX1M0"},
    "teensy40": {"class": Teensy40Config, "display": "Teensy 4.0", "mcu": "IMXRT1062"},
    "teensy41": {"class": Teensy41Config, "display": "Teensy 4.1", "mcu": "IMXRT1062"},
}

def list_supported_boards() -> list[str]:
    return sorted(BOARD_REGISTRY.keys())
