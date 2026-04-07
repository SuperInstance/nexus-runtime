"""
NEXUS i.MX RT1064 Hardware Configuration Module

NXP i.MX RT1064 — Cortex-M7 @ 600 MHz with 4 MB onboard flash.
Pin-compatible with RT1060 but adds internal NOR flash, eliminating
the need for external flash in many marine deployment scenarios.

Key differences from RT1060:
  - 4 MB internal NOR flash (vs external only)
  - Pin-compatible with RT1060
  - Simplified BOM for volume production
  - Same 1 MB SRAM, same peripheral set
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .config_imxrt1060 import (
    IMXRT1060Config,
    CORE_COUNT,
    CPU_FREQ_MAX,
    CPU_FREQ_DEFAULT,
    SRAM_TOTAL,
    ADC_CHANNELS,
    PWM_MODULES,
    UART_COUNT,
    CAN_FD_COUNT,
    ENET_COUNT,
    FLEXIO_COUNT,
    USB_COUNT,
    GPIOPin,
    PinFunction,
    MemoryRegion,
    MemoryLayout,
    ClockConfig,
)


FLASH_TOTAL = 4 * 1024 * 1024               # 4 MB internal NOR flash
FLASH_SECTOR_SIZE = 4096                     # 4 KB sector
FLASH_SECTORS = FLASH_TOTAL // FLASH_SECTOR_SIZE  # 1024 sectors


class IMXRT1064Config(IMXRT1060Config):
    """
    i.MX RT1064 configuration extending RT1060 with 4 MB internal flash.

    All RT1060 functionality plus onboard NOR flash for simplified
    marine deployment without external flash chips.
    """

    def __init__(self):
        super().__init__()
        self.board_model = "imxrt1064"
        self.board_name = "NXP i.MX RT1064"
        self.flash_total = FLASH_TOTAL
        self.flash_sector_size = FLASH_SECTOR_SIZE
        self.flash_sectors = FLASH_SECTORS

    def configure_marine_controller(self):
        """Set up SRAM and flash regions for marine controller."""
        super().configure_marine_controller()
        # Internal flash regions
        self.memory.add_region("bootloader", 0x60000000, 32 * FLASH_SECTOR_SIZE, "flash", "Bootloader")
        self.memory.add_region("application", 0x60000000 + 32 * FLASH_SECTOR_SIZE,
                                480 * FLASH_SECTOR_SIZE, "flash", "Application firmware")
        self.memory.add_region("settings", 0x60000000 + 512 * FLASH_SECTOR_SIZE,
                                128 * FLASH_SECTOR_SIZE, "flash", "NVS / settings")
        self.memory.add_region("calibration", 0x60000000 + 640 * FLASH_SECTOR_SIZE,
                                128 * FLASH_SECTOR_SIZE, "flash", "Sensor calibration data")
        self.memory.add_region("log", 0x60000000 + 768 * FLASH_SECTOR_SIZE,
                               256 * FLASH_SECTOR_SIZE, "flash", "Mission log")

    def fits_in_flash(self) -> bool:
        return self.memory.total_allocated("flash") <= FLASH_TOTAL

    def summary(self) -> dict:
        s = super().summary()
        s.update({
            "chip": "i.MX RT1064",
            "flash_total_bytes": FLASH_TOTAL,
            "flash_sectors": FLASH_SECTORS,
            "flash_sector_size": FLASH_SECTOR_SIZE,
        })
        return s
