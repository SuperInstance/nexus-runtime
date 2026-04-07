"""
NEXUS i.MX RT1050 Hardware Configuration Module

NXP i.MX RT1050 — Cortex-M7 @ 600 MHz with 512 KB SRAM.
Cost-optimized member of the i.MX RT family for mid-range
marine controller applications.

Key differences from RT1060:
  - 512 KB SRAM (vs 1 MB)
  - 1x ENET (vs 2x)
  - No CAN-FD (CAN 2.0 only)
  - Lower cost
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple


CORE_COUNT = 1
CPU_FREQ_MAX = 600_000_000
CPU_FREQ_DEFAULT = 600_000_000
SRAM_TOTAL = 512 * 1024               # 512 KB
FLASH_TOTAL = 0
ADC_RESOLUTION = 12
ADC_CHANNELS = 4
PWM_MODULES = 2
PWM_SUBMODULES = 8
UART_COUNT = 8
SPI_COUNT = 4
I2C_COUNT = 4
CAN_FD_COUNT = 1                      # CAN 2.0 only
ENET_COUNT = 1
USB_COUNT = 2
FLEXIO_COUNT = 2


class GPIOPin(IntEnum):
    GPIO_AD_B0_00 = 0
    GPIO_AD_B0_01 = 1
    GPIO_AD_B0_02 = 2
    GPIO_AD_B0_03 = 3
    GPIO_AD_B0_04 = 4
    GPIO_AD_B0_05 = 5
    GPIO_AD_B0_06 = 6
    GPIO_AD_B0_07 = 7
    GPIO_AD_B0_08 = 8
    GPIO_AD_B0_09 = 9
    GPIO_AD_B0_10 = 10
    GPIO_AD_B0_11 = 11
    GPIO_AD_B0_12 = 12
    GPIO_AD_B0_13 = 13
    GPIO_B0_00 = 20
    GPIO_B0_01 = 21
    GPIO_B0_02 = 22
    GPIO_B0_03 = 23
    GPIO_EMC_00 = 40


class PinFunction(IntEnum):
    I2C_SDA = 10
    I2C_SCL = 11
    SPI_MOSI = 20
    SPI_MISO = 21
    SPI_CLK = 22
    SPI_CS = 23
    UART_TX = 30
    UART_RX = 31
    ADC_DEPTH = 40
    ADC_TEMP = 41
    STATUS_LED = 80


@dataclass
class MemoryRegion:
    name: str
    start: int
    size_bytes: int
    type: str = "ram"
    purpose: str = ""

    @property
    def end(self) -> int:
        return self.start + self.size_bytes


@dataclass
class MemoryLayout:
    regions: List[MemoryRegion] = field(default_factory=list)

    def add_region(self, name: str, start: int, size: int, type: str = "ram", purpose: str = ""):
        self.regions.append(MemoryRegion(name=name, start=start, size_bytes=size, type=type, purpose=purpose))

    def total_allocated(self, type: str = "ram") -> int:
        return sum(r.size_bytes for r in self.regions if r.type == type)

    def fits_in_sram(self) -> bool:
        return self.total_allocated("ram") <= SRAM_TOTAL

    def region_by_name(self, name: str) -> Optional[MemoryRegion]:
        for r in self.regions:
            if r.name == name:
                return r
        return None


@dataclass
class ClockConfig:
    arm_freq_hz: int = CPU_FREQ_DEFAULT
    ahb_freq_hz: int = 150_000_000
    ipg_freq_hz: int = 75_000_000

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.arm_freq_hz > CPU_FREQ_MAX:
            errors.append(f"ARM freq {self.arm_freq_hz} exceeds max {CPU_FREQ_MAX}.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


class IMXRT1050Config:
    """
    Complete hardware configuration for the NXP i.MX RT1050.

    Cost-optimized Cortex-M7 @ 600 MHz with 512 KB SRAM for
    mid-range marine controller deployments.
    """

    DEFAULT_PINS: Dict[PinFunction, int] = {
        PinFunction.I2C_SDA: GPIOPin.GPIO_AD_B0_12,
        PinFunction.I2C_SCL: GPIOPin.GPIO_AD_B0_13,
        PinFunction.SPI_MOSI: GPIOPin.GPIO_AD_B0_04,
        PinFunction.SPI_MISO: GPIOPin.GPIO_AD_B0_05,
        PinFunction.SPI_CLK: GPIOPin.GPIO_AD_B0_06,
        PinFunction.SPI_CS: GPIOPin.GPIO_AD_B0_07,
        PinFunction.UART_TX: GPIOPin.GPIO_AD_B0_08,
        PinFunction.UART_RX: GPIOPin.GPIO_AD_B0_09,
        PinFunction.ADC_DEPTH: GPIOPin.GPIO_AD_B0_10,
        PinFunction.ADC_TEMP: GPIOPin.GPIO_AD_B0_11,
        PinFunction.STATUS_LED: GPIOPin.GPIO_AD_B0_03,
    }

    def __init__(self):
        self._pin_map: Dict[PinFunction, int] = {}
        self.clock = ClockConfig()
        self.memory = MemoryLayout()
        self._peripherals_used: Set[str] = set()
        self._configure_default_pins()

    def _configure_default_pins(self):
        self._pin_map = dict(self.DEFAULT_PINS)

    def get_pin(self, function: PinFunction) -> int:
        if function not in self._pin_map:
            raise KeyError(f"Pin function {function.name} not configured.")
        return self._pin_map[function]

    def remap_pin(self, function: PinFunction, pin: int):
        self._pin_map[function] = pin

    def all_pins(self) -> Dict[PinFunction, int]:
        return dict(self._pin_map)

    def used_gpio_set(self) -> set:
        return set(self._pin_map.values())

    def allocate_peripheral(self, name: str, periph_type: str, instance: int = 0) -> str:
        resource_key = f"{periph_type}{instance}"
        if resource_key in self._peripherals_used:
            raise ValueError(f"Peripheral {resource_key} already allocated.")
        self._peripherals_used.add(resource_key)
        return resource_key

    def release_peripheral(self, resource_key: str):
        self._peripherals_used.discard(resource_key)

    def allocated_peripherals(self) -> Set[str]:
        return set(self._peripherals_used)

    def configure_marine_controller(self):
        self.memory.add_region("dtcm", 0x20000000, 128 * 1024, "ram", "Data TCM")
        self.memory.add_region("itcm", 0x00000000, 64 * 1024, "ram", "Instruction TCM")
        self.memory.add_region("ocram", 0x20200000, 256 * 1024, "ram", "On-chip RAM")
        self.memory.add_region("adc_dma", 0x20240000, 16 * 1024, "ram", "ADC DMA buffer")
        self.memory.add_region("eth_buf", 0x20244000, 32 * 1024, "ram", "Ethernet buffers")

    def validate(self) -> List[str]:
        errors: List[str] = []
        errors.extend(self.clock.validate())
        if not self.memory.fits_in_sram():
            errors.append(f"SRAM overflow: {self.memory.total_allocated('ram')} > {SRAM_TOTAL}.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def summary(self) -> dict:
        return {
            "chip": "i.MX RT1050",
            "cpu": "Cortex-M7",
            "cpu_freq_hz": self.clock.arm_freq_hz,
            "sram_total_bytes": SRAM_TOTAL,
            "sram_allocated_bytes": self.memory.total_allocated(),
            "adc_channels": ADC_CHANNELS,
            "uart_count": UART_COUNT,
            "can_fd_count": CAN_FD_COUNT,
            "enet_count": ENET_COUNT,
            "flexio_count": FLEXIO_COUNT,
            "usb_count": USB_COUNT,
            "pins_configured": len(self._pin_map),
            "peripherals_allocated": list(self._peripherals_used),
        }
