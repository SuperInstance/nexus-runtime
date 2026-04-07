"""
NEXUS i.MX RT1170 Hardware Configuration Module

NXP i.MX RT1170 — Highest performance crossover MCU with dual-core
Cortex-M7 @ 1 GHz + Cortex-M4 @ 400 MHz, 3.5 MB total SRAM.

Key features:
  - Cortex-M7 @ 1 GHz (2475 DMIPS, 4.17 CoreMark/MHz) + Cortex-M4 @ 400 MHz
  - 3.5 MB total SRAM (512 KB TCM per core + 2.5 MB shared OCRAM)
  - Cortex-M7 FPU with double-precision
  - 2x FlexSPI (Octal SPI support)
  - 2x FlexCAN (CAN-FD 2.0)
  - 3x ENET (2x 10/100 + 1x 10/100 with TSN)
  - 2x USB 2.0 OTG (HS)
  - 3x ADC (2x 12-bit, 1x 2x12-bit delta-sigma)
  - Hardware JPEG codec, GPU 2D, PXP (pixel pipeline)
  - Hardware security (CAAM, SNVS, DCU)
  - LQFP144, MAPBGA516 packages
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple


# Application core (Cortex-M7)
APP_CORE_COUNT = 1
APP_CORE_FREQ_MAX = 1_000_000_000        # 1 GHz
APP_CORE_FREQ_DEFAULT = 996_000_000      # 996 MHz common config
APP_ITCM_SIZE = 256 * 1024               # 256 KB
APP_DTCM_SIZE = 256 * 1024               # 256 KB

# Network/Secondary core (Cortex-M4)
NET_CORE_COUNT = 1
NET_CORE_FREQ_MAX = 400_000_000          # 400 MHz
NET_CORE_FREQ_DEFAULT = 400_000_000
NET_TCM_SIZE = 128 * 1024                # 128 KB

# Shared
CORE_COUNT = APP_CORE_COUNT + NET_CORE_COUNT
SHARED_OCRAM_SIZE = 2560 * 1024          # 2.5 MB
APP_SRAM_TOTAL = APP_ITCM_SIZE + APP_DTCM_SIZE
NET_SRAM_TOTAL = NET_TCM_SIZE
SRAM_TOTAL = APP_SRAM_TOTAL + NET_SRAM_TOTAL + SHARED_OCRAM_SIZE
FLASH_TOTAL = 0                          # External via FlexSPI

ADC_RESOLUTION = 12
ADC_CHANNELS = 10
PWM_MODULES = 2
PWM_SUBMODULES = 8
UART_COUNT = 10
SPI_COUNT = 4
I2C_COUNT = 5
CAN_FD_COUNT = 2
ENET_COUNT = 3
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
    GPIO_EMC_01 = 41
    GPIO_SD_B0_00 = 60
    GPIO_SD_B0_01 = 61


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
    PWM_THRUSTER_1 = 50
    PWM_THRUSTER_2 = 51
    STATUS_LED = 80
    USB_DP = 90
    USB_DM = 91


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
    m7_freq_hz: int = APP_CORE_FREQ_DEFAULT
    m4_freq_hz: int = NET_CORE_FREQ_DEFAULT
    ahb_freq_hz: int = 200_000_000
    ipg_freq_hz: int = 100_000_000

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.m7_freq_hz > APP_CORE_FREQ_MAX:
            errors.append(f"M7 freq {self.m7_freq_hz} exceeds max {APP_CORE_FREQ_MAX}.")
        if self.m4_freq_hz > NET_CORE_FREQ_MAX:
            errors.append(f"M4 freq {self.m4_freq_hz} exceeds max {NET_CORE_FREQ_MAX}.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


class IMXRT1170Config:
    """
    Top-level hardware configuration for the NXP i.MX RT1170.

    Dual-core Cortex-M7 @ 1 GHz + Cortex-M4 @ 400 MHz with 3.5 MB SRAM.
    The M7 runs real-time control loops; the M4 handles sensor I/O and BLE.
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

    def configure_marine_autopilot(self):
        """Set up dual-core memory for NEXUS marine autopilot."""
        # M7 core (real-time control)
        self.memory.add_region("m7_itcm", 0x00000000, APP_ITCM_SIZE, "ram", "M7 Instruction TCM")
        self.memory.add_region("m7_dtcm", 0x20000000, APP_DTCM_SIZE, "ram", "M7 Data TCM")
        self.memory.add_region("m7_control", 0x20200000, 512 * 1024, "ram", "M7 control algorithms")
        self.memory.add_region("m7_nav", 0x20280000, 256 * 1024, "ram", "M7 navigation stack")
        # M4 core (sensor I/O)
        self.memory.add_region("m4_tcm", 0x202C0000, NET_TCM_SIZE, "ram", "M4 TCM")
        self.memory.add_region("m4_sensor", 0x202E0000, 128 * 1024, "ram", "M4 sensor processing")
        # Shared OCRAM
        self.memory.add_region("shared_ipc", 0x20300000, 256 * 1024, "ram", "Inter-core communication")
        self.memory.add_region("shared_sensor", 0x20340000, 512 * 1024, "ram", "Shared sensor buffer")
        self.memory.add_region("shared_eth", 0x203C0000, 128 * 1024, "ram", "Ethernet DMA buffers")
        self.memory.add_region("shared_can", 0x203E0000, 64 * 1024, "ram", "CAN-FD buffers")
        self.memory.add_region("shared_log", 0x203F0000, 64 * 1024, "ram", "Mission log buffer")

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
            "chip": "i.MX RT1170",
            "app_cpu": "Cortex-M7 @ 1 GHz",
            "net_cpu": "Cortex-M4 @ 400 MHz",
            "app_cpu_freq_hz": self.clock.m7_freq_hz,
            "net_cpu_freq_hz": self.clock.m4_freq_hz,
            "sram_total_bytes": SRAM_TOTAL,
            "sram_allocated_bytes": self.memory.total_allocated(),
            "adc_channels": ADC_CHANNELS,
            "pwm_modules": PWM_MODULES,
            "uart_count": UART_COUNT,
            "can_fd_count": CAN_FD_COUNT,
            "enet_count": ENET_COUNT,
            "flexio_count": FLEXIO_COUNT,
            "usb_count": USB_COUNT,
            "pins_configured": len(self._pin_map),
            "peripherals_allocated": list(self._peripherals_used),
        }
