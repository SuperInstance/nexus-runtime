"""
NEXUS nRF52810 Hardware Configuration Module

Ultra-low-cost BLE SoC for disposable marine sensor tags and minimal
beacon nodes. Minimal peripherals but excellent power efficiency.

Key differences from nRF52832/52840:
  - 192 KB Flash (smallest in nRF52 family)
  - 24 KB RAM (tightest memory constraint)
  - BLE 4.2
  - No NFC, no USB, no IEEE 802.15.4
  - Only 1 SPI, 1 I2C, 1 UART
  - Lowest cost, smallest footprint
  - Ideal for disposable sensor tags
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple


CORE_COUNT = 1
CPU_FREQ_MAX = 64_000_000
CPU_FREQ_DEFAULT = 64_000_000
FLASH_TOTAL = 192 * 1024               # 192 KB
RAM_TOTAL = 24 * 1024                  # 24 KB
GPIO_COUNT = 32
ADC_RESOLUTION = 12
ADC_CHANNELS = 1
RTC_COUNT = 2
TIMER_COUNT = 3
PWM_COUNT = 1
SPI_COUNT = 1
I2C_COUNT = 1
UART_COUNT = 1
RADIO_BLE_5 = False
NFC_AVAILABLE = False

PERIPHERAL_BASE = 0x40000000
RAM_BASE = 0x20000000
FLASH_BASE = 0x00000000
CODE_PAGE_SIZE = 4096
CODE_PAGES = FLASH_TOTAL // CODE_PAGE_SIZE  # 48 pages


class GPIOPort(IntEnum):
    PORT0 = 0


class PinDrive(IntEnum):
    S0S1 = 0
    H0S1 = 1
    S0H1 = 2
    H0H1 = 3
    D0S1 = 4
    D0H1 = 5
    S0D1 = 6
    DISCONNECTED = 7


class PinPull(IntEnum):
    NONE = 0
    PULLDOWN = 1
    PULLUP = 2
    BUSKEEPER = 3


class PinFunction(IntEnum):
    STATUS_LED = 0
    I2C_SDA = 10
    I2C_SCL = 11
    UART_TX = 20
    UART_RX = 21
    ADC_TEMP_SENSE = 40
    ADC_DEPTH_SENSE = 41


@dataclass
class PinConfig:
    function: PinFunction
    port: GPIOPort
    pin: int
    direction: str = "input"
    pull: PinPull = PinPull.NONE
    drive: PinDrive = PinDrive.S0S1
    description: str = ""

    def __post_init__(self):
        if not (0 <= self.pin <= 31):
            raise ValueError(f"Pin {self.pin} out of range for {self.port.name} (0-31).")

    @property
    def absolute_pin(self) -> int:
        return self.port * 32 + self.pin


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

    def total_ram(self) -> int:
        return self.total_allocated("ram")

    def total_flash(self) -> int:
        return self.total_allocated("flash")

    def fits_in_ram(self) -> bool:
        return self.total_ram() <= RAM_TOTAL

    def fits_in_flash(self) -> bool:
        return self.total_flash() <= FLASH_TOTAL

    def region_by_name(self, name: str) -> Optional[MemoryRegion]:
        for r in self.regions:
            if r.name == name:
                return r
        return None


@dataclass
class ProtocolConfig:
    ble_enabled: bool = True
    ble_max_conn: int = 4              # Very limited on nRF52810
    ble_mtu: int = 23                   # Default MTU only
    ble_tx_power_dbm: int = 0           # range: -20 to +4
    radio_mode: str = "ble"

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.ble_max_conn < 1 or self.ble_max_conn > 4:
            errors.append(f"BLE max connections must be 1-4, got {self.ble_max_conn}.")
        if self.ble_mtu < 23 or self.ble_mtu > 158:
            errors.append(f"BLE MTU must be 23-158, got {self.ble_mtu}.")
        if self.ble_tx_power_dbm < -20 or self.ble_tx_power_dbm > 4:
            errors.append(f"TX power must be -20..+4 dBm, got {self.ble_tx_power_dbm}.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


class NRF52810Config:
    """
    Ultra-low-cost BLE beacon configuration for disposable marine sensor tags.
    Minimal peripherals, 24 KB RAM, 192 KB Flash.
    """

    DEFAULT_PINS: Dict[PinFunction, Tuple[int, int]] = {
        PinFunction.STATUS_LED: (0, 17),
        PinFunction.I2C_SDA: (0, 26),
        PinFunction.I2C_SCL: (0, 27),
        PinFunction.UART_TX: (0, 20),
        PinFunction.UART_RX: (0, 19),
        PinFunction.ADC_TEMP_SENSE: (0, 2),
        PinFunction.ADC_DEPTH_SENSE: (0, 3),
    }

    def __init__(self):
        self._pin_map: Dict[PinFunction, PinConfig] = {}
        self.protocol = ProtocolConfig()
        self.memory = MemoryLayout()
        self._peripherals_used: Set[str] = set()
        self._configure_default_pins()

    def _configure_default_pins(self):
        for func, (port, pin) in self.DEFAULT_PINS.items():
            self._pin_map[func] = PinConfig(
                function=func, port=GPIOPort(port), pin=pin,
                description=f"{func.name} on P{port}.{pin:02d}",
            )

    def get_pin(self, function: PinFunction) -> PinConfig:
        if function not in self._pin_map:
            raise KeyError(f"Pin function {function.name} not configured.")
        return self._pin_map[function]

    def remap_pin(self, function: PinFunction, port: int, pin: int, **kwargs):
        old = self._pin_map.get(function)
        if old:
            self._pin_map[function] = PinConfig(
                function=function, port=GPIOPort(port), pin=pin,
                description=kwargs.pop("description", old.description), **kwargs,
            )
        else:
            self._pin_map[function] = PinConfig(
                function=function, port=GPIOPort(port), pin=pin, **kwargs
            )

    def all_pins(self) -> List[PinConfig]:
        return list(self._pin_map.values())

    def used_pins_set(self) -> Set[int]:
        return {pc.absolute_pin for pc in self._pin_map.values()}

    @property
    def PIN_STATUS_LED(self) -> PinConfig:
        return self.get_pin(PinFunction.STATUS_LED)

    def configure_ble(self, max_conn: int = 4, mtu: int = 23, tx_power: int = 0):
        self.protocol.ble_enabled = True
        self.protocol.radio_mode = "ble"
        self.protocol.ble_max_conn = max_conn
        self.protocol.ble_mtu = mtu
        self.protocol.ble_tx_power_dbm = tx_power

    def configure_sensor_tag(self):
        """Configure memory for minimal BLE sensor tag."""
        self.memory.add_region("stack", RAM_BASE, 2048, "ram", "Stack")
        self.memory.add_region("heap", RAM_BASE + 2048, 4096, "ram", "Heap")
        self.memory.add_region("ble_stack", RAM_BASE + 6144, 12288, "ram", "BLE stack")
        self.memory.add_region("sensor_buf", RAM_BASE + 18432, 2048, "ram", "Sensor buffer")
        self.memory.add_region("softdevice", FLASH_BASE, 34 * CODE_PAGE_SIZE, "flash", "SoftDevice")
        self.memory.add_region("application", FLASH_BASE + 34 * CODE_PAGE_SIZE, 4 * CODE_PAGE_SIZE, "flash", "App")
        self.memory.add_region("settings", FLASH_BASE + 38 * CODE_PAGE_SIZE, 1 * CODE_PAGE_SIZE, "flash", "NVS")

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

    def validate(self) -> List[str]:
        errors: List[str] = []
        errors.extend(self.protocol.validate())
        if not self.memory.fits_in_ram():
            errors.append(f"RAM overflow: {self.memory.total_ram()} > {RAM_TOTAL} bytes.")
        if not self.memory.fits_in_flash():
            errors.append(f"Flash overflow: {self.memory.total_flash()} > {FLASH_TOTAL} bytes.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def summary(self) -> dict:
        return {
            "chip": "nRF52810",
            "cpu": "Cortex-M4",
            "cpu_freq_hz": CPU_FREQ_DEFAULT,
            "flash_bytes": FLASH_TOTAL,
            "ram_bytes": RAM_TOTAL,
            "flash_allocated": self.memory.total_flash(),
            "ram_allocated": self.memory.total_ram(),
            "radio_mode": self.protocol.radio_mode,
            "ble_enabled": self.protocol.ble_enabled,
            "pins_configured": len(self._pin_map),
            "peripherals_allocated": list(self._peripherals_used),
        }
