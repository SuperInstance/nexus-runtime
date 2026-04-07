"""
NEXUS nRF52832 Hardware Configuration Module

Provides chip configuration, memory layout, pin mapping, and protocol
setup for the Nordic Semiconductor nRF52832 system-on-chip used in
NEXUS marine robotics BLE sensor beacons and relay nodes.

Key differences from nRF52840:
  - 512 KB Flash (vs 1 MB)
  - 64 KB RAM (vs 256 KB)
  - BLE 4.2 (vs 5.0)
  - No USB, no NFC
  - No IEEE 802.15.4 (Zigbee/Thread)
  - Lower power consumption
  - Smaller footprint, lower cost
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# nRF52832 Hardware Constants
# ---------------------------------------------------------------------------

CORE_COUNT = 1
CPU_FREQ_MAX = 64_000_000
CPU_FREQ_DEFAULT = 64_000_000
FLASH_TOTAL = 512 * 1024               # 512 KB
RAM_TOTAL = 64 * 1024                  # 64 KB
GPIO_COUNT = 32                         # P0.00-P0.31 only (no Port 1)
ADC_RESOLUTION = 12
ADC_CHANNELS = 1                        # 1 ADC channel (AIN0-P0.02)
RTC_COUNT = 3
TIMER_COUNT = 5
PWM_COUNT = 3
SPI_COUNT = 3
I2C_COUNT = 2
UART_COUNT = 1
RADIO_BLE_5 = False
RADIO_ZIGBEE = False
RADIO_THREAD = False
NFC_AVAILABLE = False

PERIPHERAL_BASE = 0x40000000
RAM_BASE = 0x20000000
FLASH_BASE = 0x00000000
CODE_PAGE_SIZE = 4096
CODE_PAGES = FLASH_TOTAL // CODE_PAGE_SIZE  # 128 pages


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
    BLE_ADVERT_LED = 0
    STATUS_LED = 1
    I2C_SDA = 10
    I2C_SCL = 11
    SPI_MOSI = 20
    SPI_MISO = 21
    SPI_CLK = 22
    SPI_CS = 23
    UART_TX = 30
    UART_RX = 31
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
    sense: str = "none"
    description: str = ""

    def __post_init__(self):
        max_pin = 31
        if not (0 <= self.pin <= max_pin):
            raise ValueError(f"Pin {self.pin} out of range for {self.port.name} (0-{max_pin}).")

    @property
    def absolute_pin(self) -> int:
        return self.port * 32 + self.pin

    def hardware_address(self) -> int:
        return PERIPHERAL_BASE + 0x500 + self.absolute_pin * 4


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

    def has_overlap(self, type: str = "ram") -> bool:
        typed = [r for r in self.regions if r.type == type]
        sorted_regions = sorted(typed, key=lambda r: r.start)
        for i in range(len(sorted_regions) - 1):
            if sorted_regions[i].end > sorted_regions[i + 1].start:
                return True
        return False


@dataclass
class ProtocolConfig:
    ble_enabled: bool = True
    ble_max_conn: int = 8              # nRF52832 supports fewer connections
    ble_mtu: int = 247
    ble_min_conn_interval_ms: float = 7.5
    ble_max_conn_interval_ms: float = 4000.0
    ble_tx_power_dbm: int = 0           # range: -20 to +4 (lower than 52840)
    radio_mode: str = "ble"

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.ble_max_conn < 1 or self.ble_max_conn > 8:
            errors.append(f"BLE max connections must be 1-8, got {self.ble_max_conn}.")
        if self.ble_mtu < 23 or self.ble_mtu > 247:
            errors.append(f"BLE MTU must be 23-247, got {self.ble_mtu}.")
        if self.ble_tx_power_dbm < -20 or self.ble_tx_power_dbm > 4:
            errors.append(f"TX power must be -20..+4 dBm, got {self.ble_tx_power_dbm}.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


class NRF52832Config:
    """
    Top-level hardware configuration for the NEXUS nRF52832 marine BLE node.

    Optimized for cost-sensitive sensor beacons and relay nodes with BLE 4.2.
    """

    DEFAULT_PINS: Dict[PinFunction, Tuple[int, int]] = {
        PinFunction.BLE_ADVERT_LED: (0, 13),
        PinFunction.STATUS_LED: (0, 17),
        PinFunction.I2C_SDA: (0, 26),
        PinFunction.I2C_SCL: (0, 27),
        PinFunction.UART_TX: (0, 20),
        PinFunction.UART_RX: (0, 19),
        PinFunction.SPI_MOSI: (0, 24),
        PinFunction.SPI_MISO: (0, 23),
        PinFunction.SPI_CLK: (0, 25),
        PinFunction.SPI_CS: (0, 22),
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
    def PIN_I2C_SDA(self) -> PinConfig:
        return self.get_pin(PinFunction.I2C_SDA)

    @property
    def PIN_I2C_SCL(self) -> PinConfig:
        return self.get_pin(PinFunction.I2C_SCL)

    @property
    def PIN_UART_TX(self) -> PinConfig:
        return self.get_pin(PinFunction.UART_TX)

    @property
    def PIN_UART_RX(self) -> PinConfig:
        return self.get_pin(PinFunction.UART_RX)

    @property
    def PIN_STATUS_LED(self) -> PinConfig:
        return self.get_pin(PinFunction.STATUS_LED)

    def configure_ble(self, max_conn: int = 8, mtu: int = 247, tx_power: int = 0):
        self.protocol.ble_enabled = True
        self.protocol.radio_mode = "ble"
        self.protocol.ble_max_conn = max_conn
        self.protocol.ble_mtu = mtu
        self.protocol.ble_tx_power_dbm = tx_power

    def configure_marine_node(self):
        self.memory.add_region("stack", RAM_BASE, 4096, "ram", "Main stack")
        self.memory.add_region("heap", RAM_BASE + 4096, 16384, "ram", "Application heap")
        self.memory.add_region("ble_stack", RAM_BASE + 20480, 32768, "ram", "SoftDevice BLE stack")
        self.memory.add_region("sensor_buf", RAM_BASE + 53248, 4096, "ram", "Sensor data buffer")
        self.memory.add_region("uart_buf", RAM_BASE + 57344, 2048, "ram", "UART buffer")
        self.memory.add_region("softdevice", FLASH_BASE, 100 * CODE_PAGE_SIZE, "flash", "SoftDevice")
        self.memory.add_region("application", FLASH_BASE + 100 * CODE_PAGE_SIZE, 24 * CODE_PAGE_SIZE, "flash", "App firmware")
        self.memory.add_region("settings", FLASH_BASE + 124 * CODE_PAGE_SIZE, 4 * CODE_PAGE_SIZE, "flash", "NVS")

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
            "chip": "nRF52832",
            "cpu": "Cortex-M4F",
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
