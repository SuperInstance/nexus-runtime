"""
NEXUS nRF52840 Hardware Configuration Module

Provides chip configuration, memory layout, pin mapping, and protocol
setup for the Nordic Semiconductor nRF52840 system-on-chip used in
NEXUS marine robotics wireless sensor nodes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# nRF52840 Hardware Constants
# ---------------------------------------------------------------------------

CORE_COUNT = 1                         # Single Cortex-M4
CPU_FREQ_MAX = 64_000_000              # 64 MHz
CPU_FREQ_DEFAULT = 64_000_000          # 64 MHz (only speed)
FLASH_TOTAL = 1_048_576                # 1 MB
RAM_TOTAL = 256 * 1024                 # 256 KB
GPIO_COUNT = 48                        # P0.00-P0.31 + P1.00-P1.15
ADC_RESOLUTION = 12                    # 12-bit SAADC
ADC_CHANNELS = 8                       # 8 SAADC channels
RTC_COUNT = 3                          # RTC0, RTC1, RTC2
TIMER_COUNT = 5                        # TIMER0-TIMER4
PWM_COUNT = 4                          # PWM0-PWM3
SPI_COUNT = 4                          # SPIM0-SPIM3
I2C_COUNT = 4                          # TWIM0-TWIM3
UART_COUNT = 2                         # UARTE0, UARTE1
RADIO_BLE_5 = True                     # BLE 5.0 capable
RADIO_ZIGBEE = True                    # IEEE 802.15.4 capable
RADIO_THREAD = True                    # OpenThread capable
NFC_AVAILABLE = True                   # Built-in NFC-A

# Peripheral base addresses (nRF52840)
PERIPHERAL_BASE = 0x40000000
RAM_BASE = 0x20000000
FLASH_BASE = 0x00000000
CODE_PAGE_SIZE = 4096                  # 4 KB flash page size
CODE_PAGES = FLASH_TOTAL // CODE_PAGE_SIZE  # 256 pages


class GPIOPort(IntEnum):
    """nRF52840 GPIO port identifiers."""
    PORT0 = 0
    PORT1 = 1


class PinDrive(IntEnum):
    """GPIO drive strength settings."""
    S0S1 = 0      # Standard 0, Standard 1
    H0S1 = 1      # High drive 0, Standard 1
    S0H1 = 2      # Standard 0, High drive 1
    H0H1 = 3      # High drive 0, High drive 1
    D0S1 = 4      # Disconnect 0, Standard 1
    D0H1 = 5      # Disconnect 0, High drive 1
    S0D1 = 6      # Standard 0, Disconnect 1
    DISCONNECTED = 7


class PinPull(IntEnum):
    """GPIO pull resistor configuration."""
    NONE = 0
    PULLDOWN = 1
    PULLUP = 2
    BUSKEEPER = 3


class PinFunction(IntEnum):
    """Marine sensor function identifiers for nRF52840."""
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
    ADC_CONDUCTIVITY = 42
    ADC_PH = 43
    PWM_THRUSTER = 50
    PWM_AUX = 51
    NFC_ANTENNA = 60
    GPS_PPS = 70


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PinConfig:
    """Configuration for a single GPIO pin on the nRF52840."""
    function: PinFunction
    port: GPIOPort
    pin: int
    direction: str = "input"            # input, output
    pull: PinPull = PinPull.NONE
    drive: PinDrive = PinDrive.S0S1
    sense: str = "none"                # none, high, low, both
    description: str = ""

    def __post_init__(self):
        max_pin = 31 if self.port == GPIOPort.PORT0 else 15
        if not (0 <= self.pin <= max_pin):
            raise ValueError(
                f"Pin {self.pin} out of range for {self.port.name} (0-{max_pin})."
            )

    @property
    def absolute_pin(self) -> int:
        return self.port * 32 + self.pin

    def hardware_address(self) -> int:
        return PERIPHERAL_BASE + 0x500 + self.absolute_pin * 4


@dataclass
class MemoryRegion:
    """Describes an SRAM or Flash allocation region."""
    name: str
    start: int
    size_bytes: int
    type: str = "ram"                  # ram, flash
    purpose: str = ""

    @property
    def end(self) -> int:
        return self.start + self.size_bytes


@dataclass
class MemoryLayout:
    """Memory layout for the nRF52840."""
    regions: List[MemoryRegion] = field(default_factory=list)

    def add_region(self, name: str, start: int, size: int,
                   type: str = "ram", purpose: str = ""):
        self.regions.append(
            MemoryRegion(name=name, start=start, size_bytes=size, type=type, purpose=purpose)
        )

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
    """Radio protocol configuration."""
    ble_enabled: bool = True
    zigbee_enabled: bool = False
    thread_enabled: bool = False
    nfc_enabled: bool = False
    ble_max_conn: int = 20
    ble_mtu: int = 247
    ble_min_conn_interval_ms: float = 7.5
    ble_max_conn_interval_ms: float = 4000.0
    ble_tx_power_dbm: int = 0          # range: -20 to +8
    radio_mode: str = "ble"            # ble, zigbee, thread

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.ble_max_conn < 1 or self.ble_max_conn > 20:
            errors.append(f"BLE max connections must be 1-20, got {self.ble_max_conn}.")
        if self.ble_mtu < 23 or self.ble_mtu > 517:
            errors.append(f"BLE MTU must be 23-517, got {self.ble_mtu}.")
        if self.ble_tx_power_dbm < -20 or self.ble_tx_power_dbm > 8:
            errors.append(f"TX power must be -20..+8 dBm, got {self.ble_tx_power_dbm}.")
        protocols = sum([self.ble_enabled, self.zigbee_enabled, self.thread_enabled])
        if protocols > 1:
            errors.append("Only one radio protocol can be active at a time.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


# ---------------------------------------------------------------------------
# Main Configuration Class
# ---------------------------------------------------------------------------

class NRF52840Config:
    """
    Top-level hardware configuration for the NEXUS nRF52840 marine wireless node.

    Manages GPIO pin mapping, memory layout, radio protocol selection, and
    peripheral allocation for BLE-based marine sensor telemetry.
    """

    # Default pin mapping for NEXUS marine sensor node
    DEFAULT_PINS: Dict[PinFunction, Tuple[int, int]] = {
        # (port, pin_number)
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
        PinFunction.ADC_CONDUCTIVITY: (0, 4),
        PinFunction.ADC_PH: (0, 5),
        PinFunction.PWM_THRUSTER: (0, 16),
        PinFunction.PWM_AUX: (0, 15),
        PinFunction.NFC_ANTENNA: (0, 9),
        PinFunction.GPS_PPS: (1, 0),
    }

    def __init__(self):
        self._pin_map: Dict[PinFunction, PinConfig] = {}
        self.protocol = ProtocolConfig()
        self.memory = MemoryLayout()
        self._peripherals_used: Set[str] = set()
        self._configure_default_pins()

    # -- Pin Mapping ---------------------------------------------------------

    def _configure_default_pins(self):
        for func, (port, pin) in self.DEFAULT_PINS.items():
            self._pin_map[func] = PinConfig(
                function=func,
                port=GPIOPort(port),
                pin=pin,
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
                function=function,
                port=GPIOPort(port),
                pin=pin,
                description=kwargs.pop("description", old.description),
                **kwargs,
            )
        else:
            self._pin_map[function] = PinConfig(
                function=function, port=GPIOPort(port), pin=pin, **kwargs
            )

    def all_pins(self) -> List[PinConfig]:
        return list(self._pin_map.values())

    def used_pins_set(self) -> Set[int]:
        return {pc.absolute_pin for pc in self._pin_map.values()}

    # -- Convenience properties ----------------------------------------------

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

    @property
    def PIN_ADC_TEMP(self) -> PinConfig:
        return self.get_pin(PinFunction.ADC_TEMP_SENSE)

    @property
    def PIN_NFC(self) -> PinConfig:
        return self.get_pin(PinFunction.NFC_ANTENNA)

    # -- Protocol Configuration ----------------------------------------------

    def configure_ble(self, max_conn: int = 20, mtu: int = 247, tx_power: int = 0):
        self.protocol.ble_enabled = True
        self.protocol.zigbee_enabled = False
        self.protocol.thread_enabled = False
        self.protocol.radio_mode = "ble"
        self.protocol.ble_max_conn = max_conn
        self.protocol.ble_mtu = mtu
        self.protocol.ble_tx_power_dbm = tx_power

    def configure_zigbee(self):
        self.protocol.ble_enabled = False
        self.protocol.zigbee_enabled = True
        self.protocol.thread_enabled = False
        self.protocol.radio_mode = "zigbee"

    def configure_thread(self):
        self.protocol.ble_enabled = False
        self.protocol.zigbee_enabled = False
        self.protocol.thread_enabled = True
        self.protocol.radio_mode = "thread"

    # -- Memory Layout -------------------------------------------------------

    def configure_marine_node(self):
        """Set up default memory regions for a NEXUS marine sensor node."""
        # RAM regions
        self.memory.add_region("stack", RAM_BASE, 8192, "ram", "Main stack")
        self.memory.add_region("heap", RAM_BASE + 8192, 32768, "ram", "Application heap")
        self.memory.add_region("ble_stack", RAM_BASE + 40960, 65536, "ram", "SoftDevice BLE stack")
        self.memory.add_region("sensor_buf", RAM_BASE + 106496, 16384, "ram", "Sensor data buffer")
        self.memory.add_region("uart_buf", RAM_BASE + 122880, 4096, "ram", "UART NMEA buffer")
        self.memory.add_region("adc_buf", RAM_BASE + 126976, 2048, "ram", "ADC sample buffer")
        # Flash regions
        self.memory.add_region("softdevice", FLASH_BASE, 152 * CODE_PAGE_SIZE, "flash", "Nordic SoftDevice")
        self.memory.add_region("application", FLASH_BASE + 152 * CODE_PAGE_SIZE, 96 * CODE_PAGE_SIZE, "flash", "Application firmware")
        self.memory.add_region("settings", FLASH_BASE + 248 * CODE_PAGE_SIZE, 8 * CODE_PAGE_SIZE, "flash", "NVS / settings")

    # -- Peripheral Management -----------------------------------------------

    def allocate_peripheral(self, name: str, periph_type: str, instance: int = 0) -> str:
        """Allocate a peripheral (e.g., SPI0, TWIM1) and return its resource key."""
        resource_key = f"{periph_type}{instance}"
        if resource_key in self._peripherals_used:
            raise ValueError(f"Peripheral {resource_key} already allocated.")
        self._peripherals_used.add(resource_key)
        return resource_key

    def release_peripheral(self, resource_key: str):
        self._peripherals_used.discard(resource_key)

    def allocated_peripherals(self) -> Set[str]:
        return set(self._peripherals_used)

    # -- Validation ----------------------------------------------------------

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

    # -- Summary -------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "chip": "nRF52840",
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
