"""
NEXUS nRF5340 Hardware Configuration Module

Nordic Semiconductor\'s most advanced BLE SoC with dual ARM Cortex-M33 cores,
BLE 5.3, and support for LE Audio, Direction Finding, and high-throughput
data streaming for advanced marine sensor networks.

Key features:
  - Dual Cortex-M33: Application core @ 128 MHz + Network core @ 64 MHz
  - 1 MB Flash (application) + 256 KB Flash (network)
  - 512 KB RAM (application) + 64 KB RAM (network)
  - BLE 5.3 with Direction Finding, LE Audio, 2 Mbps PHY
  - IEEE 802.15.4 (Thread/Zigbee) on network core
  - NFC-A
  - USB 2.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# nRF5340 Hardware Constants
# ---------------------------------------------------------------------------

# Application core (main processor)
APP_CORE_COUNT = 1
APP_CPU_FREQ_MAX = 128_000_000         # 128 MHz
APP_CPU_FREQ_DEFAULT = 128_000_000
APP_FLASH_TOTAL = 1_048_576            # 1 MB
APP_RAM_TOTAL = 512 * 1024             # 512 KB

# Network core (radio processor)
NET_CORE_COUNT = 1
NET_CPU_FREQ_MAX = 64_000_000          # 64 MHz
NET_CPU_FREQ_DEFAULT = 64_000_000
NET_FLASH_TOTAL = 256 * 1024           # 256 KB
NET_RAM_TOTAL = 64 * 1024              # 64 KB

# Combined
CORE_COUNT = APP_CORE_COUNT + NET_CORE_COUNT
FLASH_TOTAL = APP_FLASH_TOTAL + NET_FLASH_TOTAL
RAM_TOTAL = APP_RAM_TOTAL + NET_RAM_TOTAL

GPIO_COUNT = 48                         # P0.00-P0.31 + P1.00-P1.15
ADC_RESOLUTION = 12
ADC_CHANNELS = 2
RTC_COUNT = 4
TIMER_COUNT = 6
PWM_COUNT = 4
SPI_COUNT = 4
I2C_COUNT = 4
UART_COUNT = 3
RADIO_BLE_5 = True                     # BLE 5.3
RADIO_ZIGBEE = True
RADIO_THREAD = True
NFC_AVAILABLE = True
USB_AVAILABLE = True

PERIPHERAL_BASE = 0x40000000
APP_RAM_BASE = 0x20000000
NET_RAM_BASE = 0x21000000
APP_FLASH_BASE = 0x00000000
NET_FLASH_BASE = 0x01000000
CODE_PAGE_SIZE = 4096
APP_CODE_PAGES = APP_FLASH_TOTAL // CODE_PAGE_SIZE
NET_CODE_PAGES = NET_FLASH_TOTAL // CODE_PAGE_SIZE


class CoreID(IntEnum):
    APPLICATION = 0
    NETWORK = 1


class GPIOPort(IntEnum):
    PORT0 = 0
    PORT1 = 1


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
    ADC_CONDUCTIVITY = 42
    PWM_THRUSTER = 50
    PWM_AUX = 51
    NFC_ANTENNA = 60
    GPS_PPS = 70
    AUDIO_IN = 80
    AUDIO_OUT = 81


@dataclass
class PinConfig:
    function: PinFunction
    port: GPIOPort
    pin: int
    core: CoreID = CoreID.APPLICATION
    direction: str = "input"
    pull: PinPull = PinPull.NONE
    drive: PinDrive = PinDrive.S0S1
    description: str = ""

    def __post_init__(self):
        max_pin = 31 if self.port == GPIOPort.PORT0 else 15
        if not (0 <= self.pin <= max_pin):
            raise ValueError(f"Pin {self.pin} out of range for {self.port.name} (0-{max_pin}).")

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
    """Dual-core memory layout for nRF5340."""
    regions: List[MemoryRegion] = field(default_factory=list)

    def add_region(self, name: str, start: int, size: int, type: str = "ram", purpose: str = ""):
        self.regions.append(MemoryRegion(name=name, start=start, size_bytes=size, type=type, purpose=purpose))

    def total_allocated(self, type: str = "ram") -> int:
        return sum(r.size_bytes for r in self.regions if r.type == type)

    def total_ram(self) -> int:
        return self.total_allocated("ram")

    def total_flash(self) -> int:
        return self.total_allocated("flash")

    def fits_in_app_ram(self) -> bool:
        app_ram = sum(r.size_bytes for r in self.regions
                      if r.type == "ram" and r.start < NET_RAM_BASE)
        return app_ram <= APP_RAM_TOTAL

    def fits_in_net_ram(self) -> bool:
        net_ram = sum(r.size_bytes for r in self.regions
                      if r.type == "ram" and r.start >= NET_RAM_BASE)
        return net_ram <= NET_RAM_TOTAL

    def fits_in_ram(self) -> bool:
        return self.fits_in_app_ram() and self.fits_in_net_ram()

    def fits_in_flash(self) -> bool:
        app_flash = sum(r.size_bytes for r in self.regions
                        if r.type == "flash" and r.start < NET_FLASH_BASE)
        net_flash = sum(r.size_bytes for r in self.regions
                        if r.type == "flash" and r.start >= NET_FLASH_BASE)
        return app_flash <= APP_FLASH_TOTAL and net_flash <= NET_FLASH_TOTAL

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
    zigbee_enabled: bool = False
    thread_enabled: bool = False
    nfc_enabled: bool = False
    ble_max_conn: int = 20
    ble_mtu: int = 517                  # Extended MTU support
    ble_min_conn_interval_ms: float = 7.5
    ble_max_conn_interval_ms: float = 4000.0
    ble_tx_power_dbm: int = 0           # range: -20 to +8
    ble_version: str = "5.3"
    radio_mode: str = "ble"
    le_audio: bool = False
    direction_finding: bool = False

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


class NRF5340Config:
    """
    Top-level hardware configuration for the NEXUS nRF5340 advanced marine node.

    Dual-core Cortex-M33 with BLE 5.3, LE Audio, Direction Finding,
    and Thread support. The application core runs NEXUS firmware while
    the network core handles radio protocol processing.
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
        PinFunction.ADC_CONDUCTIVITY: (0, 4),
        PinFunction.PWM_THRUSTER: (0, 16),
        PinFunction.PWM_AUX: (0, 15),
        PinFunction.NFC_ANTENNA: (0, 9),
        PinFunction.GPS_PPS: (1, 0),
        PinFunction.AUDIO_IN: (0, 5),
        PinFunction.AUDIO_OUT: (0, 6),
    }

    def __init__(self):
        self._pin_map: Dict[PinFunction, PinConfig] = {}
        self.protocol = ProtocolConfig()
        self.memory = MemoryLayout()
        self._peripherals_used: Set[str] = set()
        self._configure_default_pins()

    def _configure_default_pins(self):
        for func, (port, pin) in self.DEFAULT_PINS.items():
            core = CoreID.NETWORK if func in (PinFunction.NFC_ANTENNA,) else CoreID.APPLICATION
            self._pin_map[func] = PinConfig(
                function=func, port=GPIOPort(port), pin=pin, core=core,
                description=f"{func.name} on P{port}.{pin:02d}",
            )

    def get_pin(self, function: PinFunction) -> PinConfig:
        if function not in self._pin_map:
            raise KeyError(f"Pin function {function.name} not configured.")
        return self._pin_map[function]

    def remap_pin(self, function: PinFunction, port: int, pin: int, **kwargs):
        old = self._pin_map.get(function)
        core = kwargs.pop("core", CoreID.APPLICATION)
        if old:
            self._pin_map[function] = PinConfig(
                function=function, port=GPIOPort(port), pin=pin, core=core,
                description=kwargs.pop("description", old.description), **kwargs,
            )
        else:
            self._pin_map[function] = PinConfig(
                function=function, port=GPIOPort(port), pin=pin, core=core, **kwargs
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
    def PIN_STATUS_LED(self) -> PinConfig:
        return self.get_pin(PinFunction.STATUS_LED)

    def configure_ble(self, max_conn: int = 20, mtu: int = 517, tx_power: int = 0):
        self.protocol.ble_enabled = True
        self.protocol.zigbee_enabled = False
        self.protocol.thread_enabled = False
        self.protocol.radio_mode = "ble"
        self.protocol.ble_max_conn = max_conn
        self.protocol.ble_mtu = mtu
        self.protocol.ble_tx_power_dbm = tx_power

    def configure_thread(self):
        self.protocol.ble_enabled = False
        self.protocol.zigbee_enabled = False
        self.protocol.thread_enabled = True
        self.protocol.radio_mode = "thread"

    def configure_audio(self):
        """Enable LE Audio features."""
        self.protocol.le_audio = True

    def configure_direction_finding(self):
        """Enable BLE Direction Finding (AoA/AoD)."""
        self.protocol.direction_finding = True

    def configure_marine_node(self):
        """Set up memory regions for NEXUS marine node with dual-core."""
        # Application core RAM
        self.memory.add_region("app_stack", APP_RAM_BASE, 8192, "ram", "App core stack")
        self.memory.add_region("app_heap", APP_RAM_BASE + 8192, 65536, "ram", "App heap")
        self.memory.add_region("app_sensor", APP_RAM_BASE + 73728, 32768, "ram", "Sensor data")
        self.memory.add_region("app_uart", APP_RAM_BASE + 106496, 8192, "ram", "UART buffers")
        self.memory.add_region("app_audio", APP_RAM_BASE + 114688, 32768, "ram", "Audio buffer")
        self.memory.add_region("app_ipc", APP_RAM_BASE + 147456, 4096, "ram", "IPC shared RAM")
        # Network core RAM
        self.memory.add_region("net_ble", NET_RAM_BASE, 40960, "ram", "BLE stack on net core")
        self.memory.add_region("net_radio", NET_RAM_BASE + 40960, 8192, "ram", "Radio driver")
        # Flash
        self.memory.add_region("app_softdevice", APP_FLASH_BASE, 180 * CODE_PAGE_SIZE, "flash", "App SoftDevice")
        self.memory.add_region("app_firmware", APP_FLASH_BASE + 180 * CODE_PAGE_SIZE, 68 * CODE_PAGE_SIZE, "flash", "App firmware")
        self.memory.add_region("app_settings", APP_FLASH_BASE + 248 * CODE_PAGE_SIZE, 8 * CODE_PAGE_SIZE, "flash", "App NVS")
        self.memory.add_region("net_firmware", NET_FLASH_BASE, 56 * CODE_PAGE_SIZE, "flash", "Net core firmware")

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
            errors.append("RAM overflow in one or both cores.")
        if not self.memory.fits_in_flash():
            errors.append("Flash overflow in one or both cores.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def summary(self) -> dict:
        return {
            "chip": "nRF5340",
            "app_cpu": "Cortex-M33 @ 128 MHz",
            "net_cpu": "Cortex-M33 @ 64 MHz",
            "app_flash_bytes": APP_FLASH_TOTAL,
            "app_ram_bytes": APP_RAM_TOTAL,
            "net_flash_bytes": NET_FLASH_TOTAL,
            "net_ram_bytes": NET_RAM_TOTAL,
            "ble_version": self.protocol.ble_version,
            "le_audio": self.protocol.le_audio,
            "direction_finding": self.protocol.direction_finding,
            "radio_mode": self.protocol.radio_mode,
            "pins_configured": len(self._pin_map),
            "peripherals_allocated": list(self._peripherals_used),
        }
