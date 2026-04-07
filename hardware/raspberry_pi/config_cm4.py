"""
NEXUS Compute Module 4 Configuration — BCM2711 SoC.

Target hardware:
  - SoC:    Broadcom BCM2711, Quad-core Cortex-A72 @ 1.5 GHz
  - RAM:    1 / 2 / 4 / 8 GB LPDDR4-3200
  - eMMC:   0 / 8 / 16 / 32 GB onboard flash
  - GPIO:   100-pin DDR2 SODIMM connector (28 BCM GPIOs via carrier)
  - CSI:    2x MIPI CSI-2 (4-lane)
  - DSI:    2x MIPI DSI (4-lane)
  - PCIe:   1x PCIe 2.0 x1 (single lane)
  - Ethernet: via carrier board
  - WiFi:   Optional (via antenna header on wireless variant)
  - Form factor: Module (67.6 x 31 mm)
  - Connector: 100-pin DDR2 SODIMM

Marine robotics use: custom carrier boards for embedded hull mounting,
industrial ROV/AUV controllers, IP67-rated sealed electronics bays,
volume production marine deployments.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


class GPIOPin(IntEnum):
    """BCM GPIO pin numbers available via the CM4 SODIMM connector."""
    GPIO0 = 0
    GPIO1 = 1
    GPIO2 = 2
    GPIO3 = 3
    GPIO4 = 4
    GPIO5 = 5
    GPIO6 = 6
    GPIO7 = 7
    GPIO8 = 8
    GPIO9 = 9
    GPIO10 = 10
    GPIO11 = 11
    GPIO12 = 12
    GPIO13 = 13
    GPIO14 = 14
    GPIO15 = 15
    GPIO16 = 16
    GPIO17 = 17
    GPIO18 = 18
    GPIO19 = 19
    GPIO20 = 20
    GPIO21 = 21
    GPIO22 = 22
    GPIO23 = 23
    GPIO24 = 24
    GPIO25 = 25
    GPIO26 = 26
    GPIO27 = 27


@dataclass(frozen=True)
class PinMapping:
    """SODIMM pin to BCM GPIO mapping for CM4 carrier board."""
    sodimm_pin: int
    bcm_gpio: int
    name: str
    default_function: str
    alt_functions: Tuple[str, ...]
    marine_default: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.sodimm_pin, self.bcm_gpio))


@dataclass(frozen=True)
class CM4Variant:
    """Compute Module 4 variant descriptor."""
    emmc_size_gb: int = 0
    wireless: bool = False
    io_board: str = "custom_carrier"
    pcie_enabled: bool = False
    nvme_enabled: bool = False


@dataclass(frozen=True)
class CarrierBoardSpec:
    """Specification for a CM4 carrier board."""
    name: str
    manufacturer: str
    has_ethernet: bool = True
    has_usb: bool = True
    has_gpio_header: bool = True
    pcie_slot: bool = False
    nvme_slot: bool = False
    industrial_temp: bool = False
    dimensions_mm: Tuple[float, float] = (85.0, 56.0)


# Pre-defined carrier boards for NEXUS marine use
CARRIER_BOARDS: Dict[str, CarrierBoardSpec] = {
    "nexus-io-board": CarrierBoardSpec(
        name="NEXUS CM4 IO Board",
        manufacturer="NEXUS Marine",
        has_ethernet=True,
        has_usb=True,
        has_gpio_header=True,
        pcie_slot=True,
        nvme_slot=True,
        industrial_temp=True,
    ),
    "raspberry-pi-io": CarrierBoardSpec(
        name="Raspberry Pi CM4 IO Board",
        manufacturer="Raspberry Pi Foundation",
    ),
}


PINOUT: List[PinMapping] = [
    PinMapping(1,  -1, "3V3",         "3.3V power",    ()),
    PinMapping(2,  -1, "5V",          "5.0V power",    ()),
    PinMapping(3,  2,  "GPIO2/SDA1",  "I2C1 SDA",      ("I2C", "marine_ctd_sda")),
    PinMapping(4,  -1, "5V",          "5.0V power",    ()),
    PinMapping(5,  3,  "GPIO3/SCL1",  "I2C1 SCL",      ("I2C", "marine_ctd_scl")),
    PinMapping(6,  -1, "GND",         "Ground",        ()),
    PinMapping(7,  4,  "GPIO4",       "GPIO",          ("GPIO",)),
    PinMapping(8,  14, "GPIO14/TXD0", "UART0 TX",      ("UART0", "marine_gps_tx")),
    PinMapping(9,  -1, "GND",         "Ground",        ()),
    PinMapping(10, 15, "GPIO15/RXD0", "UART0 RX",      ("UART0", "marine_gps_rx")),
    PinMapping(11, 17, "GPIO17",      "GPIO",          ("GPIO", "marine_leak_sensor")),
    PinMapping(12, 18, "GPIO18",      "PWM0",          ("PWM0", "marine_servo")),
    PinMapping(13, 27, "GPIO27",      "GPIO",          ("GPIO",)),
    PinMapping(14, -1, "GND",         "Ground",        ()),
    PinMapping(15, 22, "GPIO22",      "GPIO",          ("GPIO", "marine_can_cs")),
    PinMapping(16, 23, "GPIO23",      "SPI0 SCK",      ("SPI0",)),
    PinMapping(17, -1, "3V3",         "3.3V power",    ()),
    PinMapping(18, 24, "GPIO24",      "GPIO",          ("GPIO",)),
    PinMapping(19, 10, "GPIO10/MOSI", "SPI0 MOSI",     ("SPI0",)),
    PinMapping(20, -1, "GND",         "Ground",        ()),
    PinMapping(21, 9,  "GPIO9/MISO",  "SPI0 MISO",     ("SPI0",)),
    PinMapping(22, 25, "GPIO25",      "GPIO",          ("GPIO", "marine_status_led")),
    PinMapping(23, 11, "GPIO11/SCLK", "SPI0 SCLK",     ("SPI0",)),
    PinMapping(24, 8,  "GPIO8/CE0",   "SPI0 CE0",      ("SPI0", "marine_adc_pressure")),
    PinMapping(25, -1, "GND",         "Ground",        ()),
    PinMapping(26, 7,  "GPIO7/CE1",   "SPI0 CE1",      ("SPI0", "marine_adc_do")),
    PinMapping(27, 0,  "ID_SD",       "I2C0 SDA",      ("I2C0",)),
    PinMapping(28, 1,  "ID_SC",       "I2C0 SCL",      ("I2C0",)),
    PinMapping(29, 5,  "GPIO5",       "GPIO",          ("GPIO",)),
    PinMapping(30, -1, "GND",         "Ground",        ()),
    PinMapping(31, 6,  "GPIO6",       "GPIO",          ("GPIO",)),
    PinMapping(32, 12, "GPIO12",      "PWM0",          ("PWM0", "marine_esc_thrust")),
    PinMapping(33, 13, "GPIO13",      "PWM1",          ("PWM1", "marine_esc_thrust_2")),
    PinMapping(34, -1, "GND",         "Ground",        ()),
    PinMapping(35, 19, "GPIO19",      "SPI1 MISO",     ("SPI1",)),
    PinMapping(36, 16, "GPIO16",      "GPIO",          ("GPIO",)),
    PinMapping(37, 26, "GPIO26",      "GPIO",          ("GPIO", "marine_heartbeat_led")),
    PinMapping(38, 20, "GPIO20",      "SPI1 MOSI",     ("SPI1",)),
    PinMapping(39, -1, "GND",         "Ground",        ()),
    PinMapping(40, 21, "GPIO21",      "SPI1 SCLK",     ("SPI1",)),
]


@dataclass(frozen=True)
class MarineSensorPinAssignment:
    sensor_name: str
    sensor_type: str
    interface: str
    pins: Dict[str, int]
    config_params: Dict[str, Any] = field(default_factory=dict)


MARINE_SENSOR_MAPPINGS: Dict[str, MarineSensorPinAssignment] = {
    "ctd": MarineSensorPinAssignment(
        sensor_name="CTD Sensor",
        sensor_type="ctd", interface="I2C-1",
        pins={"sda": 2, "scl": 3},
        config_params={"address": 0x66, "baud_rate": 400000},
    ),
    "gps": MarineSensorPinAssignment(
        sensor_name="u-blox NEO-M8N GPS",
        sensor_type="gps", interface="UART0",
        pins={"tx": 14, "rx": 15},
        config_params={"baud_rate": 9600},
    ),
    "leak_sensor": MarineSensorPinAssignment(
        sensor_name="Water Leak Detector",
        sensor_type="leak", interface="GPIO",
        pins={"alarm": 17},
        config_params={"active_low": True, "debounce_ms": 50},
    ),
    "esc_thrust": MarineSensorPinAssignment(
        sensor_name="Brushless ESC (Thruster)",
        sensor_type="esc", interface="PWM0",
        pins={"signal": 12},
        config_params={"frequency_hz": 50, "pulse_min_us": 1000, "pulse_max_us": 2000},
    ),
}


@dataclass(frozen=True)
class ThermalProfile:
    idle_temp_c: float = 40.0
    typical_load_c: float = 55.0
    throttle_start_c: float = 80.0
    critical_c: float = 85.0
    recommended_max_ambient_c: float = 55.0
    enclosure_note: str = (
        "Thermal management is carrier-board dependent. The NEXUS IO board "
        "includes a thermal pad mount. Custom PCB thermal design required "
        "for sealed underwater enclosures."
    )


@dataclass(frozen=True)
class PowerProfile:
    idle_w: float = 2.5
    typical_load_w: float = 5.0
    max_load_w: float = 7.0
    usb_peripheral_budget_w: float = 4.5
    voltage_range: Tuple[float, float] = (5.0, 5.25)
    recommended_psu_w: float = 15.0
    battery_life_estimate_ah: float = 1.5


class CM4Config:
    """
    Complete hardware configuration for Raspberry Pi Compute Module 4.

    Designed for custom carrier board integration in NEXUS marine deployments.
    All GPIO is accessed via the 100-pin SODIMM connector.
    """

    def __init__(self, variant: Optional[CM4Variant] = None,
                 carrier: str = "nexus-io-board"):
        self.variant = variant or CM4Variant()
        self.carrier_name = carrier

        if carrier in CARRIER_BOARDS:
            self.carrier = CARRIER_BOARDS[carrier]
        else:
            self.carrier = CarrierBoardSpec(name=carrier, manufacturer="Custom")

        self.board_model = "cm4"
        self.board_name = "Raspberry Pi Compute Module 4"
        self.soc = "BCM2711"
        self.soc_manufacturer = "Broadcom Inc."
        self.cpu_arch = "cortex-a72"
        self.cpu_cores = 4
        self.cpu_clock_max_mhz = 1500
        self.cpu_clock_min_mhz = 600
        self.cpu_clock_default_mhz = 1500
        self.gpu = "VideoCore VI @ 500 MHz"
        self.ram_type = "LPDDR4-3200"
        self.ram_options_gb = (1, 2, 4, 8)
        self.default_ram_gb = 4
        self.emmc_options_gb = (0, 8, 16, 32)
        self.default_emmc_gb = 16

        self.peripherals = {
            "i2c": {"count": 2, "max_speed_hz": 400_000, "pins": [(2, 3), (0, 1)]},
            "spi": {"count": 2, "max_speed_hz": 125_000_000, "pins": [(9, 10, 11), (19, 20, 21)]},
            "uart": {"count": 5, "pins": [(14, 15), (0, 1), (4, 5), (8, 9), (12, 13)]},
            "pwm": {"count": 2, "channels": 4, "frequency_range_hz": (1, 125_000_000)},
            "csi": {"count": 2, "lanes_per_port": 4, "max_resolution": "1080p60 / 4K30"},
            "dsi": {"count": 2, "lanes_per_port": 4},
            "pcie": {"version": "2.0", "lanes": 1, "max_bandwidth_mbps": 500},
            "ethernet": {"speed_mbps": 1000, "phy": "carrier-dependent"},
            "wifi": {"standard": "802.11ac" if self.variant.wireless else "none",
                     "bands": ["2.4GHz", "5GHz"] if self.variant.wireless else [],
                     "antenna": "external" if self.variant.wireless else "none"},
            "bluetooth": {"version": "5.0" if self.variant.wireless else "none",
                          "ble": self.variant.wireless},
        }

        self.gpio_pins = list(GPIOPin)
        self.pinout = PINOUT
        self.total_gpio = 28
        self.marine_sensor_mappings = MARINE_SENSOR_MAPPINGS
        self.thermal = ThermalProfile()
        self.power = PowerProfile()
        self.nexus_platform = "raspberry_pi"
        self.nexus_role = "embedded_compute"
        self._config_hash = self._compute_hash()

    def get_pin(self, bcm_gpio: int) -> Optional[PinMapping]:
        for pin in self.pinout:
            if pin.bcm_gpio == bcm_gpio:
                return pin
        return None

    def get_pin_by_physical(self, physical: int) -> Optional[PinMapping]:
        for pin in self.pinout:
            if pin.sodimm_pin == physical:
                return pin
        return None

    def get_sensor_mapping(self, sensor_key: str) -> Optional[MarineSensorPinAssignment]:
        return self.marine_sensor_mappings.get(sensor_key)

    def list_sensor_mappings(self) -> List[str]:
        return sorted(self.marine_sensor_mappings.keys())

    def validate_gpio_assignment(self, bcm_gpio: int) -> bool:
        return bcm_gpio in [p.value for p in GPIOPin]

    def check_pin_conflict(self, sensor_key: str, other_sensor_key: str) -> List[int]:
        s1 = self.get_sensor_mapping(sensor_key)
        s2 = self.get_sensor_mapping(other_sensor_key)
        if not s1 or not s2:
            return []
        return sorted(set(s1.pins.values()) & set(s2.pins.values()))

    def compute_power_budget(self, peripherals: List[str]) -> Dict[str, float]:
        peripheral_power = {
            "csi": 0.3, "dsi": 0.5, "ethernet": 0.4,
            "wifi": 0.5, "bluetooth": 0.2, "usb": 1.0,
            "pcie": 2.0,
        }
        base = self.power.idle_w
        breakdown = {"base_w": base}
        for p in peripherals:
            w = peripheral_power.get(p, 0.2)
            breakdown[f"{p}_w"] = w
            base += w
        breakdown["total_w"] = round(base, 2)
        return breakdown

    def get_config_hash(self) -> str:
        return self._config_hash

    def summary(self) -> Dict[str, Any]:
        return {
            "board": self.board_name,
            "soc": self.soc,
            "cpu": f"{self.cpu_cores}x {self.cpu_arch} @ {self.cpu_clock_max_mhz}MHz",
            "ram_options_gb": list(self.ram_options_gb),
            "emmc_options_gb": list(self.emmc_options_gb),
            "carrier": self.carrier.name,
            "wireless": self.variant.wireless,
            "pcie": self.peripherals["pcie"]["version"],
            "gpio_count": self.total_gpio,
            "sensor_mappings": self.list_sensor_mappings(),
            "thermal_throttle_c": self.thermal.throttle_start_c,
            "power_idle_w": self.power.idle_w,
            "power_max_w": self.power.max_load_w,
            "config_hash": self._config_hash[:16],
        }

    def _compute_hash(self) -> str:
        data = (
            f"{self.soc}:{self.cpu_cores}:{self.cpu_clock_max_mhz}:"
            f"{self.total_gpio}:{len(self.marine_sensor_mappings)}:"
            f"{self.carrier_name}:{self.variant.wireless}:"
            f"{self.thermal.throttle_start_c}:{self.power.max_load_w}"
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def __repr__(self) -> str:
        wl = ", wireless" if self.variant.wireless else ""
        return (
            f"CM4Config(board='{self.board_name}', soc='{self.soc}', "
            f"ram={self.default_ram_gb}GB, carrier='{self.carrier_name}'{wl})"
        )
