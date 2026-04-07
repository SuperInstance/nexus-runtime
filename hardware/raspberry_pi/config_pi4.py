"""
NEXUS Raspberry Pi 4 Model B Configuration — BCM2711 SoC.

Target hardware:
  - SoC:    Broadcom BCM2711, Quad-core Cortex-A72 @ 1.5 GHz
  - RAM:    1 / 2 / 4 / 8 GB LPDDR4-3200
  - GPIO:   40-pin header (28 BCM GPIOs)
  - CSI:    2× MIPI CSI-2 (4-lane) for camera / sonar
  - DSI:    2× MIPI DSI (4-lane) for display
  - USB:    2× USB 3.0 + 2× USB 2.0
  - Ethernet: Gigabit (BCM54210 PHY)
  - WiFi:   802.11ac dual-band + Bluetooth 5.0

Marine robotics use: primary compute node, underwater camera processing,
navigation computer, thruster ESC control, CTD sensor aggregation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GPIO Pin Definitions
# ---------------------------------------------------------------------------

class GPIOPin(IntEnum):
    """BCM GPIO pin numbers available on the 40-pin header."""
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
    """Physical-to-BCM GPIO pin mapping entry."""
    physical_pin: int
    bcm_gpio: int
    name: str
    default_function: str
    alt_functions: Tuple[str, ...]
    marine_default: Optional[str] = None  # NEXUS marine sensor assignment

    def __hash__(self) -> int:
        return hash((self.physical_pin, self.bcm_gpio))


# Full 40-pin header mapping for Pi 4B
PINOUT: List[PinMapping] = [
    PinMapping(1,  -1, "3V3",         "3.3V power",    ()),
    PinMapping(2,  -1, "5V",          "5.0V power",    ()),
    PinMapping(3,  2,  "GPIO2/SDA1",  "I2C1 SDA",      ("I2C", "marine_ctd_sda")),
    PinMapping(4,  -1, "5V",          "5.0V power",    ()),
    PinMapping(5,  3,  "GPIO3/SCL1",  "I2C1 SCL",      ("I2C", "marine_ctd_scl")),
    PinMapping(6,  -1, "GND",         "Ground",        ()),
    PinMapping(7,  4,  "GPIO4",       "GPIO / UART2 TX", ("GPIO", "marine_telemetry_tx")),
    PinMapping(8,  14, "GPIO14/TXD0", "UART0 TX",      ("UART0", "marine_gps_tx")),
    PinMapping(9,  -1, "GND",         "Ground",        ()),
    PinMapping(10, 15, "GPIO15/RXD0", "UART0 RX",      ("UART0", "marine_gps_rx")),
    PinMapping(11, 17, "GPIO17",      "GPIO",          ("GPIO", "marine_leak_sensor")),
    PinMapping(12, 18, "GPIO18",      "PWM0",          ("PWM0", "marine_servo")),
    PinMapping(13, 27, "GPIO27",      "GPIO",          ("GPIO",)),
    PinMapping(14, -1, "GND",         "Ground",        ()),
    PinMapping(15, 22, "GPIO22",      "GPIO",          ("GPIO", "marine_can_cs")),
    PinMapping(16, 23, "GPIO23",      "GPIO / SPI0 SCK", ("SPI0",)),
    PinMapping(17, -1, "3V3",         "3.3V power",    ()),
    PinMapping(18, 24, "GPIO24",      "GPIO",          ("GPIO",)),
    PinMapping(19, 10, "GPIO10/MOSI", "SPI0 MOSI",     ("SPI0",)),
    PinMapping(20, -1, "GND",         "Ground",        ()),
    PinMapping(21,  9, "GPIO9/MISO",  "SPI0 MISO",     ("SPI0",)),
    PinMapping(22, 25, "GPIO25",      "GPIO",          ("GPIO", "marine_status_led")),
    PinMapping(23, 11, "GPIO11/SCLK", "SPI0 SCLK",     ("SPI0",)),
    PinMapping(24,  8, "GPIO8/CE0",   "SPI0 CE0",      ("SPI0", "marine_adc_pressure")),
    PinMapping(25, -1, "GND",         "Ground",        ()),
    PinMapping(26,  7, "GPIO7/CE1",   "SPI0 CE1",      ("SPI0", "marine_adc_do")),
    PinMapping(27,  0, "ID_SD",       "I2C0 SDA (EEPROM)", ("I2C0",)),
    PinMapping(28,  1, "ID_SC",       "I2C0 SCL (EEPROM)", ("I2C0",)),
    PinMapping(29,  5, "GPIO5",       "GPIO / UART2 RX", ("GPIO", "marine_telemetry_rx")),
    PinMapping(30, -1, "GND",         "Ground",        ()),
    PinMapping(31,  6, "GPIO6",       "GPIO",          ("GPIO",)),
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


# ---------------------------------------------------------------------------
# Marine Sensor Pin Assignments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarineSensorPinAssignment:
    """Pin assignment for a specific marine sensor."""
    sensor_name: str
    sensor_type: str
    interface: str
    pins: Dict[str, int]      # signal -> BCM GPIO
    config_params: Dict[str, Any] = field(default_factory=dict)


# Pre-defined sensor pin assignments for common marine sensors
MARINE_SENSOR_MAPPINGS: Dict[str, MarineSensorPinAssignment] = {
    "ctd": MarineSensorPinAssignment(
        sensor_name="CTD Sensor (Conductivity/Temperature/Depth)",
        sensor_type="ctd",
        interface="I2C-1",
        pins={"sda": 2, "scl": 3},
        config_params={"address": 0x66, "baud_rate": 100000, "poll_interval_ms": 500},
    ),
    "imu": MarineSensorPinAssignment(
        sensor_name="BNO055 9-DOF IMU",
        sensor_type="imu",
        interface="I2C-1",
        pins={"sda": 2, "scl": 3},
        config_params={"address": 0x28, "mode": "NDOF", "poll_interval_ms": 10},
    ),
    "gps": MarineSensorPinAssignment(
        sensor_name="u-blox NEO-M8N GPS",
        sensor_type="gps",
        interface="UART0",
        pins={"tx": 14, "rx": 15},
        config_params={"baud_rate": 9600, "nmea_version": "4.11"},
    ),
    "echosounder": MarineSensorPinAssignment(
        sensor_name="Blue Robotics Ping Sonar / Echosounder",
        sensor_type="echosounder",
        interface="UART1",
        pins={"tx": 0, "rx": 1},
        config_params={"baud_rate": 115200, "max_range_m": 30.0},
    ),
    "leak_sensor": MarineSensorPinAssignment(
        sensor_name="Water Leak Detector",
        sensor_type="leak",
        interface="GPIO",
        pins={"alarm": 17},
        config_params={"active_low": True, "debounce_ms": 50},
    ),
    "esc_thrust": MarineSensorPinAssignment(
        sensor_name="Brushless ESC (Thruster Port)",
        sensor_type="esc",
        interface="PWM0",
        pins={"signal": 12},
        config_params={"frequency_hz": 50, "pulse_min_us": 1000, "pulse_max_us": 2000},
    ),
    "esc_thrust_2": MarineSensorPinAssignment(
        sensor_name="Brushless ESC (Thruster Starboard)",
        sensor_type="esc",
        interface="PWM1",
        pins={"signal": 13},
        config_params={"frequency_hz": 50, "pulse_min_us": 1000, "pulse_max_us": 2000},
    ),
    "servo": MarineSensorPinAssignment(
        sensor_name="Servo (Manipulator Arm)",
        sensor_type="servo",
        interface="PWM0",
        pins={"signal": 18},
        config_params={"frequency_hz": 50, "pulse_min_us": 500, "pulse_max_us": 2500},
    ),
    "adc_pressure": MarineSensorPinAssignment(
        sensor_name="SPI ADC (Water Pressure Transducer)",
        sensor_type="adc_pressure",
        interface="SPI-0",
        pins={"mosi": 10, "miso": 9, "sclk": 11, "cs": 8},
        config_params={"spi_mode": 0, "max_speed_hz": 1000000, "channels": 8},
    ),
    "adc_do": MarineSensorPinAssignment(
        sensor_name="SPI ADC (Dissolved Oxygen Sensor)",
        sensor_type="adc_do",
        interface="SPI-0",
        pins={"mosi": 10, "miso": 9, "sclk": 11, "cs": 7},
        config_params={"spi_mode": 0, "max_speed_hz": 500000, "channels": 4},
    ),
    "power_monitor": MarineSensorPinAssignment(
        sensor_name="INA219 Power Monitor",
        sensor_type="power",
        interface="I2C-1",
        pins={"sda": 2, "scl": 3},
        config_params={"address": 0x40, "shunt_resistor_uohm": 10000},
    ),
    "can_bus": MarineSensorPinAssignment(
        sensor_name="MCP2515 CAN Bus (NMEA 2000)",
        sensor_type="can",
        interface="SPI-0",
        pins={"mosi": 10, "miso": 9, "sclk": 11, "cs": 22, "int": 25},
        config_params={"oscillator_hz": 8000000, "bus_speed_kbps": 250},
    ),
    "telemetry": MarineSensorPinAssignment(
        sensor_name="Telemetry Radio UART",
        sensor_type="telemetry",
        interface="UART2",
        pins={"tx": 4, "rx": 5},
        config_params={"baud_rate": 460800, "protocol": "MAVLink 2.0"},
    ),
}


# ---------------------------------------------------------------------------
# Thermal / Power Profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThermalProfile:
    """Thermal limits for the Pi 4B in marine enclosures."""
    idle_temp_c: float = 45.0
    typical_load_c: float = 60.0
    throttle_start_c: float = 80.0
    critical_c: float = 85.0
    recommended_max_ambient_c: float = 55.0
    enclosure_note: str = (
        "Requires heatsink + passive cooling in IP67 enclosure. "
        "Active cooling recommended above 40°C ambient water temp."
    )


@dataclass(frozen=True)
class PowerProfile:
    """Power consumption estimates for Pi 4B."""
    idle_w: float = 3.0
    typical_load_w: float = 5.5
    max_load_w: float = 7.6
    usb_peripheral_budget_w: float = 4.5
    voltage_range: Tuple[float, float] = (5.0, 5.25)
    recommended_psu_w: float = 15.0
    battery_life_estimate_ah: float = 1.6  # At 12V with 5V buck converter


# ---------------------------------------------------------------------------
# Compute Module 4 Variant
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CM4Variant:
    """Compute Module 4 specific differences from Pi 4B."""
    form_factor: str = "Module (67.6 × 31 mm)"
    connector: str = "100-pin DDR2 SODIMM"
    has_ethernet: bool = False
    has_usb: bool = False
    has_gpio_header: bool = False
    requires_carrier: bool = True
    maxcsi_lanes: int = 4
    maxdsi_lanes: int = 4
    pcie_gen: Optional[int] = None
    emmc_sizes: Tuple[int, ...] = (0, 8, 16, 32)
    wireless_options: Tuple[str, ...] = ("none", "wifi_ble")


# ---------------------------------------------------------------------------
# Main Configuration
# ---------------------------------------------------------------------------

class Pi4Config:
    """
    Complete hardware configuration for Raspberry Pi 4 Model B.

    Encapsulates SoC specs, GPIO pinout, peripheral interfaces, and
    marine sensor pin mappings for NEXUS deployments.
    """

    def __init__(self, variant: Optional[CM4Variant] = None):
        self.variant = variant

        # SoC Identification
        self.board_model = "pi4b"
        self.board_name = "Raspberry Pi 4 Model B"
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
        self.default_ram_gb = 8

        # Peripheral Interfaces
        self.peripherals = {
            "i2c": {"count": 2, "max_speed_hz": 400_000, "pins": [(2, 3), (0, 1)]},
            "spi": {"count": 2, "max_speed_hz": 125_000_000, "pins": [(9, 10, 11), (19, 20, 21)]},
            "uart": {"count": 5, "pins": [(14, 15), (0, 1), (4, 5), (8, 9), (12, 13)]},
            "pwm": {"count": 2, "channels": 4, "frequency_range_hz": (1, 125_000_000)},
            "csi": {"count": 2, "lanes_per_port": 4, "max_resolution": "1080p60 / 4K30"},
            "dsi": {"count": 2, "lanes_per_port": 4},
            "usb": {"usb3_ports": 2, "usb2_ports": 2, "total_bandwidth_gbps": 10.0},
            "ethernet": {"speed_mbps": 1000, "phy": "BCM54210"},
            "wifi": {"standard": "802.11ac", "bands": ["2.4GHz", "5GHz"], "antenna": "onboard"},
            "bluetooth": {"version": "5.0", "ble": True},
        }

        # GPIO
        self.gpio_pins = list(GPIOPin)
        self.pinout = PINOUT
        self.total_gpio = 28

        # Marine-specific
        self.marine_sensor_mappings = MARINE_SENSOR_MAPPINGS
        self.thermal = ThermalProfile()
        self.power = PowerProfile()

        # NEXUS platform metadata
        self.nexus_platform = "raspberry_pi"
        self.nexus_role = "primary_compute"

        # Build config fingerprint for integrity checks
        self._config_hash = self._compute_hash()

    # --- Public API ---

    def get_pin(self, bcm_gpio: int) -> Optional[PinMapping]:
        """Look up a pin mapping by BCM GPIO number."""
        for pin in self.pinout:
            if pin.bcm_gpio == bcm_gpio:
                return pin
        return None

    def get_pin_by_physical(self, physical: int) -> Optional[PinMapping]:
        """Look up a pin mapping by physical header pin number."""
        for pin in self.pinout:
            if pin.physical_pin == physical:
                return pin
        return None

    def get_sensor_mapping(self, sensor_key: str) -> Optional[MarineSensorPinAssignment]:
        """Get marine sensor pin assignment by sensor key."""
        return self.marine_sensor_mappings.get(sensor_key)

    def list_sensor_mappings(self) -> List[str]:
        """List all available marine sensor mapping keys."""
        return sorted(self.marine_sensor_mappings.keys())

    def validate_gpio_assignment(self, bcm_gpio: int) -> bool:
        """Check if a BCM GPIO number is valid for this board."""
        return bcm_gpio in [p.value for p in GPIOPin]

    def check_pin_conflict(
        self, sensor_key: str, other_sensor_key: str
    ) -> List[int]:
        """
        Check for GPIO pin conflicts between two sensor assignments.

        Returns list of conflicting BCM GPIO numbers (empty if no conflict).
        """
        s1 = self.get_sensor_mapping(sensor_key)
        s2 = self.get_sensor_mapping(other_sensor_key)
        if not s1 or not s2:
            return []
        pins_s1 = set(s1.pins.values())
        pins_s2 = set(s2.pins.values())
        return sorted(pins_s1 & pins_s2)

    def compute_power_budget(self, peripherals: List[str]) -> Dict[str, float]:
        """
        Estimate total power draw given a list of active peripherals.

        Args:
            peripherals: List of peripheral names (e.g. ['csi', 'ethernet', 'wifi']).

        Returns:
            Dict with estimated watts for base + each peripheral + total.
        """
        peripheral_power = {
            "csi": 0.3,      # Per camera
            "dsi": 0.5,      # Per display
            "ethernet": 0.4,
            "wifi": 0.5,
            "bluetooth": 0.2,
            "usb": 1.0,      # Per active USB device
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
        """Return SHA-256 hash of this configuration for integrity checks."""
        return self._config_hash

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary dict."""
        return {
            "board": self.board_name,
            "soc": self.soc,
            "cpu": f"{self.cpu_cores}x {self.cpu_arch} @ {self.cpu_clock_max_mhz}MHz",
            "ram_options_gb": list(self.ram_options_gb),
            "gpio_count": self.total_gpio,
            "sensor_mappings": self.list_sensor_mappings(),
            "thermal_throttle_c": self.thermal.throttle_start_c,
            "power_idle_w": self.power.idle_w,
            "power_max_w": self.power.max_load_w,
            "config_hash": self._config_hash[:16],
        }

    # --- Internal ---

    def _compute_hash(self) -> str:
        """Compute a SHA-256 hash over critical config fields."""
        data = (
            f"{self.soc}:{self.cpu_cores}:{self.cpu_clock_max_mhz}:"
            f"{self.total_gpio}:{len(self.marine_sensor_mappings)}:"
            f"{self.thermal.throttle_start_c}:{self.power.max_load_w}"
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def __repr__(self) -> str:
        variant_str = " (CM4 Variant)" if self.variant else ""
        return (
            f"Pi4Config(board='{self.board_name}', soc='{self.soc}', "
            f"ram={self.default_ram_gb}GB, gpio={self.total_gpio}{variant_str})"
        )
