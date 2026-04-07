"""
NEXUS Raspberry Pi 5 Configuration — BCM2712 SoC.

Target hardware:
  - SoC:    Broadcom BCM2712, Quad-core Cortex-A76 @ 2.4 GHz
  - RAM:    4 / 8 GB LPDDR4X-4267
  - GPU:    VideoCore VII
  - GPIO:   40-pin header (reversed JST socket), 16 extra via RP1
  - PCIe:   1× PCIe 2.0 x1 (via RP1 I/O controller)
  - CSI:    4-lane MIPI via RP1 (4K@60 capable)
  - DSI:    2× 4-lane MIPI via RP1
  - USB:    2× USB 3.0 + 2× USB 2.0 via RP1
  - Ethernet: Gigabit PoE HAT compatible
  - WiFi:   802.11ac dual-band + Bluetooth 5.0 / BLE
  - Real-time clock: Onboard

Marine robotics use: high-performance autonomy controller, 4K sonar
visualization, real-time SLAM, multi-camera navigation, heavy compute node.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GPIO Pin Definitions (Pi 5 uses RP1 I/O controller, different from BCM)
# ---------------------------------------------------------------------------

class Pi5GPIOPin(IntEnum):
    """RP1 GPIO pin numbers for Pi 5 (via RP1 south bridge)."""
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
class Pi5PinMapping:
    """Physical-to-RP1 GPIO pin mapping entry for Pi 5."""
    physical_pin: int
    rp1_gpio: int
    name: str
    default_function: str
    alt_functions: Tuple[str, ...]
    marine_default: Optional[str] = None


# Pi 5 pinout — same physical layout as Pi 4B but via RP1
PI5_PINOUT: List[Pi5PinMapping] = [
    Pi5PinMapping(1,  -1, "3V3",         "3.3V power",    ()),
    Pi5PinMapping(2,  -1, "5V",          "5.0V power",    ()),
    Pi5PinMapping(3,  2,  "GPIO2",       "I2C1 SDA",      ("I2C", "marine_ctd_sda")),
    Pi5PinMapping(4,  -1, "5V",          "5.0V power",    ()),
    Pi5PinMapping(5,  3,  "GPIO3",       "I2C1 SCL",      ("I2C", "marine_ctd_scl")),
    Pi5PinMapping(6,  -1, "GND",         "Ground",        ()),
    Pi5PinMapping(7,  4,  "GPIO4",       "UART2 TX",      ("UART", "marine_telemetry_tx")),
    Pi5PinMapping(8,  14, "GPIO14",      "UART0 TX",      ("UART", "marine_gps_tx")),
    Pi5PinMapping(9,  -1, "GND",         "Ground",        ()),
    Pi5PinMapping(10, 15, "GPIO15",      "UART0 RX",      ("UART", "marine_gps_rx")),
    Pi5PinMapping(11, 17, "GPIO17",      "GPIO",          ("GPIO", "marine_leak_sensor")),
    Pi5PinMapping(12, 18, "GPIO18",      "PWM0/1",        ("PWM", "marine_servo")),
    Pi5PinMapping(13, 27, "GPIO27",      "GPIO",          ("GPIO",)),
    Pi5PinMapping(14, -1, "GND",         "Ground",        ()),
    Pi5PinMapping(15, 22, "GPIO22",      "GPIO",          ("GPIO", "marine_can_cs")),
    Pi5PinMapping(16, 23, "GPIO23",      "SPI0 SCK",      ("SPI",)),
    Pi5PinMapping(17, -1, "3V3",         "3.3V power",    ()),
    Pi5PinMapping(18, 24, "GPIO24",      "SPI0 CS2",      ("SPI",)),
    Pi5PinMapping(19, 10, "GPIO10",      "SPI0 MOSI",     ("SPI",)),
    Pi5PinMapping(20, -1, "GND",         "Ground",        ()),
    Pi5PinMapping(21,  9, "GPIO9",       "SPI0 MISO",     ("SPI",)),
    Pi5PinMapping(22, 25, "GPIO25",      "GPIO",          ("GPIO", "marine_status_led")),
    Pi5PinMapping(23, 11, "GPIO11",      "SPI0 SCLK",     ("SPI",)),
    Pi5PinMapping(24,  8, "GPIO8",       "SPI0 CS0",      ("SPI", "marine_adc_pressure")),
    Pi5PinMapping(25, -1, "GND",         "Ground",        ()),
    Pi5PinMapping(26,  7, "GPIO7",       "SPI0 CS1",      ("SPI", "marine_adc_do")),
    Pi5PinMapping(27,  0, "ID_SD",       "I2C0 SDA",      ("I2C",)),
    Pi5PinMapping(28,  1, "ID_SC",       "I2C0 SCL",      ("I2C",)),
    Pi5PinMapping(29,  5, "GPIO5",       "UART2 RX",      ("UART", "marine_telemetry_rx")),
    Pi5PinMapping(30, -1, "GND",         "Ground",        ()),
    Pi5PinMapping(31,  6, "GPIO6",       "UART3 TX",      ("UART",)),
    Pi5PinMapping(32, 12, "GPIO12",      "PWM0/0",        ("PWM", "marine_esc_thrust")),
    Pi5PinMapping(33, 13, "GPIO13",      "PWM0/1",        ("PWM", "marine_esc_thrust_2")),
    Pi5PinMapping(34, -1, "GND",         "Ground",        ()),
    Pi5PinMapping(35, 19, "GPIO19",      "SPI1 MISO",     ("SPI",)),
    Pi5PinMapping(36, 16, "GPIO16",      "SPI1 CS0",      ("SPI",)),
    Pi5PinMapping(37, 26, "GPIO26",      "GPIO",          ("GPIO", "marine_heartbeat_led")),
    Pi5PinMapping(38, 20, "GPIO20",      "SPI1 MOSI",     ("SPI",)),
    Pi5PinMapping(39, -1, "GND",         "Ground",        ()),
    Pi5PinMapping(40, 21, "GPIO21",      "SPI1 SCLK",     ("SPI",)),
]


# ---------------------------------------------------------------------------
# Marine Sensor Pin Assignments (Pi 5 specific)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pi5SensorAssignment:
    """Sensor pin assignment for Pi 5."""
    sensor_name: str
    sensor_type: str
    interface: str
    pins: Dict[str, int]
    config_params: Dict[str, Any] = field(default_factory=dict)


PI5_MARINE_SENSORS: Dict[str, Pi5SensorAssignment] = {
    "ctd": Pi5SensorAssignment(
        sensor_name="CTD Sensor",
        sensor_type="ctd", interface="I2C-1",
        pins={"sda": 2, "scl": 3},
        config_params={"address": 0x66, "baud_rate": 400000},
    ),
    "imu": Pi5SensorAssignment(
        sensor_name="BNO055 9-DOF IMU",
        sensor_type="imu", interface="I2C-1",
        pins={"sda": 2, "scl": 3},
        config_params={"address": 0x28, "mode": "NDOF"},
    ),
    "gps": Pi5SensorAssignment(
        sensor_name="u-blox NEO-M8N GPS",
        sensor_type="gps", interface="UART0",
        pins={"tx": 14, "rx": 15},
        config_params={"baud_rate": 115200},
    ),
    "sonar": Pi5SensorAssignment(
        sensor_name="4K Sonar Imaging Module",
        sensor_type="sonar", interface="CSI-0",
        pins={"data_lanes": 4},
        config_params={"resolution": "4K@60", "protocol": "MIPI_CSI2"},
    ),
    "leak_sensor": Pi5SensorAssignment(
        sensor_name="Water Leak Detector",
        sensor_type="leak", interface="GPIO",
        pins={"alarm": 17},
        config_params={"active_low": True, "debounce_ms": 50},
    ),
    "esc_thrust": Pi5SensorAssignment(
        sensor_name="Thruster ESC",
        sensor_type="esc", interface="PWM0",
        pins={"signal": 12},
        config_params={"frequency_hz": 50, "pulse_range_us": (1000, 2000)},
    ),
}


# ---------------------------------------------------------------------------
# Thermal / Power Profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pi5ThermalProfile:
    """Thermal limits for Pi 5 — higher TDP than Pi 4B."""
    idle_temp_c: float = 50.0
    typical_load_c: float = 65.0
    throttle_start_c: float = 80.0
    critical_c: float = 85.0
    recommended_max_ambient_c: float = 50.0
    enclosure_note: str = (
        "Active cooling REQUIRED. The official active cooler or NVMe base "
        "with fan is mandatory for marine enclosures. Passive cooling is "
        "insufficient for continuous operation."
    )


@dataclass(frozen=True)
class Pi5PowerProfile:
    """Power consumption estimates for Pi 5."""
    idle_w: float = 4.0
    typical_load_w: float = 7.5
    max_load_w: float = 12.0
    usb_peripheral_budget_w: float = 6.0
    pcie_budget_w: float = 5.0
    voltage_range: Tuple[float, float] = (5.0, 5.25)
    recommended_psu_w: float = 27.0
    battery_life_estimate_ah: float = 2.5


# ---------------------------------------------------------------------------
# PCIe Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PCIeConfig:
    """PCIe 2.0 x1 configuration via RP1 on Pi 5."""
    version: str = "2.0"
    lanes: int = 1
    max_bandwidth_mbps: int = 500
    connector: str = "FPC (on-board, under board)"
    supported_devices: Tuple[str, ...] = (
        "NVMe SSD (HAT)",
        "AI accelerator (Hailo-8, Coral TPU)",
        "Multi-gigabit Ethernet",
        "SATA controller",
    )


# ---------------------------------------------------------------------------
# Main Configuration
# ---------------------------------------------------------------------------

class Pi5Config:
    """
    Complete hardware configuration for Raspberry Pi 5.

    Uses RP1 I/O controller for GPIO/SPI/I2C/UART instead of direct
    BCM pin muxing. Supports PCIe 2.0 x1 for expansion and 4K@60
    camera input via CSI.
    """

    def __init__(self):
        # SoC Identification
        self.board_model = "pi5"
        self.board_name = "Raspberry Pi 5"
        self.soc = "BCM2712"
        self.soc_manufacturer = "Broadcom Inc."
        self.io_controller = "RP1"       # Raspberry Pi South Bridge
        self.cpu_arch = "cortex-a76"
        self.cpu_cores = 4
        self.cpu_clock_max_mhz = 2400
        self.cpu_clock_min_mhz = 600
        self.cpu_clock_default_mhz = 2400
        self.gpu = "VideoCore VII @ 800 MHz"
        self.ram_type = "LPDDR4X-4267"
        self.ram_options_gb = (4, 8)
        self.default_ram_gb = 8

        # Peripheral Interfaces (via RP1)
        self.peripherals = {
            "i2c": {"count": 6, "max_speed_hz": 1_000_000, "pins": [(2, 3), (0, 1)]},
            "spi": {"count": 2, "max_speed_hz": 62_500_000, "pins": [(9, 10, 11), (19, 20, 21)]},
            "uart": {"count": 6, "pins": [(14, 15), (0, 1), (4, 5), (6, 7), (8, 9), (12, 13)]},
            "pwm": {"count": 4, "channels": 8, "frequency_range_hz": (1, 125_000_000)},
            "csi": {"count": 2, "lanes_per_port": 4, "max_resolution": "4K@60 / 8K30"},
            "dsi": {"count": 2, "lanes_per_port": 4},
            "pcie": {"version": "2.0", "lanes": 1, "max_bandwidth_mbps": 500},
            "usb": {"usb3_ports": 2, "usb2_ports": 2, "total_bandwidth_gbps": 10.0},
            "ethernet": {"speed_mbps": 1000, "phy": "on-board"},
            "wifi": {"standard": "802.11ac", "bands": ["2.4GHz", "5GHz"]},
            "bluetooth": {"version": "5.0", "ble": True},
            "rtc": True,
        }

        # GPIO
        self.gpio_pins = list(Pi5GPIOPin)
        self.pinout = PI5_PINOUT
        self.total_gpio = 28
        self.extra_rp1_gpio = 16  # Additional GPIO on RP1

        # Marine sensor mappings
        self.marine_sensor_mappings = PI5_MARINE_SENSORS
        self.thermal = Pi5ThermalProfile()
        self.power = Pi5PowerProfile()
        self.pcie = PCIeConfig()

        # NEXUS platform metadata
        self.nexus_platform = "raspberry_pi"
        self.nexus_role = "high_performance_compute"

        # Config fingerprint
        self._config_hash = self._compute_hash()

    # --- Public API ---

    def get_pin(self, rp1_gpio: int) -> Optional[Pi5PinMapping]:
        """Look up a pin mapping by RP1 GPIO number."""
        for pin in self.pinout:
            if pin.rp1_gpio == rp1_gpio:
                return pin
        return None

    def get_pin_by_physical(self, physical: int) -> Optional[Pi5PinMapping]:
        """Look up a pin mapping by physical header pin number."""
        for pin in self.pinout:
            if pin.physical_pin == physical:
                return pin
        return None

    def get_sensor_mapping(self, sensor_key: str) -> Optional[Pi5SensorAssignment]:
        """Get marine sensor assignment by key."""
        return self.marine_sensor_mappings.get(sensor_key)

    def list_sensor_mappings(self) -> List[str]:
        """List all available marine sensor mapping keys."""
        return sorted(self.marine_sensor_mappings.keys())

    def validate_gpio_assignment(self, rp1_gpio: int) -> bool:
        """Check if RP1 GPIO number is valid."""
        return rp1_gpio in [p.value for p in Pi5GPIOPin]

    def check_pin_conflict(
        self, sensor_key: str, other_sensor_key: str
    ) -> List[int]:
        """Check for GPIO conflicts between two sensor assignments."""
        s1 = self.get_sensor_mapping(sensor_key)
        s2 = self.get_sensor_mapping(other_sensor_key)
        if not s1 or not s2:
            return []
        return sorted(set(s1.pins.values()) & set(s2.pins.values()))

    def compute_power_budget(self, peripherals: List[str]) -> Dict[str, float]:
        """Estimate power draw given active peripherals."""
        peripheral_power = {
            "csi": 0.5, "dsi": 0.7, "ethernet": 0.5,
            "wifi": 0.6, "bluetooth": 0.2, "usb": 1.2,
            "pcie": 2.0, "fan": 0.3,
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
        """Return SHA-256 hash of this configuration."""
        return self._config_hash

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary dict."""
        return {
            "board": self.board_name,
            "soc": self.soc,
            "io_controller": self.io_controller,
            "cpu": f"{self.cpu_cores}x {self.cpu_arch} @ {self.cpu_clock_max_mhz}MHz",
            "ram_options_gb": list(self.ram_options_gb),
            "gpio_count": self.total_gpio,
            "extra_rp1_gpio": self.extra_rp1_gpio,
            "pcie": f"PCIe {self.pcie.version} x{self.pcie.lanes}",
            "csi_max": self.peripherals["csi"]["max_resolution"],
            "sensor_mappings": self.list_sensor_mappings(),
            "thermal_throttle_c": self.thermal.throttle_start_c,
            "power_idle_w": self.power.idle_w,
            "power_max_w": self.power.max_load_w,
            "config_hash": self._config_hash[:16],
        }

    # --- Internal ---

    def _compute_hash(self) -> str:
        data = (
            f"{self.soc}:{self.cpu_cores}:{self.cpu_clock_max_mhz}:"
            f"{self.total_gpio}:{self.extra_rp1_gpio}:"
            f"{len(self.marine_sensor_mappings)}:{self.pcie.version}:"
            f"{self.thermal.throttle_start_c}:{self.power.max_load_w}"
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def __repr__(self) -> str:
        return (
            f"Pi5Config(board='{self.board_name}', soc='{self.soc}', "
            f"io='{self.io_controller}', ram={self.default_ram_gb}GB, "
            f"gpio={self.total_gpio}, pcie='{self.pcie.version} x{self.pcie.lanes}')"
        )
