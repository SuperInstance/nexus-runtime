"""
NEXUS Raspberry Pi Zero 2 W Configuration — BCM2710 SoC.

Target hardware:
  - SoC:    Broadcom BCM2710, Quad-core Cortex-A53 @ 1.0 GHz
  - RAM:    512 MB LPDDR2
  - GPIO:   40-pin header (mini pinout, 28 usable)
  - WiFi:   802.11 b/g/n (2.4 GHz) + Bluetooth 4.2 / BLE
  - CSI:    1× MIPI CSI-2 (2-lane)
  - USB:    1× Micro-USB 2.0 OTG
  - Video:  mini-HDMI (1080p30)
  - Form factor: 65 × 30 mm, ~12 g

Marine robotics use: lightweight sensor nodes, BLE telemetry beacons,
distributed sensor networks, buoy controllers, low-power ROV subsystems.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GPIO Pin Definitions (Zero 2W — smaller header subset)
# ---------------------------------------------------------------------------

class Zero2WGPIOPin(IntEnum):
    """BCM GPIO pins available on the Pi Zero 2W mini header."""
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
class Zero2WPinMapping:
    """Pin mapping for Pi Zero 2W 40-pin mini header."""
    physical_pin: int
    bcm_gpio: int
    name: str
    default_function: str
    alt_functions: Tuple[str, ...]
    available: bool = True  # Not all pins are usable on Zero


# Zero 2W pinout (same physical layout but only 28 GPIOs accessible)
ZERO2W_PINOUT: List[Zero2WPinMapping] = [
    Zero2WPinMapping(1,  -1, "3V3",     "3.3V power",    ()),
    Zero2WPinMapping(2,  -1, "5V",      "5.0V power",    ()),
    Zero2WPinMapping(3,  2,  "GPIO2",   "I2C1 SDA",      ("I2C",)),
    Zero2WPinMapping(4,  -1, "5V",      "5.0V power",    ()),
    Zero2WPinMapping(5,  3,  "GPIO3",   "I2C1 SCL",      ("I2C",)),
    Zero2WPinMapping(6,  -1, "GND",     "Ground",        ()),
    Zero2WPinMapping(7,  4,  "GPIO4",   "GPIO",          ("GPIO",)),
    Zero2WPinMapping(8,  14, "GPIO14",  "UART0 TX",      ("UART0",)),
    Zero2WPinMapping(9,  -1, "GND",     "Ground",        ()),
    Zero2WPinMapping(10, 15, "GPIO15",  "UART0 RX",      ("UART0",)),
    Zero2WPinMapping(11, 17, "GPIO17",  "GPIO",          ("GPIO",)),
    Zero2WPinMapping(12, 18, "GPIO18",  "PWM0",          ("PWM0",)),
    Zero2WPinMapping(13, 27, "GPIO27",  "GPIO",          ("GPIO",)),
    Zero2WPinMapping(14, -1, "GND",     "Ground",        ()),
    Zero2WPinMapping(15, 22, "GPIO22",  "GPIO",          ("GPIO",)),
    Zero2WPinMapping(16, 23, "GPIO23",  "SPI0 SCK",      ("SPI0",)),
    Zero2WPinMapping(17, -1, "3V3",     "3.3V power",    ()),
    Zero2WPinMapping(18, 24, "GPIO24",  "GPIO",          ("GPIO",)),
    Zero2WPinMapping(19, 10, "GPIO10",  "SPI0 MOSI",     ("SPI0",)),
    Zero2WPinMapping(20, -1, "GND",     "Ground",        ()),
    Zero2WPinMapping(21,  9,  "GPIO9",   "SPI0 MISO",     ("SPI0",)),
    Zero2WPinMapping(22, 25, "GPIO25",  "GPIO",          ("GPIO",)),
    Zero2WPinMapping(23, 11, "GPIO11",  "SPI0 SCLK",     ("SPI0",)),
    Zero2WPinMapping(24,  8,  "GPIO8",   "SPI0 CE0",      ("SPI0",)),
    Zero2WPinMapping(25, -1, "GND",     "Ground",        ()),
    Zero2WPinMapping(26,  7,  "GPIO7",   "SPI0 CE1",      ("SPI0",)),
    Zero2WPinMapping(27,  0,  "ID_SD",   "I2C0 SDA",      ("I2C0",)),
    Zero2WPinMapping(28,  1,  "ID_SC",   "I2C0 SCL",      ("I2C0",)),
    Zero2WPinMapping(29,  5,  "GPIO5",   "GPIO",          ("GPIO",)),
    Zero2WPinMapping(30, -1, "GND",     "Ground",        ()),
    Zero2WPinMapping(31,  6,  "GPIO6",   "GPIO",          ("GPIO",)),
    Zero2WPinMapping(32, 12, "GPIO12",  "PWM0",          ("PWM0",)),
    Zero2WPinMapping(33, 13, "GPIO13",  "PWM1",          ("PWM1",)),
    Zero2WPinMapping(34, -1, "GND",     "Ground",        ()),
    Zero2WPinMapping(35, 19, "GPIO19",  "SPI1 MISO",     ("SPI1",)),
    Zero2WPinMapping(36, 16, "GPIO16",  "GPIO",          ("GPIO",)),
    Zero2WPinMapping(37, 26, "GPIO26",  "GPIO",          ("GPIO",)),
    Zero2WPinMapping(38, 20, "GPIO20",  "SPI1 MOSI",     ("SPI1",)),
    Zero2WPinMapping(39, -1, "GND",     "Ground",        ()),
    Zero2WPinMapping(40, 21, "GPIO21",  "SPI1 SCLK",     ("SPI1",)),
]


# ---------------------------------------------------------------------------
# Sensor Node Presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Zero2WSensorPreset:
    """Pre-configured sensor setup for specific marine use cases."""
    preset_name: str
    description: str
    sensors: Dict[str, Dict[str, Any]]
    power_estimate_mw: float
    sample_rate_hz: float


ZERO2W_SENSOR_PRESETS: Dict[str, Zero2WSensorPreset] = {
    "buoy_ctd": Zero2WSensorPreset(
        preset_name="Buoy CTD Node",
        description="Compact CTD sensor node for buoy deployment",
        sensors={
            "ctd": {"interface": "I2C-1", "pins": {"sda": 2, "scl": 3}, "address": 0x66},
            "leak": {"interface": "GPIO", "pins": {"alarm": 17}},
            "batt_monitor": {"interface": "I2C-1", "pins": {"sda": 2, "scl": 3}, "address": 0x36},
        },
        power_estimate_mw=850.0,
        sample_rate_hz=2.0,
    ),
    "rov_depth": Zero2WSensorPreset(
        preset_name="ROV Depth Monitor",
        description="Depth and temperature sensor for ROV hull",
        sensors={
            "ms5837": {"interface": "I2C-1", "pins": {"sda": 2, "scl": 3}, "address": 0x76},
            "imu": {"interface": "I2C-1", "pins": {"sda": 2, "scl": 3}, "address": 0x28},
        },
        power_estimate_mw=720.0,
        sample_rate_hz=10.0,
    ),
    "telemetry_beacon": Zero2WSensorPreset(
        preset_name="BLE Telemetry Beacon",
        description="Low-power BLE beacon for surface telemetry relay",
        sensors={
            "ble_radio": {"interface": "onboard"},
            "gps": {"interface": "UART0", "pins": {"tx": 14, "rx": 15}},
            "led": {"interface": "GPIO", "pins": {"signal": 26}},
        },
        power_estimate_mw=550.0,
        sample_rate_hz=1.0,
    ),
}


# ---------------------------------------------------------------------------
# Thermal / Power Profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Zero2WThermalProfile:
    """Thermal limits for Pi Zero 2W — very low power."""
    idle_temp_c: float = 35.0
    typical_load_c: float = 48.0
    throttle_start_c: float = 80.0
    critical_c: float = 85.0
    recommended_max_ambient_c: float = 60.0
    enclosure_note: str = (
        "Passive conduction cooling sufficient. Can be fully potted in "
        "marine epoxy for IP68 rated deployments. No heatsink needed."
    )


@dataclass(frozen=True)
class Zero2WPowerProfile:
    """Power consumption estimates for Pi Zero 2W."""
    idle_w: float = 0.7
    typical_load_w: float = 1.0
    max_load_w: float = 1.5
    voltage_range: Tuple[float, float] = (5.0, 5.25)
    usb_power_in_max_w: float = 2.5
    battery_life_estimate_ah: float = 0.35
    solar_compatible: bool = True
    recommended_battery_mah: int = 5000  # For 24hr operation


# ---------------------------------------------------------------------------
# Main Configuration
# ---------------------------------------------------------------------------

class PiZero2WConfig:
    """
    Complete hardware configuration for Raspberry Pi Zero 2 W.

    Optimized for marine sensor node deployments: ultra-low power,
    compact form factor, WiFi/BLE for telemetry, single CSI for
    basic camera/sonar input.
    """

    def __init__(self):
        # SoC Identification
        self.board_model = "pizerow"
        self.board_name = "Raspberry Pi Zero 2 W"
        self.soc = "BCM2710"
        self.soc_manufacturer = "Broadcom Inc."
        self.cpu_arch = "cortex-a53"
        self.cpu_cores = 4
        self.cpu_clock_max_mhz = 1000
        self.cpu_clock_min_mhz = 600
        self.cpu_clock_default_mhz = 1000
        self.gpu = "VideoCore IV @ 250 MHz"
        self.ram_type = "LPDDR2"
        self.ram_options_gb = (0.5,)
        self.default_ram_gb = 0.5
        self.form_factor = "65 × 30 × 5 mm, ~12 g"

        # Peripheral Interfaces
        self.peripherals = {
            "i2c": {"count": 1, "max_speed_hz": 400_000, "pins": [(2, 3)]},
            "spi": {"count": 1, "max_speed_hz": 62_500_000, "pins": [(9, 10, 11)]},
            "uart": {"count": 1, "pins": [(14, 15)]},
            "pwm": {"count": 1, "channels": 2, "frequency_range_hz": (1, 40_000)},
            "csi": {"count": 1, "lanes_per_port": 2, "max_resolution": "1080p30"},
            "usb": {"usb2_ports": 1, "otg": True},
            "wifi": {"standard": "802.11b/g/n", "bands": ["2.4GHz"], "antenna": "onboard"},
            "bluetooth": {"version": "4.2", "ble": True},
        }

        # GPIO
        self.gpio_pins = list(Zero2WGPIOPin)
        self.pinout = ZERO2W_PINOUT
        self.total_gpio = 28

        # Marine sensor presets
        self.sensor_presets = ZERO2W_SENSOR_PRESETS
        self.thermal = Zero2WThermalProfile()
        self.power = Zero2WPowerProfile()

        # NEXUS platform metadata
        self.nexus_platform = "raspberry_pi"
        self.nexus_role = "sensor_node"

        # Config fingerprint
        self._config_hash = self._compute_hash()

    # --- Public API ---

    def get_pin(self, bcm_gpio: int) -> Optional[Zero2WPinMapping]:
        """Look up a pin mapping by BCM GPIO number."""
        for pin in self.pinout:
            if pin.bcm_gpio == bcm_gpio:
                return pin
        return None

    def get_pin_by_physical(self, physical: int) -> Optional[Zero2WPinMapping]:
        """Look up a pin mapping by physical header pin number."""
        for pin in self.pinout:
            if pin.physical_pin == physical:
                return pin
        return None

    def get_sensor_preset(self, preset_key: str) -> Optional[Zero2WSensorPreset]:
        """Get a sensor node preset by key."""
        return self.sensor_presets.get(preset_key)

    def list_sensor_presets(self) -> List[str]:
        """List all available sensor node presets."""
        return sorted(self.sensor_presets.keys())

    def validate_gpio_assignment(self, bcm_gpio: int) -> bool:
        """Check if BCM GPIO number is valid for this board."""
        return bcm_gpio in [p.value for p in Zero2WGPIOPin]

    def estimate_battery_life(
        self, battery_mah: float, load_factor: float = 0.8
    ) -> float:
        """
        Estimate battery life in hours.

        Args:
            battery_mah: Battery capacity in mAh.
            load_factor: Fraction of max power draw (0.0-1.0).

        Returns:
            Estimated operating hours.
        """
        avg_draw_w = self.power.idle_w + (
            (self.power.max_load_w - self.power.idle_w) * load_factor
        )
        draw_ma = (avg_draw_w / 5.0) * 1000  # Approximate at 5V
        return round(battery_mah / draw_ma, 1)

    def get_config_hash(self) -> str:
        """Return SHA-256 hash of this configuration."""
        return self._config_hash

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary dict."""
        return {
            "board": self.board_name,
            "soc": self.soc,
            "cpu": f"{self.cpu_cores}x {self.cpu_arch} @ {self.cpu_clock_max_mhz}MHz",
            "ram_gb": self.default_ram_gb,
            "form_factor": self.form_factor,
            "gpio_count": self.total_gpio,
            "sensor_presets": self.list_sensor_presets(),
            "thermal_throttle_c": self.thermal.throttle_start_c,
            "power_idle_w": self.power.idle_w,
            "power_max_w": self.power.max_load_w,
            "solar_compatible": self.power.solar_compatible,
            "config_hash": self._config_hash[:16],
        }

    # --- Internal ---

    def _compute_hash(self) -> str:
        data = (
            f"{self.soc}:{self.cpu_cores}:{self.cpu_clock_max_mhz}:"
            f"{self.total_gpio}:{len(self.sensor_presets)}:"
            f"{self.thermal.throttle_start_c}:{self.power.max_load_w}"
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def __repr__(self) -> str:
        return (
            f"PiZero2WConfig(board='{self.board_name}', soc='{self.soc}', "
            f"ram={self.default_ram_gb}GB, gpio={self.total_gpio}, "
            f"role='{self.nexus_role}')"
        )
