"""
NEXUS Marine Sensor HAT Configuration.

Provides an abstraction layer for multi-sensor marine boards (HATs) that
stack on top of Raspberry Pi boards. Manages sensor registration, bus
arbitration, conflict detection, and provides a unified read interface.

Supports:
  - I2C sensor multiplexing (TCA9548A for shared buses)
  - SPI sensor chip select management
  - UART sensor port allocation
  - GPIO interrupt aggregation
  - Power rail management for sensor modules

HAT Revision History:
  - v1: Basic I2C sensor board (CTD + IMU + pressure)
  - v2: Extended board (+ SPI ADC + leak sensor + CAN bus)
  - v3: Full navigation suite (+ GPS + echosounder + DVL-ready)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sensor Interface Types
# ---------------------------------------------------------------------------

class SensorInterface(Enum):
    """Supported sensor communication interfaces."""
    I2C = "i2c"
    SPI = "spi"
    UART = "uart"
    GPIO = "gpio"
    ANALOG = "analog"
    ONBOARD = "onboard"


class SensorCategory(Enum):
    """Marine sensor categories."""
    NAVIGATION = "navigation"
    ENVIRONMENTAL = "environmental"
    PROPULSION = "propulsion"
    SAFETY = "safety"
    COMMUNICATIONS = "communications"
    POWER = "power"


# ---------------------------------------------------------------------------
# Sensor Registration
# ---------------------------------------------------------------------------

@dataclass
class SensorRegistration:
    """Registration record for a sensor on the HAT."""
    sensor_id: str
    sensor_name: str
    interface: SensorInterface
    category: SensorCategory
    address: Optional[int] = None        # I2C address or SPI CS
    bus_number: int = 1                  # I2C/SPI bus
    pins: Dict[str, int] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    poll_interval_ms: int = 100
    priority: int = 5                    # 1=highest, 10=lowest

    def bus_key(self) -> str:
        """Unique key identifying this sensor's bus position."""
        if self.interface == SensorInterface.I2C:
            return f"i2c-{self.bus_number}-0x{self.address:02x}"
        elif self.interface == SensorInterface.SPI:
            return f"spi-{self.bus_number}-cs{self.address}"
        elif self.interface == SensorInterface.UART:
            return f"uart-{self.bus_number}"
        elif self.interface == SensorInterface.GPIO:
            return f"gpio-{self.pins.get('pin', 'unknown')}"
        return f"{self.interface.value}-{id(self)}"


# ---------------------------------------------------------------------------
# I2C Multiplexer Support
# ---------------------------------------------------------------------------

@dataclass
class I2CMultiplexer:
    """TCA9548A I2C multiplexer configuration for bus expansion."""
    mux_address: int = 0x70
    mux_bus: int = 1
    active_channels: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    channel_map: Dict[str, int] = field(default_factory=dict)

    def register_channel(self, channel_name: str, channel_num: int) -> None:
        if channel_num not in self.active_channels:
            raise ValueError(f"Channel {channel_num} not in active channels")
        self.channel_map[channel_name] = channel_num


# ---------------------------------------------------------------------------
# Power Rail Configuration
# ---------------------------------------------------------------------------

class PowerRail(Enum):
    """HAT power rail identifiers."""
    RAIL_3V3 = "3.3V"
    RAIL_5V = "5.0V"
    RAIL_12V = "12.0V"     # For external sensors requiring higher voltage
    RAIL_BATTERY = "battery"


@dataclass
class PowerRailConfig:
    """Configuration for a sensor power rail."""
    rail: PowerRail
    max_current_ma: float
    enabled_by_default: bool = True
    gpio_control: Optional[int] = None   # GPIO pin for on/off control


# ---------------------------------------------------------------------------
# HAT Hardware Definition
# ---------------------------------------------------------------------------

@dataclass
class HATHardware:
    """Physical HAT hardware specification."""
    hat_name: str
    revision: str
    board_compatibility: List[str]   # Compatible RPi board models
    eeprom_address: int = 0x50
    has_power_control: bool = False
    has_i2c_mux: bool = False
    has_adc: bool = False
    has_can: bool = False
    dimensions_mm: Tuple[float, float] = (65.0, 56.5)  # Standard HAT size
    stacking_height_mm: float = 12.0


# Pre-defined HAT hardware configs
HAT_DEFS: Dict[str, HATHardware] = {
    "nexus-marine-v1": HATHardware(
        hat_name="NEXUS Marine Sensor HAT v1",
        revision="v1",
        board_compatibility=["pi4b", "pi5", "cm4"],
        has_i2c_mux=False,
        has_adc=False,
        has_can=False,
    ),
    "nexus-marine-v2": HATHardware(
        hat_name="NEXUS Marine Sensor HAT v2",
        revision="v2",
        board_compatibility=["pi4b", "pi5", "cm4"],
        has_power_control=True,
        has_i2c_mux=True,
        has_adc=True,
        has_can=True,
    ),
    "nexus-marine-v3": HATHardware(
        hat_name="NEXUS Marine Navigation Suite v3",
        revision="v3",
        board_compatibility=["pi4b", "pi5", "cm4"],
        has_power_control=True,
        has_i2c_mux=True,
        has_adc=True,
        has_can=True,
    ),
}


# ---------------------------------------------------------------------------
# Default Sensor Catalog
# ---------------------------------------------------------------------------

DEFAULT_SENSOR_CATALOG: Dict[str, Dict[str, Any]] = {
    "bme280": {
        "name": "BME280 Barometric Pressure / Temp / Humidity",
        "interface": SensorInterface.I2C,
        "category": SensorCategory.ENVIRONMENTAL,
        "addresses": [0x76, 0x77],
        "poll_interval_ms": 500,
    },
    "ms5837": {
        "name": "MS5837-30BA Water Pressure Sensor",
        "interface": SensorInterface.I2C,
        "category": SensorCategory.NAVIGATION,
        "addresses": [0x76],
        "poll_interval_ms": 20,
        "max_depth_m": 300.0,
    },
    "bno055": {
        "name": "BNO055 9-DOF IMU",
        "interface": SensorInterface.I2C,
        "category": SensorCategory.NAVIGATION,
        "addresses": [0x28, 0x29],
        "poll_interval_ms": 10,
    },
    "ina219": {
        "name": "INA219 Power Monitor",
        "interface": SensorInterface.I2C,
        "category": SensorCategory.POWER,
        "addresses": [0x40, 0x41, 0x44, 0x45],
        "poll_interval_ms": 1000,
    },
    "mcp2515": {
        "name": "MCP2515 CAN Controller (NMEA 2000)",
        "interface": SensorInterface.SPI,
        "category": SensorCategory.COMMUNICATIONS,
        "poll_interval_ms": 1,
    },
    "max11616": {
        "name": "MAX11616 12-bit ADC (Dissolved O2)",
        "interface": SensorInterface.SPI,
        "category": SensorCategory.ENVIRONMENTAL,
        "poll_interval_ms": 100,
    },
    "leak_sensor": {
        "name": "Water Ingress Leak Sensor",
        "interface": SensorInterface.GPIO,
        "category": SensorCategory.SAFETY,
        "poll_interval_ms": 50,
        "critical": True,
    },
    "gps_ublox": {
        "name": "u-blox NEO-M8N GPS Receiver",
        "interface": SensorInterface.UART,
        "category": SensorCategory.NAVIGATION,
        "poll_interval_ms": 100,
    },
    "echosounder": {
        "name": "Blue Robotics Ping Echosounder",
        "interface": SensorInterface.UART,
        "category": SensorCategory.NAVIGATION,
        "poll_interval_ms": 200,
    },
}


# ---------------------------------------------------------------------------
# Main Sensor HAT Class
# ---------------------------------------------------------------------------

class MarineSensorHAT:
    """
    Multi-sensor marine HAT manager for NEXUS platform.

    Handles sensor registration, bus conflict detection, and provides
    a unified interface for reading all sensors.

    Usage:
        hat = MarineSensorHAT(board="pi4b", revision="v2")
        hat.register_sensor("ms5837", bus="i2c", address=0x76)
        hat.register_sensor("bno055", bus="i2c", address=0x28)
        hat.register_sensor("leak_sensor", bus="gpio", pins={"pin": 17})
        report = hat.read_all()
    """

    def __init__(
        self,
        board: str = "pi4b",
        revision: str = "v2",
        hat_id: Optional[str] = None,
    ):
        self.board = board
        self.revision = revision
        self.hat_id = hat_id or f"nexus-marine-{revision}"

        # Load HAT hardware definition
        if self.hat_id in HAT_DEFS:
            self.hardware = HAT_DEFS[self.hat_id]
        else:
            self.hardware = HATHardware(
                hat_name=f"NEXUS Custom HAT ({self.hat_id})",
                revision=revision,
                board_compatibility=[board],
            )

        # Sensor registry
        self._sensors: Dict[str, SensorRegistration] = {}
        self._i2c_mux: Optional[I2CMultiplexer] = None
        self._power_rails: Dict[PowerRail, PowerRailConfig] = {}

        # Initialize default power rails
        self._init_power_rails()

    # --- Power Rail Management ---

    def _init_power_rails(self) -> None:
        """Initialize default power rails for the HAT."""
        self._power_rails = {
            PowerRail.RAIL_3V3: PowerRailConfig(
                rail=PowerRail.RAIL_3V3, max_current_ma=800
            ),
            PowerRail.RAIL_5V: PowerRailConfig(
                rail=PowerRail.RAIL_5V, max_current_ma=2000
            ),
        }

    def add_power_rail(self, config: PowerRailConfig) -> None:
        """Add a custom power rail (e.g., 12V for external sensors)."""
        self._power_rails[config.rail] = config

    def get_power_budget_ma(self) -> Dict[str, float]:
        """Calculate total power budget across all rails."""
        return {
            rail.value: cfg.max_current_ma
            for rail, cfg in self._power_rails.items()
        }

    # --- I2C Multiplexer ---

    def enable_i2c_mux(self, mux_address: int = 0x70, bus: int = 1) -> None:
        """Enable I2C multiplexer for shared bus expansion."""
        self._i2c_mux = I2CMultiplexer(mux_address=mux_address, mux_bus=bus)

    def register_mux_channel(self, name: str, channel: int) -> None:
        """Register a named channel on the I2C multiplexer."""
        if not self._i2c_mux:
            raise RuntimeError("I2C multiplexer not enabled")
        self._i2c_mux.register_channel(name, channel)

    # --- Sensor Registration ---

    def register_sensor(
        self,
        sensor_id: str,
        bus: str = "i2c",
        address: Optional[int] = None,
        pins: Optional[Dict[str, int]] = None,
        config: Optional[Dict[str, Any]] = None,
        poll_interval_ms: int = 100,
        priority: int = 5,
    ) -> SensorRegistration:
        """
        Register a sensor on the HAT.

        Args:
            sensor_id: Unique sensor identifier (e.g., 'ms5837', 'bno055').
            bus: Bus interface ('i2c', 'spi', 'uart', 'gpio').
            address: I2C address or SPI chip select number.
            pins: Pin assignments dict for GPIO/UART sensors.
            config: Additional sensor configuration parameters.
            poll_interval_ms: Polling interval in milliseconds.
            priority: Sensor priority (1=highest, 10=lowest).

        Returns:
            SensorRegistration instance.

        Raises:
            ValueError: If sensor_id is already registered or bus conflict.
        """
        if sensor_id in self._sensors:
            raise ValueError(f"Sensor '{sensor_id}' already registered")

        interface_map = {
            "i2c": SensorInterface.I2C,
            "spi": SensorInterface.SPI,
            "uart": SensorInterface.UART,
            "gpio": SensorInterface.GPIO,
            "analog": SensorInterface.ANALOG,
            "onboard": SensorInterface.ONBOARD,
        }

        iface = interface_map.get(bus.lower())
        if iface is None:
            raise ValueError(f"Unknown bus interface: {bus}")

        # Determine sensor category from catalog
        catalog_entry = DEFAULT_SENSOR_CATALOG.get(sensor_id, {})
        category = catalog_entry.get(
            "category", SensorCategory.ENVIRONMENTAL
        )
        sensor_name = catalog_entry.get("name", f"Sensor {sensor_id}")

        # Determine bus number
        bus_num = 1
        if bus.lower() == "spi" and pins and "bus" in pins:
            bus_num = pins["bus"]
        elif bus.lower() == "uart" and config and "uart_num" in config:
            bus_num = config["uart_num"]

        registration = SensorRegistration(
            sensor_id=sensor_id,
            sensor_name=sensor_name,
            interface=iface,
            category=category,
            address=address,
            bus_number=bus_num,
            pins=pins or {},
            config=config or {},
            poll_interval_ms=poll_interval_ms,
            priority=priority,
        )

        # Check for bus conflicts
        conflicts = self._detect_conflicts(registration)
        if conflicts:
            conflict_str = ", ".join(
                f"{c.sensor_id} ({c.bus_key()})" for c in conflicts
            )
            raise ValueError(
                f"Bus conflict for '{sensor_id}' on {registration.bus_key()}: "
                f"conflicts with {conflict_str}"
            )

        self._sensors[sensor_id] = registration
        return registration

    def unregister_sensor(self, sensor_id: str) -> None:
        """Remove a sensor registration."""
        if sensor_id not in self._sensors:
            raise KeyError(f"Sensor '{sensor_id}' not registered")
        del self._sensors[sensor_id]

    def _detect_conflicts(
        self, registration: SensorRegistration
    ) -> List[SensorRegistration]:
        """Detect bus conflicts with existing sensor registrations."""
        conflicts = []
        new_key = registration.bus_key()
        for existing in self._sensors.values():
            if existing.bus_key() == new_key and existing.interface in (
                SensorInterface.I2C,
                SensorInterface.SPI,
                SensorInterface.UART,
            ):
                conflicts.append(existing)
        return conflicts

    def check_compatibility(self, board: str) -> bool:
        """Check if this HAT is compatible with the given board model."""
        return board in self.hardware.board_compatibility

    # --- Read Interface ---

    def read_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Read all registered sensors (mock — in production, this calls
        actual hardware drivers).

        Returns:
            Dict mapping sensor_id to reading data.
        """
        results = {}
        for sensor_id, reg in self._sensors.items():
            if not reg.enabled:
                continue
            results[sensor_id] = {
                "sensor_name": reg.sensor_name,
                "interface": reg.interface.value,
                "bus_key": reg.bus_key(),
                "category": reg.category.value,
                "status": "simulated",
                "reading": None,  # Placeholder for actual sensor read
            }
        return results

    def read_sensor(self, sensor_id: str) -> Dict[str, Any]:
        """Read a specific sensor by ID."""
        reg = self._sensors.get(sensor_id)
        if not reg:
            raise KeyError(f"Sensor '{sensor_id}' not registered")
        if not reg.enabled:
            return {"status": "disabled"}
        return {
            "sensor_name": reg.sensor_name,
            "interface": reg.interface.value,
            "bus_key": reg.bus_key(),
            "status": "simulated",
            "reading": None,
        }

    # --- Query Methods ---

    def list_sensors(self) -> List[str]:
        """List all registered sensor IDs."""
        return sorted(self._sensors.keys())

    def list_sensors_by_category(
        self, category: SensorCategory
    ) -> List[str]:
        """List sensors filtered by category."""
        return sorted(
            sid for sid, reg in self._sensors.items()
            if reg.category == category
        )

    def list_sensors_by_interface(
        self, interface: SensorInterface
    ) -> List[str]:
        """List sensors filtered by interface type."""
        return sorted(
            sid for sid, reg in self._sensors.items()
            if reg.interface == interface
        )

    def get_sensor(self, sensor_id: str) -> Optional[SensorRegistration]:
        """Get sensor registration by ID."""
        return self._sensors.get(sensor_id)

    def sensor_count(self) -> int:
        """Count of registered sensors."""
        return len(self._sensors)

    def summary(self) -> Dict[str, Any]:
        """Return HAT summary including all registered sensors."""
        return {
            "hat_id": self.hat_id,
            "hat_name": self.hardware.hat_name,
            "revision": self.revision,
            "board": self.board,
            "compatible_boards": self.hardware.board_compatibility,
            "sensor_count": self.sensor_count(),
            "sensors": self.list_sensors(),
            "i2c_mux": self._i2c_mux is not None,
            "power_rails": self.get_power_budget_ma(),
        }

    def get_config_hash(self) -> str:
        """Compute a hash of the current HAT configuration."""
        sensor_str = "|".join(
            f"{sid}:{reg.bus_key()}" for sid, reg in sorted(self._sensors.items())
        )
        data = f"{self.hat_id}:{self.revision}:{sensor_str}".encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def __repr__(self) -> str:
        return (
            f"MarineSensorHAT(hat='{self.hat_id}', board='{self.board}', "
            f"sensors={self.sensor_count()})"
        )
