"""
NEXUS Marine Sensor Driver Configuration Module.

Provides dataclass-based configuration objects for each sensor type commonly
used in marine robotics: GPS, IMU, sonar, pressure, temperature, servo
actuators, and motor / thruster controllers.  Every config class validates
its fields at construction time and exposes helper methods for protocol-
specific serial setup.

All numerical ranges enforce sensible marine-environment bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Interface(Enum):
    """Communication interface identifiers."""

    UART = "UART"
    SPI = "SPI"
    I2C = "I2C"
    ONE_WIRE = "1WIRE"
    ANALOG = "ANALOG"


# ---------------------------------------------------------------------------
# GPS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPSSensorConfig:
    """Configuration for a GPS receiver (NMEA 0183).

    Typical marine GPS modules operate at 4800 or 9600 baud and emit NMEA
    sentences at 1–10 Hz.
    """

    baud_rate: int = 9600
    update_rate_hz: float = 5.0
    protocol: str = "NMEA"

    # NMEA sentence filter — only these talker IDs are forwarded
    nmea_filter: Tuple[str, ...] = ("GGA", "RMC", "VTG", "GSA")

    # Serial port on Arduino (Uno: "Serial", Mega: "Serial1", etc.)
    serial_port: str = "Serial"

    def __post_init__(self) -> None:
        if self.baud_rate <= 0:
            raise ValueError("baud_rate must be positive")
        if not (0.1 <= self.update_rate_hz <= 20.0):
            raise ValueError("update_rate_hz must be between 0.1 and 20.0")

    def serial_params(self) -> dict:
        """Return a dict suitable for ``pyserial.Serial(...)``."""
        return {"baudrate": self.baud_rate, "timeout": 1.0}


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IMUSensorConfig:
    """Configuration for an inertial measurement unit.

    Supports both SPI and I2C interfaces.  Common marine IMUs include the
    MPU-6050, MPU-9250, BNO-055, and ADIS16470.
    """

    accel_range_g: float = 16.0     # ±16 g
    gyro_range_dps: float = 2000.0  # ±2000 °/s
    mag_range_uT: float = 4800.0    # ±4800 µT
    protocol: str = "I2C"

    # I2C address (default for MPU-9250 in I2C mode)
    i2c_address: int = 0x68

    # SPI chip-select pin (-1 if using I2C)
    spi_cs_pin: int = -1

    # Enable / disable individual axes
    enable_accel: bool = True
    enable_gyro: bool = True
    enable_mag: bool = True

    def __post_init__(self) -> None:
        valid_protocols = {"SPI", "I2C"}
        if self.protocol not in valid_protocols:
            raise ValueError(f"protocol must be one of {valid_protocols}")
        if self.accel_range_g <= 0:
            raise ValueError("accel_range_g must be positive")
        if self.gyro_range_dps <= 0:
            raise ValueError("gyro_range_dps must be positive")
        if self.mag_range_uT < 0:
            raise ValueError("mag_range_uT must be non-negative")

    @property
    def interface(self) -> Interface:
        return Interface.SPI if self.protocol == "SPI" else Interface.I2C


# ---------------------------------------------------------------------------
# Sonar
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SonarConfig:
    """Configuration for an ultrasonic distance sensor (e.g. HC-SR04).

    Marine-rated sonar modules (JSN-SR04T, A02YYUW) use the same trigger/
    echo interface and can operate up to 4.5 m in water.
    """

    max_range_cm: float = 400.0    # 4 m default
    min_range_cm: float = 2.0
    trigger_pulse_us: int = 10     # µs
    echo_timeout_us: int = 30_000  # 30 ms  (~5 m round-trip)
    speed_of_sound_cm_us: float = 0.034  # cm/µs in air (adjust for water)

    # Sound speed in water ≈ 0.015 cm/µs — override for underwater use
    medium: str = "air"  # "air" or "water"

    def __post_init__(self) -> None:
        if self.max_range_cm <= self.min_range_cm:
            raise ValueError("max_range_cm must exceed min_range_cm")
        if self.trigger_pulse_us <= 0:
            raise ValueError("trigger_pulse_us must be positive")
        if self.echo_timeout_us <= 0:
            raise ValueError("echo_timeout_us must be positive")
        if self.speed_of_sound_cm_us <= 0:
            raise ValueError("speed_of_sound_cm_us must be positive")

    @property
    def effective_timeout_ms(self) -> float:
        """Timeout in milliseconds."""
        return self.echo_timeout_us / 1000.0

    def distance_cm(self, echo_us: float) -> float:
        """Convert raw echo pulse duration to distance in cm."""
        return (echo_us * self.speed_of_sound_cm_us) / 2.0


# ---------------------------------------------------------------------------
# Pressure / depth sensor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PressureSensorConfig:
    """Configuration for a pressure sensor used for depth measurement.

    Common marine pressure sensors: MS5837-30BA (300 bar), MS5837-02BA
    (2 bar), MPXHZ6400A series.  Depth is derived from hydrostatic pressure.
    """

    range_kpa: float = 300.0       # 300 kPa for shallow-water MS5837
    resolution_bits: int = 24
    protocol: str = "I2C"

    # I2C address (default for MS5837)
    i2c_address: int = 0x76

    # Water density for depth conversion (kg/m³, seawater ≈ 1025)
    water_density_kg_m3: float = 1025.0

    # Gravity (m/s²)
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        valid_protocols = {"I2C", "SPI", "ANALOG"}
        if self.protocol not in valid_protocols:
            raise ValueError(f"protocol must be one of {valid_protocols}")
        if self.range_kpa <= 0:
            raise ValueError("range_kpa must be positive")
        if self.resolution_bits not in {8, 10, 12, 16, 24}:
            raise ValueError("resolution_bits must be 8, 10, 12, 16, or 24")

    def pressure_to_depth_m(self, pressure_kpa: float) -> float:
        """Convert gauge pressure (kPa) to depth in metres."""
        p_pa = pressure_kpa * 1000.0
        return p_pa / (self.water_density_kg_m3 * self.gravity_m_s2)


# ---------------------------------------------------------------------------
# Temperature sensor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemperatureSensorConfig:
    """Configuration for a temperature sensor.

    Supports 1-Wire (DS18B20), I2C (MCP9808, TMP117), and analog (NTC
    thermistor with voltage divider) interfaces.
    """

    range_celsius: Tuple[float, float] = (-40.0, 85.0)  # DS18B20 range
    resolution_bits: int = 12
    protocol: str = "1WIRE"

    # 1-Wire data pin (-1 for I2C)
    one_wire_pin: int = -1

    # I2C address (for I2C sensors)
    i2c_address: int = 0x48

    # Analog reference voltage (for NTC thermistor)
    analog_ref_v: float = 5.0

    # Beta coefficient for NTC thermistor (Steinhart-Hart simplified)
    ntc_beta: float = 3950.0
    ntc_r25_ohm: float = 10000.0  # 10 kΩ @ 25 °C

    def __post_init__(self) -> None:
        valid_protocols = {"1WIRE", "I2C", "ANALOG"}
        if self.protocol not in valid_protocols:
            raise ValueError(f"protocol must be one of {valid_protocols}")
        if self.range_celsius[0] >= self.range_celsius[1]:
            raise ValueError("range_celsius must be (min, max) with min < max")
        if self.resolution_bits not in {9, 10, 11, 12}:
            raise ValueError("resolution_bits must be 9, 10, 11, or 12")


# ---------------------------------------------------------------------------
# Servo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServoConfig:
    """Configuration for a standard PWM servo (rudder, ballast valve, etc.)."""

    min_pulse_us: int = 1000       # 1 ms  (0°)
    max_pulse_us: int = 2000       # 2 ms  (180°)
    frequency_hz: int = 50         # Standard RC servo rate

    # Pin assigned to this servo (-1 = not assigned)
    pin: int = -1

    # Initial angle (degrees)
    initial_angle_deg: float = 90.0

    # Invert direction (useful for mirrored servo mounts)
    invert: bool = False

    def __post_init__(self) -> None:
        if self.min_pulse_us <= 0:
            raise ValueError("min_pulse_us must be positive")
        if self.max_pulse_us <= self.min_pulse_us:
            raise ValueError("max_pulse_us must exceed min_pulse_us")
        if not (1 <= self.frequency_hz <= 333):
            raise ValueError("frequency_hz must be between 1 and 333")
        if not (0.0 <= self.initial_angle_deg <= 180.0):
            raise ValueError("initial_angle_deg must be between 0 and 180")

    @property
    def pulse_range_us(self) -> int:
        return self.max_pulse_us - self.min_pulse_us

    def angle_to_pulse(self, angle_deg: float) -> int:
        """Map an angle in degrees to a pulse width in microseconds."""
        if not (0.0 <= angle_deg <= 180.0):
            raise ValueError("angle_deg must be between 0 and 180")
        ratio = angle_deg / 180.0
        pulse = int(self.min_pulse_us + ratio * self.pulse_range_us)
        if self.invert:
            pulse = self.max_pulse_us - (pulse - self.min_pulse_us)
        return pulse


# ---------------------------------------------------------------------------
# Motor / thruster controller
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorControllerConfig:
    """Configuration for a brushless ESC or brushed motor controller.

    Typical AUV/ROV thrusters use PWM in the 1000–2000 µs range at 50 Hz,
    identical to standard RC servo signals.
    """

    max_current_a: float = 30.0
    pwm_frequency_hz: int = 50
    channels: int = 1             # Single or dual ESC

    min_throttle_us: int = 1000   # Armed / zero thrust
    max_throttle_us: int = 2000   # Full throttle

    # Pin assigned to this ESC channel
    pins: Tuple[int, ...] = (10,)

    # Arm delay (ms) — ESCs require a brief low-throttle signal to arm
    arm_delay_ms: int = 2000

    # Enable reverse thrust (bidirectional ESCs)
    bidirectional: bool = False

    def __post_init__(self) -> None:
        if self.max_current_a <= 0:
            raise ValueError("max_current_a must be positive")
        if self.channels < 1:
            raise ValueError("channels must be >= 1")
        if self.min_throttle_us <= 0:
            raise ValueError("min_throttle_us must be positive")
        if self.max_throttle_us <= self.min_throttle_us:
            raise ValueError("max_throttle_us must exceed min_throttle_us")
        if len(self.pins) != self.channels:
            raise ValueError("len(pins) must equal channels")

    @property
    def throttle_range_us(self) -> int:
        return self.max_throttle_us - self.min_throttle_us

    def throttle_to_pulse(self, fraction: float) -> int:
        """Map a throttle fraction [0.0, 1.0] to a pulse width in µs."""
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("fraction must be between 0.0 and 1.0")
        return int(self.min_throttle_us + fraction * self.throttle_range_us)
