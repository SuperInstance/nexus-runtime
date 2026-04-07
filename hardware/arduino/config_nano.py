"""
NEXUS Arduino Nano Hardware Configuration Module.

Defines board-level metadata, serial communication parameters, pin mappings
for common marine sensors, and NEXUS wire protocol constants specific to the
Arduino Nano (ATmega328P).

The Arduino Nano shares the same ATmega328P processor as the Uno R3 in a
compact breadboard-friendly form factor (30-pin).  Pin mappings are adapted
for the Nano's A0-A7 analog range and Mini-USB connector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Board hardware specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardConfig:
    """Immutable hardware specification for the Arduino Nano."""

    board_name: str = "Arduino Nano"
    cpu: str = "ATmega328P"
    clock_hz: int = 16_000_000
    flash_kb: int = 32
    ram_kb: int = 2
    eeprom_kb: int = 1
    uart_count: int = 1
    spi_count: int = 1
    i2c_count: int = 1
    pwm_pins: int = 6            # D3, D5, D6, D9, D10, D11
    adc_count: int = 8           # A0-A7
    adc_resolution_bits: int = 10
    gpio_count: int = 22         # D0-D13 + A0-A7
    operating_voltage_v: float = 5.0


# ---------------------------------------------------------------------------
# Serial communication parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SerialConfig:
    """UART settings for the NEXUS link on Arduino Nano."""

    baud_rate: int = 115_200
    data_bits: int = 8
    parity: str = "N"
    stop_bits: int = 1
    timeout_s: float = 1.0


# ---------------------------------------------------------------------------
# NEXUS wire-protocol constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WireProtocolConfig:
    """Low-level framing parameters for the NEXUS serial protocol."""

    frame_preamble: bytes = b"\xAA\x55"
    max_frame_size: int = 256
    heartbeat_interval_ms: int = 500
    crc_polynomial: int = 0x8005
    message_id_offset: int = 2
    payload_offset: int = 3


# ---------------------------------------------------------------------------
# Pin mapping - marine-sensor defaults for Nano
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PinMapping:
    """GPIO pin assignments for marine-sensor integration on the Arduino Nano.

    Pin numbers follow the Arduino digital/analog numbering convention
    (D0-D13, A0-A7).  Analog pins are referenced by their analog index
    (0-7) and map to digital pin numbers (14-21) when used as digital I/O.
    """

    # GPS module (UART shared with USB - consider SoftwareSerial)
    GPS_TX: int = 0               # D0 / RX
    GPS_RX: int = 1               # D1 / TX

    # IMU (I2C)
    IMU_SDA: int = 18             # A4 on Nano = D18
    IMU_SCL: int = 19             # A5 on Nano = D19

    # Sonar (HC-SR04 or similar)
    SONAR_TRIG: int = 7           # D7
    SONAR_ECHO: int = 8           # D8

    # Temperature sensor (analog NTC or DS18B20 1-Wire)
    TEMP_PIN: int = 0             # A0

    # Pressure / depth sensor (analog or I2C)
    PRESSURE_PIN: int = 1         # A1

    # Extra analog sensors (Nano has A0-A7 vs Uno's A0-A5)
    EXTRA_ADC_1: int = 2          # A2
    EXTRA_ADC_2: int = 3          # A3

    # Servo output (PWM-capable)
    SERVO_PINS: Tuple[int, ...] = (9,)

    # Onboard LED
    LED_PIN: int = 13             # D13

    # Thruster / ESC output (PWM-capable)
    THRUSTER_PWM: int = 10        # D10 (PWM)


# ---------------------------------------------------------------------------
# Interface pin definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterfacePins:
    """Hardware interface pin assignments for the Arduino Nano."""

    # UART
    UART_TX: int = 1              # D1
    UART_RX: int = 0              # D0

    # SPI
    SPI_SS: int = 10              # D10
    SPI_MOSI: int = 11            # D11
    SPI_MISO: int = 12            # D12
    SPI_SCK: int = 13             # D13

    # I2C
    I2C_SDA: int = 18             # A4
    I2C_SCL: int = 19             # A5

    # PWM-capable pins
    PWM_PINS: Tuple[int, ...] = (3, 5, 6, 9, 10, 11)


# ---------------------------------------------------------------------------
# Composite Nano configuration
# ---------------------------------------------------------------------------

@dataclass
class NanoConfig:
    """Top-level configuration container for the Arduino Nano."""

    board_config: BoardConfig = field(default_factory=BoardConfig)
    serial_config: SerialConfig = field(default_factory=SerialConfig)
    wire_protocol: WireProtocolConfig = field(default_factory=WireProtocolConfig)
    pin_mapping: PinMapping = field(default_factory=PinMapping)
    interface_pins: InterfacePins = field(default_factory=InterfacePins)


def get_nano_config(**overrides) -> NanoConfig:
    """Return a ``NanoConfig`` instance with optional keyword overrides.

    Accepted override keys correspond to the nested dataclass attributes.
    Unrecognised keys raise ``ValueError``.
    """
    cfg = NanoConfig()

    # Serial overrides
    serial_fields = {f.name for f in SerialConfig.__dataclass_fields__.values()}
    serial_overrides = {k: v for k, v in overrides.items() if k in serial_fields}
    if serial_overrides:
        current = {
            f.name: getattr(cfg.serial_config, f.name)
            for f in SerialConfig.__dataclass_fields__.values()
        }
        current.update(serial_overrides)
        cfg.serial_config = SerialConfig(**current)

    # Pin overrides
    pin_fields = {f.name for f in PinMapping.__dataclass_fields__.values()}
    pin_overrides = {k: v for k, v in overrides.items() if k in pin_fields}
    if pin_overrides:
        current = {
            f.name: getattr(cfg.pin_mapping, f.name)
            for f in PinMapping.__dataclass_fields__.values()
        }
        current.update(pin_overrides)
        cfg.pin_mapping = PinMapping(**current)

    # Board overrides
    board_fields = {f.name for f in BoardConfig.__dataclass_fields__.values()}
    board_overrides = {k: v for k, v in overrides.items() if k in board_fields}
    if board_overrides:
        current = {
            f.name: getattr(cfg.board_config, f.name)
            for f in BoardConfig.__dataclass_fields__.values()
        }
        current.update(board_overrides)
        cfg.board_config = BoardConfig(**current)

    # Wire-protocol overrides
    wp_fields = {f.name for f in WireProtocolConfig.__dataclass_fields__.values()}
    wp_overrides = {k: v for k, v in overrides.items() if k in wp_fields}
    if wp_overrides:
        current = {
            f.name: getattr(cfg.wire_protocol, f.name)
            for f in WireProtocolConfig.__dataclass_fields__.values()
        }
        current.update(wp_overrides)
        cfg.wire_protocol = WireProtocolConfig(**current)

    # Detect unknown keys
    known = pin_fields | board_fields | wp_fields | serial_fields
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    return cfg
