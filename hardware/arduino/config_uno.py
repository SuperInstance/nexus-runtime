"""
NEXUS Arduino Uno R3 Hardware Configuration Module.

Defines board-level metadata, serial communication parameters, pin mappings
for common marine sensors, and NEXUS wire protocol constants specific to the
Arduino Uno R3 (ATmega328P).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Board hardware specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardConfig:
    """Immutable hardware specification for the Arduino Uno R3."""

    board_name: str = "Arduino Uno R3"
    cpu: str = "ATmega328P"
    clock_hz: int = 16_000_000
    flash_kb: int = 32
    ram_kb: int = 2
    eeprom_kb: int = 1
    uart_count: int = 1
    spi_count: int = 1
    i2c_count: int = 1
    pwm_pins: int = 6          # D3, D5, D6, D9, D10, D11
    adc_count: int = 6         # A0–A5
    adc_resolution_bits: int = 10
    gpio_count: int = 14       # D0–D13
    operating_voltage_v: float = 5.0


# ---------------------------------------------------------------------------
# Serial communication parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SerialConfig:
    """UART settings for the NEXUS link on Arduino Uno."""

    baud_rate: int = 115_200
    data_bits: int = 8
    parity: str = "N"           # None
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
    crc_polynomial: int = 0x8005   # CRC-16/ARC
    message_id_offset: int = 2     # byte index of MSG_ID inside frame
    payload_offset: int = 3        # byte index of first payload byte


# ---------------------------------------------------------------------------
# Pin mapping — marine-sensor defaults for Uno
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PinMapping:
    """Default GPIO pin assignments for marine-sensor integration on Uno.

    Pin numbers follow the Arduino digital/analog numbering convention
    (D0-D13, A0-A5).  Analog pins are referenced by their analog index
    (0–5) and map to the corresponding digital pin number (14–19) when
    used as digital I/O.
    """

    # GPS module (UART shared with USB — consider SoftwareSerial)
    GPS_TX: int = 0               # D0 / RX
    GPS_RX: int = 1               # D1 / TX

    # IMU (I2C)
    IMU_SDA: int = 18             # A4 on Uno = D18
    IMU_SCL: int = 19             # A5 on Uno = D19

    # Sonar (HC-SR04 or similar)
    SONAR_TRIG: int = 7           # D7
    SONAR_ECHO: int = 8           # D8

    # Temperature sensor (analog NTC or DS18B20 1-Wire)
    TEMP_PIN: int = 0             # A0

    # Pressure / depth sensor (analog or I2C)
    PRESSURE_PIN: int = 1         # A1

    # Servo output (PWM-capable)
    SERVO_PINS: tuple = (9,)      # D9 (PWM)

    # Onboard LED
    LED_PIN: int = 13             # D13

    # Thruster / ESC output (PWM-capable)
    THRUSTER_PWM: int = 10        # D10 (PWM)


# ---------------------------------------------------------------------------
# Composite Uno configuration
# ---------------------------------------------------------------------------

@dataclass
class UnoConfig:
    """Top-level configuration container for the Arduino Uno R3."""

    board_config: BoardConfig = field(default_factory=BoardConfig)
    serial_config: SerialConfig = field(default_factory=SerialConfig)
    wire_protocol: WireProtocolConfig = field(default_factory=WireProtocolConfig)
    pin_mapping: PinMapping = field(default_factory=PinMapping)


def get_uno_config(**overrides) -> UnoConfig:
    """Return an ``UnoConfig`` instance with optional keyword overrides.

    Accepted override keys correspond to the nested dataclass attributes,
    e.g. ``baud_rate=9600`` or ``GPS_TX=4``.  Unrecognised keys raise
    ``ValueError``.
    """
    cfg = UnoConfig()

    # Serial overrides
    if "baud_rate" in overrides:
        cfg.serial_config = SerialConfig(
            baud_rate=overrides["baud_rate"],
            data_bits=cfg.serial_config.data_bits,
            parity=cfg.serial_config.parity,
            stop_bits=cfg.serial_config.stop_bits,
        )

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
    known = pin_fields | board_fields | wp_fields | {"baud_rate"}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    return cfg
