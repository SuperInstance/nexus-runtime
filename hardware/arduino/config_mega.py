"""
NEXUS Arduino Mega 2560 Hardware Configuration Module.

Defines board-level metadata, multi-UART serial parameters, expanded pin
mappings for multi-sensor marine setups, and NEXUS wire protocol constants
specific to the Arduino Mega 2560 (ATmega2560).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Board hardware specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardConfig:
    """Immutable hardware specification for the Arduino Mega 2560."""

    board_name: str = "Arduino Mega 2560"
    cpu: str = "ATmega2560"
    clock_hz: int = 16_000_000
    flash_kb: int = 256
    ram_kb: int = 8
    eeprom_kb: int = 4
    uart_count: int = 4           # Serial, Serial1, Serial2, Serial3
    spi_count: int = 1
    i2c_count: int = 1
    pwm_pins: int = 15            # D2-D13, D44-D46
    adc_count: int = 16           # A0–A15
    adc_resolution_bits: int = 10
    gpio_count: int = 54          # D0–D53
    operating_voltage_v: float = 5.0


# ---------------------------------------------------------------------------
# Serial communication parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SerialConfig:
    """UART settings for the NEXUS link on Arduino Mega 2560.

    The Mega exposes four hardware UARTs.  ``serial_port`` indicates which
    UART the primary NEXUS link uses (default: ``Serial1``).
    """

    baud_rate: int = 115_200
    data_bits: int = 8
    parity: str = "N"
    stop_bits: int = 1
    timeout_s: float = 1.0
    serial_port: str = "Serial1"

    # Dedicated UART pin assignments for reference
    UART0_PINS: Tuple[int, int] = (0, 1)       # D0/D1  (USB serial)
    UART1_PINS: Tuple[int, int] = (19, 18)      # RX1/TX1
    UART2_PINS: Tuple[int, int] = (17, 16)      # RX2/TX2
    UART3_PINS: Tuple[int, int] = (15, 14)      # RX3/TX3


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
# Pin mapping — multi-sensor marine layout for Mega 2560
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PinMapping:
    """GPIO pin assignments for a multi-sensor marine setup on the Mega 2560.

    The Mega's generous pin count allows simultaneous connection of GPS,
    dual IMUs, up to three sonar modules, multiple temperature / pressure
    sensors, several servos, and multiple thruster ESCs.
    """

    # ---- GPS (dedicated UART — Serial1) ----
    GPS_TX: int = 19             # RX1
    GPS_RX: int = 18             # TX1

    # ---- IMU (I2C) ----
    IMU_SDA: int = 20            # SDA / D20
    IMU_SCL: int = 21            # SCL / D21
    IMU2_SDA: int = 20           # Secondary IMU shares bus
    IMU2_SCL: int = 21

    # ---- Sonar array (up to 3 units) ----
    SONAR1_TRIG: int = 22
    SONAR1_ECHO: int = 23
    SONAR2_TRIG: int = 24
    SONAR2_ECHO: int = 25
    SONAR3_TRIG: int = 26
    SONAR3_ECHO: int = 27

    # ---- Temperature sensors ----
    TEMP_PIN_1: int = 0          # A0 — forward hull
    TEMP_PIN_2: int = 1          # A1 — motor bay
    TEMP_PIN_3: int = 2          # A2 — ballast area

    # ---- Pressure / depth sensor ----
    PRESSURE_PIN: int = 3        # A3

    # ---- Servo outputs (PWM-capable pins) ----
    SERVO_PINS: Tuple[int, ...] = (44, 45, 46)   # D44, D45, D46

    # ---- Thruster ESC outputs (PWM-capable) ----
    THRUSTER_PORT: int = 44      # D44 (reuse of servo range)
    THRUSTER_STARBOARD: int = 45 # D45
    THRUSTER_VERTICAL: int = 46  # D46

    # ---- Companion link (Serial2) ----
    COMPANION_TX: int = 16       # TX2
    COMPANION_RX: int = 17       # RX2

    # ---- Auxiliary serial (Serial3) ----
    AUX_TX: int = 14             # TX3
    AUX_RX: int = 15             # RX3

    # ---- Onboard LED ----
    LED_PIN: int = 13

    # ---- Relay / power control ----
    RELAY_PIN: int = 30


# ---------------------------------------------------------------------------
# Composite Mega configuration
# ---------------------------------------------------------------------------

@dataclass
class MegaConfig:
    """Top-level configuration container for the Arduino Mega 2560."""

    board_config: BoardConfig = field(default_factory=BoardConfig)
    serial_config: SerialConfig = field(default_factory=SerialConfig)
    wire_protocol: WireProtocolConfig = field(default_factory=WireProtocolConfig)
    pin_mapping: PinMapping = field(default_factory=PinMapping)


def get_mega_config(**overrides) -> MegaConfig:
    """Return a ``MegaConfig`` instance with optional keyword overrides.

    Accepted override keys: any attribute name from ``BoardConfig``,
    ``SerialConfig``, ``WireProtocolConfig``, or ``PinMapping``.  Unrecognised
    keys raise ``ValueError``.
    """
    cfg = MegaConfig()

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
