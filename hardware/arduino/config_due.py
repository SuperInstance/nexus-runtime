"""
NEXUS Arduino Due Hardware Configuration Module.

Defines board-level metadata, multi-UART serial parameters, extended pin
mappings for high-performance marine sensor arrays, and NEXUS wire protocol
constants specific to the Arduino Due (AT91SAM3X8E ARM Cortex-M3).

The Due offers significantly more processing power than AVR-based boards with
its 84 MHz ARM Cortex-M3, 512 KB flash, and 96 KB SRAM, making it ideal for
NEXUS edge computing tasks that require local sensor fusion, PID control, and
real-time data processing before forwarding telemetry to the central hub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Board hardware specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardConfig:
    """Immutable hardware specification for the Arduino Due."""

    board_name: str = "Arduino Due"
    cpu: str = "AT91SAM3X8E"
    cpu_arch: str = "ARM Cortex-M3"
    clock_hz: int = 84_000_000
    flash_kb: int = 512
    ram_kb: int = 96
    eeprom_kb: int = 0            # No native EEPROM
    uart_count: int = 4           # Serial (programming), Serial1-3
    spi_count: int = 2            # SPI (extended), SPI1
    i2c_count: int = 2            # Wire, Wire1
    pwm_pins: int = 12            # D2-D13 (all PWM-capable on Due)
    adc_count: int = 12           # A0-A11
    adc_resolution_bits: int = 12
    dac_count: int = 2            # DAC0, DAC1 (unique to Due)
    gpio_count: int = 54          # D0-D53
    operating_voltage_v: float = 3.3


# ---------------------------------------------------------------------------
# Serial communication parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SerialConfig:
    """UART settings for the NEXUS link on Arduino Due.

    The Due exposes four hardware UARTs.  ``serial_port`` indicates which
    UART the primary NEXUS link uses (default: ``Serial1``).
    """

    baud_rate: int = 115_200
    data_bits: int = 8
    parity: str = "N"
    stop_bits: int = 1
    timeout_s: float = 1.0
    serial_port: str = "Serial1"

    # Dedicated UART pin assignments
    UART0_PINS: Tuple[int, int] = (0, 1)       # D0/D1  (USB/programming)
    UART1_PINS: Tuple[int, int] = (19, 18)      # RX1/TX1
    UART2_PINS: Tuple[int, int] = (17, 16)      # RX2/TX2
    UART3_PINS: Tuple[int, int] = (15, 14)      # RX3/TX3


# ---------------------------------------------------------------------------
# NEXUS wire-protocol constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WireProtocolConfig:
    """Low-level framing parameters for the NEXUS serial protocol.

    The Due's increased memory allows larger frame sizes for complex
    telemetry payloads.
    """

    frame_preamble: bytes = b"\xAA\x55"
    max_frame_size: int = 512
    heartbeat_interval_ms: int = 250
    crc_polynomial: int = 0x8005
    message_id_offset: int = 2
    payload_offset: int = 3


# ---------------------------------------------------------------------------
# High-performance NEXUS edge configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NexusEdgeConfig:
    """NEXUS edge computing configuration for the Arduino Due.

    The Due's ARM Cortex-M3 at 84 MHz enables local sensor fusion,
    PID control loops, and real-time data preprocessing, reducing the
    load on the central NEXUS hub.
    """

    edge_processing_enabled: bool = True
    sensor_fusion_rate_hz: float = 50.0
    pid_loop_rate_hz: float = 100.0
    telemetry_aggregation_window_ms: int = 100
    max_sensor_streams: int = 8
    watchdog_timeout_ms: int = 1000
    buffer_size_frames: int = 64


# ---------------------------------------------------------------------------
# Pin mapping - high-performance marine layout for Due
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PinMapping:
    """GPIO pin assignments for a high-performance multi-sensor marine setup.

    The Due's generous pin count (54 digital + 12 analog + 2 DAC) supports
    simultaneous connection of GPS, dual IMUs, sonar arrays, temperature
    sensors, pressure transducers, multiple servos, and thruster ESCs,
    with dedicated UARTs for each subsystem.
    """

    # ---- GPS (dedicated UART - Serial1) ----
    GPS_TX: int = 19             # RX1
    GPS_RX: int = 18             # TX1

    # ---- IMU (I2C - Wire) ----
    IMU_SDA: int = 20            # SDA / D20
    IMU_SCL: int = 21            # SCL / D21
    IMU2_SDA: int = 70           # SDA1 / Wire1 (extended header)
    IMU2_SCL: int = 71           # SCL1 / Wire1 (extended header)

    # ---- Sonar array (up to 4 units) ----
    SONAR1_TRIG: int = 22
    SONAR1_ECHO: int = 23
    SONAR2_TRIG: int = 24
    SONAR2_ECHO: int = 25
    SONAR3_TRIG: int = 26
    SONAR3_ECHO: int = 27
    SONAR4_TRIG: int = 28
    SONAR4_ECHO: int = 29

    # ---- Temperature sensors ----
    TEMP_PIN_1: int = 0          # A0 - forward hull
    TEMP_PIN_2: int = 1          # A1 - motor bay
    TEMP_PIN_3: int = 2          # A2 - ballast area
    TEMP_PIN_4: int = 3          # A3 - battery compartment

    # ---- Pressure / depth sensor ----
    PRESSURE_PIN: int = 4        # A4

    # ---- Analog sensors ----
    WATER_QUALITY_PIN: int = 5   # A5 - turbidity / dissolved O2
    EXTRA_ADC_1: int = 6         # A6
    EXTRA_ADC_2: int = 7         # A7

    # ---- DAC outputs (unique to Due) ----
    DAC0_PIN: int = 66           # DAC0 (analog output)
    DAC1_PIN: int = 67           # DAC1 (analog output)

    # ---- Servo outputs (all Due digital pins are PWM-capable) ----
    SERVO_PINS: Tuple[int, ...] = (34, 35, 36)

    # ---- Thruster ESC outputs ----
    THRUSTER_PORT: int = 34      # D34
    THRUSTER_STARBOARD: int = 35 # D35
    THRUSTER_VERTICAL: int = 36  # D36
    THRUSTER_LATERAL: int = 37   # D37

    # ---- Companion link (Serial2) ----
    COMPANION_TX: int = 16       # TX2
    COMPANION_RX: int = 17       # RX2

    # ---- Auxiliary serial (Serial3) ----
    AUX_TX: int = 14             # TX3
    AUX_RX: int = 15             # RX3

    # ---- Onboard LED ----
    LED_PIN: int = 13

    # ---- Relay / power control ----
    RELAY_PIN_1: int = 30
    RELAY_PIN_2: int = 31


# ---------------------------------------------------------------------------
# Interface pin definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterfacePins:
    """Hardware interface pin assignments for the Arduino Due."""

    # UART
    UART0_TX: int = 1            # D1
    UART0_RX: int = 0            # D0
    UART1_TX: int = 18           # TX1
    UART1_RX: int = 19           # RX1
    UART2_TX: int = 16           # TX2
    UART2_RX: int = 17           # RX2
    UART3_TX: int = 14           # TX3
    UART3_RX: int = 15           # RX3

    # SPI (main)
    SPI_SS: int = 10             # D10
    SPI_MOSI: int = 11           # D11
    SPI_MISO: int = 12           # D12
    SPI_SCK: int = 13            # D13

    # SPI (extended)
    SPI1_SS: int = 52            # D52
    SPI1_MOSI: int = 50          # D50
    SPI1_MISO: int = 51          # D51
    SPI1_SCK: int = 53           # D53

    # I2C (Wire)
    I2C_SDA: int = 20            # D20
    I2C_SCL: int = 21            # D21

    # I2C (Wire1)
    I2C1_SDA: int = 70           # SDA1
    I2C1_SCL: int = 71           # SCL1

    # PWM-capable pins (D2-D13 on Due)
    PWM_PINS: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)


# ---------------------------------------------------------------------------
# Composite Due configuration
# ---------------------------------------------------------------------------

@dataclass
class DueConfig:
    """Top-level configuration container for the Arduino Due."""

    board_config: BoardConfig = field(default_factory=BoardConfig)
    serial_config: SerialConfig = field(default_factory=SerialConfig)
    wire_protocol: WireProtocolConfig = field(default_factory=WireProtocolConfig)
    nexus_edge: NexusEdgeConfig = field(default_factory=NexusEdgeConfig)
    pin_mapping: PinMapping = field(default_factory=PinMapping)
    interface_pins: InterfacePins = field(default_factory=InterfacePins)


def get_due_config(**overrides) -> DueConfig:
    """Return a ``DueConfig`` instance with optional keyword overrides.

    Accepted override keys: any attribute name from ``BoardConfig``,
    ``SerialConfig``, ``WireProtocolConfig``, ``NexusEdgeConfig``, or
    ``PinMapping``.  Unrecognised keys raise ``ValueError``.
    """
    cfg = DueConfig()

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

    # Edge config overrides
    edge_fields = {f.name for f in NexusEdgeConfig.__dataclass_fields__.values()}
    edge_overrides = {k: v for k, v in overrides.items() if k in edge_fields}
    if edge_overrides:
        current = {
            f.name: getattr(cfg.nexus_edge, f.name)
            for f in NexusEdgeConfig.__dataclass_fields__.values()
        }
        current.update(edge_overrides)
        cfg.nexus_edge = NexusEdgeConfig(**current)

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

    # Detect unknown keys
    known = (pin_fields | board_fields | wp_fields | serial_fields | edge_fields)
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    return cfg
