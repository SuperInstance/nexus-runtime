"""
NEXUS Arduino MKR WiFi 1010 Hardware Configuration Module.

Defines board-level metadata, serial communication parameters, WiFi/BLE
pin mappings for the NEXUS serial bridge, and NEXUS wire protocol constants
specific to the Arduino MKR WiFi 1010 (SAMD21 + u-blox NINA-W102).

The MKR WiFi 1010 uses the SAMD21G18A (ARM Cortex-M0+) at 48 MHz with an
onboard u-blox NINA-W102 module providing 802.11 b/g/n WiFi and Bluetooth
LE 4.2 connectivity, ideal for NEXUS wireless sensor gateway applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Board hardware specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardConfig:
    """Immutable hardware specification for the Arduino MKR WiFi 1010."""

    board_name: str = "Arduino MKR WiFi 1010"
    cpu: str = "SAMD21G18A"
    cpu_arch: str = "ARM Cortex-M0+"
    clock_hz: int = 48_000_000
    flash_kb: int = 256
    ram_kb: int = 32
    eeprom_kb: int = 0            # Emulated in flash
    uart_count: int = 1           # Serial1 (USB is native CDC)
    spi_count: int = 1
    i2c_count: int = 1
    pwm_pins: int = 12            # D0-D8, D10, A3-A4
    adc_count: int = 7            # A0-A6
    adc_resolution_bits: int = 12
    gpio_count: int = 22          # D0-D14 + A0-A6
    operating_voltage_v: float = 3.3


# ---------------------------------------------------------------------------
# Serial communication parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SerialConfig:
    """UART settings for the NEXUS link on MKR WiFi 1010.

    The MKR uses a native USB CDC serial for programming/debug and a
    separate hardware UART (Serial1) on pins 13/14 for the NEXUS link.
    """

    baud_rate: int = 115_200
    data_bits: int = 8
    parity: str = "N"
    stop_bits: int = 1
    timeout_s: float = 1.0
    serial_port: str = "Serial1"

    # Hardware UART pin assignments
    UART_TX: int = 14             # D14
    UART_RX: int = 13             # D13


# ---------------------------------------------------------------------------
# NEXUS wire-protocol constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WireProtocolConfig:
    """Low-level framing parameters for the NEXUS serial protocol."""

    frame_preamble: bytes = b"\xAA\x55"
    max_frame_size: int = 512
    heartbeat_interval_ms: int = 500
    crc_polynomial: int = 0x8005
    message_id_offset: int = 2
    payload_offset: int = 3


# ---------------------------------------------------------------------------
# WiFi / BLE configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WiFiConfig:
    """WiFi and BLE settings for the onboard NINA-W102 module."""

    wifi_enabled: bool = True
    ble_enabled: bool = True
    wifi_ssid: str = ""
    wifi_password: str = ""
    mqtt_broker: str = ""
    mqtt_port: int = 1883
    ap_mode: bool = False
    ap_ssid: str = "NEXUS-MKR"
    wifi_channel: int = 6


# ---------------------------------------------------------------------------
# NEXUS serial bridge configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NexusBridgeConfig:
    """NEXUS serial bridge settings for the MKR WiFi 1010.

    The MKR acts as a WiFi-to-serial bridge, forwarding NEXUS telemetry
    frames between the onboard UART and the NINA-W102 WiFi module.
    """

    bridge_enabled: bool = True
    bridge_protocol: str = "MQTT"    # MQTT or WebSocket
    telemetry_topic: str = "nexus/telemetry"
    command_topic: str = "nexus/command"
    status_topic: str = "nexus/status"
    max_reconnect_attempts: int = 5
    reconnect_interval_ms: int = 5000


# ---------------------------------------------------------------------------
# Pin mapping - marine-sensor defaults for MKR WiFi 1010
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PinMapping:
    """GPIO pin assignments for marine-sensor integration on the MKR WiFi 1010.

    The MKR form factor uses D0-D14 and A0-A6.  Pin assignments follow
    the MKR pinout convention.
    """

    # GPS module (Serial1 - dedicated UART)
    GPS_TX: int = 13             # RX1 / D13
    GPS_RX: int = 14             # TX1 / D14

    # IMU (I2C - Wire)
    IMU_SDA: int = 11            # D11 / SDA
    IMU_SCL: int = 12            # D12 / SCL

    # Sonar (HC-SR04 or similar)
    SONAR_TRIG: int = 6          # D6
    SONAR_ECHO: int = 7          # D7

    # Temperature sensor (analog or 1-Wire)
    TEMP_PIN: int = 0            # A0

    # Pressure / depth sensor (I2C)
    PRESSURE_PIN: int = 1        # A1

    # Servo output (PWM-capable)
    SERVO_PINS: Tuple[int, ...] = (4, 5)

    # Onboard LED
    LED_PIN: int = 6             # D6 (RGB LED, green channel)

    # Thruster / ESC output (PWM-capable)
    THRUSTER_PWM: int = 4        # D4 (PWM)

    # SPI pins (for WiFi module is internal; external SPI available)
    SPI_SS: int = 5              # D5
    SPI_MOSI: int = 8            # D8
    SPI_MISO: int = 10           # D10
    SPI_SCK: int = 9             # D9


# ---------------------------------------------------------------------------
# Interface pin definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterfacePins:
    """Hardware interface pin assignments for the MKR WiFi 1010."""

    # UART (Serial1)
    UART_TX: int = 14            # D14
    UART_RX: int = 13            # D13

    # SPI
    SPI_SS: int = 5              # D5
    SPI_MOSI: int = 8            # D8
    SPI_MISO: int = 10           # D10
    SPI_SCK: int = 9             # D9

    # I2C (Wire)
    I2C_SDA: int = 11            # D11
    I2C_SCL: int = 12            # D12

    # PWM-capable pins
    PWM_PINS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 18, 19)


# ---------------------------------------------------------------------------
# Composite MKR WiFi 1010 configuration
# ---------------------------------------------------------------------------

@dataclass
class MKRWiFiConfig:
    """Top-level configuration container for the Arduino MKR WiFi 1010."""

    board_config: BoardConfig = field(default_factory=BoardConfig)
    serial_config: SerialConfig = field(default_factory=SerialConfig)
    wire_protocol: WireProtocolConfig = field(default_factory=WireProtocolConfig)
    wifi_config: WiFiConfig = field(default_factory=WiFiConfig)
    nexus_bridge: NexusBridgeConfig = field(default_factory=NexusBridgeConfig)
    pin_mapping: PinMapping = field(default_factory=PinMapping)
    interface_pins: InterfacePins = field(default_factory=InterfacePins)


def get_mkr_wifi_config(**overrides) -> MKRWiFiConfig:
    """Return a ``MKRWiFiConfig`` instance with optional keyword overrides.

    Accepted override keys correspond to the nested dataclass attributes.
    Unrecognised keys raise ``ValueError``.
    """
    cfg = MKRWiFiConfig()

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

    # WiFi overrides
    wifi_fields = {f.name for f in WiFiConfig.__dataclass_fields__.values()}
    wifi_overrides = {k: v for k, v in overrides.items() if k in wifi_fields}
    if wifi_overrides:
        current = {
            f.name: getattr(cfg.wifi_config, f.name)
            for f in WiFiConfig.__dataclass_fields__.values()
        }
        current.update(wifi_overrides)
        cfg.wifi_config = WiFiConfig(**current)

    # Bridge overrides
    bridge_fields = {f.name for f in NexusBridgeConfig.__dataclass_fields__.values()}
    bridge_overrides = {k: v for k, v in overrides.items() if k in bridge_fields}
    if bridge_overrides:
        current = {
            f.name: getattr(cfg.nexus_bridge, f.name)
            for f in NexusBridgeConfig.__dataclass_fields__.values()
        }
        current.update(bridge_overrides)
        cfg.nexus_bridge = NexusBridgeConfig(**current)

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
    known = (pin_fields | board_fields | wp_fields | serial_fields
             | wifi_fields | bridge_fields)
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    return cfg
