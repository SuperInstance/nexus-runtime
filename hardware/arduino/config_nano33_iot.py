"""
NEXUS Arduino Nano 33 IoT Hardware Configuration Module.

Defines board-level metadata, serial communication parameters, WiFi/BLE
mesh configuration for the NEXUS distributed intelligence platform, and
NEXUS wire protocol constants specific to the Arduino Nano 33 IoT
(SAMD21 + u-blox NINA-W102).

The Nano 33 IoT combines the SAMD21G18A (ARM Cortex-M0+) at 48 MHz with
an onboard u-blox NINA-W102 module providing 802.11 b/g/n WiFi and
Bluetooth LE 4.2. Its compact form factor and wireless capabilities make
it ideal for NEXUS mesh sensor nodes in distributed marine robotics fleets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Board hardware specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardConfig:
    """Immutable hardware specification for the Arduino Nano 33 IoT."""

    board_name: str = "Arduino Nano 33 IoT"
    cpu: str = "SAMD21G18A"
    cpu_arch: str = "ARM Cortex-M0+"
    clock_hz: int = 48_000_000
    flash_kb: int = 256
    ram_kb: int = 32
    eeprom_kb: int = 0            # Emulated in flash
    uart_count: int = 1           # Serial1 (USB is native CDC)
    spi_count: int = 1
    i2c_count: int = 1
    pwm_pins: int = 4             # D2, D3, D5, D9 (limited PWM on SAMD21)
    adc_count: int = 8            # A0-A7
    adc_resolution_bits: int = 12
    gpio_count: int = 20          # D0-D13 + A0-A5 (some shared)
    operating_voltage_v: float = 3.3


# ---------------------------------------------------------------------------
# Serial communication parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SerialConfig:
    """UART settings for the NEXUS link on Nano 33 IoT.

    The Nano 33 IoT uses native USB CDC for programming/debug and
    Serial1 (D1/D0) for hardware UART communications.
    """

    baud_rate: int = 115_200
    data_bits: int = 8
    parity: str = "N"
    stop_bits: int = 1
    timeout_s: float = 1.0
    serial_port: str = "Serial1"

    # Hardware UART pin assignments
    UART_TX: int = 1              # D1
    UART_RX: int = 0              # D0


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
# WiFi / BLE mesh configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WiFiConfig:
    """WiFi and BLE settings for the onboard NINA-W102 module.

    Configured for NEXUS mesh networking with automatic reconnection
    and low-power operation suitable for battery-powered marine nodes.
    """

    wifi_enabled: bool = True
    ble_enabled: bool = True
    wifi_ssid: str = ""
    wifi_password: str = ""
    mqtt_broker: str = ""
    mqtt_port: int = 1883
    ap_mode: bool = False
    ap_ssid: str = "NEXUS-NANO33"
    wifi_channel: int = 6
    low_power_mode: bool = True
    wifi_timeout_ms: int = 15000
    max_reconnect_attempts: int = 10


# ---------------------------------------------------------------------------
# NEXUS mesh configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NexusMeshConfig:
    """NEXUS mesh networking configuration for the Nano 33 IoT.

    The Nano 33 IoT operates as a mesh node in the NEXUS distributed
    intelligence network, relaying sensor data and receiving commands
    over WiFi/BLE.
    """

    mesh_enabled: bool = True
    node_id: str = ""
    node_role: str = "end_device"    # coordinator, router, end_device
    parent_node_id: str = ""
    telemetry_topic: str = "nexus/telemetry"
    command_topic: str = "nexus/command"
    heartbeat_topic: str = "nexus/heartbeat"
    telemetry_interval_ms: int = 1000
    status_report_interval_s: int = 60


# ---------------------------------------------------------------------------
# Pin mapping - marine-sensor defaults for Nano 33 IoT
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PinMapping:
    """GPIO pin assignments for marine-sensor integration on Nano 33 IoT.

    Pin numbers follow the Arduino Nano 33 IoT convention (D0-D13, A0-A7).
    """

    # GPS module (UART - Serial1)
    GPS_TX: int = 0               # D0 / RX
    GPS_RX: int = 1               # D1 / TX

    # IMU (I2C - Wire)
    IMU_SDA: int = 18             # A4 on Nano 33 IoT
    IMU_SCL: int = 19             # A5 on Nano 33 IoT

    # Sonar (HC-SR04 or similar)
    SONAR_TRIG: int = 7           # D7
    SONAR_ECHO: int = 8           # D8

    # Temperature sensor (analog or 1-Wire)
    TEMP_PIN: int = 0             # A0

    # Pressure / depth sensor (I2C)
    PRESSURE_PIN: int = 1         # A1

    # Extra analog sensors
    EXTRA_ADC_1: int = 2          # A2
    EXTRA_ADC_2: int = 3          # A3

    # Servo output (PWM-capable)
    SERVO_PINS: Tuple[int, ...] = (9,)

    # Onboard LED
    LED_PIN: int = 13             # D13
    LED_BUILTIN: int = 13         # Alias
    LED_R: int = 26               # RGB red (not exposed on header)
    LED_G: int = 25               # RGB green
    LED_B: int = 27               # RGB blue

    # Thruster / ESC output (PWM-capable)
    THRUSTER_PWM: int = 3         # D3 (PWM)

    # Built-in IMU (LSM6DS3 - on I2C)
    BUILTIN_IMU_SDA: int = 18     # A4
    BUILTIN_IMU_SCL: int = 19     # A5


# ---------------------------------------------------------------------------
# Interface pin definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterfacePins:
    """Hardware interface pin assignments for the Nano 33 IoT."""

    # UART (Serial1)
    UART_TX: int = 1              # D1
    UART_RX: int = 0              # D0

    # SPI
    SPI_SS: int = 10              # D10
    SPI_MOSI: int = 11            # D11
    SPI_MISO: int = 12            # D12
    SPI_SCK: int = 13             # D13

    # I2C (Wire)
    I2C_SDA: int = 18             # A4
    I2C_SCL: int = 19             # A5

    # PWM-capable pins
    PWM_PINS: Tuple[int, ...] = (2, 3, 5, 9)


# ---------------------------------------------------------------------------
# Composite Nano 33 IoT configuration
# ---------------------------------------------------------------------------

@dataclass
class Nano33IoTConfig:
    """Top-level configuration container for the Arduino Nano 33 IoT."""

    board_config: BoardConfig = field(default_factory=BoardConfig)
    serial_config: SerialConfig = field(default_factory=SerialConfig)
    wire_protocol: WireProtocolConfig = field(default_factory=WireProtocolConfig)
    wifi_config: WiFiConfig = field(default_factory=WiFiConfig)
    nexus_mesh: NexusMeshConfig = field(default_factory=NexusMeshConfig)
    pin_mapping: PinMapping = field(default_factory=PinMapping)
    interface_pins: InterfacePins = field(default_factory=InterfacePins)


def get_nano33_iot_config(**overrides) -> Nano33IoTConfig:
    """Return a ``Nano33IoTConfig`` instance with optional keyword overrides.

    Accepted override keys correspond to the nested dataclass attributes.
    Unrecognised keys raise ``ValueError``.
    """
    cfg = Nano33IoTConfig()

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

    # Mesh overrides
    mesh_fields = {f.name for f in NexusMeshConfig.__dataclass_fields__.values()}
    mesh_overrides = {k: v for k, v in overrides.items() if k in mesh_fields}
    if mesh_overrides:
        current = {
            f.name: getattr(cfg.nexus_mesh, f.name)
            for f in NexusMeshConfig.__dataclass_fields__.values()
        }
        current.update(mesh_overrides)
        cfg.nexus_mesh = NexusMeshConfig(**current)

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
             | wifi_fields | mesh_fields)
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    return cfg
