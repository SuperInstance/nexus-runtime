"""NEXUS Marine Robotics Platform - ESP8266 Board Configuration.

Provides dataclass-based hardware configuration for the ESP8266 (ESP-12E/NodeMCU)
as used in the NEXUS marine robotics platform. Covers pin mappings, serial
parameters, wire protocol framing, WiFi communications, and board-level specs.

The ESP8266 (Xtensa L106) is a cost-effective WiFi SoC popular for IoT and
marine sensor nodes, offering 80 MHz/160 MHz operation, 4 MB flash, and
802.11 b/g/n WiFi in a compact, low-power package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Pin Map
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ESP8266PinMap:
    """GPIO pin assignments for peripherals connected to the ESP8266 (NodeMCU).

    The ESP8266-12E has 17 GPIO pins (GPIO0-GPIO16), though some are
    strapping pins with boot-time restrictions.  This mapping follows the
    NodeMCU pin numbering convention alongside the native GPIO numbers.
    """

    # GPS (SoftwareSerial - ESP8266 has limited UART)
    gps_tx: int = 1               # TX (GPIO1 / TXD0)
    gps_rx: int = 3               # RX (GPIO3 / RXD0)

    # IMU (I2C)
    imu_sda: int = 4              # D2 / GPIO4 (SDA)
    imu_scl: int = 5              # D1 / GPIO5 (SCL)

    # Sonar
    sonar_trig: int = 14          # D5 / GPIO14
    sonar_echo: int = 12          # D6 / GPIO12

    # Servo
    servo_1: int = 0              # D3 / GPIO0 (PWM)
    servo_2: int = 15             # D8 / GPIO15 (PWM)

    # Onboard LED
    led: int = 2                  # D4 / GPIO2 (active low)

    # Temperature sensor (analog - ESP8266 has only 1 ADC)
    temp_pin: int = 17            # A0 / ADC0 (10-bit, 0-1V)

    # Pressure sensor (I2C, shared bus)
    pressure_sda: int = 4         # GPIO4
    pressure_scl: int = 5         # GPIO5

    # Relay / power control
    relay: int = 16               # D0 / GPIO16

    # NEXUS companion serial (SoftwareSerial on GPIO13/GPIO15)
    nexus_tx: int = 15            # D8 / GPIO15
    nexus_rx: int = 13            # D7 / GPIO13


# ---------------------------------------------------------------------------
# Serial configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SerialConfig:
    """UART serial port settings for ESP8266.

    The ESP8266 has a single hardware UART (TXD0/RXD0 on GPIO1/GPIO3).
    A second UART (TXD1 on GPIO2) is TX-only and used for debug logging.
    """

    baud_rate: int = 115200
    data_bits: int = 8
    parity: str = "N"
    stop_bits: int = 1


# ---------------------------------------------------------------------------
# Wire-protocol framing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WireProtocolConfig:
    """Low-level wire protocol parameters for NEXUS inter-device communication."""

    frame_preamble: bytes = b"\xaa\x55"
    max_frame_size: int = 512
    heartbeat_ms: int = 1000


# ---------------------------------------------------------------------------
# Communications (WiFi)
# ---------------------------------------------------------------------------

@dataclass
class CommsConfig:
    """Network communications configuration for ESP8266.

    The ESP8266 supports 802.11 b/g/n in station and AP modes.
    """

    wifi_ssid: str = ""
    wifi_password: str = ""
    mqtt_broker: str = ""
    mqtt_port: int = 1883
    ble_enabled: bool = False     # ESP8266 has no BLE
    wifi_channel: int = 1
    ap_mode: bool = False
    ap_ssid: str = "NEXUS-ESP8266"
    ap_password: str = ""
    ap_channel: int = 6
    static_ip: str = ""
    gateway: str = ""
    subnet: str = "255.255.255.0"
    dns: str = "8.8.8.8"
    mdns_enabled: bool = True


# ---------------------------------------------------------------------------
# Top-level board configuration
# ---------------------------------------------------------------------------

@dataclass
class ESP8266BoardConfig:
    """Complete hardware configuration for an ESP8266 (ESP-12E/NodeMCU) board.

    Attributes:
        board_name: Human-readable board identifier.
        cpu: Processor architecture string.
        clock_mhz: CPU clock frequency in MHz.
        flash_mb: On-board flash size in MiB.
        sram_kb: Internal SRAM size in KiB.
        gpio_count: Total available GPIO pins.
        adc_channels: Number of ADC channels (1 on ESP8266).
        pwm_channels: Number of PWM channels.
        uart_count: Number of hardware UART peripherals.
        spi_count: Number of hardware SPI peripherals.
        i2c_count: Number of hardware I2C peripherals.
        wifi: Whether the chip has built-in WiFi.
        ble: Whether the chip has built-in Bluetooth LE.
        pin_map: GPIO peripheral pin mapping.
        serial: Default serial port configuration.
        protocol: Wire-protocol framing configuration.
        comms: Network communications settings.
    """

    board_name: str = "ESP8266 (ESP-12E/NodeMCU)"
    cpu: str = "Xtensa L106"
    clock_mhz: int = 80
    flash_mb: int = 4
    sram_kb: int = 80
    gpio_count: int = 17
    adc_channels: int = 1
    pwm_channels: int = 10
    uart_count: int = 1
    spi_count: int = 1
    i2c_count: int = 1
    wifi: bool = True
    ble: bool = False
    pin_map: ESP8266PinMap = field(default_factory=ESP8266PinMap)
    serial: SerialConfig = field(default_factory=SerialConfig)
    protocol: WireProtocolConfig = field(default_factory=WireProtocolConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_esp8266_config(**overrides: Any) -> ESP8266BoardConfig:
    """Create an :class:`ESP8266BoardConfig` with optional overrides.

    Any keyword argument whose name matches a field on
    :class:`ESP8266BoardConfig` (or a nested dataclass) will replace
    the default value.

    Examples:
        >>> cfg = create_esp8266_config(board_name="ESP8266-SENSOR", clock_mhz=160)
        >>> cfg.board_name
        'ESP8266-SENSOR'
        >>> cfg.clock_mhz
        160
    """
    config = ESP8266BoardConfig()

    nested_map = {
        "pin_map": ESP8266PinMap,
        "serial": SerialConfig,
        "protocol": WireProtocolConfig,
        "comms": CommsConfig,
    }

    for key, value in overrides.items():
        if key in nested_map and isinstance(value, dict):
            current = getattr(config, key)
            if hasattr(current, "__dataclass_fields__"):
                from dataclasses import replace
                merged = replace(current, **value)
                setattr(config, key, merged)
            else:
                setattr(config, key, value)
        elif hasattr(config, key):
            setattr(config, key, value)
        else:
            raise TypeError(
                f"create_esp8266_config() got an unexpected keyword argument '{key}'"
            )

    return config
