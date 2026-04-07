"""NEXUS Marine Robotics Platform - ESP32 Classic Board Configuration.

Provides dataclass-based hardware configuration for the ESP32 (Xtensa LX6)
as used in the NEXUS marine robotics platform. Covers pin mappings, serial
parameters, wire protocol framing, communications, and board-level specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


# ---------------------------------------------------------------------------
# Pin Map
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ESP32PinMap:
    """GPIO pin assignments for peripherals connected to the ESP32."""

    gps_tx: int = 9
    gps_rx: int = 10
    imu_sda: int = 21
    imu_scl: int = 22
    sonar_trig: int = 5
    sonar_echo: int = 18
    servo_1: int = 13
    servo_2: int = 14
    led: int = 2
    temp_pin: int = 34
    pressure_sda: int = 21
    pressure_scl: int = 22


# ---------------------------------------------------------------------------
# Serial configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SerialConfig:
    """UART serial port settings."""

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
    max_frame_size: int = 1024
    heartbeat_ms: int = 1000


# ---------------------------------------------------------------------------
# Communications (WiFi / BLE)
# ---------------------------------------------------------------------------

@dataclass
class CommsConfig:
    """Network communications configuration."""

    wifi_ssid: str = ""
    wifi_password: str = ""
    mqtt_broker: str = ""
    ble_enabled: bool = True
    wifi_channel: int = 6
    ap_mode: bool = False


# ---------------------------------------------------------------------------
# Top-level board configuration
# ---------------------------------------------------------------------------

@dataclass
class ESP32BoardConfig:
    """Complete hardware configuration for an ESP32 (classic) board.

    Attributes:
        board_name: Human-readable board identifier.
        cpu: Processor architecture string.
        clock_mhz: CPU clock frequency in MHz.
        cores: Number of CPU cores.
        flash_mb: On-board flash size in MiB.
        sram_kb: Internal SRAM size in KiB.
        gpio_count: Total available GPIO pins.
        adc_channels: Number of ADC channels.
        pwm_channels: Number of PWM (LEDC) channels.
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

    board_name: str = "ESP32"
    cpu: str = "Xtensa LX6"
    clock_mhz: int = 240
    cores: int = 2
    flash_mb: int = 4
    sram_kb: int = 520
    gpio_count: int = 34
    adc_channels: int = 18
    pwm_channels: int = 16
    uart_count: int = 3
    spi_count: int = 2
    i2c_count: int = 2
    wifi: bool = True
    ble: bool = True
    pin_map: ESP32PinMap = field(default_factory=ESP32PinMap)
    serial: SerialConfig = field(default_factory=SerialConfig)
    protocol: WireProtocolConfig = field(default_factory=WireProtocolConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_esp32_config(**overrides: Any) -> ESP32BoardConfig:
    """Create an :class:`ESP32BoardConfig` with optional overrides.

    Any keyword argument whose name matches a field on
    :class:`ESP32BoardConfig` (or a nested dataclass accessible via
    ``__dict__`` on a mutable field) will replace the default value.

    Examples:
        >>> cfg = create_esp32_config(board_name="ESP32-NAV", clock_mhz=160)
        >>> cfg.board_name
        'ESP32-NAV'
        >>> cfg.clock_mhz
        160
    """
    config = ESP32BoardConfig()

    # Build mapping of nested config attribute names for convenience
    nested_map = {
        "pin_map": ESP32PinMap,
        "serial": SerialConfig,
        "protocol": WireProtocolConfig,
        "comms": CommsConfig,
    }

    for key, value in overrides.items():
        if key in nested_map and isinstance(value, dict):
            current = getattr(config, key)
            if hasattr(current, "__dataclass_fields__"):
                merged = replace(current, **value)
                setattr(config, key, merged)
            else:
                setattr(config, key, value)
        elif hasattr(config, key):
            setattr(config, key, value)
        else:
            raise TypeError(
                f"create_esp32_config() got an unexpected keyword argument '{key}'"
            )

    return config
