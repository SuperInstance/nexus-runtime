"""NEXUS Marine Robotics Platform - ESP32-S3 Board Configuration.

Provides dataclass-based hardware configuration for the ESP32-S3
(Xtensa LX7 dual-core) as used in the NEXUS marine robotics platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Pin Map
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ESP32S3PinMap:
    """GPIO pin assignments for peripherals connected to the ESP32-S3.

    The ESP32-S3 has 45 GPIO pins. This mapping reflects the typical
    routing for NEXUS peripherals.
    """

    gps_tx: int = 43
    gps_rx: int = 44
    imu_sda: int = 8
    imu_scl: int = 9
    sonar_trig: int = 5
    sonar_echo: int = 6
    servo_1: int = 10
    servo_2: int = 11
    led: int = 21
    temp_pin: int = 4
    pressure_sda: int = 8
    pressure_scl: int = 9
    camera_d0: int = 15
    camera_d1: int = 16
    camera_d2: int = 17
    camera_d3: int = 18
    camera_d4: int = 38
    camera_d5: int = 39
    camera_d6: int = 40
    camera_d7: int = 41
    camera_vsync: int = 42
    camera_href: int = 47
    camera_pclk: int = 48
    usb_dm: int = 20
    usb_dp: int = 19


# ---------------------------------------------------------------------------
# Board configuration
# ---------------------------------------------------------------------------

@dataclass
class ESP32S3BoardConfig:
    """Complete hardware configuration for an ESP32-S3 board.

    The ESP32-S3 features a dual-core Xtensa LX7 CPU at 240 MHz,
    512 KB SRAM, 8 MB PSRAM (octal SPI), 8 MB flash, WiFi 6,
    BLE 5.0, USB OTG, and a DVP camera interface.
    """

    board_name: str = "ESP32-S3"
    cpu: str = "Xtensa LX7"
    clock_mhz: int = 240
    cores: int = 2
    flash_mb: int = 8
    sram_kb: int = 512
    psram_mb: int = 8
    gpio_count: int = 45
    adc_channels: int = 20
    pwm_channels: int = 8
    uart_count: int = 2
    spi_count: int = 2
    i2c_count: int = 2
    wifi: bool = True
    wifi_6: bool = True
    ble: bool = True
    ble_version: str = "5.0"
    usb_otg: bool = True
    camera_dvp: bool = True
    pin_map: ESP32S3PinMap = field(default_factory=ESP32S3PinMap)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_esp32_s3_config(**overrides: Any) -> ESP32S3BoardConfig:
    """Create an :class:`ESP32S3BoardConfig` with optional overrides.

    Examples:
        >>> cfg = create_esp32_s3_config(board_name="S3-PRIMARY", psram_mb=16)
        >>> cfg.board_name
        'S3-PRIMARY'
    """
    config = ESP32S3BoardConfig()

    nested_map = {
        "pin_map": ESP32S3PinMap,
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
                f"create_esp32_s3_config() got an unexpected keyword argument '{key}'"
            )

    return config
