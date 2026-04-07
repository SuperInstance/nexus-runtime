"""NEXUS Marine Robotics Platform - ESP32-C6 Board Configuration.

Provides dataclass-based hardware configuration for the ESP32-C6
(RISC-V single-core) as used in the NEXUS marine robotics platform.
The ESP32-C6 is a cost-effective option with WiFi 6, BLE 5, Zigbee,
Thread, and Matter support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Pin Map
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ESP32C6PinMap:
    """GPIO pin assignments for peripherals connected to the ESP32-C6.

    The ESP32-C6 has 30 GPIO pins. This mapping reflects the typical
    routing for NEXUS sensor and actuator peripherals.
    """

    gps_tx: int = 6
    gps_rx: int = 7
    imu_sda: int = 8
    imu_scl: int = 9
    sonar_trig: int = 2
    sonar_echo: int = 3
    servo_1: int = 4
    servo_2: int = 5
    led: int = 10
    temp_pin: int = 1
    pressure_sda: int = 8
    pressure_scl: int = 9


# ---------------------------------------------------------------------------
# Board configuration
# ---------------------------------------------------------------------------

@dataclass
class ESP32C6BoardConfig:
    """Complete hardware configuration for an ESP32-C6 board.

    The ESP32-C6 features a single-core RISC-V CPU at 160 MHz,
    512 KB SRAM, 4 MB flash, WiFi 6, BLE 5, Zigbee, Thread,
    and Matter protocol support.
    """

    board_name: str = "ESP32-C6"
    cpu: str = "RISC-V"
    clock_mhz: int = 160
    cores: int = 1
    flash_mb: int = 4
    sram_kb: int = 512
    psram_mb: int = 0
    gpio_count: int = 30
    adc_channels: int = 7
    pwm_channels: int = 6
    uart_count: int = 2
    spi_count: int = 1
    i2c_count: int = 1
    wifi: bool = True
    wifi_6: bool = True
    ble: bool = True
    ble_version: str = "5.0"
    zigbee: bool = True
    thread: bool = True
    matter: bool = True
    pin_map: ESP32C6PinMap = field(default_factory=ESP32C6PinMap)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_esp32_c6_config(**overrides: Any) -> ESP32C6BoardConfig:
    """Create an :class:`ESP32C6BoardConfig` with optional overrides.

    Examples:
        >>> cfg = create_esp32_c6_config(board_name="C6-SENSOR-NODE")
        >>> cfg.board_name
        'C6-SENSOR-NODE'
    """
    config = ESP32C6BoardConfig()

    nested_map = {
        "pin_map": ESP32C6PinMap,
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
                f"create_esp32_c6_config() got an unexpected keyword argument '{key}'"
            )

    return config
