"""NEXUS Marine Robotics Platform - ESP32-C3 Board Configuration.

Provides dataclass-based hardware configuration for the ESP32-C3
(RISC-V single-core) as used in the NEXUS marine robotics platform.
The ESP32-C3 is an ultra-low-power RISC-V MCU with WiFi 4, BLE 5 (LE),
and support for Zigbee, Thread, and Matter protocols, making it ideal
for battery-powered NEXUS sensor nodes requiring minimal energy consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Pin Map
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ESP32C3PinMap:
    """GPIO pin assignments for peripherals connected to the ESP32-C3.

    The ESP32-C3 has 22 GPIO pins (GPIO0-GPIO21). This mapping reflects
    the typical routing for NEXUS ultra-low-power sensor peripherals.
    """

    gps_tx: int = 4
    gps_rx: int = 5
    imu_sda: int = 8
    imu_scl: int = 9
    sonar_trig: int = 2
    sonar_echo: int = 3
    servo_1: int = 6
    servo_2: int = 7
    led: int = 10
    temp_pin: int = 1
    pressure_sda: int = 8
    pressure_scl: int = 9


# ---------------------------------------------------------------------------
# Ultra-low-power NEXUS node configuration
# ---------------------------------------------------------------------------

@dataclass
class NexusNodeConfig:
    """NEXUS ultra-low-power node configuration for ESP32-C3.

    The ESP32-C3's RISC-V architecture and low-power modes enable
    deployment as long-lived battery-powered NEXUS sensor nodes.
    """

    deep_sleep_enabled: bool = True
    deep_sleep_us: int = 5_000_000        # 5 seconds between readings
    wake_on_gpio: bool = True
    wake_gpio_pin: int = 3               # Sonar echo triggers wake
    low_power_clock_mhz: int = 32         # XTAL 32 kHz for deep sleep
    sensor_burst_mode: bool = True       # Rapid sensor sampling then sleep
    max_sensor_channels: int = 4
    telemetry_compression: bool = True    # Compress telemetry before TX
    battery_monitor_adc: int = 0          # ADC channel for battery voltage
    low_battery_threshold_mv: int = 3000  # 3.0 V cutoff
    node_role: str = "end_device"


# ---------------------------------------------------------------------------
# Board configuration
# ---------------------------------------------------------------------------

@dataclass
class ESP32C3BoardConfig:
    """Complete hardware configuration for an ESP32-C3 board.

    The ESP32-C3 features a single-core 32-bit RISC-V CPU at 160 MHz,
    400 KB SRAM, 4 MB flash, WiFi 4 (802.11 b/g/n), BLE 5 (LE),
    and ultra-low-power deep sleep modes suitable for battery-powered
    NEXUS sensor nodes.
    """

    board_name: str = "ESP32-C3"
    cpu: str = "RISC-V"
    clock_mhz: int = 160
    cores: int = 1
    flash_mb: int = 4
    sram_kb: int = 400
    psram_mb: int = 0
    gpio_count: int = 22
    adc_channels: int = 6
    pwm_channels: int = 6
    uart_count: int = 2
    spi_count: int = 1
    i2c_count: int = 1
    wifi: bool = True
    wifi_6: bool = False
    ble: bool = True
    ble_version: str = "5.0"
    zigbee: bool = True
    thread: bool = True
    matter: bool = True
    deep_sleep: bool = True
    pin_map: ESP32C3PinMap = field(default_factory=ESP32C3PinMap)
    nexus_node: NexusNodeConfig = field(default_factory=NexusNodeConfig)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_esp32_c3_config(**overrides: Any) -> ESP32C3BoardConfig:
    """Create an :class:`ESP32C3BoardConfig` with optional overrides.

    Examples:
        >>> cfg = create_esp32_c3_config(board_name="C3-SENSOR-NODE")
        >>> cfg.board_name
        'C3-SENSOR-NODE'
    """
    config = ESP32C3BoardConfig()

    nested_map = {
        "pin_map": ESP32C3PinMap,
        "nexus_node": NexusNodeConfig,
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
                f"create_esp32_c3_config() got an unexpected keyword argument '{key}'"
            )

    return config
