"""NEXUS Marine Robotics Platform - ESP32-H2 Board Configuration.

Provides dataclass-based hardware configuration for the ESP32-H2
(RISC-V single-core) as used in the NEXUS marine robotics platform.
The ESP32-H2 is optimized for IoT gateway applications with native
Matter/Thread/Zigbee support, Bluetooth 5 LE, and IEEE 802.15.4,
making it ideal as a NEXUS fleet IoT gateway bridging wireless
sensor networks to the NEXUS backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Pin Map
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ESP32H2PinMap:
    """GPIO pin assignments for peripherals connected to the ESP32-H2.

    The ESP32-H2 has 19 GPIO pins (GPIO0-GPIO18). This mapping reflects
    the typical routing for NEXUS IoT gateway peripherals.
    """

    gps_tx: int = 4
    gps_rx: int = 5
    imu_sda: int = 6
    imu_scl: int = 7
    sonar_trig: int = 2
    sonar_echo: int = 3
    servo_1: int = 8
    servo_2: int = 9
    led: int = 10
    temp_pin: int = 1
    pressure_sda: int = 6
    pressure_scl: int = 7


# ---------------------------------------------------------------------------
# IoT Gateway configuration
# ---------------------------------------------------------------------------

@dataclass
class IoTGatewayConfig:
    """NEXUS IoT gateway configuration for the ESP32-H2.

    The ESP32-H2 serves as a gateway node in the NEXUS fleet, bridging
    Matter/Thread/Zigbee wireless sensor networks to the NEXUS backbone
    over serial or Ethernet connections.
    """

    gateway_enabled: bool = True
    gateway_id: str = ""
    thread_enabled: bool = True
    thread_network_name: str = "NEXUS-Thread"
    thread_pan_id: int = 0xDEAD
    thread_channel: int = 15
    matter_fabric: bool = True
    zigbee_coordinator: bool = True
    max_thread_nodes: int = 32
    max_zigbee_nodes: int = 64
    max_matter_devices: int = 32
    telemetry_bridging: bool = True
    command_forwarding: bool = True
    ota_update_enabled: bool = True
    serial_baud_rate: int = 115200
    backbone_interface: str = "serial"   # serial, ethernet, wifi-bridge


# ---------------------------------------------------------------------------
# Board configuration
# ---------------------------------------------------------------------------

@dataclass
class ESP32H2BoardConfig:
    """Complete hardware configuration for an ESP32-H2 board.

    The ESP32-H2 features a single-core 32-bit RISC-V CPU at 96 MHz,
    256 KB SRAM, 4 MB flash, Bluetooth 5 LE, IEEE 802.15.4 (for Zigbee
    and Thread), and native Matter protocol support.  Note: the ESP32-H2
    does NOT have WiFi — it uses 802.15.4 for low-power mesh networking
    instead.
    """

    board_name: str = "ESP32-H2"
    cpu: str = "RISC-V"
    clock_mhz: int = 96
    cores: int = 1
    flash_mb: int = 4
    sram_kb: int = 256
    psram_mb: int = 0
    gpio_count: int = 19
    adc_channels: int = 2
    pwm_channels: int = 4
    uart_count: int = 1
    spi_count: int = 1
    i2c_count: int = 1
    wifi: bool = False               # No WiFi on H2
    wifi_6: bool = False
    ble: bool = True
    ble_version: str = "5.0"
    zigbee: bool = True
    thread: bool = True
    matter: bool = True
    ieee_802_15_4: bool = True
    deep_sleep: bool = True
    pin_map: ESP32H2PinMap = field(default_factory=ESP32H2PinMap)
    iot_gateway: IoTGatewayConfig = field(default_factory=IoTGatewayConfig)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_esp32_h2_config(**overrides: Any) -> ESP32H2BoardConfig:
    """Create an :class:`ESP32H2BoardConfig` with optional overrides.

    Examples:
        >>> cfg = create_esp32_h2_config(board_name="H2-GATEWAY-01")
        >>> cfg.board_name
        'H2-GATEWAY-01'
    """
    config = ESP32H2BoardConfig()

    nested_map = {
        "pin_map": ESP32H2PinMap,
        "iot_gateway": IoTGatewayConfig,
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
                f"create_esp32_h2_config() got an unexpected keyword argument '{key}'"
            )

    return config
