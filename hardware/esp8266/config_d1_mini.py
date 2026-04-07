"""NEXUS Marine Robotics Platform - Wemos D1 Mini Board Configuration.

Provides dataclass-based hardware configuration for the Wemos D1 Mini,
a compact ESP8266-based development board popular for marine IoT sensor
nodes in the NEXUS platform.

The D1 Mini shares the same ESP8266-12E SoC as the NodeMCU but in a smaller
form factor with a different pin arrangement.  It supports 80 MHz/160 MHz
operation, 4 MB flash, and 802.11 b/g/n WiFi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Pin Map (D1 Mini numbering)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class D1MiniPinMap:
    """GPIO pin assignments for peripherals on the Wemos D1 Mini.

    The D1 Mini uses the same GPIO0-GPIO16 as the NodeMCU but with a
    compact header layout.  Pin labels D1-D8 map to specific GPIOs.
    """

    # D1 Mini pin header mapping
    D0: int = 16                 # GPIO16 (wakeup)
    D1: int = 5                  # GPIO5 (SCL)
    D2: int = 4                  # GPIO4 (SDA)
    D3: int = 0                  # GPIO0 (boot strapping)
    D4: int = 2                  # GPIO2 (LED, boot strapping)
    D5: int = 14                 # GPIO14 (SPI SCK)
    D6: int = 12                 # GPIO12 (SPI MISO)
    D7: int = 13                 # GPIO13 (SPI MOSI)
    D8: int = 15                 # GPIO15 (SPI SS, boot strapping)

    # A0 analog input
    A0: int = 17                 # ADC0 (10-bit, 0-3.2V)

    # GPS (SoftwareSerial)
    gps_tx: int = 1              # TXD0
    gps_rx: int = 3              # RXD0

    # IMU (I2C on D1/D2)
    imu_sda: int = 4             # D2 / GPIO4
    imu_scl: int = 5             # D1 / GPIO5

    # Sonar (D5/D6)
    sonar_trig: int = 14         # D5 / GPIO14
    sonar_echo: int = 12         # D6 / GPIO12

    # Servo (D3)
    servo_1: int = 0             # D3 / GPIO0

    # Onboard LED (D4, active low)
    led: int = 2                 # D4 / GPIO2

    # Temperature (analog)
    temp_pin: int = 17           # A0

    # Pressure (I2C, shared bus)
    pressure_sda: int = 4        # GPIO4
    pressure_scl: int = 5        # GPIO5

    # Relay
    relay: int = 16              # D0 / GPIO16

    # SPI pins
    spi_sck: int = 14            # D5
    spi_miso: int = 12           # D6
    spi_mosi: int = 13           # D7
    spi_ss: int = 15             # D8


# ---------------------------------------------------------------------------
# Top-level D1 Mini configuration
# ---------------------------------------------------------------------------

@dataclass
class D1MiniBoardConfig:
    """Complete hardware configuration for a Wemos D1 Mini board.

    Inherits the same ESP8266-12E SoC specs as the NodeMCU variant
    but with the D1 Mini form factor and pin arrangement.
    """

    board_name: str = "Wemos D1 Mini"
    cpu: str = "Xtensa L106"
    clock_mhz: int = 80
    flash_mb: int = 4
    sram_kb: int = 80
    gpio_count: int = 11          # Exposed on header (D0-D8, A0, RST)
    adc_channels: int = 1
    pwm_channels: int = 10
    uart_count: int = 1
    spi_count: int = 1
    i2c_count: int = 1
    wifi: bool = True
    ble: bool = False
    pin_map: D1MiniPinMap = field(default_factory=D1MiniPinMap)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_d1_mini_config(**overrides: Any) -> D1MiniBoardConfig:
    """Create a :class:`D1MiniBoardConfig` with optional overrides.

    Examples:
        >>> cfg = create_d1_mini_config(board_name="D1-TEMP-SENSOR")
        >>> cfg.board_name
        'D1-TEMP-SENSOR'
    """
    config = D1MiniBoardConfig()

    nested_map = {
        "pin_map": D1MiniPinMap,
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
                f"create_d1_mini_config() got an unexpected keyword argument '{key}'"
            )

    return config
