# NEXUS RP2040 Hardware Configuration Library

**Marine Robotics Microcontroller Configuration for the NEXUS Distributed Intelligence Platform**

The NEXUS RP2040 library provides production-grade hardware configuration for the Raspberry Pi Pico (RP2040) microcontroller in marine robotics applications. Leverage the dual-core Cortex-M0+ architecture, 264KB SRAM, and programmable I/O (PIO) state machines to drive sonar arrays, servo-based steering mechanisms, UART sensor bridges, and real-time marine environmental monitoring.

## Features

- **Dual-Core Cortex-M0+ @ 133 MHz** — concurrent sensor polling and control loops
- **264 KB SRAM** — ample buffer space for sonar return data and sensor fusion pipelines
- **PIO State Machines** — deterministic timing for sonar ping/echo, servo PWM, and UART bridging
- **Marine Sensor Pin Mapping** — pre-configured GPIO assignments for sonar, IMU, temperature, pressure, and GPS modules
- **Pico W Extension** — WiFi + BLE wireless telemetry for NEXUS distributed intelligence
- **Comprehensive Test Suite** — 25+ unit tests covering pin validation, frequency calculations, and PIO program assembly

## Installation

```bash
pip install nexus-hardware-rp2040
```

## Quick Start

```python
from nexus_hardware.rp2040 import RP2040Config, PicoWConfig
from nexus_hardware.rp2040.pio_programs import SonarPingProgram, ServoPWMProgram

# Base RP2040 configuration for marine sensor array
config = RP2040Config()
config.set_clock_frequency(133_000_000)
config.configure_marine_sensors()

# Deploy PIO sonar ping program
sonar = SonarPingProgram(trigger_pin=config.PIN_SONAR_TRIG, echo_pin=config.PIN_SONAR_ECHO)
sonar.assemble()

# Pico W with wireless telemetry
pico_w = PicoWConfig()
pico_w.configure_wifi(ssid="NEXUS_MARINE", password="secure_key")
pico_w.configure_ble()
```

## PIO Programs

| Program | Description | State Machines |
|---------|-------------|----------------|
| `SonarPingProgram` | Sonar ping/echo timing with microsecond precision | 1 |
| `ServoPWMProgram` | Multi-channel servo PWM (50 Hz, 1-2 ms pulse) | 1 |
| `UARTBridgeProgram` | Full-duplex UART-to-PIO bridging for sensor telemetry | 2 |

## Pin Mapping Reference

| Function | GPIO Pin | Notes |
|----------|----------|-------|
| Sonar Trigger | GP0 | 10 us pulse |
| Sonar Echo | GP1 | Timeout-safe |
| Servo 1-4 | GP2-GP5 | 50 Hz PWM via PIO |
| I2C SDA/SCL | GP4/GP5 | IMU + barometer |
| UART TX/RX | GP8/GP9 | GPS NMEA bridge |
| SPI MOSI/MISO/CLK | GP11/GP12/GP10 | External flash / ADC |

## Documentation

Full API documentation and marine deployment guides are available at the NEXUS Platform Docs.

## License

MIT License — NEXUS Marine Robotics Project
