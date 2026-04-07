# NEXUS Raspberry Pi Marine Robotics Configuration

> **NEXUS Distributed Intelligence Platform** — Hardware Abstraction Layer for Raspberry Pi Boards in Marine Robotics Applications

[![NEXUS Platform](https://img.shields.io/badge/NEXUS-Marine_Robotics-0077b6?style=for-the-badge)]()
[![Raspberry Pi](https://img.shields.io/badge/RPi-4B%20%7C%205%20%7C%20Zero2W%20%7C%20CM4-c51a4a?style=flat-square)]()
[![Tests](https://img.shields.io/badge/tests-189%20passing-brightgreen)]()
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()

---

## Overview

The **NEXUS Raspberry Pi hardware configuration library** provides production-ready board descriptions, GPIO pin mappings, and sensor interfaces for deploying Raspberry Pi single-board computers in **marine robotics** environments. This module is part of the NEXUS distributed intelligence platform and handles:

- **Board identification** — SoC, CPU, RAM, and peripheral capability detection
- **GPIO pin mapping** — BCM-to-physical pin translation for marine sensor wiring
- **Marine sensor interfaces** — Pre-configured I2C/SPI/UART routes for common oceanographic sensors (CTD, IMU, GPS, echosounder)
- **Thermal profiling** — Passive cooling limits for waterproof enclosures
- **Power budget modeling** — Current draw estimates for battery-powered ROV/AUV deployments

### Supported Boards

| Board | SoC | CPU | RAM | Use Case |
|-------|-----|-----|-----|----------|
| **Raspberry Pi 4B** | BCM2711 | 4× Cortex-A72 @ 1.5 GHz | 1/2/4/8 GB | Primary compute node, vision processing |
| **Raspberry Pi 5** | BCM2712 | 4× Cortex-A76 @ 2.4 GHz | 4/8 GB | High-performance autonomy, 4K sonar viz |
| **Raspberry Pi Zero 2W** | BCM2710 | 4× Cortex-A53 @ 1.0 GHz | 512 MB | Lightweight sensor node, BLE telemetry |
| **Compute Module 4** | BCM2711 | 4× Cortex-A72 @ 1.5 GHz | 1/2/4/8 GB | Custom carrier boards, embedded hull mounts |

---

## Quick Start

```bash
pip install nexus-hal
```

```python
from nexus.hardware.raspberry_pi import detect_board, get_config

board = detect_board()  # Auto-detect current RPi model
config = get_config(board)

print(f"Board: {config.name}")
print(f"SoC:   {config.soc}")
print(f"GPIO pins available: {len(config.gpio_pins)}")
```

---

## 40-Pin GPIO Header (Pi 4B / Pi 5 / CM4)

```
  ┌──────────────┬──────────────┐
  │ 3V3    1  │  2   5V       │
  │ GPIO2   3  │  4   5V       │
  │ GPIO3   5  │  6   GND      │
  │ GPIO4   7  │  8   GPIO14   │
  │ GND     9  │ 10   GPIO15   │
  │ GPIO17 11  │ 12   GPIO18   │
  │ GPIO27 13  │ 14   GND      │
  │ GPIO22 15  │ 16   GPIO23   │
  │ 3V3    17  │ 18   GPIO24   │
  │ MOSI  19  │ 20   GND      │
  │ MISO  21  │ 22   GPIO25   │
  │ SCLK  23  │ 24   CE0      │
  │ GND    25  │ 26   CE1      │
  │ ID_SD 27  │ 28   ID_SC    │
  │ GPIO5  29  │ 30   GND      │
  │ GPIO6  31  │ 32   GPIO12   │
  │ GPIO13 33  │ 34   GND      │
  │ MISO  35  │ 36   GPIO16   │
  │ GPIO19 37  │ 38   GPIO20   │
  │ GND    39  │ 40   GPIO21   │
  └──────────────┴──────────────┘
```

## Marine Sensor Default Pin Assignments (Pi 4B)

| Sensor | Interface | Pin(s) | Notes |
|--------|-----------|--------|-------|
| CTD (Conductivity/Temperature/Depth) | I2C-1 | GPIO2 (SDA), GPIO3 (SCL) | Address 0x66 |
| BNO055 IMU | I2C-1 | GPIO2 (SDA), GPIO3 (SCL) | Address 0x28 |
| u-blox GPS | UART0 | GPIO14 (TX), GPIO15 (RX) | 9600–115200 baud |
| Blue Robotics Echosounder | UART1 | GPIO0 (TX), GPIO1 (RX) | 115200 baud |
| Water Leak Sensor | GPIO | GPIO17 | Active-low alarm |
| ESC / Thruster PWM | PWM0/1 | GPIO12, GPIO13 | 50 Hz output |
| Servo (manipulator) | PWM | GPIO18 | Standard servo 50 Hz |
| SPI ADC (pressure) | SPI-0 | GPIO10–13, CE0 | Water pressure |
| SPI ADC (dissolved O₂) | SPI-0 | GPIO10–13, CE1 | Dissolved oxygen |
| CAN Bus (NMEA 2000) | SPI-0 | GPIO10–13, CS=GPIO22 | MCP2515 |
| Telemetry Radio | UART2 | GPIO4 (TX), GPIO5 (RX) | 460800 baud |

## Thermal & Power Guidelines

| Board | Idle | Max | Throttle | Enclosure |
|-------|------|-----|----------|-----------|
| Pi 4B | 3.0 W | 7.6 W | 80 °C | Heatsink + passive |
| Pi 5 | 4.0 W | 12.0 W | 80 °C | Active cooler required |
| Pi Zero 2W | 0.7 W | 1.5 W | 80 °C | Conduction-cooled, potting OK |
| CM4 | 2.5 W | 7.0 W | 80 °C | Custom PCB thermal |

> ⚠️ All boards require conformal coating (IP67+) for saltwater deployment.

## Architecture

```
nexus/hardware/raspberry_pi/
├── __init__.py          # Package init, board detection
├── config_pi4.py        # Pi 4B (BCM2711) — 54 tests
├── config_pi5.py        # Pi 5  (BCM2712) — 43 tests
├── config_pizerow.py    # Pi Zero 2W (BCM2710) — 39 tests
├── sensor_hat.py        # Marine sensor HAT — 53 tests
└── tests/
    ├── conftest.py
    ├── test_config_pi4.py
    ├── test_config_pi5.py
    ├── test_config_pizerow.py
    └── test_sensor_hat.py
```

## License

Proprietary — NEXUS Marine Robotics Platform. All rights reserved.
