# NEXUS Raspberry Pi Marine Robotics Configuration

> **NEXUS Distributed Intelligence Platform** — Hardware Abstraction Layer for Raspberry Pi Boards in Marine Robotics Applications

[![NEXUS Platform](https://img.shields.io/badge/NEXUS-Marine_Robotics-0077b6?style=for-the-badge)]()
[![Raspberry Pi](https://img.shields.io/badge/RPi-4B%20%7C%205%20%7C%203B%2B%20%7C%20400%20%7C%20Zero2W%20%7C%20CM4%20%7C%20Pico2-c51a4a?style=flat-square)]()
[![Tests](https://img.shields.io/badge/tests-189%20passing-brightgreen)]()
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()

---

## Overview

The **NEXUS Raspberry Pi hardware configuration library** provides production-ready board descriptions, GPIO pin mappings, and sensor interfaces for deploying Raspberry Pi single-board computers and microcontrollers in **marine robotics** environments. This module handles:

- **Board identification** — SoC, CPU, RAM, and peripheral capability detection
- **GPIO pin mapping** — BCM/RP1-to-physical pin translation for marine sensor wiring
- **Marine sensor interfaces** — Pre-configured I2C/SPI/UART routes for common oceanographic sensors (CTD, IMU, GPS, echosounder)
- **Thermal profiling** — Passive cooling limits for waterproof enclosures
- **Power budget modeling** — Current draw estimates for battery-powered ROV/AUV deployments

### Supported Boards

| Board | SoC | CPU | RAM | Form Factor | Use Case |
|-------|-----|-----|-----|-------------|----------|
| **Raspberry Pi 4B** | BCM2711 | 4× Cortex-A72 @ 1.5 GHz | 1/2/4/8 GB | SBC (85×56 mm) | Primary compute node, vision processing |
| **Raspberry Pi 5** | BCM2712 | 4× Cortex-A76 @ 2.4 GHz | 4/8 GB | SBC (85×56 mm) | High-performance autonomy, 4K sonar viz |
| **Raspberry Pi 3B+** | BCM2837B0 | 4× Cortex-A53 @ 1.4 GHz | 1 GB | SBC (85×56 mm) | Mid-range compute, cost-effective controller |
| **Raspberry Pi 400** | BCM2711 | 4× Cortex-A72 @ 1.5 GHz | 4 GB | Integrated keyboard | Surface control station, portable terminal |
| **Pi Zero 2W** | BCM2710 | 4× Cortex-A53 @ 1.0 GHz | 512 MB | Compact (65×30 mm) | Lightweight sensor node, BLE telemetry |
| **Compute Module 4** | BCM2711 | 4× Cortex-A72 @ 1.5 GHz | 1/2/4/8 GB | SODIMM module | Custom carrier boards, embedded hull mount |
| **Pico 2** | RP2350 | Dual Cortex-M33 @ 150 MHz | 520 KB | Compact (52×21 mm) | Real-time sensor processing, PIO protocols |

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

## Architecture

```
nexus/hardware/raspberry_pi/
  __init__.py          # Package init, board detection, board registry
  config_pi4.py        # Pi 4B (BCM2711)
  config_pi5.py        # Pi 5  (BCM2712, RP1 I/O controller)
  config_pi3b.py       # Pi 3B+ (BCM2837B0)
  config_pi400.py      # Pi 400 (BCM2711, keyboard form factor)
  config_pizerow.py    # Pi Zero 2W (BCM2710)
  config_cm4.py        # Compute Module 4 (BCM2711, SODIMM)
  config_pico2.py      # Pico 2 (RP2350, dual Cortex-M33)
  sensor_hat.py        # Marine sensor HAT configuration
  tests/
    conftest.py
    test_config_pi4.py
    test_config_pi5.py
    test_config_pi3b.py
    test_config_pi400.py
    test_config_cm4.py
    test_config_pizerow.py
    test_config_pico2.py
    test_sensor_hat.py
```

## License

Proprietary — NEXUS Marine Robotics Platform. All rights reserved.
