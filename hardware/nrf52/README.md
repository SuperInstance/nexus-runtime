# NEXUS nRF52 Hardware Configuration Library

**BLE Marine Robotics Wireless Sensor Configuration for the NEXUS Distributed Intelligence Platform**

The NEXUS nRF52 library provides hardware configuration and BLE GATT profile definitions for Nordic Semiconductor nRF52840 wireless microcontrollers deployed in marine robotics sensor networks. Supports BLE 5.0, Zigbee, and Thread protocols for real-time wireless sensor telemetry, NEXUS mesh networking, and underwater-to-surface data relay systems.

## Features

- **ARM Cortex-M4 @ 64 MHz with FPU** — floating-point math for sensor data processing on-chip
- **1 MB Flash + 256 KB RAM** — ample storage for wireless protocol stacks and sensor firmware
- **BLE 5.0 with 2 Mbps PHY** — high-throughput Bluetooth Low Energy for marine sensor data
- **Zigbee & Thread** — IEEE 802.15.4 mesh networking for multi-sensor deployments
- **NEXUS GATT Profiles** — custom BLE service and characteristic definitions for marine sensor data
- **NFC** — on-chip NFC-A tag for pairing and configuration
- **Comprehensive Test Suite** — 25+ unit tests covering chip config, memory layout, and BLE profile validation

## Installation

```bash
pip install nexus-hardware-nrf52
```

## Quick Start

```python
from nexus_hardware.nrf52 import NRF52840Config
from nexus_hardware.nrf52.ble_profiles import NEXUSMarineService

# Configure nRF52840 for marine sensor node
chip = NRF52840Config()
chip.configure_marine_node()

# Set up NEXUS marine BLE service
service = NEXUSMarineService()
service.add_temperature_characteristic()
service.add_depth_characteristic()
service.add_gps_characteristic()
gatt_table = service.build_gatt_table()
```

## Supported Protocols

| Protocol | Version | Use Case |
|----------|---------|----------|
| BLE | 5.0 | Primary sensor telemetry |
| Zigbee | 3.0 | Multi-sensor mesh networks |
| Thread | 1.3 | IP-based mesh networking |
| NFC-A | ISO 14443-3A | Device pairing / config |

## NEXUS BLE Services

| Service UUID | Description |
|-------------|-------------|
| `6E787873-0001-4000-8000-001122334455` | NEXUS Marine Sensor Service |
| `6E787873-0002-4000-8000-001122334455` | NEXUS Device Info Service |

## License

MIT License — NEXUS Marine Robotics Project
