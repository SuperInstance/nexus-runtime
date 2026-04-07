# NEXUS nRF52 Hardware Configuration Library

**BLE Marine Robotics Wireless Sensor Configuration for the NEXUS Distributed Intelligence Platform**

The NEXUS nRF52 library provides hardware configuration, BLE GATT profile definitions, and Bluetooth mesh networking for Nordic Semiconductor nRF52 series wireless microcontrollers deployed in marine robotics sensor networks.

## Supported Chips

| Chip | CPU | Clock | Flash | RAM | BLE | Use Case |
|------|-----|-------|-------|-----|-----|----------|
| **nRF52840** | Cortex-M4F | 64 MHz | 1 MB | 256 KB | 5.0 | Primary wireless sensor node |
| **nRF52832** | Cortex-M4F | 64 MHz | 512 KB | 64 KB | 4.2 | Cost-effective BLE relay/beacon |
| **nRF52810** | Cortex-M4 | 64 MHz | 192 KB | 24 KB | 4.2 | Ultra-low-cost disposable sensor tag |
| **nRF5340** | Dual M33 | M7@128 / M4@64 MHz | 1 MB + 256 KB | 512 KB + 64 KB | 5.3 | Advanced multi-sensor, LE Audio, Direction Finding |

## Features

- **ARM Cortex-M4 / M33** — floating-point math for on-chip sensor data processing
- **BLE 4.2 / 5.0 / 5.3** — Bluetooth Low Energy for marine sensor data telemetry
- **Zigbee & Thread** — IEEE 802.15.4 mesh networking for multi-sensor deployments
- **NEXUS GATT Profiles** — custom BLE service and characteristic definitions for marine sensor data
- **NFC** — on-chip NFC-A tag for pairing and configuration (nRF52840, nRF5340)
- **Bluetooth Mesh** — provisioning, relay, friend/low-power node configuration for multi-hop networks
- **Dual-core (nRF5340)** — application core runs NEXUS firmware, network core handles radio
- **Comprehensive Test Suite** — 60+ unit tests covering chip config, memory layout, BLE profile, and mesh validation

## Installation

```bash
pip install nexus-hardware-nrf52
```

## Quick Start

```python
from hardware.nrf52.config_nrf52840 import NRF52840Config
from hardware.nrf52.ble_profiles import NEXUSMarineService

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

| Protocol | nRF52810 | nRF52832 | nRF52840 | nRF5340 |
|----------|----------|----------|----------|----------|
| BLE 4.2 | Yes | Yes | Yes | Yes |
| BLE 5.0 | — | — | Yes | Yes |
| BLE 5.3 | — | — | — | Yes |
| LE Audio | — | — | — | Yes |
| Direction Finding | — | — | — | Yes |
| Zigbee | — | — | Yes | Yes |
| Thread | — | — | Yes | Yes |
| NFC-A | — | — | Yes | Yes |

## Architecture

```
hardware/nrf52/
  __init__.py              Package exports
  config_nrf52840.py       nRF52840 (Cortex-M4F @ 64MHz, 1MB flash, BLE 5.0)
  config_nrf52832.py       nRF52832 (Cortex-M4F @ 64MHz, 512KB flash, BLE 4.2)
  config_nrf52810.py       nRF52810 (Cortex-M4 @ 64MHz, 192KB flash, BLE 4.2)
  config_nrf5340.py        nRF5340 (Dual M33, BLE 5.3, LE Audio, Direction Finding)
  ble_profiles.py          NEXUS GATT service definitions
  mesh_config.py           Bluetooth mesh provisioning and relay configuration
  tests/
    test_config_nrf52840.py
    test_config_nrf52832.py
    test_config_nrf52810.py
    test_config_nrf5340.py
    test_mesh_config.py
```

## License

MIT License — NEXUS Marine Robotics Project
