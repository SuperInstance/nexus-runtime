# Hardware Configuration Roadmap

## Current State (v0.2.0)

### Supported Platform Families: 11

| # | Platform | Boards | MCU Family | Primary Role |
|---|----------|--------|------------|-------------|
| 1 | Arduino | 6 | ATmega / SAMD | Sensor nodes, actuators, serial bridge |
| 2 | ESP32 | 5 | Xtensa LX6 / RISC-V | WiFi/BLE mesh, deep sleep, IoT gateway |
| 3 | ESP8266 | 2 | Xtensa L106 | Legacy WiFi sensor nodes |
| 4 | NVIDIA Jetson | 6 | ARM + Maxwell/Pascal/Volta/Ampere | AI inference, fleet command, CV |
| 5 | Raspberry Pi | 7 | BCM / RP2040/RP2350 | Companion computers, SBC control |
| 6 | STM32 | 7 | Cortex-M0+/M4/M7/A7 | Real-time control, CAN, LoRa, hybrid |
| 7 | Nordic nRF | 4 | Cortex-M4/M33 | BLE mesh, low-power tracking |
| 8 | Teensy | 3 | i.MX RT / Kinetis | High-speed USB, FlexIO protocols |
| 9 | i.MX RT | 4 | Cortex-M7 | Industrial control, custom protocols |
| 10 | RP2040 | 2 | Cortex-M0+ | PIO-based custom interfaces |
| 11 | BeagleBone | 2 | ARM + DSP + PRU | DSP processing, real-time motor control |

**Total: 48 board configurations**

### Board-to-NEXUS Role Mapping

| NEXUS Role | Recommended Hardware | Trust Level |
|------------|---------------------|-------------|
| `edge_sensor` | Arduino Nano/Uno, STM32G0, nRF52810, ESP8266 | L1-L2 |
| `network_relay` | Teensy 4.1, ESP32-C3, Raspberry Pi 3B+ | L2-L3 |
| `ai_perception` | Jetson Nano, Jetson TX2 | L3-L4 |
| `fleet_commander` | Jetson Orin NX, Jetson AGX Orin | L4-L5 |
| `io_gateway` | BeagleBone Black, STM32MP1 | L3 |
| `mesh_beacon` | nRF52840, ESP32-H2, STM32WL | L1-L2 |
| `marine_cv` | Jetson Xavier NX, Jetson TX2 | L3-L4 |
| `motor_controller` | STM32F4/F7, BeagleBone AI-64 (PRU) | L2-L3 |

## Planned Expansions

### Phase 7 — Additional Platform Families

| Platform | Boards | Priority | Notes |
|----------|--------|----------|-------|
| **STM32H5** | H562, H563, H573 | High | New ultra-efficient Cortex-M33 |
| **STM32U5** | U585, U595, U599 | High | Latest ultra-low-power line |
| **CH32V** | CH32V003, CH32V103, CH32V307 | Medium | RISC-V MCUs, low cost |
| **WCH ESP32-C6** | C6 variants | Medium | RISC-V WiFi 6 |
| **MCP2515 CAN** | Standalone CAN controller | Medium | CAN bus for Arduino/RPi |
| **BME680/BME280** | Environmental sensor configs | Medium | Weather/ocean sensors |
| **Pixhawk** | Pixhawk 4, Cube Orange | High | ArduPilot/PX4 integration |
| **ArduPilot Companion** | Companion computer configs | High | MAVLink bridge to NEXUS |

### Phase 8 — Marine-Grade Hardware

| Hardware | Config | Notes |
|----------|--------|-------|
| **BlueROV2** | ESC/Thruster configs | Pre-configured ROV deployment |
| **WaterLinked** | UWB underwater positioning | Localization integration |
| **Navio2** | RPi HAT for GPS/IMU | Marine navigation |
| **Raspberry Pi CM4** | Industrial carrier boards | Enclosed deployments |
| **PCIE-based Jetson** | Orin AGX industrial | Flagship marine computing |

### Phase 9 — Fleet-Scale Infrastructure

| Component | Description |
|-----------|-------------|
| **Fleet manifest schema** | JSON schema for multi-vessel hardware descriptions |
| **Auto-detection service** | Runtime board detection via serial signatures |
| **OTA update profiles** | Board-specific firmware/binary update configs |
| **Hardware abstraction tests** | HIL test harness templates per platform |
| **Performance benchmark suite** | Per-board CPU/memory/latency benchmarks |

## SEO Strategy

Each board configuration lives in its own directory path that matches common search patterns:

```
hardware/arduino/uno/          → "nexus arduino uno"
hardware/esp32/s3/             → "nexus esp32-s3"
hardware/jetson/orin-nano/     → "nexus jetson orin nano"
hardware/stm32/f4/             → "nexus stm32f4"
```

READMEs in each directory include:
- Board specs and pin mappings
- Getting started with NEXUS on this board
- Example deployment commands
- Links to related boards in the NEXUS ecosystem
