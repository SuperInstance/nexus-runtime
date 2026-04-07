# NEXUS Runtime

Distributed intelligence platform for industrial marine robotics. LLM agents — not humans — are the primary authors, interpreters, and validators of control code, executing on a bytecode VM that runs on embedded hardware (ESP32-S3) with AI cognition on edge GPUs (Jetson Orin Nano).

## Architecture

```
Tier 3: CLOUD    — Heavy training, fleet management
Tier 2: Jetson   — AI inference, reflex synthesis, trust engine
Tier 1: ESP32-S3 — Bytecode VM, real-time control, safety enforcement
```

## Supported Hardware (50+ Boards)

NEXUS ships with pre-configured deployment profiles for **11 platform families** spanning microcontrollers, edge GPUs, and single-board computers:

| Platform | Boards | Architecture |
|----------|--------|-------------|
| **Arduino** | Uno, Mega, Nano, Due, MKR WiFi 1010, Nano 33 IoT | ATmega328P / AT91SAM3X8E / SAMD21 |
| **ESP32** | Classic, S3, C3, C6, H2 | Xtensa LX6 / RISC-V |
| **ESP8266** | ESP-12E NodeMCU, Wemos D1 Mini | Xtensa L106 |
| **NVIDIA Jetson** | Nano, TX2, Xavier NX, Orin Nano, Orin NX, AGX Orin | ARM A57/A72 + Maxwell/Pascal/Volta/Ampere GPU |
| **Raspberry Pi** | Zero W, 3B+, 4B, 400, 5, CM4, Pico 2 | ARM Cortex-A53/A76 / RP2350 |
| **STM32** | F4, F7, G0, L4, WL, H7, MP1 | ARM Cortex-M0+/M4/M7/A7 |
| **Nordic nRF** | 52810, 52832, 52840, 5340 | ARM Cortex-M4/M33 + BLE/Bluetooth Mesh |
| **Teensy** | 3.6, 4.0, 4.1 | NXP i.MX RT1062 / K66 |
| **i.MX RT** | 1050, 1060, 1064, 1170 | ARM Cortex-M7 (600MHz-1GHz) |
| **RP2040** | Pico, Pico W | ARM Cortex-M0+ + PIO |
| **BeagleBone** | Black, AI-64 | ARM Cortex-A8/A15 + DSP + PRU |

### Quick Discovery

```python
from hardware import list_platforms, list_boards, total_board_count

print(f"{total_board_count()} boards across {len(list_platforms())} platforms")
list_boards("jetson_nano")  # ['jetson-agx-orin', 'jetson-nano', ...]
```

## Directory Structure

- `firmware/` — ESP-IDF project for ESP32-S3 (C, FreeRTOS)
  - `nexus_vm/` — 32-opcode bytecode VM interpreter
  - `wire_protocol/` — COBS/CRC-16/frame/message dispatch
  - `safety/` — Safety state machine, watchdog, heartbeat
  - `drivers/` — Sensor bus and actuator drivers
- `hardware/` — Pre-configured deployment profiles (11 families, 48+ boards)
  - `arduino/`, `esp32/`, `esp8266/`, `jetson_nano/`
  - `raspberry_pi/`, `stm32/`, `nrf52/`, `teensy/`
  - `imx_rt/`, `rp2040/`, `beaglebone/`
- `jetson/` — Python SDK for Jetson (38 modules)
  - `wire_protocol/` — Wire protocol client
  - `reflex_compiler/` — JSON-to-bytecode compiler
  - `trust_engine/` — INCREMENTS trust algorithm
  - `agent_runtime/` — AAB codec, A2A opcodes
  - `swarm/`, `rl/`, `vision/`, `navigation/`, `sensor_fusion/`
  - `fleet_coordination/`, `cooperative_perception/`, `maritime_domain/`
  - `decision_engine/`, `adaptive_autonomy/`, `self_healing/`
  - `xai/`, `knowledge_graph/`, `simulation/`, `digital_twin/`
  - `energy/`, `maintenance/`, `security/`, `compliance/`
  - `marketplace/`, `mission/`, `performance/`, `nl_commands/`
  - `data_pipeline/`, `api_gateway/`, `config_mgmt/`, `learning/`
  - `integration/`, `runtime_verification/`
- `nexus/` — Core runtime modules (vm, wire, trust, aab, bridge, orchestrator)
- `shared/` — Cross-platform definitions (opcodes, instruction format)
- `tests/` — Test suites (pytest, 2200+ tests)
- `schemas/` — JSON schemas for wire protocol, autonomy, reflex definitions

## Quick Start

### Firmware (requires ESP-IDF)

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

### Jetson SDK

```bash
cd jetson
pip install -r requirements.txt
python -m pytest tests/ -v
```

### Hardware Discovery (no hardware required)

```bash
cd hardware
python -c "from hardware import list_all_boards; import pprint; pprint.pprint(list_all_boards())"
```

### Run All Tests

```bash
python -m pytest --tb=short -q
```

## Opcodes

### Core (0x00-0x1F) — 32 opcodes

| Range | Category | Count |
|-------|----------|-------|
| 0x00-0x07 | Stack (NOP, PUSH, POP, DUP, SWAP, ROT) | 8 |
| 0x08-0x10 | Arithmetic (ADD, SUB, MUL, DIV, NEG, ABS, MIN, MAX, CLAMP) | 9 |
| 0x11-0x15 | Comparison (EQ, LT, GT, LTE, GTE) | 5 |
| 0x16-0x19 | Logic (AND, OR, XOR, NOT) | 4 |
| 0x1A-0x1C | I/O (READ_PIN, WRITE_PIN, READ_TIMER) | 3 |
| 0x1D-0x1F | Control (JUMP, JUMP_IF_FALSE, JUMP_IF_TRUE) | 3 |

### A2A (0x20-0x56) — 29 opcodes (NOP on ESP32)

Intent, Agent Communication, Capability Negotiation, Safety Augmentation.

## Safety

Four-tier safety architecture (non-negotiable):
1. **Hardware**: Kill switch, watchdog IC, polyfuses
2. **Firmware**: ISR guard, safe-state outputs
3. **Supervisory**: Heartbeat monitoring, state machine
4. **Application**: Trust-score-gated autonomy (L0-L5)

## License

MIT
