# NEXUS Runtime

Distributed intelligence platform for industrial robotics. LLM agents — not humans — are the primary authors, interpreters, and validators of control code, executing on a bytecode VM that runs on embedded hardware (ESP32-S3) with AI cognition on edge GPUs (Jetson Orin Nano).

## Architecture

```
Tier 3: CLOUD    — Heavy training, fleet management
Tier 2: Jetson   — AI inference, reflex synthesis, trust engine
Tier 1: ESP32-S3 — Bytecode VM, real-time control, safety enforcement
```

## Directory Structure

- `firmware/` — ESP-IDF project for ESP32-S3 (C, FreeRTOS)
  - `nexus_vm/` — 32-opcode bytecode VM interpreter
  - `wire_protocol/` — COBS/CRC-16/frame/message dispatch
  - `safety/` — Safety state machine, watchdog, heartbeat
  - `drivers/` — Sensor bus and actuator drivers
- `jetson/` — Python SDK for Jetson Orin Nano
  - `wire_protocol/` — Wire protocol client
  - `reflex_compiler/` — JSON-to-bytecode compiler
  - `trust_engine/` — INCREMENTS trust algorithm
  - `agent_runtime/` — AAB codec, A2A opcodes
  - `learning/` — Observation recording
- `shared/` — Cross-platform definitions (opcodes, instruction format)
- `tests/` — Test suites (Unity for firmware, pytest for Jetson, HIL skeletons)

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
python -m pytest ../tests/jetson/ -v
```

### Host Tests (no hardware required)

```bash
mkdir -p tests/firmware/build && cd tests/firmware/build
cmake .. -DCMAKE_SOURCE_DIR=../..
make -j$(nproc)
./test_firmware
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
