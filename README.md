<p align="center">
  <img src="https://img.shields.io/badge/tests-2287%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
  <img src="https://img.shields.io/badge/python-%3E%3D3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/boards-50%2B-orange" alt="Boards">
  <img src="https://img.shields.io/badge/platforms-11-informational" alt="Platforms">
  <img src="https://img.shields.io/badge/opcodes-32%20core%20%2B%2029%20A2A-purple" alt="Opcodes">
</p>

<h1 align="center">NEXUS Runtime</h1>

<p align="center">
  <strong>LLM agents write the control code. A bytecode VM executes it. A trust engine decides if it's safe.</strong>
</p>

---

NEXUS is a distributed intelligence platform for industrial marine robotics where **LLM agents — not humans — are the primary authors of control code**. Agents synthesize intent into bytecode, which runs on a deterministic VM embedded on ESP32-S3 microcontrollers, while AI cognition and trust-based safety enforcement run on edge GPUs (Jetson Orin Nano).

> **The key insight**: instead of hand-coding every control loop, NEXUS lets LLM agents express *intent* (e.g., "maintain 2m depth, avoid obstacles, surface if battery &lt; 15%"), which is compiled into verified bytecode and executed with hardware-enforced safety guarantees.

## Animated Pipeline

```
  Step 1         Step 2            Step 3              Step 4
┌──────────┐  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐
│  INTENT   │  │   BYTECODE   │  │  SAFETY CHECK  │  │  EXECUTION   │
│           │  │              │  │                │  │              │
│ "Maintain │  │ LOAD_CONST   │  │ ✓ Validator    │  │ ESP32-S3 VM  │
│  depth    │──│ READ_IO      │──│ ✓ Watchdog     │──│  ┌─┐ ┌─┐    │
│  at 2m"   │  │ CMP          │  │ ✓ Trust gate   │  │  ├─┤ ├─┤    │
│           │  │ JNZ dive     │  │ ✓ Kill switch  │  │  └─┘ └─┘    │
└──────────┘  └──────────────┘  └────────────────┘  └──────────────┘
  LLM Agent     Reflex Compiler   4-Tier Safety     Real-time HW
```

## Why NEXUS?

| Problem | NEXUS Answer |
|---|---|
| Hand-coded control loops don't scale across 50+ board types | **LLM-authored bytecode** — write intent once, deploy everywhere |
| LLM-generated code is unreliable for safety-critical systems | **Deterministic VM** — only 32 verified opcodes, no dynamic memory, cycle-exact |
| Multi-agent fleets need cooperation without a central controller | **INCREMENTS trust engine** — mathematically grounded, multi-dimensional trust |
| Marine environments destroy electronics | **4-tier safety architecture** — hardware kill switch → firmware ISR → supervisory heartbeat → application trust gate |
| Fleet heterogeneity makes coordination impossible | **3-tier architecture** — ESP32 for real-time, Jetson for cognition, Cloud for training |

## Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║  Tier 3: CLOUD                                                       ║
║  Heavy model training, fleet management, mission planning             ║
║  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐             ║
║  │  Training    │  │  Fleet Mgmt  │  │  Digital Twin    │             ║
║  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘             ║
╠═════════╪════════════════╪════════════════════╪═════════════════════╣
║  Tier 2: JETSON (Edge GPU)                                            ║
║  AI inference, reflex synthesis, trust engine, swarm coordination      ║
║  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌────────────────┐         ║
║  │ Vision   │ │ Trust Eng  │ │ Swarm    │ │ Reflex Compiler│         ║
║  └────┬─────┘ └─────┬──────┘ └────┬─────┘ └───────┬────────┘         ║
╠══════╪═════════════╪══════════════╪════════════════╪════════════════╣
║  Tier 1: ESP32-S3 (Microcontroller)                                     ║
║  Bytecode VM, real-time sensor/actuator control, safety enforcement    ║
║  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌────────────────┐         ║
║  │ 32-op VM │ │ Wire Proto │ │ Safety   │ │ Sensor/Act DRV │         ║
║  └──────────┘ └────────────┘ └──────────┘ └────────────────┘         ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Quick Start

```bash
# Install (editable mode)
cd /tmp/nexus-runtime
pip install -e .

# Verify installation
python -c "import nexus; print(f'NEXUS {nexus.__version__}')"

# Run all tests
python -m pytest --tb=short -q

# Hardware discovery (no hardware required)
python -c "from hardware import total_board_count, list_platforms; \
  print(f'{total_board_count()} boards across {len(list_platforms())} platforms')"

# Try an example
python examples/bytecode_playground.py
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

## Modules

### Core Runtime (`nexus/`)
| Module | Description |
|--------|-------------|
| `nexus.vm` | 32-opcode bytecode VM with assembler, disassembler, and validator |
| `nexus.trust` | INCREMENTS multi-dimensional trust engine |
| `nexus.wire` | COBS/CRC-16 framed wire protocol |
| `nexus.aab` | Autonomous Agent Behavior codec and roles |
| `nexus.bridge` | Git-based bytecode deployment bridge |
| `nexus.orchestrator` | Fleet orchestration and coordination |

### Jetson SDK (`jetson/`) — 38 Modules
| Category | Modules |
|----------|---------|
| **Cognition** | `vision`, `sensor_fusion`, `decision_engine`, `nl_commands` |
| **Cooperation** | `swarm`, `fleet_coordination`, `cooperative_perception`, `trust` |
| **Autonomy** | `adaptive_autonomy`, `self_healing`, `reflex`, `agent_runtime` |
| **Maritime** | `maritime_domain`, `navigation`, `mission`, `compliance` |
| **Infrastructure** | `energy`, `maintenance`, `security`, `performance`, `data_pipeline` |
| **Observability** | `explainability` (XAI), `knowledge_graph`, `digital_twin`, `runtime_verification` |
| **Learning** | `rl`, `learning`, `marketplace`, `mpc` |

### Hardware (`hardware/`) — 11 Platform Families
Pre-configured deployment profiles with pin maps, clock configs, and peripheral drivers for all supported boards.

### Firmware (`firmware/`)
ESP-IDF C project for ESP32-S3: VM interpreter, wire protocol, safety state machine, sensor/actuator drivers.

## Opcode Reference

### Core Opcodes (0x00-0x1F) — 32 opcodes

| Range | Category | Opcodes |
|-------|----------|---------|
| 0x00-0x01 | Control Flow | NOP, LOAD_CONST |
| 0x02-0x03 | Memory | LOAD_REG, STORE_REG |
| 0x04-0x07 | Arithmetic | ADD, SUB, MUL, DIV |
| 0x08-0x0D | Bitwise | AND, OR, XOR, NOT, SHL, SHR |
| 0x0E | Compare | CMP |
| 0x0F-0x13 | Branch | JMP, JZ, JNZ, CALL, RET |
| 0x14-0x15 | Stack | PUSH, POP |
| 0x16-0x17 | I/O | READ_IO, WRITE_IO |
| 0x18-0x19 | System | HALT, SLEEP |
| 0x1A-0x1B | Comms | SEND, RECV |
| 0x1C-0x1F | Memory Mgmt | ALLOC, FREE, DMA_COPY, INTERRUPT |

### A2A Opcodes (0x20-0x56) — 29 opcodes (NOP on ESP32)

Agent-to-agent opcodes for intent broadcasting, capability negotiation, safety augmentation, and cooperative perception fusion. These opcodes are interpreted at the Jetson tier; on ESP32 they execute as NOPs.

## Safety Architecture

Safety is non-negotiable. NEXUS implements a **four-tier defense-in-depth** model:

```
┌─────────────────────────────────────────────────────────────┐
│  TIER 1: HARDWARE                                           │
│  Kill switch • Watchdog IC • Polyfuses • Power rails        │
│  Response time: <1us                                        │
├─────────────────────────────────────────────────────────────┤
│  TIER 2: FIRMWARE (ESP32-S3)                                │
│  ISR guard • Safe-state outputs • Stack canary              │
│  Response time: <1ms                                        │
├─────────────────────────────────────────────────────────────┤
│  TIER 3: SUPERVISORY (Jetson)                               │
│  Heartbeat monitoring • State machine • Watchdog daemon     │
│  Response time: <100ms                                      │
├─────────────────────────────────────────────────────────────┤
│  TIER 4: APPLICATION                                        │
│  Trust-score-gated autonomy (L0-L5) • Bytecode validation   │
│  Response time: <1s                                         │
└─────────────────────────────────────────────────────────────┘
```

- **L0** — Manual control only, all automation disabled
- **L1** — Assisted mode, human approves every action
- **L2** — Supervised autonomy, human can veto
- **L3** — Conditional autonomy, trust-score-gated
- **L4** — High autonomy, fleet cooperation enabled
- **L5** — Full autonomy, emergency-only human intervention

## Examples

| Example | Description |
|---------|-------------|
| [`examples/bytecode_playground.py`](examples/bytecode_playground.py) | Assemble and execute bytecode in the VM emulator |
| [`examples/trust_scenario.py`](examples/trust_scenario.py) | INCREMENTS trust between 5 AUV agents |
| [`examples/flocking_simulation.py`](examples/flocking_simulation.py) | 10-agent Reynolds flocking simulation |
| [`examples/hardware_discovery.py`](examples/hardware_discovery.py) | Explore the hardware catalog |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. All contributions are subject to our [Code of Conduct](CODE_OF_CONDUCT.md). For security concerns, see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) — Copyright (c) 2025 NEXUS Project
