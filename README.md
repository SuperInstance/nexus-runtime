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

## Overview

NEXUS Runtime is the **executable core** of the NEXUS distributed intelligence platform — a Python package (v0.2.1) that provides the complete software stack for autonomous marine robotics fleets. Where the companion [Edge-Native](https://github.com/nexus-platform/Edge-Native) repository defines *what to build* through exhaustive specifications, NEXUS Runtime is *what runs*: a deterministic bytecode virtual machine, a COBS-framed wire protocol, a multi-dimensional trust engine, a fleet orchestrator, and hardware drivers for 50+ embedded boards.

The runtime embodies a radical inversion of the conventional robotics paradigm: **LLM agents — not humans — are the primary authors of control code**. Agents express intent in natural language (e.g., "maintain 2m depth, avoid obstacles, surface if battery < 15%"), which a reflex compiler translates into verified bytecode executed on ESP32-S3 microcontrollers. Meanwhile, AI cognition and trust-based safety enforcement run on edge GPUs (Jetson Orin Nano). The result is a system where each "limb" (ESP32 node) thinks, reacts, and learns independently — like a biological ribosome translating mRNA into proteins without comprehension.

The runtime ships with 2,287 passing tests across unit, integration, and hardware-in-loop suites, targets Python 3.11+, and supports 11 platform families spanning microcontrollers, edge GPUs, and single-board computers.

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
                        ┌─────────────────────────────────────────────────────┐
                        │  TIER 3: CLOUD                                     │
                        │  Heavy model training • Fleet management            │
                        │  Mission planning • Digital twin simulation         │
                        │                                                     │
                        │  ┌───────────┐  ┌───────────┐  ┌───────────────┐  │
                        │  │ Training  │  │ Fleet Mgmt│  │ Digital Twin  │  │
                        │  └─────┬─────┘  └─────┬─────┘  └──────┬────────┘  │
                        └────────┼──────────────┼───────────────┼────────────┘
                                 │   Starlink/   │               │
                                 │   5G / LTE    │               │
                        ┌────────┼──────────────┼───────────────┼────────────┐
                        │  TIER 2: JETSON ORIN NANO (Edge GPU)             │
                        │  40 TOPS INT8 • 8GB LPDDR5 • AI Inference        │
                        │                                                     │
                        │  ┌─────────┐ ┌──────────┐ ┌───────┐ ┌─────────┐   │
  LLM Agent ────────────┤  │ Vision  │ │  Trust   │ │ Swarm │ │  Reflex │   │
  Intent NL  ───────────┤  │ Pipeline│ │  Engine  │ │ Coord │ │Compiler │   │
                        │  └────┬────┘ └────┬─────┘ └───┬───┘ └────┬────┘   │
                        └───────┼───────────┼───────────┼──────────┼─────────┘
                                │           │  TRUST    │  BYTECODE │
                                │           │  SCORES   │  DEPLOY  │
                        ┌───────┼───────────┼───────────┼──────────┼─────────┐
                        │  TIER 1: ESP32-S3 (Microcontroller)               │
                        │  240MHz Dual-Core • 8MB PSRAM • 45 GPIO           │
                        │                                                     │
                        │  ┌───────┐ ┌─────────┐ ┌───────┐ ┌───────────┐   │
                        │  │ 32-op│ │  Wire   │ │Safety │ │ Sensor /  │   │
                        │  │  VM  │ │ Protocol│ │  SM   │ │ Actuator  │   │
                        │  └───────┘ └─────────┘ └───────┘ └───────────┘   │
                        └───────────────────────────────────────────────────┘

                        ┌───────────────────────────────────────────────────┐
                        │  HARDWARE INTERLOCK (Tier 0)                      │
                        │  Kill switch • Watchdog IC • Polyfuses            │
                        │  Response: <1µs  —  operates regardless of FW     │
                        └───────────────────────────────────────────────────┘
```

**Data flow**: An LLM agent expresses intent → Jetson compiles reflex to bytecode → Bytecode validated by 4-tier safety pipeline → Deployed via COBS/CRC-16 wire protocol at 921,600 baud → ESP32-S3 VM executes at 1ms ticks → Sensor data streamed back → Trust engine scores interactions → Fleet orchestrator distributes tasks.

## Core Concepts

### Bytecode VM (`nexus.vm`)

A deterministic, register-based virtual machine with 32 opcodes, 32 registers (16 GP + 16 IO-mapped), 64KB addressable memory, and a 1024-entry hardware stack. Every instruction is 8 bytes, fixed-width, little-endian. The VM enforces cycle budgets (default 100,000 cycles), bounds-checked memory, and IO-register isolation. No dynamic allocation. No garbage collection. Given the same inputs, it produces the same outputs in the same number of cycles — every time.

### INCREMENTS Trust Engine (`nexus.trust`)

A multi-dimensional trust model that computes composite trust scores between agents:

```
T(a,b,t) = α·T_history + β·T_capability + γ·T_latency + δ·T_consistency
```

Where `α=0.35, β=0.25, γ=0.20, δ=0.20` by default. Trust decays exponentially toward neutral (0.5) over time, requiring sustained good behavior to maintain high scores. This creates a natural 25:1 loss-to-gain ratio — trust is earned slowly and lost rapidly, ensuring safety.

### Wire Protocol (`nexus.wire`)

A reliable serial communication layer using COBS (Consistent Overhead Byte Stuffing) framing with CRC-16/CCITT-FALSE integrity checks. Supports 28 message types across system, sensor, command, telemetry, trust, swarm, A2A, and data categories. Frame format: `[0xAA55 preamble][length][COBS-encoded payload][CRC-16]`. Maximum frame size: 4096 bytes.

### Fleet Orchestrator (`nexus.orchestrator`)

Manages multi-vessel fleets with task submission, prioritized scheduling, resource-aware assignment, and workload balancing. Tasks have lifecycle states (PENDING → ASSIGNED → IN_PROGRESS → COMPLETED/FAILED/CANCELLED) and four priority levels (LOW, NORMAL, HIGH, CRITICAL). Vessels are matched to tasks via capability profiles and current load.

### Node Lifecycle (`nexus.core.node`)

Every NEXUS node follows a deterministic lifecycle: `INIT → CONNECTING → ACTIVE → DEGRADED → RECOVERY → SHUTDOWN`. Transitions are validated, hookable, and fully auditable. Nodes track health metrics with configurable thresholds and maintain complete state histories.

### Autonomous Agent Behavior (`nexus.aab`)

An agent-first extension to the bytecode format. AAB adds TLV metadata (Intent, Capability, Safety, Trust, Narrative tags) alongside the 8-byte core instructions. On ESP32, metadata is stripped at zero overhead. On Jetson, agents read and reason about the annotations for cooperative decision-making.

## Quick Start

```bash
# Install (editable mode)
cd /tmp/nexus-runtime
pip install -e .

# Verify installation
python -c "import nexus; print(f'NEXUS {nexus.__version__}')"

# Run all tests (2,287 tests)
python -m pytest --tb=short -q

# Hardware discovery (no hardware required)
python -c "from hardware import total_board_count, list_platforms; \
  print(f'{total_board_count()} boards across {len(list_platforms())} platforms')"

# Try an example
python examples/bytecode_playground.py
```

## API Reference

### VM Executor

```python
from nexus.vm.executor import Executor, Opcodes, Instruction

# Create a VM with IO callbacks
vm = Executor(
    program=bytecode,
    io_read_cb=lambda idx: sensor_values[idx],
    io_write_cb=lambda idx, val: set_actuator(idx, val),
)

# Execute
vm.run(max_cycles=100_000)          # Run until halted or budget
insn = vm.step()                    # Single-step execution
state = vm.get_state()              # Snapshot: PC, registers, stack, flags

# IO simulation
vm.push_recv(42)                    # Feed data to RECV opcode
msg = vm.pop_send()                 # Retrieve data from SEND opcode
vm.push_interrupt(3)                # Queue external interrupt
```

### Trust Engine

```python
from nexus.trust.engine import TrustEngine, CapabilityProfile

engine = TrustEngine()
engine.register_agent("AUV-001", capabilities=CapabilityProfile(navigation=0.9, sensing=0.7))
engine.register_agent("AUV-002", capabilities=CapabilityProfile(sensing=0.95))

# Record interactions
engine.record_interaction("AUV-001", "AUV-002", success=True, latency_ms=45.0)
score = engine.get_trust("AUV-001", "AUV-002")    # → 0.0-1.0

# Find best partner
agent_id, trust = engine.get_most_trusted("AUV-001")
```

### Wire Protocol

```python
from nexus.wire.protocol import Message, MessageType, encode_frame, decode_frame

msg = Message(msg_type=MessageType.SENSOR_DATA, source=1, destination=0, payload=data)
frame = encode_frame(msg)              # → bytes with COBS + CRC-16
decoded = decode_frame(frame)           # → Message or None (CRC fail)

# Direct CRC and COBS access
from nexus.wire.protocol import CRC16, COBSCodec
crc = CRC16.compute(data)              # CRC-16/CCITT-FALSE
encoded = COBSCodec.encode(data)        # Zero-free encoding
```

### Fleet Orchestrator

```python
from nexus.orchestrator.fleet import FleetOrchestrator, VesselInfo, TaskPriority

fleet = FleetOrchestrator()
fleet.register_vessel(VesselInfo(vessel_id="AUV-001", capabilities={"navigation": 0.9}))
task = fleet.submit_task("Survey Area A", priority=TaskPriority.HIGH,
                         required_capabilities={"navigation": 0.7})
result = fleet.assign_task(task.task_id)   # Auto-selects best vessel
status = fleet.get_fleet_status()          # Full fleet snapshot
```

### Node Lifecycle

```python
from nexus.core.node import Node, HealthMetric

node = Node(node_id="AUV-001", name="Port Scanner")
node.on_transition(lambda n, old, new: print(f"{old} → {new}"))
node.start()                                        # INIT → CONNECTING → ACTIVE
node.add_health_metric(HealthMetric("battery", 85, "%", 20, 100))
node.report_degraded("Low light conditions")
node.recover()                                       # RECOVERY → ACTIVE
```

## Integration

### Python Package Integration

```python
# pyproject.toml or requirements.txt
nexus-runtime = ">=0.2.1"
```

```python
import nexus
from nexus.vm import Executor, Assembler
from nexus.trust import TrustEngine
from nexus.wire import encode_frame, Message, MessageType
from nexus.orchestrator import FleetOrchestrator
from nexus.aab import BehaviorCodec
```

### Firmware Integration (C)

The Python runtime mirrors the ESP-IDF firmware implementation byte-for-byte. Shared headers in `shared/` define opcodes, instruction formats, and wire protocol constants used by both the Python emulator and the C firmware:

```
shared/
├── instruction.h    # 8-byte instruction format (C struct)
├── opcodes.h        # Opcode enum (C)
└── opcodes.py       # Opcode enum (Python)
```

The firmware under `firmware/` is a complete ESP-IDF project with:
- `firmware/src/core/nexus_vm/` — C VM interpreter (vm_core.c, vm_validate.c, vm_opcodes.c)
- `firmware/src/core/wire_protocol/` — COBS, CRC-16, frame encode/decode, message dispatch
- `firmware/src/safety/` — Watchdog, heartbeat, safety state machine, E-stop ISR
- `firmware/src/drivers/` — Sensor bus (I2C/SPI/1-Wire), actuator drivers, HAL

### Hardware Platform Integration

Deploy to any of the 11 supported platform families using pre-configured profiles:

```python
from hardware.esp32.config_esp32_s3 import ESP32S3Config
from hardware.jetson_nano.config_orin_nano import OrinNanoConfig
from hardware.raspberry_pi.config_pi5 import Pi5Config

# Each config provides: pin_maps, clock_settings, peripheral_drivers, memory_layout
config = ESP32S3Config()
print(config.gpio_pins, config.clock_freq, config.flash_size)
```

### Jetson SDK Integration

The `jetson/` directory provides 38 modules for the cognitive edge layer, organized into 8 categories. Key entry points:

```python
from jetson.reflex.compiler import ReflexCompiler        # NL → bytecode
from jetson.adaptive_autonomy.levels import AutonomyLevel # L0-L5 control
from jetson.swarm.flocking import FlockSimulation          # Multi-agent behavior
from jetson.navigation.pilot import Pilot                  # Waypoint + collision avoidance
from jetson.energy.power_budget import PowerBudget         # Energy-aware planning
from jetson.security.safety_monitor import SafetyMonitor   # Runtime safety checks
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

---

<img src="callsign1.jpg" width="128" alt="callsign">
