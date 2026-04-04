# NEXUS Platform Architecture Plan
## Modular, Git-Native, IDE-Friendly

**Version:** 0.2.0
**Date:** 2026-04-05
**Status:** Design Phase — Pre-Build

---

## 1. What We're Building

NEXUS is a distributed intelligence platform where LLM agents — not humans — are the primary authors, interpreters, and validators of control code. The system executes on a bytecode VM running on embedded hardware (ESP32-S3) with AI cognition on edge GPUs (Jetson Orin Nano).

### What's Built vs What's Missing

| Component | Status | Tests | Spec Coverage |
|-----------|--------|-------|---------------|
| Bytecode VM (32 opcodes) | Built | 52 pass | Matches build-spec |
| Wire Protocol (COBS/CRC-16/Frame) | Built | 30 pass | Matches build-spec |
| Safety System (4-tier state machine) | Built | 21 pass | Partial |
| HAL (sensor/actuator abstraction) | Built | 21 pass | Partial |
| Trust Engine (INCREMENTS) | Built | 40 pass | Matches trust spec |
| AAB Codec (13 TLV tags) | Built | 38 pass | Matches A2A spec |
| A2A Opcodes (29 opcodes) | Built | 38 pass | Matches A2A spec |
| Reflex Compiler (JSON to bytecode) | Built | 25 pass | Partial |
| Safety Policy Validator | Partial | ~10 pass | Missing SR-001 to SR-010 |
| Observation Pipeline | Stub | 0 | Missing 72-field UnifiedObservation |
| A/B Testing Framework | Stub | 0 | Missing Welch's t-test |
| LLM Inference Pipeline | Stub | 0 | Missing Qwen2.5 integration |
| gRPC Cluster API | Missing | 0 | Missing 6 services |
| MQTT Bridge | Missing | 0 | Missing 13 topics |
| Git-Native Workflow | Missing | 0 | This document |

Total: 252 tests pass. 7 modules built. 7 modules remaining.

---

## 2. Modular Architecture — 7 Independent Modules

```
nexus-runtime/
├── core/                    # Module 0: Foundation (REQUIRED by all)
│   ├── opcodes/              # Shared opcode definitions (C + Python)
│   ├── instruction/          # 8-byte instruction format packers
│   └── wire_protocol/        # COBS, CRC-16, frame, message, dispatch
│
├── vm/                      # Module 1: Bytecode VM
│   ├── firmware/             # C for ESP32 (vm_core.c, vm_opcodes.c, vm_validate.c)
│   └── emulator/             # Python VM emulator for host testing
│
├── safety/                   # Module 2: Safety System
│   ├── state_machine/        # 4-tier: NORMAL → DEGRADED → SAFE_STATE → FAULT
│   ├── heartbeat/            # Heartbeat monitor (configurable intervals)
│   ├── watchdog/             # MAX6818-style alternating pattern driver
│   └── policy/               # safety_policy.json SR-001 to SR-010 enforcement
│
├── hal/                      # Module 3: Hardware Abstraction Layer
│   ├── sensor_bus/           # 64 sensor channels with freshness tracking
│   ├── actuator/             # 64 actuator channels with 7 safety profiles
│   └── drivers/              # I2C/SPI/UART concrete drivers
│
├── trust/                    # Module 4: INCREMENTS Trust Engine
│   ├── algorithm/            # 12-parameter 3-branch delta formula
│   ├── events/               # 15 event type classifiers
│   └── levels/               # 6 autonomy levels (L0-L5) with promotion rules
│
├── a2a/                      # Module 5: Agent-to-Agent Runtime
│   ├── aab_codec/            # Agent-Annotated Bytecode (8-byte core + 13 TLV tags)
│   ├── opcodes/              # 29 A2A opcodes (NOP on ESP32, full on Jetson)
│   └── communication/        # TELL/ASK/DELEGATE message routing
│
└── jetson/                   # Module 6: Cognitive SDK (requires all above)
    ├── reflex_compiler/      # JSON → bytecode compilation + safety validation
    ├── trust_engine/         # Per-subsystem trust tracking
    ├── agent_runtime/        # A2A opcode interpreter
    ├── learning/             # Observation + pattern discovery + A/B testing
    └── llm/                  # LLM inference (Qwen2.5-Coder-7B)
```

### Minimum Install Paths

| Tier | Install | What You Get |
|------|---------|--------------|
| Bare Metal | `pip install nexus-core nexus-vm` | Opcodes, wire protocol, VM emulator |
| Safety | + `nexus-safety` | Safety state machine, policy enforcement |
| Platform | + `nexus-hal nexus-trust nexus-a2a` | Full embedded stack |
| Full Stack | `pip install nexus-runtime` | Everything + LLM + learning |

---

## 3. Git-Native Workflows

### 3.1 Branches = Reflex Variants

Every reflex is a git repo. Branches are A/B test variants.

```
reflex-heading-hold/
├── main/                    # Currently deployed
├── experiment/pid-tuning/    # A/B test variant
└── rollback/                # Known-good fallback
```

Switch variants by switching branches. Instant rollback.

### 3.2 Pull Requests = Safety Validation

Every PR triggers automated safety checks:
1. JSON schema validation
2. Bytecode compilation
3. SR-001 through SR-010 enforcement
4. Stack depth analysis (max 256)
5. Cycle budget check (max 10,000)
6. CLAMP_F before WRITE_PIN verification
7. Jump bounds validation
8. Cross-LLM safety validation

### 3.3 Git History = Audit Trail

For regulatory compliance (IEC 61508, EU AI Act):
- Every change recorded with who (human or agent), when, why, and who validated
- Tags = trusted, signed versions
- git log IS the compliance evidence

### 3.4 Pre-commit Hooks

Local safety enforcement before code reaches the repo:
- safety_policy.json validation
- Bytecode structure verification
- CLAMP_F enforcement check

---

## 4. Domain Portability

The VM and wire protocol are domain-agnostic. Domain specificity lives in configuration:

```
domains/
├── marine/
│   ├── safety_policy.json    # COLREGs rules, 80% throttle cap, 45° rudder limit
│   ├── actuator_profiles.json # servo, motor_pwm, relay, solenoid profiles
│   └── reflexes/             # heading-hold, collision-avoidance, waypoint-follow
├── agriculture/
│   ├── safety_policy.json    # 2m proximity stop, 15 km/h speed limit
│   └── reflexes/
├── factory/
│   ├── safety_policy.json    # 0.3m human-robot distance
│   └── reflexes/
└── hvac/
    ├── safety_policy.json    # 5°C-40°C temp range, 70% humidity cap
    └── reflexes/
```

Core bytecode is IDENTICAL across domains. Only parameters differ.

---

## 5. Build Plan: Next Sprints

| Sprint | Goal | Duration |
|--------|------|----------|
| 0.6 | Modular restructure | 2 days |
| 0.7 | Safety policy enforcement (SR-001 to SR-010) | 3 days |
| 0.8 | Observation pipeline (72-field UnifiedObservation) | 3 days |
| 0.9 | A/B testing framework (Welch's t-test, rollback) | 2 days |
| 0.10 | LLM integration (system prompt, GBNF grammar) | 3 days |
| 0.11 | Git-native workflow (hooks, PR templates, reflex repos) | 2 days |
| 0.12 | Domain portability (marine config, HVAC config) | 2 days |

---

*"Bytecode is the source of truth. Human specs are documentation."*
*— Rosetta Stone, 1.2*
