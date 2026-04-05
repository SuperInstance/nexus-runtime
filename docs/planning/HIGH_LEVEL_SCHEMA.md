# NEXUS High-Level Integration Schema

> **Version**: 2.0 — Git-Native Modular Paradigm
> **Date**: 2026-04-05
> **Status**: DRAFT — Open for parallel discussion before implementation

---

## 1. Executive Summary

NEXUS is a distributed intelligence platform for marine robotics. After completing 5 foundation sprints (252/252 tests passing, ~11K LOC), we are entering Phase 2: **modularization, git-agent integration, and fleet orchestration**.

This document defines the high-level architecture connecting three layers:

1. **EDGE LAYER** — NEXUS Runtime (ESP32 firmware + Jetson SDK)
2. **GIT-AGENT LAYER** — Orchestration intelligence (repo-as-agent paradigm)
3. **FLEET LAYER** — Cocapn ecosystem (fleet management, trust, coordination)

The key insight: **Git is the nervous system.** Every firmware update, trust event, mission log, and coordination decision flows through git operations. This gives us automatic audit trails, A/B testing via branches, rollback via tags, and collaborative building via PRs — all for free.

---

## 2. Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FLEET ORCHESTRATION LAYER                          │
│                     (Cocapn Ecosystem)                                 │
│                                                                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │ fleet-orchestrator │  │ edgenative-ai  │  │ increments-    │            │
│  │ Captain's Bridge   │  │ NEXUS Knowledge│  │ fleet-trust    │            │
│  │ HCQ │ DEB │ Council │  │ VM │ Trust │  │ L0→L5 │ 25:1  │            │
│  └───────┬────────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                       │                    │                     │
│  ┌───────┴───────────────────────┴────────────────────┴────────┐        │
│  │             COORDINATION BUS (GitHub API)                   │        │
│  │  Issues │ PRs │ Branches │ Discussions │ Tags │ Forks       │        │
│  └──────────────────────┬─────────────────────────────────────┘        │
│                         │ MQTT / Webhook / API                         │
├─────────────────────────┼───────────────────────────────────────────────┤
│                     GIT-AGENT LAYER                                    │
│                     (Per-Vessel Orchestration)                          │
│                                                                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │ git-agent #1   │  │ git-agent #2   │  │ git-agent #N   │            │
│  │ Vessel Alpha   │  │ Vessel Bravo   │  │ Vessel Charlie │            │
│  │ .agent/identity│  │ .agent/identity│  │ .agent/identity│            │
│  │ .agent/next    │  │ .agent/next    │  │ .agent/next    │            │
│  │ Heartbeat Loop │  │ Heartbeat Loop │  │ Heartbeat Loop │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                    │                     │
│  ┌───────┴───────────────────┴────────────────────┴────────┐        │
│  │          NEXUS BRIDGE (Wire Protocol Adapter)           │        │
│  │  bytecode deployment │ telemetry ingestion │ trust sync  │        │
│  └──────────────────────┬─────────────────────────────────────┘        │
│                         │ NEXUS Wire Protocol (Serial/UART)           │
├─────────────────────────┼───────────────────────────────────────────────┤
│                     EDGE LAYER (NEXUS Runtime)                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │                    NEXUS RUNTIME                             │       │
│  │                                                              │       │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │       │
│  │  │  Jetson SDK  │    │  ESP32 HAL   │    │  Bytecode VM │   │       │
│  │  │  (Python)    │◄──►│  (C/RTOS)    │◄──►│  (32 opcodes)│   │       │
│  │  │  Compiler    │    │  Sensor Bus  │    │  AAB Ready   │   │       │
│  │  │  Trust Engine│    │  Actuators   │    │  8B instruct.│   │       │
│  │  │  AAB Codec   │    │  Safety SM   │    │  Validator   │   │       │
│  │  └──────────────┘    └──────────────┘    └──────────────┘   │       │
│  │                                                              │       │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │       │
│  │  │  Wire Proto  │    │  Safety Sub  │    │  Learning    │   │       │
│  │  │  COBS+CRC16  │    │  4-Tier      │    │  Observations│   │       │
│  │  │  28 Msg Types│    │  Watchdog    │    │  (minimal)   │   │       │
│  │  │  Frame Parser│    │  E-Stop ISR  │    │              │   │       │
│  │  └──────────────┘    └──────────────┘    └──────────────┘   │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                         │                                             │
│  ┌──────────────────────┴────────────────────────────────────────┐    │
│  │              MARINE ROBOT HARDWARE                             │    │
│  │  Sensors (IMU, GPS, Sonar, Temp, Pressure, Current)           │    │
│  │  Actuators (Thrusters, Rudder, Winches, Lights)               │    │
│  │  Comms (RS-422, WiFi, NMEA 2000, Satellite)                  │    │
│  └──────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component-to-Component Mapping

### 3.1 NEXUS Runtime ↔ git-agent

| NEXUS Component | git-agent Concept | Integration Method |
|---|---|---|
| **AAB Bytecode Programs** | `.agent/next` task queue | Agent compiles behavior → commits bytecode as PR → review → merge → deploy to ESP32 |
| **Wire Protocol** | `lib/comm-link.ts` | Bridge module translates Wire Protocol messages ↔ GitHub API events |
| **Bytecode VM** | `lib/git-cognitive-map.ts` | Git branches = parallel execution paths; PRs = proposals; merges = synthesis |
| **HAL (Sensors/Actuators)** | `lib/equipment.ts` | EquipmentManifest declares each sensor/actuator as discoverable capability |
| **Trust Engine (INCREMENTS)** | `lib/trust-engine.ts` + `lib/forgiveness-trust.ts` | Same mathematical model (25:1 ratio, 3-branch delta, 6 levels) |
| **Safety State Machine** | `lib/emergency-protocol.ts` | Safety events trigger git-agent emergency protocol → fleet-wide alert |
| **Jetson SDK (Python)** | `src/worker.ts` (Cloudflare Worker) | Edge heartbeat on Jetson syncs with cloud heartbeat via MQTT/webhook |
| **Observation Buffer** | `.agent/done` task log | Mission telemetry recorded as git commits with structured metadata |

### 3.2 NEXUS Runtime ↔ Fleet Orchestration

| NEXUS Component | Fleet Repo | Integration Method |
|---|---|---|
| **Trust Events** | `increments-fleet-trust` | Wire Protocol trust events → Cloudflare Worker → fleet-wide trust computation |
| **VM Execution** | `edgenative-ai` | `/api/vm/execute` endpoint for remote bytecode testing and validation |
| **Safety Validation** | `edgenative-ai` | `/api/safety/validate` endpoint for pre-deployment bytecode checks |
| **Rosetta Stone** | `edgenative-ai` | `/api/rosetta/translate` for human intent → bytecode translation |
| **Fleet Registration** | `fleet-orchestrator` | Each vessel registers with HCQ health checks and DEB task bonds |
| **Context Handoff** | `baton-ai` | HMAC-signed context transfer between vessels during relay missions |
| **Consensus** | `tripartite-rs` | Pathos + Logos + Ethos → Rust on ESP32/Jetson |

### 3.3 git-agent ↔ Fleet Orchestration

| git-agent Concept | Fleet Repo | Integration Method |
|---|---|---|
| **Agent Identity** | `fleet-orchestrator` | `.agent/identity` → fleet registry vessel definition |
| **Task Queue** | `fleet-orchestrator` | `.agent/next` ↔ DEB (Deterministic Execution Bonds) |
| **Trust Score** | `increments-fleet-trust` | Per-agent trust propagated fleet-wide (0.85x attenuation, 3-hop) |
| **Coordination** | `cocapn` fleet protocol | Issues/PRs as coordination substrate; Council of Captains for disputes |
| **Discovery** | `lib/discovery.ts` | Scan org repos for `.agent/identity` → auto-discover fleet members |
| **Skill Loading** | `I-know-kung-fu` | Skill cartridges loaded as git submodules or JSON configs |

---

## 4. Data Flow: Mission Lifecycle

```
CAPTAIN creates GitHub Issue: "Survey grid section B7, depth profile to 200m"
                              │
                              ▼
            fleet-orchestrator assigns task to git-agent on Vessel Alpha
                              │
                              ▼
            git-agent heartbeat processes task:
            1. Consult strategist (Kimi K2.5) for approach
            2. Call main LLM (DeepSeek) for plan
            3. Generate AAB bytecode for survey pattern
                              │
                              ▼
            AAB bytecode committed as PR to vessel repo
                              │
                              ▼
            edgenative-ai validates: safety checks + VM dry-run
                              │
                              ▼
            PR merged → bytecode deployed via NEXUS Wire Protocol
                              │
                              ▼
            ESP32 executes bytecode VM:
            - Read depth sensor (opcode READ_PIN)
            - Log depth reading (syscall RECORD_SNAPSHOT)
            - Navigate to next waypoint (opcode JUMP_IF_TRUE)
            - Report position (opcode EMIT_EVENT)
                              │
                              ▼
            Telemetry flows back:
            - ESP32 → Jetson (Wire Protocol over UART)
            - Jetson → git-agent (MQTT/webhook)
            - git-agent commits telemetry as structured data
            - fleet-orchestrator updates vessel health
            - increments-fleet-trust scores events
                              │
                              ▼
            git-agent analyzes results:
            - Dead-reckoning engine processes depth data
            - Working theory: "Section B7 has shelf drop at 45m"
            - Creates PR with bathymetric findings
                              │
                              ▼
            CAPTAIN reviews PR → closes Issue with "Grid B7 complete"
```

---

## 5. Modular Architecture

### 5.1 Minimum Install (Power User)

The bare minimum for a developer to build ground-up on the NEXUS framework:

```
nexus-core/                    # ~3,500 LOC — the irreducible core
├── firmware/
│   ├── nexus_vm/              # Bytecode VM (32 opcodes, ~1,300 LOC)
│   └── wire_protocol/         # COBS, CRC-16, frame parser (~850 LOC)
├── jetson/
│   └── nexus_vm/              # Python compiler, validator, tests (~2,200 LOC)
├── shared/
│   └── bytecode/              # Cross-platform opcode definitions (~210 LOC)
└── CMakeLists.txt             # Monorepo build
```

### 5.2 Standard Install (Marine Developer)

```
nexus-core/                    # (minimum install)
nexus-marine/                  # Marine domain extension
├── firmware/
│   ├── safety/                # 4-tier safety system (~440 LOC)
│   ├── drivers/               # HAL + sensor/actuator drivers (~800 LOC)
│   └── main/app_main.c        # FreeRTOS task creation (~170 LOC)
├── jetson/
│   ├── reflex_compiler/       # JSON→bytecode pipeline (~760 LOC)
│   ├── trust_engine/          # INCREMENTS trust (~570 LOC)
│   └── agent_runtime/         # AAB codec + A2A opcodes (~600 LOC)
├── schemas/                   # JSON config schemas
└── configs/                   # Pin configs, safety limits, trust params
```

### 5.3 Fleet Install (Full Deployment)

```
nexus-core/                    # (minimum install)
nexus-marine/                  # (standard install)
nexus-fleet/                   # Fleet orchestration
├── git-agent-bridge/          # NEXUS ↔ git-agent bridge
├── fleet-config/              # Vessel identities, fleet registry
└── scripts/                   # Deployment, registration, reporting
```

### 5.4 Plugin Ecosystem

| Module | Description | Depends On |
|---|---|---|
| `nexus-learning/` | Pattern discovery, A/B testing, observation storage | nexus-marine |
| `nexus-vision/` | YOLOv8/EfficientNet species classification | nexus-marine |
| `nexus-voice/` | Whisper-tiny captain voice interface | nexus-marine |
| `nexus-nav/` | Dead reckoning, GPS fusion, waypoint navigation | nexus-marine |
| `nexus-consensus/` | Tripartite-rs safety-critical decision | nexus-marine |
| `nexus-crdt/` | CRDT-based offline fleet state sync | nexus-fleet |
| `nexus-skill-loader/` | I-know-kung-fu skill cartridge system | nexus-fleet |

---

## 6. Git-Native Workflows

### 6.1 What Git Already Gives Us (For Free)

| Workflow | Git Mechanism | NEXUS Application |
|---|---|---|
| **Audit Trail** | `git log` | Every bytecode deployment is a commit. Full history. |
| **A/B Testing** | Branches | Competing reflex behaviors on separate branches. Metrics decide winner. |
| **Rollback** | `git revert` / tags | Bad firmware? Revert to tagged release. |
| **Code Review** | Pull Requests | Every bytecode change reviewed before deployment. |
| **Collaborative Building** | Forks + PRs | Multiple agents/humans work on same vessel. |
| **Release Management** | Tags + Releases | Firmware versions are git tags. |
| **Experiment Tracking** | Branches + commits | Each experiment is a branch. Results committed. |
| **Emergency Response** | Branches + Issues | Emergency protocol creates Issue with RED label → fleet reacts. |

### 6.2 Branch Strategy

```
main (production)
├── deploy/production          # Tagged releases deployed to vessels
├── deploy/staging             # Pre-release testing
├── reflex/navigation-v2       # A/B test: new navigation algorithm
├── reflex/fishing-pattern-alpha # A/B test: new fishing behavior
├── fleet/vessel-alpha-config  # Per-vessel configuration
├── experiment/depth-profiling # Research experiment
└── safety/incident-2026-04-05 # Safety incident investigation
```

---

## 7. Trust Architecture (Unified Model)

The INCREMENTS trust engine spans all three layers with identical math:

```
EDGE (ESP32)         GIT-AGENT (Cloud)     FLEET (Orchestration)
─────────────        ─────────────────     ─────────────────────
τ_gain=500           τ_gain=500            τ_gain=500
τ_loss=20            τ_loss=20             τ_loss=20
25:1 ratio           25:1 ratio            25:1 ratio
L0→L5 levels         L0→L5 levels          L0→L5 levels

Trust earned on the edge propagates UP through git-agent to fleet.
Trust lost at any layer propagates DOWN to reduce autonomy.
Fleet propagation: 0.85x attenuation, 3-hop max radius.
```

---

## 8. Safety Architecture (Non-Negotiable)

Four tiers remain absolute regardless of git-agent integration:

```
Tier 1: HARDWARE INTERLOCK — Physical kill switch, cuts actuator power
Tier 2: FIRMWARE ISR       — E-Stop interrupt, highest priority, bypasses VM
Tier 3: SUPERVISORY TASK   — FreeRTOS safety task, monitors VM/sensors/trust
Tier 4: APPLICATION CONTROL — VM bytecode, operates WITHIN Tier 1-3 constraints
```

**Git-agent does NOT touch Tiers 1-2.** Fleet orchestration can only influence Tier 4 (what bytecode is deployed) and Tier 3 (trust threshold parameters).

---

## 9. Repository Structure (Target State)

```
SuperInstance/
├── nexus-runtime/              # THIS REPO — core runtime
│   ├── firmware/               # ESP32 C firmware (VM, Wire, Safety, HAL)
│   ├── jetson/                 # Python SDK (compiler, trust, AAB)
│   ├── shared/                 # Cross-platform definitions
│   ├── tests/                  # Unity C + pytest
│   ├── schemas/                # JSON config schemas
│   ├── docs/planning/          # Schemas, roadmaps, builder kits
│   └── CMakeLists.txt          # Monorepo build
│
├── nexus-fleet/                # NEW — git-agent bridge + fleet config
│   ├── git-agent-bridge/       # NEXUS ↔ git-agent TypeScript bridge
│   ├── fleet-config/           # Vessel identities, fleet registry
│   └── scripts/                # Deployment, registration, reporting
│
├── nexus-knowledge/            # NEW — extracted from edge-ref
│   ├── knowledge-base/         # 27 articles (~334K words)
│   ├── specs/                  # Production specifications (~19K lines)
│   ├── schemas/                # JSON schemas
│   └── addenda/                # Engineering checklists, pitfalls
│
└── nexus-onboarding/           # NEW — agent onboarding suite
    ├── context-map.md
    ├── methodology.md
    └── builder-education.md
```

---

## 10. Key Interfaces

### 10.1 New Interfaces (For git-agent Integration)

- `NexusModuleManifest` — module name, version, dependencies, trust requirements
- `ReflexPackage` — bundled bytecode + source IR + test vectors + provenance
- `TrustAttestation` — cryptographic proof of trust state at deployment time
- `NexusAgentIdentity` — agent name, vessel, role, capabilities, equipment

### 10.2 Existing Interfaces (Preserved from edge-ref)

All 12 core interfaces verified as implemented in nexus-runtime:
`instruction_t`, 32 opcodes, `vm_state_t`, COBS, CRC-16, message header,
28+ message types, 75 error codes, safety state machine, INCREMENTS trust,
`reflex_definition.json`, `node_role_config.json`, `autonomy_state.json`

---

## 11. Open Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Repo topology | Monorepo vs polyrepo vs hybrid | Monorepo for core+marine, separate repos for fleet+knowledge |
| D2 | git-agent deployment | Cloudflare vs Jetson-local vs hybrid | Hybrid: cloud orchestration, Jetson edge heartbeat |
| D3 | Trust persistence | Git-backed vs KV vs SQLite | Git for history, SQLite for fast lookup |
| D4 | Fleet communication | MQTT vs GitHub API vs hybrid | GitHub for coordination, MQTT for real-time telemetry |
| D5 | OTA signing | Ed25519 vs ECDSA-P256 | Ed25519 (simpler, git-compatible) |
| D6 | Edge LLM | Local llama.cpp vs cloud vs hybrid | Hybrid: cloud reasoning, local reflex generation |
| D7 | CRDT library | Yjs vs Automerge vs crdts-rs | crdts-rs (Rust, ESP32-compatible) |
| D8 | Constraint Theory | ESP32 vs Jetson vs cloud | Jetson-only initially |
