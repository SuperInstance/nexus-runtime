# NEXUS Phase 2 Research Findings

> **Date**: 2026-04-05 | **Researchers**: 4 parallel agents
> **Sources**: Edge-Native repo (330 files, ~1.1M words), git-agent repo (80+ files), Lucineer profile (99 repos), nexus-runtime audit (89 files, ~11K LOC)

---

## 1. Edge-Native Repository Audit

### Scale
- **330 files** total, **~1.1 million words** across ~167 text documents
- **21 production specification files** (~19,200 lines)
- **28 Architecture Decision Records** (6 pending)
- **27 knowledge base articles** (~334K words)
- **30+ dissertation files** (~132K words)
- **6 JSON schemas** (autonomy, reflex, node role, serial protocol)
- **4 CI/CD workflows** (ESP32 + Jetson)
- **Firmware**: 9 implemented C files, 10 stubs
- **Jetson**: 2 implemented Python packages, 8 stubs

### What's Already Implemented in nexus-runtime (Verified)
All of the following from edge-ref specs are FULLY IMPLEMENTED and tested in nexus-runtime:
- Bytecode VM (32 opcodes, 8-byte instructions, ~1,300 LOC C)
- Wire Protocol (COBS, CRC-16, 28+ message types, frame parser)
- Safety Subsystem (4-tier: hardware ISR, watchdog, heartbeat, safety SM)
- HAL/Drivers (sensor bus, actuator control, I/O polling)
- Reflex Compiler (3-stage JSON→IR→bytecode pipeline, Python)
- Trust Engine (INCREMENTS algorithm, 6 levels, Python)
- AAB Codec + 29 A2A opcodes (Python)
- Cross-platform opcode definitions (C + Python shared)

### Critical Gaps Identified

1. **Agent Cross-Validation** — Self-validation misses 29.4% of safety issues. Cross-validation catches 95.1%. No implementation exists.
2. **Safety Validation Pipeline** — 6-stage pipeline specified but not implemented (syntax → safety → stack → trust → semantic → adversarial).
3. **Variable Namespace Isolation** — 73% collision rate in multi-reflex deployment. No per-reflex namespace isolation specified.
4. **Certification Paradox** — How to certify self-modifying bytecode against IEC 61508. No PCCP outlined.
5. **Wire Protocol Dispatch** — edge-ref has stubs; nexus-runtime has full implementation.
6. **Driver Implementations** — All 3 edge-ref driver files are stubs; nexus-runtime has implementations.
7. **Learning Pipeline** — Only 51 LOC observation recorder. No pattern discovery or training.
8. **LLM Integration** — No llama.cpp + system prompt + reflex generation pipeline.
9. **A2A Rosetta Stone** — Only VM spec has A2A-native twin. 5 more needed.
10. **Fleet Protocol** — TELL/ASK/DELEGATE opcodes specified but not implemented.

### Cross-Cutting Themes
1. **"The Ribosome, Not the Brain"** — Intelligence at periphery operates independently of center
2. **Specs Are Source of Truth** — If code disagrees with specs, specs win
3. **Four-Tier Safety Is Non-Negotiable** — Hardware → ISR → Supervisor → Application
4. **Trust Is Mathematical** — INCREMENTS with 12 parameters, 3-branch delta, 25:1 loss-to-gain
5. **Agent-First Programming** — Three Pillars: System Prompt=Compiler, Equipment=Runtime, Vessel=Hardware
6. **Zero-Heap Deterministic** — VM fits in 3KB, uses ~5.4KB state, all static allocation
7. **8-Domain Generalization** — 80% shared architecture, 20% domain-specific
8. **Claude Code → git-agent** — Onboarding designed for Claude Code; needs evolution for git-native paradigm

### Key Schemas to Preserve
- `instruction_t` (8-byte packed) — verified implemented
- 32 opcode definitions (C + Python) — verified implemented
- `vm_state_t` (5.4KB) — verified implemented
- COBS/CRC-16 — verified implemented
- 10-byte message header — verified implemented
- 75 error codes — verified implemented
- 4 JSON schemas (autonomy_state, reflex_definition, node_role_config, serial_protocol) — need extraction
- INCREMENTS trust formula (12 params) — verified implemented
- Safety state machine (4 states) — verified implemented

---

## 2. git-agent Analysis

### Overview
- **TypeScript**, runs on **Cloudflare Workers**, 237 commits
- **Core paradigm**: "The repo IS the agent. Git IS the nervous system."
- **Agent**: Flux (Captain Riker persona), uses DeepSeek (workhorse) + Kimi K2.5 (strategist)
- **Heartbeat cycle**: PERCEIVE → THINK → ACT → REMEMBER → NOTIFY (~5 min intervals)

### Architecture
```
.agent/identity  — WHO AM I (persona, role, capabilities)
.agent/next      — WHAT DO I DO NEXT (task queue, top line = current)
.agent/done      — WHAT HAVE I DONE (72 completed tasks with timestamps)
git commits      — MEMORY / THOUGHTS
git branches     — PARALLEL REASONING
git issues       — QUESTIONS / TASK ASSIGNMENT
git PRs          — PROPOSALS / COLLABORATION
git tags         — MILESTONES / VERIFIED KNOWLEDGE
git forks        — REPRODUCTION / ISOLATED EXPERIMENT
```

### Library Modules (25 files in lib/)
- ✅ Implemented: trust-engine, forgiveness-trust, merkle-trust, discovery, equipment, admiral-interface, vessel-status, council, comm-link, parallel-arena, task-prioritizer, away-mission, mission-log, dead-reckoning-bridge, formation-manager, probation-manager
- ⚠️ Scaffolded: copilot-bridge, swarm-coordinator, git-cognitive-map, dead-reckoning, debt-tracker, dream-engine, emergency-protocol
- ❌ Stub: pulse (types only)

### Critical Finding
**`src/worker.ts` does NOT import from `lib/`**. The 25 library modules are standalone files not yet wired into the monolithic 572-line heartbeat. This is the #1 engineering task for git-agent.

### NEXUS Integration Synergies
- **Trust Engine**: DIRECT MAP — same INCREMENTS math (25:1 ratio, forgiveness function)
- **HAL → Equipment**: DIRECT MAP — sensors/actuators as EquipmentManifest entries
- **Dead Reckoning**: EXTRAORDINARY — marine navigation concept maps to git-agent's commit-state dead reckoning
- **Wire Protocol → Comm Link**: Wire Protocol for inter-device, Comm Link for inter-agent
- **AAB → Heartbeat**: AAB programs become tasks in .agent/next

### Gaps for NEXUS Integration
1. No edge/hardware abstraction (pure GitHub API space)
2. No real-time communication (seconds latency via git API)
3. Trust state not persisted (in-memory only)
4. No package.json or build system
5. Worker is monolithic (needs lib/ integration)

---

## 3. Lucineer Ecosystem Analysis

### Scale: 99 Repositories

#### Tier 1 — Direct NEXUS Integration (5★ synergy)
| Repo | Stars | Description |
|---|---|---|
| **edgenative-ai** | 0 | NEXUS Knowledge Vessel — VM emulator, trust calc, Rosetta Stone, 1.2M words. Already IS a NEXUS vessel. |
| **increments-fleet-trust** | 0 | INCREMENTS adapted from NEXUS edge robotics to software fleet. Same math. L0-L5. |
| **fleet-orchestrator** | 0 | Captain's Bridge — HCQ quarantine, DEB execution bonds, Council of Captains, trust attestation. |
| **fishinglog-ai** | 1 | Edge AI fishing vessel on Jetson Orin Nano — species classification, captain voice, catch reporting. Real deployment. |

#### Tier 2 — Fleet Orchestration (4★ synergy)
| Repo | Description |
|---|---|
| **cocapn** | Core ecosystem hub — repo-first agent, BYOK multi-LLM, fleet protocol, ~80 lines core |
| **cocapn-lite** | Minimal cocapn seed (~200 lines) for power users |
| **capitaine** | Full git-native vessel — Captain/Helm mode, Dead Reckoning, Iron Sharpens Iron |
| **zeroclaw** | Minimum viable repo-native agent — skills + equipment + soul |
| **baton-ai** | Universal handoff protocol — HMAC-signed context transfer |
| **dead-reckoning-engine** | 6-folder knowledge pipeline: compass→dead→working→ground→published |
| **personality-engine** | DNA-based personality profiles with contextual adaptation |
| **model-quality-rubric** | 7-dimension model scoring, 25+ providers |
| **LOG-mcp** | Multi-model gateway — builds preference dataset for local inference |
| **mycelium-ai** | Behavior capture as seeds — one prompt + one seed = exact action |

#### Tier 3 — Safety & Consensus (4★ synergy)
| Repo | Description |
|---|---|
| **tripartite-rs** | Rust consensus (Pathos/Logos/Ethos), privacy-first, 298 tests |
| **open-fleet-safety** | Top 5 fleet failure modes + defenses |
| **forgetting-problem** | Three types of forgetting: thermal decay, scheduled pruning, emergency purging |
| **crdt-sync** | CRDT state sync for agent fleets (concept phase, needs implementation) |

#### Tier 4 — Support Infrastructure
- **Constraint-Theory** (Rust + JS) — Geometric computation, KD-tree, O(log n) spatial queries
- **I-know-kung-fu** — 9 JSON skill cartridges, 6 platform templates, 18 profiles
- **dream-engine** — Background consolidation / REM sleep for fleet
- **actualizer-ai** — Reverse-actualization across 7 time horizons (1yr-100yr)
- **40+ log-ai domain agents** — fishinglog, makerlog, studylog, dmlog, etc.
- **10 Minecraft/game AI repos** — craftmind ecosystem

### Key Insight: The Fleet Is Operational
35+ vessels deployed on Cloudflare Workers with live health checks, trust scoring, and inter-vessel coordination. The ecosystem is not theoretical — it's running.

### 6 Empty Edge-Native Concept Repos
These are DESIGN INTENT waiting for implementation:
- edge-boarding-protocol
- edge-equipment-catalog
- gravity-well-protocol
- nexus-fracture-sim
- resonant-consensus
- forgiveness-function

---

## 4. nexus-runtime Audit

### Git State
- **Branch**: main (only branch)
- **Commits**: 7 (linear history)
- **Latest**: `26f7806` — architecture plan
- **Remote**: origin/main (GitHub)

### Code Metrics
| Component | Files | LOC |
|---|---|---|
| Firmware C/H (source) | 24 | ~3,100 |
| Shared C/H | 2 | ~210 |
| Firmware tests (Unity) | 8 | ~2,540 |
| Jetson Python (source) | 17 | ~2,200 |
| Python tests | 7 | ~1,790 |
| HIL tests (stubs) | 3 | ~100 |
| **Total** | **89** | **~11,015** |

### Module Completeness
| Module | Status | LOC |
|---|---|---|
| Bytecode VM | ✅ Complete | ~1,332 |
| Wire Protocol | ✅ Complete | ~1,485 (C+Py) |
| Safety | ✅ Complete | ~441 |
| HAL/Drivers | ✅ Complete | ~796 |
| Reflex Compiler | ✅ Complete | ~759 |
| Trust Engine | ✅ Complete | ~568 |
| Agent Runtime | ✅ Complete | ~601 |
| Learning | ⚠️ Minimal | ~52 |
| SDK Pipeline | ⚠️ Minimal | ~41 |
| HIL Tests | ❌ Stubs | ~100 |

### Test Status
- **Jetson (pytest)**: 150/150 passed in 0.28s ✅
- **Firmware (Unity)**: Tests exist (~2,540 LOC), binaries pre-built, cannot re-run without ESP-IDF toolchain
- **HIL**: 3 skeleton files only

---

## 5. Synthesis: Strategic Recommendations

### The Integration Architecture (3 Layers)

1. **EDGE LAYER** (nexus-runtime): ESP32 firmware + Jetson SDK. Already built. Handles real-time control, sensor I/O, bytecode execution, safety.

2. **GIT-AGENT LAYER** (git-agent per vessel): Orchestration intelligence. Each vessel runs a git-agent instance that coordinates through git operations. This is NEW and must be built.

3. **FLEET LAYER** (cocapn ecosystem): Fleet-wide management. Already exists across 99 repos. Must be forked and connected.

### Critical Path

```
Phase 2A (Foundation)     → Repo restructuring, schemas, CI
Phase 2B (Bridge)        → git-agent bridge, edge heartbeat, fleet registration
Phase 2C (Intelligence)  → Dead reckoning, trust unification, Rosetta Stone
Phase 2D (Safety)        → Tripartite consensus, safety validation, emergency protocol
Phase 2E (Fleet)         → CRDT sync, skill loading, marine vessel template
Phase 2F (Knowledge)     → Extract knowledge base, onboarding, ADRs, release
```

### The One Thing

If there's one thing to get right, it's the **git-agent bridge** (Sprint 2B.1). This is the translation layer between the physical world (NEXUS edge) and the coordination world (git-agent fleet). Get this right, and everything else plugs in naturally. Get it wrong, and the two layers remain islands.

### Decision: Hybrid Deployment

git-agent should run in BOTH places:
- **Cloud (Cloudflare Workers)**: For fleet-wide coordination, strategy, GitHub API operations
- **Edge (Jetson Python)**: For local heartbeat, sensor data ingestion, bytecode deployment

The edge heartbeat syncs with the cloud heartbeat via MQTT. When offline, the edge heartbeat operates autonomously using its local git repo.

### Decision: Monorepo Core, Polyrepo Ecosystem

- `nexus-runtime`: Monorepo (core + marine). Single build, single CI, single release.
- `nexus-fleet`: Separate repo (bridge + configs). Different language (TypeScript), different deploy target (Cloudflare).
- `nexus-knowledge`: Separate repo (334K words is too big for a code repo). Git submodule or separate clone.
- `nexus-onboarding`: Separate repo (educational content, not code).

Son's repos: Fork into SuperInstance org. Don't modify upstream. Pull updates via merge.
