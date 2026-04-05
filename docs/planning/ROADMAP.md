# NEXUS Phase 2 Master Roadmap

> **Version**: 2.0 | **Date**: 2026-04-05 | **Status**: DRAFT

---

## Current State Summary

| Metric | Value |
|---|---|
| **Sprints completed** | 5 (0.1 → 0.5) + 1 bugfix + 1 architecture plan |
| **Tests passing** | 252/252 (102 C/Unity + 150 pytest) |
| **Total LOC** | ~11,015 (firmware: ~3,300, jetson: ~2,200, tests: ~4,430) |
| **Branches** | 1 (main) |
| **Commits** | 7 |
| **Edge-Native docs** | ~1.1M words across 330 files at `/tmp/edge-ref` |
| **Son's ecosystem** | 99 repos at github.com/lucineer, 5★ NEXUS synergy |
| **git-agent** | Cloned to `/tmp/git-agent-ref`, TypeScript, 237 commits |

---

## Phase 2 Overview: From Monolith to Fleet

### Strategic Goals

1. **Modularize** — Split nexus-runtime into composable modules (core/marine/fleet)
2. **Git-Native** — Every process leverages git for audit trails, rollback, A/B testing
3. **git-Agent Bridge** — Connect NEXUS edge to git-agent orchestration layer
4. **Fleet Integration** — Wire into cocapn ecosystem (fleet-orchestrator, increments-fleet-trust, edgenative-ai)
5. **Knowledge Migration** — Extract and refracture ~1.1M words from edge-ref into structured repos

### Phase Structure

```
Phase 2A: Foundation (Weeks 1-2)   — Repo restructuring, module boundaries, CI modernization
Phase 2B: Bridge (Weeks 3-4)       — git-agent bridge, equipment manifests, edge heartbeat
Phase 2C: Intelligence (Weeks 5-6) — Dead reckoning, trust unification, Rosetta Stone
Phase 2D: Safety & Consensus (Weeks 7-8) — Tripartite consensus, safety validation as PR checks
Phase 2E: Fleet Maturity (Weeks 9-10) — CRDT sync, skill loading, full fleet deployment
Phase 2F: Knowledge (Weeks 11-12)  — Extract knowledge base, specs, schemas from edge-ref
```

---

## Phase 2A: Foundation (Weeks 1-2)

### Sprint 2A.1: Repository Restructuring

**Goal**: Reorganize nexus-runtime into modular directory structure without breaking any tests.

```
BEFORE:                          AFTER:
nexus-runtime/                    nexus-runtime/
├── firmware/src/                 ├── firmware/src/
│   ├── nexus_vm/                 │   ├── core/            # VM + wire protocol
│   ├── wire_protocol/            │   ├── safety/          # 4-tier safety
│   ├── safety/                   │   ├── drivers/         # HAL + I/O
│   └── drivers/                  │   └── main/            # app_main + FreeRTOS
├── jetson/                       ├── jetson/
│   ├── nexus_vm/                 │   ├── core/            # compiler + validator
│   ├── reflex_compiler/          │   ├── reflex/          # JSON→bytecode pipeline
│   ├── trust_engine/             │   ├── trust/           # INCREMENTS engine
│   ├── agent_runtime/            │   ├── agent/           # AAB codec + A2A
│   ├── learning/                 │   ├── learning/        # observation + patterns
│   └── nexus_sdk/                │   └── sdk/             # pipeline orchestrator
├── shared/                       ├── shared/              # unchanged
├── tests/                        ├── tests/              # reorganized by module
└── CMakeLists.txt                ├── schemas/            # moved from edge-ref
                                  └── CMakeLists.txt      # updated targets
```

**Tasks**:
- [ ] Create new directory structure
- [ ] Move files preserving git history (`git mv`)
- [ ] Update all `#include` paths in C files
- [ ] Update all `import` paths in Python files
- [ ] Update CMakeLists.txt targets
- [ ] Run full test suite (252/252 must remain green)
- [ ] Commit: "Sprint 2A.1: Modular directory restructuring"

**Builder Notes**: Use `git mv` for every file move to preserve blame/history. Run `grep -r '#include' firmware/` before and after to catch broken includes. Run pytest after every batch of Python import changes.

### Sprint 2A.2: Schema Extraction from edge-ref

**Goal**: Migrate the 4 critical JSON schemas from edge-ref into nexus-runtime.

```
Source: /tmp/edge-ref/schemas/
  ├── autonomy_state.json       → nexus-runtime/schemas/autonomy_state.json
  ├── reflex_definition.json    → nexus-runtime/schemas/reflex_definition.json
  ├── node_role_config.json     → nexus-runtime/schemas/node_role_config.json
  └── serial_protocol.json      → nexus-runtime/schemas/serial_protocol.json
```

**Tasks**:
- [ ] Copy schemas to nexus-runtime/schemas/
- [ ] Write Python validation code for each schema (jsonschema library)
- [ ] Write tests that validate existing config files against schemas
- [ ] Write C header equivalents for struct layouts
- [ ] Run tests
- [ ] Commit: "Sprint 2A.2: Schema extraction and validation"

### Sprint 2A.3: CI Modernization

**Goal**: Add git-native CI checks that run on every PR.

**New CI workflows**:
- `pr-checks.yml`: Runs on PRs to main
  - Firmware tests (Unity)
  - Python tests (pytest)
  - Schema validation (all configs validated against JSON schemas)
  - Bytecode safety check (new: validate bytecode doesn't violate safety rules)
  - ADR review (check if any new ADRs need to be added)
- `deploy-staging.yml`: On merge to main
  - Build firmware binary
  - Run integration tests
  - Create draft release
- `fleet-health.yml`: Scheduled (hourly)
  - Ping fleet-orchestrator for vessel status
  - Report trust scores

**Tasks**:
- [ ] Write pr-checks.yml workflow
- [ ] Write deploy-staging.yml workflow
- [ ] Add bytecode safety validator to CI
- [ ] Test CI on a feature branch PR
- [ ] Merge to main
- [ ] Commit: "Sprint 2A.3: Git-native CI with PR checks"

### Sprint 2A.4: Knowledge Migration Part 1 — Specs

**Goal**: Extract production specifications from edge-ref into a structured format.

```
Source: /tmp/edge-ref/specs/ → nexus-runtime/docs/specs/
  ├── 00_MASTER_INDEX.md
  ├── ARCHITECTURE_DECISION_RECORDS.md
  ├── NEXUS_Phase1_Foundation_Complete_Specs.md
  └── SENIOR_ENGINEER_BUILD_GUIDE.md
```

**Tasks**:
- [ ] Copy spec files
- [ ] Reconcile specs against what's implemented in nexus-runtime
- [ ] Mark each spec section as ✅ IMPLEMENTED / ⚠️ PARTIAL / ❌ TODO
- [ ] Create `docs/specs/IMPLEMENTATION_STATUS.md` tracking document
- [ ] Commit: "Sprint 2A.4: Spec migration with implementation status"

**Deliverable for Phase 2A**: Modular repo with green tests, validated schemas, CI on PRs, migrated specs.

---

## Phase 2B: Bridge (Weeks 3-4)

### Sprint 2B.1: git-agent Bridge Module

**Goal**: Create the TypeScript bridge that connects NEXUS Wire Protocol to git-agent operations.

```
nexus-fleet/
├── git-agent-bridge/
│   ├── src/
│   │   ├── nexus_bridge.ts      # Main bridge: Wire Protocol ↔ GitHub API
│   │   ├── equipment_manifest.ts # HAL capabilities → EquipmentManifest
│   │   ├── bytecode_deployer.ts  # PR merge → bytecode deployment
│   │   ├── telemetry_ingester.ts # Sensor data → git commits
│   │   └── trust_sync.ts         # INCREMENTS events ↔ fleet trust
│   ├── tests/
│   │   ├── test_bridge.ts
│   │   ├── test_deployer.ts
│   │   └── test_trust_sync.ts
│   ├── package.json
│   └── tsconfig.json
```

**Key bridge translations**:

| NEXUS Event | git-agent Action |
|---|---|
| Bytecode compiled | Create file in repo + commit |
| Bytecode ready for deploy | Create PR with safety check results |
| PR merged | Deploy bytecode via Wire Protocol to ESP32 |
| Sensor reading received | Commit telemetry data with structured metadata |
| Trust event (GOOD/BAD) | Create trust attestation commit |
| Safety violation | Create Issue with RED label + fleet alert |
| Mission complete | Close Issue with results summary |

**Tasks**:
- [ ] Initialize TypeScript project with dependencies
- [ ] Implement nexus_bridge.ts (Wire Protocol message parsing + GitHub API)
- [ ] Implement equipment_manifest.ts (sensor/actuator discovery)
- [ ] Implement bytecode_deployer.ts (PR watcher → serial deploy)
- [ ] Implement telemetry_ingester.ts (Wire Protocol → git commit)
- [ ] Implement trust_sync.ts (INCREMENTS ↔ fleet trust API)
- [ ] Write tests for all modules
- [ ] Create sample vessel configuration
- [ ] Commit: "Sprint 2B.1: git-agent bridge module"

### Sprint 2B.2: Edge Heartbeat on Jetson

**Goal**: A lightweight Python heartbeat on Jetson that syncs with cloud git-agent.

```
nexus-fleet/
├── edge_heartbeat/
│   ├── heartbeat.py          # Main heartbeat loop (reads .agent/next, acts, commits)
│   ├── nexus_adapter.py      # Connects to NEXUS Wire Protocol (serial)
│   ├── github_client.py      # GitHub API for .agent/ files, Issues, PRs
│   ├── mqtt_bridge.py        # Bidirectional MQTT ↔ GitHub bridge
│   └── config.py             # Vessel-specific configuration
```

**Tasks**:
- [ ] Implement heartbeat.py (5-minute cycle: perceive → think → act → remember → notify)
- [ ] Implement nexus_adapter.py (connect to NEXUS runtime on localhost)
- [ ] Implement github_client.py (read/write .agent/ files, create issues/PRs)
- [ ] Implement mqtt_bridge.py (MQTT for real-time, GitHub for persistent)
- [ ] Write tests
- [ ] Test on Jetson hardware (or simulate)
- [ ] Commit: "Sprint 2B.2: Edge heartbeat for Jetson"

### Sprint 2B.3: Fleet Registration

**Goal**: Register NEXUS vessels with fleet-orchestrator and increments-fleet-trust.

**Tasks**:
- [ ] Fork fleet-orchestrator repo
- [ ] Add NEXUS vessel type to fleet registry
- [ ] Configure HCQ (Hierarchical Circuit Quarantine) parameters for edge devices
- [ ] Configure DEB (Deterministic Execution Bonds) for task distribution
- [ ] Fork increments-fleet-trust
- [ ] Create NEXUS event adapter (Wire Protocol events → INCREMENTS format)
- [ ] Test: register vessel, receive task, report trust event
- [ ] Commit: "Sprint 2B.3: Fleet registration and trust integration"

**Deliverable for Phase 2B**: Working bridge between NEXUS edge and git-agent cloud. Edge heartbeat running. Fleet registration complete.

---

## Phase 2C: Intelligence (Weeks 5-6)

### Sprint 2C.1: Dead Reckoning Integration

**Goal**: Wire the dead-reckoning-engine pipeline into NEXUS sensor data flow.

```
Sensor Data → compass-bearing → dead-reckoning → working-theory → ground-truth → published
```

**Tasks**:
- [ ] Fork dead-reckoning-engine repo
- [ ] Create NEXUS compass-bearing input adapter (sensor telemetry format)
- [ ] Configure storyboarder models for marine domain
- [ ] Configure inbetweener for efficient processing
- [ ] Implement knowledge persistence (ground-truth → git branch)
- [ ] Test with simulated sensor data
- [ ] Commit: "Sprint 2C.1: Dead reckoning pipeline for marine data"

### Sprint 2C.2: Trust Unification

**Goal**: Ensure the same INCREMENTS math runs on ESP32 (C), Jetson (Python), and fleet (TypeScript).

**Tasks**:
- [ ] Write cross-language test vectors (same inputs → same outputs in C, Python, TypeScript)
- [ ] Port any missing trust features between implementations
- [ ] Implement trust propagation (edge → git-agent → fleet)
- [ ] Implement trust attenuation (fleet → git-agent → edge, 0.85x per hop)
- [ ] Add trust attestation signing (HMAC or Ed25519)
- [ ] Test: generate trust event on ESP32, verify it arrives at fleet with correct attenuation
- [ ] Commit: "Sprint 2C.2: Unified trust across edge, agent, and fleet"

### Sprint 2C.3: Rosetta Stone Completion

**Goal**: Complete the A2A Rosetta Stone specs that translate human intent to bytecode.

**Tasks**:
- [ ] Read existing Rosetta Stone spec from edge-ref (`a2a-native-specs/`)
- [ ] Implement Rosetta Stone translator in Python (intent → bytecode)
- [ ] Add Rosetta Stone API endpoint to edgenative-ai vessel
- [ ] Create test vectors: human instruction → bytecode → VM execution → expected behavior
- [ ] Document the 4-layer translation pipeline
- [ ] Commit: "Sprint 2C.3: Rosetta Stone intent-to-bytecode translation"

**Deliverable for Phase 2C**: Sensor data flows through dead-reckoning pipeline. Trust unified across all three layers. Rosetta Stone translates human intent to bytecode.

---

## Phase 2D: Safety & Consensus (Weeks 7-8)

### Sprint 2D.1: Tripartite Consensus on Edge

**Goal**: Deploy tripartite-rs (Pathos/Logos/Ethos) on ESP32 and Jetson.

**Tasks**:
- [ ] Clone tripartite-rs (298 tests, Rust)
- [ ] Add NEXUS-specific agents:
  - Pathos: Maritime intent interpretation
  - Logos: Navigation/operation planning
  - Ethos: Safety verification against marine regulations
- [ ] Cross-compile for ESP32 (thumbv6m-none-eabi target)
- [ ] Compile for Jetson (aarch64-unknown-linux-gnu)
- [ ] Create C FFI bindings for ESP32 firmware integration
- [ ] Write integration tests
- [ ] Commit: "Sprint 2D.1: Tripartite consensus for safety-critical decisions"

### Sprint 2D.2: Safety Validation as PR Checks

**Goal**: Every PR that contains bytecode must pass automated safety validation.

**Tasks**:
- [ ] Implement bytecode safety validator (6-stage pipeline from edge-ref specs):
  1. Syntax check (well-formed instructions)
  2. Safety rules (no forbidden opcode sequences)
  3. Stack analysis (no underflow/overflow)
  4. Trust check (opcode privileges match trust level)
  5. Semantic analysis (no infinite loops, no I/O on protected pins)
  6. Adversarial probing (fuzz boundary conditions)
- [ ] Add to CI as PR check
- [ ] Create GitHub Action that comments safety report on PR
- [ ] Write tests for each validation stage
- [ ] Commit: "Sprint 2D.2: Automated safety validation on PRs"

### Sprint 2D.3: Emergency Protocol Bridge

**Goal**: Connect NEXUS Tier 2/3 safety events to git-agent emergency protocol.

**Tasks**:
- [ ] Implement safety event → GitHub Issue creation
- [ ] Implement fleet-wide alert propagation
- [ ] Implement automatic trust degradation on safety events
- [ ] Implement recovery workflow (fix PR → safety revalidation → redeploy)
- [ ] Test: trigger E-Stop → verify Issue created → verify fleet notified
- [ ] Commit: "Sprint 2D.3: Emergency protocol bridge"

**Deliverable for Phase 2D**: Tripartite consensus running on edge. Safety validation automated on every PR. Emergency events propagated fleet-wide.

---

## Phase 2E: Fleet Maturity (Weeks 9-10)

### Sprint 2E.1: CRDT State Sync

**Goal**: Implement conflict-free state synchronization for offline marine fleet.

**Tasks**:
- [ ] Implement CRDT sync layer using crdts-rs (or Automerge if Rust proves too complex for ESP32)
- [ ] Define shared state types (vessel position, trust scores, task assignments)
- [ ] Implement sync over NMEA 2000 / WiFi / satellite
- [ ] Test: disconnect two vessels, modify state independently, reconnect, verify convergence
- [ ] Commit: "Sprint 2E.1: CRDT-based offline fleet sync"

### Sprint 2E.2: Skill Loading System

**Goal**: Load marine operation skills from I-know-kung-fu cartridges.

**Tasks**:
- [ ] Define NEXUS skill cartridge format (JSON)
- [ ] Create initial marine skill cartridges:
  - navigation (waypoint following, GPS fusion)
  - fishing (pattern generation, species detection triggers)
  - emergency (man-overboard, fire, collision avoidance)
  - docking (approach, mooring, departure)
- [ ] Implement skill loader on Jetson (reads cartridges, generates AAB bytecode)
- [ ] Test: load navigation skill → generate bytecode → deploy to ESP32 → verify behavior
- [ ] Commit: "Sprint 2E.2: Marine skill cartridge system"

### Sprint 2E.3: Fishinglog Template

**Goal**: Fork fishinglog-ai as the NEXUS marine vessel template.

**Tasks**:
- [ ] Fork fishinglog-ai to nexus-marinelog
- [ ] Replace fishinglog-specific features with generic NEXUS vessel features
- [ ] Add NEXUS fleet integration (trust, health, coordination)
- [ ] Add navigation overlay dashboard
- [ ] Add fleet status dashboard
- [ ] Test: deploy as NEXUS vessel template
- [ ] Commit: "Sprint 2E.3: Marine vessel template from fishinglog"

**Deliverable for Phase 2E**: Fleet syncs offline. Skills load dynamically. Marine vessel template ready.

---

## Phase 2F: Knowledge (Weeks 11-12)

### Sprint 2F.1: Knowledge Base Extraction

**Goal**: Extract the 27-article knowledge base from edge-ref into nexus-knowledge repo.

**Tasks**:
- [ ] Create nexus-knowledge repo
- [ ] Extract knowledge-base/ articles (334K words)
- [ ] Reorganize into categories:
  - Technical (VM, embedded, edge AI, distributed systems)
  - Domain (marine autonomous, robotics, trust psychology)
  - Meta (formal verification, post-coding paradigms, evolution of VMs)
  - Reference (glossary with 310 terms, annotated bibliography with 178 refs)
- [ ] Write index document linking all articles
- [ ] Commit: "Sprint 2F.1: Knowledge base extraction"

### Sprint 2F.2: Onboarding Suite Extraction

**Goal**: Extract agent onboarding materials into nexus-onboarding repo.

**Tasks**:
- [ ] Create nexus-onboarding repo
- [ ] Extract onboarding/context-map.md (9,500 words)
- [ ] Extract onboarding/methodology.md (7,500 words)
- [ ] Extract onboarding/gamified-intro.md (7,900 words)
- [ ] Extract onboarding/concept-playground.md (6,800 words)
- [ ] Update for git-native paradigm (replace Claude Code references with git-agent)
- [ ] Add "How to add a new reflex module" guide
- [ ] Add "How to add a new domain port" guide
- [ ] Commit: "Sprint 2F.2: Onboarding suite extraction"

### Sprint 2F.3: ADR Evolution

**Goal**: Add new ADRs for git-native paradigm decisions.

**New ADRs to create**:
- ADR-029: Git-native vs monolithic repo (modular architecture)
- ADR-030: git-agent as orchestration layer (Cloudflare Workers + edge heartbeat)
- ADR-031: Hybrid trust persistence (git commits for history + SQLite for lookup)
- ADR-032: Tripartite consensus for safety-critical edge decisions
- ADR-033: CRDT-based offline fleet synchronization
- ADR-034: Skill cartridge format (JSON-based, I-know-kung-fu compatible)
- ADR-035: Bytecode safety validation as PR pipeline (6-stage)

**Tasks**:
- [ ] Write each ADR following the existing format (context, decision, consequences)
- [ ] Each ADR gets its own file (git-native: one ADR per PR)
- [ ] Update master index
- [ ] Commit: "Sprint 2F.3: Phase 2 ADRs"

### Sprint 2F.4: Final Documentation & Release

**Goal**: Update all documentation to reflect Phase 2 state.

**Tasks**:
- [ ] Update README.md with new architecture, modular install instructions
- [ ] Update ARCHITECTURE_PLAN.md to reflect Phase 2 completion
- [ ] Update CLAUDE.md (or create GIT_AGENT.md) for git-agent onboarding
- [ ] Create CHANGELOG.md for all Phase 2 changes
- [ ] Tag release: v0.6.0
- [ ] Push all repos to GitHub
- [ ] Commit: "Sprint 2F.4: Phase 2 documentation and v0.6.0 release"

**Deliverable for Phase 2F**: Knowledge extracted, onboarding modernized, ADRs evolved, v0.6.0 released.

---

## Migration Priority Matrix from Edge-Native

| Source (edge-ref) | Target (nexus) | Sprint | Priority |
|---|---|---|---|
| `schemas/*.json` (4 files) | `nexus-runtime/schemas/` | 2A.2 | CRITICAL |
| `specs/ARCHITECTURE_DECISION_RECORDS.md` | `nexus-runtime/docs/specs/` | 2A.4 | CRITICAL |
| `claude-build/build-specification.md` | `nexus-runtime/docs/specs/build-spec.md` | 2A.4 | CRITICAL |
| `claude.md` | `nexus-runtime/docs/GIT_AGENT.md` (evolved) | 2F.4 | HIGH |
| `onboarding/*.md` (11 files) | `nexus-onboarding/` repo | 2F.2 | HIGH |
| `knowledge-base/*.md` (27 files) | `nexus-knowledge/` repo | 2F.1 | MEDIUM |
| `addenda/*.md` (7 files) | `nexus-knowledge/addenda/` | 2F.1 | HIGH |
| `incubator/manifesto.md` | `nexus-knowledge/manifesto.md` | 2F.1 | MEDIUM |
| `autopilot/configs/*.json` (6 files) | `nexus-runtime/configs/` | 2A.2 | MEDIUM |
| `vessel-platform/configs/*.json` (6 files) | `nexus-runtime/configs/` | 2A.2 | MEDIUM |
| `a2a-native-specs/` (8 files) | `nexus-runtime/docs/specs/a2a/` | 2C.3 | HIGH |
| `framework/*.txt` (7 files) | `nexus-knowledge/framework/` | 2F.1 | LOW |
| `genesis-colony/` (40 files) | Archive (historical) | — | LOW |
| `dissertation/` (30 files) | `nexus-knowledge/dissertation/` | 2F.1 | LOW |
| `v31-docs/` (13 files) | Archive (superseded) | — | LOW |

---

## Son's Ecosystem Integration Priority

| Repo | Integration Sprint | Type | Effort |
|---|---|---|---|
| `edgenative-ai` | 2B.3 (fork + connect) | Fork | 2 days |
| `increments-fleet-trust` | 2B.3 (fork + adapter) | Fork | 3 days |
| `fleet-orchestrator` | 2B.3 (fork + register) | Fork | 3 days |
| `git-agent` | 2B.1 (bridge + heartbeat) | Subtree | 5 days |
| `fishinglog-ai` | 2E.3 (fork as template) | Fork | 4 days |
| `dead-reckoning-engine` | 2C.1 (fork + configure) | Fork | 3 days |
| `tripartite-rs` | 2D.1 (cross-compile) | Submodule | 5 days |
| `baton-ai` | 2C.2 (integrate with Wire) | Fork | 2 days |
| `capitaine` | 2E.3 (template for vessels) | Fork | 3 days |
| `cocapn` | 2B.3 (deploy domain ctrl) | Fork | 2 days |
| `zeroclaw` | 2E.2 (lightweight edge agent) | Fork | 3 days |
| `crdt-sync` | 2E.1 (implement + integrate) | Fork | 5 days |
| `open-fleet-safety` | 2D.3 (adapt for marine) | Reference | 2 days |
| `forgetting-problem` | 2E.1 (implement on edge) | Reference | 3 days |
| `Constraint-Theory` | 2E.1 (Jetson spatial) | Submodule | 3 days |
| `I-know-kung-fu` | 2E.2 (skill cartridge format) | Reference | 2 days |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| git-agent lib/ not wired to worker.ts | HIGH | HIGH | Refactor worker.ts to import lib/ modules |
| CRDT performance on ESP32 | MEDIUM | HIGH | Start with Jetson-only; defer ESP32 if too slow |
| Trust score divergence between layers | MEDIUM | HIGH | Cross-language test vectors; periodic reconciliation |
| Offline fleet sync data loss | MEDIUM | CRITICAL | Append-only git log; CRDT merge on reconnect |
| OTA firmware corruption | LOW | CRITICAL | Ed25519 signing + rollback to tagged release |
| LLM costs for fleet orchestration | HIGH | MEDIUM | Cache strategist calls; use cheap models for routine tasks |
| git-agent rate limits (GitHub API) | MEDIUM | MEDIUM | Batch operations; respect rate limits; exponential backoff |
| Tripartite consensus latency | MEDIUM | MEDIUM | Timeout with safe default (maintain current behavior) |
