---
## Task ID: phase2-fix-and-build — fix-and-build-agent
### Work Task
Fix 10 critical bugs identified by research phase, then build the NexusOrchestrator integration harness.

### Work Summary

**Bug Fixes (10 bugs fixed):**

1. **C1: Jump Target Field Mismatch** (`bytecode_deployer.py`): Changed jump validation from checking operand1 to operand2, which is the actual target field in the instruction encoding.

2. **C2: .agent/next Format Contradiction** (`bridge.py`): Unified .agent/next from directory-of-JSON-files to flat text format (one mission per line), matching heartbeat's `mission_runner.py` expected format. Updated `_init_agent_dirs()` to create `.agent/next` as a text file and removed it from the directories list. Updated `get_mission_queue()` and `complete_mission()` for text-based format.

3. **C3: HALT Always Fails Safety at Trust < L5** (`pipeline.py`): Modified both `stage2_safety_rules()` and `stage4_trust_check()` in `BytecodeSafetyPipeline` to exempt HALT (SYSCALL syscall_id=0x01) from trust-level gating. HALT is a safety termination opcode that must be available at all trust levels.

4. **C4: Trust Score Type Mismatch** (`bridge.py`): Updated `get_status()` to extract float scores from `SubsystemTrust` objects by checking for `trust_score` attribute and falling back to direct float conversion.

5. **C5: Integer/Float Mix in Wait Loop** (`intent_compiler.py`): Changed `_compile_wait()` to use `PUSH_F32` with float operands instead of `PUSH_I8` with integer operands, since `SUB_F` operates on floats.

6. **C6: emergency_surface Skill Doesn't Surface** (`builtin_skills.py`): Added `em.emit_write_pin(7)` to emergency_surface bytecode to actually trigger the ascent actuator. Changed `trust_required` from 0 to 2 (WRITE_PIN requires L2). Updated version to 1.0.1.

7. **I1: Dual Safety Validators** (`bridge.py`): Bridge now delegates to the 6-stage `BytecodeSafetyPipeline` (64-stack limit) instead of its internal `BytecodeDeployer.validate_bytecode()` (16-stack limit). Falls back to deployer when pipeline unavailable.

8. **S1: Hardcoded HMAC Key** (`attestation.py`): HMAC signing key now loads from `NEXUS_ATTESTATION_KEY` environment variable with fallback to the default key.

9. **S5: Emergency Claims Fleet Notified When No Bridge** (`response.py`): Changed `fleet_notified = True` to `fleet_notified = False` in the no-bridge branch of `respond_red()`.

10. **S6: CLAMP_F Bypass** (`bytecode_deployer.py`): Enhanced CLAMP_F-before-WRITE check to detect intervening PUSH instructions (PUSH_I8, PUSH_I16, PUSH_F32) between CLAMP_F and WRITE_PIN, which would bypass the safety clamp.

**Integration Orchestrator** (already existed, verified functional):
- `NexusOrchestrator`: Central coordinator wiring all modules
- `MissionSimulator`: Software VM simulation without hardware
- `SystemStatus`/`StatusAggregator`: System status aggregation
- 62 end-to-end integration tests

**Test Updates:**
- Updated bridge tests for flat-text .agent/next format
- Updated safety pipeline tests for HALT exemption at all trust levels
- Updated rosetta tests for PUSH_F32 in wait loop
- Updated skill system tests for emergency_surface at L2 with WRITE_PIN
- Updated orchestrator tests for trust seeding thresholds

**Test Results:** 991 passed, 3 failed (pre-existing fleet_sync CRDT failures unrelated to our changes).

**Commit:** `281f707` — "Phase 2 critical fixes + integration orchestrator"
### Work Task
Implement Sprint 2D.3 + partial 2F: Emergency Protocol Bridge and Phase 2 ADRs (ADR-029 to ADR-036).

### Work Summary
Created the complete Emergency Protocol Bridge subsystem and 8 Phase 2 ADRs.

**Modules Created (4 files):**
- `jetson/agent/emergency_protocol/__init__.py` — Package exports for all public types
- `jetson/agent/emergency_protocol/protocol.py` — Core EmergencyProtocol class with 4-level state machine (GREEN/YELLOW/ORANGE/RED), Incident dataclass, EmergencyAssessment/EscalationResult/DeescalationResult result types, configurable thresholds, incident lifecycle tracking, and incident report generation
- `jetson/agent/emergency_protocol/detectors.py` — 5 specialized detectors: SensorFailureDetector (offline/stale/out-of-range/low-quality), TrustCollapseDetector (per-subsystem thresholds + multi-subsystem degradation), CommunicationLossDetector (timeout/dead), SafetyViolationDetector (E-Stop/FAULT/CRITICAL/WARNING/watchdog), MissionTimeoutDetector (overrun detection)
- `jetson/agent/emergency_protocol/response.py` — EmergencyResponder with level-specific automated responses: YELLOW (log+monitor), ORANGE (alert+reduce autonomy+notify fleet), RED (halt ops+safe actuators+GitHub Issue+git commit+trust reduction+fleet alert+watchdog), plus structured fleet alert generation

**ADRs Created (8 files):**
- ADR-029: Git-native modular architecture (monorepo core + polyrepo ecosystem)
- ADR-030: git-agent as orchestration layer (Cloudflare + Jetson hybrid heartbeat)
- ADR-031: Hybrid trust persistence (git commits for history + in-memory for speed)
- ADR-032: Tripartite consensus for safety-critical edge decisions
- ADR-033: CRDT-based offline fleet synchronization
- ADR-034: Skill cartridge format (JSON + bytecode + metadata)
- ADR-035: Bytecode safety validation as PR pipeline (6-stage)
- ADR-036: Emergency protocol bridge (NEXUS safety → git-agent fleet alert)

**Tests:** 79 tests across 16 test classes — all passing (0.20s).
- Protocol module tests: enums, dataclasses, assessment properties, ID generation
- SensorFailureDetector: 8 tests (healthy, offline, stale, out-of-range, quality, multiple)
- TrustCollapseDetector: 8 tests (healthy, yellow/orange/red thresholds, multi-subsystem, mixed)
- CommunicationLossDetector: 4 tests (recent, timeout, dead, custom thresholds)
- SafetyViolationDetector: 8 tests (nominal, empty, E-Stop, FAULT, CRITICAL, WARNING, watchdog, no double-count)
- MissionTimeoutDetector: 5 tests (on-time, timeout, just-over, infinite, zero)
- EmergencyResponder: 10 tests (yellow/orange/red responses, bridge integration, fleet alerts)
- EmergencyProtocol integration: 12 tests (healthy state, all 5 emergency types, monitoring multiplier, incident accumulation)
- Escalation: 4 tests (yellow/orange/red, bridge integration)
- De-escalation: 4 tests (unknown, resolve, to-green, partial)
- Report: 3 tests (empty, with incidents, resolved)
- Thresholds: 3 tests

**Commit:** `4967904` — "Sprint 2D.3+2F: Emergency protocol bridge + Phase 2 ADRs (ADR-029 to ADR-036)"
---
## Task ID: Round 8 (Sprint 2C.3) — rosetta-stone
### Work Task
Build a 4-layer translation pipeline (Rosetta Stone) that converts human-readable intent into NEXUS bytecode through: Text → Intent → IR → Validated IR → Bytecode.

### Work Summary
Created 7 modules in `/tmp/nexus-runtime/jetson/agent/rosetta_stone/`:

1. **`intent_parser.py`** (Layer 1): Rule-based regex parser converting 11 intent patterns into structured `Intent` dataclasses — READ, WRITE, CONDITIONAL, LOOP, WAIT, PID, NAVIGATE, SYSCALL, plus compound intents (monitor+trigger, patrol).

2. **`intent_compiler.py`** (Layer 2): Compiles `Intent` objects into `IRInstruction` lists with label-based control flow. Handles conditionals (JUMP_IF_FALSE branching), loops (counter-based iteration with backward jumps), PID syscalls, navigation waypoints, and CLAMP_F safety bounds before every actuator write.

3. **`ir_validator.py`** (Layer 3): Validates IR for pin ranges (sensors 0-63, actuators 0-63), stack depth (max 16), cycle budgets, trust level restrictions (L0-L5), jump target resolution, infinite loop detection, and NaN/Inf guards. Peephole optimizations: PUSH/POP removal, constant arithmetic folding (ADD/SUB/MUL/DIV), NOP collapsing.

4. **`bytecode_generator.py`** (Layer 4): Generates valid 8-byte NEXUS bytecode using the existing `reflex.bytecode_emitter.BytecodeEmitter`. Resolves IR labels to instruction indices. Includes disassembler. All bytecode passes the existing `reflex.safety_validator.SafetyValidator`.

5. **`rosetta.py`**: Top-level `RosettaStone` class chaining all 4 layers with `translate()`, `translate_many()`, and `translate_combined()` APIs. Configurable trust level and optimization toggle.

6. **`__init__.py`**: Public API exports.

7. **`tests/test_rosetta.py`**: 124 tests covering all intent patterns, compilation, validation, optimization, bytecode generation, full pipeline, safety verification, trust levels, error handling, compound intents, and edge cases. All 124 tests pass. All 150 existing tests continue to pass.

**Key design decisions:**
- Bytecode encoding reuses `BytecodeEmitter` from `reflex.bytecode_emitter` for compatibility
- Actuator writes always preceded by `CLAMP_F(-1, 1)` for safety
- Syscalls use `NOP + FLAGS_SYSCALL` encoding matching the instruction.h spec
- Labels resolved to instruction indices during bytecode generation (not stored in bytecode)
- Trust levels L0-L5 restrict available operations, validated before bytecode emission

**Commit**: `805d672`
