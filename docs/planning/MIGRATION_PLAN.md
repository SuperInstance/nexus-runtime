# Edge-Native → NEXUS Migration Plan

> **Version**: 1.0 | **Date**: 2026-04-05 | **Source**: `/tmp/edge-ref` | **Target**: `nexus-runtime` + new repos

---

## Migration Philosophy

The Edge-Native repository is a **design monastery** — ~1.1M words of specifications, research, and philosophical exploration that informed what NEXUS is today. Much of it has been absorbed into the code (252 tests, ~11K LOC prove that). But the remaining knowledge must be **refractured** — not copied wholesale, but restructured into the new git-native, modular paradigm.

### Principles

1. **Don't dump** — Don't copy 1.1M words into a code repo. Extract what's needed.
2. **Preserve truth** — If a spec says X and the code does Y, investigate. Specs are source of truth unless code has explicit ADR override.
3. **Modular separation** — Knowledge base separate from specs. Specs separate from code. Onboarding separate from everything.
4. **Git-native format** — One ADR per file (not one mega-file). One spec per component (not one mega-spec). PRs for changes.
5. **Evolve, don't transliterate** — Update Claude Code references to git-agent. Update monolithic assumptions to modular.

---

## Phase A: Immediate (Sprint 2A)

### A1. Schemas (CRITICAL)

| Source | Target | Action |
|---|---|---|
| `schemas/post_coding/autonomy_state.json` | `nexus-runtime/schemas/autonomy_state.json` | Copy + add validation tests |
| `schemas/post_coding/reflex_definition.json` | `nexus-runtime/schemas/reflex_definition.json` | Copy + add validation tests |
| `schemas/post_coding/node_role_config.json` | `nexus-runtime/schemas/node_role_config.json` | Copy + add validation tests |
| `schemas/post_coding/serial_protocol.json` | `nexus-runtime/schemas/serial_protocol.json` | Copy + add validation tests |

### A2. Production Specs (CRITICAL)

| Source | Target | Action |
|---|---|---|
| `specs/00_MASTER_INDEX.md` | `nexus-runtime/docs/specs/00_MASTER_INDEX.md` | Copy + reconcile with implementation status |
| `specs/ARCHITECTURE_DECISION_RECORDS.md` | `nexus-runtime/docs/specs/ARCHITECTURE_DECISION_RECORDS.md` | Copy + split into individual ADR files |
| `specs/NEXUS_Phase1_Foundation_Complete_Specs.md` | `nexus-runtime/docs/specs/PHASE1_SPECS.md` | Copy + mark implemented vs pending |
| `specs/SENIOR_ENGINEER_BUILD_GUIDE.md` | `nexus-runtime/docs/specs/BUILD_GUIDE.md` | Copy + update for git-native paradigm |

### A3. Build Specification (CRITICAL)

| Source | Target | Action |
|---|---|---|
| `claude-build/build-specification.md` | `nexus-runtime/docs/specs/build-specification.md` | Copy + refactor into per-component sections |

### A4. Agent Onboarding (HIGH)

| Source | Target | Action |
|---|---|---|
| `claude.md` | `nexus-runtime/docs/GIT_AGENT.md` | Rewrite for git-agent paradigm |
| `onboarding/context-map.md` | `nexus-onboarding/context-map.md` | Copy + update references |
| `onboarding/methodology.md` | `nexus-onboarding/methodology.md` | Copy + update references |

### A5. Configuration Files (MEDIUM)

| Source | Target | Action |
|---|---|---|
| `autopilot/configs/firmware_config.json` | `nexus-runtime/configs/firmware.json` | Copy + validate against schema |
| `autopilot/configs/safety_config.json` | `nexus-runtime/configs/safety.json` | Copy + validate against schema |
| `autopilot/configs/pin_config.json` | `nexus-runtime/configs/pins.json` | Copy + validate against schema |
| `autopilot/configs/task_config.json` | `nexus-runtime/configs/tasks.json` | Copy |
| `vessel-platform/configs/cluster_config.json` | `nexus-runtime/configs/cluster.json` | Copy |
| `vessel-platform/configs/vessel_network_config.json` | `nexus-runtime/configs/network.json` | Copy |

---

## Phase B: Near-Term (Sprint 2B-2C)

### B1. Engineering Addenda (HIGH)

| Source | Target | Action |
|---|---|---|
| `addenda/engineering_pitfalls.md` | `nexus-runtime/docs/addenda/pitfalls.md` | Copy to runtime repo |
| `addenda/safety_validation_playbook.md` | `nexus-runtime/docs/addenda/safety_playbook.md` | Copy to runtime repo |
| `addenda/integration_test_plan.md` | `nexus-runtime/docs/addenda/integration_tests.md` | Copy to runtime repo |
| `addenda/code_review_checklist.md` | `nexus-runtime/docs/addenda/review_checklist.md` | Copy to runtime repo |
| `addenda/hardware_bringup_checklist.md` | `nexus-runtime/docs/addenda/hardware_bringup.md` | Copy to runtime repo |

### B2. A2A Native Specs (HIGH)

| Source | Target | Action |
|---|---|---|
| `a2a-native-specs/README.md` | `nexus-runtime/docs/specs/a2a/README.md` | Copy |
| `a2a-native-specs/bytecode_vm_a2a_native.md` | `nexus-runtime/docs/specs/a2a/vm_spec.md` | Copy |
| (remaining a2a specs) | `nexus-runtime/docs/specs/a2a/` | Copy + create missing Rosetta Stone twins |

### B3. Incubator Manifesto (MEDIUM)

| Source | Target | Action |
|---|---|---|
| `incubator/manifesto.md` | `nexus-knowledge/manifesto.md` | Copy to knowledge repo |

---

## Phase C: Medium-Term (Sprint 2D-2E)

### C1. Knowledge Base (MEDIUM)

| Source | Target | Action |
|---|---|---|
| `knowledge-base/glossary.md` (310 terms) | `nexus-knowledge/reference/glossary.md` | Copy |
| `knowledge-base/annotated_bibliography.md` (178 refs) | `nexus-knowledge/reference/bibliography.md` | Copy |
| `knowledge-base/open_problems.md` (29 problems) | `nexus-knowledge/reference/open_problems.md` | Copy |
| `knowledge-base/developer_onboarding_guide.md` | `nexus-knowledge/guides/developer_onboarding.md` | Copy + update |
| (remaining 23 articles) | `nexus-knowledge/articles/` | Copy, organize by category |

### C2. Onboarding Suite (MEDIUM)

| Source | Target | Action |
|---|---|---|
| `onboarding/gamified-intro.md` | `nexus-onboarding/gamified-intro.md` | Copy + update for git-native |
| `onboarding/concept-playground.md` | `nexus-onboarding/concept-playground.md` | Copy |
| `onboarding/builder-education.md` | `nexus-onboarding/builder-education.md` | Copy + update |
| `onboarding/architecture-patterns.md` | `nexus-onboarding/architecture-patterns.md` | Copy |
| `onboarding/use-case-scenarios.md` | `nexus-onboarding/use-cases.md` | Copy |
| `onboarding/expansion-guide.md` | `nexus-onboarding/expansion-guide.md` | Copy |

---

## Phase D: Long-Term (Sprint 2F)

### D1. Framework & Autopilot (LOW)

| Source | Target | Action |
|---|---|---|
| `framework/*.txt` (7 files, ~57K words) | `nexus-knowledge/framework/` | Archive — concepts absorbed into specs |
| `autopilot/*.md` (15 files, ~63K words) | `nexus-knowledge/autopilot/` | Archive — marine domain reference |

### D2. Dissertation (LOW)

| Source | Target | Action |
|---|---|---|
| `dissertation/*.md` (30+ files, ~132K words) | `nexus-knowledge/dissertation/` | Archive — research findings |

### D3. Genesis Colony & V31 (LOW)

| Source | Target | Action |
|---|---|---|
| `genesis-colony/` (40 files, ~210K words) | Archive | Historical — concepts distilled into a2a-native |
| `v31-docs/` (13 files) | Archive | Superseded by current docs |

### D4. Human-Readable (LOW)

| Source | Target | Action |
|---|---|---|
| `human-readable/*.md` (3 files, ~27K words) | `nexus-knowledge/human-readable/` | Copy — for stakeholder communication |

---

## Files NOT Migrated (Intentionally)

| File/Directory | Reason |
|---|---|
| `worklog.md` | Historical artifact only |
| `archives/*.zip` | Binary archives, outdated |
| `.github/workflows/` | NEXUS has its own CI; reference only |
| `firmware/safety/*.c` (stubs) | NEXUS has full implementations |
| `firmware/wire_protocol/wire_rx.c` (stub) | NEXUS has full implementation |
| `jetson/main/nexus_main.py` (stub) | NEXUS has SDK pipeline |
| `jetson/**/__init__.py` (empty) | Generated during Python setup |
| `claude-build/CLAUDE_BUILD_SPECIFICATION.md` | Superseded by build-specification.md |

---

## Reconciliation Checklist

For each spec section migrated, verify against nexus-runtime implementation:

- [ ] VM instruction format: 8-byte packed, verified identical ✅
- [ ] Opcodes: 32 core + 29 A2A, verified identical ✅
- [ ] Wire protocol: COBS + CRC-16 + 10-byte header, verified ✅
- [ ] Trust engine: INCREMENTS with 12 params, verified ✅
- [ ] Safety: 4-tier architecture, verified ✅
- [ ] Error codes: 75 defined, verify all handled in firmware
- [ ] Message types: 28 defined, verify 29 implemented (extra?)
- [ ] Syscalls: 4 defined (HALT, PID_COMPUTE, RECORD_SNAPSHOT, EMIT_EVENT), verify ✅
- [ ] I/O pin routing: sensors 0-63, vars 64-319, actuators 0-63, verify ✅
- [ ] Cycle budgets: 10K max per tick, verify enforced ✅
- [ ] Stack depth: 64 entries, verify enforced ✅
- [ ] Memory layout: ~5.4KB vm_state_t, verify struct matches ✅

### Discrepancies Found (if any)

To be filled during Sprint 2A.4 when specs are formally reconciled against code.
