# NEXUS Builder Skill Kit

> **Version**: 2.0 | **Date**: 2026-04-05 | **For**: AI coding agents and human builders

---

## How to Use This Document

This kit equips builders (human or agentic) with everything needed to implement the NEXUS Phase 2 roadmap. Each section provides:

1. **Context** — What you're building and why
2. **Prerequisites** — What must exist before you start
3. **Step-by-step instructions** — Exact commands and file changes
4. **Verification** — How to prove it works
5. **Common pitfalls** — What breaks and how to fix it

---

## Builder Profiles

### Profile A: Power User (Minimum Install)

You want the bytecode VM and serial communication. Nothing else.

```bash
# Clone
git clone https://github.com/SuperInstance/nexus-runtime.git
cd nexus-runtime

# What you need:
#   firmware/src/core/nexus_vm/     — The VM (32 opcodes, C)
#   firmware/src/core/wire_protocol/ — Serial framing (COBS + CRC-16)
#   jetson/core/nexus_vm/           — Python compiler + validator
#   shared/bytecode/               — Cross-platform opcode definitions

# Build tests (host-side, no ESP32 needed)
mkdir build && cd build
cmake .. -DNEXUS_HOST_TEST=ON
make -j4
./test_firmware_main    # Unity tests
./test_firmware_vm      # VM-specific tests

# Run Python tests
cd ../jetson
pip install -r requirements.txt
python -m pytest nexus_vm/ -v
```

### Profile B: Marine Developer (Standard Install)

You want the full marine robotics stack with safety, trust, and reflex compilation.

```bash
# Same as above, plus:
#   firmware/src/safety/      — 4-tier safety system
#   firmware/src/drivers/     — HAL + sensor/actuator drivers
#   jetson/reflex/            — JSON→bytecode reflex compiler
#   jetson/trust/             — INCREMENTS trust engine
#   jetson/agent/             — AAB codec + A2A opcodes
#   schemas/                  — JSON config schemas

# Compile a reflex behavior
cd jetson
python -c "
from reflex.compiler import ReflexCompiler
compiler = ReflexCompiler()
bytecode = compiler.compile_file('configs/my_reflex.json')
with open('my_reflex.bin', 'wb') as f:
    f.write(bytecode)
print(f'Compiled {len(bytecode)} bytes of bytecode')
"

# Validate trust
python -c "
from trust.engine import TrustEngine
engine = TrustEngine()
engine.record_event('sensor', 'GOOD', severity=1)
print(f'Trust: {engine.get_trust_score(\"sensor\"):.3f} (Level {engine.get_level(\"sensor\")})')
"
```

### Profile C: Fleet Builder (Full Deployment)

You want git-agent integration, fleet orchestration, and the full ecosystem.

```bash
# Clone all repos
git clone https://github.com/SuperInstance/nexus-runtime.git
git clone https://github.com/SuperInstance/nexus-fleet.git
git clone https://github.com/SuperInstance/nexus-knowledge.git
git clone https://github.com/SuperInstance/nexus-onboarding.git

# Fork son's ecosystem repos (under SuperInstance org)
# gh repo fork Lucineer/edgenative-ai --clone
# gh repo fork Lucineer/increments-fleet-trust --clone
# gh repo fork Lucineer/fleet-orchestrator --clone
# gh repo fork Lucineer/git-agent --clone

# Deploy edge heartbeat on Jetson
cd nexus-fleet/edge_heartbeat
pip install -r requirements.txt
cp config/vessel_template.json config/my_vessel.json
# Edit my_vessel.json with vessel name, sensors, capabilities
python heartbeat.py --config config/my_vessel.json
```

---

## Critical Technical Reference

### The 8-Byte Instruction Format

Every NEXUS bytecode instruction is exactly 8 bytes, packed as:

```
Byte 0:   opcode        (uint8, 0x00-0x1F for core, 0x20-0x56 for A2A)
Byte 1:   flags         (uint8, bitfield: IS_CALL, IS_COND, etc.)
Byte 2-3: operand1      (uint16, little-endian)
Byte 4-7: operand2      (uint32, little-endian)
```

**CRITICAL**: JUMP targets use `operand2` (uint32), NOT `operand1`. This was bug #1.
**CRITICAL**: CLAMP_F uses IEEE 754 float16 bit patterns in `operand2`. This was bug #2.

### The 32 Core Opcodes

```
0x00 NOP          0x08 PUSH_I16     0x10 ADD_F        0x18 READ_PIN
0x01 PUSH_F32     0x09 POP          0x11 SUB_F        0x19 WRITE_PIN
0x02 PUSH_I32     0x0A DUP          0x12 MUL_F        0x1A CALL
0x03 PUSH_U8      0x0B SWAP         0x13 DIV_F        0x1B RET
0x04 LOAD_VAR     0x0C EQ           0x14 MOD_F        0x1C JUMP
0x05 STORE_VAR    0x0D LT           0x15 NEG_F        0x1D JUMP_IF_TRUE
0x06 ADD_I        0x0E GT           0x16 ABS_F        0x1E CLAMP_F
0x07 SUB_I        0x0F AND          0x17 MIN_MAX_F    0x1F SYSCALL
```

### I/O Pin Routing

```
Sensors:   pins 0-63     → READ_PIN loads value to stack
Variables: pins 64-319   → LOAD_VAR/STORE_VAR (256 variables)
Actuators: pins 0-63     → WRITE_PIN pops value from stack
```

### Wire Protocol Frame

```
[0x00][payload...][0x00]    ← COBS-encoded payload (delineated by zero bytes)
                             ↑ CRC-16 appended to payload before COBS encoding
```

Message header (10 bytes, first in payload after COBS decode):
```
Byte 0:   msg_type (uint8, 28 types)
Byte 1-2: msg_id (uint16, incrementing)
Byte 3:   source_id (uint8)
Byte 4:   dest_id (uint8)
Byte 5-6: payload_len (uint16)
Byte 7-8: flags (uint16)
Byte 9:   sequence (uint8)
```

### INCREMENTS Trust Formula

```
For GOOD events:  δ = +1 × (severity / τ_gain)
For BAD events:   δ = -1 × (severity / τ_loss) × α_consecutive
For IDLE:         δ = -1 × (1 / τ_decay)

τ_gain = 500 (slow to earn trust)
τ_loss = 20  (fast to lose trust)
25:1 ratio (one bad event = 25 good events to recover)

Levels:
  L0 (0.00-0.10): Manual only
  L1 (0.10-0.30): Supervised
  L2 (0.30-0.50): Assisted
  L3 (0.50-0.70): Conditional autonomy
  L4 (0.70-0.85): High autonomy
  L5 (0.85-1.00): Full autonomy
```

---

## Sprint-by-Sprint Builder Instructions

### Sprint 2A.1: Repository Restructuring

**What you're doing**: Reorganizing files into a cleaner module structure.

**Prerequisites**:
- Clean git working tree (`git status` shows nothing)
- All 252 tests passing
- Backup branch created: `git branch backup/pre-2A`

**Step-by-step**:

```bash
cd /tmp/nexus-runtime

# 1. Create new directory structure
mkdir -p firmware/src/core firmware/src/safety firmware/src/drivers firmware/src/main
mkdir -p jetson/core jetson/reflex jetson/trust jetson/agent jetson/learning jetson/sdk

# 2. Move firmware files (using git mv for history preservation)
git mv firmware/src/nexus_vm firmware/src/core/nexus_vm
git mv firmware/src/wire_protocol firmware/src/core/wire_protocol
git mv firmware/src/safety firmware/src/safety
git mv firmware/src/drivers firmware/src/drivers
# main/app_main.c stays or moves to firmware/src/main/

# 3. Move jetson files
git mv jetson/nexus_vm jetson/core/nexus_vm
git mv jetson/reflex_compiler jetson/reflex
git mv jetson/trust_engine jetson/trust
git mv jetson/agent_runtime jetson/agent
git mv jetson/learning jetson/learning
git mv jetson/nexus_sdk jetson/sdk

# 4. Fix C includes
# firmware/src/core/nexus_vm/vm_core.c has: #include "vm.h"
# After move, vm.h is in same directory — should still work.
# But if anything includes "wire_protocol/cobs.h" it needs updating.
rg '#include.*wire_protocol' firmware/ --files-with-matches
# Fix each one:
# OLD: #include "wire_protocol/cobs.h"
# NEW: #include "core/wire_protocol/cobs.h"
# (or add -I firmware/src/core to CMakeLists.txt)

# 5. Fix Python imports
rg 'from nexus_vm|from reflex_compiler|from trust_engine|from agent_runtime' jetson/ --files-with-matching
# Fix each one:
# OLD: from reflex_compiler.compiler import ReflexCompiler
# NEW: from reflex.compiler import ReflexCompiler

# 6. Update CMakeLists.txt
# Point source paths to new locations

# 7. Run tests
cd build && cmake .. -DNEXUS_HOST_TEST=ON && make -j4
./test_firmware_main
./test_firmware_vm
cd ../jetson && python -m pytest -v

# 8. If all green, commit
git add -A
git commit -m "Sprint 2A.1: Modular directory restructuring"
```

**Common pitfalls**:
- Forgetting to update CMakeLists.txt source paths → build fails
- Python relative imports breaking → add `__init__.py` or update `sys.path`
- Test files importing from old paths → update test imports too
- CI breaking because paths changed → update workflow files

### Sprint 2A.2: Schema Extraction

**What you're doing**: Bringing JSON schemas from edge-ref into nexus-runtime with validation.

**Prerequisites**: Sprint 2A.1 complete, tests green.

```bash
# 1. Copy schemas
cp /tmp/edge-ref/schemas/post_coding/autonomy_state.json /tmp/nexus-runtime/schemas/
cp /tmp/edge-ref/schemas/post_coding/reflex_definition.json /tmp/nexus-runtime/schemas/
cp /tmp/edge-ref/schemas/post_coding/node_role_config.json /tmp/nexus-runtime/schemas/
cp /tmp/edge-ref/schemas/post_coding/serial_protocol.json /tmp/nexus-runtime/schemas/

# 2. Create schema validator
cat > jetson/core/schema_validator.py << 'PY'
import json
import jsonschema

def validate_config(config_path, schema_path):
    with open(schema_path) as f:
        schema = json.load(f)
    with open(config_path) as f:
        config = json.load(f)
    jsonschema.validate(config, schema)
    return True
PY

# 3. Write tests for each schema
# Use the configs from edge-ref as test data

# 4. Run tests
python -m pytest jetson/core/test_schema_validator.py -v

# 5. Commit
git add -A
git commit -m "Sprint 2A.2: Schema extraction and validation"
```

### Sprint 2B.1: git-agent Bridge

**What you're doing**: Creating a TypeScript module that translates between NEXUS Wire Protocol and git-agent GitHub API operations.

**Prerequisites**: Sprint 2A complete. Node.js 18+ and npm installed. GitHub PAT available.

```bash
# 1. Create the bridge project
mkdir -p /tmp/nexus-fleet/git-agent-bridge/src
mkdir -p /tmp/nexus-fleet/git-agent-bridge/tests
cd /tmp/nexus-fleet/git-agent-bridge

# 2. Initialize
cat > package.json << 'PKG'
{
  "name": "nexus-git-agent-bridge",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "test": "vitest run",
    "build": "tsc"
  },
  "dependencies": {
    "@octokit/rest": "^20.0.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "vitest": "^1.0.0"
  }
}
PKG

npm install

# 3. Implement the bridge
# Key interface: Wire Protocol message → GitHub action
# Key interface: GitHub event → Wire Protocol command

# 4. Write tests
# Mock both GitHub API and Wire Protocol

# 5. Commit
git init
git add -A
git commit -m "Sprint 2B.1: git-agent bridge module"
```

**Key design decisions for builders**:
- The bridge should be a **transform layer**, not a runtime. It takes a Wire Protocol message and produces a GitHub API call, or vice versa.
- All bridge operations should be **idempotent** — replaying a commit should produce the same state.
- Trust events should be **batched** — don't create a commit per sensor reading. Aggregate into 5-minute windows.

---

## Testing Strategy

### Unit Tests (What we have)

| Component | Tests | Framework | Coverage |
|---|---|---|---|
| VM opcodes | 34+ | Unity (C) | All 32 opcodes |
| COBS | 8 | Unity (C) | Edge cases |
| CRC-16 | 9 | Unity (C) | Standard vectors |
| Wire roundtrip | 40+ | pytest | Frame encode/decode |
| Trust engine | 45+ | pytest | All trust scenarios |
| AAB codec | 35+ | pytest | All 29 A2A opcodes |
| Compiler | 30+ | pytest | JSON→bytecode |

### Integration Tests (What we need)

| Test | Purpose | Sprint |
|---|---|---|
| Jetson → ESP32 bytecode deploy | Full pipeline: compile → validate → deploy → execute | 2B.2 |
| Trust event propagation | Edge → git-agent → fleet attenuation | 2C.2 |
| Safety validation pipeline | 6-stage bytecode validation | 2D.2 |
| Emergency protocol | E-Stop → Issue → fleet alert | 2D.3 |
| CRDT convergence | Offline sync → reconnect → merge | 2E.1 |
| Skill loading | Cartridge → bytecode → deploy → verify | 2E.2 |

### Cross-Language Consistency Tests

**Critical**: The same INCREMENTS trust computation must produce identical results in C, Python, and TypeScript.

```python
# Test vector: 10 GOOD events followed by 1 BAD event
# Expected: trust ≈ 0.020 (L0) after 10 GOOD, drops significantly after 1 BAD
# All three languages must agree within floating-point epsilon (1e-6)
```

---

## Coding Standards

### C Firmware (ESP32)

- **Zero heap allocation** after init. All buffers static.
- **Deterministic execution**: No variadic functions, no recursion, no dynamic memory.
- **Every function has a max cycle count** documented in comments.
- **Error codes are uint8** from the 75-code table. Never use negative numbers.
- **All public APIs return error codes**, never void (except vm_init).
- **Safety-first**: If in doubt, HALT. A halted VM is always safer than a wrong VM.

### Python (Jetson SDK)

- **Python 3.11+**, type hints everywhere, mypy strict mode.
- **No async in the core compiler** (keep it simple and deterministic).
- **All bytecode output is bytes**, never str. No encoding surprises.
- **Float32 → Float16**: Always use `f32_to_f16_bits()` for IEEE 754 bit patterns, never integer scaling.
- **Test coverage**: Every public function has at least one test.

### TypeScript (git-agent Bridge)

- **ESM modules** (import/export, no require).
- **No external runtime dependencies** beyond @octokit/rest.
- **All functions are pure** where possible. Side effects isolated in adapters.
- **Error handling**: Never throw. Return `{ok: true, value: T} | {ok: false, error: string}`.

---

## IDE Accessibility

Any IDE can work with this codebase:

- **Cursor / VS Code**: C extension (clangd), Python extension (Pylance), TypeScript extension
- **CLion**: Native CMake support, Python plugin
- **Vim/Neovim**: LSP servers (clangd, pyright, typescript-language-server)
- **Agent-assisted**: Open repo in any IDE, any human or AI agent can read the code, understand the architecture from docs/, make changes, submit PRs.

**Key files for IDE context**:
1. `docs/planning/HIGH_LEVEL_SCHEMA.md` — Architecture overview
2. `docs/specs/00_MASTER_INDEX.md` — Spec navigation
3. `shared/bytecode/opcodes.h` — Opcode definitions
4. `shared/bytecode/instruction.h` — Instruction format
5. `schemas/*.json` — Config validation

---

## Quick Reference Commands

```bash
# Run all tests
cd /tmp/nexus-runtime
mkdir -p build && cd build && cmake .. -DNEXUS_HOST_TEST=ON && make -j4
./test_firmware_main && ./test_firmware_vm
cd ../jetson && python -m pytest -v

# Compile a reflex
cd jetson && python -m reflex.compiler compile configs/example.json -o output.bin

# Validate bytecode
cd jetson && python -m core.validator validate output.bin

# Check trust
cd jetson && python -c "from trust.engine import TrustEngine; e=TrustEngine(); e.record_event('nav','GOOD',1); print(e.get_level('nav'))"

# Git status
cd /tmp/nexus-runtime && git log --oneline -10 && git status

# Push to GitHub
cd /tmp/nexus-runtime && git push origin main
```
