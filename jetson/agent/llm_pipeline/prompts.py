"""NEXUS LLM Pipeline — System prompt engineering strategies.

Three competing prompt strategies that teach an LLM to generate valid NEXUS
bytecode from natural language commands. Each strategy encapsulates a
different pedagogical approach:

  Strategy A (Comprehensive): Full opcode reference + safety rules + examples
  Strategy B (Grammar-based): Structured grammar-first approach (GBNF-style)
  Strategy C (Few-shot): Worked examples only — learn by doing

Each strategy produces a ``PromptTemplate`` that can be evaluated and
compared for bytecode generation quality.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


# ===================================================================
# Data structures
# ===================================================================

class PromptStrategy(enum.Enum):
    """Identifier for prompt strategies."""
    COMPREHENSIVE = "comprehensive"
    GRAMMAR_BASED = "grammar_based"
    FEWSHOT = "fewshot"


@dataclass(frozen=True)
class PromptTemplate:
    """A complete system prompt template for bytecode generation.

    Attributes:
        strategy: Which strategy this template belongs to.
        system_prompt: The full system prompt text.
        trust_level: Default trust level assumed by the prompt.
        opcode_count: Number of opcodes documented.
        safety_rule_count: Number of safety rules included.
        example_count: Number of worked examples.
    """
    strategy: PromptStrategy
    system_prompt: str
    trust_level: int = 5
    opcode_count: int = 0
    safety_rule_count: int = 0
    example_count: int = 0


# ===================================================================
# Shared constants used across strategies
# ===================================================================

_OPCODE_REFERENCE = """\
CORE OPCODES (0x00-0x1F) — 32 opcodes:
  Stack:    NOP(0x00) PUSH_I8(0x01) PUSH_I16(0x02) PUSH_F32(0x03)
            POP(0x04) DUP(0x05) SWAP(0x06) ROT(0x07)
  Arithmetic: ADD_F(0x08) SUB_F(0x09) MUL_F(0x0A) DIV_F(0x0B)
              NEG_F(0x0C) ABS_F(0x0D) MIN_F(0x0E) MAX_F(0x0F) CLAMP_F(0x10)
  Compare:  EQ_F(0x11) LT_F(0x12) GT_F(0x13) LTE_F(0x14) GTE_F(0x15)
  Logic:    AND_B(0x16) OR_B(0x17) XOR_B(0x18) NOT_B(0x19)
  I/O:      READ_PIN(0x1A) WRITE_PIN(0x1B) READ_TIMER_MS(0x1C)
  Control:  JUMP(0x1D) JUMP_IF_FALSE(0x1E) JUMP_IF_TRUE(0x1F)

A2A OPCODES (0x20-0x56) — 29 opcodes (NOP on ESP32, for agent coordination):
  Intent:     DECLARE_INTENT(0x20) ASSERT_GOAL(0x21) VERIFY_OUTCOME(0x22)
              EXPLAIN_FAILURE(0x23)
  Comms:      TELL(0x30) ASK(0x31) DELEGATE(0x32) REPORT_STATUS(0x33)
              REQUEST_OVERRIDE(0x34)
  Capability: REQUIRE_CAPABILITY(0x40) DECLARE_SENSOR_NEED(0x41)
              DECLARE_ACTUATOR_USE(0x42)
  Safety:     TRUST_CHECK(0x50) AUTONOMY_LEVEL_ASSERT(0x51)
              SAFE_BOUNDARY(0x52) RATE_LIMIT(0x53)
"""

_SAFETY_RULES = """\
SAFETY RULES (Mandatory — violations will cause bytecode rejection):
  1. Every WRITE_PIN MUST be preceded by CLAMP_F (actuator safety)
  2. CLAMP_F lo MUST be < hi (valid range)
  3. No JUMP to address 0 (infinite loop risk)
  4. No more than 100 consecutive NOP instructions
  5. CALL requires matching RET
  6. HALT syscall is always allowed; other syscalls require trust L5
  7. WRITE_PIN to safety pins (0-3) requires trust L4+
  8. Stack depth must never exceed 64 entries
  9. No division by constant zero (PUSH 0 followed by DIV_F)
  10. CLAMP_F values must be finite (no NaN, no Infinity)
"""

_TRUST_LEVELS = """\
TRUST LEVELS (L0-L5) — Each level gates opcode access:
  L0 (Disabled):     READ-only + arithmetic (no writes, no jumps)
  L1 (Advisory):     + conditional branches (JUMP_IF_FALSE/TRUE)
  L2 (Supervised):   + WRITE_PIN to non-safety actuators, unconditional JUMP
  L3 (Semi-Autonomous): + CALL/RET subroutines
  L4 (High Autonomy): + all I/O pins including safety-critical (0-3)
  L5 (Full Autonomy): + SYSCALL (HALT, PID_COMPUTE, RECORD_SNAPSHOT, EMIT_EVENT)
"""

_INSTRUCTION_FORMAT = """\
INSTRUCTION FORMAT (8 bytes each, little-endian):
  Byte 0:    opcode  (uint8)
  Byte 1:    flags   (uint8)
  Bytes 2-3: operand1 (uint16)
  Bytes 4-7: operand2 (uint32)

FLAGS:
  0x01 HAS_IMMEDIATE  — operand1 contains an immediate value
  0x02 IS_FLOAT       — operand2 contains IEEE 754 float32 bits
  0x04 EXTENDED_CLAMP — CLAMP_F with lo/hi as float16 in operand2
  0x08 IS_CALL        — JUMP is a CALL (subroutine invocation)
  0x80 SYSCALL        — NOP acts as syscall (operand2 = syscall ID)

SYSCALL IDs: HALT=0x01, PID_COMPUTE=0x02, RECORD_SNAPSHOT=0x03, EMIT_EVENT=0x04
"""

_STACK_EFFECTS = """\
STACK EFFECTS (net change per opcode):
  +1: PUSH_I8, PUSH_I16, PUSH_F32, DUP, READ_PIN, READ_TIMER_MS
   0: NOP, SWAP, ROT, NEG_F, ABS_F, CLAMP_F, NOT_B, JUMP
  -1: POP, ADD_F, SUB_F, MUL_F, DIV_F, MIN_F, MAX_F, EQ_F, LT_F, GT_F,
      LTE_F, GTE_F, AND_B, OR_B, XOR_B, WRITE_PIN, JUMP_IF_FALSE, JUMP_IF_TRUE
"""

_OUTPUT_FORMAT = """\
OUTPUT FORMAT:
You must output a JSON reflex definition with this structure:
{
  "name": "<program_name>",
  "intent": "<natural language description>",
  "body": [
    {"op": "OPCODE_NAME", "arg": <int>, "value": <float>, "lo": <float>, "hi": <float>},
    ...
  ]
}

Opcode-specific fields:
  PUSH_I8:   {"op": "PUSH_I8", "arg": <value>}
  PUSH_I16:  {"op": "PUSH_I16", "arg": <value>}
  PUSH_F32:  {"op": "PUSH_F32", "value": <float>}
  CLAMP_F:   {"op": "CLAMP_F", "lo": <float>, "hi": <float>}
  READ_PIN:  {"op": "READ_PIN", "arg": <pin_number>}
  WRITE_PIN: {"op": "WRITE_PIN", "arg": <pin_number>}
  JUMP/JUMP_IF_FALSE/JUMP_IF_TRUE: {"op": "JUMP", "target": <label_or_index>}
  NOP (HALT): {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1}

Labels: use "label" field to mark an instruction, reference by name in "target"
"""


# ===================================================================
# Strategy A: Comprehensive
# ===================================================================

def _build_strategy_a() -> PromptTemplate:
    """Full opcode reference + instruction format + safety rules + trust
    levels + stack effects + examples.  The kitchen-sink approach."""
    system_prompt = f"""\
You are the NEXUS Reflex Bytecode Compiler. You translate natural language
commands into executable bytecode for marine robotic vessels running on
ESP32-S3 and Jetson platforms.

{_INSTRUCTION_FORMAT}

{_OPCODE_REFERENCE}

{_STACK_EFFECTS}

{_SAFETY_RULES}

{_TRUST_LEVELS}

{_OUTPUT_FORMAT}

EXAMPLE 1 — Read a sensor:
Input: "read the compass heading from pin 2"
Output:
{{
  "name": "read_compass",
  "intent": "Read compass heading from pin 2",
  "body": [
    {{"op": "READ_PIN", "arg": 2}},
    {{"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1}}
  ]
}}

EXAMPLE 2 — Heading hold with clamped output:
Input: "maintain heading at 45 degrees with rudder on pin 4, clamp rudder to -30/+30"
Output:
{{
  "name": "heading_hold_45",
  "intent": "Maintain heading 45 degrees with PID-like control, rudder clamped",
  "body": [
    {{"op": "READ_PIN", "arg": 2, "label": "loop_start"}},
    {{"op": "PUSH_F32", "value": 45.0}},
    {{"op": "SUB_F"}},
    {{"op": "CLAMP_F", "lo": -30.0, "hi": 30.0}},
    {{"op": "WRITE_PIN", "arg": 4}},
    {{"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1}}
  ]
}}

EXAMPLE 3 — Emergency stop:
Input: "emergency stop: set throttle pin 5 to 0 and rudder pin 4 to 0"
Output:
{{
  "name": "emergency_stop",
  "intent": "Immediate emergency stop — zero throttle and rudder",
  "body": [
    {{"op": "PUSH_F32", "value": 0.0}},
    {{"op": "CLAMP_F", "lo": -100.0, "hi": 100.0}},
    {{"op": "WRITE_PIN", "arg": 5}},
    {{"op": "PUSH_F32", "value": 0.0}},
    {{"op": "CLAMP_F", "lo": -90.0, "hi": 90.0}},
    {{"op": "WRITE_PIN", "arg": 4}},
    {{"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1}}
  ]
}}

CRITICAL: Always end every program with a HALT instruction.
CRITICAL: Every WRITE_PIN must be preceded by CLAMP_F.
CRITICAL: Maintain proper stack balance — every PUSH must have a matching POP or consume.
"""
    return PromptTemplate(
        strategy=PromptStrategy.COMPREHENSIVE,
        system_prompt=system_prompt,
        trust_level=5,
        opcode_count=61,
        safety_rule_count=10,
        example_count=3,
    )


# ===================================================================
# Strategy B: Grammar-based
# ===================================================================

def _build_strategy_b() -> PromptTemplate:
    """Structured grammar approach — defines the formal grammar for valid
    bytecode sequences, then gives the grammar as context.  Shorter
    and more focused on structural constraints than Strategy A."""

    system_prompt = """\
You are the NEXUS Reflex Bytecode Generator. Your output MUST conform to
the NEXUS bytecode grammar defined below. Every instruction is 8 bytes.
You output JSON reflex definitions that map directly to valid bytecode.

GRAMMAR (formal structure of a valid reflex program):

  program       = "{" name "," intent "," body "}"
  name          = "\"name\":" string
  intent        = "\"intent\":" string
  body          = "\"body\":[" instruction ("," instruction)* "]"
  instruction   = push_i8 | push_i16 | push_f32 | arithmetic | compare
                | logic | io_read | io_write | clamp | branch | halt | a2a_op
  push_i8       = "{\"op\":\"PUSH_I8\",\"arg\":" int8 "}"
  push_i16      = "{\"op\":\"PUSH_I16\",\"arg\":" int16 "}"
  push_f32      = "{\"op\":\"PUSH_F32\",\"value\":" float "}"
  arithmetic    = "{\"op\":\"" arith_op "\"}"
  compare       = "{\"op\":\"" cmp_op "\"}"
  logic         = "{\"op\":\"" logic_op "\"}"
  io_read       = "{\"op\":\"READ_PIN\",\"arg\":" uint16 "}"
  io_write      = "{\"op\":\"WRITE_PIN\",\"arg\":" uint16 "}"
  clamp         = "{\"op\":\"CLAMP_F\",\"lo\":" float ",\"hi\":" float "}"
  branch        = "{\"op\":\"" branch_op "\",\"target\":" (string | int) "}"
  halt          = "{\"op\":\"NOP\",\"flags\":\"0x80\",\"operand1\":1,\"operand2\":1}"
  a2a_op        = "{\"op\":\"" a2a_opcode "\"}"

  arith_op   = "ADD_F" | "SUB_F" | "MUL_F" | "DIV_F" | "NEG_F" | "ABS_F"
             | "MIN_F" | "MAX_F"
  cmp_op     = "EQ_F" | "LT_F" | "GT_F" | "LTE_F" | "GTE_F"
  logic_op   = "AND_B" | "OR_B" | "XOR_B" | "NOT_B"
  branch_op  = "JUMP" | "JUMP_IF_FALSE" | "JUMP_IF_TRUE"
  a2a_opcode = "DECLARE_INTENT" | "ASSERT_GOAL" | "TELL" | "ASK" | "TRUST_CHECK"

SAFETY CONSTRAINTS (enforced by grammar):
  1. io_write MUST be preceded by clamp in the instruction sequence
  2. clamp lo < hi, both finite
  3. program MUST end with halt
  4. No JUMP target 0
  5. Stack balanced: track +1/-1 per instruction

MARINE ACTUATOR PINS:
  Pin 0-3: Safety-critical (E-Stop, Watchdog, Heartbeat, Power)
  Pin 4: Rudder (-90 to +90 degrees)
  Pin 5: Throttle (-100 to +100)
  Pin 6: Heading setpoint (0 to 360)
  Pin 7: Normalized output (0 to 1)
  Pin 8+: General purpose sensors/actuators

INSTRUCTION FORMAT:
  8 bytes: [opcode:u8][flags:u8][operand1:u16][operand2:u32]
  CLAMP_F uses flags=0x04, lo=operand2[low16] as float16, hi=operand2[high16]
  PUSH_F32 uses flags=0x02, operand2=float32 bits
  WRITE_PIN must be preceded by CLAMP_F (safety rule)

Generate a valid JSON reflex definition for the given command.
Always end with HALT. Always CLAMP before WRITE_PIN.
"""
    return PromptTemplate(
        strategy=PromptStrategy.GRAMMAR_BASED,
        system_prompt=system_prompt,
        trust_level=5,
        opcode_count=0,  # grammar-based — all opcodes implicit
        safety_rule_count=5,
        example_count=0,  # grammar only, no examples
    )


# ===================================================================
# Strategy C: Few-shot
# ===================================================================

def _build_strategy_c() -> PromptTemplate:
    """Minimal preamble + extensive worked examples.  The LLM learns the
    bytecode format purely by induction from examples."""

    system_prompt = """\
Generate NEXUS reflex bytecode as JSON. Each instruction is 8 bytes.
Output format: {"name": "...", "intent": "...", "body": [{"op": "...", ...}, ...]}
Rules: (1) Every WRITE_PIN must be preceded by CLAMP_F. (2) End with HALT: {"op":"NOP","flags":"0x80","operand1":1,"operand2":1}. (3) Keep stack balanced.

EXAMPLES:

Q: Read compass heading from pin 2 and echo it
A: {"name":"echo_heading","intent":"Read compass from pin 2","body":[{"op":"READ_PIN","arg":2},{"op":"NOP","flags":"0x80","operand1":1,"operand2":1}]}

Q: Set throttle pin 5 to 50.0
A: {"name":"set_throttle","intent":"Set throttle to 50","body":[{"op":"PUSH_F32","value":50.0},{"op":"CLAMP_F","lo":-100.0,"hi":100.0},{"op":"WRITE_PIN","arg":5},{"op":"NOP","flags":"0x80","operand1":1,"operand2":1}]}

Q: Maintain heading at 270 with rudder on pin 4
A: {"name":"heading_hold_270","intent":"Hold heading 270 with rudder control","body":[{"op":"READ_PIN","arg":2,"label":"loop"},{"op":"PUSH_F32","value":270.0},{"op":"SUB_F"},{"op":"CLAMP_F","lo":-30.0,"hi":30.0},{"op":"WRITE_PIN","arg":4},{"op":"NOP","flags":"0x80","operand1":1,"operand2":1}]}

Q: Collision avoidance — if distance sensor pin 8 reads less than 5.0, reduce throttle pin 5
A: {"name":"collision_avoid","intent":"Reduce speed when obstacle detected within 5m","body":[{"op":"READ_PIN","arg":8},{"op":"PUSH_F32","value":5.0},{"op":"LT_F"},{"op":"JUMP_IF_FALSE","target":"safe"},{"op":"PUSH_F32","value":10.0},{"op":"CLAMP_F","lo":-100.0,"hi":100.0},{"op":"WRITE_PIN","arg":5,"label":"safe"},{"op":"NOP","flags":"0x80","operand1":1,"operand2":1}]}

Q: Emergency stop all actuators
A: {"name":"estop","intent":"Immediate full stop","body":[{"op":"PUSH_F32","value":0.0},{"op":"CLAMP_F","lo":-100.0,"hi":100.0},{"op":"WRITE_PIN","arg":5},{"op":"PUSH_F32","value":0.0},{"op":"CLAMP_F","lo":-90.0,"hi":90.0},{"op":"WRITE_PIN","arg":4},{"op":"NOP","flags":"0x80","operand1":1,"operand2":1}]}

Q: Station keeping at position (10.5, 20.3) with drift compensation
A: {"name":"station_keep","intent":"Hold position near (10.5, 20.3)","body":[{"op":"READ_PIN","arg":9},{"op":"PUSH_F32","value":10.5},{"op":"SUB_F"},{"op":"PUSH_F32","value":2.0},{"op":"MUL_F"},{"op":"CLAMP_F","lo":-30.0,"hi":30.0},{"op":"WRITE_PIN","arg":4},{"op":"READ_PIN","arg":10},{"op":"PUSH_F32","value":20.3},{"op":"SUB_F"},{"op":"PUSH_F32","value":2.0},{"op":"MUL_F"},{"op":"CLAMP_F","lo":-50.0,"hi":50.0},{"op":"WRITE_PIN","arg":5},{"op":"NOP","flags":"0x80","operand1":1,"operand2":1}]}

Q: If heading error exceeds 10 degrees, declare intent "course_correction"
A: {"name":"course_correction","intent":"Declare intent on large heading error","body":[{"op":"READ_PIN","arg":2},{"op":"PUSH_F32","value":45.0},{"op":"SUB_F"},{"op":"ABS_F"},{"op":"PUSH_F32","value":10.0},{"op":"GT_F"},{"op":"JUMP_IF_FALSE","target":"done"},{"op":"DECLARE_INTENT","label":"declared"},{"op":"NOP","flags":"0x80","operand1":1,"operand2":1,"label":"done"}]}
"""
    return PromptTemplate(
        strategy=PromptStrategy.FEWSHOT,
        system_prompt=system_prompt,
        trust_level=5,
        opcode_count=0,
        safety_rule_count=3,
        example_count=7,
    )


# ===================================================================
# Module-level instances
# ===================================================================

strategy_a_comprehensive: PromptTemplate = _build_strategy_a()
strategy_b_grammar: PromptTemplate = _build_strategy_b()
strategy_c_fewshot: PromptTemplate = _build_strategy_c()

# Ordered by preference (best first) — Strategy A is most comprehensive
_STRATEGIES_ORDERED: list[PromptTemplate] = [
    strategy_a_comprehensive,
    strategy_c_fewshot,
    strategy_b_grammar,
]


def best_prompt(trust_level: int = 5) -> PromptTemplate:
    """Return the best available prompt template, adjusted for trust level.

    Args:
        trust_level: Current trust level (0-5).  The template text
                     references the trust level for gating context.

    Returns:
        A ``PromptTemplate`` ready for use with an LLM.
    """
    # Pick the best strategy (A by default)
    template = _STRATEGIES_ORDERED[0]
    # Adjust trust level in the metadata
    if template.trust_level != trust_level:
        # For now we return the template as-is; trust-level-specific
        # filtering is handled by the safety pipeline, not the prompt.
        pass
    return template


def all_strategies() -> dict[PromptStrategy, PromptTemplate]:
    """Return all registered prompt strategies."""
    return {
        s.strategy: s
        for s in _STRATEGIES_ORDERED
    }
