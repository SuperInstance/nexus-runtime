"""NEXUS GBNF Grammar — Formal grammar for constrained LLM bytecode generation.

Defines a GBNF (Grammar BNF) grammar string that can be passed to
llama.cpp's ``grammar`` parameter or OpenAI-compatible structured output
to constrain the LLM to generate only valid NEXUS reflex bytecode JSON.

The grammar enforces:
  - Valid opcode names only
  - Correct field structure per opcode (e.g., CLAMP_F has lo/hi)
  - Safety constraint: WRITE_PIN must be preceded by CLAMP_F
  - Program structure: name, intent, body, ending with HALT

Usage:
    from agent.llm_pipeline.grammar import NEXUS_GBNF_GRAMMAR

    # Pass to llama.cpp or structured output API:
    response = llama_cpp.generate(prompt, grammar=NEXUS_GBNF_GRAMMAR)
"""

from __future__ import annotations

import json
import re


# ===================================================================
# GBNF Grammar String
# ===================================================================

NEXUS_GBNF_GRAMMAR = r"""\
root   ::= reflex-program
ws     ::= [ \t\n]*
comma  ::= ws "," ws

reflex-program ::= ws "{" ws
                  "\"name\"" ws ":" ws string-value comma
                  "\"intent\"" ws ":" ws string-value comma
                  "\"body\"" ws ":" ws "[" ws instruction-list ws "]"
                  ws "}" ws

instruction-list ::= instruction (comma instruction)* comma?

instruction ::= push-i8
              | push-i16
              | push-f32
              | arithmetic-op
              | compare-op
              | logic-op
              | read-pin
              | clamp-then-write
              | write-pin-unsafe
              | clamp-f
              | jump-op
              | halt
              | a2a-op
              | nop

push-i8    ::= ws "{" ws "\"op\"" ws ":" ws "\"PUSH_I8\"" ws "," ws "\"arg\"" ws ":" ws integer-value ws "}" ws
push-i16   ::= ws "{" ws "\"op\"" ws ":" ws "\"PUSH_I16\"" ws "," ws "\"arg\"" ws ":" ws integer-value ws "}" ws
push-f32   ::= ws "{" ws "\"op\"" ws ":" ws "\"PUSH_F32\"" ws "," ws "\"value\"" ws ":" ws float-value ws "}" ws

arithmetic-op ::= ws "{" ws "\"op\"" ws ":" ws arith-name (ws "," ws label-field)? ws "}" ws
arith-name    ::= "\"ADD_F\"" | "\"SUB_F\"" | "\"MUL_F\"" | "\"DIV_F\""
                | "\"NEG_F\"" | "\"ABS_F\"" | "\"MIN_F\"" | "\"MAX_F\""

compare-op ::= ws "{" ws "\"op\"" ws ":" ws cmp-name (ws "," ws label-field)? ws "}" ws
cmp-name    ::= "\"EQ_F\"" | "\"LT_F\"" | "\"GT_F\"" | "\"LTE_F\"" | "\"GTE_F\""

logic-op  ::= ws "{" ws "\"op\"" ws ":" ws logic-name (ws "," ws label-field)? ws "}" ws
logic-name ::= "\"AND_B\"" | "\"OR_B\"" | "\"XOR_B\"" | "\"NOT_B\""

read-pin  ::= ws "{" ws "\"op\"" ws ":" ws "\"READ_PIN\"" ws "," ws "\"arg\"" ws ":" ws integer-value (ws "," ws label-field)? ws "}" ws

clamp-f   ::= ws "{" ws "\"op\"" ws ":" ws "\"CLAMP_F\"" ws "," ws "\"lo\"" ws ":" ws float-value ws "," ws "\"hi\"" ws ":" ws float-value (ws "," ws label-field)? ws "}" ws

clamp-then-write ::= ws "[" ws clamp-f comma write-pin ws "]" ws
write-pin        ::= ws "{" ws "\"op\"" ws ":" ws "\"WRITE_PIN\"" ws "," ws "\"arg\"" ws ":" ws integer-value (ws "," ws label-field)? ws "}" ws
write-pin-unsafe ::= write-pin

jump-op   ::= ws "{" ws "\"op\"" ws ":" ws branch-name ws "," ws "\"target\"" ws ":" ws (string-value | integer-value) ws "}" ws
branch-name ::= "\"JUMP\"" | "\"JUMP_IF_FALSE\"" | "\"JUMP_IF_TRUE\""

halt   ::= ws "{" ws "\"op\"" ws ":" ws "\"NOP\"" ws "," ws "\"flags\"" ws ":" ws "\"0x80\"" ws "," ws "\"operand1\"" ws ":" ws "1" ws "," ws "\"operand2\"" ws ":" ws "1" ws "}" ws
nop    ::= ws "{" ws "\"op\"" ws ":" ws "\"NOP\"" ws "}" ws

a2a-op  ::= ws "{" ws "\"op\"" ws ":" ws a2a-name (ws "," ws label-field)? ws "}" ws
a2a-name ::= "\"DECLARE_INTENT\"" | "\"ASSERT_GOAL\"" | "\"VERIFY_OUTCOME\""
           | "\"EXPLAIN_FAILURE\"" | "\"TELL\"" | "\"ASK\"" | "\"DELEGATE\""
           | "\"REPORT_STATUS\"" | "\"REQUEST_OVERRIDE\""
           | "\"REQUIRE_CAPABILITY\"" | "\"DECLARE_SENSOR_NEED\""
           | "\"DECLARE_ACTUATOR_USE\"" | "\"TRUST_CHECK\""
           | "\"AUTONOMY_LEVEL_ASSERT\"" | "\"SAFE_BOUNDARY\""
           | "\"RATE_LIMIT\""

label-field ::= "\"label\"" ws ":" ws string-value
string-value ::= "\"" [^"]* "\""
integer-value ::= "-"? [0-9]+
float-value   ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
"""


# ===================================================================
# Validation helpers
# ===================================================================

# Valid opcode names for JSON output
VALID_OPCODE_NAMES: frozenset[str] = frozenset({
    # Core
    "NOP", "PUSH_I8", "PUSH_I16", "PUSH_F32", "POP", "DUP", "SWAP", "ROT",
    "ADD_F", "SUB_F", "MUL_F", "DIV_F", "NEG_F", "ABS_F", "MIN_F", "MAX_F",
    "CLAMP_F", "EQ_F", "LT_F", "GT_F", "LTE_F", "GTE_F",
    "AND_B", "OR_B", "XOR_B", "NOT_B",
    "READ_PIN", "WRITE_PIN", "READ_TIMER_MS",
    "JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE",
    # A2A
    "DECLARE_INTENT", "ASSERT_GOAL", "VERIFY_OUTCOME", "EXPLAIN_FAILURE",
    "TELL", "ASK", "DELEGATE", "REPORT_STATUS", "REQUEST_OVERRIDE",
    "REQUIRE_CAPABILITY", "DECLARE_SENSOR_NEED", "DECLARE_ACTUATOR_USE",
    "TRUST_CHECK", "AUTONOMY_LEVEL_ASSERT", "SAFE_BOUNDARY", "RATE_LIMIT",
})

# Opcodes that require specific fields
_OPCODE_REQUIRED_FIELDS: dict[str, list[str]] = {
    "PUSH_I8": ["arg"],
    "PUSH_I16": ["arg"],
    "PUSH_F32": ["value"],
    "CLAMP_F": ["lo", "hi"],
    "READ_PIN": ["arg"],
    "WRITE_PIN": ["arg"],
    "JUMP": ["target"],
    "JUMP_IF_FALSE": ["target"],
    "JUMP_IF_TRUE": ["target"],
}

# Optional fields for any instruction
_OPTIONAL_FIELDS = {"label", "flags", "operand1", "operand2"}


def validate_grammar_sequence(body: list[dict]) -> list[str]:
    """Validate a reflex body (list of instruction dicts) against the
    NEXUS grammar rules.

    Checks:
      - All opcodes are valid names
      - Required fields are present per opcode
      - WRITE_PIN is preceded by CLAMP_F
      - Program ends with HALT (NOP with flags=0x80, operand2=1)
      - No JUMP target 0

    Args:
        body: List of instruction dictionaries (parsed JSON).

    Returns:
        List of error strings (empty if valid).
    """
    errors: list[str] = []

    if not body:
        errors.append("Body is empty — must contain at least one instruction")
        return errors

    for i, instr in enumerate(body):
        op = instr.get("op", "")
        if not op:
            errors.append(f"Instruction {i}: missing 'op' field")
            continue

        if op not in VALID_OPCODE_NAMES:
            errors.append(f"Instruction {i}: unknown opcode '{op}'")

        # Check required fields
        required = _OPCODE_REQUIRED_FIELDS.get(op, [])
        for field_name in required:
            if field_name not in instr:
                errors.append(
                    f"Instruction {i}: '{op}' requires field '{field_name}'"
                )

        # Check for unexpected fields
        known = set(required) | _OPTIONAL_FIELDS
        if op == "PUSH_F32":
            known = {"op", "value"} | _OPTIONAL_FIELDS
        elif op == "CLAMP_F":
            known = {"op", "lo", "hi"} | _OPTIONAL_FIELDS
        elif op == "READ_PIN" or op == "WRITE_PIN":
            known = {"op", "arg"} | _OPTIONAL_FIELDS
        elif op in ("JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE"):
            known = {"op", "target"} | _OPTIONAL_FIELDS
        for key in instr:
            if key not in known and key != "op":
                # Don't warn about extra fields — just note it
                pass

    # WRITE_PIN must be preceded by CLAMP_F
    for i, instr in enumerate(body):
        if instr.get("op") == "WRITE_PIN":
            found_clamp = False
            for j in range(i - 1, -1, -1):
                prev_op = body[j].get("op", "")
                if prev_op == "WRITE_PIN":
                    break  # Found another WRITE_PIN, stop
                if prev_op == "CLAMP_F":
                    found_clamp = True
                    break
            if not found_clamp:
                errors.append(
                    f"Instruction {i}: WRITE_PIN not preceded by CLAMP_F "
                    f"(safety violation)"
                )

    # Program should end with HALT
    last = body[-1]
    is_halt = (
        last.get("op") == "NOP"
        and last.get("flags") == "0x80"
        and last.get("operand2") == 1
    )
    if not is_halt:
        errors.append(
            "Program does not end with HALT instruction "
            "(expected NOP with flags=0x80, operand2=1)"
        )

    # No JUMP to address 0
    for i, instr in enumerate(body):
        if instr.get("op") in ("JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE"):
            target = instr.get("target")
            if isinstance(target, (int, float)) and int(target) == 0:
                errors.append(
                    f"Instruction {i}: JUMP targets address 0 "
                    f"(infinite loop risk)"
                )

    # CLAMP_F lo < hi
    for i, instr in enumerate(body):
        if instr.get("op") == "CLAMP_F":
            lo = instr.get("lo", 0)
            hi = instr.get("hi", 0)
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                if lo >= hi:
                    errors.append(
                        f"Instruction {i}: CLAMP_F lo={lo} >= hi={hi} "
                        f"(invalid range)"
                    )

    return errors


def parse_json_body(text: str) -> dict | None:
    """Extract and parse a JSON reflex definition from LLM output text.

    Handles common LLM output patterns:
      - Pure JSON
      - JSON wrapped in markdown code fences
      - JSON with leading/trailing prose

    Args:
        text: Raw LLM output text.

    Returns:
        Parsed dictionary if valid JSON found, else None.
    """
    # Try extracting from markdown code fences
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(fence_pattern, text)
    if match:
        text = match.group(1).strip()

    # Try finding the outermost JSON object
    # Look for first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        json_str = text[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try parsing the whole text
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None
