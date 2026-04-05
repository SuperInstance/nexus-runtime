"""NEXUS Reflex Compiler - JSON-to-bytecode compilation.

Compiles human-readable reflex definitions (JSON) into
8-byte fixed-length bytecode binary for the ESP32 VM.

Compilation pipeline: JSON -> Parse -> Validate -> Assemble -> Emit -> Verify

Input format:
{
    "name": "heading_hold",
    "intent": "Maintain heading 270 degrees",
    "sensors": ["compass_heading", "wind_speed"],
    "actuators": ["rudder_angle"],
    "trust_min": 0.50,
    "author": "human",
    "body": [
        {"op": "READ_PIN", "arg": 0},
        {"op": "PUSH_F32", "value": 270.0},
        {"op": "SUB_F"},
        {"op": "CLAMP_F", "lo": -30.0, "hi": 30.0},
        {"op": "WRITE_PIN", "arg": 0},
        {"op": "NOP", "flags": "0x80", "operand1": 1}
    ]
}
"""

from __future__ import annotations

from reflex_compiler.bytecode_emitter import (
    BytecodeEmitter,
    INSTR_SIZE,
    FLAGS_SYSCALL,
    float_to_u32,
    pack_instruction,
)
from reflex_compiler.safety_validator import SafetyValidator

# Valid core opcode names
VALID_OPCODES: set[str] = {
    "NOP", "PUSH_I8", "PUSH_I16", "PUSH_F32", "POP", "DUP", "SWAP", "ROT",
    "ADD_F", "SUB_F", "MUL_F", "DIV_F", "NEG_F", "ABS_F", "MIN_F", "MAX_F",
    "CLAMP_F", "EQ_F", "LT_F", "GT_F", "LTE_F", "GTE_F",
    "AND_B", "OR_B", "XOR_B", "NOT_B",
    "READ_PIN", "WRITE_PIN", "READ_TIMER_MS",
    "JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE",
    # A2A opcodes (valid in reflex code)
    "DECLARE_INTENT", "ASSERT_GOAL", "VERIFY_OUTCOME", "EXPLAIN_FAILURE",
    "SET_PRIORITY", "REQUEST_RESOURCE", "RELEASE_RESOURCE",
    "TELL", "ASK", "DELEGATE", "REPORT_STATUS", "REQUEST_OVERRIDE",
    "REQUIRE_CAPABILITY", "DECLARE_SENSOR_NEED", "DECLARE_ACTUATOR_USE",
    "CHECK_AVAILABILITY", "RESERVE_RESOURCE",
    "TRUST_CHECK", "AUTONOMY_LEVEL_ASSERT", "SAFE_BOUNDARY", "RATE_LIMIT",
    "EMERGENCY_CLAIM", "RELEASE_CLAIM", "VERIFY_AUTHORITY",
}

# Opcode name -> value mapping (core + A2A)
OPCODE_MAP: dict[str, int] = {
    # Core
    "NOP": 0x00, "PUSH_I8": 0x01, "PUSH_I16": 0x02, "PUSH_F32": 0x03,
    "POP": 0x04, "DUP": 0x05, "SWAP": 0x06, "ROT": 0x07,
    "ADD_F": 0x08, "SUB_F": 0x09, "MUL_F": 0x0A, "DIV_F": 0x0B,
    "NEG_F": 0x0C, "ABS_F": 0x0D, "MIN_F": 0x0E, "MAX_F": 0x0F,
    "CLAMP_F": 0x10,
    "EQ_F": 0x11, "LT_F": 0x12, "GT_F": 0x13, "LTE_F": 0x14, "GTE_F": 0x15,
    "AND_B": 0x16, "OR_B": 0x17, "XOR_B": 0x18, "NOT_B": 0x19,
    "READ_PIN": 0x1A, "WRITE_PIN": 0x1B, "READ_TIMER_MS": 0x1C,
    "JUMP": 0x1D, "JUMP_IF_FALSE": 0x1E, "JUMP_IF_TRUE": 0x1F,
    # A2A Intent
    "DECLARE_INTENT": 0x20, "ASSERT_GOAL": 0x21, "VERIFY_OUTCOME": 0x22,
    "EXPLAIN_FAILURE": 0x23, "SET_PRIORITY": 0x24, "REQUEST_RESOURCE": 0x25,
    "RELEASE_RESOURCE": 0x26,
    # A2A Communication
    "TELL": 0x30, "ASK": 0x31, "DELEGATE": 0x32,
    "REPORT_STATUS": 0x33, "REQUEST_OVERRIDE": 0x34,
    # A2A Capability
    "REQUIRE_CAPABILITY": 0x40, "DECLARE_SENSOR_NEED": 0x41,
    "DECLARE_ACTUATOR_USE": 0x42, "CHECK_AVAILABILITY": 0x43,
    "RESERVE_RESOURCE": 0x44,
    # A2A Safety
    "TRUST_CHECK": 0x50, "AUTONOMY_LEVEL_ASSERT": 0x51,
    "SAFE_BOUNDARY": 0x52, "RATE_LIMIT": 0x53,
    "EMERGENCY_CLAIM": 0x54, "RELEASE_CLAIM": 0x55, "VERIFY_AUTHORITY": 0x56,
}

MAX_CYCLE_BUDGET = 1000
MAX_STACK_DEPTH = 16


def _parse_int(value) -> int:
    """Parse an integer from various formats (int, str like '0x80', etc.)."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.startswith("0x") or value.startswith("0X"):
            return int(value, 16)
        return int(value)
    return int(value)


class ReflexCompiler:
    """JSON-to-bytecode compiler."""

    def __init__(self) -> None:
        self._emitter = BytecodeEmitter()
        self._validator = SafetyValidator()
        self._labels: dict[str, int] = {}  # label -> instruction index
        self._label_refs: list[tuple[int, str]] = []  # (instr_index, label_name)

    def compile(self, reflex_json: dict) -> bytes:
        """Compile a reflex JSON definition to bytecode.

        Args:
            reflex_json: Reflex definition as a dictionary.

        Returns:
            Compiled bytecode bytes (multiple of 8 bytes).

        Raises:
            ValueError: If compilation fails.
        """
        # Validate first
        errors = self.validate(reflex_json)
        if errors:
            raise ValueError(f"Validation failed: {'; '.join(errors)}")

        body = reflex_json.get("body", [])
        if not body:
            return b""

        self._emitter.reset()
        self._labels.clear()
        self._label_refs.clear()

        # First pass: collect labels
        for i, instr in enumerate(body):
            label = instr.get("label")
            if label:
                self._labels[label] = i

        # Second pass: emit instructions
        for i, instr in enumerate(body):
            self._emit_instruction(instr, i, len(body))

        bytecode = self._emitter.get_bytecode()

        # Verify
        verify_errors = self._validator.validate_bytecode(bytecode)
        if verify_errors:
            raise ValueError(f"Verification failed: {'; '.join(verify_errors)}")

        return bytecode

    def validate(self, reflex_json: dict) -> list[str]:
        """Validate a reflex JSON definition.

        Args:
            reflex_json: Reflex definition as a dictionary.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []

        # Check name
        if "name" not in reflex_json:
            errors.append("Missing 'name' field")

        # Check body
        body = reflex_json.get("body")
        if body is None:
            errors.append("Missing 'body' field")
            return errors

        if not isinstance(body, list):
            errors.append("'body' must be a list")
            return errors

        if len(body) == 0:
            errors.append("'body' is empty")
            return errors

        if len(body) > MAX_CYCLE_BUDGET:
            errors.append(f"Body has {len(body)} instructions, exceeds cycle budget of {MAX_CYCLE_BUDGET}")

        # Validate each instruction
        for i, instr in enumerate(body):
            op_name = instr.get("op")
            if not op_name:
                errors.append(f"Instruction {i}: missing 'op' field")
                continue

            if op_name not in VALID_OPCODES:
                errors.append(f"Instruction {i}: unknown opcode '{op_name}'")

            # Validate CLAMP_F has lo/hi
            if op_name == "CLAMP_F":
                if "lo" not in instr or "hi" not in instr:
                    errors.append(f"Instruction {i}: CLAMP_F requires 'lo' and 'hi' fields")

        return errors

    def _emit_instruction(self, instr: dict, index: int, total: int) -> None:
        """Emit a single instruction from JSON definition."""
        op_name = instr.get("op", "NOP")
        opcode = OPCODE_MAP.get(op_name, 0x00)
        arg = instr.get("arg")
        value = instr.get("value")
        flags = _parse_int(instr.get("flags", 0))
        operand1 = _parse_int(instr.get("operand1", 0))
        lo = instr.get("lo")
        hi = instr.get("hi")

        if op_name == "PUSH_F32" and value is not None:
            self._emitter.emit_push_f32(float(value))
        elif op_name == "PUSH_I8" and arg is not None:
            self._emitter.emit_push_i8(int(arg))
        elif op_name == "PUSH_I16" and arg is not None:
            self._emitter.emit_push_i16(int(arg))
        elif op_name == "CLAMP_F" and lo is not None and hi is not None:
            self._emitter.emit_clamp_f(float(lo), float(hi))
        elif op_name == "READ_PIN" and arg is not None:
            self._emitter.emit_read_pin(int(arg))
        elif op_name == "WRITE_PIN" and arg is not None:
            self._emitter.emit_write_pin(int(arg))
        elif op_name in ("JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE"):
            # Handle label references
            target = instr.get("target")
            if isinstance(target, str):
                # Label reference - resolve later, placeholder for now
                target_idx = self._labels.get(target, 0)
                self._label_refs.append((index, target))
                if op_name == "JUMP":
                    self._emitter.emit_jump(target_idx)
                elif op_name == "JUMP_IF_FALSE":
                    self._emitter.emit_jump_if_false(target_idx)
                else:
                    self._emitter.emit_jump_if_true(target_idx)
            elif arg is not None:
                if op_name == "JUMP":
                    self._emitter.emit_jump(int(arg))
                elif op_name == "JUMP_IF_FALSE":
                    self._emitter.emit_jump_if_false(int(arg))
                else:
                    self._emitter.emit_jump_if_true(int(arg))
            else:
                self._emitter.emit_raw(opcode, flags, operand1, 0)
        elif op_name == "NOP" and flags:
            # NOP with custom flags (e.g., HALT syscall)
            self._emitter.emit_raw(opcode, flags, operand1, _parse_int(instr.get("operand2", 0)))
        else:
            # Default: emit raw with given flags/operands
            self._emitter.emit_raw(opcode, flags, operand1, _parse_int(instr.get("operand2", 0)))
