"""NEXUS Jetson tests - Reflex compiler tests.

10+ tests covering JSON-to-bytecode compilation.
"""

import struct

import pytest

from reflex_compiler.compiler import ReflexCompiler
from reflex_compiler.bytecode_emitter import (
    BytecodeEmitter,
    INSTR_SIZE,
    FLAGS_SYSCALL,
    float_to_u32,
    pack_instruction,
    unpack_instruction,
)
from reflex_compiler.safety_validator import SafetyValidator


HEADING_HOLD_REFLEX = {
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
        {"op": "NOP", "flags": "0x80", "operand1": 1},
    ],
}

SIMPLE_ARITHMETIC = {
    "name": "simple_add",
    "body": [
        {"op": "PUSH_F32", "value": 10.0},
        {"op": "PUSH_F32", "value": 20.0},
        {"op": "ADD_F"},
        {"op": "PUSH_F32", "value": 2.0},
        {"op": "DIV_F"},
        {"op": "NOP", "flags": "0x80", "operand1": 1},
    ],
}

EMPTY_BODY = {"name": "empty", "body": []}

NO_CLAMP_REFLEX = {
    "name": "no_clamp",
    "body": [
        {"op": "READ_PIN", "arg": 0},
        {"op": "WRITE_PIN", "arg": 0},
        {"op": "NOP", "flags": "0x80", "operand1": 1},
    ],
}


class TestReflexCompiler:
    def test_compiler_exists(self) -> None:
        compiler = ReflexCompiler()
        assert compiler is not None

    def test_compile_heading_hold(self) -> None:
        compiler = ReflexCompiler()
        bytecode = compiler.compile(HEADING_HOLD_REFLEX)
        assert len(bytecode) == 6 * INSTR_SIZE
        assert len(bytecode) % INSTR_SIZE == 0

        opcode, flags, op1, op2 = unpack_instruction(bytecode, 0)
        assert opcode == 0x1A
        assert op1 == 0

        opcode, flags, op1, op2 = unpack_instruction(bytecode, 1 * INSTR_SIZE)
        assert opcode == 0x03
        value = struct.unpack("<f", struct.pack("<I", op2))[0]
        assert abs(value - 270.0) < 0.01

        opcode, _, _, _ = unpack_instruction(bytecode, 2 * INSTR_SIZE)
        assert opcode == 0x09

        opcode, _, _, op2 = unpack_instruction(bytecode, 3 * INSTR_SIZE)
        assert opcode == 0x10

        opcode, _, op1, _ = unpack_instruction(bytecode, 4 * INSTR_SIZE)
        assert opcode == 0x1B
        assert op1 == 0

        opcode, flags, _, _ = unpack_instruction(bytecode, 5 * INSTR_SIZE)
        assert opcode == 0x00
        assert flags & FLAGS_SYSCALL

    def test_compile_simple_arithmetic(self) -> None:
        compiler = ReflexCompiler()
        bytecode = compiler.compile(SIMPLE_ARITHMETIC)
        assert len(bytecode) == 6 * INSTR_SIZE

        opcode, _, _, op2 = unpack_instruction(bytecode, 0)
        assert opcode == 0x03
        value = struct.unpack("<f", struct.pack("<I", op2))[0]
        assert abs(value - 10.0) < 0.01

        opcode, _, _, _ = unpack_instruction(bytecode, 2 * INSTR_SIZE)
        assert opcode == 0x08

    def test_reject_invalid_opcode(self) -> None:
        compiler = ReflexCompiler()
        errors = compiler.validate({"name": "bad", "body": [{"op": "NONEXISTENT_OP"}]})
        assert len(errors) > 0
        assert any("unknown opcode" in e.lower() for e in errors)

    def test_reject_empty_body(self) -> None:
        compiler = ReflexCompiler()
        errors = compiler.validate(EMPTY_BODY)
        assert len(errors) > 0
        assert any("empty" in e.lower() for e in errors)

    def test_compile_empty_dict(self) -> None:
        compiler = ReflexCompiler()
        errors = compiler.validate({"name": "test", "body": []})
        assert len(errors) > 0

    def test_validate_empty(self) -> None:
        compiler = ReflexCompiler()
        errors = compiler.validate({})
        assert len(errors) > 0


class TestLabelResolution:
    def test_label_jump(self) -> None:
        reflex = {
            "name": "loop_test",
            "body": [
                {"op": "READ_PIN", "arg": 0, "label": "start"},
                {"op": "JUMP", "target": "start"},
                {"op": "NOP", "flags": "0x80", "operand1": 1},
            ],
        }
        compiler = ReflexCompiler()
        bytecode = compiler.compile(reflex)
        opcode, _, op1, _ = unpack_instruction(bytecode, 1 * INSTR_SIZE)
        assert opcode == 0x1D
        assert op1 == 0


class TestClampEnforcement:
    def test_clamp_before_write_passes(self) -> None:
        compiler = ReflexCompiler()
        validator = SafetyValidator()
        bytecode = compiler.compile(HEADING_HOLD_REFLEX)
        errors = validator.validate_bytecode(bytecode)
        clamp_errors = [e for e in errors if "CLAMP_F" in e]
        assert len(clamp_errors) == 0

    def test_missing_clamp_detected(self) -> None:
        """WRITE_PIN without preceding CLAMP_F should be caught by validator."""
        # Build bytecode manually (bypassing compiler's verification)
        emitter = BytecodeEmitter()
        emitter.emit_read_pin(0)
        emitter.emit_write_pin(0)
        emitter.emit_halt()
        bytecode = emitter.get_bytecode()

        validator = SafetyValidator()
        errors = validator.validate_bytecode(bytecode)
        clamp_errors = [e for e in errors if "CLAMP_F" in e]
        assert len(clamp_errors) > 0


class TestCycleBudget:
    def test_excessive_instructions_rejected(self) -> None:
        compiler = ReflexCompiler()
        validator = SafetyValidator(max_cycles=10)
        reflex = {"name": "too_long", "body": [{"op": "NOP"} for _ in range(15)]}
        bytecode = compiler.compile(reflex)
        errors = validator.validate_bytecode(bytecode)
        cycle_errors = [e for e in errors if "cycle budget" in e.lower()]
        assert len(cycle_errors) > 0


class TestStackDepth:
    def test_stack_depth_computation(self) -> None:
        validator = SafetyValidator()
        compiler = ReflexCompiler()
        bytecode = compiler.compile(HEADING_HOLD_REFLEX)
        depth = validator.check_stack_depth(bytecode)
        assert depth is not None
        assert depth > 0

    def test_stack_overflow_detected(self) -> None:
        validator = SafetyValidator(max_stack=2)
        bytecode = (
            pack_instruction(0x03, 0, 0, float_to_u32(1.0))
            + pack_instruction(0x03, 0, 0, float_to_u32(2.0))
            + pack_instruction(0x03, 0, 0, float_to_u32(3.0))
        )
        errors = validator.validate_bytecode(bytecode)
        stack_errors = [e for e in errors if "stack" in e.lower()]
        assert len(stack_errors) > 0

    def test_stack_underflow_detected(self) -> None:
        validator = SafetyValidator()
        bytecode = pack_instruction(0x04, 0, 0, 0)  # POP
        depth = validator.check_stack_depth(bytecode)
        assert depth is None


class TestJumpBounds:
    def test_out_of_bounds_jump(self) -> None:
        validator = SafetyValidator()
        bytecode = pack_instruction(0x1D, 0, 99, 0)  # JUMP 99
        errors = validator.validate_bytecode(bytecode)
        jump_errors = [e for e in errors if "Jump" in e]
        assert len(jump_errors) > 0


class TestBinaryOutput:
    def test_output_is_multiple_of_8(self) -> None:
        compiler = ReflexCompiler()
        bytecode = compiler.compile(HEADING_HOLD_REFLEX)
        assert len(bytecode) % 8 == 0
        assert len(bytecode) > 0

    def test_binary_encoding_correctness(self) -> None:
        compiler = ReflexCompiler()
        bytecode = compiler.compile(HEADING_HOLD_REFLEX)
        expected = pack_instruction(0x1A, 0x01, 0, 0)
        assert bytecode[0:8] == expected

    def test_nop_with_syscall_flag(self) -> None:
        emitter = BytecodeEmitter()
        emitter.emit_halt()
        bytecode = emitter.get_bytecode()
        opcode, flags, op1, op2 = unpack_instruction(bytecode, 0)
        assert opcode == 0x00
        assert flags & 0x80
        assert op2 == 1


class TestBytecodeEmitter:
    def test_initial_state(self) -> None:
        emitter = BytecodeEmitter()
        assert emitter.instruction_count() == 0
        assert emitter.get_bytecode() == b""

    def test_emit_nop(self) -> None:
        emitter = BytecodeEmitter()
        emitter.emit_nop()
        assert emitter.instruction_count() == 1
        assert len(emitter.get_bytecode()) == 8

    def test_emit_halt(self) -> None:
        emitter = BytecodeEmitter()
        emitter.emit_halt()
        bytecode = emitter.get_bytecode()
        assert len(bytecode) == 8
        assert bytecode[1] == 0x80

    def test_reset(self) -> None:
        emitter = BytecodeEmitter()
        emitter.emit_nop()
        emitter.reset()
        assert emitter.instruction_count() == 0
        assert emitter.get_bytecode() == b""

    def test_multiple_instructions(self) -> None:
        emitter = BytecodeEmitter()
        emitter.emit_nop()
        emitter.emit_push_f32(3.14)
        emitter.emit_add_f()
        assert emitter.instruction_count() == 3
        assert len(emitter.get_bytecode()) == 24
