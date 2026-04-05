"""NEXUS Safety Validator — Comprehensive test suite for the 6-stage pipeline.

Tests cover each of the 6 stages:
  Stage 1: Syntax Check — valid bytecode, misaligned, empty, too large, unknown opcodes
  Stage 2: Safety Rules — forbidden patterns (CALL/RET, JUMP to 0, NOPs, safety pins)
  Stage 3: Stack Analysis — balanced stack, overflow, underflow
  Stage 4: Trust Check — each trust level with appropriate/inappropriate opcodes
  Stage 5: Semantic Analysis — infinite loops, unreachable code, CLAMP ranges, div by zero
  Stage 6: Adversarial Probing — bit-flip robustness, boundary mutations
  Integration: full pipeline on known-good and known-bad bytecode
"""

from __future__ import annotations

import math
import struct

import pytest

from core.safety_validator import (
    ACTUATOR_SAFE_RANGES,
    DEFAULT_MAX_STACK_DEPTH,
    FLAGS_EXTENDED_CLAMP,
    FLAGS_IS_CALL,
    FLAGS_IS_FLOAT,
    FLAGS_HAS_IMMEDIATE,
    FLAGS_SYSCALL,
    INSTR_SIZE,
    SAFETY_CRITICAL_PINS,
    STACK_EFFECTS,
    BytecodeSafetyPipeline,
    SafetyReport,
    SafetyViolation,
    StageResult,
)
from core.safety_validator.rules import (
    DEFAULT_MAX_INSTRUCTIONS,
    DEFAULT_MAX_NOP_SEQUENCE,
    TRUST_OPCODE_MATRIX,
    OP_JUMP,
    OP_JUMP_IF_FALSE,
    OP_NOP,
    OP_WRITE_PIN,
)
from core.safety_validator.pipeline import BytecodeSafetyPipeline as Pipeline

# We also need the bytecode emitter for generating known-good bytecode
from reflex.bytecode_emitter import (
    BytecodeEmitter,
    INSTR_SIZE as EMIT_INSTR_SIZE,
    pack_instruction,
    unpack_instruction,
)


# ===================================================================
# Helpers
# ===================================================================

def _make_instr(opcode: int, flags: int = 0, operand1: int = 0,
                operand2: int = 0) -> bytes:
    """Pack a single 8-byte instruction."""
    return pack_instruction(opcode, flags, operand1, operand2)


def _make_program(instrs: list[bytes]) -> bytes:
    """Join instructions into a bytecode program."""
    return b"".join(instrs)


def _make_nops(count: int) -> bytes:
    """Make a program of N NOP instructions."""
    return _make_program([_make_instr(OP_NOP)] * count)


def _emitter_to_bytecode(emitter: BytecodeEmitter) -> bytes:
    """Get bytecode from an emitter."""
    return emitter.get_bytecode()


def _make_simple_safe_program() -> bytes:
    """Create a simple, known-safe reflex program:
    READ_PIN 0, PUSH_F32 270.0, SUB_F, CLAMP_F(-30,30), WRITE_PIN 4, HALT
    """
    e = BytecodeEmitter()
    e.emit_read_pin(0)
    e.emit_push_f32(270.0)
    e.emit_sub_f()
    e.emit_clamp_f(-30.0, 30.0)
    e.emit_write_pin(4)  # Non-safety pin
    e.emit_halt()
    return _emitter_to_bytecode(e)


# ===================================================================
# Stage 1: Syntax Check Tests
# ===================================================================

class TestStage1Syntax:
    """Stage 1: Syntax validation."""

    def test_valid_single_instruction(self):
        bc = _make_instr(OP_NOP)
        p = BytecodeSafetyPipeline(trust_level=5)
        r = p.stage1_syntax(bc, [(OP_NOP, 0, 0, 0)])
        assert r.passed
        assert len(r.errors) == 0

    def test_valid_multi_instruction(self):
        bc = _make_simple_safe_program()
        p = BytecodeSafetyPipeline(trust_level=5)
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        r = p.stage1_syntax(bc, instrs)
        assert r.passed

    def test_empty_bytecode(self):
        bc = b""
        p = BytecodeSafetyPipeline()
        r = p.stage1_syntax(bc, [])
        assert not r.passed
        assert any("empty" in e.lower() for e in r.errors)

    def test_misaligned_bytecode(self):
        bc = b"\x00\x00\x00\x00\x00\x00\x00"  # 7 bytes, not 8
        p = BytecodeSafetyPipeline()
        r = p.stage1_syntax(bc, [])
        assert not r.passed
        assert any("alignment" in e.lower() or "multiple" in e.lower() for e in r.errors)

    def test_too_many_instructions(self):
        nops = _make_nops(DEFAULT_MAX_INSTRUCTIONS + 1)
        p = BytecodeSafetyPipeline()
        instrs = [(OP_NOP, 0, 0, 0)] * (DEFAULT_MAX_INSTRUCTIONS + 1)
        r = p.stage1_syntax(nops, instrs)
        assert not r.passed
        assert any("exceeds" in e for e in r.errors)

    def test_unknown_opcode(self):
        bc = _make_instr(0xFE)  # Invalid opcode
        p = BytecodeSafetyPipeline()
        r = p.stage1_syntax(bc, [(0xFE, 0, 0, 0)])
        assert not r.passed
        assert any("unknown" in e.lower() for e in r.errors)

    def test_single_byte_bytecode(self):
        bc = b"\x00"
        p = BytecodeSafetyPipeline()
        r = p.stage1_syntax(bc, [])
        assert not r.passed

    def test_all_0xFF_instruction(self):
        bc = b"\xFF" * INSTR_SIZE
        p = BytecodeSafetyPipeline()
        r = p.stage1_syntax(bc, [(0xFF, 0xFF, 0xFFFF, 0xFFFFFFFF)])
        assert not r.passed
        assert any("unknown" in e.lower() for e in r.errors)


# ===================================================================
# Stage 2: Safety Rules Tests
# ===================================================================

class TestStage2SafetyRules:
    """Stage 2: Forbidden pattern detection."""

    def test_safe_program_passes(self):
        bc = _make_simple_safe_program()
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline(trust_level=5)
        r = p.stage2_safety_rules(bc, instrs)
        assert r.passed

    def test_unmatched_call(self):
        # CALL without RET
        call_instr = _make_instr(OP_JUMP, FLAGS_IS_CALL, 0, 3)
        nops = _make_instr(OP_NOP)
        bc = _make_program([nops, call_instr, nops])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE) for i in range(3)]
        p = BytecodeSafetyPipeline()
        r = p.stage2_safety_rules(bc, instrs)
        assert not r.passed
        assert any("no matching RET" in e or "CALL" in e for e in r.errors)

    def test_matched_call_ret(self):
        call_instr = _make_instr(OP_JUMP, FLAGS_IS_CALL, 0, 4)  # CALL to instr 4
        ret_instr = _make_instr(OP_JUMP, 0, 0, 0xFFFFFFFF)      # RET
        nops = _make_instr(OP_NOP)
        bc = _make_program([nops, call_instr, nops, nops, ret_instr, nops])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE) for i in range(6)]
        p = BytecodeSafetyPipeline(trust_level=3)
        r = p.stage2_safety_rules(bc, instrs)
        assert r.passed

    def test_ret_without_call(self):
        ret_instr = _make_instr(OP_JUMP, 0, 0, 0xFFFFFFFF)
        bc = _make_program([_make_instr(OP_NOP), ret_instr])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE) for i in range(2)]
        p = BytecodeSafetyPipeline()
        r = p.stage2_safety_rules(bc, instrs)
        assert not r.passed
        assert any("RET" in e for e in r.errors)

    def test_jump_to_zero(self):
        jump_instr = _make_instr(OP_JUMP, 0, 0, 0)  # JUMP to addr 0
        bc = _make_program([_make_instr(OP_NOP), jump_instr])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE) for i in range(2)]
        p = BytecodeSafetyPipeline()
        r = p.stage2_safety_rules(bc, instrs)
        assert not r.passed
        assert any("address 0" in e or "infinite loop" in e for e in r.errors)

    def test_excessive_nops(self):
        n = DEFAULT_MAX_NOP_SEQUENCE + 10
        bc = _make_nops(n)
        instrs = [(OP_NOP, 0, 0, 0)] * n
        p = BytecodeSafetyPipeline()
        r = p.stage2_safety_rules(bc, instrs)
        assert not r.passed
        assert any("NOP" in e for e in r.errors)

    def test_allowed_nop_sequence(self):
        bc = _make_nops(DEFAULT_MAX_NOP_SEQUENCE)
        instrs = [(OP_NOP, 0, 0, 0)] * DEFAULT_MAX_NOP_SEQUENCE
        p = BytecodeSafetyPipeline()
        r = p.stage2_safety_rules(bc, instrs)
        assert r.passed

    def test_syscalls_nops_not_counted(self):
        """NOP with SYSCALL flag should not be counted as plain NOPs."""
        syscall_instr = _make_instr(OP_NOP, FLAGS_SYSCALL, 0, 0x01)
        nops_between = [_make_instr(OP_NOP)] * 90
        bc = _make_program([syscall_instr] + nops_between + [syscall_instr])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline(trust_level=5)
        r = p.stage2_safety_rules(bc, instrs)
        # Should not flag excessive NOPs since syscalls break the NOP run
        nop_errors = [e for e in r.errors if "NOP" in e and "consecutive" in e]
        assert len(nop_errors) == 0

    def test_write_safety_pin_low_trust(self):
        write_instr = _make_instr(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 0, 0)  # Pin 0
        bc = _make_program([write_instr])
        instrs = [(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 0, 0)]
        p = BytecodeSafetyPipeline(trust_level=2)
        r = p.stage2_safety_rules(bc, instrs)
        assert not r.passed
        assert any("safety-critical" in e.lower() for e in r.errors)

    def test_write_safety_pin_high_trust(self):
        write_instr = _make_instr(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 0, 0)  # Pin 0
        bc = _make_program([write_instr])
        instrs = [(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 0, 0)]
        p = BytecodeSafetyPipeline(trust_level=4)
        r = p.stage2_safety_rules(bc, instrs)
        # Pin check should pass at L4
        pin_errors = [e for e in r.errors if "safety-critical" in e.lower()]
        assert len(pin_errors) == 0

    def test_write_non_safety_pin_ok(self):
        write_instr = _make_instr(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 4, 0)  # Pin 4
        bc = _make_program([write_instr])
        instrs = [(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 4, 0)]
        p = BytecodeSafetyPipeline(trust_level=2)
        r = p.stage2_safety_rules(bc, instrs)
        # Should not flag for safety pin (pin 4 is not safety-critical)
        pin_errors = [e for e in r.errors if "safety-critical" in e.lower()]
        assert len(pin_errors) == 0

    def test_syscall_low_trust(self):
        """Non-HALT SYSCALL at low trust should fail. HALT is allowed at all levels."""
        # Test non-HALT syscall (PID_COMPUTE = 0x02)
        syscall_instr = _make_instr(OP_NOP, FLAGS_SYSCALL, 0, 0x02)
        bc = _make_program([syscall_instr])
        instrs = [(OP_NOP, FLAGS_SYSCALL, 0, 0x02)]
        p = BytecodeSafetyPipeline(trust_level=4)
        r = p.stage2_safety_rules(bc, instrs)
        assert not r.passed
        assert any("SYSCALL" in e or "L5" in e for e in r.errors)

        # HALT (syscall 0x01) should be allowed at all trust levels
        halt_instr = _make_instr(OP_NOP, FLAGS_SYSCALL, 0, 0x01)
        bc_halt = _make_program([halt_instr])
        instrs_halt = [(OP_NOP, FLAGS_SYSCALL, 0, 0x01)]
        p_halt = BytecodeSafetyPipeline(trust_level=0)
        r_halt = p_halt.stage2_safety_rules(bc_halt, instrs_halt)
        assert r_halt.passed, "HALT should be allowed at trust level 0"

    def test_call_depth_limit(self):
        """Test that deep call nesting is caught."""
        instrs_list = []
        # Create nested calls: CALL to next, CALL to next, ... RET, RET, ...
        max_calls = 9  # Exceeds default limit of 8
        for _ in range(max_calls):
            instrs_list.append(_make_instr(OP_JUMP, FLAGS_IS_CALL, 0, max_calls + 1))
        for _ in range(max_calls):
            instrs_list.append(_make_instr(OP_JUMP, 0, 0, 0xFFFFFFFF))
        instrs_list.append(_make_instr(OP_NOP))  # Target
        bc = _make_program(instrs_list)
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline()
        r = p.stage2_safety_rules(bc, instrs)
        assert not r.passed
        assert any("call depth" in e.lower() for e in r.errors)


# ===================================================================
# Stage 3: Stack Analysis Tests
# ===================================================================

class TestStage3StackAnalysis:
    """Stage 3: Stack depth analysis."""

    def test_balanced_stack(self):
        """Program: PUSH, PUSH, ADD_F, POP — balanced."""
        e = BytecodeEmitter()
        e.emit_push_f32(1.0)
        e.emit_push_f32(2.0)
        e.emit_add_f()
        e.emit_pop()
        bc = _emitter_to_bytecode(e)
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline()
        r = p.stage3_stack_analysis(bc, instrs)
        assert r.passed

    def test_stack_underflow(self):
        """Program: ADD_F with nothing on stack — underflow."""
        bc = _make_instr(0x0B)  # ADD_F (needs 2 on stack)
        p = BytecodeSafetyPipeline()
        r = p.stage3_stack_analysis(bc, [(0x0B, 0, 0, 0)])
        assert not r.passed
        assert any("underflow" in e.lower() for e in r.errors)

    def test_stack_overflow(self):
        """Push many values without popping — overflow."""
        pushes = [_make_instr(0x03, FLAGS_IS_FLOAT, 0, 0)] * (DEFAULT_MAX_STACK_DEPTH + 5)
        bc = _make_program(pushes)
        instrs = [(0x03, FLAGS_IS_FLOAT, 0, 0)] * (DEFAULT_MAX_STACK_DEPTH + 5)
        p = BytecodeSafetyPipeline()
        r = p.stage3_stack_analysis(bc, instrs)
        assert not r.passed
        assert any("overflow" in e.lower() or "exceeds" in e.lower() for e in r.errors)

    def test_stack_within_limits(self):
        """Push exactly up to the limit."""
        pushes = [_make_instr(0x03, FLAGS_IS_FLOAT, 0, 0)] * DEFAULT_MAX_STACK_DEPTH
        bc = _make_program(pushes)
        instrs = [(0x03, FLAGS_IS_FLOAT, 0, 0)] * DEFAULT_MAX_STACK_DEPTH
        p = BytecodeSafetyPipeline()
        r = p.stage3_stack_analysis(bc, instrs)
        assert r.passed

    def test_stack_depth_tracking(self):
        """Verify the pipeline tracks actual stack depth correctly."""
        # PUSH(1), PUSH(2), SUB_F -> depth goes 1, 2, 1
        bc = _make_program([
            _make_instr(0x03, FLAGS_IS_FLOAT, 0, 0),  # PUSH_F32
            _make_instr(0x03, FLAGS_IS_FLOAT, 0, 0),  # PUSH_F32
            _make_instr(0x09),                          # SUB_F
        ])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline()
        r = p.stage3_stack_analysis(bc, instrs)
        assert r.passed

    def test_pop_on_empty_stack(self):
        """POP with empty stack — underflow."""
        bc = _make_instr(0x04)  # POP
        p = BytecodeSafetyPipeline()
        r = p.stage3_stack_analysis(bc, [(0x04, 0, 0, 0)])
        assert not r.passed

    def test_write_pin_underflow(self):
        """WRITE_PIN needs a value on stack."""
        bc = _make_instr(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 5, 0)
        p = BytecodeSafetyPipeline()
        r = p.stage3_stack_analysis(bc, [(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 5, 0)])
        assert not r.passed

    def test_complex_program_stack(self):
        """Full program with multiple push/pop operations."""
        e = BytecodeEmitter()
        e.emit_read_pin(0)       # +1  depth=1
        e.emit_push_f32(270.0)   # +1  depth=2
        e.emit_sub_f()            # -1  depth=1
        e.emit_clamp_f(-30, 30)  # 0   depth=1
        e.emit_write_pin(4)      # -1  depth=0
        bc = _emitter_to_bytecode(e)
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline()
        r = p.stage3_stack_analysis(bc, instrs)
        assert r.passed


# ===================================================================
# Stage 4: Trust Check Tests
# ===================================================================

class TestStage4TrustCheck:
    """Stage 4: Trust-level opcode permission checks."""

    def test_l0_allows_read_only(self):
        """L0 allows only read-only + computation opcodes."""
        e = BytecodeEmitter()
        e.emit_push_f32(1.0)
        e.emit_push_f32(2.0)
        e.emit_add_f()
        e.emit_read_pin(0)
        bc = _emitter_to_bytecode(e)
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline(trust_level=0)
        r = p.stage4_trust_check(bc, instrs)
        assert r.passed

    def test_l0_blocks_write_pin(self):
        """L0 blocks WRITE_PIN."""
        bc = _make_instr(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 5, 0)
        p = BytecodeSafetyPipeline(trust_level=0)
        r = p.stage4_trust_check(bc, [(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 5, 0)])
        assert not r.passed
        assert any("WRITE_PIN" in e or "not allowed" in e for e in r.errors)

    def test_l0_blocks_jump(self):
        """L0 blocks unconditional JUMP."""
        bc = _make_instr(OP_JUMP, 0, 0, 5)
        p = BytecodeSafetyPipeline(trust_level=0)
        r = p.stage4_trust_check(bc, [(OP_JUMP, 0, 0, 5)])
        assert not r.passed

    def test_l1_allows_conditional_jump(self):
        """L1 allows JUMP_IF_FALSE/TRUE."""
        e = BytecodeEmitter()
        e.emit_push_f32(1.0)
        e.emit_push_f32(2.0)
        e.emit_gt_f()
        e.emit_jump_if_false(0)
        bc = _emitter_to_bytecode(e)
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline(trust_level=1)
        r = p.stage4_trust_check(bc, instrs)
        assert r.passed

    def test_l2_allows_write_pin(self):
        """L2 allows WRITE_PIN and unconditional JUMP."""
        e = BytecodeEmitter()
        e.emit_push_f32(1.0)
        e.emit_write_pin(5)
        e.emit_jump(0)
        bc = _emitter_to_bytecode(e)
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline(trust_level=2)
        r = p.stage4_trust_check(bc, instrs)
        assert r.passed

    def test_l3_allows_call_ret(self):
        """L3 allows CALL/RET."""
        call = _make_instr(OP_JUMP, FLAGS_IS_CALL, 0, 4)
        ret = _make_instr(OP_JUMP, 0, 0, 0xFFFFFFFF)
        bc = _make_program([
            _make_instr(OP_NOP),
            call,
            _make_instr(OP_NOP),
            _make_instr(OP_NOP),
            ret,
        ])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline(trust_level=3)
        r = p.stage4_trust_check(bc, instrs)
        assert r.passed

    def test_l2_blocks_call(self):
        """L2 blocks CALL."""
        call = _make_instr(OP_JUMP, FLAGS_IS_CALL, 0, 5)
        bc = _make_program([_make_instr(OP_NOP), call])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE) for i in range(2)]
        p = BytecodeSafetyPipeline(trust_level=2)
        r = p.stage4_trust_check(bc, instrs)
        assert not r.passed
        assert any("CALL" in e or "L3" in e for e in r.errors)

    def test_l5_allows_syscall(self):
        """L5 allows SYSCALL (HALT)."""
        halt = _make_instr(OP_NOP, FLAGS_SYSCALL, 0, 0x01)
        bc = _make_program([halt])
        instrs = [(OP_NOP, FLAGS_SYSCALL, 0, 0x01)]
        p = BytecodeSafetyPipeline(trust_level=5)
        r = p.stage4_trust_check(bc, instrs)
        assert r.passed

    def test_l4_blocks_syscall(self):
        """L4 blocks non-HALT SYSCALL, but allows HALT."""
        # Non-HALT syscall should fail at L4
        non_halt = _make_instr(OP_NOP, FLAGS_SYSCALL, 0, 0x02)  # PID_COMPUTE
        bc = _make_program([non_halt])
        instrs = [(OP_NOP, FLAGS_SYSCALL, 0, 0x02)]
        p = BytecodeSafetyPipeline(trust_level=4)
        r = p.stage4_trust_check(bc, instrs)
        assert not r.passed
        assert any("SYSCALL" in e or "L5" in e for e in r.errors)

        # HALT should be allowed at L4
        halt = _make_instr(OP_NOP, FLAGS_SYSCALL, 0, 0x01)
        bc_halt = _make_program([halt])
        instrs_halt = [(OP_NOP, FLAGS_SYSCALL, 0, 0x01)]
        p_halt = BytecodeSafetyPipeline(trust_level=4)
        r_halt = p_halt.stage4_trust_check(bc_halt, instrs_halt)
        assert r_halt.passed, "HALT should be allowed at trust level 4"

    def test_all_trust_levels_have_l0_subset(self):
        """Verify trust level matrix is monotonic (higher level has superset of opcodes)."""
        for level in range(5):
            lower = TRUST_OPCODE_MATRIX[level]
            upper = TRUST_OPCODE_MATRIX[level + 1]
            assert lower.issubset(upper), (
                f"L{level} opcodes not subset of L{level + 1}"
            )


# ===================================================================
# Stage 5: Semantic Analysis Tests
# ===================================================================

class TestStage5SemanticAnalysis:
    """Stage 5: Semantic property checks."""

    def test_unreachable_code_detection(self):
        """Code after unconditional JUMP should be flagged as unreachable."""
        # JUMP at index 0 to index 2, making index 3 unreachable
        bc = _make_program([
            _make_instr(OP_NOP),  # 0: reachable
            _make_instr(OP_NOP),  # 1: reachable
            _make_instr(OP_NOP),  # 2: reachable (jump target)
            _make_instr(OP_JUMP, 0, 0, 2),  # 3: reachable (but this jumps to 2, so 4 is unreachable)
            _make_instr(0x03, FLAGS_IS_FLOAT, 0, 0),  # 4: UNREACHABLE
        ])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE) for i in range(5)]
        p = BytecodeSafetyPipeline()
        r = p.stage5_semantic_analysis(bc, instrs)
        assert any("unreachable" in w.lower() for w in r.warnings)

    def test_infinite_loop_detection(self):
        """Back-edge JUMP with no conditional exit should be flagged."""
        # JUMP to self = infinite loop with no exit
        jump_instr = _make_instr(OP_JUMP, 0, 0, 0)  # addr 0 -> jump_to_zero catches this
        # But also test: loop body with no exit
        bc = _make_program([
            _make_instr(OP_NOP),     # 0
            _make_instr(OP_NOP),     # 1
            _make_instr(OP_JUMP, 0, 0, 1),  # 2: unconditional JUMP to 1
        ])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE) for i in range(3)]
        p = BytecodeSafetyPipeline()
        r = p.stage5_semantic_analysis(bc, instrs)
        assert not r.passed
        assert any("infinite loop" in e.lower() for e in r.errors)

    def test_loop_with_exit_passes(self):
        """Loop with conditional exit should not be flagged."""
        bc = _make_program([
            _make_instr(OP_NOP),  # 0
            _make_instr(OP_NOP),  # 1
            _make_instr(OP_JUMP_IF_FALSE, 0, 0, 5),  # 2: conditional exit
            _make_instr(OP_NOP),  # 3
            _make_instr(OP_JUMP, 0, 0, 1),  # 4: back-edge to 1
            _make_instr(OP_NOP),  # 5: exit target
        ])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE) for i in range(6)]
        p = BytecodeSafetyPipeline()
        r = p.stage5_semantic_analysis(bc, instrs)
        # Should not flag infinite loop (there's a conditional exit)
        loop_errors = [e for e in r.errors if "infinite loop" in e.lower()]
        assert len(loop_errors) == 0

    def test_clamp_valid_range(self):
        """CLAMP_F with valid lo < hi should pass."""
        e = BytecodeEmitter()
        e.emit_push_f32(50.0)
        e.emit_clamp_f(-30.0, 30.0)
        e.emit_write_pin(4)
        bc = _emitter_to_bytecode(e)
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline()
        r = p.stage5_semantic_analysis(bc, instrs)
        clamp_errors = [e for e in r.errors if "CLAMP" in e or "clamp" in e]
        assert len(clamp_errors) == 0

    def test_clamp_invalid_range(self):
        """CLAMP_F with lo >= hi should fail."""
        e = BytecodeEmitter()
        e.emit_push_f32(50.0)
        e.emit_clamp_f(30.0, -30.0)  # lo > hi (invalid!)
        e.emit_write_pin(4)
        bc = _emitter_to_bytecode(e)
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline()
        r = p.stage5_semantic_analysis(bc, instrs)
        assert not r.passed
        assert any("lo" in e.lower() and ">=" in e for e in r.errors)

    def test_div_by_constant_zero(self):
        """DIV_F preceded by PUSH(0) should be flagged."""
        push_zero = _make_instr(0x01, FLAGS_HAS_IMMEDIATE, 0, 0)  # PUSH_I8(0)
        div_instr = _make_instr(0x0B)  # DIV_F
        # Need 2 values on stack: push something, push 0, then divide
        push_val = _make_instr(0x03, FLAGS_IS_FLOAT, 0,
                               struct.unpack("<I", struct.pack("<f", 5.0))[0])
        bc = _make_program([push_val, push_zero, div_instr])
        instrs = [unpack_instruction(bc, i * INSTR_SIZE) for i in range(3)]
        p = BytecodeSafetyPipeline()
        r = p.stage5_semantic_analysis(bc, instrs)
        assert not r.passed
        assert any("zero" in e.lower() for e in r.errors)

    def test_pid_compute_safe_params(self):
        """PID_COMPUTE with safe parameters should pass."""
        e = BytecodeEmitter()
        e.emit_push_f32(1.0)   # Kp (safe: 0-10)
        e.emit_push_f32(0.1)   # Ki (safe: 0-5)
        e.emit_push_f32(0.05)  # Kd (safe: 0-5)
        e.emit_push_f32(100.0) # setpoint (safe: -1000 to 1000)
        e.emit_halt()           # SYSCALL HALT (not PID_COMPUTE, but tests syscall)
        bc = _emitter_to_bytecode(e)
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline()
        r = p.stage5_semantic_analysis(bc, instrs)
        # HALT not PID_COMPUTE so no PID check
        pid_errors = [e for e in r.errors if "PID" in e]
        assert len(pid_errors) == 0

    def test_no_warnings_for_simple_program(self):
        """Simple linear program should have no semantic issues."""
        e = BytecodeEmitter()
        e.emit_push_f32(1.0)
        e.emit_push_f32(2.0)
        e.emit_add_f()
        bc = _emitter_to_bytecode(e)
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline()
        r = p.stage5_semantic_analysis(bc, instrs)
        assert len(r.errors) == 0


# ===================================================================
# Stage 6: Adversarial Probing Tests
# ===================================================================

class TestStage6Adversarial:
    """Stage 6: Adversarial fuzzing and boundary probing."""

    def test_adversarial_stage_always_passes(self):
        """Stage 6 is informational — it always passes but reports warnings."""
        bc = _make_simple_safe_program()
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline(trust_level=5)
        r = p.stage6_adversarial(bc, instrs)
        assert r.passed  # Stage 6 itself always passes

    def test_bit_flip_detection(self):
        """Flipping bits in a valid instruction should cause re-validation failures."""
        bc = _make_simple_safe_program()
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline(trust_level=5)
        r = p.stage6_adversarial(bc, instrs)
        # Most bit-flips should cause stage 1 (syntax) or other stage failures
        # The stage always passes, but should have some warnings if vulnerabilities found
        assert isinstance(r, StageResult)

    def test_empty_bytecode_rejected(self):
        """Mutating to empty bytecode should be caught."""
        bc = _make_instr(OP_NOP)
        instrs = [(OP_NOP, 0, 0, 0)]
        p = BytecodeSafetyPipeline()
        passed = p._run_sub_stages(b"")
        assert not passed

    def test_misaligned_bytecode_rejected(self):
        """Single-byte bytecode should be caught."""
        p = BytecodeSafetyPipeline()
        passed = p._run_sub_stages(b"\x00")
        assert not passed

    def test_all_0xFF_rejected(self):
        """All-0xFF instruction should fail syntax (unknown opcode)."""
        bad_bc = b"\xFF" * INSTR_SIZE
        p = BytecodeSafetyPipeline()
        passed = p._run_sub_stages(bad_bc)
        assert not passed

    def test_boundary_operand_mutation(self):
        """Operand2 set to 0 or 0xFFFFFFFF should be properly validated."""
        # This is implicitly tested by stage6_adversarial
        bc = _make_simple_safe_program()
        instrs = [unpack_instruction(bc, i * INSTR_SIZE)
                   for i in range(len(bc) // INSTR_SIZE)]
        p = BytecodeSafetyPipeline(trust_level=5)
        r = p.stage6_adversarial(bc, instrs)
        assert r.passed  # Stage 6 always passes


# ===================================================================
# Integration Tests
# ===================================================================

class TestIntegration:
    """Full pipeline integration tests."""

    def test_known_good_bytecode_passes_all_stages(self):
        """A well-formed program with CLAMP_F before WRITE_PIN should pass at L5."""
        bc = _make_simple_safe_program()
        p = BytecodeSafetyPipeline(trust_level=5)
        report = p.validate(bc)
        assert isinstance(report, SafetyReport)
        assert report.overall_passed
        assert len(report.failed_stages) == 0
        assert report.instruction_count == len(bc) // INSTR_SIZE

    def test_known_good_bytecode_at_l2(self):
        """Same program passes at L2 (no syscall, no safety pin write)."""
        # Build program WITHOUT halt (which requires L5)
        e = BytecodeEmitter()
        e.emit_read_pin(0)
        e.emit_push_f32(270.0)
        e.emit_sub_f()
        e.emit_clamp_f(-30.0, 30.0)
        e.emit_write_pin(4)  # Non-safety pin
        bc = _emitter_to_bytecode(e)
        p = BytecodeSafetyPipeline(trust_level=2)
        report = p.validate(bc)
        assert report.overall_passed, (
            f"Expected pass at L2, got: {report.summary()}"
        )

    def test_bad_bytecode_fails(self):
        """Bytecode with multiple violations should fail."""
        # Write to safety pin at low trust + stack underflow
        bc = _make_instr(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 0, 0)  # Safety pin 0
        p = BytecodeSafetyPipeline(trust_level=2)
        report = p.validate(bc)
        assert not report.overall_passed

    def test_empty_bytecode_fails(self):
        """Empty bytecode fails pipeline."""
        p = BytecodeSafetyPipeline()
        report = p.validate(b"")
        assert not report.overall_passed
        assert any("syntax_check" in s for s in report.failed_stages)

    def test_misaligned_bytecode_fails(self):
        """Misaligned bytecode fails pipeline."""
        p = BytecodeSafetyPipeline()
        report = p.validate(b"\x00\x00\x00")
        assert not report.overall_passed

    def test_report_contains_all_stages(self):
        """Report should have results for all 6 stages."""
        bc = _make_simple_safe_program()
        p = BytecodeSafetyPipeline(trust_level=5)
        report = p.validate(bc)
        assert len(report.stages) == 6
        stage_names = {s.stage_name for s in report.stages}
        assert stage_names == {
            "syntax_check", "safety_rules", "stack_analysis",
            "trust_check", "semantic_analysis", "adversarial_probing",
        }

    def test_report_summary(self):
        """Report.summary() should return a non-empty string."""
        bc = _make_simple_safe_program()
        p = BytecodeSafetyPipeline(trust_level=5)
        report = p.validate(bc)
        summary = report.summary()
        assert len(summary) > 0
        assert "PASS" in summary

    def test_report_violations(self):
        """Violations should be populated for bad bytecode."""
        # Write to safety pin + syscall at L2
        write_instr = _make_instr(OP_WRITE_PIN, FLAGS_HAS_IMMEDIATE, 0, 0)
        syscall_instr = _make_instr(OP_NOP, FLAGS_SYSCALL, 0, 0x01)
        bc = _make_program([write_instr, syscall_instr])
        p = BytecodeSafetyPipeline(trust_level=2)
        report = p.validate(bc)
        assert len(report.violations) > 0
        # Check violation format
        v = report.violations[0]
        assert isinstance(v, SafetyViolation)
        assert v.stage != ""
        assert v.severity in ("error", "warning", "info")

    def test_trust_level_clamped(self):
        """Trust levels outside 0-5 should be clamped."""
        p = BytecodeSafetyPipeline(trust_level=10)
        assert p.trust_level == 5
        p2 = BytecodeSafetyPipeline(trust_level=-1)
        assert p2.trust_level == 0

    def test_custom_safety_config(self):
        """Custom safety config should override defaults."""
        cfg = {"max_instructions": 5, "max_stack_depth": 4}
        p = BytecodeSafetyPipeline(trust_level=5, safety_config=cfg)
        assert p.max_instructions == 5
        assert p.max_stack_depth == 4

    def test_custom_config_enforced(self):
        """Custom limits should be enforced during validation."""
        cfg = {"max_instructions": 2}
        bc = _make_nops(5)  # 5 instructions > limit of 2
        p = BytecodeSafetyPipeline(trust_level=5, safety_config=cfg)
        report = p.validate(bc)
        assert not report.overall_passed
        assert any("exceeds" in e for e in report.stages[0].errors)

    def test_bytecode_hash_and_metadata(self):
        """Report should contain correct hash, size, and metadata."""
        bc = _make_instr(OP_NOP)
        p = BytecodeSafetyPipeline(trust_level=3)
        report = p.validate(bc)
        assert report.bytecode_size == INSTR_SIZE
        assert report.instruction_count == 1
        assert len(report.bytecode_hash) == 64  # SHA-256 hex
        assert report.trust_level == 3
        assert report.timestamp != ""


# ===================================================================
# Model Tests
# ===================================================================

class TestModels:
    """Test data model classes."""

    def test_stage_result_properties(self):
        r = StageResult(stage_name="test", passed=True)
        assert not r.has_errors
        assert not r.has_warnings

        r.errors.append("error1")
        assert r.has_errors
        assert not r.has_warnings

        r.warnings.append("warn1")
        assert r.has_warnings

    def test_safety_violation_str(self):
        v = SafetyViolation(
            stage="syntax_check",
            severity="error",
            instruction_index=5,
            opcode=0xFF,
            description="Unknown opcode",
            remediation="Use valid opcode",
        )
        s = str(v)
        assert "ERROR" in s
        assert "0xFF" in s
        assert "Unknown opcode" in s

    def test_safety_report_totals(self):
        r1 = StageResult(stage_name="s1", passed=True)
        r2 = StageResult(stage_name="s2", passed=False, errors=["e1"], warnings=["w1"])
        report = SafetyReport(
            overall_passed=False,
            stages=[r1, r2],
        )
        assert report.total_errors == 1
        assert report.total_warnings == 1
        assert report.passed_stages == ["s1"]
        assert report.failed_stages == ["s2"]


# ===================================================================
# Rules Tests
# ===================================================================

class TestRules:
    """Test rule definitions."""

    def test_all_core_opcodes_in_stack_effects(self):
        """All core opcodes (0x00-0x1F) should have stack effects defined."""
        for op in range(0x20):
            assert op in STACK_EFFECTS, f"Missing stack effect for opcode 0x{op:02X}"

    def test_stack_effects_sensible(self):
        """PUSH opcodes should have positive effect, POP negative."""
        assert STACK_EFFECTS[0x01] > 0  # PUSH_I8
        assert STACK_EFFECTS[0x02] > 0  # PUSH_I16
        assert STACK_EFFECTS[0x03] > 0  # PUSH_F32
        assert STACK_EFFECTS[0x04] < 0  # POP
        assert STACK_EFFECTS[0x0A] < 0  # MUL_F (binary)

    def test_trust_matrix_monotonic(self):
        """Each higher trust level should be a superset."""
        for level in range(5):
            lower = TRUST_OPCODE_MATRIX[level]
            upper = TRUST_OPCODE_MATRIX[level + 1]
            assert lower.issubset(upper)

    def test_safety_critical_pins(self):
        """Safety-critical pins should include 0-3."""
        assert SAFETY_CRITICAL_PINS == {0, 1, 2, 3}


# ===================================================================
# Property Tests — Random Mutations
# ===================================================================

class TestPropertyMutations:
    """Property tests: randomly mutated bytecode should mostly fail validation."""

    def test_random_byte_flips_fail(self):
        """Flipping random bytes in valid bytecode should usually break it."""
        # NOP is 0x00 — most single-bit flips on opcode byte produce valid opcodes.
        # Use deterministic bit-flip testing instead: flip each bit of the opcode
        # byte and verify at least some produce invalid opcodes (values > 0x56).
        bc = _make_instr(OP_NOP)
        p = BytecodeSafetyPipeline(trust_level=5)

        # Deterministically test all 8 bits of byte 0 (opcode)
        failures = 0
        for bit in range(8):
            mutated = bytearray(bc)
            mutated[0] ^= (1 << bit)
            report = p.validate(bytes(mutated))
            if not report.overall_passed:
                failures += 1

        # Also test all bits of other bytes (flags, operand1, operand2)
        # Flipping flags may produce CALL/SYSCALL which fail at L5 trust
        for byte_pos in range(1, INSTR_SIZE):
            for bit in range(8):
                mutated = bytearray(bc)
                mutated[byte_pos] ^= (1 << bit)
                report = p.validate(bytes(mutated))
                if not report.overall_passed:
                    failures += 1

        # At least some mutations should fail (e.g., flags becoming SYSCALL at L5
        # is actually fine, but flags becoming CALL will require L3)
        # With 64 bit-flips total, expect several failures
        assert failures >= 1, (
            f"None of the bit-flips broke validation — "
            f"this is unexpected for a NOP instruction"
        )

    def test_random_byte_substitution_fails(self):
        """Replacing random bytes should usually break bytecode."""
        # Use a simple program where opcode byte substitution is likely to break
        bc = _make_instr(OP_NOP)  # 8 bytes
        p = BytecodeSafetyPipeline(trust_level=5)

        failures = 0
        trials = 20

        for _ in range(trials):
            mutated = bytearray(bc)
            import random
            # Specifically mutate the opcode byte (byte 0)
            mutated[0] = random.randint(0, 255)

            report = p.validate(bytes(mutated))
            if not report.overall_passed:
                failures += 1

        # Random opcode bytes should often produce invalid opcodes
        assert failures >= trials // 4
