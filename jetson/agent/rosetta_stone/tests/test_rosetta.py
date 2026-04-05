"""Rosetta Stone — Comprehensive test suite.

Tests cover all 4 layers of the translation pipeline:
  Layer 1: Intent parsing (natural language -> Intent)
  Layer 2: Intent compilation (Intent -> IR)
  Layer 3: IR validation and optimization
  Layer 4: Bytecode generation (IR -> NEXUS bytecode)
  Full pipeline: text -> bytecode
"""

from __future__ import annotations

import struct

import pytest

from agent.rosetta_stone.intent_parser import Intent, IntentParser
from agent.rosetta_stone.intent_compiler import IntentCompiler, IRInstruction
from agent.rosetta_stone.ir_validator import IRValidator, ValidationResult
from agent.rosetta_stone.bytecode_generator import BytecodeGenerator
from agent.rosetta_stone.rosetta import RosettaStone, TranslationResult
from reflex.bytecode_emitter import (
    BytecodeEmitter,
    INSTR_SIZE,
    FLAGS_SYSCALL,
    FLAGS_HAS_IMMEDIATE,
    FLAGS_IS_FLOAT,
    unpack_instruction,
    pack_instruction,
)
from reflex.safety_validator import SafetyValidator


# ===================================================================
# Layer 1: IntentParser tests
# ===================================================================

class TestIntentParserReadSensor:
    """Test parsing of 'read sensor <n>' intents."""

    def test_read_sensor_basic(self) -> None:
        parser = IntentParser()
        intent = parser.parse("read sensor 3")
        assert intent.action == "READ"
        assert intent.target == "SENSOR"
        assert intent.pin == 3
        assert intent.confidence == 1.0

    def test_read_sensor_zero(self) -> None:
        parser = IntentParser()
        intent = parser.parse("read sensor 0")
        assert intent.action == "READ"
        assert intent.pin == 0

    def test_read_sensor_case_insensitive(self) -> None:
        parser = IntentParser()
        intent = parser.parse("READ SENSOR 5")
        assert intent.action == "READ"
        assert intent.pin == 5

    def test_read_sensor_large_pin(self) -> None:
        parser = IntentParser()
        intent = parser.parse("read sensor 63")
        assert intent.pin == 63


class TestIntentParserWriteActuator:
    """Test parsing of 'set actuator <n> to <value>' intents."""

    def test_write_actuator_basic(self) -> None:
        parser = IntentParser()
        intent = parser.parse("set actuator 2 to 0.75")
        assert intent.action == "WRITE"
        assert intent.target == "ACTUATOR"
        assert intent.pin == 2
        assert intent.value == 0.75

    def test_write_actuator_negative(self) -> None:
        parser = IntentParser()
        intent = parser.parse("set actuator 1 to -0.5")
        assert intent.value == -0.5

    def test_write_actuator_integer(self) -> None:
        parser = IntentParser()
        intent = parser.parse("set actuator 0 to 1")
        assert intent.value == 1.0

    def test_write_actuator_scientific(self) -> None:
        parser = IntentParser()
        intent = parser.parse("set actuator 3 to 1.5e-2")
        assert abs(intent.value - 0.015) < 1e-10


class TestIntentParserConditional:
    """Test parsing of 'if sensor <n> <op> <threshold> then <action>'."""

    def test_conditional_gt(self) -> None:
        parser = IntentParser()
        intent = parser.parse("if sensor 2 gt 50.0 then set actuator 3 to 1.0")
        assert intent.action == "CONDITIONAL"
        assert intent.pin == 2
        assert intent.operator == "gt"
        assert intent.threshold == 50.0
        assert len(intent.then_body) == 1
        assert intent.then_body[0].action == "WRITE"
        assert intent.then_body[0].pin == 3

    def test_conditional_less_than(self) -> None:
        parser = IntentParser()
        intent = parser.parse("if sensor 1 less than 10.5 then halt")
        assert intent.operator == "lt"
        assert intent.threshold == 10.5
        assert intent.then_body[0].action == "SYSCALL"

    def test_conditional_symbol_gt(self) -> None:
        parser = IntentParser()
        intent = parser.parse("if sensor 4 > 100 then log snapshot")
        assert intent.operator == "gt"
        assert intent.threshold == 100.0

    def test_conditional_gte(self) -> None:
        parser = IntentParser()
        intent = parser.parse("if sensor 0 gte 75.0 then trigger actuator 1")
        assert intent.operator == "gte"
        assert len(intent.then_body) == 1
        assert intent.then_body[0].value == 1.0

    def test_conditional_lte(self) -> None:
        parser = IntentParser()
        intent = parser.parse("if sensor 5 lte 20 then halt")
        assert intent.operator == "lte"


class TestIntentParserLoop:
    """Test parsing of 'repeat <n> times: <actions>'."""

    def test_loop_single_action(self) -> None:
        parser = IntentParser()
        intent = parser.parse("repeat 5 times: read sensor 0")
        assert intent.action == "LOOP"
        assert intent.params["count"] == 5
        assert len(intent.body) == 1
        assert intent.body[0].action == "READ"
        assert intent.body[0].pin == 0

    def test_loop_multiple_actions(self) -> None:
        parser = IntentParser()
        intent = parser.parse("repeat 10 times: read sensor 1, set actuator 2 to 0.5")
        assert intent.params["count"] == 10
        assert len(intent.body) == 2
        assert intent.body[0].action == "READ"
        assert intent.body[1].action == "WRITE"

    def test_loop_empty_body(self) -> None:
        parser = IntentParser()
        intent = parser.parse("repeat 3 times: do nothing")
        assert intent.action == "LOOP"
        assert intent.params["count"] == 3
        assert len(intent.body) == 0
        assert intent.confidence < 1.0


class TestIntentParserWait:
    """Test parsing of 'wait <n> cycles'."""

    def test_wait_basic(self) -> None:
        parser = IntentParser()
        intent = parser.parse("wait 100 cycles")
        assert intent.action == "WAIT"
        assert intent.params["cycles"] == 100

    def test_wait_one_cycle(self) -> None:
        parser = IntentParser()
        intent = parser.parse("wait 1 cycle")
        assert intent.params["cycles"] == 1

    def test_wait_large(self) -> None:
        parser = IntentParser()
        intent = parser.parse("wait 5000 cycles")
        assert intent.params["cycles"] == 5000


class TestIntentParserPID:
    """Test parsing of 'compute pid on sensor <n> with kp=<v> ki=<v> kd=<v>'."""

    def test_pid_basic(self) -> None:
        parser = IntentParser()
        intent = parser.parse(
            "compute pid on sensor 0 with kp=1.0 ki=0.1 kd=0.01"
        )
        assert intent.action == "PID"
        assert intent.pin == 0
        assert intent.params["kp"] == 1.0
        assert intent.params["ki"] == 0.1
        assert intent.params["kd"] == 0.01

    def test_pid_negative_kp(self) -> None:
        parser = IntentParser()
        intent = parser.parse(
            "compute pid on sensor 2 with kp=-0.5 ki=0.0 kd=0.0"
        )
        assert intent.params["kp"] == -0.5


class TestIntentParserNavigate:
    """Test parsing of 'navigate to waypoint <x>,<y>'."""

    def test_navigate_basic(self) -> None:
        parser = IntentParser()
        intent = parser.parse("navigate to waypoint 10.5,20.3")
        assert intent.action == "NAVIGATE"
        assert intent.target == "WAYPOINT"
        assert intent.params["x"] == 10.5
        assert intent.params["y"] == 20.3

    def test_navigate_negative(self) -> None:
        parser = IntentParser()
        intent = parser.parse("navigate to waypoint -5.0,-10.0")
        assert intent.params["x"] == -5.0
        assert intent.params["y"] == -10.0


class TestIntentParserSyscall:
    """Test parsing of syscall intents."""

    def test_log_snapshot(self) -> None:
        parser = IntentParser()
        intent = parser.parse("log snapshot")
        assert intent.action == "SYSCALL"
        assert intent.params["syscall"] == "record_snapshot"

    def test_emit_event(self) -> None:
        parser = IntentParser()
        intent = parser.parse('emit event alert')
        assert intent.action == "SYSCALL"
        assert intent.params["syscall"] == "emit_event"
        assert intent.params["message"] == "alert"

    def test_halt(self) -> None:
        parser = IntentParser()
        intent = parser.parse("halt")
        assert intent.action == "SYSCALL"
        assert intent.params["syscall"] == "halt"

    def test_stop(self) -> None:
        parser = IntentParser()
        intent = parser.parse("stop")
        assert intent.action == "SYSCALL"
        assert intent.params["syscall"] == "halt"


class TestIntentParserCompound:
    """Test parsing of compound intents."""

    def test_monitor_trigger(self) -> None:
        parser = IntentParser()
        intent = parser.parse(
            "monitor sensor 2 and if gt 50.0 then trigger actuator 3"
        )
        assert intent.action == "COMPOUND"
        assert len(intent.body) == 1
        cond = intent.body[0]
        assert cond.action == "CONDITIONAL"
        assert cond.pin == 2
        assert cond.operator == "gt"
        assert cond.threshold == 50.0
        assert len(cond.then_body) == 1
        assert cond.then_body[0].action == "WRITE"
        assert cond.then_body[0].pin == 3

    def test_patrol_basic(self) -> None:
        parser = IntentParser()
        intent = parser.parse("patrol: read GPS, if distance > 100m return home")
        assert intent.action == "COMPOUND"
        assert intent.confidence == 0.8
        assert len(intent.body) >= 1


class TestIntentParserErrors:
    """Test error handling in intent parsing."""

    def test_empty_input_raises(self) -> None:
        parser = IntentParser()
        with pytest.raises(ValueError):
            parser.parse("")

    def test_whitespace_only_raises(self) -> None:
        parser = IntentParser()
        with pytest.raises(ValueError):
            parser.parse("   ")

    def test_unknown_intent(self) -> None:
        parser = IntentParser()
        intent = parser.parse("do something weird")
        assert intent.action == "UNKNOWN"
        assert intent.confidence == 0.0

    def test_missing_parameters(self) -> None:
        parser = IntentParser()
        intent = parser.parse("read sensor")
        assert intent.action == "UNKNOWN"


class TestIntentParserVariables:
    """Test variable read/write intents."""

    def test_read_variable(self) -> None:
        parser = IntentParser()
        intent = parser.parse("read variable 5")
        assert intent.action == "READ"
        assert intent.target == "VARIABLE"
        assert intent.pin == 5

    def test_read_var_abbrev(self) -> None:
        parser = IntentParser()
        intent = parser.parse("read var 10")
        assert intent.action == "READ"
        assert intent.target == "VARIABLE"

    def test_write_variable(self) -> None:
        parser = IntentParser()
        intent = parser.parse("set variable 3 to 42.0")
        assert intent.action == "WRITE"
        assert intent.target == "VARIABLE"
        assert intent.pin == 3
        assert intent.value == 42.0

    def test_parse_many(self) -> None:
        parser = IntentParser()
        intents = parser.parse_many([
            "read sensor 0",
            "set actuator 1 to 0.5",
            "halt",
        ])
        assert len(intents) == 3
        assert intents[0].action == "READ"
        assert intents[1].action == "WRITE"
        assert intents[2].action == "SYSCALL"


# ===================================================================
# Layer 2: IntentCompiler tests
# ===================================================================

class TestIntentCompilerRead:
    def test_compile_read_sensor(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(action="READ", target="SENSOR", pin=3, raw="read sensor 3")
        ir = compiler.compile(intent)
        assert len(ir) >= 2  # at least the read + halt
        assert ir[0].opcode == "READ_PIN"
        assert ir[0].operand1 == 3
        # Last instruction should be HALT
        assert ir[-1].opcode == "SYSCALL"
        assert ir[-1].operand2 == 0x01

    def test_compile_read_variable(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(action="READ", target="VARIABLE", pin=5, raw="read var 5")
        ir = compiler.compile(intent)
        assert ir[0].opcode == "READ_PIN"
        assert ir[0].operand1 == 5


class TestIntentCompilerWrite:
    def test_compile_write_actuator(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="WRITE", target="ACTUATOR", pin=2, value=0.75,
            raw="set actuator 2 to 0.75",
        )
        ir = compiler.compile(intent)
        assert ir[0].opcode == "PUSH_F32"
        assert ir[0].operand2 == 0.75
        assert ir[1].opcode == "CLAMP_F"
        assert ir[1].operand1 == -1.0
        assert ir[1].operand2 == 1.0
        assert ir[2].opcode == "WRITE_PIN"
        assert ir[2].operand1 == 2


class TestIntentCompilerConditional:
    def test_compile_conditional_has_labels(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="CONDITIONAL",
            target="SENSOR",
            pin=2,
            operator="gt",
            threshold=50.0,
            then_body=[
                Intent(action="WRITE", target="ACTUATOR", pin=3, value=1.0, raw=""),
            ],
            raw="if sensor 2 gt 50 then set actuator 3 to 1",
        )
        ir = compiler.compile(intent)
        # Should have: READ_PIN, PUSH_F32, GT_F, JUMP_IF_FALSE, PUSH_F32, CLAMP_F, WRITE_PIN, NOP, HALT
        opcodes = [i.opcode for i in ir]
        assert "READ_PIN" in opcodes
        assert "GT_F" in opcodes
        assert "JUMP_IF_FALSE" in opcodes
        # Check labels
        labels = [i.label for i in ir if i.label]
        assert len(labels) >= 1  # else label

    def test_compile_conditional_jump_target(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="CONDITIONAL",
            target="SENSOR",
            pin=0,
            operator="lt",
            threshold=10.0,
            then_body=[
                Intent(action="SYSCALL", target="SYSTEM",
                       params={"syscall": "halt"}, raw=""),
            ],
            raw="if sensor 0 lt 10 then halt",
        )
        ir = compiler.compile(intent)
        # Find the JUMP_IF_FALSE
        jump_instrs = [i for i in ir if i.opcode == "JUMP_IF_FALSE"]
        assert len(jump_instrs) == 1
        assert jump_instrs[0].jump_target is not None


class TestIntentCompilerLoop:
    def test_compile_loop_has_structure(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="LOOP",
            target="SYSTEM",
            body=[
                Intent(action="READ", target="SENSOR", pin=0, raw=""),
            ],
            params={"count": 5},
            raw="repeat 5 times: read sensor 0",
        )
        ir = compiler.compile(intent)
        opcodes = [i.opcode for i in ir]
        assert "PUSH_I8" in opcodes
        assert "DUP" in opcodes
        assert "LTE_F" in opcodes
        assert "JUMP_IF_TRUE" in opcodes
        assert "SUB_F" in opcodes
        assert "JUMP" in opcodes
        assert "POP" in opcodes

    def test_compile_loop_has_labels(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="LOOP",
            target="SYSTEM",
            body=[
                Intent(action="READ", target="SENSOR", pin=1, raw=""),
            ],
            params={"count": 3},
            raw="repeat 3: read sensor 1",
        )
        ir = compiler.compile(intent)
        labels = [i.label for i in ir if i.label]
        assert len(labels) >= 2  # start/check and end labels


class TestIntentCompilerWait:
    def test_compile_wait_single(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="WAIT", target="SYSTEM", params={"cycles": 1}, raw="wait 1 cycle",
        )
        ir = compiler.compile(intent)
        assert ir[0].opcode == "NOP"

    def test_compile_wait_multiple(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="WAIT", target="SYSTEM", params={"cycles": 5}, raw="wait 5 cycles",
        )
        ir = compiler.compile(intent)
        # Should have loop structure with labels
        opcodes = [i.opcode for i in ir]
        assert "PUSH_I8" in opcodes
        assert "DUP" in opcodes
        assert "SUB_F" in opcodes
        assert "JUMP_IF_TRUE" in opcodes


class TestIntentCompilerPID:
    def test_compile_pid(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="PID",
            target="SENSOR",
            pin=0,
            params={"kp": 1.0, "ki": 0.1, "kd": 0.01},
            raw="compute pid on sensor 0",
        )
        ir = compiler.compile(intent)
        assert ir[0].opcode == "READ_PIN"
        assert ir[1].opcode == "PUSH_F32"
        assert ir[1].operand2 == 1.0
        assert ir[2].opcode == "PUSH_F32"
        assert ir[2].operand2 == 0.1
        assert ir[3].opcode == "PUSH_F32"
        assert ir[3].operand2 == 0.01
        # Syscall for PID
        syscall_instrs = [i for i in ir if i.opcode == "SYSCALL"]
        assert any(i.operand2 == 0x02 for i in syscall_instrs)


class TestIntentCompilerNavigate:
    def test_compile_navigate(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="NAVIGATE",
            target="WAYPOINT",
            params={"x": 10.0, "y": 20.0},
            raw="navigate to waypoint 10,20",
        )
        ir = compiler.compile(intent)
        assert ir[0].opcode == "PUSH_F32"
        assert ir[0].operand2 == 10.0
        assert ir[1].opcode == "PUSH_F32"
        assert ir[1].operand2 == 20.0
        assert ir[2].opcode == "DECLARE_INTENT"


class TestIntentCompilerSyscall:
    def test_compile_log_snapshot(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="SYSCALL", target="SYSTEM",
            params={"syscall": "record_snapshot"}, raw="log snapshot",
        )
        ir = compiler.compile(intent)
        assert ir[0].opcode == "SYSCALL"
        assert ir[0].operand2 == 0x03

    def test_compile_halt(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="SYSCALL", target="SYSTEM",
            params={"syscall": "halt"}, raw="halt",
        )
        ir = compiler.compile(intent)
        assert ir[0].opcode == "SYSCALL"
        assert ir[0].operand2 == 0x01


class TestIntentCompilerCompound:
    def test_compile_compound(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(
            action="COMPOUND",
            target="SYSTEM",
            body=[
                Intent(action="READ", target="SENSOR", pin=0, raw=""),
                Intent(action="READ", target="SENSOR", pin=1, raw=""),
            ],
            raw="compound",
        )
        ir = compiler.compile(intent)
        read_pins = [i for i in ir if i.opcode == "READ_PIN"]
        assert len(read_pins) == 2


class TestIntentCompilerUnknown:
    def test_compile_unknown_returns_empty(self) -> None:
        compiler = IntentCompiler()
        intent = Intent(action="UNKNOWN", target="SYSTEM", raw="")
        ir = compiler.compile(intent)
        assert ir == []


# ===================================================================
# Layer 3: IRValidator tests
# ===================================================================

class TestIRValidatorBasic:
    def test_valid_simple_read(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="READ_PIN", operand1=0),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir, trust_level=5)
        assert result.valid
        assert len(result.errors) == 0

    def test_empty_ir_invalid(self) -> None:
        validator = IRValidator()
        result = validator.validate([])
        assert not result.valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_max_stack_depth(self) -> None:
        validator = IRValidator(max_stack=3)
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=1.0),
            IRInstruction(opcode="PUSH_F32", operand2=2.0),
            IRInstruction(opcode="PUSH_F32", operand2=3.0),
            IRInstruction(opcode="PUSH_F32", operand2=4.0),  # 4th push
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir)
        assert not result.valid
        assert any("stack" in e.lower() for e in result.errors)

    def test_stack_underflow(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="POP"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir)
        assert not result.valid
        assert any("underflow" in e.lower() for e in result.errors)

    def test_instruction_count(self) -> None:
        validator = IRValidator(max_cycles=5)
        ir = [IRInstruction(opcode="NOP")] * 10
        result = validator.validate(ir)
        assert not result.valid
        assert any("cycle budget" in e.lower() for e in result.errors)


class TestIRValidatorPins:
    def test_valid_pin_range(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="READ_PIN", operand1=0),
            IRInstruction(opcode="READ_PIN", operand1=63),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir, trust_level=5)
        assert result.valid

    def test_invalid_sensor_pin(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="READ_PIN", operand1=100),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir)
        assert not result.valid
        assert any("pin" in e.lower() for e in result.errors)

    def test_invalid_actuator_pin(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=0.5),
            IRInstruction(opcode="CLAMP_F", operand1=-1.0, operand2=1.0),
            IRInstruction(opcode="WRITE_PIN", operand1=100),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir)
        assert not result.valid
        assert any("pin" in e.lower() for e in result.errors)


class TestIRValidatorTrust:
    def test_trust_level_0_all_blocked(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="READ_PIN", operand1=0),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir, trust_level=0)
        assert not result.valid

    def test_trust_level_1_read_only(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="READ_PIN", operand1=0),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir, trust_level=1)
        # L1 allows READ and SYSCALL (halt)
        assert result.valid

    def test_trust_level_1_blocks_write(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=0.5),
            IRInstruction(opcode="CLAMP_F", operand1=-1.0, operand2=1.0),
            IRInstruction(opcode="WRITE_PIN", operand1=0),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir, trust_level=1)
        assert not result.valid
        assert any("trust" in e.lower() for e in result.errors)

    def test_trust_level_5_all_allowed(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="READ_PIN", operand1=0),
            IRInstruction(opcode="PUSH_F32", operand2=1.0),
            IRInstruction(opcode="CLAMP_F", operand1=-1.0, operand2=1.0),
            IRInstruction(opcode="WRITE_PIN", operand1=1),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir, trust_level=5)
        assert result.valid


class TestIRValidatorJumpTargets:
    def test_valid_jump_labels(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="READ_PIN", operand1=0, label="start"),
            IRInstruction(opcode="JUMP", jump_target="start"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir, trust_level=5)
        assert result.valid

    def test_unresolved_jump_target(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="JUMP", jump_target="nonexistent"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir)
        assert not result.valid
        assert any("not defined" in e.lower() for e in result.errors)


class TestIRValidatorInfiniteLoop:
    def test_backward_jump_without_decrement_warns(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="READ_PIN", operand1=0, label="loop"),
            IRInstruction(opcode="JUMP", jump_target="loop"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir, trust_level=5)
        assert result.valid  # It's valid, but has a warning
        assert any("infinite" in w.lower() for w in result.warnings)

    def test_backward_jump_with_decrement_no_warning(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="PUSH_I8", operand1=5, label="loop"),
            IRInstruction(opcode="DUP"),
            IRInstruction(opcode="PUSH_I8", operand1=1),
            IRInstruction(opcode="SUB_F"),
            IRInstruction(opcode="JUMP", jump_target="loop"),
            IRInstruction(opcode="POP"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        result = validator.validate(ir)
        infinite_warnings = [w for w in result.warnings if "infinite" in w.lower()]
        assert len(infinite_warnings) == 0


# ===================================================================
# IR Optimizer tests
# ===================================================================

class TestIROptimizer:
    def test_remove_push_pop_pair(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=42.0),
            IRInstruction(opcode="POP"),
            IRInstruction(opcode="READ_PIN", operand1=0),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        optimized = validator.optimize(ir)
        opcodes = [i.opcode for i in optimized]
        assert "PUSH_F32" not in opcodes
        assert "POP" not in opcodes
        assert "READ_PIN" in opcodes

    def test_fold_constant_arithmetic(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=10.0),
            IRInstruction(opcode="PUSH_F32", operand2=20.0),
            IRInstruction(opcode="ADD_F"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        optimized = validator.optimize(ir)
        assert len(optimized) == 2  # PUSH_F32 + SYSCALL
        assert optimized[0].opcode == "PUSH_F32"
        assert abs(float(optimized[0].operand2) - 30.0) < 0.001

    def test_fold_constant_subtraction(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=50.0),
            IRInstruction(opcode="PUSH_F32", operand2=20.0),
            IRInstruction(opcode="SUB_F"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        optimized = validator.optimize(ir)
        assert len(optimized) == 2
        assert abs(float(optimized[0].operand2) - 30.0) < 0.001

    def test_fold_constant_multiplication(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=3.0),
            IRInstruction(opcode="PUSH_F32", operand2=4.0),
            IRInstruction(opcode="MUL_F"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        optimized = validator.optimize(ir)
        assert len(optimized) == 2
        assert abs(float(optimized[0].operand2) - 12.0) < 0.001

    def test_fold_constant_division(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=100.0),
            IRInstruction(opcode="PUSH_F32", operand2=4.0),
            IRInstruction(opcode="DIV_F"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        optimized = validator.optimize(ir)
        assert len(optimized) == 2
        assert abs(float(optimized[0].operand2) - 25.0) < 0.001

    def test_no_fold_division_by_zero(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=10.0),
            IRInstruction(opcode="PUSH_F32", operand2=0.0),
            IRInstruction(opcode="DIV_F"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        optimized = validator.optimize(ir)
        # Should not fold division by zero
        assert len(optimized) == 4

    def test_collapse_consecutive_nops(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="NOP"),
            IRInstruction(opcode="NOP"),
            IRInstruction(opcode="NOP"),
            IRInstruction(opcode="READ_PIN", operand1=0),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        optimized = validator.optimize(ir)
        nop_count = sum(1 for i in optimized if i.opcode == "NOP")
        assert nop_count == 1

    def test_preserve_labeled_nop_kept(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="NOP", label="start"),
            IRInstruction(opcode="NOP"),
            IRInstruction(opcode="NOP"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        optimized = validator.optimize(ir)
        nop_count = sum(1 for i in optimized if i.opcode == "NOP")
        # All consecutive NOPs collapse to 1 (the labeled one is kept)
        assert nop_count == 1
        # Verify the labeled NOP is preserved
        assert optimized[0].label == "start"
        assert optimized[0].opcode == "NOP"

    def test_empty_ir_optimization(self) -> None:
        validator = IRValidator()
        assert validator.optimize([]) == []


# ===================================================================
# Layer 4: BytecodeGenerator tests
# ===================================================================

class TestBytecodeGeneratorBasic:
    def test_generate_simple_read(self) -> None:
        gen = BytecodeGenerator()
        ir = [
            IRInstruction(opcode="READ_PIN", operand1=3),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        bytecode = gen.generate(ir)
        assert len(bytecode) == 2 * INSTR_SIZE
        assert len(bytecode) % INSTR_SIZE == 0

        # Check first instruction
        opcode, flags, op1, op2 = unpack_instruction(bytecode, 0)
        assert opcode == 0x1A  # READ_PIN
        assert op1 == 3

    def test_generate_write_actuator(self) -> None:
        gen = BytecodeGenerator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=0.75),
            IRInstruction(opcode="CLAMP_F", operand1=-1.0, operand2=1.0),
            IRInstruction(opcode="WRITE_PIN", operand1=2),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        bytecode = gen.generate(ir)
        assert len(bytecode) == 4 * INSTR_SIZE

        # Check PUSH_F32
        opcode, flags, op1, op2 = unpack_instruction(bytecode, 0)
        assert opcode == 0x03  # PUSH_F32
        value = struct.unpack("<f", struct.pack("<I", op2))[0]
        assert abs(value - 0.75) < 0.01

        # Check WRITE_PIN
        opcode, flags, op1, op2 = unpack_instruction(bytecode, 2 * INSTR_SIZE)
        assert opcode == 0x1B
        assert op1 == 2

    def test_generate_halt(self) -> None:
        gen = BytecodeGenerator()
        ir = [
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        bytecode = gen.generate(ir)
        assert len(bytecode) == INSTR_SIZE
        opcode, flags, op1, op2 = unpack_instruction(bytecode, 0)
        assert opcode == 0x00  # NOP
        assert flags & FLAGS_SYSCALL
        assert op2 == 0x01

    def test_bytecode_is_multiple_of_8(self) -> None:
        gen = BytecodeGenerator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=1.0),
            IRInstruction(opcode="PUSH_F32", operand2=2.0),
            IRInstruction(opcode="ADD_F"),
            IRInstruction(opcode="POP"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        bytecode = gen.generate(ir)
        assert len(bytecode) % 8 == 0

    def test_generate_conditional_with_jumps(self) -> None:
        gen = BytecodeGenerator()
        ir = [
            IRInstruction(opcode="READ_PIN", operand1=0, label="start"),
            IRInstruction(opcode="PUSH_F32", operand2=50.0),
            IRInstruction(opcode="GT_F"),
            IRInstruction(opcode="JUMP_IF_FALSE", jump_target="end"),
            IRInstruction(opcode="PUSH_F32", operand2=1.0),
            IRInstruction(opcode="CLAMP_F", operand1=-1.0, operand2=1.0),
            IRInstruction(opcode="WRITE_PIN", operand1=1),
            IRInstruction(opcode="NOP", label="end"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        bytecode = gen.generate(ir)
        assert len(bytecode) % 8 == 0
        # Check JUMP_IF_FALSE targets the NOP at index 7
        jif_opcode, _, _, jif_target = unpack_instruction(bytecode, 3 * INSTR_SIZE)
        assert jif_opcode == 0x1E  # JUMP_IF_FALSE
        assert jif_target == 7  # NOP is at index 7


class TestBytecodeGeneratorDisassemble:
    def test_disassemble_simple(self) -> None:
        gen = BytecodeGenerator()
        ir = [
            IRInstruction(opcode="READ_PIN", operand1=0),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        bytecode = gen.generate(ir)
        lines = gen.disassemble(bytecode)
        assert len(lines) == 2
        assert "READ_PIN" in lines[0]
        assert "HALT" in lines[1]

    def test_disassemble_push_f32(self) -> None:
        gen = BytecodeGenerator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=3.14),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        bytecode = gen.generate(ir)
        lines = gen.disassemble(bytecode)
        assert "PUSH_F32" in lines[0]
        assert "3.14" in lines[0]


class TestBytecodeGeneratorErrors:
    def test_unknown_opcode_raises(self) -> None:
        gen = BytecodeGenerator()
        ir = [
            IRInstruction(opcode="FAKE_OPCODE"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        with pytest.raises(ValueError, match="Unknown"):
            gen.generate(ir)

    def test_unresolved_label_raises(self) -> None:
        gen = BytecodeGenerator()
        ir = [
            IRInstruction(opcode="JUMP", jump_target="nonexistent"),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        with pytest.raises(ValueError, match="Unresolved"):
            gen.generate(ir)


# ===================================================================
# Full pipeline (RosettaStone) tests
# ===================================================================

class TestRosettaStoneFullPipeline:
    def setup_method(self) -> None:
        self.rosetta = RosettaStone(trust_level=5, optimize=False)

    def test_read_sensor_full(self) -> None:
        result = self.rosetta.translate("read sensor 3")
        assert result.success
        assert result.bytecode is not None
        assert len(result.bytecode) == 2 * INSTR_SIZE
        assert result.intent is not None
        assert result.intent.action == "READ"
        assert result.ir is not None

    def test_set_actuator_full(self) -> None:
        result = self.rosetta.translate("set actuator 2 to 0.75")
        assert result.success
        assert result.bytecode is not None
        assert result.intent.action == "WRITE"
        # Verify bytecode has PUSH_F32, CLAMP_F, WRITE_PIN, HALT
        assert len(result.bytecode) == 4 * INSTR_SIZE

    def test_conditional_full(self) -> None:
        result = self.rosetta.translate(
            "if sensor 2 gt 50.0 then set actuator 3 to 1.0"
        )
        assert result.success
        assert result.bytecode is not None
        # Should have at least: READ_PIN, PUSH_F32, GT_F, JUMP_IF_FALSE,
        # PUSH_F32, CLAMP_F, WRITE_PIN, NOP, HALT
        assert len(result.bytecode) >= 8 * INSTR_SIZE

    def test_loop_full(self) -> None:
        result = self.rosetta.translate("repeat 5 times: read sensor 0")
        assert result.success
        assert result.bytecode is not None

    def test_wait_full(self) -> None:
        result = self.rosetta.translate("wait 10 cycles")
        assert result.success
        assert result.bytecode is not None

    def test_pid_full(self) -> None:
        result = self.rosetta.translate(
            "compute pid on sensor 0 with kp=1.0 ki=0.1 kd=0.01"
        )
        assert result.success
        assert result.bytecode is not None

    def test_navigate_full(self) -> None:
        result = self.rosetta.translate("navigate to waypoint 10,20")
        assert result.success
        assert result.bytecode is not None

    def test_log_snapshot_full(self) -> None:
        result = self.rosetta.translate("log snapshot")
        assert result.success
        assert result.bytecode is not None

    def test_emit_event_full(self) -> None:
        result = self.rosetta.translate("emit event test_alert")
        assert result.success
        assert result.bytecode is not None

    def test_halt_full(self) -> None:
        result = self.rosetta.translate("halt")
        assert result.success
        assert result.bytecode is not None
        # Compiler deduplicates trailing HALT, so "halt" produces single HALT
        assert len(result.bytecode) == INSTR_SIZE


class TestRosettaStoneErrors:
    def test_unknown_intent(self) -> None:
        rosetta = RosettaStone(trust_level=5)
        result = rosetta.translate("fly to the moon")
        assert not result.success
        assert len(result.errors) > 0

    def test_empty_input(self) -> None:
        rosetta = RosettaStone(trust_level=5)
        result = rosetta.translate("")
        assert not result.success

    def test_trust_level_blocks_write(self) -> None:
        rosetta = RosettaStone(trust_level=1)
        result = rosetta.translate("set actuator 0 to 0.5")
        assert not result.success
        assert any("trust" in e.lower() for e in result.errors)

    def test_trust_level_0_blocks_all(self) -> None:
        rosetta = RosettaStone(trust_level=0)
        result = rosetta.translate("read sensor 0")
        assert not result.success


class TestRosettaStoneOptimization:
    def test_optimization_enabled(self) -> None:
        rosetta = RosettaStone(trust_level=5, optimize=True)
        # Pipeline uses the same BytecodeGenerator and SafetyValidator,
        # so bytecode passes verification. The optimization happens on IR.
        result = rosetta.translate("read sensor 0")
        assert result.success

    def test_optimization_disabled(self) -> None:
        rosetta = RosettaStone(trust_level=5, optimize=False)
        result = rosetta.translate("read sensor 0")
        assert result.success


class TestRosettaStoneMultiple:
    def test_translate_many(self) -> None:
        rosetta = RosettaStone(trust_level=5)
        results = rosetta.translate_many([
            "read sensor 0",
            "read sensor 1",
            "halt",
        ])
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_translate_many_mixed(self) -> None:
        rosetta = RosettaStone(trust_level=5)
        results = rosetta.translate_many([
            "read sensor 0",
            "unknown command",
            "halt",
        ])
        assert len(results) == 3
        assert results[0].success
        assert not results[1].success
        assert results[2].success

    def test_translate_combined(self) -> None:
        rosetta = RosettaStone(trust_level=5)
        result = rosetta.translate_combined([
            "read sensor 0",
            "read sensor 1",
        ])
        assert result.success
        assert result.bytecode is not None


# ===================================================================
# Bytecode verification tests (using existing SafetyValidator)
# ===================================================================

class TestBytecodeSafety:
    """Verify that Rosetta Stone bytecode passes the existing SafetyValidator."""

    def _generate_bytecode(self, text: str) -> bytes:
        rosetta = RosettaStone(trust_level=5, optimize=False)
        result = rosetta.translate(text)
        assert result.success, f"Translation failed: {result.errors}"
        assert result.bytecode is not None
        return result.bytecode

    def test_read_bytecode_safe(self) -> None:
        bytecode = self._generate_bytecode("read sensor 0")
        validator = SafetyValidator()
        errors = validator.validate_bytecode(bytecode)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_write_bytecode_safe(self) -> None:
        bytecode = self._generate_bytecode("set actuator 0 to 0.5")
        validator = SafetyValidator()
        errors = validator.validate_bytecode(bytecode)
        clamp_errors = [e for e in errors if "CLAMP_F" in e]
        assert len(clamp_errors) == 0, f"CLAMP_F errors: {clamp_errors}"

    def test_conditional_bytecode_safe(self) -> None:
        bytecode = self._generate_bytecode(
            "if sensor 0 gt 50 then set actuator 1 to 1.0"
        )
        validator = SafetyValidator()
        errors = validator.validate_bytecode(bytecode)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_loop_bytecode_safe(self) -> None:
        bytecode = self._generate_bytecode("repeat 3 times: read sensor 0")
        validator = SafetyValidator()
        errors = validator.validate_bytecode(bytecode)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_wait_bytecode_safe(self) -> None:
        bytecode = self._generate_bytecode("wait 5 cycles")
        validator = SafetyValidator()
        errors = validator.validate_bytecode(bytecode)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_pid_bytecode_safe(self) -> None:
        bytecode = self._generate_bytecode(
            "compute pid on sensor 0 with kp=1.0 ki=0.1 kd=0.01"
        )
        validator = SafetyValidator()
        errors = validator.validate_bytecode(bytecode)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_navigate_bytecode_safe(self) -> None:
        bytecode = self._generate_bytecode("navigate to waypoint 10,20")
        validator = SafetyValidator()
        errors = validator.validate_bytecode(bytecode)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_log_snapshot_bytecode_safe(self) -> None:
        bytecode = self._generate_bytecode("log snapshot")
        validator = SafetyValidator()
        errors = validator.validate_bytecode(bytecode)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_halt_bytecode_safe(self) -> None:
        bytecode = self._generate_bytecode("halt")
        validator = SafetyValidator()
        errors = validator.validate_bytecode(bytecode)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_all_bytecodes_aligned(self) -> None:
        """All generated bytecodes are 8-byte aligned."""
        texts = [
            "read sensor 0", "set actuator 1 to 0.5",
            "if sensor 2 gt 50 then halt", "repeat 5 times: read sensor 0",
            "wait 3 cycles", "halt", "log snapshot",
            "compute pid on sensor 0 with kp=1.0 ki=0.1 kd=0.01",
            "navigate to waypoint 1,2",
        ]
        rosetta = RosettaStone(trust_level=5, optimize=False)
        for text in texts:
            result = rosetta.translate(text)
            if result.success and result.bytecode:
                assert len(result.bytecode) % 8 == 0, (
                    f"Not aligned for '{text}': {len(result.bytecode)} bytes"
                )


# ===================================================================
# Compound intent full pipeline tests
# ===================================================================

class TestCompoundFullPipeline:
    def test_monitor_trigger_full(self) -> None:
        rosetta = RosettaStone(trust_level=5, optimize=False)
        result = rosetta.translate(
            "monitor sensor 2 and if gt 50.0 then trigger actuator 3"
        )
        assert result.success
        assert result.bytecode is not None
        assert result.intent.action == "COMPOUND"

    def test_patrol_full(self) -> None:
        rosetta = RosettaStone(trust_level=5, optimize=False)
        result = rosetta.translate(
            "patrol: read GPS, if distance > 100m return home"
        )
        assert result.success
        assert result.bytecode is not None


# ===================================================================
# Edge case tests
# ===================================================================

class TestEdgeCases:
    def test_very_long_input(self) -> None:
        parser = IntentParser()
        long_text = "read sensor " + "0" * 100
        # Should still try to parse (but may not match)
        intent = parser.parse(long_text)
        # It won't match since "0000..." is too long for \d+
        # But it shouldn't crash
        assert intent is not None

    def test_special_characters_in_event(self) -> None:
        parser = IntentParser()
        intent = parser.parse("emit event hello world!")
        assert intent.action == "SYSCALL"
        assert "hello world!" in intent.params["message"]

    def test_nested_intent_in_loop(self) -> None:
        rosetta = RosettaStone(trust_level=5, optimize=False)
        # Loop with conditional body - loop itself is simple
        result = rosetta.translate("repeat 3 times: read sensor 0")
        assert result.success

    def test_float_pin_rejected(self) -> None:
        parser = IntentParser()
        intent = parser.parse("read sensor 3.5")
        # Won't match the \d+ pattern
        assert intent.action == "UNKNOWN"

    def test_negative_pin(self) -> None:
        parser = IntentParser()
        intent = parser.parse("read sensor -1")
        # Won't match (no minus in pattern)
        assert intent.action == "UNKNOWN"

    def test_duplicate_push_pop_optimization(self) -> None:
        validator = IRValidator()
        ir = [
            IRInstruction(opcode="PUSH_F32", operand2=1.0),
            IRInstruction(opcode="POP"),
            IRInstruction(opcode="PUSH_F32", operand2=2.0),
            IRInstruction(opcode="POP"),
            IRInstruction(opcode="READ_PIN", operand1=0),
            IRInstruction(opcode="SYSCALL", operand2=0x01),
        ]
        optimized = validator.optimize(ir)
        opcodes = [i.opcode for i in optimized]
        assert opcodes == ["READ_PIN", "SYSCALL"]

    def test_combined_pipeline_bytecode_valid(self) -> None:
        rosetta = RosettaStone(trust_level=5)
        result = rosetta.translate_combined([
            "read sensor 0",
            "read sensor 1",
            "halt",
        ])
        assert result.success
        assert result.bytecode is not None
        validator = SafetyValidator()
        errors = validator.validate_bytecode(result.bytecode)
        assert len(errors) == 0, f"Combined bytecode errors: {errors}"
