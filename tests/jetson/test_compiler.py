"""NEXUS Jetson tests - Reflex compiler tests."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "jetson"))

from reflex_compiler.compiler import ReflexCompiler
from reflex_compiler.bytecode_emitter import BytecodeEmitter


class TestReflexCompiler:
    """Reflex compiler tests."""

    def test_compiler_exists(self) -> None:
        """Verify compiler can be instantiated."""
        compiler = ReflexCompiler()
        assert compiler is not None

    def test_compile_empty(self) -> None:
        """Verify empty reflex compiles to empty bytecode."""
        compiler = ReflexCompiler()
        result = compiler.compile({})
        assert result == b""

    def test_validate_empty(self) -> None:
        """Verify empty reflex has no validation errors."""
        compiler = ReflexCompiler()
        errors = compiler.validate({})
        assert errors == []


class TestBytecodeEmitter:
    """Bytecode emitter tests."""

    def test_initial_state(self) -> None:
        """Verify emitter starts empty."""
        emitter = BytecodeEmitter()
        assert emitter.instruction_count() == 0
        assert emitter.get_bytecode() == b""

    def test_emit_nop(self) -> None:
        """Verify NOP emission."""
        emitter = BytecodeEmitter()
        emitter.emit_nop()
        assert emitter.instruction_count() == 1
        bytecode = emitter.get_bytecode()
        assert len(bytecode) == 8

    def test_emit_halt(self) -> None:
        """Verify HALT emission."""
        emitter = BytecodeEmitter()
        emitter.emit_halt()
        assert emitter.instruction_count() == 1
        bytecode = emitter.get_bytecode()
        assert bytecode[1] == 0x80  # SYSCALL flag

    def test_reset(self) -> None:
        """Verify emitter reset."""
        emitter = BytecodeEmitter()
        emitter.emit_nop()
        emitter.reset()
        assert emitter.instruction_count() == 0
