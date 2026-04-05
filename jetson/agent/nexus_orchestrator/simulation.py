"""NEXUS Orchestrator — Mission simulation engine.

Simulates bytecode execution in a virtual VM without real hardware.
Tracks stack operations, I/O reads/writes, trust events, cycle count,
and safety violations during simulation.
"""

from __future__ import annotations

import struct
import math
from dataclasses import dataclass, field
from typing import Any

from core.safety_validator.rules import (
    INSTR_SIZE,
    STACK_EFFECTS,
    FLAGS_SYSCALL,
    FLAGS_IS_FLOAT,
    FLAGS_HAS_IMMEDIATE,
)


# ── Data structures ───────────────────────────────────────────────

@dataclass
class SimStep:
    """Single step of simulated bytecode execution."""

    cycle: int
    opcode: int
    opcode_name: str
    flags: int
    operand1: int
    operand2: int
    stack_before: list[float] = field(default_factory=list)
    stack_after: list[float] = field(default_factory=list)
    action: str = ""  # human-readable description
    is_io_read: bool = False
    is_io_write: bool = False
    io_pin: int = -1
    io_value: float = 0.0
    is_syscall: bool = False
    is_halt: bool = False


@dataclass
class SimulationResult:
    """Result of simulating bytecode execution."""

    success: bool = True
    total_cycles: int = 0
    max_cycles_reached: bool = False
    halted: bool = False
    halt_reason: str = ""
    final_stack: list[float] = field(default_factory=list)
    max_stack_depth: int = 0
    steps: list[SimStep] = field(default_factory=list)
    io_reads: list[dict[str, Any]] = field(default_factory=list)
    io_writes: list[dict[str, Any]] = field(default_factory=list)
    trust_events: list[dict[str, Any]] = field(default_factory=list)
    safety_violations: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class ComparisonResult:
    """Result of comparing two bytecode programs side by side."""

    bytecode_a_hash: str = ""
    bytecode_b_hash: str = ""
    cycles_a: int = 0
    cycles_b: int = 0
    io_reads_match: bool = True
    io_writes_match: bool = True
    final_stack_a: list[float] = field(default_factory=list)
    final_stack_b: list[float] = field(default_factory=list)
    result_a: SimulationResult = field(default_factory=SimulationResult)
    result_b: SimulationResult = field(default_factory=SimulationResult)
    differences: list[str] = field(default_factory=list)
    verdict: str = ""  # IDENTICAL, EQUIVALENT, DIFFERENT


# ── Opcode name lookup ────────────────────────────────────────────

def _opcode_name(opcode: int) -> str:
    from shared.opcodes import opcode_name as _on
    return _on(opcode)


# ── MissionSimulator ──────────────────────────────────────────────

class MissionSimulator:
    """Simulates bytecode execution without real hardware.

    Creates a virtual VM state, executes bytecode, tracks:
    - Stack operations
    - I/O reads (returns simulated sensor values)
    - I/O writes (records actuator commands)
    - Trust events
    - Cycle count
    - Safety violations

    The simulator uses the same 8-byte instruction format as the ESP32 VM.
    """

    def __init__(self, max_stack: int = 32) -> None:
        self.max_stack = max_stack

    def simulate(
        self,
        bytecode: bytes,
        sensor_data: dict[int, float] | None = None,
        max_cycles: int = 1000,
    ) -> SimulationResult:
        """Execute bytecode in simulation.

        Args:
            bytecode: Raw NEXUS bytecode bytes.
            sensor_data: Optional dict mapping pin numbers to simulated
                         sensor values. Default returns 0.0 for all pins.
            max_cycles: Maximum number of cycles (instructions) to execute.

        Returns:
            SimulationResult with full execution trace.
        """
        sensors = sensor_data or {}
        result = SimulationResult()

        if len(bytecode) == 0 or len(bytecode) % INSTR_SIZE != 0:
            result.success = False
            result.error = "Invalid bytecode: empty or misaligned"
            return result

        n_instr = len(bytecode) // INSTR_SIZE
        stack: list[float] = []
        pc = 0  # program counter (instruction index)
        cycle = 0
        result.total_cycles = 0
        result.max_stack_depth = 0

        while cycle < max_cycles and pc < n_instr:
            offset = pc * INSTR_SIZE
            opcode, flags, op1, op2 = struct.unpack_from("<BBHI", bytecode, offset)
            op_name = _opcode_name(opcode)
            is_syscall = (opcode == 0x00 and (flags & FLAGS_SYSCALL))

            stack_before = list(stack)

            step = SimStep(
                cycle=cycle,
                opcode=opcode,
                opcode_name=op_name,
                flags=flags,
                operand1=op1,
                operand2=op2,
                stack_before=stack_before,
            )

            # ── HALT ──
            if is_syscall and op2 == 0x01:
                step.is_halt = True
                step.is_syscall = True
                step.action = "HALT"
                step.stack_after = list(stack)
                result.steps.append(step)
                result.halted = True
                result.halt_reason = "SYSCALL HALT"
                result.total_cycles = cycle + 1
                result.final_stack = list(stack)
                return result

            # ── NOP (non-syscall) ──
            if opcode == 0x00 and not is_syscall:
                step.action = "NOP"
                step.stack_after = list(stack)
                result.steps.append(step)
                pc += 1
                cycle += 1
                continue

            # ── PUSH_I8 ──
            if opcode == 0x01:
                val = op1
                if val > 127:
                    val -= 256
                stack.append(float(val))
                step.action = f"PUSH_I8 {val}"
                step.stack_after = list(stack)
                result.steps.append(step)
                pc += 1
                cycle += 1
                self._update_max_stack(stack, result)
                continue

            # ── PUSH_I16 ──
            if opcode == 0x02:
                val = op1
                if val > 32767:
                    val -= 65536
                stack.append(float(val))
                step.action = f"PUSH_I16 {val}"
                step.stack_after = list(stack)
                result.steps.append(step)
                pc += 1
                cycle += 1
                self._update_max_stack(stack, result)
                continue

            # ── PUSH_F32 ──
            if opcode == 0x03 and (flags & FLAGS_IS_FLOAT):
                raw_bytes = struct.pack("<I", op2)
                val = struct.unpack("<f", raw_bytes)[0]
                stack.append(val)
                step.action = f"PUSH_F32 {val}"
                step.stack_after = list(stack)
                result.steps.append(step)
                pc += 1
                cycle += 1
                self._update_max_stack(stack, result)
                continue

            # ── POP ──
            if opcode == 0x04:
                if stack:
                    stack.pop()
                else:
                    result.safety_violations.append(
                        f"Cycle {cycle}: POP on empty stack"
                    )
                step.action = "POP"
                step.stack_after = list(stack)
                result.steps.append(step)
                pc += 1
                cycle += 1
                continue

            # ── DUP ──
            if opcode == 0x05:
                if stack:
                    stack.append(stack[-1])
                else:
                    result.safety_violations.append(
                        f"Cycle {cycle}: DUP on empty stack"
                    )
                step.action = "DUP"
                step.stack_after = list(stack)
                result.steps.append(step)
                pc += 1
                cycle += 1
                self._update_max_stack(stack, result)
                continue

            # ── SWAP ──
            if opcode == 0x06:
                if len(stack) >= 2:
                    stack[-1], stack[-2] = stack[-2], stack[-1]
                step.action = "SWAP"
                step.stack_after = list(stack)
                result.steps.append(step)
                pc += 1
                cycle += 1
                continue

            # ── ROT ──
            if opcode == 0x07:
                if len(stack) >= 3:
                    a, b, c = stack[-3], stack[-2], stack[-1]
                    stack[-3], stack[-2], stack[-1] = c, a, b
                step.action = "ROT"
                step.stack_after = list(stack)
                result.steps.append(step)
                pc += 1
                cycle += 1
                continue

            # ── Arithmetic ──
            if opcode == 0x08:  # ADD_F
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(a + b)
                    step.action = f"ADD_F {a} + {b} = {stack[-1]}"
                else:
                    result.safety_violations.append(
                        f"Cycle {cycle}: ADD_F needs 2 stack values"
                    )

            elif opcode == 0x09:  # SUB_F
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(a - b)
                    step.action = f"SUB_F {a} - {b} = {stack[-1]}"
                else:
                    result.safety_violations.append(
                        f"Cycle {cycle}: SUB_F needs 2 stack values"
                    )

            elif opcode == 0x0A:  # MUL_F
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(a * b)
                    step.action = f"MUL_F {a} * {b} = {stack[-1]}"
                else:
                    result.safety_violations.append(
                        f"Cycle {cycle}: MUL_F needs 2 stack values"
                    )

            elif opcode == 0x0B:  # DIV_F
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    if b == 0.0:
                        result.safety_violations.append(
                            f"Cycle {cycle}: DIV_F by zero"
                        )
                        stack.append(0.0)
                    else:
                        stack.append(a / b)
                    step.action = f"DIV_F {a} / {b}"
                else:
                    result.safety_violations.append(
                        f"Cycle {cycle}: DIV_F needs 2 stack values"
                    )

            elif opcode == 0x0C:  # NEG_F
                if stack:
                    stack[-1] = -stack[-1]
                    step.action = f"NEG_F -> {stack[-1]}"

            elif opcode == 0x0D:  # ABS_F
                if stack:
                    stack[-1] = abs(stack[-1])
                    step.action = f"ABS_F -> {stack[-1]}"

            elif opcode == 0x0E:  # MIN_F
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(min(a, b))
                    step.action = f"MIN_F min({a}, {b}) = {stack[-1]}"

            elif opcode == 0x0F:  # MAX_F
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(max(a, b))
                    step.action = f"MAX_F max({a}, {b}) = {stack[-1]}"

            elif opcode == 0x10:  # CLAMP_F
                # Extended CLAMP_F: lo/hi encoded in operand2
                lo16 = op2 & 0xFFFF
                hi16 = (op2 >> 16) & 0xFFFF
                lo = self._f16_to_f32(lo16)
                hi = self._f16_to_f32(hi16)
                if stack:
                    stack[-1] = max(lo, min(hi, stack[-1]))
                    step.action = f"CLAMP_F [{lo}, {hi}] -> {stack[-1]}"
                else:
                    step.action = f"CLAMP_F [{lo}, {hi}] (no stack value)"

            # ── Comparison ──
            elif opcode == 0x11:  # EQ_F
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(1.0 if a == b else 0.0)
                    step.action = f"EQ_F {a} == {b} -> {stack[-1]}"

            elif opcode == 0x12:  # LT_F
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(1.0 if a < b else 0.0)
                    step.action = f"LT_F {a} < {b} -> {stack[-1]}"

            elif opcode == 0x13:  # GT_F
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(1.0 if a > b else 0.0)
                    step.action = f"GT_F {a} > {b} -> {stack[-1]}"

            elif opcode == 0x14:  # LTE_F
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(1.0 if a <= b else 0.0)
                    step.action = f"LTE_F {a} <= {b} -> {stack[-1]}"

            elif opcode == 0x15:  # GTE_F
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(1.0 if a >= b else 0.0)
                    step.action = f"GTE_F {a} >= {b} -> {stack[-1]}"

            # ── I/O ──
            elif opcode == 0x1A:  # READ_PIN
                pin = op1
                val = sensors.get(pin, 0.0)
                stack.append(val)
                step.is_io_read = True
                step.io_pin = pin
                step.io_value = val
                step.action = f"READ_PIN pin={pin} -> {val}"
                result.io_reads.append({
                    "cycle": cycle, "pin": pin, "value": val,
                })

            elif opcode == 0x1B:  # WRITE_PIN
                pin = op1
                val = stack.pop() if stack else 0.0
                step.is_io_write = True
                step.io_pin = pin
                step.io_value = val
                step.action = f"WRITE_PIN pin={pin} value={val}"
                result.io_writes.append({
                    "cycle": cycle, "pin": pin, "value": val,
                })

            elif opcode == 0x1C:  # READ_TIMER_MS
                stack.append(float(cycle))
                step.action = f"READ_TIMER_MS -> {cycle}"

            # ── Control flow ──
            elif opcode == 0x1D:  # JUMP
                target = op2
                if target >= n_instr:
                    result.safety_violations.append(
                        f"Cycle {cycle}: JUMP to {target} (max {n_instr})"
                    )
                    step.action = f"JUMP -> {target} (OUT OF BOUNDS)"
                else:
                    step.action = f"JUMP -> {target}"
                    step.stack_after = list(stack)
                    result.steps.append(step)
                    pc = target
                    cycle += 1
                    continue

            elif opcode == 0x1E:  # JUMP_IF_FALSE
                target = op2
                cond = stack.pop() if stack else 0.0
                if cond == 0.0:
                    step.action = f"JUMP_IF_FALSE (0.0) -> {target}"
                    step.stack_after = list(stack)
                    result.steps.append(step)
                    if target < n_instr:
                        pc = target
                    else:
                        pc += 1
                    cycle += 1
                    continue
                else:
                    step.action = f"JUMP_IF_FALSE ({cond}) -> no jump"

            elif opcode == 0x1F:  # JUMP_IF_TRUE
                target = op2
                cond = stack.pop() if stack else 0.0
                if cond != 0.0:
                    step.action = f"JUMP_IF_TRUE ({cond}) -> {target}"
                    step.stack_after = list(stack)
                    result.steps.append(step)
                    if target < n_instr:
                        pc = target
                    else:
                        pc += 1
                    cycle += 1
                    continue
                else:
                    step.action = f"JUMP_IF_TRUE (0.0) -> no jump"

            # ── Logic ──
            elif opcode == 0x16:  # AND_B
                if len(stack) >= 2:
                    b, a = int(stack.pop()), int(stack.pop())
                    stack.append(float(a & b))
                    step.action = f"AND_B {a} & {b}"

            elif opcode == 0x17:  # OR_B
                if len(stack) >= 2:
                    b, a = int(stack.pop()), int(stack.pop())
                    stack.append(float(a | b))
                    step.action = f"OR_B {a} | {b}"

            elif opcode == 0x18:  # XOR_B
                if len(stack) >= 2:
                    b, a = int(stack.pop()), int(stack.pop())
                    stack.append(float(a ^ b))
                    step.action = f"XOR_B {a} ^ {b}"

            elif opcode == 0x19:  # NOT_B
                if stack:
                    stack[-1] = float(~int(stack[-1]))
                    step.action = f"NOT_B -> {stack[-1]}"

            # ── Syscalls (non-HALT) ──
            elif is_syscall:
                step.is_syscall = True
                if op2 == 0x02:
                    step.action = "SYSCALL PID_COMPUTE"
                elif op2 == 0x03:
                    step.action = "SYSCALL RECORD_SNAPSHOT"
                    result.trust_events.append({
                        "cycle": cycle, "type": "snapshot", "data": "recorded",
                    })
                elif op2 == 0x04:
                    step.action = f"SYSCALL EMIT_EVENT event_id={op1}"
                else:
                    step.action = f"SYSCALL id={op2}"

            # ── A2A opcodes (treated as NOP) ──
            elif opcode >= 0x20 and opcode <= 0x56:
                step.action = f"A2A:{op_name} (NOP on sim)"

            # ── Unknown ──
            else:
                step.action = f"UNKNOWN opcode 0x{opcode:02X}"
                result.safety_violations.append(
                    f"Cycle {cycle}: Unknown opcode 0x{opcode:02X}"
                )

            step.stack_after = list(stack)
            result.steps.append(step)

            self._update_max_stack(stack, result)
            pc += 1
            cycle += 1

        # Check termination
        result.total_cycles = cycle
        result.final_stack = list(stack)

        if cycle >= max_cycles:
            result.max_cycles_reached = True
            result.success = False
            result.error = f"Exceeded max cycles ({max_cycles})"

        return result

    def compare_bytecode(
        self,
        bytecode_a: bytes,
        bytecode_b: bytes,
        sensor_data: dict[int, float] | None = None,
        max_cycles: int = 1000,
    ) -> ComparisonResult:
        """Compare two bytecode programs side by side (A/B testing).

        Args:
            bytecode_a: First bytecode program.
            bytecode_b: Second bytecode program.
            sensor_data: Shared simulated sensor data.
            max_cycles: Maximum cycles per simulation.

        Returns:
            ComparisonResult with differences and verdict.
        """
        import hashlib

        result_a = self.simulate(bytecode_a, sensor_data, max_cycles)
        result_b = self.simulate(bytecode_b, sensor_data, max_cycles)

        hash_a = hashlib.sha256(bytecode_a).hexdigest()[:16]
        hash_b = hashlib.sha256(bytecode_b).hexdigest()[:16]

        comp = ComparisonResult(
            bytecode_a_hash=hash_a,
            bytecode_b_hash=hash_b,
            cycles_a=result_a.total_cycles,
            cycles_b=result_b.total_cycles,
            final_stack_a=result_a.final_stack,
            final_stack_b=result_b.final_stack,
            result_a=result_a,
            result_b=result_b,
        )

        diffs: list[str] = []

        # Compare cycles
        if result_a.total_cycles != result_b.total_cycles:
            diffs.append(
                f"Cycle count differs: A={result_a.total_cycles} B={result_b.total_cycles}"
            )

        # Compare I/O reads
        if len(result_a.io_reads) != len(result_b.io_reads):
            diffs.append(
                f"IO read count differs: A={len(result_a.io_reads)} B={len(result_b.io_reads)}"
            )
        else:
            for i, (ra, rb) in enumerate(zip(result_a.io_reads, result_b.io_reads)):
                if ra.get("pin") != rb.get("pin") or ra.get("value") != rb.get("value"):
                    diffs.append(f"IO read {i} differs: A={ra} B={rb}")
                    comp.io_reads_match = False

        # Compare I/O writes
        if len(result_a.io_writes) != len(result_b.io_writes):
            diffs.append(
                f"IO write count differs: A={len(result_a.io_writes)} B={len(result_b.io_writes)}"
            )
        else:
            for i, (wa, wb) in enumerate(zip(result_a.io_writes, result_b.io_writes)):
                if wa.get("pin") != wb.get("pin") or wa.get("value") != wb.get("value"):
                    diffs.append(f"IO write {i} differs: A={wa} B={wb}")
                    comp.io_writes_match = False

        # Compare final stacks
        if result_a.final_stack != result_b.final_stack:
            diffs.append(
                f"Final stack differs: A={result_a.final_stack} B={result_b.final_stack}"
            )

        comp.differences = diffs

        if not diffs:
            comp.verdict = "IDENTICAL"
        elif (comp.io_reads_match and comp.io_writes_match
              and result_a.final_stack == result_b.final_stack):
            comp.verdict = "EQUIVALENT"
        else:
            comp.verdict = "DIFFERENT"

        return comp

    def _update_max_stack(self, stack: list[float], result: SimulationResult) -> None:
        """Track maximum stack depth."""
        if len(stack) > result.max_stack_depth:
            result.max_stack_depth = len(stack)
        if len(stack) > self.max_stack:
            result.safety_violations.append(
                f"Stack depth {len(stack)} exceeds limit {self.max_stack}"
            )

    @staticmethod
    def _f16_to_f32(half_bits: int) -> float:
        """Convert IEEE 754 half-precision (16-bit) to float."""
        sign = (half_bits >> 15) & 0x1
        exponent = (half_bits >> 10) & 0x1F
        mantissa = half_bits & 0x3FF

        if exponent == 0:
            if mantissa == 0:
                return -0.0 if sign else 0.0
            # Subnormal
            val = mantissa / 1024.0
            return -val if sign else val
        elif exponent == 31:
            if mantissa == 0:
                return float('-inf') if sign else float('inf')
            return float('nan')

        val = 1.0 + mantissa / 1024.0
        val *= 2 ** (exponent - 15)
        return -val if sign else val
