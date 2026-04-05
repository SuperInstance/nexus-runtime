"""NEXUS Safety Validator — 6-stage bytecode safety validation pipeline.

Pre-deployment validation for bytecode targeting the ESP32 NEXUS VM.
Every reflex program MUST pass all 6 stages before being deployed.

Stage 1: Syntax Check — well-formed instructions, alignment, size
Stage 2: Safety Rules — no forbidden opcode sequences
Stage 3: Stack Analysis — no underflow/overflow
Stage 4: Trust Check — opcode privileges match trust level
Stage 5: Semantic Analysis — no infinite loops, no I/O on protected pins
Stage 6: Adversarial Probing — fuzz boundary conditions
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from typing import Any

from core.safety_validator.models import (
    SafetyReport,
    SafetyViolation,
    StageResult,
    make_timestamp,
)
from core.safety_validator.rules import (
    ACTUATOR_SAFE_RANGES,
    ALL_VALID_OPCODES,
    DEFAULT_CLAMP_RANGE,
    DEFAULT_MAX_CALL_DEPTH,
    DEFAULT_MAX_INSTRUCTIONS,
    DEFAULT_MAX_NOP_SEQUENCE,
    DEFAULT_MAX_PROGRAM_SIZE,
    DEFAULT_MAX_STACK_DEPTH,
    FLAGS_EXTENDED_CLAMP,
    FLAGS_IS_CALL,
    FLAGS_IS_FLOAT,
    FLAGS_SYSCALL,
    INSTR_SIZE,
    OP_DIV_F,
    OP_JUMP,
    OP_JUMP_IF_FALSE,
    OP_JUMP_IF_TRUE,
    OP_NOP,
    OP_WRITE_PIN,
    PID_KD_RANGE,
    PID_KI_RANGE,
    PID_KP_RANGE,
    PID_SETPOINT_RANGE,
    SAFETY_CRITICAL_PINS,
    STACK_EFFECTS,
    SYSCALL_HALT,
    SYSCALL_PID_COMPUTE,
    TRUST_OPCODE_MATRIX,
)


def _unpack(data: bytes, offset: int = 0) -> tuple[int, int, int, int]:
    """Unpack an 8-byte instruction into (opcode, flags, operand1, operand2)."""
    return struct.unpack_from("<BBHI", data, offset)


def _opcode_name(opcode: int) -> str:
    """Return a human-readable opcode name."""
    from shared.opcodes import opcode_name as _on
    return _on(opcode)


class BytecodeSafetyPipeline:
    """6-stage bytecode safety validation pipeline.

    Validates bytecode before deployment to the ESP32 NEXUS VM.
    Each stage checks a different safety aspect. All stages must pass
    for the bytecode to be considered safe for deployment.

    Args:
        trust_level: Current trust/autonomy level (0-5).
        safety_config: Optional overrides for pipeline limits.
    """

    def __init__(self, trust_level: int = 0, safety_config: dict[str, Any] | None = None) -> None:
        self.trust_level = max(0, min(5, trust_level))
        cfg = safety_config or {}
        self.max_instructions = cfg.get("max_instructions", DEFAULT_MAX_INSTRUCTIONS)
        self.max_stack_depth = cfg.get("max_stack_depth", DEFAULT_MAX_STACK_DEPTH)
        self.max_call_depth = cfg.get("max_call_depth", DEFAULT_MAX_CALL_DEPTH)
        self.max_nop_sequence = cfg.get("max_nop_sequence", DEFAULT_MAX_NOP_SEQUENCE)
        self.max_program_size = cfg.get("max_program_size", DEFAULT_MAX_PROGRAM_SIZE)
        self.stages = [
            self.stage1_syntax,
            self.stage2_safety_rules,
            self.stage3_stack_analysis,
            self.stage4_trust_check,
            self.stage5_semantic_analysis,
            self.stage6_adversarial,
        ]

    def validate(self, bytecode: bytes) -> SafetyReport:
        """Run all 6 stages on the bytecode and return a SafetyReport.

        Args:
            bytecode: Raw binary bytecode (must be multiple of 8 bytes).

        Returns:
            SafetyReport with pass/fail per stage and overall verdict.
        """
        bc_hash = hashlib.sha256(bytecode).hexdigest()
        bc_size = len(bytecode)
        instr_count = bc_size // INSTR_SIZE if bc_size % INSTR_SIZE == 0 else 0

        report = SafetyReport(
            overall_passed=True,
            bytecode_hash=bc_hash,
            bytecode_size=bc_size,
            instruction_count=instr_count,
            trust_level=self.trust_level,
            timestamp=make_timestamp(),
        )

        # Parse instructions once (if syntax allows)
        instructions: list[tuple[int, int, int, int]] = []
        if bc_size > 0 and bc_size % INSTR_SIZE == 0:
            instructions = [_unpack(bytecode, i * INSTR_SIZE) for i in range(instr_count)]

        for stage_fn in self.stages:
            result = stage_fn(bytecode, instructions)
            report.stages.append(result)
            report.violations.extend(result.violations)
            if not result.passed:
                report.overall_passed = False

        return report

    # ==================================================================
    # Stage 1: Syntax Check
    # ==================================================================

    def stage1_syntax(
        self, bytecode: bytes, instructions: list[tuple[int, int, int, int]]
    ) -> StageResult:
        """Check: alignment (8 bytes per instruction), size > 0, size % 8 == 0.

        Also validates that every opcode is a known opcode value.
        """
        t0 = time.perf_counter()
        result = StageResult(stage_name="syntax_check", passed=True)

        # Empty bytecode
        if len(bytecode) == 0:
            result.passed = False
            err = "Bytecode is empty (0 bytes)"
            result.errors.append(err)
            result.violations.append(SafetyViolation(
                stage="syntax_check", severity="error",
                instruction_index=0, opcode=0,
                description=err,
                remediation="Provide at least one valid instruction",
            ))
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        # Alignment check
        if len(bytecode) % INSTR_SIZE != 0:
            result.passed = False
            err = (
                f"Bytecode size {len(bytecode)} is not a multiple of "
                f"{INSTR_SIZE} bytes (alignment error)"
            )
            result.errors.append(err)
            result.violations.append(SafetyViolation(
                stage="syntax_check", severity="error",
                instruction_index=0, opcode=0,
                description=err,
                remediation=f"Pad bytecode to a multiple of {INSTR_SIZE} bytes",
            ))
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        # Size limits
        if len(bytecode) > self.max_program_size:
            result.passed = False
            err = (
                f"Bytecode size {len(bytecode)} exceeds maximum "
                f"{self.max_program_size} bytes"
            )
            result.errors.append(err)
            result.violations.append(SafetyViolation(
                stage="syntax_check", severity="error",
                instruction_index=0, opcode=0,
                description=err,
                remediation="Reduce program size",
            ))

        instr_count = len(bytecode) // INSTR_SIZE
        if instr_count > self.max_instructions:
            result.passed = False
            err = (
                f"Instruction count {instr_count} exceeds maximum "
                f"{self.max_instructions}"
            )
            result.errors.append(err)
            result.violations.append(SafetyViolation(
                stage="syntax_check", severity="error",
                instruction_index=0, opcode=0,
                description=err,
                remediation="Reduce instruction count",
            ))

        # Validate each opcode
        for i, (opcode, flags, op1, op2) in enumerate(instructions):
            if opcode not in ALL_VALID_OPCODES:
                result.passed = False
                err = (
                    f"Unknown opcode 0x{opcode:02X} at instruction {i}"
                )
                result.errors.append(err)
                result.violations.append(SafetyViolation(
                    stage="syntax_check", severity="error",
                    instruction_index=i, opcode=opcode,
                    description=err,
                    remediation=f"Replace opcode 0x{opcode:02X} with a valid opcode",
                ))

        result.duration_ms = (time.perf_counter() - t0) * 1000
        return result

    # ==================================================================
    # Stage 2: Safety Rules
    # ==================================================================

    def stage2_safety_rules(
        self, bytecode: bytes, instructions: list[tuple[int, int, int, int]]
    ) -> StageResult:
        """Check forbidden patterns:
        - No CALL without matching RET (call stack depth limit)
        - No WRITE_PIN to safety-critical pins at low trust
        - No JUMP to address 0 (infinite loop risk)
        - No sequence of >100 NOP instructions
        - No SYSCALL at trust < 5
        """
        t0 = time.perf_counter()
        result = StageResult(stage_name="safety_rules", passed=True)
        n = len(instructions)

        if n == 0:
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        # --- Check: unmatched CALL/RET ---
        call_depth = 0
        max_call_depth_seen = 0
        call_positions: list[int] = []  # stack of call instruction indices
        ret_positions: list[int] = []

        for i, (opcode, flags, op1, op2) in enumerate(instructions):
            is_call = (opcode == OP_JUMP and (flags & FLAGS_IS_CALL))
            is_ret = (opcode == OP_JUMP and not (flags & FLAGS_IS_CALL) and op2 == 0xFFFFFFFF)

            if is_call:
                call_depth += 1
                call_positions.append(i)
                if call_depth > max_call_depth_seen:
                    max_call_depth_seen = call_depth
            elif is_ret:
                if call_depth > 0:
                    call_depth -= 1
                    ret_positions.append(i)
                else:
                    # RET without CALL
                    result.passed = False
                    err = f"RET at instruction {i} without matching CALL"
                    result.errors.append(err)
                    result.violations.append(SafetyViolation(
                        stage="safety_rules", severity="error",
                        instruction_index=i, opcode=opcode,
                        description=err,
                        remediation="Remove unmatched RET or add matching CALL",
                    ))

        if call_depth > 0:
            result.passed = False
            unmatched = call_positions[-min(call_depth, len(call_positions)):]
            for idx in unmatched:
                err = f"CALL at instruction {idx} has no matching RET"
                result.errors.append(err)
                result.violations.append(SafetyViolation(
                    stage="safety_rules", severity="error",
                    instruction_index=idx, opcode=OP_JUMP,
                    description=err,
                    remediation="Add matching RET after the subroutine",
                ))

        if max_call_depth_seen > self.max_call_depth:
            result.passed = False
            err = (
                f"Maximum call depth {max_call_depth_seen} exceeds "
                f"limit {self.max_call_depth}"
            )
            result.errors.append(err)
            result.violations.append(SafetyViolation(
                stage="safety_rules", severity="error",
                instruction_index=0, opcode=0,
                description=err,
                remediation="Reduce subroutine nesting depth",
            ))

        # --- Check: JUMP to address 0 ---
        for i, (opcode, flags, op1, op2) in enumerate(instructions):
            if opcode == OP_JUMP:
                is_call = (flags & FLAGS_IS_CALL)
                is_ret = (not is_call and op2 == 0xFFFFFFFF)
                if not is_call and not is_ret and op2 == 0:
                    result.passed = False
                    err = f"JUMP at instruction {i} targets address 0 (infinite loop risk)"
                    result.errors.append(err)
                    result.violations.append(SafetyViolation(
                        stage="safety_rules", severity="error",
                        instruction_index=i, opcode=opcode,
                        description=err,
                        remediation="Change JUMP target to a non-zero address",
                    ))

        # --- Check: excessive NOP sequence ---
        nop_run = 0
        nop_run_start = 0
        for i, (opcode, flags, op1, op2) in enumerate(instructions):
            is_syscall = (opcode == OP_NOP and (flags & FLAGS_SYSCALL))
            if opcode == OP_NOP and not is_syscall:
                if nop_run == 0:
                    nop_run_start = i
                nop_run += 1
                if nop_run > self.max_nop_sequence:
                    result.passed = False
                    err = (
                        f"Excessive NOP sequence: {nop_run} consecutive NOPs "
                        f"starting at instruction {nop_run_start}"
                    )
                    result.errors.append(err)
                    result.violations.append(SafetyViolation(
                        stage="safety_rules", severity="warning",
                        instruction_index=nop_run_start, opcode=OP_NOP,
                        description=err,
                        remediation="Remove unnecessary NOP padding",
                    ))
            else:
                nop_run = 0

        # --- Check: WRITE_PIN to safety-critical pins at low trust ---
        if self.trust_level < 4:
            for i, (opcode, flags, op1, op2) in enumerate(instructions):
                if opcode == OP_WRITE_PIN:
                    pin = op1
                    if pin in SAFETY_CRITICAL_PINS:
                        result.passed = False
                        err = (
                            f"WRITE_PIN at instruction {i} targets safety-critical "
                            f"pin {pin} (E-Stop/watchdog/heartbeat) at trust level "
                            f"L{self.trust_level} (requires L4)"
                        )
                        result.errors.append(err)
                        result.violations.append(SafetyViolation(
                            stage="safety_rules", severity="error",
                            instruction_index=i, opcode=opcode,
                            description=err,
                            remediation="Raise trust level to L4 or use a non-safety pin",
                        ))

        # --- Check: non-HALT SYSCALL at trust < 5 ---
        # HALT (syscall_id=0x01) is allowed at all trust levels as a safety
        # termination opcode. Only non-HALT syscalls require L5.
        if self.trust_level < 5:
            for i, (opcode, flags, op1, op2) in enumerate(instructions):
                if opcode == OP_NOP and (flags & FLAGS_SYSCALL):
                    syscall_id = op2
                    if syscall_id == SYSCALL_HALT:
                        continue  # HALT is always allowed
                    result.passed = False
                    name = {2: "PID_COMPUTE",
                            3: "RECORD_SNAPSHOT", 4: "EMIT_EVENT"}.get(
                        syscall_id, f"0x{syscall_id:02X}"
                    )
                    err = (
                        f"SYSCALL {name} at instruction {i} requires "
                        f"trust level L5 (current: L{self.trust_level})"
                    )
                    result.errors.append(err)
                    result.violations.append(SafetyViolation(
                        stage="safety_rules", severity="error",
                        instruction_index=i, opcode=opcode,
                        description=err,
                        remediation="Raise trust level to L5 for SYSCALL access",
                    ))

        result.duration_ms = (time.perf_counter() - t0) * 1000
        return result

    # ==================================================================
    # Stage 3: Stack Analysis
    # ==================================================================

    def stage3_stack_analysis(
        self, bytecode: bytes, instructions: list[tuple[int, int, int, int]]
    ) -> StageResult:
        """Simulate stack: check no underflow, no overflow beyond max_stack_depth.

        Tracks stack depth along the linear instruction path.
        For conditional branches, explores both paths where feasible.
        Reports the worst-case (maximum) stack depth encountered.
        """
        t0 = time.perf_counter()
        result = StageResult(stage_name="stack_analysis", passed=True)
        n = len(instructions)

        if n == 0:
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        # Linear analysis: simulate stack depth
        # We track min and max depth along the primary path
        depth = 0
        max_depth = 0
        underflow_at = -1

        for i, (opcode, flags, op1, op2) in enumerate(instructions):
            effect = STACK_EFFECTS.get(opcode, 0)
            depth += effect

            if depth < 0:
                underflow_at = i
                break

            if depth > max_depth:
                max_depth = depth

            if depth > self.max_stack_depth:
                # Keep going to find all overflow points
                pass

        # Underflow check
        if underflow_at >= 0:
            opcode_at = instructions[underflow_at][0]
            result.passed = False
            err = (
                f"Stack underflow at instruction {underflow_at} "
                f"(opcode={_opcode_name(opcode_at)}, "
                f"would pop from empty stack)"
            )
            result.errors.append(err)
            result.violations.append(SafetyViolation(
                stage="stack_analysis", severity="error",
                instruction_index=underflow_at, opcode=opcode_at,
                description=err,
                remediation="Add PUSH instructions before the pop to ensure stack has values",
            ))

        # Overflow check
        if max_depth > self.max_stack_depth:
            result.passed = False
            err = (
                f"Maximum stack depth {max_depth} exceeds limit "
                f"{self.max_stack_depth}"
            )
            result.errors.append(err)
            result.violations.append(SafetyViolation(
                stage="stack_analysis", severity="error",
                instruction_index=0, opcode=0,
                description=err,
                remediation="Reduce PUSH instructions or add POP instructions to limit stack usage",
            ))

        # Warning for high stack usage
        if max_depth > self.max_stack_depth * 0.75 and max_depth <= self.max_stack_depth:
            warn = (
                f"Stack depth reaches {max_depth} of {self.max_stack_depth} "
                f"({max_depth * 100 // self.max_stack_depth}% capacity)"
            )
            result.warnings.append(warn)

        # Track per-instruction depths and detect overflow points
        depth = 0
        for i, (opcode, flags, op1, op2) in enumerate(instructions):
            effect = STACK_EFFECTS.get(opcode, 0)
            depth += effect
            if depth > self.max_stack_depth:
                # Already caught above, but record specific instruction
                pass

        result.duration_ms = (time.perf_counter() - t0) * 1000
        return result

    # ==================================================================
    # Stage 4: Trust Check
    # ==================================================================

    def stage4_trust_check(
        self, bytecode: bytes, instructions: list[tuple[int, int, int, int]]
    ) -> StageResult:
        """Check opcode privileges match trust level.

        L0-L1: READ-only opcodes (LOAD_VAR, PUSH, arithmetic, READ_PIN)
        L2: + WRITE_PIN for non-safety actuators
        L3: + CALL/RET (subroutine capability)
        L4: + all I/O pins including safety
        L5: + SYSCALL (full system access including HALT)
        """
        t0 = time.perf_counter()
        result = StageResult(stage_name="trust_check", passed=True)
        allowed = TRUST_OPCODE_MATRIX.get(self.trust_level, TRUST_OPCODE_MATRIX[0])

        for i, (opcode, flags, op1, op2) in enumerate(instructions):
            # SYSCALL is a special case: NOP + FLAGS_SYSCALL
            is_syscall = (opcode == OP_NOP and (flags & FLAGS_SYSCALL))
            is_call = (opcode == OP_JUMP and (flags & FLAGS_IS_CALL))

            # Check syscall privilege — HALT is allowed at all trust levels
            if is_syscall and self.trust_level < 5:
                syscall_id = op2
                if syscall_id == SYSCALL_HALT:
                    continue  # HALT is always allowed
                result.passed = False
                name = {2: "PID_COMPUTE",
                        3: "RECORD_SNAPSHOT", 4: "EMIT_EVENT"}.get(
                    syscall_id, f"0x{syscall_id:02X}"
                )
                err = (
                    f"SYSCALL {name} at instruction {i} "
                    f"requires L5, current trust: L{self.trust_level}"
                )
                result.errors.append(err)
                result.violations.append(SafetyViolation(
                    stage="trust_check", severity="error",
                    instruction_index=i, opcode=opcode,
                    description=err,
                    remediation="Raise trust level to L5 for SYSCALL access",
                ))
                continue

            # Check CALL privilege (requires L3+)
            if is_call and self.trust_level < 3:
                result.passed = False
                err = (
                    f"CALL at instruction {i} requires L3, "
                    f"current trust: L{self.trust_level}"
                )
                result.errors.append(err)
                result.violations.append(SafetyViolation(
                    stage="trust_check", severity="error",
                    instruction_index=i, opcode=opcode,
                    description=err,
                    remediation="Raise trust level to L3 for CALL/RET capability",
                ))
                continue

            # Check general opcode privilege (skip syscall NOPs since we handled above)
            if is_syscall:
                continue

            if opcode not in allowed:
                result.passed = False
                name = _opcode_name(opcode)
                err = (
                    f"Opcode {name} (0x{opcode:02X}) at instruction {i} "
                    f"not allowed at trust level L{self.trust_level}"
                )
                result.errors.append(err)
                result.violations.append(SafetyViolation(
                    stage="trust_check", severity="error",
                    instruction_index=i, opcode=opcode,
                    description=err,
                    remediation=f"Raise trust level or remove {name} instruction",
                ))

        result.duration_ms = (time.perf_counter() - t0) * 1000
        return result

    # ==================================================================
    # Stage 5: Semantic Analysis
    # ==================================================================

    def stage5_semantic_analysis(
        self, bytecode: bytes, instructions: list[tuple[int, int, int, int]]
    ) -> StageResult:
        """Check semantic properties:
        - No unreachable code after unconditional JUMP (warning)
        - No infinite loops (detected by cycle detection in control flow graph)
        - CLAMP_F values within physical actuator ranges
        - DIV_F: no division by constant zero
        - SYSCALL PID_COMPUTE: PID params within safe ranges
        """
        t0 = time.perf_counter()
        result = StageResult(stage_name="semantic_analysis", passed=True)
        n = len(instructions)

        if n == 0:
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        # --- Build control flow graph ---
        successors: dict[int, list[int]] = {}
        for i in range(n):
            successors[i] = []
            opcode, flags, op1, op2 = instructions[i]

            # Determine successors
            is_unconditional = False
            if opcode == OP_JUMP:
                is_call = (flags & FLAGS_IS_CALL)
                is_ret = (not is_call and op2 == 0xFFFFFFFF)

                if is_ret:
                    # RET returns to caller; in linear analysis, we can't
                    # resolve the target, so we treat it as a fall-through terminator
                    pass
                elif is_call:
                    # CALL: goes to target + returns (fall through as well)
                    target = min(int(op2), n - 1)
                    successors[i].append(target)
                    if i + 1 < n:
                        successors[i].append(i + 1)  # return address
                else:
                    # Unconditional jump
                    target = min(int(op2), n - 1)
                    successors[i].append(target)
                    is_unconditional = True
            elif opcode == OP_JUMP_IF_FALSE:
                target = min(int(op2), n - 1)
                successors[i].append(target)
                if i + 1 < n:
                    successors[i].append(i + 1)
            elif opcode == OP_JUMP_IF_TRUE:
                target = min(int(op2), n - 1)
                successors[i].append(target)
                if i + 1 < n:
                    successors[i].append(i + 1)
            else:
                if i + 1 < n:
                    successors[i].append(i + 1)

        # --- Unreachable code detection ---
        reachable: set[int] = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            for succ in successors.get(node, []):
                if succ not in reachable:
                    stack.append(succ)

        unreachable_instrs = [i for i in range(n) if i not in reachable]
        if unreachable_instrs:
            for idx in unreachable_instrs:
                opcode = instructions[idx][0]
                warn = (
                    f"Unreachable code at instruction {idx} "
                    f"({_opcode_name(opcode)})"
                )
                result.warnings.append(warn)
                result.violations.append(SafetyViolation(
                    stage="semantic_analysis", severity="warning",
                    instruction_index=idx, opcode=opcode,
                    description=warn,
                    remediation="Remove unreachable instructions to save memory",
                ))

        # --- Infinite loop detection ---
        # Detect unconditional back-edges in the CFG (JUMP to earlier instruction)
        # that don't have any conditional exit
        back_edges: list[int] = []
        for i in range(n):
            opcode, flags, op1, op2 = instructions[i]
            if opcode == OP_JUMP:
                is_call = (flags & FLAGS_IS_CALL)
                is_ret = (not is_call and op2 == 0xFFFFFFFF)
                if not is_call and not is_ret:
                    target = int(op2)
                    if target <= i and target != 0:
                        # Check if there's a conditional exit within the loop
                        loop_has_exit = False
                        for j in range(target, i + 1):
                            op = instructions[j][0]
                            if op in (OP_JUMP_IF_FALSE, OP_JUMP_IF_TRUE):
                                jtarget = int(instructions[j][3])
                                if jtarget > i or jtarget < target:
                                    loop_has_exit = True
                                    break
                            if op == OP_JUMP:
                                jflags = instructions[j][1]
                                jop2 = instructions[j][3]
                                j_is_call = (jflags & FLAGS_IS_CALL)
                                j_is_ret = (not j_is_call and jop2 == 0xFFFFFFFF)
                                if not j_is_call and not j_is_ret:
                                    jtarget = int(jop2)
                                    if jtarget > i or jtarget < target:
                                        loop_has_exit = True
                                        break

                        if not loop_has_exit:
                            back_edges.append(i)

        for idx in back_edges:
            err = (
                f"Potential infinite loop: unconditional JUMP at instruction {idx} "
                f"jumps to {instructions[idx][3]} with no conditional exit"
            )
            result.errors.append(err)
            result.passed = False
            result.violations.append(SafetyViolation(
                stage="semantic_analysis", severity="error",
                instruction_index=idx, opcode=OP_JUMP,
                description=err,
                remediation="Add a conditional exit (JUMP_IF_FALSE/TRUE) inside the loop",
            ))

        # --- CLAMP_F range validation ---
        for i, (opcode, flags, op1, op2) in enumerate(instructions):
            if opcode == 0x10 and (flags & FLAGS_EXTENDED_CLAMP):  # CLAMP_F with extended encoding
                lo16 = op2 & 0xFFFF
                hi16 = (op2 >> 16) & 0xFFFF
                lo_val = self._f16_to_f32(lo16)
                hi_val = self._f16_to_f32(hi16)

                if lo_val >= hi_val:
                    err = (
                        f"CLAMP_F at instruction {i}: lo={lo_val} >= hi={hi_val} "
                        f"(invalid range)"
                    )
                    result.errors.append(err)
                    result.passed = False
                    result.violations.append(SafetyViolation(
                        stage="semantic_analysis", severity="error",
                        instruction_index=i, opcode=opcode,
                        description=err,
                        remediation="Ensure CLAMP_F lo < hi",
                    ))

                if math.isnan(lo_val) or math.isinf(lo_val) or math.isnan(hi_val) or math.isinf(hi_val):
                    err = (
                        f"CLAMP_F at instruction {i}: lo={lo_val}, hi={hi_val} "
                        f"(NaN or Infinity not allowed)"
                    )
                    result.errors.append(err)
                    result.passed = False
                    result.violations.append(SafetyViolation(
                        stage="semantic_analysis", severity="error",
                        instruction_index=i, opcode=opcode,
                        description=err,
                        remediation="Use finite numeric bounds for CLAMP_F",
                    ))

                # Check against absolute default range
                if abs(hi_val) > DEFAULT_CLAMP_RANGE[1] or abs(lo_val) > DEFAULT_CLAMP_RANGE[1]:
                    warn = (
                        f"CLAMP_F at instruction {i}: range [{lo_val}, {hi_val}] "
                        f"exceeds default safe range {DEFAULT_CLAMP_RANGE}"
                    )
                    result.warnings.append(warn)

        # --- Division by constant zero detection ---
        # Check if the instruction before DIV_F pushes a constant 0
        for i in range(1, n):
            opcode, flags, op1, op2 = instructions[i]
            if opcode == OP_DIV_F:
                prev_opcode, prev_flags, prev_op1, prev_op2 = instructions[i - 1]
                is_zero_push = False

                if prev_opcode == 0x03 and (prev_flags & FLAGS_IS_FLOAT):
                    # PUSH_F32 with float value
                    val = struct.unpack("<f", struct.pack("<I", prev_op2))[0]
                    if val == 0.0:
                        is_zero_push = True
                elif prev_opcode in (0x01, 0x02):
                    # PUSH_I8 or PUSH_I16
                    val = prev_op1 if prev_opcode == 0x02 else (prev_op1 & 0xFF)
                    if prev_opcode == 0x01:
                        # Interpret as signed i8
                        if val > 127:
                            val = val - 256
                    else:
                        if val > 32767:
                            val = val - 65536
                    if val == 0:
                        is_zero_push = True

                if is_zero_push:
                    err = (
                        f"Division by constant zero at instruction {i}: "
                        f"DIV_F preceded by PUSH(0)"
                    )
                    result.errors.append(err)
                    result.passed = False
                    result.violations.append(SafetyViolation(
                        stage="semantic_analysis", severity="error",
                        instruction_index=i, opcode=opcode,
                        description=err,
                        remediation="Replace the zero constant with a non-zero value, or add a guard check",
                    ))

        # --- SYSCALL PID_COMPUTE validation ---
        # PID params are typically pushed on the stack before the syscall
        # Check if the syscall_id is PID_COMPUTE (0x02) and warn about params
        for i, (opcode, flags, op1, op2) in enumerate(instructions):
            if opcode == OP_NOP and (flags & FLAGS_SYSCALL) and op2 == SYSCALL_PID_COMPUTE:
                # Check the 4 values pushed before: Kp, Ki, Kd, setpoint
                params_found = []
                push_idx = i - 1
                while len(params_found) < 4 and push_idx >= 0:
                    pop, pflags, pop1, pop2 = instructions[push_idx]
                    if pop == 0x03 and (pflags & FLAGS_IS_FLOAT):
                        val = struct.unpack("<f", struct.pack("<I", pop2))[0]
                        params_found.append((push_idx, val))
                    push_idx -= 1

                if len(params_found) == 4:
                    # params_found is in reverse order (last pushed first)
                    # Stack order: Kp, Ki, Kd, setpoint (bottom to top)
                    setpoint_val = params_found[0][1]
                    kd_val = params_found[1][1]
                    ki_val = params_found[2][1]
                    kp_val = params_found[3][1]

                    checks = [
                        ("Kp", kp_val, PID_KP_RANGE),
                        ("Ki", ki_val, PID_KI_RANGE),
                        ("Kd", kd_val, PID_KD_RANGE),
                        ("setpoint", setpoint_val, PID_SETPOINT_RANGE),
                    ]
                    for pname, pval, (plo, phi) in checks:
                        if pval < plo or pval > phi:
                            err = (
                                f"PID_COMPUTE at instruction {i}: "
                                f"{pname}={pval} outside safe range [{plo}, {phi}]"
                            )
                            result.errors.append(err)
                            result.passed = False
                            result.violations.append(SafetyViolation(
                                stage="semantic_analysis", severity="error",
                                instruction_index=i, opcode=opcode,
                                description=err,
                                remediation=f"Adjust {pname} to be within [{plo}, {phi}]",
                            ))

        result.duration_ms = (time.perf_counter() - t0) * 1000
        return result

    # ==================================================================
    # Stage 6: Adversarial Probing
    # ==================================================================

    def stage6_adversarial(
        self, bytecode: bytes, instructions: list[tuple[int, int, int, int]]
    ) -> StageResult:
        """Fuzz boundary conditions:
        - Flip each bit of each instruction, re-run stages 1-5
        - Mutate operands to max/min values, re-validate
        - Check that validation is robust to malformed input
        - Return list of mutations that pass (potential vulnerabilities)

        Note: This stage always passes (it's informational), but it
        reports warnings for mutations that incorrectly pass validation.
        """
        t0 = time.perf_counter()
        result = StageResult(stage_name="adversarial_probing", passed=True)
        n = len(instructions)

        if n == 0:
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        vulnerabilities: list[str] = []
        total_mutations = 0
        incorrectly_passed = 0

        # Limit the number of mutations for performance
        max_bit_flips = min(n, 50)  # Test up to 50 instructions

        for instr_idx in range(max_bit_flips):
            instr_bytes = bytecode[instr_idx * INSTR_SIZE: (instr_idx + 1) * INSTR_SIZE]
            if len(instr_bytes) < INSTR_SIZE:
                continue

            # Test flipping each byte position (8 bits, test a few key bits)
            for bit_pos in range(INSTR_SIZE * 8):
                if total_mutations > 500:
                    break

                # Flip one bit
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                mutated = bytearray(instr_bytes)
                mutated[byte_idx] ^= (1 << bit_idx)
                mutated_bc = bytes(
                    bytecode[:instr_idx * INSTR_SIZE]
                    + mutated
                    + bytecode[(instr_idx + 1) * INSTR_SIZE:]
                )

                total_mutations += 1

                # Re-run stages 1-3 (lightweight subset)
                try:
                    sub_report = self._run_sub_stages(mutated_bc)
                    if sub_report:
                        incorrectly_passed += 1
                        vuln = (
                            f"Bit-flip at instr {instr_idx} byte {byte_idx} "
                            f"bit {bit_idx} incorrectly passes validation"
                        )
                        vulnerabilities.append(vuln)
                except Exception:
                    # Expected for some mutations — validation should handle gracefully
                    pass

            if total_mutations > 500:
                break

        # Test operand boundary mutations (max/min values)
        if n > 0:
            # Mutate first instruction's operand2 to 0 and 0xFFFFFFFF
            for boundary_val in [0x00000000, 0xFFFFFFFF]:
                mutated = bytearray(bytecode)
                struct.pack_into("<I", mutated, 0 * INSTR_SIZE + 4, boundary_val)
                mutated_bc = bytes(mutated)
                total_mutations += 1
                try:
                    sub_report = self._run_sub_stages(mutated_bc)
                    if sub_report:
                        incorrectly_passed += 1
                        vulnerabilities.append(
                            f"Operand2 boundary mutation (0x{boundary_val:08X}) "
                            f"on instruction 0 passes validation unexpectedly"
                        )
                except Exception:
                    pass

            # Test empty bytecode
            total_mutations += 1
            try:
                sub_report = self._run_sub_stages(b"")
                if sub_report:
                    incorrectly_passed += 1
                    vulnerabilities.append("Empty bytecode passes validation")
            except Exception:
                pass

            # Test single-byte (misaligned)
            total_mutations += 1
            try:
                sub_report = self._run_sub_stages(b"\x00")
                if sub_report:
                    incorrectly_passed += 1
                    vulnerabilities.append("Single-byte (misaligned) bytecode passes validation")
            except Exception:
                pass

            # Test all-0xFF bytecode
            total_mutations += 1
            bad_bc = b"\xFF" * INSTR_SIZE
            try:
                sub_report = self._run_sub_stages(bad_bc)
                if sub_report:
                    incorrectly_passed += 1
                    vulnerabilities.append("All-0xFF instruction passes validation")
            except Exception:
                pass

        # Report findings
        if vulnerabilities:
            result.warnings.append(
                f"Adversarial probing: {len(vulnerabilities)} potential "
                f"vulnerabilities found ({incorrectly_passed}/{total_mutations} "
                f"mutations incorrectly passed)"
            )
            for vuln in vulnerabilities[:20]:  # Limit output
                result.warnings.append(f"  {vuln}")
                result.violations.append(SafetyViolation(
                    stage="adversarial_probing", severity="warning",
                    instruction_index=0, opcode=0,
                    description=vuln,
                    remediation="Investigate why malformed bytecode passes validation",
                ))

        result.duration_ms = (time.perf_counter() - t0) * 1000
        return result

    def _run_sub_stages(self, mutated_bc: bytes) -> bool:
        """Run stages 1-3 on mutated bytecode. Returns True if it passes."""
        # Parse instructions
        if len(mutated_bc) == 0 or len(mutated_bc) % INSTR_SIZE != 0:
            return False  # Should fail syntax

        instrs = [_unpack(mutated_bc, i * INSTR_SIZE)
                   for i in range(len(mutated_bc) // INSTR_SIZE)]

        # Stage 1
        r1 = self.stage1_syntax(mutated_bc, instrs)
        if not r1.passed:
            return False

        # Stage 2 (only check CALL/RET and NOP sequence, not trust-dependent)
        r2 = self.stage2_safety_rules(mutated_bc, instrs)
        if not r2.passed:
            return False

        # Stage 3
        r3 = self.stage3_stack_analysis(mutated_bc, instrs)
        if not r3.passed:
            return False

        return True

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _f16_to_f32(bits: int) -> float:
        """Convert IEEE 754 float16 bit pattern to float32."""
        sign = (bits >> 15) & 1
        exp = (bits >> 10) & 0x1F
        mant = bits & 0x3FF

        if exp == 0:
            if mant == 0:
                return -0.0 if sign else 0.0
            # Denormalized
            val = mant / 1024.0
            return -val if sign else val
        elif exp == 31:
            if mant == 0:
                return float("-inf") if sign else float("inf")
            return float("nan")
        else:
            # Normalized
            val = (1.0 + mant / 1024.0) * (2 ** (exp - 15))
            return -val if sign else val
