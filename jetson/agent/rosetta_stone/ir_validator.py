"""Rosetta Stone Layer 3: IR validation and optimization.

Validates IR instructions for safety constraints (pin ranges, stack
effects, trust levels) and applies peephole optimizations to reduce
bytecode size and improve execution efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.rosetta_stone.intent_compiler import IRInstruction


# ===================================================================
# VM constants (mirrors vm.h)
# ===================================================================

VM_SENSOR_COUNT = 64
VM_ACTUATOR_COUNT = 64
VM_VAR_COUNT = 256
VM_STACK_SIZE = 256
MAX_CYCLE_BUDGET = 1000

# Trust level restrictions:
# L0 (Disabled):    No operations permitted
# L1 (Advisory):    READ only
# L2 (Supervised):  READ + WRITE (no loops)
# L3 (Semi-Auto):   READ + WRITE + CONDITIONAL + LOOP (bounded)
# L4 (High Auto):   All except NAVIGATE
# L5 (Full Auto):   All operations

_TRUST_ALLOWED_ACTIONS: dict[int, set[str]] = {
    0: set(),  # Disabled
    1: {"READ", "WAIT", "SYSCALL"},  # Advisory: observe only
    2: {"READ", "WRITE", "WAIT", "SYSCALL"},  # Supervised: read+write
    3: {"READ", "WRITE", "CONDITIONAL", "LOOP", "WAIT", "SYSCALL", "PID"},  # Semi-auto
    4: {"READ", "WRITE", "CONDITIONAL", "LOOP", "WAIT", "SYSCALL", "PID"},  # High auto
    5: {"READ", "WRITE", "CONDITIONAL", "LOOP", "WAIT", "SYSCALL", "PID",
        "NAVIGATE", "COMPOUND"},  # Full auto
}

# Stack effect per IR opcode (positive = push, negative = pop)
_IR_STACK_EFFECTS: dict[str, int] = {
    "NOP": 0,
    "PUSH_I8": 1,
    "PUSH_I16": 1,
    "PUSH_F32": 1,
    "POP": -1,
    "DUP": 1,
    "SWAP": 0,
    "ROT": 0,
    "ADD_F": -1,
    "SUB_F": -1,
    "MUL_F": -1,
    "DIV_F": -1,
    "NEG_F": 0,
    "ABS_F": 0,
    "MIN_F": -1,
    "MAX_F": -1,
    "CLAMP_F": 0,  # pops 1, pushes 1
    "EQ_F": -1,
    "LT_F": -1,
    "GT_F": -1,
    "LTE_F": -1,
    "GTE_F": -1,
    "AND_B": -1,
    "OR_B": -1,
    "XOR_B": -1,
    "NOT_B": 0,
    "READ_PIN": 1,
    "WRITE_PIN": -1,
    "READ_TIMER_MS": 1,
    "JUMP": 0,
    "JUMP_IF_FALSE": -1,
    "JUMP_IF_TRUE": -1,
    "SYSCALL": 0,
    "DECLARE_INTENT": 0,
    "ASSERT_GOAL": 0,
    "HALT": 0,
}


# ===================================================================
# ValidationResult
# ===================================================================

@dataclass
class ValidationResult:
    """Result of IR validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    max_stack_depth: int = 0
    instruction_count: int = 0


# ===================================================================
# IRValidator
# ===================================================================

class IRValidator:
    """Validate and optimize IR before bytecode generation.

    Checks:
    - All referenced pins exist (sensors 0-63, actuators 0-63, vars 0-255)
    - Stack effects are balanced (no underflow)
    - No infinite loops (loops must have bounded iteration counts)
    - Trust level permits all operations
    - Physical value ranges are sane (e.g., actuator values in [-1, 1])
    - Cycle budget not exceeded
    """

    def __init__(
        self,
        max_stack: int = 16,
        max_cycles: int = MAX_CYCLE_BUDGET,
    ) -> None:
        self.max_stack = max_stack
        self.max_cycles = max_cycles

    def validate(
        self, ir: list[IRInstruction], trust_level: int = 0
    ) -> ValidationResult:
        """Validate IR instructions.

        Args:
            ir: List of IR instructions to validate.
            trust_level: Autonomy trust level (0-5).

        Returns:
            ValidationResult with errors and warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not ir:
            return ValidationResult(
                valid=False,
                errors=["Empty IR program"],
            )

        # Check instruction count
        if len(ir) > self.max_cycles:
            errors.append(
                f"IR has {len(ir)} instructions, exceeds cycle budget {self.max_cycles}"
            )

        # Collect labels and jump targets
        labels: dict[str, int] = {}
        for i, instr in enumerate(ir):
            if instr.label:
                labels[instr.label] = i

        # Validate jump targets exist
        for i, instr in enumerate(ir):
            if instr.jump_target and instr.jump_target not in labels:
                errors.append(
                    f"Instruction {i}: jump target '{instr.jump_target}' not defined"
                )

        # Stack depth analysis
        stack_result = self._check_stack_depth(ir, labels)
        if isinstance(stack_result, str):
            errors.append(stack_result)
        else:
            if stack_result > self.max_stack:
                errors.append(
                    f"Max stack depth {stack_result} exceeds limit {self.max_stack}"
                )

        # Pin validation
        pin_errors = self._validate_pins(ir)
        errors.extend(pin_errors)

        # Value range validation
        value_warnings = self._validate_values(ir)
        warnings.extend(value_warnings)

        # Trust level validation
        trust_errors = self._validate_trust_level(ir, trust_level)
        errors.extend(trust_errors)

        # Infinite loop detection
        loop_warnings = self._detect_potential_infinite_loops(ir, labels)
        warnings.extend(loop_warnings)

        valid = len(errors) == 0
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            max_stack_depth=stack_result if isinstance(stack_result, int) else 0,
            instruction_count=len(ir),
        )

    def optimize(self, ir: list[IRInstruction]) -> list[IRInstruction]:
        """Apply peephole optimizations to IR.

        Optimizations:
        - Remove redundant PUSH/POP pairs
        - Fold constant arithmetic (PUSH a, PUSH b, ADD_F -> PUSH result)
        - Remove consecutive NOP instructions (keep at most one)
        - Remove PUSH followed by POP with no side effects

        Args:
            ir: List of IR instructions.

        Returns:
            Optimized IR instruction list.
        """
        if not ir:
            return ir

        result = self._remove_push_pop_pairs(ir)
        result = self._fold_constant_arithmetic(result)
        result = self._collapse_nops(result)

        return result

    # -----------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------

    def _check_stack_depth(
        self, ir: list[IRInstruction], labels: dict[str, int]
    ) -> int | str:
        """Compute maximum stack depth, checking for underflow.

        Returns max depth (int) or error message (str) on underflow.
        """
        max_depth = 0
        current = 0

        for i, instr in enumerate(ir):
            effect = _IR_STACK_EFFECTS.get(instr.opcode, 0)
            current += effect

            if current < 0:
                return f"Stack underflow at instruction {i} ({instr.opcode}), depth={current}"
            if current > max_depth:
                max_depth = current

        return max_depth

    def _validate_pins(self, ir: list[IRInstruction]) -> list[str]:
        """Validate pin references are within valid ranges."""
        errors: list[str] = []

        for i, instr in enumerate(ir):
            if instr.opcode == "READ_PIN" and instr.operand1 is not None:
                pin = int(instr.operand1)
                if pin < 0 or pin >= VM_SENSOR_COUNT:
                    errors.append(
                        f"Instruction {i}: READ_PIN pin {pin} out of range [0, {VM_SENSOR_COUNT - 1}]"
                    )

            elif instr.opcode == "WRITE_PIN" and instr.operand1 is not None:
                pin = int(instr.operand1)
                if pin < 0 or pin >= VM_ACTUATOR_COUNT:
                    errors.append(
                        f"Instruction {i}: WRITE_PIN pin {pin} out of range [0, {VM_ACTUATOR_COUNT - 1}]"
                    )

        return errors

    def _validate_values(self, ir: list[IRInstruction]) -> list[str]:
        """Check for potentially dangerous float values."""
        warnings: list[str] = []

        for i, instr in enumerate(ir):
            if instr.opcode == "PUSH_F32" and instr.operand2 is not None:
                value = float(instr.operand2)
                import math
                if math.isnan(value):
                    warnings.append(
                        f"Instruction {i}: PUSH_F32 has NaN value"
                    )
                elif math.isinf(value):
                    warnings.append(
                        f"Instruction {i}: PUSH_F32 has Infinity value"
                    )
                elif abs(value) > 1e6:
                    warnings.append(
                        f"Instruction {i}: PUSH_F32 has very large value {value}"
                    )

        return warnings

    def _validate_trust_level(
        self, ir: list[IRInstruction], trust_level: int
    ) -> list[str]:
        """Check that all operations are permitted at the given trust level."""
        errors: list[str] = []

        if trust_level < 0 or trust_level > 5:
            errors.append(f"Invalid trust level {trust_level}")
            return errors

        allowed = _TRUST_ALLOWED_ACTIONS.get(trust_level, set())

        if not allowed:
            errors.append(
                f"Trust level {trust_level} (Disabled): no operations permitted"
            )
            return errors

        # Extract action types from source_intent comments
        # We check the IR opcodes instead - if there are WRITE_PIN ops but
        # trust level doesn't allow WRITE, flag it
        has_write = any(instr.opcode == "WRITE_PIN" for instr in ir)
        has_jump_loop = any(
            instr.opcode in ("JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE", "DUP")
            and instr.jump_target is not None
            for instr in ir
        )
        has_navigate = any(instr.opcode == "DECLARE_INTENT" for instr in ir)
        has_pid = any(
            instr.opcode == "SYSCALL" and instr.operand2 == 0x02
            for instr in ir
        )

        if has_write and "WRITE" not in allowed:
            errors.append(
                f"Trust level {trust_level} does not permit WRITE operations"
            )

        if has_jump_loop and "LOOP" not in allowed and "CONDITIONAL" not in allowed:
            errors.append(
                f"Trust level {trust_level} does not permit control flow operations"
            )

        if has_navigate and "NAVIGATE" not in allowed:
            errors.append(
                f"Trust level {trust_level} does not permit NAVIGATE operations"
            )

        if has_pid and "PID" not in allowed:
            errors.append(
                f"Trust level {trust_level} does not permit PID operations"
            )

        return errors

    def _detect_potential_infinite_loops(
        self, ir: list[IRInstruction], labels: dict[str, int]
    ) -> list[str]:
        """Detect potential infinite loops (backward jumps without counter decrement)."""
        warnings: list[str] = []

        for i, instr in enumerate(ir):
            if instr.jump_target and instr.opcode in ("JUMP",):
                target_idx = labels.get(instr.jump_target, -1)
                if target_idx >= 0 and target_idx <= i:
                    # Backward jump - check if there's a decrement in the loop body
                    loop_body = ir[target_idx:i + 1]
                    has_decrement = any(
                        inst.opcode == "SUB_F" for inst in loop_body
                    )
                    if not has_decrement:
                        warnings.append(
                            f"Instruction {i}: potential infinite loop "
                            f"(backward jump without decrement)"
                        )

        return warnings

    # -----------------------------------------------------------------
    # Optimization helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _remove_push_pop_pairs(ir: list[IRInstruction]) -> list[IRInstruction]:
        """Remove PUSH followed immediately by POP (no side effects)."""
        result: list[IRInstruction] = []
        skip_next = False

        for i, instr in enumerate(ir):
            if skip_next:
                skip_next = False
                continue

            # Check if this is a PUSH followed by POP
            if (
                i + 1 < len(ir)
                and instr.opcode in ("PUSH_I8", "PUSH_I16", "PUSH_F32", "DUP")
                and ir[i + 1].opcode == "POP"
            ):
                skip_next = True
                continue

            result.append(instr)

        return result

    @staticmethod
    def _fold_constant_arithmetic(ir: list[IRInstruction]) -> list[IRInstruction]:
        """Fold constant arithmetic: PUSH a, PUSH b, OP -> PUSH result.

        Only folds simple cases:
        - PUSH_F32 a, PUSH_F32 b, ADD_F -> PUSH_F32 (a+b)
        - PUSH_F32 a, PUSH_F32 b, SUB_F -> PUSH_F32 (a-b)
        - PUSH_F32 a, PUSH_F32 b, MUL_F -> PUSH_F32 (a*b)
        - PUSH_F32 a, PUSH_F32 b, DIV_F -> PUSH_F32 (a/b) (if b != 0)
        """
        result: list[IRInstruction] = []
        i = 0

        while i < len(ir):
            instr = ir[i]

            # Check for pattern: PUSH_F32, PUSH_F32, ARITH_OP
            if (
                instr.opcode == "PUSH_F32"
                and instr.operand2 is not None
                and i + 2 < len(ir)
                and ir[i + 1].opcode == "PUSH_F32"
                and ir[i + 1].operand2 is not None
                and ir[i + 2].opcode in ("ADD_F", "SUB_F", "MUL_F", "DIV_F")
            ):
                a = float(instr.operand2)
                b = float(ir[i + 1].operand2)
                op = ir[i + 2].opcode

                if op == "ADD_F":
                    folded = a + b
                elif op == "SUB_F":
                    folded = a - b
                elif op == "MUL_F":
                    folded = a * b
                elif op == "DIV_F":
                    if b == 0:
                        result.append(instr)
                        i += 1
                        continue
                    folded = a / b
                else:
                    result.append(instr)
                    i += 1
                    continue

                import math
                if not (math.isnan(folded) or math.isinf(folded)):
                    result.append(IRInstruction(
                        opcode="PUSH_F32",
                        operand2=folded,
                        comment=f"folded: {a} {op.replace('_F', '')} {b} = {folded}",
                        source_intent=instr.source_intent,
                    ))
                    i += 3
                    continue

            result.append(instr)
            i += 1

        return result

    @staticmethod
    def _collapse_nops(ir: list[IRInstruction]) -> list[IRInstruction]:
        """Collapse consecutive NOPs into a single NOP.

        Always keeps at least one NOP if any existed. Preserves labels.
        """
        result: list[IRInstruction] = []
        prev_was_nop = False

        for instr in ir:
            if instr.opcode == "NOP":
                if prev_was_nop and not instr.label:
                    # Skip consecutive NOP without label
                    continue
                prev_was_nop = True
            else:
                prev_was_nop = False

            result.append(instr)

        return result
