"""Rosetta Stone Layer 2: Intent-to-IR compiler.

Compiles structured Intent objects into an intermediate representation
(IR) that maps 1:1 to NEXUS bytecode instructions but uses human-readable
names and carries source-intent tracing information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.rosetta_stone.intent_parser import Intent


# ===================================================================
# IR Instruction data structure
# ===================================================================

@dataclass
class IRInstruction:
    """Single instruction in the intermediate representation.

    Maps 1:1 to an 8-byte NEXUS bytecode instruction but with
    human-readable names, optional labels for control flow, and
    source-intent tracing.

    Attributes:
        opcode:       Human-readable opcode name (e.g. 'READ_PIN', 'PUSH_F32').
        operand1:     First operand (pin number, immediate value, etc).
        operand2:     Second operand (immediate value, jump target, etc).
        comment:      Human-readable comment for disassembly.
        source_intent: Original intent text that produced this instruction.
        label:        Optional label name defined at this instruction offset.
        jump_target:  For JUMP/JUMP_IF_FALSE/JUMP_IF_TRUE: label to jump to.
    """

    opcode: str
    operand1: int | float | None = None
    operand2: int | float | None = None
    comment: str = ""
    source_intent: str = ""
    label: str | None = None
    jump_target: str | None = None


# ===================================================================
# Unique label generator
# ===================================================================

class _LabelGenerator:
    """Generates unique labels for control flow."""

    def __init__(self) -> None:
        self._counter = 0

    def next(self, prefix: str = "L") -> str:
        name = f"{prefix}{self._counter}"
        self._counter += 1
        return name


# ===================================================================
# IntentCompiler
# ===================================================================

class IntentCompiler:
    """Compile structured Intent objects into IR instruction lists.

    Each Intent maps to a sequence of IRInstructions. The compiler handles:
    - Simple reads, writes, waits, syscalls
    - Conditionals with label-based control flow
    - Loops with counter-based iteration
    - PID computation syscalls
    - Navigation waypoints
    - Compound intents (flattened into sequential IR)
    """

    def __init__(self) -> None:
        self._labels = _LabelGenerator()

    def compile(self, intent: Intent) -> list[IRInstruction]:
        """Compile a single Intent into IR instructions.

        Args:
            intent: Parsed Intent object from IntentParser.

        Returns:
            List of IRInstruction objects. Always ends with a HALT.
        """
        ir: list[IRInstruction] = []

        if intent.action == "UNKNOWN":
            return ir

        ir.extend(self._compile_intent(intent))

        # Add trailing HALT if last instruction is not already a HALT
        if not ir or ir[-1].opcode != "SYSCALL" or ir[-1].operand2 != 0x01:
            ir.append(self._make_halt(intent.raw))
        return ir

    def compile_many(self, intents: list[Intent]) -> list[IRInstruction]:
        """Compile multiple Intents into a single IR program.

        Args:
            intents: List of Intent objects.

        Returns:
            Combined IR instruction list ending with HALT.
        """
        ir: list[IRInstruction] = []
        for intent in intents:
            if intent.action == "UNKNOWN":
                continue
            ir.extend(self._compile_intent(intent))

        if ir:
            ir.append(self._make_halt(""))
        return ir

    # -----------------------------------------------------------------
    # Intent dispatch
    # -----------------------------------------------------------------

    def _compile_intent(self, intent: Intent) -> list[IRInstruction]:
        """Dispatch compilation based on intent action type."""
        action = intent.action

        if action == "READ":
            return self._compile_read(intent)
        elif action == "WRITE":
            return self._compile_write(intent)
        elif action == "CONDITIONAL":
            return self._compile_conditional(intent)
        elif action == "LOOP":
            return self._compile_loop(intent)
        elif action == "WAIT":
            return self._compile_wait(intent)
        elif action == "PID":
            return self._compile_pid(intent)
        elif action == "NAVIGATE":
            return self._compile_navigate(intent)
        elif action == "SYSCALL":
            return self._compile_syscall(intent)
        elif action == "COMPOUND":
            return self._compile_compound(intent)
        else:
            return []

    # -----------------------------------------------------------------
    # Simple intents
    # -----------------------------------------------------------------

    def _compile_read(self, intent: Intent) -> list[IRInstruction]:
        """Compile a READ intent: read sensor/variable pin."""
        pin = intent.pin
        if pin is None:
            return []
        return [
            IRInstruction(
                opcode="READ_PIN",
                operand1=pin,
                comment=f"read {intent.target.lower()} {pin}",
                source_intent=intent.raw,
            )
        ]

    def _compile_write(self, intent: Intent) -> list[IRInstruction]:
        """Compile a WRITE intent: push value, clamp, write to pin."""
        pin = intent.pin
        value = intent.value
        if pin is None or value is None:
            return []

        ir: list[IRInstruction] = []
        ir.append(IRInstruction(
            opcode="PUSH_F32",
            operand2=value,
            comment=f"push {value}",
            source_intent=intent.raw,
        ))
        # Clamp actuator values to [-1, 1]
        if intent.target == "ACTUATOR":
            ir.append(IRInstruction(
                opcode="CLAMP_F",
                operand1=-1.0,
                operand2=1.0,
                comment="clamp [-1, 1]",
                source_intent=intent.raw,
            ))
        ir.append(IRInstruction(
            opcode="WRITE_PIN",
            operand1=pin,
            comment=f"write {intent.target.lower()} {pin}",
            source_intent=intent.raw,
        ))
        return ir

    def _compile_wait(self, intent: Intent) -> list[IRInstruction]:
        """Compile a WAIT intent: push cycle count, read timer, compare, loop.

        Implements: read timer, add wait cycles, read timer again, compare.
        Simplified: just emit NOPs for the wait cycles (deterministic).
        """
        cycles = intent.params.get("cycles", 0) if intent.params else 0
        if not isinstance(cycles, int) or cycles <= 0:
            return []

        ir: list[IRInstruction] = []

        if cycles == 1:
            ir.append(IRInstruction(
                opcode="NOP",
                comment=f"wait 1 cycle",
                source_intent=intent.raw,
            ))
        else:
            # Loop-based wait: counter from cycles down to 0
            loop_start = self._labels.next("wait_start")
            loop_end = self._labels.next("wait_end")

            ir.append(IRInstruction(
                opcode="PUSH_I8",
                operand1=cycles,
                comment=f"wait {cycles} cycles: counter",
                source_intent=intent.raw,
                label=loop_start,
            ))
            ir.append(IRInstruction(
                opcode="DUP",
                comment="dup counter",
                source_intent=intent.raw,
            ))
            ir.append(IRInstruction(
                opcode="PUSH_I8",
                operand1=1,
                comment="decrement",
                source_intent=intent.raw,
            ))
            ir.append(IRInstruction(
                opcode="SUB_F",
                comment="counter - 1",
                source_intent=intent.raw,
            ))
            ir.append(IRInstruction(
                opcode="DUP",
                comment="dup new counter",
                source_intent=intent.raw,
            ))
            ir.append(IRInstruction(
                opcode="PUSH_I8",
                operand1=0,
                comment="compare with 0",
                source_intent=intent.raw,
            ))
            ir.append(IRInstruction(
                opcode="GT_F",
                comment="counter > 0?",
                source_intent=intent.raw,
            ))
            ir.append(IRInstruction(
                opcode="JUMP_IF_TRUE",
                jump_target=loop_start,
                comment="loop if counter > 0",
                source_intent=intent.raw,
            ))
            ir.append(IRInstruction(
                opcode="POP",
                comment="clean up counter",
                source_intent=intent.raw,
                label=loop_end,
            ))

        return ir

    def _compile_pid(self, intent: Intent) -> list[IRInstruction]:
        """Compile a PID intent: read sensor, then syscall PID_COMPUTE.

        The PID parameters (kp, ki, kd) are passed via push instructions
        before the syscall. The sensor value is read first.
        """
        pin = intent.pin
        params = intent.params or {}
        kp = params.get("kp", 1.0)
        ki = params.get("ki", 0.0)
        kd = params.get("kd", 0.0)

        if pin is None:
            return []

        ir: list[IRInstruction] = [
            IRInstruction(
                opcode="READ_PIN",
                operand1=pin,
                comment=f"read sensor {pin} for PID",
                source_intent=intent.raw,
            ),
            IRInstruction(
                opcode="PUSH_F32",
                operand2=kp,
                comment=f"kp={kp}",
                source_intent=intent.raw,
            ),
            IRInstruction(
                opcode="PUSH_F32",
                operand2=ki,
                comment=f"ki={ki}",
                source_intent=intent.raw,
            ),
            IRInstruction(
                opcode="PUSH_F32",
                operand2=kd,
                comment=f"kd={kd}",
                source_intent=intent.raw,
            ),
        ]

        # PID compute as a syscall
        ir.append(IRInstruction(
            opcode="SYSCALL",
            operand1=0,
            operand2=0x02,  # SYSCALL_PID_COMPUTE
            comment="PID compute",
            source_intent=intent.raw,
        ))

        return ir

    def _compile_navigate(self, intent: Intent) -> list[IRInstruction]:
        """Compile a NAVIGATE intent: declare intent with waypoint coords.

        Uses DECLARE_INTENT opcode with waypoint coordinates as operands.
        """
        params = intent.params or {}
        x = params.get("x", 0.0)
        y = params.get("y", 0.0)

        ir: list[IRInstruction] = [
            IRInstruction(
                opcode="PUSH_F32",
                operand2=x,
                comment=f"waypoint x={x}",
                source_intent=intent.raw,
            ),
            IRInstruction(
                opcode="PUSH_F32",
                operand2=y,
                comment=f"waypoint y={y}",
                source_intent=intent.raw,
            ),
            IRInstruction(
                opcode="DECLARE_INTENT",
                operand1=0,
                operand2=0,
                comment="navigate to waypoint",
                source_intent=intent.raw,
            ),
        ]

        return ir

    def _compile_syscall(self, intent: Intent) -> list[IRInstruction]:
        """Compile a SYSCALL intent: halt, record_snapshot, emit_event."""
        params = intent.params or {}
        syscall_name = params.get("syscall", "")

        if syscall_name == "halt":
            return [self._make_halt(intent.raw)]
        elif syscall_name == "record_snapshot":
            return [
                IRInstruction(
                    opcode="SYSCALL",
                    operand1=0,
                    operand2=0x03,  # SYSCALL_RECORD_SNAPSHOT
                    comment="log snapshot",
                    source_intent=intent.raw,
                )
            ]
        elif syscall_name == "emit_event":
            msg = params.get("message", "")
            # Encode message as a simple event_id (hash to 16-bit)
            event_id = self._hash_event_id(msg)
            return [
                IRInstruction(
                    opcode="SYSCALL",
                    operand1=event_id,
                    operand2=0x04,  # SYSCALL_EMIT_EVENT
                    comment=f'emit event "{msg}"',
                    source_intent=intent.raw,
                )
            ]

        return []

    # -----------------------------------------------------------------
    # Complex intents
    # -----------------------------------------------------------------

    def _compile_conditional(self, intent: Intent) -> list[IRInstruction]:
        """Compile a CONDITIONAL intent with label-based branching.

        Structure:
            READ_PIN <sensor>
            PUSH_F32 <threshold>
            <CMP_OP>
            JUMP_IF_FALSE <else_label>
            <then_body IR>
            <else_label:
        """
        pin = intent.pin
        threshold = intent.threshold
        operator = intent.operator
        then_body = intent.then_body or []

        if pin is None or threshold is None or operator is None:
            return []

        cmp_opcode = self._operator_to_opcode(operator)
        else_label = self._labels.next("else")

        ir: list[IRInstruction] = [
            IRInstruction(
                opcode="READ_PIN",
                operand1=pin,
                comment=f"if sensor {pin}",
                source_intent=intent.raw,
            ),
            IRInstruction(
                opcode="PUSH_F32",
                operand2=threshold,
                comment=f"compare {operator} {threshold}",
                source_intent=intent.raw,
            ),
            IRInstruction(
                opcode=cmp_opcode,
                comment=f"{operator}",
                source_intent=intent.raw,
            ),
            IRInstruction(
                opcode="JUMP_IF_FALSE",
                jump_target=else_label,
                comment="branch if false",
                source_intent=intent.raw,
            ),
        ]

        # Compile then-body
        for sub_intent in then_body:
            ir.extend(self._compile_intent(sub_intent))

        # Else label (no else body in basic conditionals)
        ir.append(IRInstruction(
            opcode="NOP",
            comment="else branch / continue",
            source_intent=intent.raw,
            label=else_label,
        ))

        return ir

    def _compile_loop(self, intent: Intent) -> list[IRInstruction]:
        """Compile a LOOP intent with counter-based iteration.

        Structure:
            PUSH_I8 <count>
            <loop_start:>
            DUP
            PUSH_I8 0
            LTE_F
            JUMP_IF_TRUE <loop_end>
            <body IR>
            <loop_start>
            POP (counter on stack)
            <loop_end:>
        """
        body = intent.body or []
        params = intent.params or {}
        count = params.get("count", 0)

        if not isinstance(count, int) or count <= 0:
            return []
        if not body:
            return []

        loop_start = self._labels.next("loop")
        loop_check = self._labels.next("loop_check")
        loop_end = self._labels.next("loop_end")

        ir: list[IRInstruction] = [
            # Initialize counter
            IRInstruction(
                opcode="PUSH_I8",
                operand1=count,
                comment=f"loop {count} times: init counter",
                source_intent=intent.raw,
                label=loop_start,
            ),
            # Check: counter <= 0?
            IRInstruction(
                opcode="DUP",
                comment="dup counter",
                source_intent=intent.raw,
                label=loop_check,
            ),
            IRInstruction(
                opcode="PUSH_I8",
                operand1=0,
                comment="compare with 0",
                source_intent=intent.raw,
            ),
            IRInstruction(
                opcode="LTE_F",
                comment="counter <= 0?",
                source_intent=intent.raw,
            ),
            IRInstruction(
                opcode="JUMP_IF_TRUE",
                jump_target=loop_end,
                comment="exit loop",
                source_intent=intent.raw,
            ),
            # Decrement counter
            IRInstruction(
                opcode="PUSH_I8",
                operand1=1,
                comment="decrement by 1",
                source_intent=intent.raw,
            ),
            IRInstruction(
                opcode="SUB_F",
                comment="counter - 1",
                source_intent=intent.raw,
            ),
        ]

        # Body
        for sub_intent in body:
            ir.extend(self._compile_intent(sub_intent))

        # Jump back to check
        ir.append(IRInstruction(
            opcode="JUMP",
            jump_target=loop_check,
            comment="loop back",
            source_intent=intent.raw,
        ))

        # Clean up counter and end label
        ir.append(IRInstruction(
            opcode="POP",
            comment="clean up loop counter",
            source_intent=intent.raw,
            label=loop_end,
        ))

        return ir

    def _compile_compound(self, intent: Intent) -> list[IRInstruction]:
        """Compile a COMPOUND intent by flattening sub-intents."""
        body = intent.body or []
        ir: list[IRInstruction] = []
        for sub_intent in body:
            ir.extend(self._compile_intent(sub_intent))
        return ir

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _operator_to_opcode(op: str) -> str:
        """Map intent operator to comparison opcode."""
        mapping = {
            "gt": "GT_F",
            "lt": "LT_F",
            "eq": "EQ_F",
            "gte": "GTE_F",
            "lte": "LTE_F",
        }
        return mapping.get(op, "GT_F")

    @staticmethod
    def _make_halt(source: str) -> IRInstruction:
        """Create a HALT instruction."""
        return IRInstruction(
            opcode="SYSCALL",
            operand1=0,
            operand2=0x01,  # SYSCALL_HALT
            comment="halt",
            source_intent=source,
        )

    @staticmethod
    def _hash_event_id(msg: str) -> int:
        """Hash an event message string to a 16-bit event_id."""
        h = hash(msg)
        return h & 0xFFFF
