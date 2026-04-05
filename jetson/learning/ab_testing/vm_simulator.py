"""NEXUS A/B Testing — Reflex Comparator with lightweight VM simulator.

Python stack machine that executes NEXUS bytecode with simulated sensor
inputs. Records actuator outputs, cycle counts, and safety events.
Used to run N iterations of each variant for statistical significance.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any

from shared.opcodes import (
    ADD_F,
    AND_B,
    ABS_F,
    CLAMP_F,
    DIV_F,
    DUP,
    EQ_F,
    GT_F,
    GTE_F,
    JUMP,
    JUMP_IF_FALSE,
    JUMP_IF_TRUE,
    LT_F,
    LTE_F,
    MAX_F,
    MIN_F,
    MUL_F,
    NEG_F,
    NOP,
    NOT_B,
    OR_B,
    POP,
    PUSH_F32,
    PUSH_I16,
    PUSH_I8,
    READ_PIN,
    READ_TIMER_MS,
    ROT,
    SUB_F,
    SWAP,
    WRITE_PIN,
    XOR_B,
    opcode_name,
)
from reflex.bytecode_emitter import (
    FLAGS_HAS_IMMEDIATE,
    FLAGS_IS_CALL,
    FLAGS_IS_FLOAT,
    FLAGS_SYSCALL,
    FLAGS_EXTENDED_CLAMP,
    INSTR_SIZE,
    SYSCALL_HALT,
    f32_to_f16_bits,
    u32_to_float,
    unpack_instruction,
)


# Maximum execution cycles to prevent infinite loops
MAX_CYCLES = 10000
MAX_STACK_SIZE = 64


class VMSimulatorError(Exception):
    """Error during VM simulation."""


@dataclass
class SimulationIteration:
    """Result of a single VM simulation iteration."""

    iteration: int
    cycle_count: int
    actuator_outputs: dict[int, float] = field(default_factory=dict)
    final_stack: list[float] = field(default_factory=list)
    safety_events: list[str] = field(default_factory=list)
    error: str | None = None
    execution_time_ms: float = 0.0
    output_value: float = 0.0  # Last written value (primary output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "cycle_count": self.cycle_count,
            "actuator_outputs": self.actuator_outputs,
            "final_stack": self.final_stack[-10:],  # Last 10 stack items
            "safety_events": self.safety_events,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "output_value": self.output_value,
        }


@dataclass
class SimulatedSensorInput:
    """Configuration for simulated sensor inputs."""

    pin_values: dict[int, float] = field(default_factory=dict)
    noise_std: float = 0.0
    timer_ms: int = 0


class StackVM:
    """Lightweight NEXUS bytecode stack machine simulator.

    Implements a subset of the full NEXUS opcodes sufficient for
    reflex simulation and A/B testing.
    """

    def __init__(
        self,
        sensor_inputs: dict[int, float] | None = None,
        max_cycles: int = MAX_CYCLES,
        max_stack: int = MAX_STACK_SIZE,
    ) -> None:
        self.stack: list[float] = []
        self.pc: int = 0  # Program counter (instruction index)
        self.cycle_count: int = 0
        self.halted: bool = False
        self.max_cycles = max_cycles
        self.max_stack = max_stack
        self.sensor_inputs: dict[int, float] = sensor_inputs or {}
        self.actuator_outputs: dict[int, float] = {}
        self.safety_events: list[str] = []
        self.call_stack: list[int] = []
        self.timer_ms: int = 0
        self.last_output: float = 0.0

    def reset(self) -> None:
        """Reset VM state for a new execution."""
        self.stack.clear()
        self.pc = 0
        self.cycle_count = 0
        self.halted = False
        self.actuator_outputs.clear()
        self.safety_events.clear()
        self.call_stack.clear()
        self.last_output = 0.0

    def push(self, value: float) -> None:
        if len(self.stack) >= self.max_stack:
            self.safety_events.append("stack_overflow")
            self.halted = True
            return
        self.stack.append(value)

    def pop(self) -> float:
        if not self.stack:
            self.safety_events.append("stack_underflow")
            self.halted = True
            return 0.0
        return self.stack.pop()

    def execute(self, bytecode: bytes) -> SimulationIteration:
        """Execute bytecode and return simulation results."""
        self.reset()
        start_time = time.perf_counter()

        num_instructions = len(bytecode) // INSTR_SIZE
        if num_instructions == 0:
            self.safety_events.append("empty_bytecode")

        while not self.halted and self.cycle_count < self.max_cycles:
            if self.pc < 0 or self.pc >= num_instructions:
                self.safety_events.append(
                    f"pc_out_of_bounds:{self.pc}/{num_instructions}"
                )
                break

            opcode, flags, operand1, operand2 = unpack_instruction(
                bytecode, self.pc * INSTR_SIZE
            )

            self.cycle_count += 1

            if not self._execute_instruction(opcode, flags, operand1, operand2):
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return SimulationIteration(
            iteration=0,
            cycle_count=self.cycle_count,
            actuator_outputs=dict(self.actuator_outputs),
            final_stack=list(self.stack),
            safety_events=list(self.safety_events),
            execution_time_ms=elapsed_ms,
            output_value=self.last_output,
        )

    def _execute_instruction(
        self, opcode: int, flags: int, operand1: int, operand2: int
    ) -> bool:
        """Execute a single instruction. Returns False if execution should stop."""

        # Syscall HALT
        if opcode == NOP and (flags & FLAGS_SYSCALL):
            if operand2 == SYSCALL_HALT or operand1 == 1:
                self.halted = True
                return False
            # Other syscalls: NOP for now
            self.pc += 1
            return True

        if opcode == NOP:
            self.pc += 1
            return True

        # Stack operations
        if opcode == PUSH_I8:
            value = operand1 if operand1 < 128 else operand1 - 256
            self.push(float(value))
            self.pc += 1
            return True

        if opcode == PUSH_I16:
            value = operand1 if operand1 < 32768 else operand1 - 65536
            self.push(float(value))
            self.pc += 1
            return True

        if opcode == PUSH_F32:
            value = u32_to_float(operand2)
            self.push(value)
            self.pc += 1
            return True

        if opcode == POP:
            self.pop()
            self.pc += 1
            return True

        if opcode == DUP:
            if self.stack:
                self.push(self.stack[-1])
            self.pc += 1
            return True

        if opcode == SWAP:
            if len(self.stack) >= 2:
                self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
            self.pc += 1
            return True

        if opcode == ROT:
            if len(self.stack) >= 3:
                a = self.stack.pop()
                b = self.stack.pop()
                c = self.stack.pop()
                self.stack.extend([a, b, c])
            self.pc += 1
            return True

        # Arithmetic
        if opcode == ADD_F:
            b, a = self.pop(), self.pop()
            self.push(a + b)
            self.pc += 1
            return True

        if opcode == SUB_F:
            b, a = self.pop(), self.pop()
            self.push(a - b)
            self.pc += 1
            return True

        if opcode == MUL_F:
            b, a = self.pop(), self.pop()
            self.push(a * b)
            self.pc += 1
            return True

        if opcode == DIV_F:
            b, a = self.pop(), self.pop()
            if abs(b) < 1e-15:
                self.safety_events.append("division_by_zero")
                self.push(0.0)
            else:
                self.push(a / b)
            self.pc += 1
            return True

        if opcode == NEG_F:
            a = self.pop()
            self.push(-a)
            self.pc += 1
            return True

        if opcode == ABS_F:
            a = self.pop()
            self.push(abs(a))
            self.pc += 1
            return True

        if opcode == MIN_F:
            b, a = self.pop(), self.pop()
            self.push(min(a, b))
            self.pc += 1
            return True

        if opcode == MAX_F:
            b, a = self.pop(), self.pop()
            self.push(max(a, b))
            self.pc += 1
            return True

        if opcode == CLAMP_F:
            a = self.pop()
            lo16 = operand2 & 0xFFFF
            hi16 = (operand2 >> 16) & 0xFFFF
            lo = _f16_to_f32(lo16)
            hi = _f16_to_f32(hi16)
            self.push(max(lo, min(hi, a)))
            self.pc += 1
            return True

        # Comparison
        if opcode == EQ_F:
            b, a = self.pop(), self.pop()
            self.push(1.0 if abs(a - b) < 1e-9 else 0.0)
            self.pc += 1
            return True

        if opcode == LT_F:
            b, a = self.pop(), self.pop()
            self.push(1.0 if a < b else 0.0)
            self.pc += 1
            return True

        if opcode == GT_F:
            b, a = self.pop(), self.pop()
            self.push(1.0 if a > b else 0.0)
            self.pc += 1
            return True

        if opcode == LTE_F:
            b, a = self.pop(), self.pop()
            self.push(1.0 if a <= b else 0.0)
            self.pc += 1
            return True

        if opcode == GTE_F:
            b, a = self.pop(), self.pop()
            self.push(1.0 if a >= b else 0.0)
            self.pc += 1
            return True

        # Logic
        if opcode == AND_B:
            b, a = self.pop(), self.pop()
            self.push(1.0 if (a > 0.5 and b > 0.5) else 0.0)
            self.pc += 1
            return True

        if opcode == OR_B:
            b, a = self.pop(), self.pop()
            self.push(1.0 if (a > 0.5 or b > 0.5) else 0.0)
            self.pc += 1
            return True

        if opcode == XOR_B:
            b, a = self.pop(), self.pop()
            self.push(1.0 if ((a > 0.5) != (b > 0.5)) else 0.0)
            self.pc += 1
            return True

        if opcode == NOT_B:
            a = self.pop()
            self.push(1.0 if a <= 0.5 else 0.0)
            self.pc += 1
            return True

        # I/O
        if opcode == READ_PIN:
            pin = operand1
            value = self.sensor_inputs.get(pin, 0.0)
            self.push(value)
            self.pc += 1
            return True

        if opcode == WRITE_PIN:
            pin = operand1
            value = self.pop()
            self.actuator_outputs[pin] = value
            self.last_output = value
            self.pc += 1
            return True

        if opcode == READ_TIMER_MS:
            self.push(float(self.timer_ms))
            self.pc += 1
            return True

        # Control flow
        if opcode == JUMP:
            # RET instruction: JUMP with operand2==0xFFFFFFFF and no CALL flag
            if operand2 == 0xFFFFFFFF and not (flags & FLAGS_IS_CALL):
                if self.call_stack:
                    self.pc = self.call_stack.pop()
                else:
                    self.halted = True
                    return False
                return True
            target = operand2
            if flags & FLAGS_IS_CALL:
                self.call_stack.append(self.pc + 1)
            self.pc = target // INSTR_SIZE
            return True

        if opcode == JUMP_IF_FALSE:
            condition = self.pop()
            if condition <= 0.5:
                self.pc = operand2 // INSTR_SIZE
            else:
                self.pc += 1
            return True

        if opcode == JUMP_IF_TRUE:
            condition = self.pop()
            if condition > 0.5:
                self.pc = operand2 // INSTR_SIZE
            else:
                self.pc += 1
            return True

        # Unknown opcode — skip
        self.pc += 1
        return True


def _f16_to_f32(bits: int) -> float:
    """Convert IEEE 754 float16 bit pattern to float32."""
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF

    if exp == 0:
        if mant == 0:
            # Zero
            return -0.0 if sign else 0.0
        # Denormal
        value = mant / 1024.0
    elif exp == 31:
        if mant == 0:
            return float("-inf") if sign else float("inf")
        else:
            return float("nan")
    else:
        value = (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))

    return -value if sign else value


class ReflexComparator:
    """Compare reflex bytecode variants using VM simulation.

    Runs N iterations of each variant with simulated sensor inputs,
    records metrics, and feeds results into the A/B testing framework.
    """

    def __init__(
        self,
        sensor_inputs: dict[int, float] | None = None,
        noise_std: float = 0.0,
        max_cycles: int = MAX_CYCLES,
        seed: int | None = None,
    ) -> None:
        self.sensor_inputs = sensor_inputs or {}
        self.noise_std = noise_std
        self.max_cycles = max_cycles
        self.seed = seed
        self._rng_seed = seed

    def simulate_single(
        self,
        bytecode: bytes,
        sensor_override: dict[int, float] | None = None,
        timer_ms: int = 0,
    ) -> SimulationIteration:
        """Run a single simulation of bytecode."""
        sensors = dict(self.sensor_inputs)
        if sensor_override:
            sensors.update(sensor_override)

        # Add noise if configured
        if self.noise_std > 0:
            import random
            rng = random.Random(self._rng_seed)
            self._rng_seed = rng.randint(0, 2**31)
            for pin in list(sensors.keys()):
                sensors[pin] += rng.gauss(0, self.noise_std)

        vm = StackVM(
            sensor_inputs=sensors,
            max_cycles=self.max_cycles,
        )
        vm.timer_ms = timer_ms
        result = vm.execute(bytecode)
        return result

    def simulate_variant(
        self,
        bytecode: bytes,
        n_iterations: int = 100,
        sensor_scenarios: list[dict[int, float]] | None = None,
    ) -> list[SimulationIteration]:
        """Run N iterations of a bytecode variant.

        If sensor_scenarios is provided, each iteration uses the next
        scenario (cycling if fewer scenarios than iterations).
        Otherwise, uses self.sensor_inputs with optional noise.
        """
        import random
        rng = random.Random(self.seed)

        results = []
        for i in range(n_iterations):
            if sensor_scenarios and len(sensor_scenarios) > 0:
                scenario = sensor_scenarios[i % len(sensor_scenarios)]
                sensors = dict(scenario)
            else:
                sensors = dict(self.sensor_inputs)
                if self.noise_std > 0:
                    for pin in list(sensors.keys()):
                        sensors[pin] += rng.gauss(0, self.noise_std)

            vm = StackVM(
                sensor_inputs=sensors,
                max_cycles=self.max_cycles,
            )
            vm.timer_ms = i * 10  # 10ms per iteration
            iter_result = vm.execute(bytecode)
            iter_result.iteration = i + 1
            results.append(iter_result)

        return results

    def extract_metrics(self, results: list[SimulationIteration]) -> dict[str, list[float]]:
        """Extract standard metrics from simulation results.

        Returns dict with keys matching MetricType values.
        """
        metrics = {
            "cycle_time_ms": [],
            "accuracy": [],
            "trust_delta": [],
            "safety_events": [],
            "error_rate": [],
        }

        for r in results:
            # Cycle time: execution time proxy (cycles * 0.001ms each)
            metrics["cycle_time_ms"].append(r.execution_time_ms)
            # Accuracy: 1.0 if no errors, 0.0 if any safety events
            metrics["accuracy"].append(
                0.0 if r.error else (1.0 if not r.safety_events else 0.5)
            )
            # Trust delta: small positive for clean runs
            metrics["trust_delta"].append(0.01 if not r.error and not r.safety_events else -0.05)
            # Safety events: count
            metrics["safety_events"].append(float(len(r.safety_events)))
            # Error rate: 1.0 if error else 0.0
            metrics["error_rate"].append(1.0 if r.error else 0.0)

        return metrics

    def compare(
        self,
        suite: Any,
        n_iterations: int = 100,
        sensor_scenarios: list[dict[int, float]] | None = None,
    ) -> None:
        """Run simulations for all variants in an ABTestSuite and record metrics.

        Populates the suite with metric observations from simulation.
        """
        from .experiment import ABTestSuite, MetricType

        if not isinstance(suite, ABTestSuite):
            raise TypeError("suite must be an ABTestSuite")

        for variant_name, variant in suite.variants.items():
            results = self.simulate_variant(
                variant.bytecode,
                n_iterations=n_iterations,
                sensor_scenarios=sensor_scenarios,
            )
            metrics = self.extract_metrics(results)

            for mt in suite.metrics:
                if isinstance(mt, str):
                    mt = MetricType(mt)
                values = metrics.get(mt.value, [])
                for i, v in enumerate(values):
                    suite.record_metric(
                        variant_name, mt, v, timestamp_ms=i * 10
                    )
