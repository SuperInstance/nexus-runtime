"""Bytecode Playground — Assemble and execute bytecode in the NEXUS VM.

Demonstrates the full pipeline: write assembly source, compile to bytecode,
validate it, and execute in the emulator with I/O callbacks.

Run:
    cd /tmp/nexus-runtime && python examples/bytecode_playground.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nexus.vm import Executor, Assembler, Validator, Opcodes, Instruction


def main() -> None:
    print("=" * 60)
    print("  NEXUS Bytecode Playground")
    print("=" * 60)

    # ── Assembly source: read sensor, add constant, write actuator ──
    # This program:
    #   1. Reads a sensor value from IO channel 16 into R0
    #   2. Loads a calibration constant (100) into R1
    #   3. Adds R1 to R0 -> R2 (sensor + calibration)
    #   4. Writes the result to IO channel 17 (actuator)
    #   5. Pushes R2 to stack for later inspection
    #   6. Halts
    source = """
        ; ---- NEXUS Bytecode Playground ----
        ; Read sensor, add calibration offset, write actuator

        READ_IO  R0, R16          ; Read sensor channel 16 into R0
        LOAD_CONST R1, 100        ; Calibration constant
        ADD      R2, R0, R1       ; R2 = sensor_value + 100
        WRITE_IO R17, R2          ; Write result to actuator channel 17
        PUSH     R2               ; Push result for inspection
        HALT                       ; Stop execution
    """

    print(f"\nAssembly source:\n{source}")

    # ── Step 1: Assemble ────────────────────────────────────────────
    asm = Assembler()
    bytecode = asm.assemble(source.strip())
    num_instructions = len(bytecode) // 8

    print(f"Assembled: {len(bytecode)} bytes, {num_instructions} instructions")
    print(f"Labels: {list(asm.labels.keys()) if asm.labels else '(none)'}")

    # ── Step 2: Validate ────────────────────────────────────────────
    validator = Validator()
    result = validator.validate(bytecode)

    if not result.valid:
        print(f"\nValidation FAILED with {len(result.errors)} errors:")
        for err in result.errors:
            print(f"  {err}")
        return

    if result.warnings:
        print(f"Validation: PASSED (with {len(result.warnings)} warnings)")
    else:
        print("Validation: PASSED")

    # ── Step 3: Set up I/O callbacks ────────────────────────────────
    io_log_read: list = []
    io_log_write: list = []

    def io_read(channel: int) -> int:
        """Simulate a sensor reading."""
        value = 42  # simulated sensor value
        io_log_read.append((channel, value))
        print(f"  [IO READ]  channel={channel} -> value={value}")
        return value

    def io_write(channel: int, value: int) -> None:
        """Simulate actuator output."""
        io_log_write.append((channel, value))
        print(f"  [IO WRITE] channel={channel} <- value={value}")

    # ── Step 4: Execute ─────────────────────────────────────────────
    print("\n--- Execution Trace ---")
    executor = Executor(
        program=bytecode,
        io_read_cb=io_read,
        io_write_cb=io_write,
    )

    cycles = executor.run(max_cycles=100)
    print(f"\nExecution complete: {cycles} cycles, halted={executor.halted}")

    # ── Step 5: Display register state ──────────────────────────────
    print("\n" + "-" * 60)
    print("  Final Register State")
    print("-" * 60)
    for i in range(16):
        val = executor.registers[i]
        if val != 0:
            print(f"  R{i:2d} = {val:>10d}  (0x{val:08X})")

    if executor.stack:
        print(f"\n  Stack: {executor.stack}")

    print(f"\n  Flags:  zero={executor.flags_zero}  negative={executor.flags_negative}")
    print(f"  PC:     {executor.pc}")
    print(f"  Memory: {executor._memory_size} bytes")

    # ── Step 6: I/O summary ─────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  I/O Summary")
    print("-" * 60)
    print(f"  Reads:  {io_log_read}")
    print(f"  Writes: {io_log_write}")

    # ── Bonus: show raw bytecode hex ────────────────────────────────
    print("\n" + "-" * 60)
    print("  Raw Bytecode (hex)")
    print("-" * 60)
    opcode_values = {op.value for op in Opcodes}
    for i in range(0, len(bytecode), 8):
        insn_bytes = bytecode[i:i + 8]
        insn = Instruction.decode(bytecode, i)
        name = Opcodes(insn.opcode).name if insn.opcode in opcode_values else f"OP_{insn.opcode:#04x}"
        hex_str = " ".join(f"{b:02x}" for b in insn_bytes)
        print(f"  {i:04x}:  {hex_str}  ; {name} rd={insn.rd} rs1={insn.rs1} rs2={insn.rs2} imm={insn.imm32}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
