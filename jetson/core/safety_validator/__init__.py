"""NEXUS Safety Validator — 6-stage bytecode safety validation pipeline.

This package provides the BytecodeSafetyPipeline class for pre-deployment
validation of bytecode targeting the ESP32 NEXUS VM.

Usage:
    from core.safety_validator import BytecodeSafetyPipeline

    pipeline = BytecodeSafetyPipeline(trust_level=3)
    report = pipeline.validate(bytecode)

    if report.overall_passed:
        print("Bytecode is safe for deployment")
    else:
        print(report.summary())
"""

from core.safety_validator.models import (
    SafetyReport,
    SafetyViolation,
    StageResult,
    make_timestamp,
)
from core.safety_validator.pipeline import BytecodeSafetyPipeline
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
    FLAGS_HAS_IMMEDIATE,
    FLAGS_IS_CALL,
    FLAGS_IS_FLOAT,
    FLAGS_SYSCALL,
    INSTR_SIZE,
    SAFETY_CRITICAL_PINS,
    STACK_EFFECTS,
    TRUST_OPCODE_MATRIX,
)

__all__ = [
    "BytecodeSafetyPipeline",
    "SafetyReport",
    "SafetyViolation",
    "StageResult",
    "make_timestamp",
    # Constants
    "INSTR_SIZE",
    "STACK_EFFECTS",
    "ALL_VALID_OPCODES",
    "TRUST_OPCODE_MATRIX",
    "SAFETY_CRITICAL_PINS",
    "DEFAULT_MAX_STACK_DEPTH",
    "DEFAULT_MAX_INSTRUCTIONS",
    "DEFAULT_MAX_CALL_DEPTH",
    "DEFAULT_MAX_NOP_SEQUENCE",
    "DEFAULT_MAX_PROGRAM_SIZE",
    "DEFAULT_CLAMP_RANGE",
    "ACTUATOR_SAFE_RANGES",
    "FLAGS_HAS_IMMEDIATE",
    "FLAGS_IS_CALL",
    "FLAGS_IS_FLOAT",
    "FLAGS_EXTENDED_CLAMP",
    "FLAGS_SYSCALL",
]
