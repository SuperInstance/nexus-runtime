"""Rosetta Stone: 4-layer intent-to-bytecode translation pipeline.

Translates human-readable intent strings into NEXUS VM bytecode through
a 4-layer pipeline:

    Layer 1 (IntentParser):    Natural language -> structured Intent
    Layer 2 (IntentCompiler):  Structured Intent -> IR instructions
    Layer 3 (IRValidator):     IR validation and optimization
    Layer 4 (BytecodeGenerator): Validated IR -> NEXUS bytecode (8-byte instructions)

Usage:
    from agent.rosetta_stone import RosettaStone

    rosetta = RosettaStone(trust_level=5)
    result = rosetta.translate("read sensor 3")
    if result.success:
        bytecode = result.bytecode  # 8-byte aligned NEXUS bytecode
"""

from agent.rosetta_stone.rosetta import RosettaStone, TranslationResult
from agent.rosetta_stone.intent_parser import Intent, IntentParser
from agent.rosetta_stone.intent_compiler import IntentCompiler, IRInstruction
from agent.rosetta_stone.ir_validator import IRValidator, ValidationResult
from agent.rosetta_stone.bytecode_generator import BytecodeGenerator

__all__ = [
    "RosettaStone",
    "TranslationResult",
    "Intent",
    "IntentParser",
    "IntentCompiler",
    "IRInstruction",
    "IRValidator",
    "ValidationResult",
    "BytecodeGenerator",
]
