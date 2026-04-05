"""Rosetta Stone: 4-layer intent-to-bytecode translation pipeline.

Full pipeline: Text -> Intent -> IR -> Validated IR -> Bytecode

This is the top-level API for the Rosetta Stone translator. It chains
together all four layers:
    Layer 1: IntentParser    - Natural language -> structured Intent
    Layer 2: IntentCompiler   - Structured Intent -> IR instructions
    Layer 3: IRValidator      - IR validation and optimization
    Layer 4: BytecodeGenerator - Validated IR -> NEXUS bytecode
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.rosetta_stone.intent_parser import Intent, IntentParser
from agent.rosetta_stone.intent_compiler import IntentCompiler, IRInstruction
from agent.rosetta_stone.ir_validator import IRValidator, ValidationResult
from agent.rosetta_stone.bytecode_generator import BytecodeGenerator


# ===================================================================
# Translation result
# ===================================================================

@dataclass
class TranslationResult:
    """Result of translating an intent string to bytecode.

    Attributes:
        success:   Whether translation completed successfully.
        bytecode:  Compiled NEXUS bytecode (multiple of 8 bytes), or None on failure.
        intent:    Parsed Intent object, or None if parsing failed.
        ir:        IR instruction list, or None if compilation failed.
        errors:    List of error messages.
        warnings:  List of warning messages (non-fatal issues).
    """

    success: bool
    bytecode: bytes | None = None
    intent: Intent | None = None
    ir: list[IRInstruction] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ===================================================================
# RosettaStone
# ===================================================================

class RosettaStone:
    """4-layer translation pipeline: Text -> Intent -> IR -> Validated IR -> Bytecode.

    Usage:
        rosetta = RosettaStone()
        result = rosetta.translate("read sensor 3")
        if result.success:
            print(f"Generated {len(result.bytecode)} bytes of bytecode")
        else:
            print(f"Errors: {result.errors}")
    """

    def __init__(
        self,
        trust_level: int = 5,
        optimize: bool = True,
    ) -> None:
        """Initialize the Rosetta Stone translator.

        Args:
            trust_level: Default trust level (0-5). Higher levels permit
                         more operations. Default is 5 (Full Autonomy).
            optimize: Whether to apply IR optimizations. Default True.
        """
        self.trust_level = trust_level
        self.optimize = optimize
        self._parser = IntentParser()
        self._compiler = IntentCompiler()
        self._validator = IRValidator()
        self._generator = BytecodeGenerator()

    def translate(
        self, text: str, trust_level: int | None = None
    ) -> TranslationResult:
        """Full pipeline: parse -> compile -> validate -> generate.

        Args:
            text: Human-readable intent string.
            trust_level: Override default trust level for this translation.

        Returns:
            TranslationResult with bytecode, intent, IR, and any errors/warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        level = trust_level if trust_level is not None else self.trust_level

        # Layer 1: Parse intent
        try:
            intent = self._parser.parse(text)
        except ValueError as e:
            return TranslationResult(
                success=False,
                errors=[str(e)],
            )

        if intent.action == "UNKNOWN":
            return TranslationResult(
                success=False,
                intent=intent,
                errors=[f"Cannot parse intent: '{text}'"],
            )

        if intent.confidence < 0.5:
            warnings.append(
                f"Low confidence ({intent.confidence:.1f}) parsing: '{text}'"
            )

        # Layer 2: Compile to IR
        ir = self._compiler.compile(intent)
        if not ir:
            return TranslationResult(
                success=False,
                intent=intent,
                errors=["Compilation produced empty IR"],
            )

        # Layer 3: Validate and optionally optimize IR
        validation = self._validator.validate(ir, trust_level=level)
        errors.extend(validation.errors)
        warnings.extend(validation.warnings)

        if not validation.valid:
            return TranslationResult(
                success=False,
                intent=intent,
                ir=ir,
                errors=errors,
                warnings=warnings,
            )

        # Optimize
        if self.optimize:
            ir = self._validator.optimize(ir)

        # Layer 4: Generate bytecode
        try:
            bytecode = self._generator.generate(ir)
        except ValueError as e:
            errors.append(str(e))
            return TranslationResult(
                success=False,
                intent=intent,
                ir=ir,
                errors=errors,
                warnings=warnings,
            )

        return TranslationResult(
            success=True,
            bytecode=bytecode,
            intent=intent,
            ir=ir,
            errors=errors,
            warnings=warnings,
        )

    def translate_many(
        self, texts: list[str], trust_level: int | None = None
    ) -> list[TranslationResult]:
        """Translate multiple intent strings.

        Each intent is translated independently (separate bytecode programs).

        Args:
            texts: List of human-readable intent strings.
            trust_level: Override default trust level.

        Returns:
            List of TranslationResult objects, one per input text.
        """
        return [self.translate(t, trust_level=trust_level) for t in texts]

    def translate_combined(
        self, texts: list[str], trust_level: int | None = None
    ) -> TranslationResult:
        """Translate multiple intents into a single combined bytecode program.

        Intents are parsed individually, then compiled into a single IR
        program. The combined program ends with a single HALT.

        Args:
            texts: List of human-readable intent strings.
            trust_level: Override default trust level.

        Returns:
            Single TranslationResult with combined bytecode.
        """
        errors: list[str] = []
        warnings: list[str] = []

        level = trust_level if trust_level is not None else self.trust_level

        # Parse all intents
        intents: list[Intent] = []
        for text in texts:
            try:
                intent = self._parser.parse(text)
                if intent.action == "UNKNOWN":
                    errors.append(f"Cannot parse intent: '{text}'")
                else:
                    intents.append(intent)
                    if intent.confidence < 0.5:
                        warnings.append(
                            f"Low confidence ({intent.confidence:.1f}): '{text}'"
                        )
            except ValueError as e:
                errors.append(str(e))

        if not intents and errors:
            return TranslationResult(success=False, errors=errors)

        # Compile all intents into combined IR
        ir = self._compiler.compile_many(intents)
        if not ir:
            return TranslationResult(
                success=False,
                errors=["Combined compilation produced empty IR"],
            )

        # Validate
        validation = self._validator.validate(ir, trust_level=level)
        errors.extend(validation.errors)
        warnings.extend(validation.warnings)

        if not validation.valid:
            return TranslationResult(
                success=False,
                ir=ir,
                errors=errors,
                warnings=warnings,
            )

        # Optimize
        if self.optimize:
            ir = self._validator.optimize(ir)

        # Generate
        try:
            bytecode = self._generator.generate(ir)
        except ValueError as e:
            errors.append(str(e))
            return TranslationResult(
                success=False,
                ir=ir,
                errors=errors,
                warnings=warnings,
            )

        return TranslationResult(
            success=True,
            bytecode=bytecode,
            intent=intents[0] if intents else None,
            ir=ir,
            errors=errors,
            warnings=warnings,
        )
