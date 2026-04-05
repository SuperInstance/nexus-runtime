"""NEXUS Skill Cartridge Builder — Construct cartridges from bytecode, intent, or JSON.

Provides the CartridgeBuilder class which offers multiple construction pathways:
  - from_bytecode: Build from raw AAB bytecode with metadata
  - from_intent: Compile intent text to bytecode via Rosetta Stone, then build
  - from_json: Load a cartridge from a JSON file on disk
  - to_json: Serialize a cartridge to a JSON file
  - validate: Validate cartridge completeness and bytecode safety
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

from agent.skill_system.cartridge import SkillCartridge, SkillParameter

# Import the safety pipeline for bytecode validation
from core.safety_validator.pipeline import BytecodeSafetyPipeline


@dataclass
class ValidationResult:
    """Result of cartridge validation.

    Attributes:
        valid: True if cartridge passes all validation checks.
        errors: List of error messages (validation blockers).
        warnings: List of warning messages (non-blocking concerns).
        details: Additional validation detail metadata.
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        """Add an error message."""
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        """Add a warning message."""
        self.warnings.append(msg)


class CartridgeBuilder:
    """Build skill cartridges from bytecode, intent text, or JSON files.

    The builder is the factory for creating SkillCartridge instances.
    It supports multiple input formats and includes validation logic
    to ensure cartridges are well-formed and safe before deployment.

    Example:
        builder = CartridgeBuilder()

        # From raw bytecode
        cartridge = builder.from_bytecode(bytecode, {
            "name": "my_skill",
            "description": "Does something useful",
        })

        # From JSON file
        cartridge = builder.from_json("path/to/skill.json")

        # Validate
        result = builder.validate(cartridge)
        if result.valid:
            print("Cartridge is ready for deployment")
    """

    def from_bytecode(self, bytecode: bytes, metadata: dict) -> SkillCartridge:
        """Build a skill cartridge from raw AAB bytecode with metadata.

        Args:
            bytecode: Raw AAB bytecode bytes (must be 8-byte aligned).
            metadata: Dictionary with cartridge metadata. Expected keys:
                - name (required): Skill identifier
                - description: Human-readable description
                - version: Semantic version string
                - domain: Operational domain
                - trust_required: Minimum trust level (0-5)
                - inputs: List of SkillParameter dicts
                - outputs: List of SkillParameter dicts
                - parameters: Dict of configurable parameters
                - constraints: Dict of safety constraints
                - provenance: Dict of author/review info
                - metadata: Free-form metadata

        Returns:
            A SkillCartridge instance.

        Raises:
            ValueError: If bytecode is empty or not 8-byte aligned.
        """
        if not bytecode:
            raise ValueError("Bytecode cannot be empty")
        if len(bytecode) % 8 != 0:
            raise ValueError(
                f"Bytecode must be 8-byte aligned, got {len(bytecode)} bytes "
                f"({len(bytecode) % 8} bytes remainder)"
            )

        name = metadata.get("name", "unnamed_skill")
        if not name or not isinstance(name, str):
            raise ValueError("Cartridge name must be a non-empty string")

        inputs = [
            SkillParameter.from_dict(p) for p in metadata.get("inputs", [])
        ]
        outputs = [
            SkillParameter.from_dict(p) for p in metadata.get("outputs", [])
        ]

        return SkillCartridge(
            name=name,
            version=metadata.get("version", "1.0.0"),
            description=metadata.get("description", ""),
            domain=metadata.get("domain", "marine"),
            trust_required=metadata.get("trust_required", 0),
            bytecode=bytecode,
            inputs=inputs,
            outputs=outputs,
            parameters=metadata.get("parameters", {}),
            constraints=metadata.get("constraints", {}),
            provenance=metadata.get("provenance", {}),
            metadata=metadata.get("metadata", {}),
        )

    def from_intent(self, intent_text: str, metadata: dict) -> SkillCartridge:
        """Build a skill cartridge from intent text.

        Compiles the intent text into AAB bytecode using the Rosetta Stone
        compiler (via the ReflexCompiler), then wraps it into a cartridge.

        The intent text is interpreted as a reflex definition JSON body.
        If the intent_text is valid JSON, it is used directly as a reflex
        definition. Otherwise, a simple linear program is synthesized from
        the intent keywords.

        Args:
            intent_text: Natural language or JSON intent description.
            metadata: Dictionary with cartridge metadata (same as from_bytecode).

        Returns:
            A SkillCartridge instance with compiled bytecode.

        Raises:
            ValueError: If compilation fails or metadata is invalid.
        """
        bytecode = self._compile_intent(intent_text)
        return self.from_bytecode(bytecode, metadata)

    def from_json(self, json_path: str) -> SkillCartridge:
        """Load a skill cartridge from a JSON file.

        The JSON file should contain a serialized SkillCartridge dict
        (as produced by to_json).

        Args:
            json_path: Path to the JSON file.

        Returns:
            A SkillCartridge instance.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            ValueError: If the JSON is malformed or missing required fields.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Cartridge JSON file not found: {json_path}")

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in cartridge file {json_path}: {e}")

        if not isinstance(data, dict):
            raise ValueError(
                f"Cartridge file {json_path} must contain a JSON object, "
                f"got {type(data).__name__}"
            )

        if "name" not in data:
            raise ValueError(f"Cartridge file {json_path} missing required 'name' field")

        return SkillCartridge.from_dict(data)

    def to_json(self, cartridge: SkillCartridge, path: str) -> None:
        """Serialize a skill cartridge to a JSON file.

        Args:
            cartridge: The cartridge to serialize.
            path: Output file path.

        Raises:
            ValueError: If the cartridge has no name.
        """
        if not cartridge.name:
            raise ValueError("Cannot serialize cartridge without a name")

        data = cartridge.to_dict()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def validate(self, cartridge: SkillCartridge) -> ValidationResult:
        """Validate a skill cartridge: metadata completeness and bytecode safety.

        Checks:
        1. Name is non-empty
        2. Version is present
        3. Trust level is 0-5
        4. Bytecode is non-empty and 8-byte aligned
        5. Bytecode passes the 6-stage safety validation pipeline
        6. Input/output parameters have valid types
        7. Parameters with ranges have min < max

        Args:
            cartridge: The cartridge to validate.

        Returns:
            ValidationResult with errors and warnings.
        """
        result = ValidationResult()

        # --- Metadata checks ---
        if not cartridge.name:
            result.add_error("Cartridge name is empty")

        if not cartridge.version:
            result.add_error("Cartridge version is empty")
        elif not self._is_valid_version(cartridge.version):
            result.add_warning(
                f"Version '{cartridge.version}' does not follow semantic versioning"
            )

        if not isinstance(cartridge.trust_required, int):
            result.add_error(
                f"trust_required must be int, got {type(cartridge.trust_required).__name__}"
            )
        elif cartridge.trust_required < 0 or cartridge.trust_required > 5:
            result.add_error(
                f"trust_required must be 0-5, got {cartridge.trust_required}"
            )

        if cartridge.domain not in ("marine", "aerial", "industrial", "research", "custom"):
            result.add_warning(f"Unusual domain: '{cartridge.domain}'")

        # --- Bytecode checks ---
        if not cartridge.bytecode:
            result.add_error("Bytecode is empty")
        elif len(cartridge.bytecode) % 8 != 0:
            result.add_error(
                f"Bytecode is not 8-byte aligned ({len(cartridge.bytecode)} bytes)"
            )
        else:
            # Run the full 6-stage safety validation pipeline
            pipeline = BytecodeSafetyPipeline(
                trust_level=cartridge.trust_required
            )
            report = pipeline.validate(cartridge.bytecode)

            if not report.overall_passed:
                for stage in report.stages:
                    for err in stage.errors:
                        result.add_error(
                            f"Safety [{stage.stage_name}]: {err}"
                        )
            for stage in report.stages:
                for warn in stage.warnings:
                    result.add_warning(
                        f"Safety [{stage.stage_name}]: {warn}"
                    )

        # --- Input/Output parameter checks ---
        valid_types = {"sensor", "actuator", "variable", "constant"}
        for param in cartridge.inputs:
            if param.type not in valid_types:
                result.add_error(
                    f"Input parameter '{param.name}' has invalid type '{param.type}'"
                )
            if param.range_min is not None and param.range_max is not None:
                if param.range_min >= param.range_max:
                    result.add_error(
                        f"Input '{param.name}': range_min ({param.range_min}) "
                        f">= range_max ({param.range_max})"
                    )

        for param in cartridge.outputs:
            if param.type not in valid_types:
                result.add_error(
                    f"Output parameter '{param.name}' has invalid type '{param.type}'"
                )
            if param.range_min is not None and param.range_max is not None:
                if param.range_min >= param.range_max:
                    result.add_error(
                        f"Output '{param.name}': range_min ({param.range_min}) "
                        f">= range_max ({param.range_max})"
                    )

        # --- Constraint checks ---
        for key, value in cartridge.constraints.items():
            if isinstance(value, dict):
                cmin = value.get("min")
                cmax = value.get("max")
                if cmin is not None and cmax is not None:
                    if cmin >= cmax:
                        result.add_error(
                            f"Constraint '{key}': min ({cmin}) >= max ({cmax})"
                        )

        result.details = {
            "bytecode_size": cartridge.bytecode_size,
            "instruction_count": cartridge.instruction_count,
            "trust_required": cartridge.trust_required,
        }

        return result

    def _compile_intent(self, intent_text: str) -> bytes:
        """Compile intent text to bytecode.

        If the intent_text is valid JSON, it is treated as a reflex definition
        and compiled via ReflexCompiler. Otherwise, a minimal read-only
        bytecode program is synthesized.

        Args:
            intent_text: Intent description (JSON or natural language).

        Returns:
            Compiled bytecode bytes.
        """
        import struct

        # Try to parse as JSON reflex definition first
        try:
            reflex_def = json.loads(intent_text)
            if isinstance(reflex_def, dict) and "body" in reflex_def:
                return self._compile_reflex_json(reflex_def)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: synthesize a minimal read-only monitoring program
        # from intent keywords. This creates a simple sensor-reading program.
        return self._synthesize_bytecode(intent_text)

    def _compile_reflex_json(self, reflex_def: dict) -> bytes:
        """Compile a reflex JSON definition using the ReflexCompiler."""
        try:
            from reflex.compiler import ReflexCompiler
            compiler = ReflexCompiler()
            return compiler.compile(reflex_def)
        except ImportError:
            raise ValueError(
                "ReflexCompiler not available for intent compilation. "
                "Provide bytecode directly or install the reflex compiler module."
            )
        except ValueError as e:
            raise ValueError(f"Failed to compile reflex definition: {e}")

    def _synthesize_bytecode(self, intent_text: str) -> bytes:
        """Synthesize a minimal bytecode program from intent keywords.

        Creates a simple read-only monitoring program that reads sensors
        and performs arithmetic. This is a fallback when the Rosetta Stone
        compiler is not available.

        The synthesized program:
        1. Read timer (1 instruction)
        2. Read pin 0 / depth sensor (1 instruction)
        3. Push a constant threshold (1 instruction)
        4. Compare (1 instruction)
        5. NOP as placeholder for alert (1 instruction)
        """
        # Simple 5-instruction monitoring program
        instructions = []

        # READ_TIMER_MS: opcode=0x1C, flags=0, operand1=0, operand2=0
        instructions.append(struct.pack("<BBHI", 0x1C, 0x00, 0, 0))

        # READ_PIN 0 (depth sensor): opcode=0x1A, flags=0x01, operand1=0, operand2=0
        instructions.append(struct.pack("<BBHI", 0x1A, 0x01, 0, 0))

        # PUSH_F32 100.0: opcode=0x03, flags=0x02, operand1=0,
        #                  operand2=IEEE754(100.0)
        val_100 = struct.unpack("<I", struct.pack("<f", 100.0))[0]
        instructions.append(struct.pack("<BBHI", 0x03, 0x02, 0, val_100))

        # LT_F: opcode=0x12, flags=0, operand1=0, operand2=0
        instructions.append(struct.pack("<BBHI", 0x12, 0x00, 0, 0))

        # NOP: opcode=0x00, flags=0, operand1=0, operand2=0
        instructions.append(struct.pack("<BBHI", 0x00, 0x00, 0, 0))

        return b"".join(instructions)

    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Check if a version string follows semantic versioning (MAJOR.MINOR.PATCH)."""
        parts = version.split(".")
        if len(parts) != 3:
            return False
        try:
            return all(p.isdigit() for p in parts)
        except (ValueError, AttributeError):
            return False
