"""NEXUS Skill Cartridge — Data format for reusable marine operation skills.

A skill cartridge contains pre-compiled AAB bytecode plus rich metadata
describing inputs, outputs, parameters, constraints, and provenance.
Cartridges are the deployment unit for NEXUS marine behaviors.

Think of it as a "USB drive for robot skills" — plug in a cartridge,
and the agent instantly knows how to perform a new marine operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkillParameter:
    """Describes a single input, output, or internal parameter of a skill.

    Attributes:
        name: Parameter identifier (e.g., "gps_heading", "throttle_output").
        type: Parameter category — "sensor", "actuator", "variable", "constant".
        pin: Hardware pin number for sensor/actuator parameters, None otherwise.
        range_min: Minimum valid value, None if unbounded.
        range_max: Maximum valid value, None if unbounded.
        unit: Physical unit (e.g., "meters", "degrees", "celsius"), None if N/A.
        description: Human-readable description of this parameter.
    """

    name: str
    type: str  # "sensor", "actuator", "variable", "constant"
    pin: int | None = None
    range_min: float | None = None
    range_max: float | None = None
    unit: str | None = None
    description: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "pin": self.pin,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "unit": self.unit,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SkillParameter:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            type=data["type"],
            pin=data.get("pin"),
            range_min=data.get("range_min"),
            range_max=data.get("range_max"),
            unit=data.get("unit"),
            description=data.get("description", ""),
        )

    def validate_value(self, value: float) -> tuple[bool, str]:
        """Check if a value is within the parameter's allowed range.

        Returns:
            (is_valid, error_message) tuple.
        """
        if self.range_min is not None and value < self.range_min:
            return False, f"{self.name}: value {value} below minimum {self.range_min}"
        if self.range_max is not None and value > self.range_max:
            return False, f"{self.name}: value {value} above maximum {self.range_max}"
        return True, ""


@dataclass
class SkillCartridge:
    """A reusable NEXUS skill cartridge.

    Contains pre-compiled AAB bytecode and full metadata for a specific
    marine operation pattern. Cartridges are the fundamental deployment unit
    in the NEXUS skill loading system.

    Attributes:
        name: Unique skill identifier (e.g., "surface_navigation").
        version: Semantic version string (e.g., "1.0.0").
        description: Human-readable description of what this skill does.
        domain: Operational domain (e.g., "marine", "aerial", "industrial").
        trust_required: Minimum trust level (0-5) needed to deploy this skill.
        bytecode: Pre-compiled AAB bytecode as raw bytes.
        inputs: List of expected sensor/variable inputs.
        outputs: List of produced actuator/variable outputs.
        parameters: Configurable parameters (e.g., PID gains, tolerances).
        constraints: Safety constraints (max values, ranges, limits).
        provenance: Author, review status, test results metadata.
        metadata: Free-form metadata for extensibility.
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    domain: str = "marine"
    trust_required: int = 0
    bytecode: bytes = b""
    inputs: list[SkillParameter] = field(default_factory=list)
    outputs: list[SkillParameter] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def instruction_count(self) -> int:
        """Number of 8-byte instructions in the bytecode."""
        return len(self.bytecode) // 8 if len(self.bytecode) % 8 == 0 else 0

    @property
    def bytecode_size(self) -> int:
        """Size of bytecode in bytes."""
        return len(self.bytecode)

    @property
    def is_bytecode_valid(self) -> bool:
        """Check if bytecode is non-empty and 8-byte aligned."""
        return len(self.bytecode) > 0 and len(self.bytecode) % 8 == 0

    def to_dict(self) -> dict:
        """Serialize cartridge to dictionary (bytecode as hex string)."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "domain": self.domain,
            "trust_required": self.trust_required,
            "bytecode": self.bytecode.hex(),
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "parameters": self.parameters,
            "constraints": self.constraints,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SkillCartridge:
        """Deserialize cartridge from dictionary (bytecode from hex string)."""
        bytecode_hex = data.get("bytecode", "")
        bytecode = bytes.fromhex(bytecode_hex) if bytecode_hex else b""

        inputs = [SkillParameter.from_dict(p) for p in data.get("inputs", [])]
        outputs = [SkillParameter.from_dict(p) for p in data.get("outputs", [])]

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            domain=data.get("domain", "marine"),
            trust_required=data.get("trust_required", 0),
            bytecode=bytecode,
            inputs=inputs,
            outputs=outputs,
            parameters=data.get("parameters", {}),
            constraints=data.get("constraints", {}),
            provenance=data.get("provenance", {}),
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        """Return a human-readable summary of this cartridge."""
        lines = [
            f"Skill: {self.name} v{self.version}",
            f"  Domain: {self.domain}",
            f"  Trust Required: L{self.trust_required}",
            f"  Description: {self.description}",
            f"  Bytecode: {self.bytecode_size} bytes ({self.instruction_count} instructions)",
            f"  Inputs: {len(self.inputs)} ({', '.join(p.name for p in self.inputs) or 'none'})",
            f"  Outputs: {len(self.outputs)} ({', '.join(p.name for p in self.outputs) or 'none'})",
            f"  Parameters: {len(self.parameters)} keys",
        ]
        return "\n".join(lines)
