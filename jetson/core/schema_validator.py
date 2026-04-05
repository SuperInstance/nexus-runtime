"""NEXUS Schema Validator - JSON Schema validation for NEXUS config files.

Provides functions to validate JSON configs against their schemas,
generate sample configs, and report validation errors with field paths.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

# Default schema directory
SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"

# Schema file names
SCHEMA_NAMES = {
    "autonomy_state": "autonomy_state.json",
    "reflex_definition": "reflex_definition.json",
    "node_role_config": "node_role_config.json",
    "serial_protocol": "serial_protocol.json",
}


@dataclass
class ValidationError:
    """A single validation error with path information."""

    message: str
    path: list[str]
    schema_path: list[str]
    validator: str

    def __str__(self) -> str:
        path_str = ".".join(self.path) if self.path else "(root)"
        return f"[{path_str}] {self.message} (validator: {self.validator})"


@dataclass
class ValidationResult:
    """Result of a schema validation."""

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def error_summary(self) -> str:
        """Return a human-readable summary of all errors."""
        if self.valid:
            return "Validation passed."
        lines = [f"Validation failed with {len(self.errors)} error(s):"]
        for err in self.errors:
            lines.append(f"  - {err}")
        return "\n".join(lines)


class SchemaValidator:
    """Validates JSON data against NEXUS schemas."""

    def __init__(self, schema_dir: Path | str | None = None) -> None:
        """Initialize validator.

        Args:
            schema_dir: Path to directory containing JSON schemas.
                        Defaults to <project_root>/schemas/.
        """
        self._schema_dir = Path(schema_dir) if schema_dir else SCHEMA_DIR
        self._schemas: dict[str, dict[str, Any]] = {}
        self._load_all_schemas()

    def _load_all_schemas(self) -> None:
        """Load all JSON schemas from the schema directory."""
        for name, filename in SCHEMA_NAMES.items():
            path = self._schema_dir / filename
            if path.exists():
                with open(path) as f:
                    self._schemas[name] = json.load(f)

    def get_schema(self, name: str) -> dict[str, Any]:
        """Get a loaded schema by name.

        Args:
            name: Schema name (e.g. 'autonomy_state').

        Returns:
            The JSON schema as a dict.

        Raises:
            KeyError: If schema name is unknown.
            FileNotFoundError: If schema file doesn't exist.
        """
        if name not in SCHEMA_NAMES:
            raise KeyError(
                f"Unknown schema '{name}'. "
                f"Known: {list(SCHEMA_NAMES.keys())}"
            )
        if name not in self._schemas:
            raise FileNotFoundError(
                f"Schema file '{SCHEMA_NAMES[name]}' not found in {self._schema_dir}"
            )
        return self._schemas[name]

    def validate(self, data: Any, schema_name: str) -> ValidationResult:
        """Validate JSON data against a named schema.

        Args:
            data: The JSON data to validate (dict, list, etc.).
            schema_name: Name of the schema to validate against.

        Returns:
            ValidationResult with valid=True if no errors, or list of errors.
        """
        schema = self.get_schema(schema_name)
        validator_cls = jsonschema.Draft7Validator
        validator = validator_cls(schema)

        errors: list[ValidationError] = []
        for err in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
            path_strs = [str(p) for p in err.absolute_path]
            schema_path_strs = [str(p) for p in err.absolute_schema_path]
            errors.append(ValidationError(
                message=err.message,
                path=path_strs,
                schema_path=schema_path_strs,
                validator=err.validator,
            ))

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def meta_validate(self, schema_name: str) -> ValidationResult:
        """Validate that a schema is itself valid JSON Schema.

        Args:
            schema_name: Name of the schema to meta-validate.

        Returns:
            ValidationResult.
        """
        schema = self.get_schema(schema_name)
        try:
            jsonschema.Draft7Validator.check_schema(schema)
            return ValidationResult(valid=True)
        except jsonschema.SchemaError as e:
            return ValidationResult(valid=False, errors=[
                ValidationError(
                    message=str(e.message),
                    path=[str(p) for p in e.path] if hasattr(e, "path") else [],
                    schema_path=[],
                    validator="meta-schema",
                )
            ])

    def list_schemas(self) -> list[str]:
        """Return list of available schema names."""
        return list(self._schemas.keys())


def validate_config(
    data: Any,
    schema_name: str,
    schema_dir: Path | str | None = None,
) -> ValidationResult:
    """Convenience function to validate a config against a schema.

    Args:
        data: JSON data to validate.
        schema_name: Schema name (e.g. 'autonomy_state').
        schema_dir: Optional path to schema directory.

    Returns:
        ValidationResult.
    """
    validator = SchemaValidator(schema_dir=schema_dir)
    return validator.validate(data, schema_name)


def validate_config_file(
    config_path: Path | str,
    schema_name: str,
    schema_dir: Path | str | None = None,
) -> ValidationResult:
    """Validate a JSON config file against a schema.

    Args:
        config_path: Path to the JSON config file.
        schema_name: Schema name (e.g. 'autonomy_state').
        schema_dir: Optional path to schema directory.

    Returns:
        ValidationResult.
    """
    with open(config_path) as f:
        data = json.load(f)
    return validate_config(data, schema_name, schema_dir)


def generate_sample_config(schema_name: str, schema_dir: Path | str | None = None) -> dict[str, Any]:
    """Generate a sample (minimal valid) config from a schema.

    Walks the schema and fills required fields with reasonable defaults.

    Args:
        schema_name: Schema name (e.g. 'autonomy_state').
        schema_dir: Optional path to schema directory.

    Returns:
        A dict with sample data that should pass validation.
    """
    validator = SchemaValidator(schema_dir=schema_dir)
    schema = validator.get_schema(schema_name)
    return _generate_from_schema(schema)


def _generate_from_schema(schema: dict[str, Any]) -> Any:
    """Recursively generate sample data from a JSON Schema."""
    # Handle allOf
    if "allOf" in schema:
        result: dict[str, Any] = {}
        for sub in schema["allOf"]:
            sub_data = _generate_from_schema(sub)
            if isinstance(sub_data, dict):
                result.update(sub_data)
        return result

    # Handle anyOf
    if "anyOf" in schema:
        return _generate_from_schema(schema["anyOf"][0])

    # Handle $ref
    if "$ref" in schema:
        # Can't resolve refs without a resolver, just return empty
        return {}

    schema_type = schema.get("type")

    # Handle type as list
    if isinstance(schema_type, list):
        # Prefer string > number > integer > boolean > array > object > null
        for preferred in ["string", "number", "integer", "boolean", "object", "array"]:
            if preferred in schema_type:
                schema_type = preferred
                break
        else:
            schema_type = schema_type[0]

    if schema_type == "object":
        result = {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        for prop_name, prop_schema in properties.items():
            if prop_name in required:
                result[prop_name] = _generate_from_schema(prop_schema)
        return result

    if schema_type == "array":
        items = schema.get("items", {})
        min_items = schema.get("minItems", 0)
        if min_items > 0:
            return [_generate_from_schema(items)]
        return []

    if schema_type == "string":
        if "enum" in schema:
            return schema["enum"][0]
        if "default" in schema:
            return schema["default"]
        if "pattern" in schema:
            return "sample_string"
        return ""

    if schema_type == "integer":
        if "enum" in schema:
            return schema["enum"][0]
        if "default" in schema:
            return schema["default"]
        if "const" in schema:
            return schema["const"]
        minimum = schema.get("minimum", 0)
        maximum = schema.get("maximum", 100)
        if "exclusiveMinimum" in schema:
            minimum = max(minimum, schema["exclusiveMinimum"] + 1)
        return min(maximum, max(minimum, 0))

    if schema_type == "number":
        if "default" in schema:
            return schema["default"]
        if "const" in schema:
            return schema["const"]
        return 0.0

    if schema_type == "boolean":
        if "default" in schema:
            return schema["default"]
        return False

    if schema_type == "null":
        return None

    # No type specified
    if "default" in schema:
        return schema["default"]
    if "const" in schema:
        return schema["const"]
    return None


def load_schema(schema_name: str, schema_dir: Path | str | None = None) -> dict[str, Any]:
    """Load a JSON schema by name.

    Args:
        schema_name: Schema name (e.g. 'autonomy_state').
        schema_dir: Optional path to schema directory.

    Returns:
        The schema as a dict.
    """
    validator = SchemaValidator(schema_dir=schema_dir)
    return validator.get_schema(schema_name)
