"""Configuration schema validation with type checking, constraints, and serialization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional, Tuple, List, Dict


class SchemaType(Enum):
    """Supported schema field types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"
    ANY = "any"


@dataclass
class SchemaField:
    """Describes a single field in a configuration schema."""
    name: str
    type: SchemaType = SchemaType.ANY
    required: bool = True
    default: Any = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    pattern: Optional[str] = None
    description: str = ""
    # For ENUM type: list of allowed values
    allowed_values: List[Any] = field(default_factory=list)

    def validate_value(self, value: Any) -> Tuple[bool, List[str]]:
        """Validate a single value against this field definition."""
        errors: List[str] = []

        # Check type
        if self.type == SchemaType.STRING:
            if not isinstance(value, str):
                errors.append(f"Field '{self.name}': expected string, got {type(value).__name__}")
                return False, errors
            if self.pattern and not re.fullmatch(self.pattern, value):
                errors.append(
                    f"Field '{self.name}': value '{value}' does not match pattern '{self.pattern}'"
                )
        elif self.type == SchemaType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"Field '{self.name}': expected integer, got {type(value).__name__}")
                return False, errors
            if self.min_val is not None and value < self.min_val:
                errors.append(f"Field '{self.name}': value {value} below minimum {self.min_val}")
            if self.max_val is not None and value > self.max_val:
                errors.append(f"Field '{self.name}': value {value} above maximum {self.max_val}")
        elif self.type == SchemaType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"Field '{self.name}': expected float, got {type(value).__name__}")
                return False, errors
            if self.min_val is not None and value < self.min_val:
                errors.append(f"Field '{self.name}': value {value} below minimum {self.min_val}")
            if self.max_val is not None and value > self.max_val:
                errors.append(f"Field '{self.name}': value {value} above maximum {self.max_val}")
        elif self.type == SchemaType.BOOLEAN:
            if not isinstance(value, bool):
                errors.append(f"Field '{self.name}': expected boolean, got {type(value).__name__}")
        elif self.type == SchemaType.ARRAY:
            if not isinstance(value, list):
                errors.append(f"Field '{self.name}': expected array, got {type(value).__name__}")
        elif self.type == SchemaType.OBJECT:
            if not isinstance(value, dict):
                errors.append(f"Field '{self.name}': expected object, got {type(value).__name__}")
        elif self.type == SchemaType.ENUM:
            if self.allowed_values and value not in self.allowed_values:
                errors.append(
                    f"Field '{self.name}': value '{value}' not in allowed values {self.allowed_values}"
                )
        # SchemaType.ANY passes type check

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize field to dictionary."""
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SchemaField:
        """Deserialize field from dictionary."""
        d = dict(d)
        d["type"] = SchemaType(d["type"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ConfigSchema:
    """Configuration schema with field definitions and validation."""

    def __init__(self, name: str = "default", description: str = "") -> None:
        self.name = name
        self.description = description
        self._fields: Dict[str, SchemaField] = {}

    def add_field(self, field_def: SchemaField) -> None:
        """Add a field definition to the schema."""
        self._fields[field_def.name] = field_def

    def remove_field(self, name: str) -> None:
        """Remove a field from the schema by name. Raises KeyError if not found."""
        if name not in self._fields:
            raise KeyError(f"Field '{name}' not found in schema '{self.name}'")
        del self._fields[name]

    def get_field(self, name: str) -> Optional[SchemaField]:
        """Get a field definition by name. Returns None if not found."""
        return self._fields.get(name)

    @property
    def fields(self) -> Dict[str, SchemaField]:
        """Return all field definitions."""
        return dict(self._fields)

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a configuration dict against this schema. Returns (valid, errors)."""
        errors: List[str] = []

        # Check required fields
        for field_name, fdef in self._fields.items():
            if fdef.required and field_name not in config:
                if fdef.default is None:
                    errors.append(f"Missing required field: '{field_name}'")

        # Validate present values
        for key, value in config.items():
            fdef = self._fields.get(key)
            if fdef is None:
                continue  # unknown fields are allowed
            valid, field_errors = fdef.validate_value(value)
            errors.extend(field_errors)

        return len(errors) == 0, errors

    def merge(self, other: ConfigSchema) -> ConfigSchema:
        """Merge another schema into this one. Other's fields override on conflict."""
        merged = ConfigSchema(
            name=f"{self.name}+{other.name}",
            description=self.description or other.description,
        )
        merged._fields = {**self._fields, **other._fields}
        return merged

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full schema to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "fields": {name: fdef.to_dict() for name, fdef in self._fields.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ConfigSchema:
        """Deserialize a schema from a dictionary."""
        schema = cls(name=d.get("name", "default"), description=d.get("description", ""))
        for field_name, field_data in d.get("fields", {}).items():
            field_data["name"] = field_name
            schema.add_field(SchemaField.from_dict(field_data))
        return schema

    def generate_default(self) -> Dict[str, Any]:
        """Generate a default configuration dict from this schema."""
        defaults: Dict[str, Any] = {}
        for name, fdef in self._fields.items():
            if fdef.default is not None:
                defaults[name] = fdef.default
            elif fdef.type == SchemaType.BOOLEAN:
                defaults[name] = False
            elif fdef.type == SchemaType.INTEGER:
                defaults[name] = 0
            elif fdef.type == SchemaType.FLOAT:
                defaults[name] = 0.0
            elif fdef.type == SchemaType.STRING:
                defaults[name] = ""
            elif fdef.type == SchemaType.ARRAY:
                defaults[name] = []
            elif fdef.type == SchemaType.OBJECT:
                defaults[name] = {}
        return defaults

    def compute_schema_hash(self) -> str:
        """Compute a SHA-256 hash of this schema for change detection."""
        schema_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(schema_str.encode("utf-8")).hexdigest()
