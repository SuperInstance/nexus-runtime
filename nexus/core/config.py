"""
NEXUS Configuration Management — validated config with YAML/JSON loading and env overrides.

Features:
    - Nested dict-based configuration
    - Schema validation with type checking and range constraints
    - YAML and JSON file loading
    - Environment variable overrides (NEXUS_SECTION_KEY format)
    - Default values and required fields
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

@dataclass
class ConfigSchema:
    """Schema definition for configuration validation."""

    fields: Dict[str, "FieldSpec"] = field(default_factory=dict)
    required: Set[str] = field(default_factory=set)
    allow_extra: bool = True

    def add_field(
        self,
        name: str,
        field_type: type = str,
        default: Any = None,
        required: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        choices: Optional[List[Any]] = None,
        description: str = "",
    ) -> "ConfigSchema":
        """Add a field specification."""
        self.fields[name] = FieldSpec(
            name=name,
            field_type=field_type,
            default=default,
            min_value=min_value,
            max_value=max_value,
            choices=choices,
            description=description,
        )
        if required:
            self.required.add(name)
        return self


@dataclass
class FieldSpec:
    """Specification for a single configuration field."""

    name: str
    field_type: type = str
    default: Any = None
    required: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Config error
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Configuration validation error."""

    def __init__(self, message: str, field_name: str = ""):
        self.field_name = field_name
        super().__init__(f"{'[' + field_name + '] ' if field_name else ''}{message}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    """Hierarchical configuration management with validation.

    Usage::

        config = Config()
        config.set("node.id", "AUV-001")
        config.set("node.max_speed", 5.0)
        config.set("sensors.enabled", True)

        # Load from file
        config.load_yaml("config.yaml")

        # Environment overrides: NEXUS_NODE_ID=NewID
        config.apply_env(prefix="NEXUS")

        # Validate with schema
        schema = ConfigSchema()
        schema.add_field("node.id", required=True)
        schema.add_field("node.max_speed", field_type=float, min_value=0.0, max_value=20.0)
        errors = config.validate(schema)
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = data or {}
        self._schema: Optional[ConfigSchema] = None
        self._sources: List[str] = []

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self._data)

    # ----- accessors -----

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by dot-separated key path."""
        parts = key.split(".")
        current = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """Set a value by dot-separated key path."""
        parts = key.split(".")
        current = self._data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def has(self, key: str) -> bool:
        """Check if a key exists."""
        parts = key.split(".")
        current = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        return True

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        parts = key.split(".")
        current = self._data
        for part in parts[:-1]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        if isinstance(current, dict) and parts[-1] in current:
            del current[parts[-1]]
            return True
        return False

    # ----- loading -----

    def load_dict(self, data: Dict[str, Any], merge: bool = True) -> None:
        """Load from a dictionary."""
        if merge:
            self._data = self._deep_merge(self._data, data)
        else:
            self._data = copy.deepcopy(data)

    def load_json(self, path: str | Path, merge: bool = True) -> None:
        """Load configuration from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {path}")
        with open(p, "r") as f:
            data = json.load(f)
        self.load_dict(data, merge=merge)
        self._sources.append(str(p))

    def load_yaml(self, path: str | Path, merge: bool = True) -> None:
        """Load configuration from a YAML file."""
        if not _HAS_YAML:
            raise ConfigError("PyYAML is not installed")
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {path}")
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        self.load_dict(data, merge=merge)
        self._sources.append(str(p))

    def load_json_string(self, text: str, merge: bool = True) -> None:
        """Load from a JSON string."""
        data = json.loads(text)
        self.load_dict(data, merge=merge)

    # ----- environment overrides -----

    def apply_env(self, prefix: str = "NEXUS", separator: str = "_") -> int:
        """Apply environment variable overrides.

        Variables like ``NEXUS_NODE_ID`` map to key ``node.id``.

        Returns count of overrides applied.
        """
        count = 0
        prefix_upper = prefix.upper() + separator
        for key, value in os.environ.items():
            if not key.startswith(prefix_upper):
                continue
            # Strip prefix and convert to dot path
            config_key = key[len(prefix_upper):].lower()
            config_key = config_key.replace(separator.lower(), ".")
            # Try to parse value
            parsed = self._parse_env_value(value)
            self.set(config_key, parsed)
            count += 1
        return count

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse an environment variable value to its native type."""
        if value.lower() in ("true", "yes", "on", "1"):
            return True
        if value.lower() in ("false", "no", "off", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    # ----- validation -----

    def validate(self, schema: Optional[ConfigSchema] = None) -> List[ConfigError]:
        """Validate current config against a schema. Returns list of errors."""
        schema = schema or self._schema
        if schema is None:
            return []

        errors: List[ConfigError] = []

        # Check required fields
        for name in schema.required:
            if not self.has(name):
                errors.append(ConfigError(f"Required field missing", name))

        # Check field types and constraints
        for name, spec in schema.fields.items():
            value = self.get(name)
            if value is None:
                if spec.required:
                    errors.append(ConfigError(f"Required field missing", name))
                continue

            # Type check
            if not isinstance(value, spec.field_type):
                # Allow int where float is expected
                if spec.field_type == float and isinstance(value, int):
                    pass
                else:
                    errors.append(ConfigError(
                        f"Expected {spec.field_type.__name__}, got {type(value).__name__}",
                        name,
                    ))
                    continue

            # Range check
            if isinstance(value, (int, float)):
                if spec.min_value is not None and value < spec.min_value:
                    errors.append(ConfigError(
                        f"Value {value} below minimum {spec.min_value}", name
                    ))
                if spec.max_value is not None and value > spec.max_value:
                    errors.append(ConfigError(
                        f"Value {value} above maximum {spec.max_value}", name
                    ))

            # Choices check
            if spec.choices is not None and value not in spec.choices:
                errors.append(ConfigError(
                    f"Value {value!r} not in allowed choices: {spec.choices}", name
                ))

        return errors

    # ----- helpers -----

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep-merge two dictionaries."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def flatten(self, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested config to dot-separated keys."""
        result: Dict[str, Any] = {}
        self._flatten_recursive(self._data, prefix, result)
        return result

    def _flatten_recursive(self, data: Any, prefix: str, result: Dict[str, Any]) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._flatten_recursive(value, new_prefix, result)
        else:
            result[prefix] = data

    def __repr__(self) -> str:
        return f"Config(keys={len(self.flatten())}, sources={self._sources})"
