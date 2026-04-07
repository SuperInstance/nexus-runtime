"""Multi-source configuration loading with override resolution and reference substitution."""

from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

from .schema import ConfigSchema


class ConfigSource(Enum):
    """Configuration source types."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    MEMORY = "memory"
    REMOTE = "remote"


class ConfigLoader:
    """Load configuration from multiple sources with merge, override, and validation."""

    def __init__(self) -> None:
        self._loaded_configs: Dict[str, Dict[str, Any]] = {}

    def load_from_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._loaded_configs[str(path)] = data
        return data

    def load_from_env(self, prefix: str = "NEXUS") -> Dict[str, Any]:
        """Load configuration from environment variables with a given prefix.

        Variables like NEXUS_LOG_LEVEL become {'log_level': value}.
        Nested keys via double underscore: NEXUS_DB__HOST -> {'db': {'host': value}}
        """
        config: Dict[str, Any] = {}
        prefix_upper = prefix.upper() + "_"

        for key, value in sorted(os.environ.items()):
            if not key.startswith(prefix_upper):
                continue
            # Strip prefix and lowercase
            remainder = key[len(prefix_upper):].lower()
            # Convert double underscore to nesting
            parts = remainder.split("__")
            target = config
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    target[part] = {}
                target = target[part]
            # Attempt JSON decode
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed = value
            target[parts[-1]] = parsed

        self._loaded_configs[f"env:{prefix}"] = config
        return config

    def load_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from an in-memory dictionary."""
        data = deepcopy(data)
        self._loaded_configs["memory:dict"] = data
        return data

    def load_with_override(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge overrides into a base configuration."""
        result = deepcopy(base)
        self._deep_merge(result, overrides)
        return result

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge override into base in place."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = deepcopy(value)

    def resolve_references(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve substitution references like ${section.key} within config values.

        Resolution is iterative with a max depth of 20 to prevent infinite loops.
        """
        resolved = deepcopy(config)
        for _ in range(20):
            changed = False
            resolved = self._resolve_pass(resolved)
            if not changed:
                break
        return resolved

    def _resolve_pass(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Single pass of reference resolution."""
        changed = False

        def _resolve_value(value: Any) -> Any:
            nonlocal changed
            if isinstance(value, str):
                new_val, did_change = self._substitute_refs(value, config)
                if did_change:
                    changed = True
                    # Try to parse numeric/bool
                    return self._coerce(new_val)
                return value
            elif isinstance(value, dict):
                return {k: _resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_resolve_value(v) for v in value]
            return value

        return {k: _resolve_value(v) for k, v in config.items()}

    def _substitute_refs(self, value: str, config: Dict[str, Any]) -> Tuple[str, bool]:
        """Substitute ${...} references in a string value."""
        pattern = r'\$\{([^}]+)\}'
        if not re.search(pattern, value):
            return value, False

        def replacer(match: re.Match) -> str:
            ref_path = match.group(1)
            parts = ref_path.split(".")
            target = config
            for part in parts:
                if isinstance(target, dict) and part in target:
                    target = target[part]
                else:
                    return match.group(0)  # keep unresolved
            if isinstance(target, (str, int, float, bool)):
                return str(target)
            return match.group(0)

        result = re.sub(pattern, replacer, value)
        return result, result != value

    def _coerce(self, value: str) -> Any:
        """Attempt to coerce a string to int, float, or bool."""
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            logger.debug("Could not coerce config value %r to int or float, keeping as string", value)
            pass
        return value

    def validate_against_schema(
        self, config: Dict[str, Any], schema: ConfigSchema
    ) -> Tuple[bool, List[str]]:
        """Validate a config dict against a schema. Delegates to ConfigSchema.validate."""
        return schema.validate(config)

    def load_and_validate(
        self, path: Union[str, Path], schema: ConfigSchema
    ) -> Dict[str, Any]:
        """Load config from file and validate against schema. Raises ValueError on invalid."""
        config = self.load_from_file(path)
        valid, errors = self.validate_against_schema(config, schema)
        if not valid:
            raise ValueError(f"Config validation failed: {'; '.join(errors)}")
        return config
