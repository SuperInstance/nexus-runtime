"""Configuration templates with variable substitution, inheritance, and diffing."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .schema import ConfigSchema


@dataclass
class ConfigTemplate:
    """A reusable configuration template with variable slots."""
    name: str
    template_vars: Dict[str, Any] = field(default_factory=dict)
    base_config: Dict[str, Any] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)


class TemplateEngine:
    """Manage configuration templates — create, render, extend, diff, and validate."""

    def __init__(self) -> None:
        self._templates: Dict[str, ConfigTemplate] = {}

    def create_template(
        self,
        name: str,
        base_config: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> ConfigTemplate:
        """Create and register a new configuration template."""
        tpl = ConfigTemplate(
            name=name,
            template_vars=variables or {},
            base_config=deepcopy(base_config),
        )
        self._templates[name] = tpl
        return tpl

    def render(self, template: ConfigTemplate, values: Dict[str, Any]) -> Dict[str, Any]:
        """Render a template by substituting variable values into the base config.

        Variables can appear as {{var_name}} in string values, or as top-level
        template variable keys whose values get injected.
        """
        config = deepcopy(template.base_config)

        # First: apply variable values to template variable slots
        merged_vars = {**template.template_vars, **values}

        # Second: substitute {{var}} patterns in string values
        config = self._substitute_vars(config, merged_vars)

        # Third: apply overrides
        if template.overrides:
            self._deep_merge(config, template.overrides)

        return config

    def _substitute_vars(
        self, config: Dict[str, Any], variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively substitute {{var}} patterns in config values."""
        result: Dict[str, Any] = {}
        for key, value in config.items():
            if isinstance(value, str):
                result[key] = self._replace_var_pattern(value, variables)
            elif isinstance(value, dict):
                result[key] = self._substitute_vars(value, variables)
            elif isinstance(value, list):
                result[key] = [
                    self._replace_var_pattern(v, variables) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def _replace_var_pattern(self, value: str, variables: Dict[str, Any]) -> Any:
        """Replace {{var}} patterns in a string, with type coercion."""
        pattern = r'\{\{(\w+)\}\}'

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in variables:
                return str(variables[var_name])
            return match.group(0)  # keep unreplaced

        new_val = re.sub(pattern, replacer, value)

        # If the entire value was a single variable, return typed
        full_match = re.fullmatch(r'\{\{(\w+)\}\}', value)
        if full_match:
            var_name = full_match.group(1)
            if var_name in variables:
                return variables[var_name]

        return new_val

    def extend_template(
        self, base_template: ConfigTemplate, overrides: Dict[str, Any]
    ) -> ConfigTemplate:
        """Create a new template that extends a base template with overrides."""
        merged_config = deepcopy(base_template.base_config)
        self._deep_merge(merged_config, overrides)
        new_name = f"{base_template.name}_extended"
        tpl = ConfigTemplate(
            name=new_name,
            template_vars=dict(base_template.template_vars),
            base_config=merged_config,
            overrides=deepcopy(base_template.overrides),
        )
        self._templates[new_name] = tpl
        return tpl

    def compute_diff(
        self, template_a: ConfigTemplate, template_b: ConfigTemplate
    ) -> Dict[str, Any]:
        """Compute differences between two templates' base configurations.

        Returns a dict with 'added', 'removed', 'changed' keys, each containing
        a dict of field path -> value.
        """
        a = template_a.base_config
        b = template_b.base_config
        return self._dict_diff(a, b)

    def _dict_diff(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """Recursive dict diff."""
        added: Dict[str, Any] = {}
        removed: Dict[str, Any] = {}
        changed: Dict[str, Any] = {}

        all_keys = set(a.keys()) | set(b.keys())
        for key in all_keys:
            if key not in a:
                added[key] = deepcopy(b[key])
            elif key not in b:
                removed[key] = deepcopy(a[key])
            elif a[key] != b[key]:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    nested = self._dict_diff(a[key], b[key])
                    if nested["added"] or nested["removed"] or nested["changed"]:
                        changed[key] = nested
                else:
                    changed[key] = {"old": a[key], "new": b[key]}

        return {"added": added, "removed": removed, "changed": changed}

    def list_templates(self) -> List[str]:
        """List all registered template names."""
        return sorted(self._templates.keys())

    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """Get a template by name. Returns None if not found."""
        return self._templates.get(name)

    def validate_template(
        self, template: ConfigTemplate, schema: ConfigSchema
    ) -> Tuple[bool, List[str]]:
        """Validate a template's base_config against a schema."""
        return schema.validate(template.base_config)

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = deepcopy(value)
