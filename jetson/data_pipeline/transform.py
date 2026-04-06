"""Data transformations — Transform and TransformPipeline classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Transform:
    """Named transformation wrapping a callable."""
    name: str
    transform_fn: Callable[[Any], Any]
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


class TransformPipeline:
    """Ordered chain of named transforms with schema validation."""

    def __init__(self) -> None:
        self._transforms: List[Transform] = []
        self._schemas: Dict[str, Dict[str, Any]] = {}

    # ── transform management ───────────────────────────────────

    def add_transform(self, transform: Transform) -> None:
        """Append a transform to the pipeline."""
        self._transforms.append(transform)

    def remove_transform(self, name: str) -> bool:
        """Remove the first transform whose name matches.  Returns True if found."""
        for i, t in enumerate(self._transforms):
            if t.name == name:
                self._transforms.pop(i)
                return True
        return False

    # ── execution ──────────────────────────────────────────────

    def apply(self, data_point: Any) -> Any:
        """Run a single data point through every transform in order."""
        result = data_point
        for t in self._transforms:
            result = t.transform_fn(result)
        return result

    def apply_batch(self, points: List[Any]) -> List[Any]:
        """Run a batch of points through the pipeline."""
        return [self.apply(p) for p in points]

    @staticmethod
    def compose(transforms: List[Transform]) -> Callable[[Any], Any]:
        """Compose multiple transforms into a single callable."""
        fns = [t.transform_fn for t in transforms]

        def composed(data: Any) -> Any:
            result = data
            for fn in fns:
                result = fn(result)
            return result

        return composed

    # ── schema management ──────────────────────────────────────

    def validate_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Lightweight structural validation.

        Supports ``type`` (exact type match), ``required_keys`` (dict),
        ``min_value`` / ``max_value`` (numeric), and ``allow_none``.
        """
        # allow_none shortcut
        if data is None:
            return schema.get("allow_none", False)

        expected_type = schema.get("type")
        if expected_type is not None:
            if not isinstance(data, expected_type):
                return False

        if isinstance(data, dict):
            required = schema.get("required_keys", [])
            for key in required:
                if key not in data:
                    return False

        if isinstance(data, (int, float)):
            min_v = schema.get("min_value")
            max_v = schema.get("max_value")
            if min_v is not None and data < min_v:
                return False
            if max_v is not None and data > max_v:
                return False

        return True

    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Register a named schema for later use."""
        self._schemas[name] = schema

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a registered schema by name."""
        return self._schemas.get(name)

    # ── introspection ──────────────────────────────────────────

    @property
    def transform_count(self) -> int:
        return len(self._transforms)

    def list_transforms(self) -> List[str]:
        return [t.name for t in self._transforms]
