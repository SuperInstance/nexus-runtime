"""NEXUS Configuration Management — schema validation, multi-source loading, templates, hot reload, secrets."""

from .schema import SchemaType, SchemaField, ConfigSchema
from .loader import ConfigSource, ConfigLoader
from .templates import ConfigTemplate, TemplateEngine
from .hot_reload import ReloadEvent, ConfigWatcher
from .secrets import SecretEntry, SecretManager

__all__ = [
    "SchemaType", "SchemaField", "ConfigSchema",
    "ConfigSource", "ConfigLoader",
    "ConfigTemplate", "TemplateEngine",
    "ReloadEvent", "ConfigWatcher",
    "SecretEntry", "SecretManager",
]
