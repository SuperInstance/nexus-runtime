"""Tests for loader.py — ConfigSource, ConfigLoader."""

import json
import os
import pytest
import tempfile
from pathlib import Path

from jetson.config_mgmt.loader import ConfigSource, ConfigLoader
from jetson.config_mgmt.schema import ConfigSchema, SchemaField, SchemaType


# ── ConfigSource ────────────────────────────────────────────────────────────

class TestConfigSource:
    def test_all_values(self):
        expected = {"FILE", "ENVIRONMENT", "DATABASE", "MEMORY", "REMOTE"}
        actual = {s.name for s in ConfigSource}
        assert actual == expected

    def test_file_value(self):
        assert ConfigSource.FILE.value == "file"

    def test_environment_value(self):
        assert ConfigSource.ENVIRONMENT.value == "environment"

    def test_memory_value(self):
        assert ConfigSource.MEMORY.value == "memory"

    def test_remote_value(self):
        assert ConfigSource.REMOTE.value == "remote"

    def test_database_value(self):
        assert ConfigSource.DATABASE.value == "database"


# ── ConfigLoader ────────────────────────────────────────────────────────────

class TestConfigLoader:
    def _write_json(self, data, suffix=".json"):
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.write(fd, json.dumps(data).encode())
        os.close(fd)
        return path

    def test_load_from_file(self):
        data = {"host": "localhost", "port": 8080}
        path = self._write_json(data)
        try:
            loader = ConfigLoader()
            result = loader.load_from_file(path)
            assert result == data
        finally:
            os.unlink(path)

    def test_load_from_file_not_found(self):
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_from_file("/nonexistent/path/config.json")

    def test_load_from_file_nested(self):
        data = {"database": {"host": "db.local", "port": 5432}, "debug": True}
        path = self._write_json(data)
        try:
            loader = ConfigLoader()
            result = loader.load_from_file(path)
            assert result["database"]["host"] == "db.local"
        finally:
            os.unlink(path)

    def test_load_from_file_empty(self):
        path = self._write_json({})
        try:
            loader = ConfigLoader()
            result = loader.load_from_file(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_load_from_env_with_prefix(self, monkeypatch):
        monkeypatch.setenv("NEXUS_HOST", "localhost")
        monkeypatch.setenv("NEXUS_PORT", "8080")
        monkeypatch.setenv("UNRELATED_VAR", "ignore")
        loader = ConfigLoader()
        result = loader.load_from_env("NEXUS")
        assert result["host"] == "localhost"
        assert result["port"] == 8080
        assert "unrelated_var" not in result

    def test_load_from_env_nested_double_underscore(self, monkeypatch):
        monkeypatch.setenv("NEXUS_DB__HOST", "db.local")
        monkeypatch.setenv("NEXUS_DB__PORT", "5432")
        loader = ConfigLoader()
        result = loader.load_from_env("NEXUS")
        assert result["db"]["host"] == "db.local"
        assert result["db"]["port"] == 5432

    def test_load_from_env_json_decode(self, monkeypatch):
        monkeypatch.setenv("NEXUS_DEBUG", "true")
        monkeypatch.setenv("NEXUS_PORT", "8080")
        loader = ConfigLoader()
        result = loader.load_from_env("NEXUS")
        assert result["debug"] is True
        assert result["port"] == 8080

    def test_load_from_env_no_matching(self, monkeypatch):
        monkeypatch.setenv("OTHER_VAR", "value")
        loader = ConfigLoader()
        result = loader.load_from_env("NEXUS")
        assert result == {}

    def test_load_from_dict(self):
        loader = ConfigLoader()
        data = {"key": "value", "num": 42}
        result = loader.load_from_dict(data)
        assert result == data

    def test_load_from_dict_is_copy(self):
        loader = ConfigLoader()
        data = {"key": "value"}
        result = loader.load_from_dict(data)
        data["key"] = "changed"
        assert result["key"] == "value"

    def test_load_with_override_simple(self):
        loader = ConfigLoader()
        base = {"host": "localhost", "port": 8080}
        overrides = {"port": 9090}
        result = loader.load_with_override(base, overrides)
        assert result["host"] == "localhost"
        assert result["port"] == 9090

    def test_load_with_override_deep_merge(self):
        loader = ConfigLoader()
        base = {"db": {"host": "db1.local", "port": 5432}, "cache": {"enabled": True}}
        overrides = {"db": {"host": "db2.local"}}
        result = loader.load_with_override(base, overrides)
        assert result["db"]["host"] == "db2.local"
        assert result["db"]["port"] == 5432  # preserved
        assert result["cache"]["enabled"] is True  # preserved

    def test_load_with_override_new_keys(self):
        loader = ConfigLoader()
        base = {"a": 1}
        overrides = {"b": 2, "c": 3}
        result = loader.load_with_override(base, overrides)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_load_with_override_empty_base(self):
        loader = ConfigLoader()
        result = loader.load_with_override({}, {"key": "val"})
        assert result == {"key": "val"}

    def test_resolve_references_simple(self):
        loader = ConfigLoader()
        config = {
            "host": "localhost",
            "url": "${host}:8080",
        }
        result = loader.resolve_references(config)
        assert result["url"] == "localhost:8080"

    def test_resolve_references_nested(self):
        loader = ConfigLoader()
        config = {
            "database": {"host": "db.local", "port": "5432"},
            "connection_string": "postgresql://${database.host}:${database.port}/mydb",
        }
        result = loader.resolve_references(config)
        assert result["connection_string"] == "postgresql://db.local:5432/mydb"

    def test_resolve_references_no_refs(self):
        loader = ConfigLoader()
        config = {"a": 1, "b": "plain"}
        result = loader.resolve_references(config)
        assert result == {"a": 1, "b": "plain"}

    def test_resolve_references_unresolved_stays(self):
        loader = ConfigLoader()
        config = {"url": "${missing.key}"}
        result = loader.resolve_references(config)
        assert result["url"] == "${missing.key}"

    def test_resolve_references_bool_coercion(self):
        loader = ConfigLoader()
        config = {"flag": True, "ref": "${flag}"}
        result = loader.resolve_references(config)
        assert result["ref"] is True

    def test_resolve_references_int_coercion(self):
        loader = ConfigLoader()
        config = {"count": 42, "ref": "${count}"}
        result = loader.resolve_references(config)
        assert result["ref"] == 42

    def test_resolve_references_float_coercion(self):
        loader = ConfigLoader()
        config = {"ratio": 3.14, "ref": "${ratio}"}
        result = loader.resolve_references(config)
        assert result["ref"] == 3.14

    def test_resolve_references_in_list(self):
        loader = ConfigLoader()
        config = {"host": "localhost", "urls": ["http://${host}", "https://${host}"]}
        result = loader.resolve_references(config)
        assert result["urls"] == ["http://localhost", "https://localhost"]

    def test_resolve_references_in_nested_dict(self):
        loader = ConfigLoader()
        config = {"db": {"host": "db.local"}, "conn": {"dsn": "host=${db.host}"}}
        result = loader.resolve_references(config)
        assert result["conn"]["dsn"] == "host=db.local"

    def test_validate_against_schema_valid(self):
        schema = ConfigSchema(name="test")
        schema.add_field(SchemaField(name="port", type=SchemaType.INTEGER, min_val=1))
        loader = ConfigLoader()
        ok, errs = loader.validate_against_schema({"port": 8080}, schema)
        assert ok is True
        assert errs == []

    def test_validate_against_schema_invalid(self):
        schema = ConfigSchema(name="test")
        schema.add_field(SchemaField(name="port", type=SchemaType.INTEGER, required=True))
        loader = ConfigLoader()
        ok, errs = loader.validate_against_schema({}, schema)
        assert ok is False
        assert len(errs) > 0

    def test_load_and_validate_valid(self):
        schema = ConfigSchema(name="test")
        schema.add_field(SchemaField(name="host", type=SchemaType.STRING, default="localhost"))
        path = self._write_json({"host": "0.0.0.0"})
        try:
            loader = ConfigLoader()
            result = loader.load_and_validate(path, schema)
            assert result["host"] == "0.0.0.0"
        finally:
            os.unlink(path)

    def test_load_and_validate_invalid(self):
        schema = ConfigSchema(name="test")
        schema.add_field(SchemaField(name="port", type=SchemaType.INTEGER, required=True))
        path = self._write_json({"port": "not-int"})
        try:
            loader = ConfigLoader()
            with pytest.raises(ValueError, match="validation failed"):
                loader.load_and_validate(path, schema)
        finally:
            os.unlink(path)
