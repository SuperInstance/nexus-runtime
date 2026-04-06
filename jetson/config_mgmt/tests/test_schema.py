"""Tests for schema.py — SchemaType, SchemaField, ConfigSchema."""

import pytest
from jetson.config_mgmt.schema import SchemaType, SchemaField, ConfigSchema


# ── SchemaType ──────────────────────────────────────────────────────────────

class TestSchemaType:
    def test_all_values_exist(self):
        expected = {"STRING", "INTEGER", "FLOAT", "BOOLEAN", "ARRAY", "OBJECT", "ENUM", "ANY"}
        actual = {t.name for t in SchemaType}
        assert actual == expected

    def test_string_value(self):
        assert SchemaType.STRING.value == "string"

    def test_integer_value(self):
        assert SchemaType.INTEGER.value == "integer"

    def test_float_value(self):
        assert SchemaType.FLOAT.value == "float"

    def test_boolean_value(self):
        assert SchemaType.BOOLEAN.value == "boolean"

    def test_enum_value(self):
        assert SchemaType.ENUM.value == "enum"

    def test_any_value(self):
        assert SchemaType.ANY.value == "any"

    def test_from_string(self):
        assert SchemaType("string") is SchemaType.STRING
        assert SchemaType("any") is SchemaType.ANY


# ── SchemaField ─────────────────────────────────────────────────────────────

class TestSchemaField:
    def test_creation_defaults(self):
        f = SchemaField(name="test")
        assert f.name == "test"
        assert f.type == SchemaType.ANY
        assert f.required is True
        assert f.default is None
        assert f.min_val is None
        assert f.max_val is None
        assert f.pattern is None
        assert f.description == ""
        assert f.allowed_values == []

    def test_creation_with_all_params(self):
        f = SchemaField(
            name="port", type=SchemaType.INTEGER, required=True,
            default=8080, min_val=1, max_val=65535,
            pattern=None, description="HTTP port",
            allowed_values=[]
        )
        assert f.name == "port"
        assert f.type == SchemaType.INTEGER
        assert f.default == 8080
        assert f.min_val == 1
        assert f.max_val == 65535

    def test_validate_string_valid(self):
        f = SchemaField(name="host", type=SchemaType.STRING)
        ok, errs = f.validate_value("localhost")
        assert ok is True
        assert errs == []

    def test_validate_string_invalid_type(self):
        f = SchemaField(name="host", type=SchemaType.STRING)
        ok, errs = f.validate_value(123)
        assert ok is False
        assert len(errs) == 1

    def test_validate_string_pattern_match(self):
        f = SchemaField(name="ip", type=SchemaType.STRING, pattern=r'\d+\.\d+\.\d+\.\d+')
        ok, errs = f.validate_value("192.168.1.1")
        assert ok is True

    def test_validate_string_pattern_mismatch(self):
        f = SchemaField(name="ip", type=SchemaType.STRING, pattern=r'\d+\.\d+\.\d+\.\d+')
        ok, errs = f.validate_value("not-an-ip")
        assert ok is False
        assert "pattern" in errs[0].lower()

    def test_validate_integer_valid(self):
        f = SchemaField(name="count", type=SchemaType.INTEGER)
        ok, errs = f.validate_value(42)
        assert ok is True

    def test_validate_integer_bool_rejected(self):
        f = SchemaField(name="flag", type=SchemaType.INTEGER)
        ok, errs = f.validate_value(True)
        assert ok is False  # bool is subclass of int, but we reject it

    def test_validate_integer_min_val(self):
        f = SchemaField(name="port", type=SchemaType.INTEGER, min_val=1)
        ok, errs = f.validate_value(0)
        assert ok is False
        assert "minimum" in errs[0].lower()

    def test_validate_integer_max_val(self):
        f = SchemaField(name="port", type=SchemaType.INTEGER, max_val=100)
        ok, errs = f.validate_value(101)
        assert ok is False
        assert "maximum" in errs[0].lower()

    def test_validate_integer_in_range(self):
        f = SchemaField(name="port", type=SchemaType.INTEGER, min_val=1, max_val=65535)
        ok, errs = f.validate_value(8080)
        assert ok is True

    def test_validate_float_valid(self):
        f = SchemaField(name="ratio", type=SchemaType.FLOAT)
        ok, errs = f.validate_value(3.14)
        assert ok is True

    def test_validate_float_int_accepted(self):
        f = SchemaField(name="ratio", type=SchemaType.FLOAT)
        ok, errs = f.validate_value(5)
        assert ok is True

    def test_validate_float_out_of_range(self):
        f = SchemaField(name="ratio", type=SchemaType.FLOAT, min_val=0.0, max_val=1.0)
        ok, errs = f.validate_value(1.5)
        assert ok is False

    def test_validate_boolean_true(self):
        f = SchemaField(name="debug", type=SchemaType.BOOLEAN)
        ok, errs = f.validate_value(True)
        assert ok is True

    def test_validate_boolean_invalid(self):
        f = SchemaField(name="debug", type=SchemaType.BOOLEAN)
        ok, errs = f.validate_value("yes")
        assert ok is False

    def test_validate_array_valid(self):
        f = SchemaField(name="items", type=SchemaType.ARRAY)
        ok, errs = f.validate_value([1, 2, 3])
        assert ok is True

    def test_validate_array_invalid(self):
        f = SchemaField(name="items", type=SchemaType.ARRAY)
        ok, errs = f.validate_value("not-array")
        assert ok is False

    def test_validate_object_valid(self):
        f = SchemaField(name="metadata", type=SchemaType.OBJECT)
        ok, errs = f.validate_value({"key": "val"})
        assert ok is True

    def test_validate_object_invalid(self):
        f = SchemaField(name="metadata", type=SchemaType.OBJECT)
        ok, errs = f.validate_value([1])
        assert ok is False

    def test_validate_enum_valid(self):
        f = SchemaField(name="env", type=SchemaType.ENUM, allowed_values=["dev", "prod"])
        ok, errs = f.validate_value("dev")
        assert ok is True

    def test_validate_enum_invalid(self):
        f = SchemaField(name="env", type=SchemaType.ENUM, allowed_values=["dev", "prod"])
        ok, errs = f.validate_value("staging")
        assert ok is False
        assert "allowed" in errs[0].lower()

    def test_validate_any_passes(self):
        f = SchemaField(name="anything", type=SchemaType.ANY)
        ok, errs = f.validate_value({"complex": [1, "two"]})
        assert ok is True

    def test_to_dict(self):
        f = SchemaField(name="host", type=SchemaType.STRING, required=False, default="0.0.0.0")
        d = f.to_dict()
        assert d["name"] == "host"
        assert d["type"] == "string"
        assert d["required"] is False
        assert d["default"] == "0.0.0.0"

    def test_from_dict_roundtrip(self):
        f = SchemaField(name="port", type=SchemaType.INTEGER, min_val=1, max_val=65535, default=80)
        d = f.to_dict()
        f2 = SchemaField.from_dict(d)
        assert f2.name == f.name
        assert f2.type == f.type
        assert f2.min_val == f.min_val
        assert f2.max_val == f.max_val
        assert f2.default == f.default


# ── ConfigSchema ────────────────────────────────────────────────────────────

class TestConfigSchema:
    def _make_schema(self):
        s = ConfigSchema(name="test", description="A test schema")
        s.add_field(SchemaField(name="host", type=SchemaType.STRING, default="localhost"))
        s.add_field(SchemaField(name="port", type=SchemaType.INTEGER, min_val=1, max_val=65535, default=8080))
        s.add_field(SchemaField(name="debug", type=SchemaType.BOOLEAN, default=False))
        s.add_field(SchemaField(name="timeout", type=SchemaType.FLOAT, min_val=0.0, default=30.0))
        return s

    def test_creation(self):
        s = ConfigSchema(name="app", description="App config")
        assert s.name == "app"
        assert s.description == "App config"
        assert s.fields == {}

    def test_add_field(self):
        s = ConfigSchema()
        f = SchemaField(name="key", type=SchemaType.STRING)
        s.add_field(f)
        assert "key" in s.fields
        assert s.fields["key"] is f

    def test_remove_field(self):
        s = ConfigSchema()
        s.add_field(SchemaField(name="key", type=SchemaType.STRING))
        s.remove_field("key")
        assert "key" not in s.fields

    def test_remove_field_not_found(self):
        s = ConfigSchema()
        with pytest.raises(KeyError):
            s.remove_field("nonexistent")

    def test_get_field_exists(self):
        s = ConfigSchema()
        f = SchemaField(name="x", type=SchemaType.INTEGER)
        s.add_field(f)
        assert s.get_field("x") is f

    def test_get_field_not_exists(self):
        s = ConfigSchema()
        assert s.get_field("missing") is None

    def test_fields_returns_copy(self):
        s = ConfigSchema()
        s.add_field(SchemaField(name="a", type=SchemaType.STRING))
        fields = s.fields
        fields["b"] = "hacked"
        assert "b" not in s.fields

    def test_validate_valid_config(self):
        s = self._make_schema()
        config = {"host": "0.0.0.0", "port": 443, "debug": True, "timeout": 30.0}
        ok, errs = s.validate(config)
        assert ok is True
        assert errs == []

    def test_validate_missing_required_no_default(self):
        s = ConfigSchema()
        s.add_field(SchemaField(name="name", type=SchemaType.STRING, required=True))
        ok, errs = s.validate({})
        assert ok is False
        assert any("name" in e for e in errs)

    def test_validate_missing_required_with_default_ok(self):
        s = self._make_schema()
        ok, errs = s.validate({})
        assert ok is True  # all have defaults

    def test_validate_type_error(self):
        s = self._make_schema()
        ok, errs = s.validate({"host": "0.0.0.0", "port": "not-a-number"})
        assert ok is False
        assert any("port" in e.lower() for e in errs)

    def test_validate_out_of_range(self):
        s = self._make_schema()
        ok, errs = s.validate({"port": 99999})
        assert ok is False
        assert any("maximum" in e.lower() for e in errs)

    def test_validate_unknown_field_allowed(self):
        s = self._make_schema()
        ok, errs = s.validate({"host": "ok", "extra_field": True})
        assert ok is True

    def test_validate_empty_schema(self):
        s = ConfigSchema()
        ok, errs = s.validate({"anything": "goes"})
        assert ok is True

    def test_merge_two_schemas(self):
        s1 = ConfigSchema(name="base")
        s1.add_field(SchemaField(name="a", type=SchemaType.STRING))
        s2 = ConfigSchema(name="ext")
        s2.add_field(SchemaField(name="b", type=SchemaType.INTEGER))
        merged = s1.merge(s2)
        assert "a" in merged.fields
        assert "b" in merged.fields

    def test_merge_override(self):
        s1 = ConfigSchema(name="base")
        s1.add_field(SchemaField(name="a", type=SchemaType.STRING))
        s2 = ConfigSchema(name="ext")
        s2.add_field(SchemaField(name="a", type=SchemaType.INTEGER))
        merged = s1.merge(s2)
        assert merged.get_field("a").type == SchemaType.INTEGER

    def test_to_dict(self):
        s = self._make_schema()
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "A test schema"
        assert "host" in d["fields"]
        assert d["fields"]["host"]["type"] == "string"

    def test_from_dict_roundtrip(self):
        s = self._make_schema()
        d = s.to_dict()
        s2 = ConfigSchema.from_dict(d)
        assert s2.name == s.name
        assert s2.get_field("host").type == SchemaType.STRING
        assert s2.get_field("port").type == SchemaType.INTEGER
        assert s2.get_field("debug").type == SchemaType.BOOLEAN
        assert s2.get_field("timeout").type == SchemaType.FLOAT

    def test_generate_default(self):
        s = self._make_schema()
        defaults = s.generate_default()
        assert defaults["host"] == "localhost"
        assert defaults["port"] == 8080
        assert defaults["debug"] is False

    def test_generate_default_empty(self):
        s = ConfigSchema()
        assert s.generate_default() == {}

    def test_compute_schema_hash_consistent(self):
        s = self._make_schema()
        h1 = s.compute_schema_hash()
        h2 = s.compute_schema_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_compute_schema_hash_differs(self):
        s1 = ConfigSchema(name="one")
        s1.add_field(SchemaField(name="x", type=SchemaType.STRING))
        s2 = ConfigSchema(name="two")
        s2.add_field(SchemaField(name="x", type=SchemaType.INTEGER))
        assert s1.compute_schema_hash() != s2.compute_schema_hash()

    def test_validate_float_negative_ok(self):
        s = ConfigSchema()
        s.add_field(SchemaField(name="temp", type=SchemaType.FLOAT, min_val=-40.0, max_val=80.0))
        ok, errs = s.validate({"temp": -10.5})
        assert ok is True

    def test_validate_multiple_errors(self):
        s = self._make_schema()
        ok, errs = s.validate({"port": -1, "timeout": "bad"})
        assert ok is False
        assert len(errs) >= 2
