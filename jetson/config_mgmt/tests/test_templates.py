"""Tests for templates.py — ConfigTemplate, TemplateEngine."""

import pytest
from jetson.config_mgmt.templates import ConfigTemplate, TemplateEngine
from jetson.config_mgmt.schema import ConfigSchema, SchemaField, SchemaType


class TestConfigTemplate:
    def test_creation_defaults(self):
        tpl = ConfigTemplate(name="base")
        assert tpl.name == "base"
        assert tpl.template_vars == {}
        assert tpl.base_config == {}
        assert tpl.overrides == {}

    def test_creation_with_data(self):
        tpl = ConfigTemplate(
            name="dev",
            template_vars={"port": 8080},
            base_config={"host": "localhost"},
            overrides={"debug": True},
        )
        assert tpl.name == "dev"
        assert tpl.template_vars == {"port": 8080}
        assert tpl.base_config == {"host": "localhost"}
        assert tpl.overrides == {"debug": True}


class TestTemplateEngine:
    def _make_engine(self):
        return TemplateEngine()

    # ── create_template ─────────────────────────────────────────────────

    def test_create_template_basic(self):
        engine = self._make_engine()
        tpl = engine.create_template("base", {"host": "localhost", "port": 8080})
        assert tpl.name == "base"
        assert tpl.base_config == {"host": "localhost", "port": 8080}
        assert "base" in engine.list_templates()

    def test_create_template_with_variables(self):
        engine = self._make_engine()
        tpl = engine.create_template(
            "env", {"host": "{{HOST}}"}, variables={"HOST": "localhost"}
        )
        assert tpl.template_vars == {"HOST": "localhost"}

    def test_create_template_overwrites(self):
        engine = self._make_engine()
        engine.create_template("base", {"a": 1})
        engine.create_template("base", {"a": 2})
        assert engine.get_template("base").base_config == {"a": 2}

    def test_create_template_registers(self):
        engine = self._make_engine()
        engine.create_template("app", {"x": 1})
        assert engine.get_template("app") is not None
        assert engine.get_template("app").name == "app"

    # ── render ──────────────────────────────────────────────────────────

    def test_render_no_vars(self):
        engine = self._make_engine()
        tpl = engine.create_template("simple", {"host": "localhost", "port": 8080})
        result = engine.render(tpl, {})
        assert result == {"host": "localhost", "port": 8080}

    def test_render_with_var_substitution(self):
        engine = self._make_engine()
        tpl = engine.create_template("tpl", {"host": "{{HOST}}", "port": "{{PORT}}"})
        result = engine.render(tpl, {"HOST": "db.local", "PORT": 5432})
        assert result["host"] == "db.local"
        assert result["port"] == 5432

    def test_render_partial_substitution(self):
        engine = self._make_engine()
        tpl = engine.create_template("tpl", {"url": "http://{{HOST}}:{{PORT}}/api"})
        result = engine.render(tpl, {"HOST": "api.local", "PORT": 443})
        assert result["url"] == "http://api.local:443/api"

    def test_render_unresolved_var_kept(self):
        engine = self._make_engine()
        tpl = engine.create_template("tpl", {"key": "{{MISSING}}"})
        result = engine.render(tpl, {})
        assert result["key"] == "{{MISSING}}"

    def test_render_with_template_defaults(self):
        engine = self._make_engine()
        tpl = engine.create_template(
            "tpl", {"host": "{{HOST}}"}, variables={"HOST": "default.local"}
        )
        result = engine.render(tpl, {})
        assert result["host"] == "default.local"

    def test_render_override_template_vars(self):
        engine = self._make_engine()
        tpl = engine.create_template(
            "tpl", {"host": "{{HOST}}"}, variables={"HOST": "default.local"}
        )
        result = engine.render(tpl, {"HOST": "override.local"})
        assert result["host"] == "override.local"

    def test_render_with_overrides(self):
        engine = self._make_engine()
        tpl = ConfigTemplate(
            name="tpl",
            base_config={"host": "localhost", "port": 8080},
            overrides={"port": 9090},
        )
        result = engine.render(tpl, {})
        assert result["port"] == 9090

    def test_render_nested_config(self):
        engine = self._make_engine()
        tpl = engine.create_template("tpl", {
            "database": {"host": "{{DB_HOST}}", "name": "mydb"}
        })
        result = engine.render(tpl, {"DB_HOST": "db.local"})
        assert result["database"]["host"] == "db.local"
        assert result["database"]["name"] == "mydb"

    def test_render_does_not_mutate_template(self):
        engine = self._make_engine()
        tpl = engine.create_template("tpl", {"host": "{{HOST}}", "port": 8080})
        engine.render(tpl, {"HOST": "localhost"})
        assert tpl.base_config["host"] == "{{HOST}}"

    def test_render_var_in_list(self):
        engine = self._make_engine()
        tpl = engine.create_template("tpl", {"urls": ["http://{{HOST}}", "http://{{HOST}}:443"]})
        result = engine.render(tpl, {"HOST": "api.local"})
        assert result["urls"] == ["http://api.local", "http://api.local:443"]

    def test_render_mixed_types(self):
        engine = self._make_engine()
        tpl = engine.create_template("tpl", {
            "name": "{{NAME}}",
            "count": 5,
            "debug": True,
        })
        result = engine.render(tpl, {"NAME": "myapp"})
        assert result["name"] == "myapp"
        assert result["count"] == 5
        assert result["debug"] is True

    # ── extend_template ─────────────────────────────────────────────────

    def test_extend_template(self):
        engine = self._make_engine()
        base = engine.create_template("base", {"host": "localhost", "port": 8080})
        extended = engine.extend_template(base, {"port": 9090, "debug": True})
        assert extended.name == "base_extended"
        assert extended.base_config["host"] == "localhost"
        assert extended.base_config["port"] == 9090
        assert extended.base_config["debug"] is True
        assert "base_extended" in engine.list_templates()

    def test_extend_deep_merge(self):
        engine = self._make_engine()
        base = engine.create_template("base", {
            "db": {"host": "db1", "port": 5432}, "cache": True
        })
        extended = engine.extend_template(base, {"db": {"host": "db2"}})
        assert extended.base_config["db"]["host"] == "db2"
        assert extended.base_config["db"]["port"] == 5432
        assert extended.base_config["cache"] is True

    def test_extend_preserves_template_vars(self):
        engine = self._make_engine()
        base = engine.create_template(
            "base", {"host": "{{HOST}}"}, variables={"HOST": "default"}
        )
        extended = engine.extend_template(base, {"port": 8080})
        assert extended.template_vars == {"HOST": "default"}

    # ── compute_diff ────────────────────────────────────────────────────

    def test_diff_identical(self):
        engine = self._make_engine()
        a = ConfigTemplate(name="a", base_config={"x": 1, "y": 2})
        b = ConfigTemplate(name="b", base_config={"x": 1, "y": 2})
        diff = engine.compute_diff(a, b)
        assert diff["added"] == {}
        assert diff["removed"] == {}
        assert diff["changed"] == {}

    def test_diff_added_keys(self):
        engine = self._make_engine()
        a = ConfigTemplate(name="a", base_config={"x": 1})
        b = ConfigTemplate(name="b", base_config={"x": 1, "y": 2})
        diff = engine.compute_diff(a, b)
        assert diff["added"] == {"y": 2}
        assert diff["removed"] == {}

    def test_diff_removed_keys(self):
        engine = self._make_engine()
        a = ConfigTemplate(name="a", base_config={"x": 1, "y": 2})
        b = ConfigTemplate(name="b", base_config={"x": 1})
        diff = engine.compute_diff(a, b)
        assert diff["added"] == {}
        assert diff["removed"] == {"y": 2}

    def test_diff_changed_values(self):
        engine = self._make_engine()
        a = ConfigTemplate(name="a", base_config={"port": 8080})
        b = ConfigTemplate(name="b", base_config={"port": 9090})
        diff = engine.compute_diff(a, b)
        assert "port" in diff["changed"]
        assert diff["changed"]["port"]["old"] == 8080
        assert diff["changed"]["port"]["new"] == 9090

    def test_diff_nested_changes(self):
        engine = self._make_engine()
        a = ConfigTemplate(name="a", base_config={"db": {"host": "db1", "port": 5432}})
        b = ConfigTemplate(name="b", base_config={"db": {"host": "db2", "port": 5432}})
        diff = engine.compute_diff(a, b)
        assert "db" in diff["changed"]
        assert "host" in diff["changed"]["db"]["changed"]

    # ── list_templates / get_template ───────────────────────────────────

    def test_list_templates_empty(self):
        engine = self._make_engine()
        assert engine.list_templates() == []

    def test_list_templates_sorted(self):
        engine = self._make_engine()
        engine.create_template("charlie", {})
        engine.create_template("alpha", {})
        engine.create_template("bravo", {})
        assert engine.list_templates() == ["alpha", "bravo", "charlie"]

    def test_get_template_found(self):
        engine = self._make_engine()
        tpl = engine.create_template("found", {"x": 1})
        assert engine.get_template("found") is tpl

    def test_get_template_not_found(self):
        engine = self._make_engine()
        assert engine.get_template("missing") is None

    # ── validate_template ───────────────────────────────────────────────

    def test_validate_template_valid(self):
        engine = self._make_engine()
        schema = ConfigSchema(name="test")
        schema.add_field(SchemaField(name="port", type=SchemaType.INTEGER))
        tpl = engine.create_template("tpl", {"port": 8080})
        ok, errs = engine.validate_template(tpl, schema)
        assert ok is True

    def test_validate_template_invalid(self):
        engine = self._make_engine()
        schema = ConfigSchema(name="test")
        schema.add_field(SchemaField(name="port", type=SchemaType.INTEGER, required=True))
        tpl = engine.create_template("tpl", {"host": "localhost"})
        ok, errs = engine.validate_template(tpl, schema)
        assert ok is False
        assert len(errs) > 0

    def test_multiple_templates_independent(self):
        engine = self._make_engine()
        t1 = engine.create_template("t1", {"a": 1})
        t2 = engine.create_template("t2", {"b": 2})
        assert engine.render(t1, {}) == {"a": 1}
        assert engine.render(t2, {}) == {"b": 2}
        assert engine.list_templates() == ["t1", "t2"]
