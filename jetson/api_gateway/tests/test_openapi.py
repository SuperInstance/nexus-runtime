"""Tests for openapi.py — OpenAPI 3.0 specification generation."""

import json

import pytest

from jetson.api_gateway.openapi import OpenAPIParam, OpenAPIResponse, OpenAPISpec


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def spec():
    return OpenAPISpec(title="Test API", version="2.0.0", description="A test API")


# ── OpenAPIParam ─────────────────────────────────────────────────────

class TestOpenAPIParam:
    def test_defaults(self):
        p = OpenAPIParam(name="page")
        assert p.name == "page"
        assert p.location == "query"
        assert p.type == "string"
        assert p.required is False
        assert p.description == ""

    def test_custom(self):
        p = OpenAPIParam(name="id", location="path", type="integer", required=True, description="User ID")
        assert p.location == "path"
        assert p.type == "integer"
        assert p.required is True
        assert p.description == "User ID"


# ── OpenAPIResponse ──────────────────────────────────────────────────

class TestOpenAPIResponse:
    def test_defaults(self):
        r = OpenAPIResponse()
        assert r.status_code == "200"
        assert r.description == ""
        assert r.schema is None

    def test_with_schema(self):
        r = OpenAPIResponse(
            status_code="201",
            description="Created",
            schema={"type": "object", "properties": {"id": {"type": "string"}}},
        )
        assert r.status_code == "201"
        assert r.schema is not None


# ── OpenAPISpec: add_path ────────────────────────────────────────────

class TestAddPath:
    def test_add_get_path(self, spec):
        spec.add_path("get", "/users", summary="List users")
        paths = spec.list_paths()
        assert "/users" in paths

    def test_add_post_path(self, spec):
        spec.add_path("post", "/users", summary="Create user")
        generated = spec.generate_spec()
        assert "post" in generated["paths"]["/users"]

    def test_add_multiple_methods(self, spec):
        spec.add_path("get", "/items", summary="List items")
        spec.add_path("post", "/items", summary="Create item")
        generated = spec.generate_spec()
        assert "get" in generated["paths"]["/items"]
        assert "post" in generated["paths"]["/items"]

    def test_add_with_params(self, spec):
        params = [
            OpenAPIParam(name="id", location="path", type="integer", required=True),
            OpenAPIParam(name="expand", location="query", type="string"),
        ]
        spec.add_path("get", "/users/{id}", params=params)
        generated = spec.generate_spec()
        path_params = generated["paths"]["/users/{id}"]["get"]["parameters"]
        assert len(path_params) == 2
        assert path_params[0]["name"] == "id"
        assert path_params[0]["in"] == "path"
        assert path_params[0]["required"] is True

    def test_add_with_responses(self, spec):
        responses = [
            OpenAPIResponse(status_code="200", description="OK"),
            OpenAPIResponse(status_code="404", description="Not found"),
        ]
        spec.add_path("get", "/users/{id}", responses=responses)
        generated = spec.generate_spec()
        resp_obj = generated["paths"]["/users/{id}"]["get"]["responses"]
        assert "200" in resp_obj
        assert "404" in resp_obj
        assert resp_obj["404"]["description"] == "Not found"

    def test_add_with_response_schema(self, spec):
        responses = [
            OpenAPIResponse(
                status_code="200",
                description="OK",
                schema={"type": "array", "items": {"type": "string"}},
            )
        ]
        spec.add_path("get", "/tags", responses=responses)
        generated = spec.generate_spec()
        content = generated["paths"]["/tags"]["get"]["responses"]["200"]["content"]
        assert "application/json" in content

    def test_add_with_tags(self, spec):
        spec.add_path("get", "/users", tags=["users"])
        generated = spec.generate_spec()
        assert generated["paths"]["/users"]["get"]["tags"] == ["users"]

    def test_add_with_operation_id(self, spec):
        spec.add_path("get", "/users", operation_id="listUsers")
        generated = spec.generate_spec()
        assert generated["paths"]["/users"]["get"]["operationId"] == "listUsers"

    def test_path_count(self, spec):
        spec.add_path("get", "/a")
        spec.add_path("get", "/b")
        spec.add_path("post", "/a")
        assert spec.path_count() == 2  # /a and /b

    def test_method_case_normalized(self, spec):
        spec.add_path("GET", "/uppercase")
        generated = spec.generate_spec()
        assert "get" in generated["paths"]["/uppercase"]


# ── OpenAPISpec: add_schema ──────────────────────────────────────────

class TestAddSchema:
    def test_add_schema(self, spec):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        spec.add_schema("User", schema)
        assert "User" in spec.list_schemas()

    def test_remove_schema(self, spec):
        spec.add_schema("User", {"type": "object"})
        assert spec.remove_schema("User") is True
        assert "User" not in spec.list_schemas()

    def test_remove_nonexistent(self, spec):
        assert spec.remove_schema("Nope") is False

    def test_schema_in_generated_spec(self, spec):
        schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
        spec.add_schema("Item", schema)
        generated = spec.generate_spec()
        assert generated["components"]["schemas"]["Item"] == schema


# ── OpenAPISpec: add_security_scheme ─────────────────────────────────

class TestAddSecurityScheme:
    def test_add_api_key_scheme(self, spec):
        spec.add_security_scheme("ApiKeyAuth", "apiKey", **{"in": "header", "name_field": "X-API-Key"})
        generated = spec.generate_spec()
        scheme = generated["components"]["securitySchemes"]["ApiKeyAuth"]
        assert scheme["type"] == "apiKey"
        assert scheme["in"] == "header"

    def test_add_bearer_scheme(self, spec):
        spec.add_security_scheme("BearerAuth", "http", scheme="bearer", bearerFormat="JWT")
        generated = spec.generate_spec()
        scheme = generated["components"]["securitySchemes"]["BearerAuth"]
        assert scheme["type"] == "http"


# ── OpenAPISpec: add_server ──────────────────────────────────────────

class TestAddServer:
    def test_add_server(self, spec):
        spec.add_server("https://api.example.com", "Production")
        generated = spec.generate_spec()
        assert len(generated["servers"]) == 1
        assert generated["servers"][0]["url"] == "https://api.example.com"
        assert generated["servers"][0]["description"] == "Production"

    def test_add_server_no_description(self, spec):
        spec.add_server("http://localhost:8080")
        generated = spec.generate_spec()
        assert generated["servers"][0]["url"] == "http://localhost:8080"

    def test_add_multiple_servers(self, spec):
        spec.add_server("https://prod.api.com", "Production")
        spec.add_server("https://staging.api.com", "Staging")
        generated = spec.generate_spec()
        assert len(generated["servers"]) == 2


# ── OpenAPISpec: add_tag ─────────────────────────────────────────────

class TestAddTag:
    def test_add_tag(self, spec):
        spec.add_tag("users", "User management")
        generated = spec.generate_spec()
        assert len(generated["tags"]) == 1
        assert generated["tags"][0]["name"] == "users"
        assert generated["tags"][0]["description"] == "User management"

    def test_remove_tag(self, spec):
        spec.add_tag("users", "User management")
        assert spec.remove_tag("users") is True
        generated = spec.generate_spec()
        assert "tags" not in generated or len(generated["tags"]) == 0

    def test_remove_nonexistent_tag(self, spec):
        assert spec.remove_tag("nope") is False


# ── OpenAPISpec: generate_spec ───────────────────────────────────────

class TestGenerateSpec:
    def test_openapi_version(self, spec):
        generated = spec.generate_spec()
        assert generated["openapi"] == "3.0.0"

    def test_info_title(self, spec):
        generated = spec.generate_spec()
        assert generated["info"]["title"] == "Test API"

    def test_info_version(self, spec):
        generated = spec.generate_spec()
        assert generated["info"]["version"] == "2.0.0"

    def test_info_description(self, spec):
        generated = spec.generate_spec()
        assert generated["info"]["description"] == "A test API"

    def test_empty_spec(self, spec):
        generated = spec.generate_spec()
        assert generated["paths"] == {}

    def test_no_description_omitted(self):
        spec = OpenAPISpec(title="API", version="1.0")
        generated = spec.generate_spec()
        assert "description" not in generated["info"]

    def test_components_only_when_schemas(self, spec):
        generated = spec.generate_spec()
        assert "components" not in generated
        spec.add_schema("X", {"type": "string"})
        generated = spec.generate_spec()
        assert "components" in generated

    def test_servers_only_when_added(self, spec):
        generated = spec.generate_spec()
        assert "servers" not in generated
        spec.add_server("http://localhost")
        generated = spec.generate_spec()
        assert "servers" in generated


# ── OpenAPISpec: validate_spec ───────────────────────────────────────

class TestValidateSpec:
    def test_valid_generated_spec(self, spec):
        spec.add_path("get", "/health", summary="Health check")
        valid, errors = spec.validate_spec()
        assert valid is True
        assert errors == []

    def test_minimal_valid_spec(self):
        s = OpenAPISpec(title="API", version="1.0")
        generated = s.generate_spec()
        valid, errors = s.validate_spec(generated)
        assert valid is True

    def test_missing_openapi(self, spec):
        valid, errors = spec.validate_spec({"info": {"title": "x", "version": "1"}, "paths": {}})
        assert valid is False
        assert any("openapi" in e for e in errors)

    def test_missing_info(self, spec):
        valid, errors = spec.validate_spec({"openapi": "3.0.0", "paths": {}})
        assert valid is False
        assert any("info" in e for e in errors)

    def test_missing_paths(self, spec):
        valid, errors = spec.validate_spec({"openapi": "3.0.0", "info": {"title": "x", "version": "1"}})
        assert valid is False
        assert any("paths" in e for e in errors)

    def test_missing_title(self, spec):
        valid, errors = spec.validate_spec({"openapi": "3.0.0", "info": {"version": "1"}, "paths": {}})
        assert valid is False
        assert any("title" in e for e in errors)

    def test_missing_version(self, spec):
        valid, errors = spec.validate_spec({"openapi": "3.0.0", "info": {"title": "x"}, "paths": {}})
        assert valid is False
        assert any("version" in e for e in errors)

    def test_non_dict_spec(self, spec):
        valid, errors = spec.validate_spec("not a dict")
        assert valid is False
        assert any("dict" in e for e in errors)

    def test_path_not_starting_with_slash(self, spec):
        bad_spec = {
            "openapi": "3.0.0",
            "info": {"title": "X", "version": "1"},
            "paths": {"badpath": {"get": {"summary": "x"}}},
        }
        valid, errors = spec.validate_spec(bad_spec)
        assert valid is False
        assert any("must start with" in e for e in errors)

    def test_invalid_servers(self, spec):
        bad_spec = {
            "openapi": "3.0.0",
            "info": {"title": "X", "version": "1"},
            "paths": {},
            "servers": "not a list",
        }
        valid, errors = spec.validate_spec(bad_spec)
        assert valid is False
        assert any("servers" in e for e in errors)

    def test_server_missing_url(self, spec):
        bad_spec = {
            "openapi": "3.0.0",
            "info": {"title": "X", "version": "1"},
            "paths": {},
            "servers": [{"description": "no url"}],
        }
        valid, errors = spec.validate_spec(bad_spec)
        assert valid is False
        assert any("url" in e for e in errors)

    def test_validate_uses_generated_when_none(self, spec):
        spec.add_path("get", "/ok")
        valid, errors = spec.validate_spec(None)
        assert valid is True

    def test_multiple_errors(self, spec):
        valid, errors = spec.validate_spec({})
        assert valid is False
        assert len(errors) >= 2


# ── OpenAPISpec: list_paths / list_schemas / path_count / to_json ───

class TestSpecUtilities:
    def test_list_paths(self, spec):
        spec.add_path("get", "/a")
        spec.add_path("post", "/b")
        paths = spec.list_paths()
        assert "/a" in paths
        assert "/b" in paths

    def test_list_schemas(self, spec):
        spec.add_schema("S1", {})
        spec.add_schema("S2", {})
        schemas = spec.list_schemas()
        assert schemas == ["S1", "S2"]

    def test_path_count_empty(self, spec):
        assert spec.path_count() == 0

    def test_to_json(self, spec):
        spec.add_path("get", "/health")
        json_str = spec.to_json()
        parsed = json.loads(json_str)
        assert parsed["openapi"] == "3.0.0"

    def test_to_json_indent(self, spec):
        json_str = spec.to_json(indent=4)
        assert "    " in json_str  # 4-space indent

    def test_full_spec_round_trip(self, spec):
        """Generate, validate, serialize, and parse."""
        spec.add_path("get", "/users", summary="List users", tags=["users"])
        spec.add_path("post", "/users", summary="Create user", tags=["users"])
        spec.add_path("get", "/users/{id}", summary="Get user", tags=["users"],
                      params=[OpenAPIParam(name="id", location="path", type="integer", required=True)],
                      responses=[OpenAPIResponse(status_code="200", description="OK"),
                                 OpenAPIResponse(status_code="404", description="Not found")])
        spec.add_schema("User", {"type": "object", "properties": {"id": {"type": "string"}, "name": {"type": "string"}}})
        spec.add_security_scheme("BearerAuth", "http", scheme="bearer")
        spec.add_server("https://api.test.com", "Test")
        spec.add_tag("users", "User operations")

        generated = spec.generate_spec()
        valid, errors = spec.validate_spec(generated)
        assert valid is True, f"Validation errors: {errors}"

        # Round-trip through JSON
        json_str = json.dumps(generated)
        parsed = json.loads(json_str)
        assert parsed["openapi"] == "3.0.0"
        assert parsed["info"]["title"] == "Test API"
        assert len(parsed["paths"]) == 2
        assert "components" in parsed
        assert "schemas" in parsed["components"]
        assert "securitySchemes" in parsed["components"]
