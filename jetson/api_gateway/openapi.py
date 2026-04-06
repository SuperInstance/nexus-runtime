"""OpenAPI 3.0 specification generation from route definitions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class OpenAPIParam:
    """An OpenAPI parameter definition."""
    name: str
    location: str = "query"       # query, header, path, cookie
    type: str = "string"
    required: bool = False
    description: str = ""


@dataclass
class OpenAPIResponse:
    """An OpenAPI response definition."""
    status_code: str = "200"
    description: str = ""
    schema: Optional[Dict[str, Any]] = None


@dataclass
class _PathEntry:
    """Internal representation of a registered path."""
    method: str
    path: str
    summary: str = ""
    params: List[OpenAPIParam] = field(default_factory=list)
    responses: List[OpenAPIResponse] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    operation_id: str = ""


class OpenAPISpec:
    """Generates OpenAPI 3.0 JSON specifications.

    Provides methods to register paths, schemas, security schemes,
    servers, and tags, then produces a complete OpenAPI 3.0 document.
    """

    def __init__(
        self,
        title: str = "NEXUS API",
        version: str = "1.0.0",
        description: str = "",
    ) -> None:
        self._title = title
        self._version = version
        self._description = description
        self._paths: Dict[str, Dict[str, Any]] = {}
        self._path_entries: List[_PathEntry] = []
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._security_schemes: Dict[str, Dict[str, Any]] = {}
        self._servers: List[Dict[str, str]] = []
        self._tags: List[Dict[str, str]] = {}

    def add_path(
        self,
        method: str,
        path: str,
        summary: str = "",
        params: Optional[List[OpenAPIParam]] = None,
        responses: Optional[List[OpenAPIResponse]] = None,
        tags: Optional[List[str]] = None,
        operation_id: str = "",
    ) -> None:
        """Register an API path.

        ``method`` should be lowercase (get, post, etc.).
        """
        entry = _PathEntry(
            method=method.lower(),
            path=path,
            summary=summary,
            params=params or [],
            responses=responses or [],
            tags=tags or [],
            operation_id=operation_id,
        )
        self._path_entries.append(entry)
        self._build_path(entry)

    def _build_path(self, entry: _PathEntry) -> None:
        """Build the OpenAPI path entry structure."""
        if entry.path not in self._paths:
            self._paths[entry.path] = {}

        method_obj: Dict[str, Any] = {}
        if entry.summary:
            method_obj["summary"] = entry.summary
        if entry.operation_id:
            method_obj["operationId"] = entry.operation_id
        if entry.tags:
            method_obj["tags"] = entry.tags

        # Parameters
        if entry.params:
            method_obj["parameters"] = []
            for p in entry.params:
                param_obj: Dict[str, Any] = {
                    "name": p.name,
                    "in": p.location,
                    "required": p.required,
                    "schema": {"type": p.type},
                }
                if p.description:
                    param_obj["description"] = p.description
                method_obj["parameters"].append(param_obj)

        # Responses
        if entry.responses:
            method_obj["responses"] = {}
            for r in entry.responses:
                resp_obj: Dict[str, Any] = {"description": r.description}
                if r.schema:
                    resp_obj["content"] = {
                        "application/json": {"schema": r.schema}
                    }
                method_obj["responses"][r.status_code] = resp_obj

        self._paths[entry.path][entry.method] = method_obj

    def add_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Add a reusable schema component."""
        self._schemas[name] = schema

    def remove_schema(self, name: str) -> bool:
        """Remove a schema. Returns True if it existed."""
        if name in self._schemas:
            del self._schemas[name]
            return True
        return False

    def add_security_scheme(self, name: str, scheme_type: str, **kwargs: Any) -> None:
        """Add a security scheme (apiKey, http, oauth2, openIdConnect)."""
        scheme: Dict[str, Any] = {"type": scheme_type}
        scheme.update(kwargs)
        self._security_schemes[name] = scheme

    def add_server(self, url: str, description: str = "") -> None:
        """Add a server URL."""
        server: Dict[str, str] = {"url": url}
        if description:
            server["description"] = description
        self._servers.append(server)

    def add_tag(self, name: str, description: str = "") -> None:
        """Add a tag definition."""
        self._tags[name] = {"name": name, "description": description}

    def remove_tag(self, name: str) -> bool:
        """Remove a tag. Returns True if it existed."""
        if name in self._tags:
            del self._tags[name]
            return True
        return False

    def generate_spec(self) -> Dict[str, Any]:
        """Generate the full OpenAPI 3.0 specification as a dict."""
        spec: Dict[str, Any] = {
            "openapi": "3.0.0",
            "info": {
                "title": self._title,
                "version": self._version,
            },
            "paths": self._paths,
        }

        if self._description:
            spec["info"]["description"] = self._description

        if self._servers:
            spec["servers"] = self._servers

        tag_list = list(self._tags.values())
        if tag_list:
            spec["tags"] = tag_list

        components: Dict[str, Any] = {}

        if self._schemas:
            components["schemas"] = self._schemas

        if self._security_schemes:
            components["securitySchemes"] = self._security_schemes

        if components:
            spec["components"] = components

        return spec

    def validate_spec(self, spec: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """Validate an OpenAPI 3.0 spec. Returns (valid, errors).

        Checks for required fields and structural correctness.
        """
        if spec is None:
            spec = self.generate_spec()

        errors: List[str] = []

        # Must be a dict
        if not isinstance(spec, dict):
            return (False, ["Spec must be a dict"])

        # openapi version
        if "openapi" not in spec:
            errors.append("Missing 'openapi' field")
        elif not isinstance(spec["openapi"], str):
            errors.append("'openapi' must be a string")

        # info
        if "info" not in spec:
            errors.append("Missing 'info' field")
        else:
            info = spec["info"]
            if not isinstance(info, dict):
                errors.append("'info' must be a dict")
            else:
                if "title" not in info:
                    errors.append("Missing 'info.title'")
                if "version" not in info:
                    errors.append("Missing 'info.version'")

        # paths
        if "paths" not in spec:
            errors.append("Missing 'paths' field")
        elif not isinstance(spec["paths"], dict):
            errors.append("'paths' must be a dict")
        else:
            for path_name, methods in spec["paths"].items():
                if not isinstance(methods, dict):
                    errors.append(f"'paths.{path_name}' must be a dict")
                    continue
                if not path_name.startswith("/"):
                    errors.append(f"Path '{path_name}' must start with /")
                for method_name, method_spec in methods.items():
                    if method_name in ("get", "post", "put", "delete", "patch", "options"):
                        if not isinstance(method_spec, dict):
                            errors.append(
                                f"'paths.{path_name}.{method_name}' must be a dict"
                            )
                    # Skip $ref and other extensions

        # components
        if "components" in spec:
            if not isinstance(spec["components"], dict):
                errors.append("'components' must be a dict")
            else:
                if "securitySchemes" in spec["components"]:
                    sec = spec["components"]["securitySchemes"]
                    if not isinstance(sec, dict):
                        errors.append("'components.securitySchemes' must be a dict")

        # servers
        if "servers" in spec:
            if not isinstance(spec["servers"], list):
                errors.append("'servers' must be a list")
            else:
                for i, server in enumerate(spec["servers"]):
                    if not isinstance(server, dict):
                        errors.append(f"'servers[{i}]' must be a dict")
                    elif "url" not in server:
                        errors.append(f"'servers[{i}]' must have 'url'")

        return (len(errors) == 0, errors)

    def list_paths(self) -> List[str]:
        """Return all registered paths."""
        return list(self._paths.keys())

    def list_schemas(self) -> List[str]:
        """Return all registered schema names."""
        return list(self._schemas.keys())

    def path_count(self) -> int:
        """Return the number of registered paths."""
        return len(self._paths)

    def to_json(self, indent: int = 2) -> str:
        """Generate spec and serialize to JSON string."""
        import json
        return json.dumps(self.generate_spec(), indent=indent)
