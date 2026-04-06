"""URL routing with path parameters, named routes, and multi-method matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class HTTPMethod(Enum):
    """Supported HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"


@dataclass(frozen=True)
class Route:
    """A registered route definition."""
    path: str
    method: HTTPMethod
    handler: Callable
    name: str = ""
    middleware: list = field(default_factory=list)


@dataclass
class Request:
    """Incoming HTTP request representation."""
    method: HTTPMethod
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[Any] = None
    path_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class Response:
    """Outgoing HTTP response representation."""
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Any] = None
    content_type: str = "application/json"


# Regex for path parameter extraction: {name}
_PARAM_RE = re.compile(r"\{(\w+)(?::([^}]+))?\}")


def _compile_path_pattern(path: str) -> re.Pattern:
    """Convert a path like /users/{id}/posts/{post_id} to a compiled regex.

    Supports optional type hints like {id:int} and {name:str}.
    """
    param_names: list[str] = []
    regex_parts = ["^"]

    last_end = 0
    for match in _PARAM_RE.finditer(path):
        # Literal segment before the parameter
        literal = path[last_end:match.start()]
        regex_parts.append(re.escape(literal))

        param_name = match.group(1)
        type_hint = match.group(2)
        param_names.append(param_name)

        if type_hint == "int":
            regex_parts.append(r"(?P<" + param_name + r">[0-9]+)")
        elif type_hint == "str":
            regex_parts.append(r"(?P<" + param_name + r">[^/]+)")
        else:
            regex_parts.append(r"(?P<" + param_name + r">[^/]+)")

        last_end = match.end()

    # Trailing literal
    regex_parts.append(re.escape(path[last_end:]))
    regex_parts.append("$")

    pattern = "".join(regex_parts)
    return re.compile(pattern)


class Router:
    """URL router supporting path parameters, named routes, and method-based dispatch."""

    def __init__(self) -> None:
        self._routes: List[Route] = []
        self._named_routes: Dict[str, Route] = {}
        self._compiled_cache: Dict[Tuple[str, HTTPMethod], re.Pattern] = {}

    def add_route(
        self,
        method: HTTPMethod,
        path: str,
        handler: Callable,
        name: str = "",
        middleware: Optional[list] = None,
    ) -> Route:
        """Register a new route. Returns the created Route."""
        route = Route(
            path=path,
            method=method,
            handler=handler,
            name=name or f"{method.value} {path}",
            middleware=middleware or [],
        )
        self._routes.append(route)

        if name:
            self._named_routes[name] = route

        # Pre-compile pattern
        self._compiled_cache[(path, method)] = _compile_path_pattern(path)

        return route

    def match(self, request: Request) -> Optional[Tuple[Callable, Dict[str, str]]]:
        """Match a request to a handler. Returns (handler, path_params) or None.

        First match wins (routes registered in order).
        """
        for route in self._routes:
            if route.method != request.method:
                continue

            pattern = self._compiled_cache.get((route.path, route.method))
            if pattern is None:
                pattern = _compile_path_pattern(route.path)
                self._compiled_cache[(route.path, route.method)] = pattern

            m = pattern.match(request.path)
            if m:
                return (route.handler, m.groupdict())

        return None

    def match_any(self, request: Request) -> List[Tuple[Callable, Dict[str, str]]]:
        """Return all matching (handler, path_params) for a request."""
        results = []
        for route in self._routes:
            if route.method != request.method:
                continue

            pattern = self._compiled_cache.get((route.path, route.method))
            if pattern is None:
                pattern = _compile_path_pattern(route.path)
                self._compiled_cache[(route.path, route.method)] = pattern

            m = pattern.match(request.path)
            if m:
                results.append((route.handler, m.groupdict()))

        return results

    def build_url(self, route_name: str, params: Optional[Dict[str, str]] = None) -> str:
        """Build a URL from a named route and path parameters.

        Raises KeyError if the route name doesn't exist or a parameter is missing.
        """
        route = self._named_routes[route_name]
        path = route.path

        if params:
            for key, value in params.items():
                placeholder = "{" + key + "}"
                typed_placeholder = "{" + key + ":"
                if placeholder in path:
                    path = path.replace(placeholder, str(value))
                else:
                    # Try to replace typed placeholder
                    for part in _PARAM_RE.finditer(route.path):
                        if part.group(1) == key:
                            full = part.group(0)
                            path = path.replace(full, str(value), 1)
                            break

        return path

    def list_routes(self) -> List[Route]:
        """Return a copy of all registered routes."""
        return list(self._routes)

    def remove_route(self, method: HTTPMethod, path: str) -> bool:
        """Remove a route by method and path. Returns True if removed."""
        for i, route in enumerate(self._routes):
            if route.method == method and route.path == path:
                removed = self._routes.pop(i)
                if removed.name and removed.name in self._named_routes:
                    del self._named_routes[removed.name]
                key = (path, method)
                if key in self._compiled_cache:
                    del self._compiled_cache[key]
                return True
        return False

    def routes_count(self) -> int:
        """Return the number of registered routes."""
        return len(self._routes)
