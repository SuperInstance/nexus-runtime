"""Request/response middleware chain with ordered processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .router import Request, Response


@dataclass
class MiddlewareContext:
    """Context passed through the middleware chain."""
    request: Request
    response: Optional[Response] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    proceed: bool = True


class MiddlewareChain:
    """Ordered middleware processing pipeline.

    Each middleware is a callable: (MiddlewareContext) -> MiddlewareContext.
    If a middleware sets ``context.proceed = False``, subsequent middleware
    is skipped and the current response is returned.
    """

    def __init__(self) -> None:
        self._middleware: List[Dict[str, Any]] = []

    def add_middleware(
        self,
        middleware_fn: Callable,
        name: Optional[str] = None,
        priority: int = 0,
    ) -> int:
        """Add a middleware function. Returns its index.

        Middleware is inserted sorted by priority (higher priority first).
        """
        entry = {
            "fn": middleware_fn,
            "name": name or f"middleware_{len(self._middleware)}",
            "priority": priority,
        }

        # Insert sorted by priority descending
        inserted = False
        for i, existing in enumerate(self._middleware):
            if priority > existing["priority"]:
                self._middleware.insert(i, entry)
                return i

        self._middleware.append(entry)
        return len(self._middleware) - 1

    def remove_middleware(self, name: str) -> bool:
        """Remove middleware by name. Returns True if removed."""
        for i, entry in enumerate(self._middleware):
            if entry["name"] == name:
                self._middleware.pop(i)
                return True
        return False

    def insert_middleware(
        self,
        index: int,
        middleware_fn: Callable,
        name: Optional[str] = None,
        priority: int = 0,
    ) -> int:
        """Insert middleware at a specific index. Returns the actual index."""
        entry = {
            "fn": middleware_fn,
            "name": name or f"middleware_{len(self._middleware)}",
            "priority": priority,
        }
        # Clamp index to valid range
        index = max(0, min(index, len(self._middleware)))
        self._middleware.insert(index, entry)
        return index

    def process(self, request: Request) -> Response:
        """Run the middleware chain against a request.

        Returns a Response. If no middleware produces a response,
        a default 200 response is returned.
        """
        context = MiddlewareContext(request=request)

        for entry in self._middleware:
            if not context.proceed:
                break
            result = entry["fn"](context)
            if result is not None:
                # If middleware returns a context, use it
                if isinstance(result, MiddlewareContext):
                    context = result
                elif isinstance(result, Response):
                    context.response = result

        if context.response is None:
            context.response = Response(status_code=200, body="OK")

        return context.response

    def get_middleware_list(self) -> List[Dict[str, Any]]:
        """Return a copy of the middleware entries (without callable references)."""
        return [
            {"name": e["name"], "priority": e["priority"]}
            for e in self._middleware
        ]

    def count(self) -> int:
        """Return the number of registered middleware."""
        return len(self._middleware)

    def clear(self) -> None:
        """Remove all middleware."""
        self._middleware.clear()
