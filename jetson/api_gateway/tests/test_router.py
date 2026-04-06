"""Tests for router.py — URL routing, path parameters, named routes."""

import pytest

from jetson.api_gateway.router import HTTPMethod, Route, Request, Response, Router


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def router():
    return Router()


def _handler():
    return Response(body="ok")


def _user_handler():
    return Response(body="user")


def _post_handler():
    return Response(body="post")


# ── HTTPMethod ────────────────────────────────────────────────────────

class TestHTTPMethod:
    def test_values(self):
        assert HTTPMethod.GET.value == "GET"
        assert HTTPMethod.POST.value == "POST"
        assert HTTPMethod.PUT.value == "PUT"
        assert HTTPMethod.DELETE.value == "DELETE"
        assert HTTPMethod.PATCH.value == "PATCH"
        assert HTTPMethod.OPTIONS.value == "OPTIONS"

    def test_all_six_methods(self):
        assert len(HTTPMethod) == 6

    def test_members_are_unique(self):
        values = [m.value for m in HTTPMethod]
        assert len(values) == len(set(values))


# ── Route ─────────────────────────────────────────────────────────────

class TestRoute:
    def test_creation(self):
        r = Route(path="/test", method=HTTPMethod.GET, handler=_handler)
        assert r.path == "/test"
        assert r.method == HTTPMethod.GET
        assert r.handler is _handler
        assert r.name == ""
        assert r.middleware == []

    def test_frozen(self):
        r = Route(path="/test", method=HTTPMethod.GET, handler=_handler)
        with pytest.raises(AttributeError):
            r.path = "/changed"

    def test_name_and_middleware(self):
        r = Route(path="/a", method=HTTPMethod.POST, handler=_handler, name="create_a", middleware=["mw1"])
        assert r.name == "create_a"
        assert r.middleware == ["mw1"]


# ── Request ───────────────────────────────────────────────────────────

class TestRequest:
    def test_defaults(self):
        req = Request(method=HTTPMethod.GET, path="/")
        assert req.method == HTTPMethod.GET
        assert req.path == "/"
        assert req.headers == {}
        assert req.query_params == {}
        assert req.body is None
        assert req.path_params == {}

    def test_with_all_fields(self):
        req = Request(
            method=HTTPMethod.POST,
            path="/users",
            headers={"Content-Type": "application/json"},
            query_params={"page": "1"},
            body='{"name":"test"}',
            path_params={"id": "42"},
        )
        assert req.method == HTTPMethod.POST
        assert req.path == "/users"
        assert req.headers["Content-Type"] == "application/json"
        assert req.query_params["page"] == "1"
        assert req.body == '{"name":"test"}'
        assert req.path_params["id"] == "42"


# ── Response ──────────────────────────────────────────────────────────

class TestResponse:
    def test_defaults(self):
        resp = Response()
        assert resp.status_code == 200
        assert resp.headers == {}
        assert resp.body is None
        assert resp.content_type == "application/json"

    def test_custom(self):
        resp = Response(status_code=201, body={"created": True}, content_type="text/plain")
        assert resp.status_code == 201
        assert resp.body == {"created": True}
        assert resp.content_type == "text/plain"


# ── Router: add_route ────────────────────────────────────────────────

class TestRouterAddRoute:
    def test_add_single_route(self, router):
        route = router.add_route(HTTPMethod.GET, "/", _handler)
        assert isinstance(route, Route)
        assert route.path == "/"
        assert router.routes_count() == 1

    def test_add_named_route(self, router):
        router.add_route(HTTPMethod.GET, "/users", _handler, name="list_users")
        routes = router.list_routes()
        assert routes[0].name == "list_users"

    def test_add_route_with_middleware(self, router):
        mw = ["auth", "logging"]
        router.add_route(HTTPMethod.GET, "/", _handler, middleware=mw)
        routes = router.list_routes()
        assert routes[0].middleware == mw

    def test_add_multiple_routes(self, router):
        router.add_route(HTTPMethod.GET, "/a", _handler)
        router.add_route(HTTPMethod.POST, "/b", _handler)
        router.add_route(HTTPMethod.PUT, "/c", _handler)
        assert router.routes_count() == 3

    def test_add_same_path_different_methods(self, router):
        router.add_route(HTTPMethod.GET, "/users", _handler)
        router.add_route(HTTPMethod.POST, "/users", _handler)
        assert router.routes_count() == 2


# ── Router: match ────────────────────────────────────────────────────

class TestRouterMatch:
    def test_match_exact(self, router):
        router.add_route(HTTPMethod.GET, "/health", _handler)
        req = Request(method=HTTPMethod.GET, path="/health")
        result = router.match(req)
        assert result is not None
        handler, params = result
        assert handler is _handler
        assert params == {}

    def test_match_no_result(self, router):
        router.add_route(HTTPMethod.GET, "/health", _handler)
        req = Request(method=HTTPMethod.GET, path="/missing")
        assert router.match(req) is None

    def test_match_wrong_method(self, router):
        router.add_route(HTTPMethod.GET, "/health", _handler)
        req = Request(method=HTTPMethod.POST, path="/health")
        assert router.match(req) is None

    def test_match_path_param(self, router):
        router.add_route(HTTPMethod.GET, "/users/{id}", _user_handler)
        req = Request(method=HTTPMethod.GET, path="/users/42")
        result = router.match(req)
        assert result is not None
        handler, params = result
        assert handler is _user_handler
        assert params == {"id": "42"}

    def test_match_multiple_path_params(self, router):
        router.add_route(HTTPMethod.GET, "/users/{user_id}/posts/{post_id}", _post_handler)
        req = Request(method=HTTPMethod.GET, path="/users/5/posts/99")
        result = router.match(req)
        assert result is not None
        _, params = result
        assert params == {"user_id": "5", "post_id": "99"}

    def test_match_typed_param_int(self, router):
        router.add_route(HTTPMethod.GET, "/items/{id:int}", _handler)
        req = Request(method=HTTPMethod.GET, path="/items/123")
        result = router.match(req)
        assert result is not None
        _, params = result
        assert params == {"id": "123"}

    def test_match_typed_param_int_rejects_non_numeric(self, router):
        router.add_route(HTTPMethod.GET, "/items/{id:int}", _handler)
        req = Request(method=HTTPMethod.GET, path="/items/abc")
        assert router.match(req) is None

    def test_match_first_wins(self, router):
        def h1(): pass
        def h2(): pass
        router.add_route(HTTPMethod.GET, "/static", h1)
        router.add_route(HTTPMethod.GET, "/static", h2)
        req = Request(method=HTTPMethod.GET, path="/static")
        result = router.match(req)
        assert result is not None
        assert result[0] is h1

    def test_match_trailing_slash(self, router):
        router.add_route(HTTPMethod.GET, "/api/", _handler)
        req = Request(method=HTTPMethod.GET, path="/api/")
        assert router.match(req) is not None

    def test_match_no_trailing_slash(self, router):
        router.add_route(HTTPMethod.GET, "/api/", _handler)
        req = Request(method=HTTPMethod.GET, path="/api")
        assert router.match(req) is None


# ── Router: match_any ────────────────────────────────────────────────

class TestRouterMatchAny:
    def test_match_any_returns_all(self, router):
        def h1(): pass
        def h2(): pass
        router.add_route(HTTPMethod.GET, "/items/{id}", h1)
        router.add_route(HTTPMethod.GET, "/items/{id:int}", h2)
        req = Request(method=HTTPMethod.GET, path="/items/42")
        results = router.match_any(req)
        assert len(results) == 2

    def test_match_any_empty(self, router):
        req = Request(method=HTTPMethod.GET, path="/nothing")
        assert router.match_any(req) == []


# ── Router: build_url ────────────────────────────────────────────────

class TestRouterBuildURL:
    def test_build_simple(self, router):
        router.add_route(HTTPMethod.GET, "/health", _handler, name="health")
        url = router.build_url("health")
        assert url == "/health"

    def test_build_with_params(self, router):
        router.add_route(HTTPMethod.GET, "/users/{id}", _handler, name="user")
        url = router.build_url("user", {"id": "42"})
        assert url == "/users/42"

    def test_build_multiple_params(self, router):
        router.add_route(
            HTTPMethod.GET, "/users/{uid}/posts/{pid}", _handler, name="user_post"
        )
        url = router.build_url("user_post", {"uid": "1", "pid": "5"})
        assert url == "/users/1/posts/5"

    def test_build_missing_name_raises(self, router):
        with pytest.raises(KeyError):
            router.build_url("nonexistent")

    def test_build_typed_param(self, router):
        router.add_route(HTTPMethod.GET, "/items/{id:int}", _handler, name="item")
        url = router.build_url("item", {"id": "99"})
        assert url == "/items/99"

    def test_build_no_params_needed(self, router):
        router.add_route(HTTPMethod.GET, "/status", _handler, name="status")
        url = router.build_url("status", {})
        assert url == "/status"


# ── Router: list_routes / routes_count ───────────────────────────────

class TestRouterList:
    def test_list_routes_returns_copy(self, router):
        router.add_route(HTTPMethod.GET, "/a", _handler)
        routes = router.list_routes()
        routes.clear()
        assert router.routes_count() == 1

    def test_routes_count(self, router):
        assert router.routes_count() == 0
        router.add_route(HTTPMethod.GET, "/a", _handler)
        assert router.routes_count() == 1
        router.add_route(HTTPMethod.POST, "/b", _handler)
        assert router.routes_count() == 2


# ── Router: remove_route ─────────────────────────────────────────────

class TestRouterRemoveRoute:
    def test_remove_existing(self, router):
        router.add_route(HTTPMethod.GET, "/to-remove", _handler, name="remove_me")
        assert router.routes_count() == 1
        assert router.remove_route(HTTPMethod.GET, "/to-remove") is True
        assert router.routes_count() == 0

    def test_remove_nonexistent(self, router):
        assert router.remove_route(HTTPMethod.GET, "/nope") is False

    def test_remove_only_matching_method(self, router):
        router.add_route(HTTPMethod.GET, "/users", _handler)
        router.add_route(HTTPMethod.POST, "/users", _handler)
        router.remove_route(HTTPMethod.GET, "/users")
        assert router.routes_count() == 1
        routes = router.list_routes()
        assert routes[0].method == HTTPMethod.POST

    def test_remove_named_route(self, router):
        router.add_route(HTTPMethod.GET, "/x", _handler, name="named_x")
        router.remove_route(HTTPMethod.GET, "/x")
        with pytest.raises(KeyError):
            router.build_url("named_x")
