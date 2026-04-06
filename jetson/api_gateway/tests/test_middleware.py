"""Tests for middleware.py — middleware chain processing."""

import pytest

from jetson.api_gateway.middleware import MiddlewareContext, MiddlewareChain
from jetson.api_gateway.router import Request, Response, HTTPMethod


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def chain():
    return MiddlewareChain()


@pytest.fixture
def http_request():
    return Request(method=HTTPMethod.GET, path="/test")


# ── MiddlewareContext ─────────────────────────────────────────────────

class TestMiddlewareContext:
    def test_defaults(self):
        req = Request(method=HTTPMethod.GET, path="/")
        ctx = MiddlewareContext(request=req)
        assert ctx.request is req
        assert ctx.response is None
        assert ctx.metadata == {}
        assert ctx.proceed is True

    def test_with_response(self):
        req = Request(method=HTTPMethod.GET, path="/")
        resp = Response(status_code=201)
        ctx = MiddlewareContext(request=req, response=resp)
        assert ctx.response is resp

    def test_with_metadata(self):
        req = Request(method=HTTPMethod.GET, path="/")
        ctx = MiddlewareContext(request=req, metadata={"key": "value"})
        assert ctx.metadata["key"] == "value"

    def test_proceed_false(self):
        req = Request(method=HTTPMethod.GET, path="/")
        ctx = MiddlewareContext(request=req, proceed=False)
        assert ctx.proceed is False


# ── MiddlewareChain: add_middleware ───────────────────────────────────

class TestAddMiddleware:
    def test_add_single(self, chain):
        idx = chain.add_middleware(lambda ctx: ctx)
        assert chain.count() == 1
        assert idx == 0

    def test_add_named(self, chain):
        chain.add_middleware(lambda ctx: ctx, name="auth")
        entries = chain.get_middleware_list()
        assert entries[0]["name"] == "auth"

    def test_add_with_priority(self, chain):
        chain.add_middleware(lambda ctx: ctx, name="low", priority=1)
        chain.add_middleware(lambda ctx: ctx, name="high", priority=10)
        entries = chain.get_middleware_list()
        assert entries[0]["name"] == "high"
        assert entries[1]["name"] == "low"

    def test_auto_name(self, chain):
        chain.add_middleware(lambda ctx: ctx)
        entries = chain.get_middleware_list()
        assert entries[0]["name"] == "middleware_0"

    def test_add_multiple(self, chain):
        chain.add_middleware(lambda ctx: ctx, name="m1")
        chain.add_middleware(lambda ctx: ctx, name="m2")
        chain.add_middleware(lambda ctx: ctx, name="m3")
        assert chain.count() == 3

    def test_priority_ordering(self, chain):
        chain.add_middleware(lambda ctx: ctx, name="p5", priority=5)
        chain.add_middleware(lambda ctx: ctx, name="p10", priority=10)
        chain.add_middleware(lambda ctx: ctx, name="p3", priority=3)
        chain.add_middleware(lambda ctx: ctx, name="p7", priority=7)
        entries = chain.get_middleware_list()
        names = [e["name"] for e in entries]
        assert names == ["p10", "p7", "p5", "p3"]


# ── MiddlewareChain: remove_middleware ────────────────────────────────

class TestRemoveMiddleware:
    def test_remove_existing(self, chain):
        chain.add_middleware(lambda ctx: ctx, name="to_remove")
        assert chain.remove_middleware("to_remove") is True
        assert chain.count() == 0

    def test_remove_nonexistent(self, chain):
        assert chain.remove_middleware("nope") is False

    def test_remove_middle(self, chain):
        chain.add_middleware(lambda ctx: ctx, name="a", priority=3)
        chain.add_middleware(lambda ctx: ctx, name="b", priority=2)
        chain.add_middleware(lambda ctx: ctx, name="c", priority=1)
        chain.remove_middleware("b")
        entries = chain.get_middleware_list()
        names = [e["name"] for e in entries]
        assert names == ["a", "c"]


# ── MiddlewareChain: insert_middleware ────────────────────────────────

class TestInsertMiddleware:
    def test_insert_at_index(self, chain):
        chain.add_middleware(lambda ctx: ctx, name="first")
        idx = chain.insert_middleware(0, lambda ctx: ctx, name="inserted")
        assert idx == 0
        entries = chain.get_middleware_list()
        assert entries[0]["name"] == "inserted"

    def test_insert_at_end(self, chain):
        chain.add_middleware(lambda ctx: ctx, name="first")
        chain.insert_middleware(10, lambda ctx: ctx, name="last")
        entries = chain.get_middleware_list()
        assert entries[-1]["name"] == "last"

    def test_insert_negative_index_clamped(self, chain):
        chain.add_middleware(lambda ctx: ctx, name="first")
        chain.insert_middleware(-5, lambda ctx: ctx, name="clamped")
        entries = chain.get_middleware_list()
        assert entries[0]["name"] == "clamped"

    def test_insert_with_name_and_priority(self, chain):
        chain.insert_middleware(0, lambda ctx: ctx, name="special", priority=99)
        entries = chain.get_middleware_list()
        assert entries[0]["name"] == "special"
        assert entries[0]["priority"] == 99


# ── MiddlewareChain: process ──────────────────────────────────────────

class TestProcess:
    def test_empty_chain_returns_default(self, chain, http_request):
        resp = chain.process(http_request)
        assert resp.status_code == 200
        assert resp.body == "OK"

    def test_middleware_sets_response(self, chain, http_request):
        def set_response(ctx):
            ctx.response = Response(status_code=403, body="Forbidden")
            return ctx

        chain.add_middleware(set_response)
        resp = chain.process(http_request)
        assert resp.status_code == 403
        assert resp.body == "Forbidden"

    def test_middleware_returns_response(self, chain, http_request):
        def return_response(ctx):
            return Response(status_code=201, body="Created")

        chain.add_middleware(return_response)
        resp = chain.process(http_request)
        assert resp.status_code == 201

    def test_middleware_modifies_request(self, chain, http_request):
        def add_header(ctx):
            ctx.request.headers["X-Custom"] = "value"

        chain.add_middleware(add_header)
        chain.process(http_request)
        assert http_request.headers["X-Custom"] == "value"

    def test_proceed_false_stops_chain(self, chain, http_request):
        order = []

        def first(ctx):
            order.append("first")
            ctx.response = Response(status_code=401, body="stop")
            ctx.proceed = False

        def second(ctx):
            order.append("second")

        chain.add_middleware(first, priority=10)
        chain.add_middleware(second, priority=5)
        chain.process(http_request)

        assert order == ["first"]

    def test_all_middleware_run(self, chain, http_request):
        order = []

        def m1(ctx):
            order.append("m1")

        def m2(ctx):
            order.append("m2")

        chain.add_middleware(m1, name="m1")
        chain.add_middleware(m2, name="m2")
        chain.process(http_request)

        assert order == ["m1", "m2"]

    def test_middleware_sets_metadata(self, chain, http_request):
        def add_meta(ctx):
            ctx.metadata["start_time"] = 12345

        chain.add_middleware(add_meta)
        chain.process(http_request)

    def test_late_middleware_overwrites_response(self, chain, http_request):
        """Higher priority runs first, so the lower-priority middleware runs last and wins."""
        def set_200(ctx):
            ctx.response = Response(status_code=200)

        def set_500(ctx):
            ctx.response = Response(status_code=500)

        chain.add_middleware(set_200, priority=10)  # runs first
        chain.add_middleware(set_500, priority=5)   # runs second, overwrites
        resp = chain.process(http_request)
        assert resp.status_code == 500

    def test_process_returns_last_response(self, chain, http_request):
        """If middleware only modifies request, default response is returned."""
        def noop(ctx):
            pass

        chain.add_middleware(noop)
        resp = chain.process(http_request)
        assert resp.status_code == 200


# ── MiddlewareChain: get_middleware_list / count / clear ──────────────

class TestChainUtilities:
    def test_get_middleware_list_no_callables(self, chain):
        chain.add_middleware(lambda ctx: ctx, name="test")
        entries = chain.get_middleware_list()
        assert "fn" not in entries[0]
        assert entries[0]["name"] == "test"

    def test_count(self, chain):
        assert chain.count() == 0
        chain.add_middleware(lambda ctx: ctx)
        assert chain.count() == 1
        chain.add_middleware(lambda ctx: ctx)
        assert chain.count() == 2

    def test_clear(self, chain):
        chain.add_middleware(lambda ctx: ctx)
        chain.add_middleware(lambda ctx: ctx)
        chain.clear()
        assert chain.count() == 0

    def test_clear_empty(self, chain):
        chain.clear()
        assert chain.count() == 0
