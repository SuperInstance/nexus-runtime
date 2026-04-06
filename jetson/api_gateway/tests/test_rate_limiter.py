"""Tests for rate_limiter.py — token bucket rate limiting."""

import time

import pytest

from jetson.api_gateway.rate_limiter import RateLimitConfig, RateLimitResult, RateLimiter


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def limiter():
    return RateLimiter(RateLimitConfig(requests_per_window=5, window_seconds=1.0, burst_size=2))


@pytest.fixture
def strict_limiter():
    return RateLimiter(RateLimitConfig(requests_per_window=2, window_seconds=1.0, burst_size=0))


# ── RateLimitConfig ──────────────────────────────────────────────────

class TestRateLimitConfig:
    def test_defaults(self):
        cfg = RateLimitConfig()
        assert cfg.requests_per_window == 100
        assert cfg.window_seconds == 60.0
        assert cfg.burst_size == 10

    def test_custom(self):
        cfg = RateLimitConfig(requests_per_window=50, window_seconds=30.0, burst_size=5)
        assert cfg.requests_per_window == 50
        assert cfg.window_seconds == 30.0
        assert cfg.burst_size == 5


# ── RateLimitResult ──────────────────────────────────────────────────

class TestRateLimitResult:
    def test_allowed(self):
        result = RateLimitResult(allowed=True, remaining=9, limit=10, reset_at=100.0)
        assert result.allowed is True
        assert result.remaining == 9
        assert result.retry_after is None

    def test_rejected(self):
        result = RateLimitResult(allowed=False, remaining=0, retry_after=2.5, limit=10)
        assert result.allowed is False
        assert result.retry_after == 2.5

    def test_defaults(self):
        result = RateLimitResult(allowed=True, remaining=0)
        assert result.limit == 100
        assert result.reset_at == 0.0


# ── RateLimiter: check ───────────────────────────────────────────────

class TestRateLimiterCheck:
    def test_initial_allow(self, limiter):
        result = limiter.check("client1")
        assert result.allowed is True
        assert result.remaining >= 0

    def test_burst_capacity(self, limiter):
        """Burst size 2 + requests_per_window 5 = 7 max tokens."""
        allowed_count = 0
        for _ in range(20):
            result = limiter.check("client2")
            if result.allowed:
                allowed_count += 1
        assert allowed_count == 7  # burst + window

    def test_strict_limit(self, strict_limiter):
        """With burst=0 and requests_per_window=2, only 2 allowed."""
        allowed = 0
        for _ in range(10):
            result = strict_limiter.check("strict_client")
            if result.allowed:
                allowed += 1
        assert allowed == 2

    def test_rejected_result(self, strict_limiter):
        strict_limiter.check("cli")
        strict_limiter.check("cli")
        result = strict_limiter.check("cli")
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after is not None
        assert result.retry_after > 0

    def test_different_identities_independent(self, strict_limiter):
        r1 = strict_limiter.check("a")
        r2 = strict_limiter.check("b")
        assert r1.allowed is True
        assert r2.allowed is True

    def test_result_has_limit(self, limiter):
        result = limiter.check("x")
        assert result.limit == 5

    def test_result_has_reset_at(self, limiter):
        result = limiter.check("x")
        assert result.reset_at > 0

    def test_custom_config_per_check(self, limiter):
        """Pass a custom config to a single check. burst=1 + requests_per_window=2 = 3."""
        cfg = RateLimitConfig(requests_per_window=2, window_seconds=1.0, burst_size=1)
        r1 = limiter.check("custom", config=cfg)
        r2 = limiter.check("custom", config=cfg)
        r3 = limiter.check("custom", config=cfg)
        r4 = limiter.check("custom", config=cfg)
        assert r1.allowed is True
        assert r2.allowed is True
        assert r3.allowed is True
        assert r4.allowed is False


# ── RateLimiter: reset ───────────────────────────────────────────────

class TestRateLimiterReset:
    def test_reset_existing(self, strict_limiter):
        strict_limiter.check("cli")
        assert strict_limiter.reset("cli") is True
        result = strict_limiter.check("cli")
        assert result.allowed is True

    def test_reset_nonexistent(self, strict_limiter):
        assert strict_limiter.reset("nope") is False

    def test_reset_clears_identity_config(self, limiter):
        limiter.set_identity_config("special", RateLimitConfig(requests_per_window=1, window_seconds=1.0, burst_size=0))
        limiter.check("special")
        limiter.reset("special")
        # After reset, should use default config
        result = limiter.check("special")
        assert result.limit == 5  # default


# ── RateLimiter: get_usage ───────────────────────────────────────────

class TestRateLimiterUsage:
    def test_usage_known_identity(self, limiter):
        limiter.check("cli")
        limiter.check("cli")
        usage = limiter.get_usage("cli")
        assert usage is not None
        assert usage["total_requests"] == 2
        assert usage["identity"] == "cli"
        assert "tokens_remaining" in usage

    def test_usage_unknown_identity(self, limiter):
        assert limiter.get_usage("unknown") is None

    def test_usage_after_check(self, limiter):
        """get_usage creates bucket on first check, then returns data."""
        limiter.check("new_cli")
        usage = limiter.get_usage("new_cli")
        assert usage is not None
        # burst (2) + requests_per_window (5) = 7
        assert usage["max_tokens"] == 7
        assert usage["total_requests"] == 1

    def test_usage_tokens_decrease(self, limiter):
        limiter.check("cli")
        usage_after_1 = limiter.get_usage("cli")
        limiter.check("cli")
        usage_after_2 = limiter.get_usage("cli")
        assert usage_after_2["tokens_remaining"] < usage_after_1["tokens_remaining"]


# ── RateLimiter: update_config ───────────────────────────────────────

class TestRateLimiterUpdateConfig:
    def test_update_default_config(self, limiter):
        new_cfg = RateLimitConfig(requests_per_window=100, window_seconds=60.0, burst_size=50)
        limiter.update_config(new_cfg)
        stats = limiter.get_stats()
        assert stats["config"]["requests_per_window"] == 100
        assert stats["config"]["burst_size"] == 50


# ── RateLimiter: set_identity_config ─────────────────────────────────

class TestRateLimiterIdentityConfig:
    def test_per_identity_config(self, limiter):
        """Per-identity config: burst=1 + requests_per_window=2 = 3 tokens."""
        cfg = RateLimitConfig(requests_per_window=2, window_seconds=1.0, burst_size=1)
        limiter.set_identity_config("vip", cfg)
        r1 = limiter.check("vip")
        r2 = limiter.check("vip")
        r3 = limiter.check("vip")
        r4 = limiter.check("vip")
        assert r1.allowed is True
        assert r2.allowed is True
        assert r3.allowed is True
        assert r4.allowed is False

    def test_default_config_for_others(self, limiter):
        cfg = RateLimitConfig(requests_per_window=1, window_seconds=1.0, burst_size=0)
        limiter.set_identity_config("vip", cfg)
        r = limiter.check("normal")
        assert r.limit == 5  # default


# ── RateLimiter: get_stats ───────────────────────────────────────────

class TestRateLimiterStats:
    def test_stats_structure(self, limiter):
        stats = limiter.get_stats()
        assert "total_checks" in stats
        assert "total_allowed" in stats
        assert "total_rejected" in stats
        assert "active_identities" in stats
        assert "config" in stats

    def test_stats_counts(self, strict_limiter):
        strict_limiter.check("a")
        strict_limiter.check("a")
        strict_limiter.check("a")  # rejected
        stats = strict_limiter.get_stats()
        assert stats["total_checks"] == 3
        assert stats["total_allowed"] == 2
        assert stats["total_rejected"] == 1

    def test_active_identities(self, limiter):
        limiter.check("a")
        limiter.check("b")
        stats = limiter.get_stats()
        assert stats["active_identities"] == 2

    def test_stats_config(self, limiter):
        stats = limiter.get_stats()
        assert stats["config"]["requests_per_window"] == 5
        assert stats["config"]["window_seconds"] == 1.0
        assert stats["config"]["burst_size"] == 2


# ── RateLimiter: get_all_usage / prune_inactive ─────────────────────

class TestRateLimiterAdvanced:
    def test_get_all_usage(self, limiter):
        limiter.check("a")
        limiter.check("b")
        all_usage = limiter.get_all_usage()
        assert "a" in all_usage
        assert "b" in all_usage

    def test_prune_inactive(self, limiter):
        limiter.check("old_client")
        # Manually set last_refill to the past
        limiter._buckets["old_client"].last_refill = time.time() - 7200
        removed = limiter.prune_inactive(max_age_seconds=3600)
        assert removed == 1
        assert limiter.get_usage("old_client") is None

    def test_prune_keeps_active(self, limiter):
        limiter.check("active_client")
        removed = limiter.prune_inactive(max_age_seconds=3600)
        assert removed == 0
        assert limiter.get_usage("active_client") is not None

    def test_refill_over_time(self, limiter):
        """Consuming all tokens, then waiting should refill some."""
        # Drain tokens
        for _ in range(20):
            limiter.check("refill_test")

        # Should be depleted
        result = limiter.check("refill_test")
        assert result.allowed is False

        # Manually simulate time passing by adjusting last_refill
        state = limiter._buckets["refill_test"]
        state.last_refill -= 1.0  # pretend 1 second passed
        state.tokens = 0.0  # fully depleted

        result = limiter.check("refill_test")
        assert result.allowed is True
