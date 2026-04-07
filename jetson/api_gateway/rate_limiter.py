"""Token-bucket rate limiter with configurable windows, burst, and per-identity tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_window: int = 100
    window_seconds: float = 60.0
    burst_size: int = 10


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    retry_after: Optional[float] = None
    limit: int = 100
    reset_at: float = 0.0


@dataclass
class _BucketState:
    """Internal token bucket state."""
    tokens: float
    last_refill: float
    total_requests: int = 0
    rejected_requests: int = 0


class RateLimiter:
    """Token-bucket rate limiter.

    Each identity (e.g. API key, IP address) gets its own bucket.
    Tokens refill at ``requests_per_window / window_seconds`` per second,
    up to ``burst_size + requests_per_window`` maximum.
    """

    _PRUNE_INTERVAL = 1000  # auto-prune every N checks
    _PRUNE_MAX_AGE = 3600.0  # prune buckets idle for 1 hour

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        self._config = config or RateLimitConfig()
        self._buckets: Dict[str, _BucketState] = {}
        self._identity_configs: Dict[str, RateLimitConfig] = {}
        self._total_checks: int = 0
        self._total_allowed: int = 0
        self._total_rejected: int = 0

    def _refill(self, state: _BucketState, config: RateLimitConfig, now: float) -> None:
        """Refill tokens based on elapsed time."""
        elapsed = now - state.last_refill
        if elapsed <= 0:
            return

        refill_rate = config.requests_per_window / config.window_seconds
        max_tokens = config.burst_size + config.requests_per_window

        state.tokens = min(max_tokens, state.tokens + elapsed * refill_rate)
        state.last_refill = now

    def _get_config(self, identity: str) -> RateLimitConfig:
        """Get config for a specific identity, falling back to default."""
        return self._identity_configs.get(identity, self._config)

    def check(self, identity: str, config: Optional[RateLimitConfig] = None) -> RateLimitResult:
        """Check if a request from ``identity`` is allowed.

        Returns a ``RateLimitResult`` with all rate limit metadata.
        """
        self._total_checks += 1
        if self._total_checks % self._PRUNE_INTERVAL == 0:
            self.prune_inactive(self._PRUNE_MAX_AGE)
        cfg = config or self._get_config(identity)
        now = time.time()
        max_tokens = cfg.burst_size + cfg.requests_per_window

        if identity not in self._buckets:
            self._buckets[identity] = _BucketState(
                tokens=float(max_tokens),
                last_refill=now,
            )

        state = self._buckets[identity]
        self._refill(state, cfg, now)
        state.total_requests += 1

        if state.tokens >= 1.0:
            state.tokens -= 1.0
            self._total_allowed += 1
            reset_at = state.last_refill + cfg.window_seconds
            return RateLimitResult(
                allowed=True,
                remaining=int(state.tokens),
                retry_after=None,
                limit=cfg.requests_per_window,
                reset_at=reset_at,
            )
        else:
            self._total_rejected += 1
            deficit = 1.0 - state.tokens
            refill_rate = cfg.requests_per_window / cfg.window_seconds
            retry_after = deficit / refill_rate if refill_rate > 0 else 0.0
            reset_at = state.last_refill + cfg.window_seconds
            return RateLimitResult(
                allowed=False,
                remaining=0,
                retry_after=retry_after,
                limit=cfg.requests_per_window,
                reset_at=reset_at,
            )

    def reset(self, identity: str) -> bool:
        """Reset the rate limit bucket for an identity. Returns True if it existed."""
        if identity in self._buckets:
            del self._buckets[identity]
            self._identity_configs.pop(identity, None)
            return True
        return False

    def get_usage(self, identity: str) -> Optional[Dict[str, any]]:
        """Get current usage stats for an identity. Returns None if unknown."""
        state = self._buckets.get(identity)
        if state is None:
            return None

        cfg = self._get_config(identity)
        max_tokens = cfg.burst_size + cfg.requests_per_window
        now = time.time()
        self._refill(state, cfg, now)

        return {
            "identity": identity,
            "tokens_remaining": state.tokens,
            "max_tokens": max_tokens,
            "total_requests": state.total_requests,
            "rejected_requests": state.rejected_requests,
            "last_refill": state.last_refill,
        }

    def update_config(self, config: RateLimitConfig) -> None:
        """Update the default rate limit configuration."""
        self._config = config

    def set_identity_config(self, identity: str, config: RateLimitConfig) -> None:
        """Set a per-identity rate limit configuration."""
        self._identity_configs[identity] = config

    def get_stats(self) -> Dict[str, any]:
        """Get aggregate rate limiter statistics."""
        return {
            "total_checks": self._total_checks,
            "total_allowed": self._total_allowed,
            "total_rejected": self._total_rejected,
            "active_identities": len(self._buckets),
            "config": {
                "requests_per_window": self._config.requests_per_window,
                "window_seconds": self._config.window_seconds,
                "burst_size": self._config.burst_size,
            },
        }

    def get_all_usage(self) -> Dict[str, Dict[str, any]]:
        """Get usage stats for all identities."""
        result = {}
        for identity in self._buckets:
            usage = self.get_usage(identity)
            if usage is not None:
                result[identity] = usage
        return result

    def prune_inactive(self, max_age_seconds: float = 3600.0) -> int:
        """Remove buckets that haven't been used recently. Returns count removed."""
        now = time.time()
        to_remove = []
        for identity, state in self._buckets.items():
            if now - state.last_refill > max_age_seconds:
                to_remove.append(identity)

        for identity in to_remove:
            del self._buckets[identity]
            self._identity_configs.pop(identity, None)

        return len(to_remove)
