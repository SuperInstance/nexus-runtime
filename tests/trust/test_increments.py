"""Tests for the NEXUS Trust Increments formulas (15+ tests)."""

import pytest
import math

from nexus.trust.increments import (
    TrustDimensions, HistoryTracker,
    compute_history, compute_capability, compute_latency,
    compute_consistency, compute_composite,
)
from nexus.trust.engine import TrustWeights


class TestHistoryTracker:
    def test_initial_state(self):
        h = HistoryTracker()
        assert h.count == 0
        assert h.get_ema() == 0.5

    def test_record_success(self):
        h = HistoryTracker(alpha=1.0)
        h.record(True)
        assert h.get_ema() == 1.0

    def test_record_failure(self):
        h = HistoryTracker(alpha=1.0)
        h.record(False)
        assert h.get_ema() == 0.0

    def test_ema_smoothing(self):
        h = HistoryTracker(alpha=0.3)
        h.record(True)
        val1 = h.get_ema()
        h.record(True)
        val2 = h.get_ema()
        assert val2 >= val1

    def test_multiple_records(self):
        h = HistoryTracker(alpha=0.5)
        for _ in range(100):
            h.record(True)
        assert h.get_ema() > 0.9
        assert h.count == 100

    def test_get_mean(self):
        h = HistoryTracker()
        h.record(True)
        h.record(True)
        h.record(False)
        assert h.get_mean() == pytest.approx(2.0 / 3.0)

    def test_get_variance(self):
        h = HistoryTracker()
        h.record(True)
        h.record(False)
        var = h.get_variance()
        assert var > 0

    def test_get_stddev(self):
        h = HistoryTracker()
        h.record(True)
        h.record(False)
        sd = h.get_stddev()
        assert sd > 0

    def test_latency_tracking(self):
        h = HistoryTracker()
        h.record(True, latency_ms=50.0)
        h.record(True, latency_ms=100.0)
        stats = h.get_latency_stats()
        assert stats["count"] == 2
        assert stats["mean"] == 75.0

    def test_latency_stats_empty(self):
        h = HistoryTracker()
        stats = h.get_latency_stats()
        assert stats["count"] == 0

    def test_reset(self):
        h = HistoryTracker()
        h.record(True)
        h.record(True)
        h.reset()
        assert h.count == 0

    def test_max_samples(self):
        h = HistoryTracker(max_samples=5)
        for _ in range(10):
            h.record(True)
        # EMA should only be influenced by last 5
        # (EMA doesn't directly use the samples list, but max_samples limits storage)
        assert h.count == 10  # count always increments


class TestComputeHistory:
    def test_compute_history(self):
        h = HistoryTracker(alpha=1.0)
        h.record(True)
        assert compute_history(h) == 1.0


class TestComputeCapability:
    def test_perfect_match(self):
        agent = {"navigation": 0.9, "sensing": 0.8}
        required = {"navigation": 0.9, "sensing": 0.8}
        assert compute_capability(agent, required) == 1.0

    def test_partial_match(self):
        agent = {"navigation": 0.5, "sensing": 0.4}
        required = {"navigation": 0.8, "sensing": 0.8}
        score = compute_capability(agent, required)
        assert 0.0 < score < 1.0

    def test_no_requirements(self):
        assert compute_capability({"nav": 0.5}, {}) == 1.0

    def test_missing_capability(self):
        agent = {}
        required = {"navigation": 0.8}
        assert compute_capability(agent, required) == 0.0


class TestComputeLatency:
    def test_below_target(self):
        assert compute_latency(50.0, target_latency_ms=100.0) == 1.0

    def test_at_target(self):
        assert compute_latency(100.0, target_latency_ms=100.0) == 1.0

    def test_above_max(self):
        assert compute_latency(2000.0, max_latency_ms=1000.0) == 0.0

    def test_between_target_and_max(self):
        score = compute_latency(500.0, target_latency_ms=100.0, max_latency_ms=1000.0)
        assert 0.0 < score < 1.0

    def test_zero_latency(self):
        assert compute_latency(0.0) == 0.5


class TestComputeConsistency:
    def test_perfect_consistency(self):
        samples = [1.0, 1.0, 1.0, 1.0]
        assert compute_consistency(samples) == 1.0

    def test_low_consistency(self):
        samples = [1.0, 0.0, 1.0, 0.0, 1.0]
        score = compute_consistency(samples)
        assert 0.0 <= score <= 1.0

    def test_single_sample(self):
        assert compute_consistency([1.0]) == 0.5

    def test_empty_samples(self):
        assert compute_consistency([]) == 0.5


class TestComputeComposite:
    def test_all_high(self):
        dims = TrustDimensions(history=1.0, capability=1.0, latency=1.0, consistency=1.0)
        assert compute_composite(dims) == 1.0

    def test_all_low(self):
        dims = TrustDimensions(history=0.0, capability=0.0, latency=0.0, consistency=0.0)
        assert compute_composite(dims) == 0.0

    def test_mixed(self):
        dims = TrustDimensions(history=0.8, capability=0.6, latency=0.5, consistency=0.9)
        score = compute_composite(dims)
        assert 0.0 <= score <= 1.0

    def test_custom_weights(self):
        dims = TrustDimensions(history=1.0, capability=0.0, latency=0.0, consistency=0.0)
        w = TrustWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
        assert compute_composite(dims, w) == 1.0

    def test_clamped(self):
        dims = TrustDimensions(history=2.0, capability=-1.0, latency=0.0, consistency=0.0)
        score = compute_composite(dims)
        assert 0.0 <= score <= 1.0
