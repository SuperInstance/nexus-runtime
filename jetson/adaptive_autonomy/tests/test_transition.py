"""Tests for jetson.adaptive_autonomy.transition."""

import time
from unittest.mock import patch

import pytest

from jetson.adaptive_autonomy.levels import AutonomyLevel
from jetson.adaptive_autonomy.transition import (
    TransitionManager,
    TransitionPolicy,
    TransitionRequest,
)


# ── TransitionRequest ─────────────────────────────────────────────

class TestTransitionRequest:
    def test_defaults(self):
        req = TransitionRequest(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
        )
        assert req.reason == ""
        assert req.urgency == "normal"
        assert req.requires_confirmation is False

    def test_custom_fields(self):
        req = TransitionRequest(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.AUTONOMOUS,
            reason="test",
            urgency="critical",
            requires_confirmation=True,
        )
        assert req.reason == "test"
        assert req.urgency == "critical"


# ── TransitionPolicy ──────────────────────────────────────────────

class TestTransitionPolicy:
    def test_default_policy_creation(self):
        policy = TransitionPolicy.default_policy()
        assert isinstance(policy, TransitionPolicy)

    def test_default_policy_allows_all(self):
        policy = TransitionPolicy.default_policy()
        for lv in AutonomyLevel:
            targets = policy.allowed_transitions.get(lv, set())
            for other in AutonomyLevel:
                if other != lv:
                    assert other in targets

    def test_default_cooldown(self):
        policy = TransitionPolicy.default_policy()
        assert policy.cooldown_seconds == 30.0

    def test_default_max_per_hour(self):
        policy = TransitionPolicy.default_policy()
        assert policy.max_transitions_per_hour == 60

    def test_confirmation_set_for_high_to_low(self):
        policy = TransitionPolicy.default_policy()
        assert (AutonomyLevel.FULL_AUTO, AutonomyLevel.MANUAL) in policy.confirmation_required


# ── TransitionManager ─────────────────────────────────────────────

class TestTransitionManager:

    @pytest.fixture()
    def tm(self):
        policy = TransitionPolicy.default_policy()
        policy.cooldown_seconds = 0.0  # speed up tests
        return TransitionManager(policy=policy)

    # current_level
    def test_initial_level_manual(self, tm):
        assert tm.current_level == AutonomyLevel.MANUAL

    # request_transition — approval
    def test_request_approved(self, tm):
        req = TransitionRequest(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
        )
        result = tm.request_transition(req)
        assert result["approved"] is True

    def test_request_denied_not_in_policy(self, tm):
        # Create a restrictive policy
        policy = TransitionPolicy(allowed_transitions={
            AutonomyLevel.MANUAL: set(),
        })
        tm2 = TransitionManager(policy=policy)
        req = TransitionRequest(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
        )
        result = tm2.request_transition(req)
        assert result["approved"] is False
        assert "not allowed" in result["reason"]

    def test_request_same_level_denied(self, tm):
        req = TransitionRequest(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.MANUAL,
        )
        result = tm.request_transition(req)
        assert result["approved"] is False

    def test_request_confirmation_high_to_low(self, tm):
        req = TransitionRequest(
            from_level=AutonomyLevel.FULL_AUTO,
            to_level=AutonomyLevel.MANUAL,
        )
        result = tm.request_transition(req)
        assert result["requires_confirmation"] is True

    def test_request_no_confirmation_adjacent(self, tm):
        req = TransitionRequest(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
        )
        result = tm.request_transition(req)
        assert result["requires_confirmation"] is False

    def test_request_explicit_confirmation(self, tm):
        req = TransitionRequest(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
            requires_confirmation=True,
        )
        result = tm.request_transition(req)
        assert result["requires_confirmation"] is True

    def test_request_urgency_critical(self, tm):
        req = TransitionRequest(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
            urgency="critical",
        )
        result = tm.request_transition(req)
        assert result["approved"] is True

    # cooldown
    def test_cooldown_blocks_normal(self):
        policy = TransitionPolicy.default_policy()
        policy.cooldown_seconds = 100.0
        tm_cool = TransitionManager(policy=policy)
        tm_cool.execute_transition(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)
        req = TransitionRequest(
            from_level=AutonomyLevel.ASSISTED,
            to_level=AutonomyLevel.SEMI_AUTO,
        )
        result = tm_cool.request_transition(req)
        assert result["approved"] is False

    def test_cooldown_bypassed_by_critical(self):
        policy = TransitionPolicy.default_policy()
        policy.cooldown_seconds = 100.0
        tm_cool = TransitionManager(policy=policy)
        tm_cool.execute_transition(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)
        req = TransitionRequest(
            from_level=AutonomyLevel.ASSISTED,
            to_level=AutonomyLevel.MANUAL,
            urgency="critical",
        )
        result = tm_cool.request_transition(req)
        assert result["approved"] is True

    def test_cooldown_clear_after_time(self):
        policy = TransitionPolicy.default_policy()
        policy.cooldown_seconds = 1.0
        tm_cool = TransitionManager(policy=policy)
        tm_cool._last_transition_time = time.time() - 2.0
        remaining = tm_cool.check_cooldown(tm_cool._last_transition_time)
        assert remaining == 0.0

    # rate limiting
    def test_rate_limit_enforced(self):
        policy = TransitionPolicy.default_policy()
        policy.cooldown_seconds = 0.0
        policy.max_transitions_per_hour = 2
        tm_rate = TransitionManager(policy=policy)
        tm_rate.execute_transition(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)
        tm_rate.execute_transition(AutonomyLevel.ASSISTED, AutonomyLevel.SEMI_AUTO)
        req = TransitionRequest(
            from_level=AutonomyLevel.SEMI_AUTO,
            to_level=AutonomyLevel.AUTO_WITH_SUPERVISION,
        )
        result = tm_rate.request_transition(req)
        assert result["approved"] is False

    # execute_transition
    def test_execute_changes_level(self, tm):
        new = tm.execute_transition(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)
        assert new == AutonomyLevel.ASSISTED
        assert tm.current_level == AutonomyLevel.ASSISTED

    def test_execute_records_history(self, tm):
        tm.execute_transition(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)
        hist = tm.get_transition_history()
        assert len(hist) == 1
        assert hist[0]["from_level"] == AutonomyLevel.MANUAL
        assert hist[0]["to_level"] == AutonomyLevel.ASSISTED

    def test_execute_multiple(self, tm):
        tm.execute_transition(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)
        tm.execute_transition(AutonomyLevel.ASSISTED, AutonomyLevel.SEMI_AUTO)
        assert tm.current_level == AutonomyLevel.SEMI_AUTO
        assert len(tm.get_transition_history()) == 2

    # get_available_transitions
    def test_available_returns_list(self, tm):
        avail = tm.get_available_transitions(AutonomyLevel.MANUAL)
        assert isinstance(avail, list)

    def test_available_sorted(self, tm):
        avail = tm.get_available_transitions(AutonomyLevel.MANUAL)
        vals = [lv.value for lv in avail]
        assert vals == sorted(vals)

    def test_available_excludes_current(self, tm):
        avail = tm.get_available_transitions(AutonomyLevel.MANUAL)
        assert AutonomyLevel.MANUAL not in avail

    # compute_transition_safety
    def test_safety_returns_float(self, tm):
        score = tm.compute_transition_safety(
            AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, {}
        )
        assert isinstance(score, float)

    def test_safety_range(self, tm):
        for cur in AutonomyLevel:
            for tgt in AutonomyLevel:
                if cur == tgt:
                    continue
                score = tm.compute_transition_safety(cur, tgt, {"risk": 0.5})
                assert 0.0 <= score <= 1.0

    def test_safety_lower_is_safer(self, tm):
        low = tm.compute_transition_safety(
            AutonomyLevel.FULL_AUTO, AutonomyLevel.MANUAL, {}
        )
        high = tm.compute_transition_safety(
            AutonomyLevel.MANUAL, AutonomyLevel.FULL_AUTO, {}
        )
        assert low >= high

    # check_cooldown
    def test_cooldown_none(self, tm):
        assert tm.check_cooldown(None) == 0.0

    def test_cooldown_just_happened(self):
        policy = TransitionPolicy.default_policy()
        policy.cooldown_seconds = 30.0
        tm_cool = TransitionManager(policy=policy)
        remaining = tm_cool.check_cooldown(time.time())
        assert remaining > 0

    # get_transition_history
    def test_history_empty_initially(self, tm):
        assert tm.get_transition_history() == []

    def test_history_dicts_have_keys(self, tm):
        tm.execute_transition(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)
        rec = tm.get_transition_history()[0]
        for key in ("from_level", "to_level", "reason", "timestamp", "approved"):
            assert key in rec
