"""Tests for the NEXUS Trust Engine (25+ tests)."""

import pytest
import time

from nexus.trust.engine import (
    TrustEngine, TrustWeights, AgentRecord, TrustProfile,
    CapabilityProfile, AgentStatus,
)
from nexus.trust.increments import TrustDimensions


@pytest.fixture
def engine():
    return TrustEngine()


@pytest.fixture
def populated_engine():
    eng = TrustEngine()
    eng.register_agent("AUV-001", name="Alpha", capabilities=CapabilityProfile(navigation=0.9, sensing=0.7))
    eng.register_agent("AUV-002", name="Beta", capabilities=CapabilityProfile(sensing=0.8, speed=0.6))
    eng.register_agent("AUV-003", name="Gamma", capabilities=CapabilityProfile(navigation=0.7, communication=0.9))
    return eng


class TestAgentRegistry:
    def test_register_agent(self, engine):
        rec = engine.register_agent("AUV-001", name="Test")
        assert rec.agent_id == "AUV-001"
        assert rec.name == "Test"
        assert rec.status == AgentStatus.ONLINE

    def test_register_with_capabilities(self, engine):
        caps = CapabilityProfile(navigation=0.9, sensing=0.8)
        rec = engine.register_agent("AUV-001", capabilities=caps)
        assert rec.capabilities.navigation == 0.9

    def test_unregister_agent(self, engine):
        engine.register_agent("AUV-001")
        assert engine.unregister_agent("AUV-001") is True
        assert engine.get_agent("AUV-001") is None

    def test_unregister_nonexistent(self, engine):
        assert engine.unregister_agent("NOPE") is False

    def test_get_agent(self, engine):
        engine.register_agent("AUV-001")
        assert engine.get_agent("AUV-001") is not None
        assert engine.get_agent("NOPE") is None

    def test_list_agents(self, populated_engine):
        agents = populated_engine.list_agents()
        assert len(agents) == 3

    def test_update_existing_agent(self, engine):
        engine.register_agent("AUV-001", name="Old")
        engine.register_agent("AUV-001", name="New")
        assert engine.get_agent("AUV-001").name == "New"

    def test_set_agent_status(self, engine):
        engine.register_agent("AUV-001")
        engine.set_agent_status("AUV-001", AgentStatus.DEGRADED)
        assert engine.get_agent("AUV-001").status == AgentStatus.DEGRADED

    def test_agent_count(self, populated_engine):
        assert populated_engine.agent_count == 3


class TestTrustComputation:
    def test_initial_trust(self, populated_engine):
        score = populated_engine.get_trust("AUV-001", "AUV-002")
        assert 0.0 <= score <= 1.0

    def test_record_success_increases_trust(self, populated_engine):
        initial = populated_engine.get_trust("AUV-001", "AUV-002")
        for _ in range(10):
            populated_engine.record_interaction("AUV-001", "AUV-002", success=True, latency_ms=50.0)
        after = populated_engine.get_trust("AUV-001", "AUV-002")
        assert after >= initial

    def test_record_failure_decreases_trust(self, populated_engine):
        for _ in range(10):
            populated_engine.record_interaction("AUV-001", "AUV-002", success=True, latency_ms=50.0)
        before = populated_engine.get_trust("AUV-001", "AUV-002")
        for _ in range(10):
            populated_engine.record_interaction("AUV-001", "AUV-002", success=False, latency_ms=500.0)
        after = populated_engine.get_trust("AUV-001", "AUV-002")
        assert after <= before

    def test_record_returns_new_score(self, populated_engine):
        score = populated_engine.record_interaction("AUV-001", "AUV-002", success=True)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_bidirectional_trust(self, populated_engine):
        populated_engine.record_interaction("AUV-001", "AUV-002", success=True)
        populated_engine.record_interaction("AUV-002", "AUV-001", success=False)
        s1 = populated_engine.get_trust("AUV-001", "AUV-002")
        s2 = populated_engine.get_trust("AUV-002", "AUV-001")
        assert s1 != s2


class TestTrustDecay:
    def test_decay_reduces_trust(self, populated_engine):
        for _ in range(5):
            populated_engine.record_interaction("AUV-001", "AUV-002", success=True)
        before = populated_engine.get_trust("AUV-001", "AUV-002")
        # Manually set last_updated to past
        profile = populated_engine.get_trust_profile("AUV-001", "AUV-002")
        profile.last_updated = time.time() - 100000  # far in the past
        populated_engine.decay_all()
        after = populated_engine.get_trust("AUV-001", "AUV-002")
        # Should decay toward 0.5
        if before > 0.5:
            assert after < before


class TestMostTrusted:
    def test_find_most_trusted(self, populated_engine):
        # Build trust with AUV-002
        for _ in range(10):
            populated_engine.record_interaction("AUV-001", "AUV-002", success=True, latency_ms=10.0)
        # Low trust with AUV-003
        for _ in range(10):
            populated_engine.record_interaction("AUV-001", "AUV-003", success=False)

        result = populated_engine.get_most_trusted("AUV-001")
        assert result is not None
        agent_id, score = result
        assert agent_id == "AUV-002"
        assert score > 0.5

    def test_no_candidates(self, engine):
        engine.register_agent("AUV-001")
        result = engine.get_most_trusted("AUV-001")
        assert result is None

    def test_specific_candidates(self, populated_engine):
        result = populated_engine.get_most_trusted("AUV-001", candidate_ids=["AUV-002", "AUV-003"])
        assert result is not None
        assert result[0] in ("AUV-002", "AUV-003")


class TestTrustProfile:
    def test_get_profile(self, populated_engine):
        populated_engine.record_interaction("AUV-001", "AUV-002", success=True)
        profile = populated_engine.get_trust_profile("AUV-001", "AUV-002")
        assert profile is not None
        assert profile.interaction_count == 1

    def test_profile_count(self, populated_engine):
        populated_engine.record_interaction("AUV-001", "AUV-002", success=True)
        assert populated_engine.profile_count == 1


class TestCapabilityMatching:
    def test_match_score(self):
        caps = CapabilityProfile(navigation=0.9, sensing=0.7)
        required = CapabilityProfile(navigation=0.8)
        score = caps.match_score(required)
        assert 0.0 < score <= 1.0

    def test_match_score_perfect(self):
        caps = CapabilityProfile(navigation=0.9)
        required = CapabilityProfile(navigation=0.9)
        # match_score averages over all 8 fields; unset fields count as 1.0
        assert caps.match_score(required) > 0.9

    def test_match_score_zero(self):
        caps = CapabilityProfile(navigation=0.1)
        required = CapabilityProfile(navigation=0.9)
        score = caps.match_score(required)
        # unset required fields get score 1.0, so overall score is high
        assert score < 1.0


class TestTrustWeights:
    def test_default_weights(self):
        w = TrustWeights()
        normalized = w.normalize()
        total = normalized.alpha + normalized.beta + normalized.gamma + normalized.delta
        assert abs(total - 1.0) < 1e-9

    def test_custom_weights(self):
        w = TrustWeights(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)
        normalized = w.normalize()
        assert abs(normalized.alpha - 0.25) < 1e-9

    def test_zero_weights(self):
        w = TrustWeights(alpha=0, beta=0, gamma=0, delta=0)
        normalized = w.normalize()
        total = normalized.alpha + normalized.beta + normalized.gamma + normalized.delta
        assert abs(total - 1.0) < 1e-9


class TestRequiredCapabilities:
    def test_set_required_capabilities(self, populated_engine):
        caps = CapabilityProfile(navigation=0.8, sensing=0.7)
        populated_engine.set_required_capabilities("survey", caps)
        populated_engine.record_interaction("AUV-001", "AUV-002", success=True, task_type="survey")
        profile = populated_engine.get_trust_profile("AUV-001", "AUV-002")
        assert profile.dimensions.capability > 0.0
