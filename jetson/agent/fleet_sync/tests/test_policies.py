"""
Tests for Conflict Resolution Policies module.
"""

import pytest
from jetson.agent.fleet_sync.policies import (
    ConflictResolver, ResolutionPolicy, StateDomain, LWWEntry,
    ConflictRecord, DEFAULT_DOMAIN_POLICIES,
)


# ==============================================================================
# LWWEntry Tests
# ==============================================================================

class TestLWWEntry:
    """Test LWW entry."""

    def test_create(self):
        entry = LWWEntry(value=42, timestamp=100.0, vessel_id="v0")
        assert entry.value == 42
        assert entry.timestamp == 100.0

    def test_vector_clock_default(self):
        entry = LWWEntry(value=1, timestamp=1.0, vessel_id="v0")
        assert entry.vector_clock == {}


# ==============================================================================
# ConflictResolver Tests
# ==============================================================================

class TestConflictResolver:
    """Test conflict resolver."""

    def test_set_get_policy(self):
        resolver = ConflictResolver()
        resolver.set_policy(StateDomain.TRUST_SCORES, ResolutionPolicy.HIGHEST_WINS)
        assert resolver.get_policy(StateDomain.TRUST_SCORES) == ResolutionPolicy.HIGHEST_WINS

    def test_default_policies(self):
        resolver = ConflictResolver()
        assert resolver.get_policy(StateDomain.SAFETY_ALERTS) == ResolutionPolicy.DOMAIN_PRIORITY
        assert resolver.get_policy(StateDomain.SKILL_VERSIONS) == ResolutionPolicy.HIGHEST_WINS
        assert resolver.get_policy(StateDomain.RESOURCE_LEVELS) == ResolutionPolicy.LOWEST_WINS
        assert resolver.get_policy(StateDomain.VESSEL_POSITIONS) == ResolutionPolicy.LWW

    def test_trust_scores(self):
        resolver = ConflictResolver()
        resolver.set_trust_scores({"v0": 0.9, "v1": 0.3})
        assert resolver.get_trust_score("v0") == 0.9
        assert resolver.get_trust_score("v1") == 0.3
        assert resolver.get_trust_score("v2") == 0.5  # default

    def test_update_trust_score(self):
        resolver = ConflictResolver()
        resolver.update_trust_score("v0", 0.8)
        assert resolver.get_trust_score("v0") == 0.8
        resolver.update_trust_score("v0", 1.5)
        assert resolver.get_trust_score("v0") == 1.0
        resolver.update_trust_score("v0", -0.5)
        assert resolver.get_trust_score("v0") == 0.0


# ==============================================================================
# LWW Resolution Tests
# ==============================================================================

class TestLWWResolution:
    """Test Last-Writer-Wins resolution."""

    def test_newer_remote_wins(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=200.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        assert winner.vessel_id == "v1"
        assert winner.value == 2

    def test_newer_local_wins(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=1, timestamp=200.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=100.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        assert winner.vessel_id == "v0"

    def test_same_timestamp_vessel_id_tiebreak(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=100.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        assert winner.vessel_id == "v1"  # "v1" > "v0"

    def test_same_timestamp_lower_vessel_wins(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v1")
        remote = LWWEntry(value=2, timestamp=100.0, vessel_id="v0")
        winner = resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        assert winner.vessel_id == "v1"  # local is v1 > v0


# ==============================================================================
# Trust-Weighted Resolution Tests
# ==============================================================================

class TestTrustWeightedResolution:
    """Test trust-weighted resolution."""

    def test_higher_trust_wins(self):
        resolver = ConflictResolver()
        resolver.set_policy(StateDomain.VESSEL_STATUSES, ResolutionPolicy.TRUST_WEIGHTED)
        resolver.set_trust_scores({"v0": 0.9, "v1": 0.3})
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=50.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.VESSEL_STATUSES, local, remote)
        assert winner.vessel_id == "v0"  # v0 has higher trust

    def test_equal_trust_falls_back_to_lww(self):
        resolver = ConflictResolver()
        resolver.set_policy(StateDomain.VESSEL_STATUSES, ResolutionPolicy.TRUST_WEIGHTED)
        resolver.set_trust_scores({"v0": 0.5, "v1": 0.5})
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=200.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.VESSEL_STATUSES, local, remote)
        assert winner.vessel_id == "v1"  # LWW fallback

    def test_close_trust_falls_back_to_lww(self):
        resolver = ConflictResolver()
        resolver.set_policy(StateDomain.VESSEL_STATUSES, ResolutionPolicy.TRUST_WEIGHTED)
        resolver.set_trust_scores({"v0": 0.6, "v1": 0.599})
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=200.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.VESSEL_STATUSES, local, remote)
        assert winner.vessel_id == "v1"  # Close trust -> LWW


# ==============================================================================
# Domain Priority Resolution Tests
# ==============================================================================

class TestDomainPriorityResolution:
    """Test domain-specific priority resolution."""

    def test_unresolved_alert_wins_over_resolved(self):
        resolver = ConflictResolver()
        local = LWWEntry(
            value={"resolved": True, "severity": "warning"},
            timestamp=200.0, vessel_id="v0",
        )
        remote = LWWEntry(
            value={"resolved": False, "severity": "warning"},
            timestamp=100.0, vessel_id="v1",
        )
        winner = resolver.resolve("k", StateDomain.SAFETY_ALERTS, local, remote)
        assert winner.vessel_id == "v1"  # unresolved wins

    def test_higher_severity_wins(self):
        resolver = ConflictResolver()
        local = LWWEntry(
            value={"resolved": False, "severity": "warning"},
            timestamp=100.0, vessel_id="v0",
        )
        remote = LWWEntry(
            value={"resolved": False, "severity": "emergency"},
            timestamp=100.0, vessel_id="v1",
        )
        winner = resolver.resolve("k", StateDomain.SAFETY_ALERTS, local, remote)
        assert winner.vessel_id == "v1"  # emergency > warning

    def test_emergency_over_critical(self):
        resolver = ConflictResolver()
        local = LWWEntry({"resolved": False, "severity": "critical"}, 100.0, "v0")
        remote = LWWEntry({"resolved": False, "severity": "emergency"}, 100.0, "v1")
        winner = resolver.resolve("k", StateDomain.SAFETY_ALERTS, local, remote)
        assert winner.value["severity"] == "emergency"

    def test_non_dict_falls_back_to_lww(self):
        resolver = ConflictResolver()
        local = LWWEntry(value="string_val", timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value="string_val_2", timestamp=200.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.SAFETY_ALERTS, local, remote)
        assert winner.vessel_id == "v1"


# ==============================================================================
# Highest/Lowest Wins Tests
# ==============================================================================

class TestNumericResolution:
    """Test highest-wins and lowest-wins resolution."""

    def test_highest_wins(self):
        resolver = ConflictResolver()
        resolver.set_policy(StateDomain.SKILL_VERSIONS, ResolutionPolicy.HIGHEST_WINS)
        local = LWWEntry(value=1.0, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2.0, timestamp=100.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.SKILL_VERSIONS, local, remote)
        assert winner.value == 2.0

    def test_lowest_wins(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=50.0, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=30.0, timestamp=100.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.RESOURCE_LEVELS, local, remote)
        assert winner.value == 30.0

    def test_lowest_wins_conservative_estimate(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=75.0, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=60.0, timestamp=100.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.RESOURCE_LEVELS, local, remote)
        # Lower fuel estimate is safer
        assert winner.value == 60.0

    def test_numeric_equal_falls_to_lww(self):
        resolver = ConflictResolver()
        resolver.set_policy(StateDomain.SKILL_VERSIONS, ResolutionPolicy.HIGHEST_WINS)
        local = LWWEntry(value=1.0, timestamp=200.0, vessel_id="v0")
        remote = LWWEntry(value=1.0, timestamp=100.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.SKILL_VERSIONS, local, remote)
        assert winner.vessel_id == "v0"

    def test_non_numeric_falls_to_local(self):
        resolver = ConflictResolver()
        resolver.set_policy(StateDomain.SKILL_VERSIONS, ResolutionPolicy.HIGHEST_WINS)
        local = LWWEntry(value="abc", timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value="def", timestamp=200.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.SKILL_VERSIONS, local, remote)
        # Non-numeric values can't be compared, falls back to local
        assert winner.vessel_id == "v0"


# ==============================================================================
# Union Resolution Tests
# ==============================================================================

class TestUnionResolution:
    """Test union resolution."""

    def test_numeric_additive(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=5.0, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=3.0, timestamp=200.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.TRUST_SCORES, local, remote)
        assert winner.value == 8.0

    def test_dict_merge(self):
        resolver = ConflictResolver()
        local = LWWEntry(value={"a": 1, "b": 2}, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value={"b": 3, "c": 4}, timestamp=200.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.TRUST_SCORES, local, remote)
        assert winner.value == {"a": 1, "b": 3, "c": 4}

    def test_list_concat_dedup(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=[1, 2, 3], timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=[3, 4, 5], timestamp=200.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.TRUST_SCORES, local, remote)
        assert set(winner.value) == {1, 2, 3, 4, 5}

    def test_string_falls_to_lww(self):
        resolver = ConflictResolver()
        local = LWWEntry(value="hello", timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value="world", timestamp=200.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.TRUST_SCORES, local, remote)
        assert winner.vessel_id == "v1"


# ==============================================================================
# First Writer Wins Tests
# ==============================================================================

class TestFirstWriterWins:
    """Test first-writer-wins resolution."""

    def test_earlier_timestamp_wins(self):
        resolver = ConflictResolver()
        resolver.set_policy(StateDomain.VESSEL_POSITIONS, ResolutionPolicy.FIRST_WRITER_WINS)
        local = LWWEntry(value=1, timestamp=200.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=100.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        assert winner.vessel_id == "v1"  # earlier timestamp

    def test_same_timestamp_lower_vessel(self):
        resolver = ConflictResolver()
        resolver.set_policy(StateDomain.VESSEL_POSITIONS, ResolutionPolicy.FIRST_WRITER_WINS)
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=100.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        assert winner.vessel_id == "v0"  # lower vessel_id wins


# ==============================================================================
# Custom Resolver Tests
# ==============================================================================

class TestCustomResolver:
    """Test custom resolver function."""

    def test_custom_resolver(self):
        resolver = ConflictResolver()
        def always_local(local, remote):
            return local
        resolver.set_custom_resolver(StateDomain.VESSEL_POSITIONS, always_local)
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=200.0, vessel_id="v1")
        winner = resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        assert winner.vessel_id == "v0"


# ==============================================================================
# Conflict Logging Tests
# ==============================================================================

class TestConflictLogging:
    """Test conflict logging."""

    def test_conflict_logged(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=200.0, vessel_id="v1")
        resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        assert resolver.get_conflict_count() >= 1

    def test_same_value_no_conflict(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=42, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=42, timestamp=100.0, vessel_id="v1")
        resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        # Same value should not generate conflict
        no_same_value = [r for r in resolver.get_conflict_log()
                        if r.local_value == r.remote_value]
        assert len(no_same_value) == 0

    def test_clear_conflict_log(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=200.0, vessel_id="v1")
        resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        resolver.clear_conflict_log()
        assert resolver.get_conflict_count() == 0

    def test_domain_conflict_counts(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=200.0, vessel_id="v1")
        resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        resolver.resolve("k2", StateDomain.SAFETY_ALERTS, local, remote)
        counts = resolver.get_domain_conflict_counts()
        assert "vessel_positions" in counts
        assert "safety_alerts" in counts

    def test_conflict_record_fields(self):
        resolver = ConflictResolver()
        local = LWWEntry(value=1, timestamp=100.0, vessel_id="v0")
        remote = LWWEntry(value=2, timestamp=200.0, vessel_id="v1")
        resolver.resolve("k", StateDomain.VESSEL_POSITIONS, local, remote)
        log = resolver.get_conflict_log()
        assert len(log) >= 1
        record = log[0]
        assert record.key == "k"
        assert record.domain == StateDomain.VESSEL_POSITIONS
        assert record.winner_vessel == "v1"
        assert record.loser_vessel == "v0"
        assert record.resolution_policy == ResolutionPolicy.LWW


# ==============================================================================
# Default Domain Policies Tests
# ==============================================================================

class TestDefaultDomainPolicies:
    """Test default policy mapping."""

    def test_all_domains_have_defaults(self):
        for domain in StateDomain:
            assert domain in DEFAULT_DOMAIN_POLICIES

    def test_safety_uses_domain_priority(self):
        assert DEFAULT_DOMAIN_POLICIES[StateDomain.SAFETY_ALERTS] == ResolutionPolicy.DOMAIN_PRIORITY

    def test_skills_use_highest_wins(self):
        assert DEFAULT_DOMAIN_POLICIES[StateDomain.SKILL_VERSIONS] == ResolutionPolicy.HIGHEST_WINS

    def test_resources_use_lowest_wins(self):
        assert DEFAULT_DOMAIN_POLICIES[StateDomain.RESOURCE_LEVELS] == ResolutionPolicy.LOWEST_WINS

    def test_trust_uses_union(self):
        assert DEFAULT_DOMAIN_POLICIES[StateDomain.TRUST_SCORES] == ResolutionPolicy.UNION
