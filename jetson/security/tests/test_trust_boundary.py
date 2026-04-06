"""Tests for trust_boundary module."""

import time
import pytest
from jetson.security.trust_boundary import (
    AccessRecord,
    AccessResult,
    DataClass,
    RateLimitResult,
    TrustBoundary,
    TrustBoundaryEnforcer,
    TrustPolicy,
)


# ── TrustBoundary Enum ──────────────────────────────────────────────

class TestTrustBoundaryEnum:
    def test_ordering(self):
        assert TrustBoundary.EXTERNAL < TrustBoundary.EDGE
        assert TrustBoundary.EDGE < TrustBoundary.AGENT
        assert TrustBoundary.AGENT < TrustBoundary.FLEET

    def test_values(self):
        assert TrustBoundary.EXTERNAL == 0
        assert TrustBoundary.EDGE == 1
        assert TrustBoundary.AGENT == 2
        assert TrustBoundary.FLEET == 3

    def test_iterable(self):
        assert len(list(TrustBoundary)) == 4


# ── DataClass Enum ──────────────────────────────────────────────────

class TestDataClassEnum:
    def test_values(self):
        assert DataClass.PUBLIC < DataClass.INTERNAL
        assert DataClass.INTERNAL < DataClass.CONFIDENTIAL
        assert DataClass.CONFIDENTIAL < DataClass.SECRET


# ── TrustPolicy ─────────────────────────────────────────────────────

class TestTrustPolicy:
    def test_default(self):
        p = TrustPolicy(source=TrustBoundary.EDGE, target=TrustBoundary.EDGE)
        assert p.max_data_class == DataClass.INTERNAL
        assert "read" in p.allowed_operations
        assert p.rate_limit == 100

    def test_custom(self):
        p = TrustPolicy(
            source=TrustBoundary.FLEET,
            target=TrustBoundary.AGENT,
            max_data_class=DataClass.SECRET,
            allowed_operations={"read", "write", "execute"},
            rate_limit=50,
        )
        assert p.rate_limit == 50
        assert "write" in p.allowed_operations


# ── TrustBoundaryEnforcer ──────────────────────────────────────────

class TestEnforcerConstruction:
    def test_construct(self):
        e = TrustBoundaryEnforcer()
        assert e.audit_access_log() == []
        assert e.get_policies() == []

    def test_default_policies_empty(self):
        e = TrustBoundaryEnforcer()
        assert len(e.get_policies()) == 0


class TestCheckAccess:
    def test_fleet_to_fleet_allowed(self):
        e = TrustBoundaryEnforcer()
        assert e.check_access(TrustBoundary.FLEET, TrustBoundary.FLEET, "read") == AccessResult.ALLOWED

    def test_fleet_to_agent_allowed(self):
        e = TrustBoundaryEnforcer()
        assert e.check_access(TrustBoundary.FLEET, TrustBoundary.AGENT, "read") == AccessResult.ALLOWED

    def test_fleet_to_edge_allowed(self):
        e = TrustBoundaryEnforcer()
        assert e.check_access(TrustBoundary.FLEET, TrustBoundary.EDGE, "read") == AccessResult.ALLOWED

    def test_fleet_to_external_allowed(self):
        e = TrustBoundaryEnforcer()
        assert e.check_access(TrustBoundary.FLEET, TrustBoundary.EXTERNAL, "read") == AccessResult.ALLOWED

    def test_agent_to_fleet_denied(self):
        e = TrustBoundaryEnforcer()
        assert e.check_access(TrustBoundary.AGENT, TrustBoundary.FLEET, "read") == AccessResult.DENIED

    def test_agent_to_agent_allowed(self):
        e = TrustBoundaryEnforcer()
        assert e.check_access(TrustBoundary.AGENT, TrustBoundary.AGENT, "read") == AccessResult.ALLOWED

    def test_edge_to_fleet_denied(self):
        e = TrustBoundaryEnforcer()
        assert e.check_access(TrustBoundary.EDGE, TrustBoundary.FLEET, "read") == AccessResult.DENIED

    def test_edge_to_agent_denied(self):
        e = TrustBoundaryEnforcer()
        assert e.check_access(TrustBoundary.EDGE, TrustBoundary.AGENT, "read") == AccessResult.DENIED

    def test_external_to_fleet_denied(self):
        e = TrustBoundaryEnforcer()
        assert e.check_access(TrustBoundary.EXTERNAL, TrustBoundary.FLEET, "read") == AccessResult.DENIED

    def test_external_to_edge_denied(self):
        e = TrustBoundaryEnforcer()
        assert e.check_access(TrustBoundary.EXTERNAL, TrustBoundary.EDGE, "read") == AccessResult.DENIED

    def test_external_to_external_allowed(self):
        e = TrustBoundaryEnforcer()
        assert e.check_access(TrustBoundary.EXTERNAL, TrustBoundary.EXTERNAL, "read") == AccessResult.ALLOWED

    def test_policy_restricts_operation(self):
        e = TrustBoundaryEnforcer()
        e.add_policy(TrustPolicy(
            source=TrustBoundary.FLEET,
            target=TrustBoundary.AGENT,
            allowed_operations={"read"},  # only read
        ))
        assert e.check_access(TrustBoundary.FLEET, TrustBoundary.AGENT, "read") == AccessResult.ALLOWED
        assert e.check_access(TrustBoundary.FLEET, TrustBoundary.AGENT, "write") == AccessResult.DENIED

    def test_custom_trust_matrix(self):
        e = TrustBoundaryEnforcer()
        e.set_trust_matrix({
            (TrustBoundary.EDGE, TrustBoundary.FLEET): True,
        })
        # Default entries removed — only explicit entries work
        # EDGE->FLEET now allowed
        assert e.check_access(TrustBoundary.EDGE, TrustBoundary.FLEET, "read") == AccessResult.ALLOWED
        # EDGE->EDGE no longer in matrix -> denied
        assert e.check_access(TrustBoundary.EDGE, TrustBoundary.EDGE, "read") == AccessResult.DENIED


class TestValidateData:
    def test_valid_data_no_policy(self):
        e = TrustBoundaryEnforcer()
        result = e.validate_data(TrustBoundary.EDGE, {"data": "hello"})
        assert result["valid"] is True

    def test_valid_data_with_policy(self):
        e = TrustBoundaryEnforcer()
        policy = TrustPolicy(source=TrustBoundary.EDGE, target=TrustBoundary.FLEET, max_data_class=DataClass.CONFIDENTIAL)
        e.add_policy(policy)
        result = e.validate_data(TrustBoundary.EDGE, {"data_class": DataClass.PUBLIC}, policy)
        assert result["valid"] is True

    def test_exceeds_data_class(self):
        e = TrustBoundaryEnforcer()
        policy = TrustPolicy(source=TrustBoundary.EDGE, target=TrustBoundary.FLEET, max_data_class=DataClass.INTERNAL)
        result = e.validate_data(TrustBoundary.EDGE, {"data_class": DataClass.SECRET}, policy)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_data_class_with_int(self):
        e = TrustBoundaryEnforcer()
        policy = TrustPolicy(source=TrustBoundary.EDGE, target=TrustBoundary.FLEET, max_data_class=DataClass.PUBLIC)
        result = e.validate_data(TrustBoundary.EDGE, {"data_class": 1}, policy)  # 1 > 0
        assert result["valid"] is False


class TestEnforceRateLimit:
    def test_under_limit(self):
        e = TrustBoundaryEnforcer()
        result = e.enforce_rate_limit(TrustBoundary.EDGE, "read", limit=5, window=60)
        assert result.allowed is True
        assert result.remaining == 4

    def test_at_limit(self):
        e = TrustBoundaryEnforcer()
        for _ in range(5):
            r = e.enforce_rate_limit(TrustBoundary.EDGE, "read", limit=5, window=60)
        # After 5 requests, the 5th succeeds with remaining=0
        assert r.remaining == 0
        # 6th request should be denied
        r6 = e.enforce_rate_limit(TrustBoundary.EDGE, "read", limit=5, window=60)
        assert r6.allowed is False

    def test_multiple_operations_tracked_separately(self):
        e = TrustBoundaryEnforcer()
        for _ in range(5):
            e.enforce_rate_limit(TrustBoundary.EDGE, "read", limit=3, window=60)
        # "write" should still be allowed
        r = e.enforce_rate_limit(TrustBoundary.EDGE, "write", limit=3, window=60)
        assert r.allowed is True

    def test_reset_time_set(self):
        e = TrustBoundaryEnforcer()
        r = e.enforce_rate_limit(TrustBoundary.EDGE, "read", limit=1, window=60)
        assert r.reset_time > time.time() - 1


class TestCrossBoundaryTransfer:
    def test_allowed_transfer(self):
        e = TrustBoundaryEnforcer()
        result = e.cross_boundary_transfer(
            TrustBoundary.FLEET, TrustBoundary.AGENT,
            {"sensor": "temp", "value": 25.0},
        )
        assert "_error" not in result
        assert result["value"] == 25.0

    def test_denied_transfer(self):
        e = TrustBoundaryEnforcer()
        result = e.cross_boundary_transfer(
            TrustBoundary.EDGE, TrustBoundary.FLEET,
            {"data": "sensitive"},
        )
        assert result.get("_error") == "access_denied"

    def test_sensitive_fields_filtered(self):
        e = TrustBoundaryEnforcer()
        result = e.cross_boundary_transfer(
            TrustBoundary.FLEET, TrustBoundary.AGENT,
            {"sensor": "temp", "password": "secret123"},
        )
        assert "password" not in result
        assert "sensor" in result

    def test_fleet_internal_keeps_secrets(self):
        e = TrustBoundaryEnforcer()
        result = e.cross_boundary_transfer(
            TrustBoundary.FLEET, TrustBoundary.FLEET,
            {"key": "secret", "data": "public"},
        )
        # Within fleet, secrets preserved (same source and target)
        assert "key" in result

    def test_token_filtered(self):
        e = TrustBoundaryEnforcer()
        result = e.cross_boundary_transfer(
            TrustBoundary.FLEET, TrustBoundary.EDGE,
            {"token": "abc123", "reading": 42},
        )
        assert "token" not in result
        assert "reading" in result

    def test_credential_filtered(self):
        e = TrustBoundaryEnforcer()
        result = e.cross_boundary_transfer(
            TrustBoundary.FLEET, TrustBoundary.AGENT,
            {"credential": "key123", "name": "sensor1"},
        )
        assert "credential" not in result


class TestAuditLog:
    def test_accesses_logged(self):
        e = TrustBoundaryEnforcer()
        e.check_access(TrustBoundary.FLEET, TrustBoundary.AGENT, "read")
        e.check_access(TrustBoundary.EDGE, TrustBoundary.FLEET, "write")
        log = e.audit_access_log()
        assert len(log) == 2

    def test_log_records_details(self):
        e = TrustBoundaryEnforcer()
        e.check_access(TrustBoundary.EDGE, TrustBoundary.FLEET, "hack")
        log = e.audit_access_log()
        assert log[0].result == AccessResult.DENIED
        assert log[0].operation == "hack"

    def test_clear_log(self):
        e = TrustBoundaryEnforcer()
        e.check_access(TrustBoundary.FLEET, TrustBoundary.AGENT, "read")
        e.clear_log()
        assert e.audit_access_log() == []

    def test_get_policies(self):
        e = TrustBoundaryEnforcer()
        p = TrustPolicy(source=TrustBoundary.FLEET, target=TrustBoundary.AGENT)
        e.add_policy(p)
        assert len(e.get_policies()) == 1
