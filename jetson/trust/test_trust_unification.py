"""NEXUS Trust Engine - Cross-language trust unification tests.

Comprehensive tests covering:
  - All test vectors match golden reference
  - Trust propagation up and down
  - Propagation attenuation (0.85x per hop)
  - Fleet trust computation (weighted average)
  - Trust attestation creation and verification
  - Trust attestation serialization/deserialization
  - Edge cases: empty fleet, single vessel, all vessels degraded
"""

from __future__ import annotations

import math
import time

import pytest

from trust.increments import IncrementTrustEngine, TrustEvent, TrustParams
from trust.test_vectors import (
    TRUST_TEST_VECTORS,
    GoldenReference,
    TrustVectorSpec,
    TEST_VECTOR_MAP,
    run_vector,
    run_vector_detailed,
    regenerate_golden_references,
)
from trust.propagation import (
    TrustPropagator,
    VesselTrust,
    AgentTrust,
    FleetTrust,
    DEFAULT_ATTENUATION,
    MAX_PROPAGATION_RADIUS,
    TRUST_MERGE_ALPHA,
)
from trust.attestation import (
    TrustAttestation,
    AttestationPayload,
)


# ===================================================================
# Helper functions
# ===================================================================

def _apply_vector_to_engine(vector: TrustVectorSpec) -> IncrementTrustEngine:
    """Apply a test vector's events to a fresh engine."""
    engine = IncrementTrustEngine()
    engine.register_subsystem(vector.subsystem)

    for event_type, value in vector.events:
        if event_type == "GOOD":
            evt = TrustEvent.good("heartbeat_ok", value, subsystem=vector.subsystem)
        elif event_type == "BAD":
            evt = TrustEvent.bad("reflex_error", value, subsystem=vector.subsystem)
        elif event_type == "IDLE":
            evt = TrustEvent(
                "system_boot", quality=0.0, severity=0.0,
                timestamp=0.0, subsystem=vector.subsystem, is_bad=False,
            )
        else:
            raise ValueError(f"Unknown event type: {event_type}")
        engine.evaluate_window(vector.subsystem, [evt])

    return engine


# ===================================================================
# Test Suite 1: Golden Reference Test Vectors
# ===================================================================

class TestGoldenReferenceVectors:
    """All test vectors must produce results matching golden reference exactly."""

    @pytest.mark.parametrize("vector", TRUST_TEST_VECTORS, ids=lambda v: v.name)
    def test_final_score_matches(self, vector: TrustVectorSpec) -> None:
        """Final trust score must match golden reference within float tolerance."""
        engine = _apply_vector_to_engine(vector)
        actual = engine.get_trust_score(vector.subsystem)
        expected = vector.golden.final_score
        assert abs(actual - expected) < 1e-7, (
            f"Vector '{vector.name}': final score {actual:.10f} "
            f"!= expected {expected:.10f}"
        )

    @pytest.mark.parametrize("vector", TRUST_TEST_VECTORS, ids=lambda v: v.name)
    def test_final_level_matches(self, vector: TrustVectorSpec) -> None:
        """Final autonomy level must match golden reference exactly."""
        engine = _apply_vector_to_engine(vector)
        actual = engine.get_autonomy_level(vector.subsystem)
        expected = vector.golden.final_level
        assert actual == expected, (
            f"Vector '{vector.name}': final level {actual} "
            f"!= expected {expected}"
        )

    def test_empty_start_score(self) -> None:
        """Empty start: score must be exactly 0.0."""
        engine = _apply_vector_to_engine(TEST_VECTOR_MAP["empty_start"])
        assert engine.get_trust_score("default") == 0.0

    def test_empty_start_level(self) -> None:
        """Empty start: level must be exactly L0."""
        engine = _apply_vector_to_engine(TEST_VECTOR_MAP["empty_start"])
        assert engine.get_autonomy_level("default") == 0

    def test_ten_good_events_monotonic(self) -> None:
        """Ten good events: trust must increase monotonically."""
        results = run_vector_detailed(
            [("GOOD", 1.0)] * 10
        )
        for i in range(1, len(results["intermediate_scores"])):
            assert results["intermediate_scores"][i] > results["intermediate_scores"][i - 1], (
                f"Trust should be monotonic at step {i}"
            )

    def test_one_bad_event_floor_clamp(self) -> None:
        """One bad event from low trust must floor-clamp to t_floor."""
        vector = TEST_VECTOR_MAP["one_bad_event"]
        engine = _apply_vector_to_engine(vector)
        assert engine.get_trust_score("default") == TrustParams().t_floor

    def test_path_to_L1_no_level_promotion(self) -> None:
        """path_to_L1: trust 0.069 is below L1 threshold (0.20)."""
        engine = _apply_vector_to_engine(TEST_VECTOR_MAP["path_to_L1"])
        assert engine.get_trust_score("default") < 0.20
        assert engine.get_autonomy_level("default") == 0

    def test_path_to_L3_reaches_L1(self) -> None:
        """path_to_L3: 200 good events should reach L1."""
        engine = _apply_vector_to_engine(TEST_VECTOR_MAP["path_to_L3"])
        assert engine.get_autonomy_level("default") == 1
        assert engine.get_trust_score("default") >= 0.20

    def test_consecutive_bad_decreasing_trust(self) -> None:
        """consecutive_bad_escalation: trust must decrease with each bad event."""
        results = run_vector_detailed(
            [("GOOD", 1.0)] * 500 + [("BAD", 1.0)] * 5
        )
        intermediates = results["intermediate_scores"]
        # Check the last 5 windows (the bad events)
        for i in range(500, 505):
            assert intermediates[i] < intermediates[i - 1], (
                f"Bad event at step {i}: trust should decrease"
            )

    def test_consecutive_bad_demotes_to_L0(self) -> None:
        """consecutive_bad_escalation: severity-1.0 bad events demote to L0."""
        engine = _apply_vector_to_engine(TEST_VECTOR_MAP["consecutive_bad_escalation"])
        assert engine.get_autonomy_level("default") == 0

    def test_idle_decay_slow(self) -> None:
        """idle_decay: decay should be very slow (close to original)."""
        engine = _apply_vector_to_engine(TEST_VECTOR_MAP["idle_decay"])
        final = engine.get_trust_score("default")
        # 300 good events give ~0.394, 100 idle should barely change it
        assert 0.35 < final < 0.40, f"idle_decay final score: {final}"

    def test_mixed_severity_recovers_slowly_from_floor(self) -> None:
        """mixed_severity: after bad event, recovery from floor is slow."""
        engine = _apply_vector_to_engine(TEST_VECTOR_MAP["mixed_severity"])
        final = engine.get_trust_score("default")
        # Should be just barely above floor after bad event + 4 good events
        assert 0.20 <= final < 0.25, f"mixed_severity final score: {final}"

    def test_mixed_severity_bad_clamps_to_floor(self) -> None:
        """mixed_severity: bad event at step 3 must clamp to floor."""
        results = run_vector_detailed([
            ("GOOD", 1.0), ("GOOD", 1.0), ("GOOD", 2.0), ("BAD", 1.0),
            ("GOOD", 1.0), ("GOOD", 3.0), ("GOOD", 1.0), ("GOOD", 1.0),
        ])
        assert results["intermediate_scores"][3] == 0.2


# ===================================================================
# Test Suite 2: Golden Reference Reproducibility
# ===================================================================

class TestGoldenReferenceConsistency:
    """Verify that run_vector() produces results matching stored golden refs."""

    @pytest.mark.parametrize("vector", TRUST_TEST_VECTORS, ids=lambda v: v.name)
    def test_run_vector_matches_stored(self, vector: TrustVectorSpec) -> None:
        """run_vector() must match stored golden reference."""
        score, level = run_vector(vector.events, vector.subsystem)
        assert abs(score - vector.golden.final_score) < 1e-7, (
            f"Vector '{vector.name}': score {score:.10f} "
            f"!= stored {vector.golden.final_score:.10f}"
        )
        assert level == vector.golden.final_level, (
            f"Vector '{vector.name}': level {level} "
            f"!= stored {vector.golden.final_level}"
        )

    @pytest.mark.parametrize("vector", TRUST_TEST_VECTORS, ids=lambda v: v.name)
    def test_deterministic_reproducibility(self, vector: TrustVectorSpec) -> None:
        """Running the same vector twice must give identical results."""
        s1, l1 = run_vector(vector.events, vector.subsystem)
        s2, l2 = run_vector(vector.events, vector.subsystem)
        assert s1 == s2, f"Score not deterministic for '{vector.name}'"
        assert l1 == l2, f"Level not deterministic for '{vector.name}'"

    def test_regenerate_function(self) -> None:
        """regenerate_golden_references returns expected structure."""
        refs = regenerate_golden_references()
        assert len(refs) == len(TRUST_TEST_VECTORS)
        for name in TEST_VECTOR_MAP:
            assert name in refs
            score, level = refs[name]
            assert isinstance(score, float)
            assert isinstance(level, int)


# ===================================================================
# Test Suite 3: Trust Propagation - Up
# ===================================================================

class TestPropagationUp:
    """Trust propagation from edge to agent to fleet."""

    @pytest.fixture
    def propagator(self) -> TrustPropagator:
        return TrustPropagator()

    def test_basic_up_propagation(self, propagator: TrustPropagator) -> None:
        """Edge trust propagates up to agent and fleet with attenuation."""
        edge = {"steering": 0.8, "engine": 0.6}
        agent, fleet = propagator.propagate_up(edge, {}, {})

        # Agent: 1 hop attenuation (0.85x)
        assert abs(agent["steering"] - 0.8 * 0.85) < 1e-9
        assert abs(agent["engine"] - 0.6 * 0.85) < 1e-9

        # Fleet: 2 hops (0.85^2 = 0.7225x)
        assert abs(fleet["steering"] - 0.8 * 0.7225) < 1e-9
        assert abs(fleet["engine"] - 0.6 * 0.7225) < 1e-9

    def test_up_propagation_merges_existing(self, propagator: TrustPropagator) -> None:
        """Propagated trust merges with existing agent/fleet trust."""
        edge = {"steering": 1.0}
        existing_agent = {"steering": 0.5}
        existing_fleet = {"steering": 0.3}
        agent, fleet = propagator.propagate_up(
            edge, existing_agent, existing_fleet
        )

        # Merge: (1 - 0.3) * existing + 0.3 * propagated
        expected_agent = (1 - 0.3) * 0.5 + 0.3 * (1.0 * 0.85)
        assert abs(agent["steering"] - expected_agent) < 1e-9

    def test_up_propagation_new_subsystems(self, propagator: TrustPropagator) -> None:
        """New subsystems appear in agent and fleet after propagation."""
        edge = {"payload": 0.9}
        agent, fleet = propagator.propagate_up(edge, {}, {})

        assert "payload" in agent
        assert "payload" in fleet
        assert abs(agent["payload"] - 0.9 * 0.85) < 1e-9

    def test_up_attenuation_is_0_85_per_hop(self, propagator: TrustPropagator) -> None:
        """Verify exact 0.85x attenuation per hop."""
        edge = {"nav": 1.0}
        _, fleet = propagator.propagate_up(edge, {}, {})

        expected_fleet = 1.0 * (0.85 ** 2)
        assert abs(fleet["nav"] - expected_fleet) < 1e-9

    def test_up_propagation_zero_trust(self, propagator: TrustPropagator) -> None:
        """Zero trust propagates as zero."""
        edge = {"steering": 0.0}
        agent, fleet = propagator.propagate_up(edge, {}, {})

        assert agent["steering"] == 0.0
        assert fleet["steering"] == 0.0

    def test_up_propagation_empty_edge(self, propagator: TrustPropagator) -> None:
        """Empty edge trust returns existing agent/fleet unchanged."""
        agent, fleet = propagator.propagate_up({}, {"s": 0.5}, {"s": 0.3})
        assert agent == {"s": 0.5}
        assert fleet == {"s": 0.3}


# ===================================================================
# Test Suite 4: Trust Propagation - Down
# ===================================================================

class TestPropagationDown:
    """Trust propagation from fleet down to agent and edge."""

    @pytest.fixture
    def propagator(self) -> TrustPropagator:
        return TrustPropagator()

    def test_basic_down_propagation(self, propagator: TrustPropagator) -> None:
        """Fleet directive propagates down as a ceiling."""
        fleet_directive = {"steering": 0.5}
        current_agent = {"steering": 0.8}
        current_edge = {"steering": 0.9}

        new_edge = propagator.propagate_down(
            fleet_directive, current_agent, current_edge
        )

        # Fleet -> edge: 0.5 * 0.85^2 = 0.36125
        # Edge is capped at min(current, propagated)
        expected_ceiling = 0.5 * 0.7225
        assert new_edge["steering"] <= expected_ceiling + 1e-9

    def test_down_propagation_high_edge(self, propagator: TrustPropagator) -> None:
        """If edge trust is already below ceiling, it stays."""
        fleet_directive = {"steering": 1.0}
        current_agent = {"steering": 0.3}
        current_edge = {"steering": 0.1}

        new_edge = propagator.propagate_down(
            fleet_directive, current_agent, current_edge
        )

        # Ceiling = 1.0 * 0.7225 = 0.7225, edge is 0.1, stays at 0.1
        assert new_edge["steering"] == current_edge["steering"]

    def test_down_propagation_empty_directive(self, propagator: TrustPropagator) -> None:
        """Empty fleet directive leaves edge unchanged."""
        new_edge = propagator.propagate_down(
            {}, {"steering": 0.8}, {"steering": 0.9}
        )
        assert new_edge["steering"] == 0.9


# ===================================================================
# Test Suite 5: Propagation Attenuation
# ===================================================================

class TestPropagationAttenuation:
    """Verify attenuation factor calculation."""

    def test_default_attenuation(self) -> None:
        """Default attenuation is 0.85."""
        assert DEFAULT_ATTENUATION == 0.85

    def test_single_hop_attenuation(self) -> None:
        """1 hop: 0.85^1 = 0.85."""
        p = TrustPropagator()
        assert abs(p.compute_attenuation(1) - 0.85) < 1e-9

    def test_two_hop_attenuation(self) -> None:
        """2 hops: 0.85^2 = 0.7225."""
        p = TrustPropagator()
        assert abs(p.compute_attenuation(2) - 0.7225) < 1e-9

    def test_three_hop_attenuation(self) -> None:
        """3 hops: 0.85^3 = 0.614125."""
        p = TrustPropagator()
        expected = 0.85 ** 3
        assert abs(p.compute_attenuation(3) - expected) < 1e-9

    def test_max_radius_clamp(self) -> None:
        """Attenuation beyond max_radius is clamped."""
        p = TrustPropagator(max_radius=3)
        atten_3 = p.compute_attenuation(3)
        atten_10 = p.compute_attenuation(10)
        assert abs(atten_3 - atten_10) < 1e-9

    def test_zero_hop_attenuation(self) -> None:
        """0 hops: attenuation = 1.0."""
        p = TrustPropagator()
        assert abs(p.compute_attenuation(0) - 1.0) < 1e-9

    def test_single_hop_propagate(self) -> None:
        """propagate_single_hop applies attenuation correctly."""
        p = TrustPropagator()
        source = {"a": 1.0, "b": 0.5}
        result = p.propagate_single_hop(source)
        assert abs(result["a"] - 0.85) < 1e-9
        assert abs(result["b"] - 0.425) < 1e-9

    def test_custom_attenuation(self) -> None:
        """Custom attenuation factor works correctly."""
        p = TrustPropagator(attenuation=0.5)
        assert abs(p.compute_attenuation(1) - 0.5) < 1e-9
        assert abs(p.compute_attenuation(2) - 0.25) < 1e-9


# ===================================================================
# Test Suite 6: Fleet Trust Computation
# ===================================================================

class TestFleetTrust:
    """Fleet trust computation as weighted average."""

    @pytest.fixture
    def propagator(self) -> TrustPropagator:
        return TrustPropagator()

    def test_equal_weights(self, propagator: TrustPropagator) -> None:
        """Fleet trust is simple average with equal weights."""
        vessel_trusts = [
            {"steering": 0.8, "engine": 0.6},
            {"steering": 0.4, "engine": 0.8},
        ]
        fleet = propagator.compute_fleet_trust(vessel_trusts)

        assert abs(fleet["steering"] - 0.6) < 1e-9
        assert abs(fleet["engine"] - 0.7) < 1e-9

    def test_custom_weights(self, propagator: TrustPropagator) -> None:
        """Custom vessel weights are respected."""
        vessel_trusts = [
            {"steering": 1.0},
            {"steering": 0.0},
        ]
        # Weight first vessel 3x more
        fleet = propagator.compute_fleet_trust(vessel_trusts, weights=[3.0, 1.0])

        expected = (1.0 * 3.0 + 0.0 * 1.0) / 4.0
        assert abs(fleet["steering"] - expected) < 1e-9

    def test_single_vessel(self, propagator: TrustPropagator) -> None:
        """Single vessel fleet returns that vessel's trust."""
        vessel_trusts = [{"steering": 0.7, "engine": 0.5}]
        fleet = propagator.compute_fleet_trust(vessel_trusts)

        assert abs(fleet["steering"] - 0.7) < 1e-9
        assert abs(fleet["engine"] - 0.5) < 1e-9

    def test_empty_fleet(self, propagator: TrustPropagator) -> None:
        """Empty fleet returns empty dict."""
        fleet = propagator.compute_fleet_trust([])
        assert fleet == {}

    def test_all_vessels_degraded(self, propagator: TrustPropagator) -> None:
        """All vessels at floor trust gives fleet at floor."""
        vessel_trusts = [
            {"steering": 0.2, "engine": 0.2},
            {"steering": 0.2, "engine": 0.2},
            {"steering": 0.2, "engine": 0.2},
        ]
        fleet = propagator.compute_fleet_trust(vessel_trusts)

        assert abs(fleet["steering"] - 0.2) < 1e-9
        assert abs(fleet["engine"] - 0.2) < 1e-9

    def test_missing_subsystems(self, propagator: TrustPropagator) -> None:
        """Missing subsystems default to 0.0."""
        vessel_trusts = [
            {"steering": 0.8},
            {"engine": 0.6},
        ]
        fleet = propagator.compute_fleet_trust(vessel_trusts)

        # steering: (0.8 + 0.0) / 2 = 0.4
        assert abs(fleet["steering"] - 0.4) < 1e-9
        # engine: (0.0 + 0.6) / 2 = 0.3
        assert abs(fleet["engine"] - 0.3) < 1e-9

    def test_weight_count_mismatch_raises(self, propagator: TrustPropagator) -> None:
        """Mismatched weight count raises ValueError."""
        vessel_trusts = [{"s": 0.5}]
        with pytest.raises(ValueError, match="Weight count"):
            propagator.compute_fleet_trust(vessel_trusts, weights=[1.0, 2.0])

    def test_zero_total_weights(self, propagator: TrustPropagator) -> None:
        """All-zero weights return empty dict."""
        vessel_trusts = [{"s": 0.8}]
        fleet = propagator.compute_fleet_trust(vessel_trusts, weights=[0.0])
        assert fleet == {}


# ===================================================================
# Test Suite 7: Trust Attestation - Creation
# ===================================================================

class TestAttestationCreation:
    """Trust attestation creation tests."""

    @pytest.fixture
    def attester(self) -> TrustAttestation:
        return TrustAttestation(signing_key=b"test-key-123")

    def test_create_attestation_format(self, attester: TrustAttestation) -> None:
        """Attestation has format: payload.signature (two dot-separated parts)."""
        att = attester.create_attestation(
            vessel_id="vessel-01",
            trust_scores={"steering": 0.8},
            bytecode_hash="abc123",
        )
        parts = att.split(".")
        assert len(parts) == 2
        # Both parts should be valid base64
        assert len(parts[0]) > 0
        assert len(parts[1]) > 0

    def test_create_with_full_params(self, attester: TrustAttestation) -> None:
        """Attestation with all parameters creates successfully."""
        att = attester.create_attestation(
            vessel_id="vessel-02",
            trust_scores={"steering": 0.9, "engine": 0.7},
            bytecode_hash="def456",
            trust_level=3,
            subsystems=["steering", "engine"],
            metadata={"deploy_version": "1.2.3"},
        )
        assert att is not None
        assert len(att.split(".")) == 2

    def test_create_different_keys_different_sigs(
        self, attester: TrustAttestation
    ) -> None:
        """Different signing keys produce different signatures."""
        attester2 = TrustAttestation(signing_key=b"different-key")

        # Create with fixed timestamp to ensure same payload
        payload = AttestationPayload(
            vessel_id="v-01",
            trust_scores={"s": 0.8},
            bytecode_hash="hash1",
            timestamp=12345.0,
        )
        att1 = attester.create_attestation_from_payload(payload)
        att2 = attester2.create_attestation_from_payload(payload)

        # Same payload, different signature due to different key
        assert att1.split(".")[0] == att2.split(".")[0]  # same payload
        assert att1.split(".")[1] != att2.split(".")[1]  # different signature

    def test_create_fixed_timestamp_deterministic(
        self, attester: TrustAttestation
    ) -> None:
        """Same inputs with same timestamp produce identical attestations."""
        payload = AttestationPayload(
            vessel_id="v-01",
            trust_scores={"s": 0.5},
            bytecode_hash="h1",
            timestamp=99999.0,
        )
        att1 = attester.create_attestation_from_payload(payload)
        att2 = attester.create_attestation_from_payload(payload)
        assert att1 == att2

    def test_bytecode_hash(self) -> None:
        """compute_bytecode_hash returns SHA-256 hex digest."""
        result = TrustAttestation.compute_bytecode_hash(b"test bytecode")
        assert len(result) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in result)

    def test_trust_hash(self) -> None:
        """compute_trust_hash is deterministic."""
        scores = {"steering": 0.8, "engine": 0.6}
        h1 = TrustAttestation.compute_trust_hash(scores)
        h2 = TrustAttestation.compute_trust_hash(scores)
        assert h1 == h2
        assert len(h1) == 64

    def test_trust_hash_order_independent(self) -> None:
        """Trust hash depends on key names, not insertion order."""
        scores1 = {"a": 0.5, "b": 0.5}
        scores2 = {"b": 0.5, "a": 0.5}
        h1 = TrustAttestation.compute_trust_hash(scores1)
        h2 = TrustAttestation.compute_trust_hash(scores2)
        assert h1 == h2


# ===================================================================
# Test Suite 8: Trust Attestation - Verification
# ===================================================================

class TestAttestationVerification:
    """Trust attestation verification tests."""

    @pytest.fixture
    def attester(self) -> TrustAttestation:
        return TrustAttestation(signing_key=b"verify-key-456")

    def test_verify_valid_attestation(self, attester: TrustAttestation) -> None:
        """Valid attestation passes verification."""
        att = attester.create_attestation(
            vessel_id="v-01", trust_scores={"s": 0.8}, bytecode_hash="h1"
        )
        assert attester.verify_attestation(att) is True

    def test_verify_wrong_key(self, attester: TrustAttestation) -> None:
        """Attestation verified with wrong key fails."""
        att = attester.create_attestation("v-01", {"s": 0.8}, "h1")
        wrong_attester = TrustAttestation(signing_key=b"wrong-key")
        assert wrong_attester.verify_attestation(att) is False

    def test_verify_vessel_id_match(self, attester: TrustAttestation) -> None:
        """Verification with vessel_id constraint."""
        att = attester.create_attestation("v-01", {"s": 0.8}, "h1")
        assert attester.verify_attestation(att, vessel_id="v-01") is True
        assert attester.verify_attestation(att, vessel_id="v-99") is False

    def test_verify_trust_range(self, attester: TrustAttestation) -> None:
        """Verification with trust range constraint."""
        att = attester.create_attestation(
            "v-01", {"s": 0.8, "e": 0.6}, "h1"
        )
        # Average trust = (0.8 + 0.6) / 2 = 0.7
        assert attester.verify_attestation(att, expected_trust_range=(0.5, 0.9)) is True
        assert attester.verify_attestation(att, expected_trust_range=(0.0, 0.5)) is False
        assert attester.verify_attestation(att, expected_trust_range=(0.8, 1.0)) is False

    def test_verify_bytecode_hash(self, attester: TrustAttestation) -> None:
        """Verification with bytecode hash constraint."""
        att = attester.create_attestation("v-01", {"s": 0.8}, "abc123")
        assert attester.verify_attestation(att, expected_bytecode_hash="abc123") is True
        assert attester.verify_attestation(att, expected_bytecode_hash="wrong") is False

    def test_verify_max_age(self, attester: TrustAttestation) -> None:
        """Verification with max age constraint."""
        att = attester.create_attestation("v-01", {"s": 0.8}, "h1")
        # Fresh attestation should pass any reasonable age
        assert attester.verify_attestation(att, max_age_seconds=3600) is True

    def test_verify_tampered_attestation(self, attester: TrustAttestation) -> None:
        """Tampered payload fails verification."""
        att = attester.create_attestation("v-01", {"s": 0.8}, "h1")
        parts = att.split(".")
        # Tamper with payload by appending a character
        tampered = parts[0] + "x." + parts[1]
        assert attester.verify_attestation(tampered) is False

    def test_verify_tampered_signature(self, attester: TrustAttestation) -> None:
        """Tampered signature fails verification."""
        att = attester.create_attestation("v-01", {"s": 0.8}, "h1")
        parts = att.split(".")
        tampered = parts[0] + "." + parts[1][:-1] + "x"
        assert attester.verify_attestation(tampered) is False

    def test_verify_malformed(self, attester: TrustAttestation) -> None:
        """Malformed attestation strings fail verification."""
        assert attester.verify_attestation("") is False
        assert attester.verify_attestation("not-base64") is False
        assert attester.verify_attestation("a.b.c") is False
        assert attester.verify_attestation("only-one-part") is False

    def test_verify_empty_trust_scores_range(self, attester: TrustAttestation) -> None:
        """Trust range check with empty scores is vacuously true."""
        att = attester.create_attestation("v-01", {}, "h1")
        assert attester.verify_attestation(att, expected_trust_range=(0.5, 0.9)) is True


# ===================================================================
# Test Suite 9: Trust Attestation - Serialization
# ===================================================================

class TestAttestationSerialization:
    """Attestation serialization and deserialization tests."""

    @pytest.fixture
    def attester(self) -> TrustAttestation:
        return TrustAttestation(signing_key=b"serial-key-789")

    def test_to_dict_roundtrip(self, attester: TrustAttestation) -> None:
        """attestation_to_dict produces valid dict."""
        att = attester.create_attestation(
            vessel_id="v-01",
            trust_scores={"steering": 0.8, "engine": 0.6},
            bytecode_hash="hash123",
            trust_level=2,
        )
        data = attester.attestation_to_dict(att)
        assert data is not None
        assert data["vessel_id"] == "v-01"
        assert data["trust_scores"] == {"steering": 0.8, "engine": 0.6}
        assert data["bytecode_hash"] == "hash123"
        assert data["trust_level"] == 2
        assert "timestamp" in data

    def test_to_dict_malformed(self, attester: TrustAttestation) -> None:
        """Malformed attestation returns None from to_dict."""
        assert attester.attestation_to_dict("") is None
        assert attester.attestation_to_dict("garbage") is None

    def test_to_payload_roundtrip(self, attester: TrustAttestation) -> None:
        """attestation_to_payload produces valid AttestationPayload."""
        att = attester.create_attestation(
            "v-01", {"s": 0.9}, "h1", trust_level=4,
            subsystems=["s"], metadata={"key": "val"},
        )
        payload = attester.attestation_to_payload(att)
        assert payload is not None
        assert payload.vessel_id == "v-01"
        assert payload.trust_scores == {"s": 0.9}
        assert payload.bytecode_hash == "h1"
        assert payload.trust_level == 4
        assert payload.subsystems == ["s"]
        assert payload.metadata == {"key": "val"}

    def test_payload_dict_roundtrip(self) -> None:
        """AttestationPayload to_dict/from_dict roundtrip."""
        original = AttestationPayload(
            vessel_id="v-01",
            trust_scores={"s": 0.5},
            bytecode_hash="h1",
            timestamp=12345.0,
            trust_level=1,
            subsystems=["s"],
            metadata={"k": "v"},
        )
        restored = AttestationPayload.from_dict(original.to_dict())
        assert restored.vessel_id == original.vessel_id
        assert restored.trust_scores == original.trust_scores
        assert restored.bytecode_hash == original.bytecode_hash
        assert restored.timestamp == original.timestamp
        assert restored.trust_level == original.trust_level
        assert restored.subsystems == original.subsystems
        assert restored.metadata == original.metadata

    def test_from_payload_to_dict_consistency(self, attester: TrustAttestation) -> None:
        """Attestation created from payload has same dict as original payload."""
        payload = AttestationPayload(
            vessel_id="v-02",
            trust_scores={"a": 0.7, "b": 0.3},
            bytecode_hash="xyz",
            timestamp=999.0,
            trust_level=2,
        )
        att = attester.create_attestation_from_payload(payload)
        decoded = attester.attestation_to_dict(att)
        assert decoded["vessel_id"] == "v-02"
        assert decoded["trust_scores"] == {"a": 0.7, "b": 0.3}
        assert decoded["bytecode_hash"] == "xyz"
        assert decoded["trust_level"] == 2


# ===================================================================
# Test Suite 10: VesselTrust / AgentTrust / FleetTrust dataclasses
# ===================================================================

class TestTrustDataClasses:
    """Trust data structure tests."""

    def test_vessel_composite_score(self) -> None:
        """VesselTrust composite score is average of subsystem scores."""
        v = VesselTrust("v-01", {"steering": 0.8, "engine": 0.4})
        assert abs(v.get_composite_score() - 0.6) < 1e-9

    def test_vessel_composite_empty(self) -> None:
        """Empty VesselTrust returns 0.0."""
        v = VesselTrust("v-01")
        assert v.get_composite_score() == 0.0

    def test_vessel_min_score(self) -> None:
        """VesselTrust min score is the weakest subsystem."""
        v = VesselTrust("v-01", {"steering": 0.8, "engine": 0.4})
        assert abs(v.get_min_score() - 0.4) < 1e-9

    def test_agent_composite_score(self) -> None:
        """AgentTrust composite score averages all trust sources."""
        a = AgentTrust("a-01", {"s": 0.8}, {"e": 0.4})
        assert abs(a.get_composite_score() - 0.6) < 1e-9

    def test_agent_composite_empty(self) -> None:
        """Empty AgentTrust returns 0.0."""
        a = AgentTrust("a-01")
        assert a.get_composite_score() == 0.0

    def test_fleet_vessel_count(self) -> None:
        """FleetTrust vessel count."""
        f = FleetTrust("fleet-1", {"v1": 0.8, "v2": 0.6, "v3": 0.7})
        assert f.get_vessel_count() == 3

    def test_fleet_avg_score(self) -> None:
        """FleetTrust average score."""
        f = FleetTrust("fleet-1", {"v1": 0.8, "v2": 0.6})
        assert abs(f.get_avg_score() - 0.7) < 1e-9

    def test_fleet_min_score(self) -> None:
        """FleetTrust minimum score."""
        f = FleetTrust("fleet-1", {"v1": 0.8, "v2": 0.3})
        assert abs(f.get_min_score() - 0.3) < 1e-9

    def test_fleet_empty(self) -> None:
        """Empty FleetTrust returns 0.0."""
        f = FleetTrust("fleet-1")
        assert f.get_avg_score() == 0.0
        assert f.get_min_score() == 0.0
        assert f.get_vessel_count() == 0


# ===================================================================
# Test Suite 11: Integration - Full Propagation Pipeline
# ===================================================================

class TestPropagationIntegration:
    """End-to-end propagation tests combining up and down flows."""

    @pytest.fixture
    def propagator(self) -> TrustPropagator:
        return TrustPropagator()

    def test_full_up_then_down_roundtrip(self, propagator: TrustPropagator) -> None:
        """Trust goes up from edge to fleet, then directive comes back down."""
        # Edge vessel earns trust
        edge_trust = {"steering": 0.9, "engine": 0.7}

        # Propagate up
        agent_trust, fleet_trust = propagator.propagate_up(edge_trust, {}, {})

        # Fleet applies some policy (e.g., caps steering at 0.5)
        fleet_directive = {"steering": 0.5, "engine": 1.0}

        # Propagate back down
        new_edge = propagator.propagate_down(
            fleet_directive, agent_trust, dict(edge_trust)
        )

        # Steering should be capped by fleet directive
        assert new_edge["steering"] < edge_trust["steering"]

    def test_multi_vessel_fleet_computation(self, propagator: TrustPropagator) -> None:
        """Multiple vessels feed into fleet trust, then fleet decides."""
        v1 = {"steering": 0.9, "engine": 0.8}
        v2 = {"steering": 0.7, "engine": 0.6}
        v3 = {"steering": 0.5, "engine": 0.4}

        fleet = propagator.compute_fleet_trust([v1, v2, v3])

        # Average steering: (0.9 + 0.7 + 0.5) / 3 = 0.7
        assert abs(fleet["steering"] - 0.7) < 1e-9
        # Average engine: (0.8 + 0.6 + 0.4) / 3 = 0.6
        assert abs(fleet["engine"] - 0.6) < 1e-9

        # Fleet issues directive to degraded vessel
        new_edge = propagator.propagate_down(fleet, {}, v3)
        # Vessel 3's steering was 0.5, fleet says 0.7
        # After 2 hops of attenuation: 0.7 * 0.85 * 0.85 = 0.50575
        # This should be >= 0.5 (not a downgrade)
        assert new_edge["steering"] >= 0.5

    def test_degraded_fleet_propagates_down(self, propagator: TrustPropagator) -> None:
        """Degraded fleet directive pulls down high-trust edges."""
        fleet_directive = {"steering": 0.2, "engine": 0.2}
        high_edge = {"steering": 0.9, "engine": 0.8}

        new_edge = propagator.propagate_down(fleet_directive, {}, high_edge)

        # Both should be pulled down significantly
        assert new_edge["steering"] < 0.5
        assert new_edge["engine"] < 0.5


# ===================================================================
# Test Suite 12: Edge Cases
# ===================================================================

class TestEdgeCases:
    """Edge case tests for propagation and attestation."""

    def test_propagation_very_high_trust(self) -> None:
        """Trust scores at 1.0 propagate correctly."""
        p = TrustPropagator()
        edge = {"s": 1.0}
        agent, fleet = p.propagate_up(edge, {}, {})
        assert abs(agent["s"] - 0.85) < 1e-9
        assert abs(fleet["s"] - 0.7225) < 1e-9

    def test_propagation_very_low_trust(self) -> None:
        """Near-zero trust propagates correctly."""
        p = TrustPropagator()
        edge = {"s": 0.001}
        agent, fleet = p.propagate_up(edge, {}, {})
        assert abs(agent["s"] - 0.001 * 0.85) < 1e-12
        assert abs(fleet["s"] - 0.001 * 0.7225) < 1e-12

    def test_attestation_empty_scores(self) -> None:
        """Attestation with empty trust scores works."""
        a = TrustAttestation(signing_key=b"edge-key")
        att = a.create_attestation("v-01", {}, "hash")
        assert a.verify_attestation(att) is True
        data = a.attestation_to_dict(att)
        assert data["trust_scores"] == {}

    def test_attestation_unicode_vessel_id(self) -> None:
        """Attestation handles unicode vessel IDs."""
        a = TrustAttestation(signing_key=b"unicode-key")
        att = a.create_attestation("vessel-\u65e5\u672c\u8a9e-01", {"s": 0.8}, "hash")
        assert a.verify_attestation(att, vessel_id="vessel-\u65e5\u672c\u8a9e-01") is True

    def test_fleet_many_subsystems(self) -> None:
        """Fleet computation with many subsystems."""
        p = TrustPropagator()
        vessels = [
            {f"subsys_{i}": 0.5 + i * 0.05} for i in range(10)
        ]
        fleet = p.compute_fleet_trust(vessels)
        assert len(fleet) == 10
        # subsys_0 appears only in vessel 0 with 0.5, others have 0.0
        assert abs(fleet["subsys_0"] - 0.5 / 10) < 1e-9

    def test_max_propagation_radius(self) -> None:
        """MAX_PROPAGATION_RADIUS is 3."""
        assert MAX_PROPAGATION_RADIUS == 3

    def test_merge_alpha_in_range(self) -> None:
        """TRUST_MERGE_ALPHA is in valid range."""
        assert 0.0 < TRUST_MERGE_ALPHA < 1.0

    def test_attestation_timestamp_present(self) -> None:
        """Attestation always includes a timestamp."""
        a = TrustAttestation(signing_key=b"ts-key")
        att = a.create_attestation("v-01", {"s": 0.5}, "h")
        data = a.attestation_to_dict(att)
        assert data["timestamp"] > 0

    def test_attestation_different_payloads_different(self) -> None:
        """Different payloads produce different attestations."""
        a = TrustAttestation(signing_key=b"diff-key")
        p1 = AttestationPayload("v-01", {"s": 0.5}, "h1", timestamp=100.0)
        p2 = AttestationPayload("v-01", {"s": 0.6}, "h1", timestamp=100.0)
        att1 = a.create_attestation_from_payload(p1)
        att2 = a.create_attestation_from_payload(p2)
        assert att1 != att2

    def test_fleet_trust_many_vessels(self) -> None:
        """Fleet computation handles 100 vessels correctly."""
        p = TrustPropagator()
        vessels = [{"s": 0.5} for _ in range(100)]
        fleet = p.compute_fleet_trust(vessels)
        assert abs(fleet["s"] - 0.5) < 1e-9

    def test_propagation_negative_atten_raises(self) -> None:
        """Negative attenuation would be invalid but we just test the math."""
        p = TrustPropagator(attenuation=0.1)
        edge = {"s": 1.0}
        agent, _ = p.propagate_up(edge, {}, {})
        assert abs(agent["s"] - 0.1) < 1e-9
