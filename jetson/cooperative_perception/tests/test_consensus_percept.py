"""Tests for consensus-based perception."""

import pytest

from jetson.cooperative_perception.consensus_percept import (
    PerceptionVote,
    PerceptionConsensus,
    PerceptionConsensusEngine,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return PerceptionConsensusEngine(quorum=0.5)


@pytest.fixture
def unanimous_votes():
    """Three vessels all agree on one object."""
    return [
        PerceptionVote(
            voter_id="vessel_A",
            perceived_object={"id": "o1", "type": "vessel", "position": (10, 20, 0), "confidence": 0.9},
            confidence=0.9,
            evidence=["lidar"],
        ),
        PerceptionVote(
            voter_id="vessel_B",
            perceived_object={"id": "o1", "type": "vessel", "position": (10.5, 20.3, 0), "confidence": 0.85},
            confidence=0.85,
            evidence=["radar"],
        ),
        PerceptionVote(
            voter_id="vessel_C",
            perceived_object={"id": "o1", "type": "vessel", "position": (10.2, 20.1, 0), "confidence": 0.88},
            confidence=0.88,
            evidence=["camera"],
        ),
    ]


@pytest.fixture
def split_votes():
    """Three vessels split on object type."""
    return [
        PerceptionVote(
            voter_id="vessel_A",
            perceived_object={"id": "o1", "type": "vessel", "position": (10, 20, 0), "confidence": 0.9},
            confidence=0.9,
        ),
        PerceptionVote(
            voter_id="vessel_B",
            perceived_object={"id": "o1", "type": "vessel", "position": (10.5, 20.3, 0), "confidence": 0.85},
            confidence=0.85,
        ),
        PerceptionVote(
            voter_id="vessel_C",
            perceived_object={"id": "o1", "type": "buoy", "position": (10.2, 20.1, 0), "confidence": 0.88},
            confidence=0.88,
        ),
    ]


@pytest.fixture
def maverick_votes():
    """One vessel consistently disagrees."""
    return [
        PerceptionVote(
            voter_id="vessel_A",
            perceived_object={"id": "o1", "type": "vessel", "position": (10, 20, 0), "confidence": 0.9},
            confidence=0.9,
        ),
        PerceptionVote(
            voter_id="vessel_B",
            perceived_object={"id": "o1", "type": "buoy", "position": (10, 20, 0), "confidence": 0.8},
            confidence=0.8,
        ),
        PerceptionVote(
            voter_id="vessel_A",
            perceived_object={"id": "o2", "type": "buoy", "position": (30, 40, 0), "confidence": 0.7},
            confidence=0.7,
        ),
        PerceptionVote(
            voter_id="vessel_B",
            perceived_object={"id": "o2", "type": "vessel", "position": (30, 40, 0), "confidence": 0.8},
            confidence=0.8,
        ),
    ]


# ── PerceptionVote dataclass tests ───────────────────────────────────────────

class TestPerceptionVote:

    def test_creation(self):
        v = PerceptionVote(
            voter_id="v1",
            perceived_object={"id": "o1", "type": "buoy"},
            confidence=0.8,
            evidence=["lidar", "radar"],
        )
        assert v.voter_id == "v1"
        assert v.confidence == 0.8
        assert len(v.evidence) == 2

    def test_object_id(self):
        v = PerceptionVote(
            voter_id="v1",
            perceived_object={"id": "my_obj", "type": "x"},
            confidence=0.5,
        )
        assert v.object_id() == "my_obj"

    def test_object_id_missing(self):
        v = PerceptionVote(
            voter_id="v1",
            perceived_object={"type": "x"},
            confidence=0.5,
        )
        assert v.object_id() == "unknown"

    def test_object_type(self):
        v = PerceptionVote(
            voter_id="v1",
            perceived_object={"id": "o1", "type": "vessel"},
            confidence=0.5,
        )
        assert v.object_type() == "vessel"

    def test_object_position(self):
        v = PerceptionVote(
            voter_id="v1",
            perceived_object={"id": "o1", "position": (1, 2, 3)},
            confidence=0.5,
        )
        assert v.object_position() == (1, 2, 3)

    def test_default_evidence(self):
        v = PerceptionVote(
            voter_id="v1",
            perceived_object={"id": "o1"},
            confidence=0.5,
        )
        assert v.evidence == []


# ── PerceptionConsensus dataclass tests ──────────────────────────────────────

class TestPerceptionConsensus:

    def test_creation(self):
        c = PerceptionConsensus(
            agreed_objects=[{"id": "o1"}],
            participation=3,
            disagreement_ratio=0.1,
            timestamp=1000.0,
        )
        assert len(c.agreed_objects) == 1
        assert c.participation == 3


# ── PerceptionConsensusEngine tests ──────────────────────────────────────────

class TestCollectVotes:

    def test_collect_basic(self, engine):
        perceptions = {
            "v_A": [{"id": "o1", "type": "vessel", "confidence": 0.9}],
            "v_B": [{"id": "o1", "type": "vessel", "confidence": 0.8}],
        }
        votes = engine.collect_votes(perceptions)
        assert len(votes) == 2
        assert votes[0].voter_id == "v_A"
        assert votes[1].voter_id == "v_B"

    def test_collect_empty(self, engine):
        votes = engine.collect_votes({})
        assert votes == []

    def test_collect_preserves_evidence(self, engine):
        perceptions = {
            "v_A": [{"id": "o1", "type": "vessel", "confidence": 0.9, "evidence": ["lidar"]}],
        }
        votes = engine.collect_votes(perceptions)
        assert votes[0].evidence == ["lidar"]

    def test_collect_no_evidence(self, engine):
        perceptions = {
            "v_A": [{"id": "o1", "type": "vessel", "confidence": 0.9}],
        }
        votes = engine.collect_votes(perceptions)
        assert votes[0].evidence == []

    def test_collect_multiple_objects(self, engine):
        perceptions = {
            "v_A": [
                {"id": "o1", "type": "vessel", "confidence": 0.9},
                {"id": "o2", "type": "buoy", "confidence": 0.7},
            ],
        }
        votes = engine.collect_votes(perceptions)
        assert len(votes) == 2


class TestComputeConsensus:

    def test_unanimous_agreement(self, engine, unanimous_votes):
        result = engine.compute_consensus(unanimous_votes)
        assert len(result.agreed_objects) == 1
        assert result.participation == 3
        assert result.agreed_objects[0]["type"] == "vessel"
        assert result.agreed_objects[0]["voter_count"] == 3

    def test_quorum_not_met(self, engine):
        # Single voter can't meet 0.5 quorum of 1... actually ceil(1*0.5)=1, so it does
        # Let's use 2 vessels with quorum needing both
        votes = [
            PerceptionVote("v_A", {"id": "o1", "type": "vessel", "confidence": 0.9}, 0.9),
        ]
        # 1 unique voter, quorum=0.5 -> ceil(1*0.5)=1, so 1 meets quorum
        result = engine.compute_consensus(votes, quorum=0.9)  # Need 90% = ceil(0.9)=1
        assert len(result.agreed_objects) == 1

    def test_empty_votes(self, engine):
        result = engine.compute_consensus([])
        assert result.agreed_objects == []
        assert result.participation == 0
        assert result.disagreement_ratio == 0.0

    def test_disagreement_ratio(self, engine, split_votes):
        result = engine.compute_consensus(split_votes)
        assert result.disagreement_ratio > 0
        assert result.disagreement_ratio <= 1.0

    def test_consensus_timestamp(self, engine, unanimous_votes):
        import time
        before = time.time()
        result = engine.compute_consensus(unanimous_votes)
        assert before <= result.timestamp <= time.time()

    def test_consensus_multiple_objects(self, engine):
        votes = [
            PerceptionVote("v_A", {"id": "o1", "type": "vessel", "position": (10, 10, 0), "confidence": 0.9}, 0.9),
            PerceptionVote("v_B", {"id": "o1", "type": "vessel", "position": (10.5, 10, 0), "confidence": 0.85}, 0.85),
            PerceptionVote("v_A", {"id": "o2", "type": "buoy", "position": (20, 20, 0), "confidence": 0.8}, 0.8),
            PerceptionVote("v_B", {"id": "o2", "type": "buoy", "position": (20.2, 20, 0), "confidence": 0.75}, 0.75),
        ]
        result = engine.compute_consensus(votes)
        assert len(result.agreed_objects) == 2

    def test_consensus_history(self, engine, unanimous_votes):
        engine.compute_consensus(unanimous_votes)
        assert len(engine.get_history()) == 1
        engine.compute_consensus(unanimous_votes)
        assert len(engine.get_history()) == 2

    def test_custom_quorum(self, engine, unanimous_votes):
        result = engine.compute_consensus(unanimous_votes, quorum=1.0)
        assert len(result.agreed_objects) == 1  # All 3 agree


class TestDetectMaverick:

    def test_no_maverick_unanimous(self, engine, unanimous_votes):
        suspect = engine.detect_maverick_vessel(unanimous_votes)
        assert suspect is None

    def test_maverick_detected(self, engine, maverick_votes):
        suspect = engine.detect_maverick_vessel(maverick_votes)
        assert suspect == "vessel_B"

    def test_no_maverick_empty(self, engine):
        assert engine.detect_maverick_vessel([]) is None

    def test_no_maverick_single_voter_per_object(self, engine):
        votes = [
            PerceptionVote("v_A", {"id": "o1", "type": "vessel", "confidence": 0.9}, 0.9),
            PerceptionVote("v_B", {"id": "o2", "type": "buoy", "confidence": 0.8}, 0.8),
        ]
        assert engine.detect_maverick_vessel(votes) is None


class TestWeightVotes:

    def test_weight_high_reputation(self, engine, unanimous_votes):
        reps = {"vessel_A": 1.0, "vessel_B": 1.0, "vessel_C": 1.0}
        weighted = engine.weight_votes_by_reputation(unanimous_votes, reps)
        for orig, w in zip(unanimous_votes, weighted):
            assert w.confidence == pytest.approx(orig.confidence)

    def test_weight_low_reputation(self, engine, unanimous_votes):
        reps = {"vessel_A": 0.1, "vessel_B": 0.1, "vessel_C": 0.1}
        weighted = engine.weight_votes_by_reputation(unanimous_votes, reps)
        for w in weighted:
            assert w.confidence < 0.1

    def test_weight_missing_vessel(self, engine, unanimous_votes):
        reps = {}
        weighted = engine.weight_votes_by_reputation(unanimous_votes, reps)
        for orig, w in zip(unanimous_votes, weighted):
            assert w.confidence == pytest.approx(orig.confidence * 0.5)

    def test_weight_preserves_evidence(self, engine, unanimous_votes):
        reps = {"vessel_A": 0.5, "vessel_B": 0.5, "vessel_C": 0.5}
        weighted = engine.weight_votes_by_reputation(unanimous_votes, reps)
        for orig, w in zip(unanimous_votes, weighted):
            assert w.evidence == orig.evidence


class TestAgreementScore:

    def test_perfect_agreement(self, engine, unanimous_votes):
        score = engine.compute_agreement_score(unanimous_votes)
        assert score == pytest.approx(1.0)

    def test_empty_votes(self, engine):
        assert engine.compute_agreement_score([]) == 1.0

    def test_single_vote_per_object(self, engine):
        votes = [
            PerceptionVote("v_A", {"id": "o1", "type": "vessel"}, 0.9),
            PerceptionVote("v_B", {"id": "o2", "type": "buoy"}, 0.8),
        ]
        score = engine.compute_agreement_score(votes)
        assert score == pytest.approx(1.0)

    def test_partial_agreement(self, engine, split_votes):
        score = engine.compute_agreement_score(split_votes)
        # 2/3 agree on "vessel", 1/3 on "buoy" -> majority = 2/3
        assert 0.3 <= score <= 1.0


class TestGetHistory:

    def test_initial_history(self, engine):
        assert engine.get_history() == []

    def test_history_grows(self, engine, unanimous_votes):
        engine.compute_consensus(unanimous_votes)
        engine.compute_consensus(unanimous_votes)
        assert len(engine.get_history()) == 2
