"""Tests for byzantine module."""

import pytest
from jetson.security.byzantine import (
    ByzantineConsensus,
    ConsensusResult,
    ConsensusState,
    Proposal,
    SuspectNode,
    Vote,
)


# ── Vote ────────────────────────────────────────────────────────────

class TestVote:
    def test_create_agree(self):
        v = Vote(voter_id="n1", proposal="go", signature="sig", timestamp=1.0, agree=True)
        assert v.voter_id == "n1"
        assert v.agree is True

    def test_create_disagree(self):
        v = Vote(voter_id="n2", proposal="go", signature="sig", timestamp=1.0, agree=False)
        assert v.agree is False


# ── ConsensusResult ────────────────────────────────────────────────

class TestConsensusResult:
    def test_default(self):
        r = ConsensusResult(decision=True)
        assert r.decision is True
        assert r.participating_nodes == []
        assert r.dissenters == []
        assert r.round_number == 0

    def test_with_data(self):
        r = ConsensusResult(
            decision=False,
            participating_nodes=["a", "b", "c"],
            dissenters=["c"],
            round_number=3,
            vote_count=3,
        )
        assert len(r.dissenters) == 1
        assert r.vote_count == 3


# ── ByzantineConsensus ─────────────────────────────────────────────

class TestByzantineConsensusConstruction:
    def test_construct(self):
        bc = ByzantineConsensus("node_0", 4)
        assert bc.node_id == "node_0"
        assert bc.total_nodes == 4
        assert bc.get_state() == ConsensusState.IDLE
        assert bc.get_current_proposal() is None

    def test_initial_round(self):
        bc = ByzantineConsensus("n", 4)
        assert bc._current_round == 0


class TestMaxFaults:
    def test_4_nodes(self):
        assert ByzantineConsensus.max_faults(4) == 1

    def test_7_nodes(self):
        assert ByzantineConsensus.max_faults(7) == 2

    def test_3_nodes(self):
        assert ByzantineConsensus.max_faults(3) == 0

    def test_1_node(self):
        assert ByzantineConsensus.max_faults(1) == 0

    def test_10_nodes(self):
        assert ByzantineConsensus.max_faults(10) == 3

    def test_13_nodes(self):
        assert ByzantineConsensus.max_faults(13) == 4

    def test_instance_faults(self):
        bc = ByzantineConsensus("n", 7)
        assert bc.f == 2


class TestIsQuorum:
    def test_quorum_4_nodes(self):
        assert ByzantineConsensus.is_quorum(3, 4) is True

    def test_no_quorum_4_nodes(self):
        assert ByzantineConsensus.is_quorum(2, 4) is False

    def test_exact_quorum_4_nodes(self):
        # 2f+1 = 3 for n=4
        assert ByzantineConsensus.is_quorum(3, 4) is True

    def test_all_nodes(self):
        assert ByzantineConsensus.is_quorum(4, 4) is True

    def test_zero_votes(self):
        assert ByzantineConsensus.is_quorum(0, 4) is False

    def test_quorum_7_nodes(self):
        # f=2, need 2*2+1=5
        assert ByzantineConsensus.is_quorum(5, 7) is True
        assert ByzantineConsensus.is_quorum(4, 7) is False


class TestPropose:
    def test_propose_increments_round(self):
        bc = ByzantineConsensus("n", 4)
        bc.propose("n", "deploy")
        assert bc._current_round == 1

    def test_propose_state(self):
        bc = ByzantineConsensus("n", 4)
        bc.propose("n", "action")
        assert bc.get_state() == ConsensusState.PROPOSING

    def test_propose_returns_proposal(self):
        bc = ByzantineConsensus("n", 4)
        p = bc.propose("n", "go")
        assert isinstance(p, Proposal)
        assert p.proposer_id == "n"
        assert p.value == "go"

    def test_multiple_proposals(self):
        bc = ByzantineConsensus("n", 4)
        bc.propose("n", "a")
        bc.propose("n", "b")
        assert bc._current_round == 2

    def test_proposal_stored(self):
        bc = ByzantineConsensus("n", 4)
        p = bc.propose("n", "x")
        assert bc.get_current_proposal().value == "x"


class TestVote:
    def test_vote_returned(self):
        bc = ByzantineConsensus("n", 4)
        bc.propose("n", "go")
        v = bc.vote("n1", "go", True)
        assert isinstance(v, Vote)
        assert v.voter_id == "n1"
        assert v.agree is True

    def test_vote_state(self):
        bc = ByzantineConsensus("n", 4)
        bc.propose("n", "go")
        bc.vote("n1", "go", True)
        assert bc.get_state() == ConsensusState.VOTING

    def test_vote_disagree(self):
        bc = ByzantineConsensus("n", 4)
        v = bc.vote("n1", "stop", False)
        assert v.agree is False


class TestCountVotes:
    def test_unanimous_agree_consensus(self):
        bc = ByzantineConsensus("n", 4)
        votes = [
            Vote("n0", "go", "", 1.0, True),
            Vote("n1", "go", "", 1.0, True),
            Vote("n2", "go", "", 1.0, True),
            Vote("n3", "go", "", 1.0, True),
        ]
        result = bc.count_votes(votes)
        assert result.decision is True
        assert result.vote_count == 4

    def test_majority_agree_consensus(self):
        bc = ByzantineConsensus("n", 4)
        votes = [
            Vote("n0", "go", "", 1.0, True),
            Vote("n1", "go", "", 1.0, True),
            Vote("n2", "go", "", 1.0, True),
            Vote("n3", "go", "", 1.0, False),
        ]
        result = bc.count_votes(votes)
        assert result.decision is True  # 3 >= 2*1+1=3
        assert len(result.dissenters) == 1

    def test_no_consensus(self):
        bc = ByzantineConsensus("n", 4)
        votes = [
            Vote("n0", "go", "", 1.0, True),
            Vote("n1", "go", "", 1.0, False),
            Vote("n2", "go", "", 1.0, False),
        ]
        result = bc.count_votes(votes)
        assert result.decision is False

    def test_committed_state(self):
        bc = ByzantineConsensus("n", 4)
        votes = [Vote(f"n{i}", "go", "", 1.0, True) for i in range(4)]
        bc.count_votes(votes)
        assert bc.get_state() == ConsensusState.COMMITTED

    def test_participating_nodes(self):
        bc = ByzantineConsensus("n", 4)
        votes = [Vote(f"n{i}", "go", "", 1.0, True) for i in range(3)]
        result = bc.count_votes(votes)
        assert set(result.participating_nodes) == {"n0", "n1", "n2"}

    def test_count_votes_uses_instance(self):
        bc = ByzantineConsensus("n", 4)
        bc.vote("n0", "go", True)
        bc.vote("n1", "go", True)
        bc.vote("n2", "go", True)
        result = bc.count_votes()
        assert result.vote_count == 3


class TestDetectByzantine:
    def test_no_suspects_uniform(self):
        bc = ByzantineConsensus("n", 4)
        votes = [Vote(f"n{i}", "go", "", 1.0, True) for i in range(4)]
        suspects = bc.detect_byzantine(votes)
        # No conflicting votes -> no suspects (evidence < threshold 3)
        assert len(suspects) == 0

    def test_sole_dissenter_flagged(self):
        bc = ByzantineConsensus("n", 4)
        votes = [
            Vote("n0", "go", "", 1.0, True),
            Vote("n1", "go", "", 1.0, True),
            Vote("n2", "go", "", 1.0, True),
            Vote("n3", "go", "", 1.0, False),
        ]
        suspects = bc.detect_byzantine(votes, threshold=1)
        assert len(suspects) >= 1
        ids = {s.node_id for s in suspects}
        assert "n3" in ids

    def test_conflicting_votes_detected(self):
        bc = ByzantineConsensus("n", 4)
        votes = [
            Vote("n0", "go", "", 1.0, True),
            Vote("n0", "stop", "", 1.0, True),  # same node, different proposal
            Vote("n1", "go", "", 1.0, True),
        ]
        suspects = bc.detect_byzantine(votes, threshold=1)
        ids = {s.node_id for s in suspects}
        assert "n0" in ids

    def test_threshold_filters(self):
        bc = ByzantineConsensus("n", 4)
        votes = [
            Vote("n0", "go", "", 1.0, True),
            Vote("n1", "go", "", 1.0, True),
            Vote("n2", "go", "", 1.0, True),
            Vote("n3", "go", "", 1.0, False),
        ]
        # With high threshold, no suspects
        suspects = bc.detect_byzantine(votes, threshold=100)
        assert len(suspects) == 0

    def test_multiple_dissenters(self):
        bc = ByzantineConsensus("n", 7)
        votes = [Vote(f"n{i}", "go", "", 1.0, True) for i in range(6)]
        votes.append(Vote("n6", "go", "", 1.0, False))  # sole dissenter
        suspects = bc.detect_byzantine(votes, threshold=1)
        ids = {s.node_id for s in suspects}
        assert "n6" in ids


class TestReset:
    def test_reset_clears_state(self):
        bc = ByzantineConsensus("n", 4)
        bc.propose("n", "go")
        bc.vote("n0", "go", True)
        bc.reset()
        assert bc.get_state() == ConsensusState.IDLE
        assert bc._current_round == 0
        assert bc.get_current_proposal() is None

    def test_reset_allows_new_proposal(self):
        bc = ByzantineConsensus("n", 4)
        bc.propose("n", "a")
        bc.reset()
        p = bc.propose("n", "b")
        assert p.value == "b"
        assert bc._current_round == 1


class TestHistory:
    def test_history_empty(self):
        bc = ByzantineConsensus("n", 4)
        assert bc.get_history() == []

    def test_history_records(self):
        bc = ByzantineConsensus("n", 4)
        bc.vote("n0", "go", True)
        bc.vote("n1", "go", True)
        bc.count_votes()
        assert len(bc.get_history()) == 1

    def test_history_multiple_rounds(self):
        bc = ByzantineConsensus("n", 4)
        bc.vote("n0", "a", True)
        bc.vote("n1", "a", True)
        bc.count_votes()
        bc.reset()
        bc.vote("n0", "b", True)
        bc.vote("n1", "b", True)
        bc.count_votes()
        assert len(bc.get_history()) == 2


# ── Proposal ────────────────────────────────────────────────────────

class TestProposal:
    def test_create(self):
        p = Proposal(proposer_id="n0", value="go", round_number=1, timestamp=1.0)
        assert p.proposer_id == "n0"
        assert p.signature == ""

    def test_with_signature(self):
        p = Proposal(proposer_id="n0", value="go", round_number=1, timestamp=1.0, signature="sig")
        assert p.signature == "sig"


# ── SuspectNode ─────────────────────────────────────────────────────

class TestSuspectNode:
    def test_create(self):
        s = SuspectNode(node_id="n_bad", reason="conflicting_votes", evidence_count=3)
        assert s.node_id == "n_bad"
        assert s.reason == "conflicting_votes"
        assert s.evidence_count == 3

    def test_default_evidence(self):
        s = SuspectNode(node_id="n", reason="test")
        assert s.evidence_count == 0
