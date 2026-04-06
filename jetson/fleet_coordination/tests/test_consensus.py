"""Tests for consensus module — ConsensusProtocol, Proposal, Vote, ConsensusResult."""

import hashlib

import pytest

from jetson.fleet_coordination.consensus import (
    ConsensusProtocol,
    ConsensusResult,
    Proposal,
    Vote,
)
from jetson.fleet_coordination.fleet_manager import VesselStatus


# ────────────────────────────────────────────────────── fixtures

@pytest.fixture
def protocol():
    return ConsensusProtocol()


@pytest.fixture
def sample_vessels():
    return [
        VesselStatus(vessel_id=f"V{i}", trust_score=0.9) for i in range(5)
    ]


@pytest.fixture
def sample_proposal():
    return Proposal(proposer="V0", value="go_north", round_number=1)


# ────────────────────────────────────────────────────── Proposal

class TestProposal:
    def test_default_proposal(self):
        p = Proposal()
        assert p.proposer == ""
        assert p.value is None
        assert p.round_number == 0

    def test_proposal_with_values(self):
        p = Proposal(proposer="V0", value="test", round_number=3)
        assert p.proposer == "V0"
        assert p.value == "test"
        assert p.round_number == 3

    def test_proposal_auto_id(self):
        p1 = Proposal()
        p2 = Proposal()
        assert p1.id != p2.id


# ────────────────────────────────────────────────────── Vote

class TestVote:
    def test_default_vote(self):
        v = Vote()
        assert v.voter_id == ""
        assert v.accept is False
        assert v.justification == ""

    def test_accept_vote(self):
        v = Vote(voter_id="V1", proposal_id="P1", accept=True, justification="looks good")
        assert v.accept is True
        assert v.justification == "looks good"


# ────────────────────────────────────────────────────── ConsensusResult

class TestConsensusResult:
    def test_default(self):
        r = ConsensusResult()
        assert r.agreed_value is None
        assert r.consensus_reached is False
        assert r.participating_nodes == []

    def test_with_values(self):
        r = ConsensusResult(
            agreed_value=42,
            participating_nodes=["V0", "V1", "V2"],
            dissenters=["V3"],
            round_number=5,
            consensus_reached=True,
        )
        assert r.agreed_value == 42
        assert len(r.participating_nodes) == 3
        assert r.consensus_reached is True


# ────────────────────────────────────────────────────── Raft election

class TestRaftElect:
    def test_elect_returns_leader(self, protocol, sample_vessels):
        leader, term = protocol.raft_elect("V0", sample_vessels)
        assert leader is not None
        assert term >= 1

    def test_elect_increments_term(self, protocol, sample_vessels):
        _, t1 = protocol.raft_elect("V0", sample_vessels)
        _, t2 = protocol.raft_elect("V0", sample_vessels)
        assert t2 > t1

    def test_elect_empty_vessels(self, protocol):
        leader, term = protocol.raft_elect("V0", [])
        assert leader is None
        assert term >= 1

    def test_elect_single_vessel(self, protocol):
        v = VesselStatus(vessel_id="ONLY", trust_score=1.0)
        leader, _ = protocol.raft_elect("ONLY", [v])
        assert leader == "ONLY"

    def test_elect_sets_leader(self, protocol, sample_vessels):
        protocol.raft_elect("V0", sample_vessels)
        assert protocol.leader_id is not None

    def test_elect_leader_in_vessels(self, protocol, sample_vessels):
        ids = {v.vessel_id for v in sample_vessels}
        leader, _ = protocol.raft_elect("V0", sample_vessels)
        assert leader in ids


# ────────────────────────────────────────────────────── Paxos

class TestPaxos:
    def test_prepare_returns_promise_count(self, protocol, sample_proposal):
        acceptors = ["A1", "A2", "A3"]
        count = protocol.paxos_prepare(sample_proposal, acceptors)
        assert isinstance(count, int)
        assert 0 <= count <= len(acceptors)

    def test_prepare_empty_acceptors(self, protocol, sample_proposal):
        assert protocol.paxos_prepare(sample_proposal, []) == 0

    def test_prepare_single_acceptor(self, protocol, sample_proposal):
        count = protocol.paxos_prepare(sample_proposal, ["A1"])
        assert count in (0, 1)

    def test_accept_with_promises(self, protocol, sample_proposal):
        promises = 3
        result = protocol.paxos_accept(sample_proposal, promises)
        assert isinstance(result, bool)

    def test_accept_zero_promises_fails(self, protocol, sample_proposal):
        assert protocol.paxos_accept(sample_proposal, 0) is False

    def test_accept_adds_to_log(self, protocol, sample_proposal):
        # Retry until accepted (probabilistic)
        for _ in range(50):
            protocol.paxos_accept(sample_proposal, 5)
        log = protocol.log
        if log:
            assert log[0].proposer == sample_proposal.proposer

    def test_paxos_round_number_preserved(self, protocol, sample_proposal):
        protocol.paxos_accept(sample_proposal, 5)
        for entry in protocol.log:
            assert entry.round_number == sample_proposal.round_number


# ────────────────────────────────────────────────────── Raft propose

class TestRaftPropose:
    def test_propose_returns_result(self, protocol, sample_vessels):
        result = protocol.raft_propose("go_north", "V0", sample_vessels)
        assert isinstance(result, ConsensusResult)

    def test_propose_leader_participates(self, protocol, sample_vessels):
        result = protocol.raft_propose("go_north", "V0", sample_vessels)
        assert "V0" in result.participating_nodes

    def test_propose_dissenters_list(self, protocol, sample_vessels):
        result = protocol.raft_propose("go_north", "V0", sample_vessels)
        # Dissenters + participants should cover all vessels
        all_ids = {v.vessel_id for v in sample_vessels}
        covered = set(result.participating_nodes) | set(result.dissenters)
        assert covered == all_ids

    def test_propose_round_number(self, protocol, sample_vessels):
        protocol.raft_elect("V0", sample_vessels)
        result = protocol.raft_propose("go_north", "V0", sample_vessels)
        assert result.round_number >= 1

    def test_propose_empty_followers(self, protocol):
        result = protocol.raft_propose("go_north", "V0", [])
        assert result.consensus_reached is True  # leader alone is majority

    def test_propose_agreed_value(self, protocol, sample_vessels):
        # Retry multiple times to get consensus
        for _ in range(20):
            result = protocol.raft_propose("go_north", "V0", sample_vessels)
            if result.consensus_reached:
                assert result.agreed_value == "go_north"
                break


# ────────────────────────────────────────────────────── Split brain

class TestSplitBrain:
    def test_no_split_single_partition(self, protocol):
        assert protocol.detect_split_brain([["V0", "V1", "V2"]]) is False

    def test_split_brain_detected(self, protocol):
        assert protocol.detect_split_brain([["V0"], ["V1", "V2"]]) is True

    def test_three_way_split(self, protocol):
        assert protocol.detect_split_brain([["V0"], ["V1"], ["V2"]]) is True

    def test_empty_partitions(self, protocol):
        assert protocol.detect_split_brain([[], []]) is False

    def test_one_partition_with_empty(self, protocol):
        assert protocol.detect_split_brain([["V0"], []]) is False

    def test_many_partitions_split(self, protocol):
        partitions = [[f"V{i}"] for i in range(5)]
        assert protocol.detect_split_brain(partitions) is True


# ────────────────────────────────────────────────────── Merkle

class TestMerkleTree:
    def test_hash_empty_state(self, protocol):
        h = protocol.merkle_tree_hash({})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_hash_simple_state(self, protocol):
        h = protocol.merkle_tree_hash({"fuel": 80, "speed": 5})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_deterministic(self, protocol):
        state = {"a": 1, "b": 2, "c": 3}
        h1 = protocol.merkle_tree_hash(state)
        h2 = protocol.merkle_tree_hash(state)
        assert h1 == h2

    def test_hash_different_states_differ(self, protocol):
        h1 = protocol.merkle_tree_hash({"x": 1})
        h2 = protocol.merkle_tree_hash({"x": 2})
        assert h1 != h2

    def test_hash_nested_dict(self, protocol):
        state = {"outer": {"inner": 42}}
        h = protocol.merkle_tree_hash(state)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_order_invariant(self, protocol):
        h1 = protocol.merkle_tree_hash({"a": 1, "b": 2})
        h2 = protocol.merkle_tree_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_known_hash_empty(self, protocol):
        h = protocol.merkle_tree_hash({})
        expected = hashlib.sha256(b"empty").hexdigest()
        assert h == expected


# ────────────────────────────────────────────────────── Reset / accessors

class TestProtocolAccessors:
    def test_initial_term(self, protocol):
        assert protocol.current_term == 0

    def test_initial_leader_none(self, protocol):
        assert protocol.leader_id is None

    def test_initial_log_empty(self, protocol):
        assert protocol.log == []

    def test_reset(self, protocol, sample_vessels):
        protocol.raft_elect("V0", sample_vessels)
        protocol.paxos_accept(Proposal(value="x"), 3)
        protocol.reset()
        assert protocol.current_term == 0
        assert protocol.leader_id is None
        assert protocol.log == []
