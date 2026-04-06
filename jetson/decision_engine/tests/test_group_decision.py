"""Tests for group_decision.py — GroupDecisionMaker, Voter, GroupDecisionResult."""

import pytest

from jetson.decision_engine.group_decision import (
    Voter, GroupDecisionResult, GroupDecisionMaker,
)


# ============================================================
# Voter dataclass
# ============================================================

class TestVoter:
    def test_voter_defaults(self):
        v = Voter(id="v1", preferences=["a", "b", "c"])
        assert v.id == "v1"
        assert v.preferences == ["a", "b", "c"]
        assert v.weight == 1.0
        assert v.reliability == 1.0

    def test_voter_full(self):
        v = Voter(id="v2", preferences=["x"], weight=2.0, reliability=0.8)
        assert v.weight == pytest.approx(2.0)
        assert v.reliability == pytest.approx(0.8)


# ============================================================
# GroupDecisionResult dataclass
# ============================================================

class TestGroupDecisionResult:
    def test_defaults(self):
        r = GroupDecisionResult()
        assert r.selected_alternative is None
        assert r.consensus_level == 0.0
        assert r.individual_votes == {}
        assert r.dissenters == []

    def test_full(self):
        r = GroupDecisionResult(
            selected_alternative="a",
            consensus_level=0.8,
            individual_votes={"v1": "a", "v2": "a", "v3": "b"},
            dissenters=["v3"],
        )
        assert r.selected_alternative == "a"
        assert r.consensus_level == pytest.approx(0.8)


# ============================================================
# GroupDecisionMaker.plurality_vote
# ============================================================

class TestPluralityVote:
    def setup_method(self):
        self.maker = GroupDecisionMaker()

    def test_plurality_clear_winner(self):
        voters = [
            Voter(id="v1", preferences=["a", "b", "c"]),
            Voter(id="v2", preferences=["a", "b", "c"]),
            Voter(id="v3", preferences=["b", "a", "c"]),
        ]
        alts = ["a", "b", "c"]
        result = self.maker.plurality_vote(voters, alts)
        assert result.selected_alternative == "a"
        assert result.consensus_level == pytest.approx(2.0 / 3.0)

    def test_plurality_tie(self):
        voters = [
            Voter(id="v1", preferences=["a"]),
            Voter(id="v2", preferences=["b"]),
        ]
        result = self.maker.plurality_vote(voters, ["a", "b"])
        # One of them wins (Counter.most_common picks first)
        assert result.selected_alternative in ["a", "b"]

    def test_plurality_empty_voters(self):
        result = self.maker.plurality_vote([], ["a", "b"])
        assert result.selected_alternative is None

    def test_plurality_empty_alternatives(self):
        result = self.maker.plurality_vote(
            [Voter(id="v1", preferences=["a"])], []
        )
        assert result.selected_alternative is None

    def test_plurality_dissenters(self):
        voters = [
            Voter(id="v1", preferences=["a"]),
            Voter(id="v2", preferences=["a"]),
            Voter(id="v3", preferences=["b"]),
        ]
        result = self.maker.plurality_vote(voters, ["a", "b"])
        assert "v3" in result.dissenters

    def test_plurality_individual_votes(self):
        voters = [
            Voter(id="v1", preferences=["a"]),
            Voter(id="v2", preferences=["b"]),
        ]
        result = self.maker.plurality_vote(voters, ["a", "b"])
        assert result.individual_votes["v1"] == "a"
        assert result.individual_votes["v2"] == "b"

    def test_plurality_preference_not_in_alternatives(self):
        voters = [
            Voter(id="v1", preferences=["z", "a"]),
            Voter(id="v2", preferences=["a"]),
        ]
        result = self.maker.plurality_vote(voters, ["a", "b"])
        assert result.selected_alternative == "a"

    def test_plurality_consensus_unanimous(self):
        voters = [
            Voter(id="v1", preferences=["a"]),
            Voter(id="v2", preferences=["a"]),
            Voter(id="v3", preferences=["a"]),
        ]
        result = self.maker.plurality_vote(voters, ["a", "b"])
        assert result.consensus_level == pytest.approx(1.0)


# ============================================================
# GroupDecisionMaker.borda_count
# ============================================================

class TestBordaCount:
    def setup_method(self):
        self.maker = GroupDecisionMaker()

    def test_borda_clear_winner(self):
        voters = [
            Voter(id="v1", preferences=["a", "b", "c"]),
            Voter(id="v2", preferences=["a", "c", "b"]),
            Voter(id="v3", preferences=["b", "a", "c"]),
        ]
        result = self.maker.borda_count(voters, ["a", "b", "c"])
        assert result.selected_alternative == "a"

    def test_borda_empty(self):
        result = self.maker.borda_count([], ["a", "b"])
        assert result.selected_alternative is None

    def test_borda_single_voter(self):
        voters = [Voter(id="v1", preferences=["a", "b", "c"])]
        result = self.maker.borda_count(voters, ["a", "b", "c"])
        assert result.selected_alternative == "a"

    def test_borda_consensus(self):
        voters = [
            Voter(id="v1", preferences=["a", "b"]),
            Voter(id="v2", preferences=["a", "b"]),
        ]
        result = self.maker.borda_count(voters, ["a", "b"])
        assert result.consensus_level > 0

    def test_borda_with_weights(self):
        voters = [
            Voter(id="v1", preferences=["a", "b"], weight=2.0),
            Voter(id="v2", preferences=["b", "a"], weight=1.0),
        ]
        result = self.maker.borda_count(voters, ["a", "b"])
        assert result.selected_alternative == "a"

    def test_borda_dissenters(self):
        voters = [
            Voter(id="v1", preferences=["a", "b"]),
            Voter(id="v2", preferences=["a", "b"]),
            Voter(id="v3", preferences=["b", "a"]),
        ]
        result = self.maker.borda_count(voters, ["a", "b"])
        assert "v3" in result.dissenters

    def test_borda_empty_alternatives(self):
        voters = [Voter(id="v1", preferences=["a"])]
        result = self.maker.borda_count(voters, [])
        assert result.selected_alternative is None


# ============================================================
# GroupDecisionMaker.condorcet_method
# ============================================================

class TestCondorcetMethod:
    def setup_method(self):
        self.maker = GroupDecisionMaker()

    def test_condorcet_winner(self):
        voters = [
            Voter(id="v1", preferences=["a", "b", "c"]),
            Voter(id="v2", preferences=["a", "b", "c"]),
            Voter(id="v3", preferences=["a", "c", "b"]),
        ]
        result = self.maker.condorcet_method(voters, ["a", "b", "c"])
        assert result.selected_alternative == "a"

    def test_condorcet_no_winner_copeland(self):
        # Condorcet paradox: a > b > c > a
        voters = [
            Voter(id="v1", preferences=["a", "b", "c"]),
            Voter(id="v2", preferences=["b", "c", "a"]),
            Voter(id="v3", preferences=["c", "a", "b"]),
        ]
        result = self.maker.condorcet_method(voters, ["a", "b", "c"])
        # Should return one of them via Copeland
        assert result.selected_alternative in ["a", "b", "c"]

    def test_condorcet_empty(self):
        result = self.maker.condorcet_method([], ["a", "b"])
        assert result.selected_alternative is None

    def test_condorcet_single_voter(self):
        voters = [Voter(id="v1", preferences=["a", "b"])]
        result = self.maker.condorcet_method(voters, ["a", "b"])
        assert result.selected_alternative == "a"

    def test_condorcet_consensus(self):
        voters = [
            Voter(id="v1", preferences=["a", "b"]),
            Voter(id="v2", preferences=["a", "b"]),
            Voter(id="v3", preferences=["a", "b"]),
        ]
        result = self.maker.condorcet_method(voters, ["a", "b"])
        assert result.consensus_level > 0

    def test_condorcet_dissenters(self):
        voters = [
            Voter(id="v1", preferences=["a", "b"]),
            Voter(id="v2", preferences=["a", "b"]),
            Voter(id="v3", preferences=["b", "a"]),
        ]
        result = self.maker.condorcet_method(voters, ["a", "b"])
        assert "v3" in result.dissenters

    def test_condorcet_two_alternatives(self):
        voters = [
            Voter(id="v1", preferences=["a", "b"]),
            Voter(id="v2", preferences=["b", "a"]),
            Voter(id="v3", preferences=["a", "b"]),
        ]
        result = self.maker.condorcet_method(voters, ["a", "b"])
        assert result.selected_alternative == "a"


# ============================================================
# GroupDecisionMaker.compute_consensus
# ============================================================

class TestComputeConsensus:
    def setup_method(self):
        self.maker = GroupDecisionMaker()

    def test_consensus_unanimous(self):
        votes = {"v1": "a", "v2": "a", "v3": "a"}
        assert self.maker.compute_consensus(votes) == pytest.approx(1.0)

    def test_consensus_split(self):
        votes = {"v1": "a", "v2": "b", "v3": "a"}
        assert self.maker.compute_consensus(votes) == pytest.approx(2.0 / 3.0)

    def test_consensus_empty(self):
        assert self.maker.compute_consensus({}) == 0.0

    def test_consensus_single(self):
        assert self.maker.compute_consensus({"v1": "x"}) == pytest.approx(1.0)

    def test_consensus_all_different(self):
        votes = {"v1": "a", "v2": "b", "v3": "c"}
        assert self.maker.compute_consensus(votes) == pytest.approx(1.0 / 3.0)


# ============================================================
# GroupDecisionMaker.detect_voting_paradox
# ============================================================

class TestDetectVotingParadox:
    def setup_method(self):
        self.maker = GroupDecisionMaker()

    def test_paradox_detected(self):
        # Classic Condorcet paradox
        voters = [
            Voter(id="v1", preferences=["a", "b", "c"]),
            Voter(id="v2", preferences=["b", "c", "a"]),
            Voter(id="v3", preferences=["c", "a", "b"]),
        ]
        assert self.maker.detect_voting_paradox(voters, ["a", "b", "c"]) is True

    def test_no_paradox(self):
        voters = [
            Voter(id="v1", preferences=["a", "b", "c"]),
            Voter(id="v2", preferences=["a", "b", "c"]),
            Voter(id="v3", preferences=["a", "b", "c"]),
        ]
        assert self.maker.detect_voting_paradox(voters, ["a", "b", "c"]) is False

    def test_no_paradox_two_alternatives(self):
        voters = [
            Voter(id="v1", preferences=["a", "b"]),
            Voter(id="v2", preferences=["b", "a"]),
        ]
        assert self.maker.detect_voting_paradox(voters, ["a", "b"]) is False

    def test_no_paradox_single_alternative(self):
        voters = [Voter(id="v1", preferences=["a"])]
        assert self.maker.detect_voting_paradox(voters, ["a"]) is False

    def test_paradox_empty_voters(self):
        assert self.maker.detect_voting_paradox([], ["a", "b", "c"]) is False


# ============================================================
# GroupDecisionMaker.weighted_voting
# ============================================================

class TestWeightedVoting:
    def setup_method(self):
        self.maker = GroupDecisionMaker()

    def test_weighted_heavier_voter_wins(self):
        voters = [
            Voter(id="v1", preferences=["a"], weight=3.0),
            Voter(id="v2", preferences=["b"], weight=1.0),
        ]
        result = self.maker.weighted_voting(voters, ["a", "b"])
        assert result.selected_alternative == "a"

    def test_weighted_empty(self):
        result = self.maker.weighted_voting([], ["a", "b"])
        assert result.selected_alternative is None

    def test_weighted_with_reliability(self):
        voters = [
            Voter(id="v1", preferences=["a"], weight=2.0, reliability=1.0),
            Voter(id="v2", preferences=["b"], weight=2.0, reliability=0.4),
        ]
        result = self.maker.weighted_voting(voters, ["a", "b"])
        assert result.selected_alternative == "a"

    def test_weighted_consensus(self):
        voters = [
            Voter(id="v1", preferences=["a"], weight=1.0, reliability=1.0),
            Voter(id="v2", preferences=["a"], weight=1.0, reliability=1.0),
            Voter(id="v3", preferences=["b"], weight=1.0, reliability=0.5),
        ]
        result = self.maker.weighted_voting(voters, ["a", "b"])
        assert result.consensus_level > 0
        assert result.selected_alternative == "a"

    def test_weighted_dissenters(self):
        voters = [
            Voter(id="v1", preferences=["a"], weight=2.0),
            Voter(id="v2", preferences=["b"], weight=1.0),
        ]
        result = self.maker.weighted_voting(voters, ["a", "b"])
        assert "v2" in result.dissenters

    def test_weighted_empty_alternatives(self):
        voters = [Voter(id="v1", preferences=["a"])]
        result = self.maker.weighted_voting(voters, [])
        assert result.selected_alternative is None

    def test_weighted_equal_weights_same_as_plurality(self):
        voters = [
            Voter(id="v1", preferences=["a", "b"]),
            Voter(id="v2", preferences=["b", "a"]),
            Voter(id="v3", preferences=["a", "b"]),
        ]
        w_result = self.maker.weighted_voting(voters, ["a", "b"])
        p_result = self.maker.plurality_vote(voters, ["a", "b"])
        assert w_result.selected_alternative == p_result.selected_alternative
