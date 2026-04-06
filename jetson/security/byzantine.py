"""Byzantine fault tolerance — 3f+1 redundancy consensus."""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Vote:
    voter_id: str
    proposal: str
    signature: str
    timestamp: float
    agree: bool = True


@dataclass
class ConsensusResult:
    decision: bool  # True = proposal accepted
    participating_nodes: List[str] = field(default_factory=list)
    dissenters: List[str] = field(default_factory=list)
    round_number: int = 0
    vote_count: int = 0


class ConsensusState(Enum):
    IDLE = "idle"
    PROPOSING = "proposing"
    VOTING = "voting"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass
class Proposal:
    proposer_id: str
    value: str
    round_number: int
    timestamp: float
    signature: str = ""


@dataclass
class SuspectNode:
    node_id: str
    reason: str
    evidence_count: int = 0


class ByzantineConsensus:
    """Byzantine fault tolerant consensus using 3f+1 redundancy."""

    def __init__(self, node_id: str, total_nodes: int) -> None:
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = self.max_faults(total_nodes)
        self._state = ConsensusState.IDLE
        self._current_round = 0
        self._proposal: Optional[Proposal] = None
        self._votes: Dict[str, Vote] = {}
        self._suspects: Dict[str, SuspectNode] = {}
        self._history: List[ConsensusResult] = []
        self._node_keys: Dict[str, bytes] = {}

    @staticmethod
    def max_faults(total_nodes: int) -> int:
        """Calculate max tolerable Byzantine faults: f = floor((n-1)/3)."""
        return (total_nodes - 1) // 3

    @staticmethod
    def is_quorum(vote_count: int, total_nodes: int) -> bool:
        """Check if votes form a quorum (2f+1 or more)."""
        f = ByzantineConsensus.max_faults(total_nodes)
        return vote_count >= (2 * f + 1)

    def set_node_key(self, node_id: str, key: bytes) -> None:
        self._node_keys[node_id] = key

    def propose(self, node_id: str, value: str) -> Proposal:
        """Create a new proposal."""
        self._current_round += 1
        self._state = ConsensusState.PROPOSING
        proposal = Proposal(
            proposer_id=node_id,
            value=value,
            round_number=self._current_round,
            timestamp=time.time(),
            signature="",
        )
        self._proposal = proposal
        return proposal

    def vote(self, node_id: str, proposal: str, agree: bool) -> Vote:
        """Cast a vote on a proposal."""
        self._state = ConsensusState.VOTING
        v = Vote(
            voter_id=node_id,
            proposal=proposal,
            signature="",
            timestamp=time.time(),
            agree=agree,
        )
        self._votes[node_id] = v
        return v

    def count_votes(
        self, votes: Optional[List[Vote]] = None, total_nodes: Optional[int] = None
    ) -> ConsensusResult:
        """Count votes and determine consensus."""
        if votes is None:
            votes = list(self._votes.values())
        if total_nodes is None:
            total_nodes = self.total_nodes

        agree_count = sum(1 for v in votes if v.agree)
        disagree_count = sum(1 for v in votes if not v.agree)
        total_votes = len(votes)

        quorum = self.is_quorum(total_votes, total_nodes)
        f = self.max_faults(total_nodes)

        # Need 2f+1 agree votes for consensus
        threshold = 2 * f + 1
        decision = quorum and agree_count >= threshold

        participating = [v.voter_id for v in votes]
        dissenters = [v.voter_id for v in votes if not v.agree]

        result = ConsensusResult(
            decision=decision,
            participating_nodes=participating,
            dissenters=dissenters,
            round_number=self._current_round,
            vote_count=total_votes,
        )

        if decision:
            self._state = ConsensusState.COMMITTED
        elif total_votes >= total_nodes:
            self._state = ConsensusState.ABORTED

        self._history.append(result)
        return result

    def detect_byzantine(
        self, votes: Optional[List[Vote]] = None, threshold: int = 3
    ) -> List[SuspectNode]:
        """Detect potentially Byzantine nodes based on voting patterns."""
        if votes is None:
            votes = list(self._votes.values())

        # Count conflicting votes (same node voting differently in same round)
        proposal_counts: Dict[str, Counter] = {}
        for v in votes:
            if v.voter_id not in proposal_counts:
                proposal_counts[v.voter_id] = Counter()
            proposal_counts[v.voter_id][v.proposal] += 1

        suspects: List[SuspectNode] = []
        for node_id, counts in proposal_counts.items():
            if len(counts) > 1:
                # Node voted for different proposals — suspicious
                if node_id not in self._suspects:
                    self._suspects[node_id] = SuspectNode(
                        node_id=node_id,
                        reason="conflicting_votes",
                        evidence_count=len(counts),
                    )
                else:
                    self._suspects[node_id].evidence_count += 1

        # Also flag nodes that dissent but were alone
        for v in votes:
            if not v.agree:
                agree_from_others = sum(
                    1 for vv in votes if vv.voter_id != v.voter_id and vv.agree
                )
                if agree_from_others == len(votes) - 1:
                    # Only dissenter — might be Byzantine
                    if v.voter_id not in self._suspects:
                        self._suspects[v.voter_id] = SuspectNode(
                            node_id=v.voter_id,
                            reason="sole_dissenter",
                            evidence_count=1,
                        )
                    else:
                        self._suspects[v.voter_id].evidence_count += 1

        suspects = [s for s in self._suspects.values() if s.evidence_count >= threshold]
        return suspects

    def get_state(self) -> ConsensusState:
        return self._state

    def get_current_proposal(self) -> Optional[Proposal]:
        return self._proposal

    def reset(self) -> None:
        self._state = ConsensusState.IDLE
        self._current_round = 0
        self._proposal = None
        self._votes.clear()
        self._suspects.clear()

    def get_history(self) -> List[ConsensusResult]:
        return list(self._history)
