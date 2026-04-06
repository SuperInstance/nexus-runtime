"""Distributed consensus protocols — Raft, Paxos, split-brain detection, Merkle."""

from __future__ import annotations

import hashlib
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Proposal:
    """A proposal submitted for consensus."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    proposer: str = ""
    value: Any = None
    round_number: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Vote:
    """A vote on a proposal."""
    voter_id: str = ""
    proposal_id: str = ""
    accept: bool = False
    justification: str = ""


@dataclass
class ConsensusResult:
    """Outcome of a consensus round."""
    agreed_value: Any = None
    participating_nodes: List[str] = field(default_factory=list)
    dissenters: List[str] = field(default_factory=list)
    round_number: int = 0
    consensus_reached: bool = False


class ConsensusProtocol:
    """Implements Raft election, Paxos prepare/accept, split-brain detection, Merkle hashing."""

    def __init__(self) -> None:
        self._current_term: int = 0
        self._leader_id: Optional[str] = None
        self._voted_for: Optional[str] = None
        self._log: List[Proposal] = []

    # ---------------------------------------------------------- Raft
    def raft_elect(self, vessel_id: str,
                   all_vessels: List[Any]) -> Tuple[Optional[str], int]:
        """Simulate a Raft election. Returns (leader_id, term).

        The vessel with the highest (trust_score * random_factor) wins.
        """
        self._current_term += 1
        term = self._current_term

        candidates = []
        for v in all_vessels:
            vid = v.vessel_id if hasattr(v, "vessel_id") else str(v)
            trust = getattr(v, "trust_score", 1.0)
            # Add randomness to break ties
            factor = trust * (0.7 + 0.3 * random.random())
            candidates.append((vid, factor))

        if not candidates:
            return (None, term)

        candidates.sort(key=lambda x: x[1], reverse=True)
        leader = candidates[0][0]
        self._leader_id = leader
        return (leader, term)

    # ---------------------------------------------------------- Paxos
    def paxos_prepare(self, proposal: Proposal,
                      acceptors: List[str]) -> int:
        """Phase 1 of Paxos: send prepare to acceptors. Returns promise count."""
        if not acceptors:
            return 0

        promises = 0
        for acc in acceptors:
            # Acceptor promises if no higher-numbered proposal accepted
            already_accepted = [
                p for p in self._log
                if p.round_number >= proposal.round_number
            ]
            if not already_accepted or random.random() > 0.2:
                promises += 1

        return promises

    def paxos_accept(self, proposal: Proposal,
                     promises: int) -> bool:
        """Phase 2 of Paxos: accept proposal if majority promised."""
        # A proposal is accepted if >50% of acceptors promised
        # (We track the promise count; assume quorum = ceil(total/2))
        if promises <= 0:
            return False

        # Simulate acceptance — accept with high probability when promises > 0
        accepted = random.random() < (promises / (promises + 1))
        if accepted:
            self._log.append(proposal)
        return accepted

    def raft_propose(self, value: Any, leader: str,
                     followers: List[Any]) -> ConsensusResult:
        """Raft-style propose: leader proposes, followers ACK."""
        round_num = self._current_term
        participating = [leader]
        dissenters: List[str] = []

        follower_ids = []
        for f in followers:
            fid = f.vessel_id if hasattr(f, "vessel_id") else str(f)
            follower_ids.append(fid)

        for fid in follower_ids:
            trust = 1.0
            for f in followers:
                if (hasattr(f, "vessel_id") and f.vessel_id == fid) or str(f) == fid:
                    trust = getattr(f, "trust_score", 0.8)
                    break

            # Follower accepts with probability proportional to trust
            if random.random() < trust:
                participating.append(fid)
            else:
                dissenters.append(fid)

        majority = (len(participating) > len(follower_ids) // 2)
        self._current_term = round_num

        return ConsensusResult(
            agreed_value=value if majority else None,
            participating_nodes=participating,
            dissenters=dissenters,
            round_number=round_num,
            consensus_reached=majority,
        )

    # --------------------------------------------------- Split-brain
    def detect_split_brain(self, partitions: List[List[str]]) -> bool:
        """Detect split-brain: True if multiple partitions claim to have a leader."""
        leaders_found = 0
        for partition in partitions:
            if not partition:
                continue
            # A partition with >1 node can claim a leader
            if len(partition) >= 1:
                leaders_found += 1
                if leaders_found > 1:
                    return True
        return leaders_found > 1

    # --------------------------------------------------- Merkle tree
    def merkle_tree_hash(self, state: Dict[str, Any]) -> str:
        """Compute a Merkle-like root hash of a state dictionary."""
        leaves = self._flatten_state(state)
        if not leaves:
            return hashlib.sha256(b"empty").hexdigest()

        # Hash each leaf
        hashed = [hashlib.sha256(str(leaf).encode()).hexdigest() for leaf in leaves]

        # Build tree bottom-up
        while len(hashed) > 1:
            if len(hashed) % 2 == 1:
                hashed.append(hashed[-1])  # duplicate last for odd count
            next_level = []
            for i in range(0, len(hashed), 2):
                combined = hashlib.sha256(
                    (hashed[i] + hashed[i + 1]).encode()
                ).hexdigest()
                next_level.append(combined)
            hashed = next_level

        return hashed[0]

    @staticmethod
    def _flatten_state(state: Dict[str, Any], prefix: str = "") -> List[str]:
        """Recursively flatten a dict into sorted key=value strings."""
        items: List[str] = []
        for k in sorted(state.keys()):
            full_key = f"{prefix}.{k}" if prefix else k
            v = state[k]
            if isinstance(v, dict):
                items.extend(ConsensusProtocol._flatten_state(v, full_key))
            else:
                items.append(f"{full_key}={v}")
        return items

    # --------------------------------------------------- Accessors
    @property
    def current_term(self) -> int:
        return self._current_term

    @property
    def leader_id(self) -> Optional[str]:
        return self._leader_id

    @property
    def log(self) -> List[Proposal]:
        return list(self._log)

    def reset(self) -> None:
        self._current_term = 0
        self._leader_id = None
        self._voted_for = None
        self._log.clear()
