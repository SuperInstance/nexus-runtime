"""Consensus-based perception for multi-vessel agreement.

Implements voting-based consensus for perceived objects across multiple
vessels, including quorum management, maverick detection, reputation-weighted
voting, and agreement scoring.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class PerceptionVote:
    """A single vessel's vote on perceived objects."""
    voter_id: str
    perceived_object: dict
    confidence: float
    evidence: List[str] = field(default_factory=list)

    def object_id(self) -> str:
        return self.perceived_object.get("id", "unknown")

    def object_type(self) -> str:
        return self.perceived_object.get("type", "unknown")

    def object_position(self) -> Tuple[float, float, float]:
        return self.perceived_object.get("position", (0.0, 0.0, 0.0))


@dataclass
class PerceptionConsensus:
    """Result of a consensus round."""
    agreed_objects: List[dict]
    participation: int
    disagreement_ratio: float
    timestamp: float


class PerceptionConsensusEngine:
    """Engine for computing perception consensus across vessels."""

    def __init__(self, quorum: float = 0.5):
        """Initialize with quorum threshold.

        Args:
            quorum: fraction of participating vessels required for consensus.
        """
        self.quorum = quorum
        self._history: List[PerceptionConsensus] = []

    def collect_votes(
        self, vessels_perceptions: Dict[str, List[dict]]
    ) -> List[PerceptionVote]:
        """Collect perception votes from multiple vessels.

        Args:
            vessels_perceptions: mapping of vessel_id -> list of perceived object dicts.

        Returns:
            List of PerceptionVote instances.
        """
        votes: List[PerceptionVote] = []
        for vessel_id, objects in vessels_perceptions.items():
            for obj in objects:
                votes.append(PerceptionVote(
                    voter_id=vessel_id,
                    perceived_object=obj,
                    confidence=obj.get("confidence", 0.5),
                    evidence=obj.get("evidence", []),
                ))
        return votes

    def compute_consensus(
        self, votes: List[PerceptionVote], quorum: Optional[float] = None
    ) -> PerceptionConsensus:
        """Compute consensus from a set of votes.

        Groups votes by object_id, checks quorum, and produces agreed objects.

        Args:
            votes: list of PerceptionVote instances.
            quorum: override quorum fraction (uses default if None).

        Returns:
            PerceptionConsensus with agreed objects, participation stats.
        """
        if quorum is None:
            quorum = self.quorum

        if not votes:
            return PerceptionConsensus(
                agreed_objects=[],
                participation=0,
                disagreement_ratio=0.0,
                timestamp=time.time(),
            )

        # Get unique voter count
        unique_voters = set(v.voter_id for v in votes)
        total_voters = len(unique_voters)
        required_voters = max(1, math.ceil(total_voters * quorum))

        # Group votes by object id
        groups: Dict[str, List[PerceptionVote]] = {}
        for vote in votes:
            oid = vote.object_id()
            groups.setdefault(oid, []).append(vote)

        agreed: List[dict] = []
        total_groups = len(groups)
        disagreeing_groups = 0

        for oid, group_votes in groups.items():
            voter_set = set(v.voter_id for v in group_votes)

            # Check if enough vessels agree on this object
            if len(voter_set) >= required_voters:
                # Check type agreement
                types = [v.object_type() for v in group_votes]
                type_counts = Counter(types)
                majority_type = type_counts.most_common(1)[0][0]
                type_agree = type_counts[majority_type] / len(types)

                # Compute weighted average position
                positions = [v.object_position() for v in group_votes]
                confidences = [v.confidence for v in group_votes]
                avg_pos = self._weighted_position(positions, confidences)
                avg_conf = sum(confidences) / len(confidences)

                # Compute intra-group disagreement
                type_disagree = 1.0 - type_agree

                agreed.append({
                    "id": oid,
                    "type": majority_type,
                    "position": avg_pos,
                    "confidence": avg_conf * type_agree,
                    "voter_count": len(voter_set),
                    "voters": list(voter_set),
                    "type_disagreement": type_disagree,
                })

                if type_disagree > 0.3:
                    disagreeing_groups += 1
            else:
                disagreeing_groups += 1

        disagreement_ratio = (
            disagreeing_groups / total_groups if total_groups > 0 else 0.0
        )

        consensus = PerceptionConsensus(
            agreed_objects=agreed,
            participation=total_voters,
            disagreement_ratio=disagreement_ratio,
            timestamp=time.time(),
        )
        self._history.append(consensus)
        return consensus

    def detect_maverick_vessel(
        self, votes: List[PerceptionVote]
    ) -> Optional[str]:
        """Detect a vessel whose votes consistently disagree with majority.

        A maverick is the voter whose object type classifications most often
        differ from the plurality vote for each object.

        Args:
            votes: list of PerceptionVote instances.

        Returns:
            The suspect vessel_id, or None if no clear maverick.
        """
        # Group by object_id
        groups: Dict[str, List[PerceptionVote]] = {}
        for vote in votes:
            oid = vote.object_id()
            groups.setdefault(oid, []).append(vote)

        if not groups:
            return None

        disagreement_counts: Dict[str, int] = {}
        total_votable = 0

        for oid, group_votes in groups.items():
            if len(group_votes) < 2:
                continue
            total_votable += 1
            types = [v.object_type() for v in group_votes]
            majority_type = Counter(types).most_common(1)[0][0]
            for v in group_votes:
                if v.object_type() != majority_type:
                    disagreement_counts[v.voter_id] = (
                        disagreement_counts.get(v.voter_id, 0) + 1
                    )

        if not disagreement_counts or total_votable == 0:
            return None

        # The vessel with highest disagreement rate
        most_disagreeing = max(
            disagreement_counts, key=disagreement_counts.get
        )
        # Only return if they disagree on more than half of votable objects
        if disagreement_counts[most_disagreeing] > total_votable * 0.5:
            return most_disagreeing
        return None

    def weight_votes_by_reputation(
        self,
        votes: List[PerceptionVote],
        reputation_scores: Dict[str, float],
    ) -> List[PerceptionVote]:
        """Re-weight votes by vessel reputation scores.

        Creates new PerceptionVote instances with adjusted confidence values.

        Args:
            votes: original votes.
            reputation_scores: mapping vessel_id -> reputation (0-1).

        Returns:
            New list of PerceptionVote with adjusted confidence.
        """
        weighted = []
        for vote in votes:
            rep = reputation_scores.get(vote.voter_id, 0.5)
            adjusted_conf = vote.confidence * rep
            weighted.append(PerceptionVote(
                voter_id=vote.voter_id,
                perceived_object=vote.perceived_object,
                confidence=adjusted_conf,
                evidence=list(vote.evidence),
            ))
        return weighted

    def compute_agreement_score(self, votes: List[PerceptionVote]) -> float:
        """Compute overall agreement score among votes (0-1).

        For each object group, measures the fraction of voters that agree
        on the majority type, then averages across all objects.

        Args:
            votes: list of PerceptionVote instances.

        Returns:
            Agreement score from 0.0 (no agreement) to 1.0 (unanimous).
        """
        groups: Dict[str, List[PerceptionVote]] = {}
        for vote in votes:
            oid = vote.object_id()
            groups.setdefault(oid, []).append(vote)

        if not groups:
            return 1.0  # No votes = perfect vacuous agreement

        type_agreements = []
        for oid, group_votes in groups.items():
            if len(group_votes) < 2:
                type_agreements.append(1.0)
                continue
            types = [v.object_type() for v in group_votes]
            type_counts = Counter(types)
            majority_count = type_counts.most_common(1)[0][1]
            type_agreements.append(majority_count / len(types))

        return sum(type_agreements) / len(type_agreements)

    def get_history(self) -> List[PerceptionConsensus]:
        """Return the history of consensus results."""
        return list(self._history)

    def _weighted_position(
        self,
        positions: List[Tuple[float, float, float]],
        confidences: List[float],
    ) -> Tuple[float, float, float]:
        total_w = sum(confidences)
        if total_w == 0:
            n = len(positions)
            if n == 0:
                return (0.0, 0.0, 0.0)
            return (
                sum(p[0] for p in positions) / n,
                sum(p[1] for p in positions) / n,
                sum(p[2] for p in positions) / n,
            )
        x = sum(p[0] * w for p, w in zip(positions, confidences)) / total_w
        y = sum(p[1] * w for p, w in zip(positions, confidences)) / total_w
        z = sum(p[2] * w for p, w in zip(positions, confidences)) / total_w
        return (x, y, z)
