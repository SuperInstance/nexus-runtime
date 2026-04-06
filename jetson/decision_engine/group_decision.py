"""Group decision making: plurality, Borda, Condorcet, weighted voting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Voter:
    """A single voter in a group decision."""
    id: str
    preferences: List[str]  # ordered list, best first
    weight: float = 1.0
    reliability: float = 1.0


@dataclass
class GroupDecisionResult:
    """Result of a group decision process."""
    selected_alternative: Optional[str] = None
    consensus_level: float = 0.0
    individual_votes: Dict[str, str] = field(default_factory=dict)
    dissenters: List[str] = field(default_factory=list)


class GroupDecisionMaker:
    """Aggregates individual preferences into group decisions."""

    # ------------------------------------------------------------------
    # Voting rules
    # ------------------------------------------------------------------

    def plurality_vote(
        self,
        voters: List[Voter],
        alternatives: List[str],
    ) -> GroupDecisionResult:
        """Each voter casts one vote for their top choice.  Most votes wins."""
        if not voters or not alternatives:
            return GroupDecisionResult()

        votes: Dict[str, int] = Counter()
        individual_votes: Dict[str, str] = {}

        for voter in voters:
            # Pick the first preference that is a valid alternative
            choice = None
            for pref in voter.preferences:
                if pref in alternatives:
                    choice = pref
                    break
            if choice is None and alternatives:
                choice = alternatives[0]
            if choice is not None:
                votes[choice] += 1
                individual_votes[voter.id] = choice

        if not votes:
            return GroupDecisionResult()

        winner = votes.most_common(1)[0][0]
        total = sum(votes.values())
        top_count = votes[winner]
        consensus = top_count / total if total > 0 else 0.0

        dissenters = [
            voter.id
            for voter in voters
            if individual_votes.get(voter.id) != winner
        ]

        return GroupDecisionResult(
            selected_alternative=winner,
            consensus_level=consensus,
            individual_votes=individual_votes,
            dissenters=dissenters,
        )

    def borda_count(
        self,
        voters: List[Voter],
        alternatives: List[str],
    ) -> GroupDecisionResult:
        """Borda count: each voter ranks all alternatives, points awarded by rank."""
        if not voters or not alternatives:
            return GroupDecisionResult()

        n_alts = len(alternatives)
        scores: Dict[str, float] = {a: 0.0 for a in alternatives}
        individual_votes: Dict[str, str] = {}

        alt_set = set(alternatives)
        for voter in voters:
            effective_prefs = [p for p in voter.preferences if p in alt_set]
            # Pad with remaining alternatives if not all ranked
            remaining = [a for a in alternatives if a not in effective_prefs]
            effective_prefs.extend(remaining)

            for rank_idx, alt in enumerate(effective_prefs):
                # Borda score: n - rank (higher rank = more points)
                points = (n_alts - 1 - rank_idx) * voter.weight
                scores[alt] += points

            if effective_prefs:
                individual_votes[voter.id] = effective_prefs[0]

        if not scores:
            return GroupDecisionResult()

        winner = max(scores, key=lambda a: scores[a])
        total = sum(voter.weight for voter in voters)
        max_possible = (n_alts - 1) * total
        winner_score = scores[winner]
        consensus = winner_score / max_possible if max_possible > 0 else 0.0

        dissenters = [
            voter.id
            for voter in voters
            if individual_votes.get(voter.id) != winner
        ]

        return GroupDecisionResult(
            selected_alternative=winner,
            consensus_level=consensus,
            individual_votes=individual_votes,
            dissenters=dissenters,
        )

    def condorcet_method(
        self,
        voters: List[Voter],
        alternatives: List[str],
    ) -> GroupDecisionResult:
        """Condorcet method: find the alternative that beats every other in pairwise contests.

        If no Condorcet winner exists, falls back to Copeland's method.
        """
        if not voters or not alternatives:
            return GroupDecisionResult()

        alt_set = set(alternatives)
        n = len(alternatives)

        # Build pairwise comparison matrix
        wins: Dict[str, int] = {a: 0 for a in alternatives}
        win_matrix: Dict[str, Dict[str, int]] = {
            a: {b: 0 for b in alternatives} for a in alternatives
        }

        for voter in voters:
            effective_prefs = [p for p in voter.preferences if p in alt_set]
            remaining = [a for a in alternatives if a not in effective_prefs]
            effective_prefs.extend(remaining)

            for i, alt_i in enumerate(effective_prefs):
                for j, alt_j in enumerate(effective_prefs):
                    if i < j:
                        # voter prefers alt_i over alt_j
                        win_matrix[alt_i][alt_j] += 1
                    elif j < i:
                        win_matrix[alt_j][alt_i] += 1

        # Check for Condorcet winner
        condorcet_winner = None
        for alt in alternatives:
            beats_all = True
            for other in alternatives:
                if other == alt:
                    continue
                if win_matrix[alt][other] <= win_matrix[other][alt]:
                    beats_all = False
                    break
            if beats_all:
                condorcet_winner = alt
                break

        if condorcet_winner is not None:
            # Compute Copeland scores for consensus
            copeland: Dict[str, float] = {}
            for alt in alternatives:
                wins_count = sum(
                    1 for other in alternatives
                    if other != alt and win_matrix[alt][other] > win_matrix[other][alt]
                )
                ties = sum(
                    0.5 for other in alternatives
                    if other != alt and win_matrix[alt][other] == win_matrix[other][alt]
                )
                copeland[alt] = wins_count + ties
            max_copeland = max(copeland.values())
            consensus = copeland[condorcet_winner] / (n - 1) if n > 1 else 1.0

            individual_votes: Dict[str, str] = {}
            for voter in voters:
                effective_prefs = [p for p in voter.preferences if p in alt_set]
                remaining = [a for a in alternatives if a not in effective_prefs]
                effective_prefs.extend(remaining)
                if effective_prefs:
                    individual_votes[voter.id] = effective_prefs[0]

            dissenters = [
                voter.id
                for voter in voters
                if individual_votes.get(voter.id) != condorcet_winner
            ]

            return GroupDecisionResult(
                selected_alternative=condorcet_winner,
                consensus_level=consensus,
                individual_votes=individual_votes,
                dissenters=dissenters,
            )

        # No Condorcet winner — use Copeland's method
        copeland: Dict[str, float] = {}
        for alt in alternatives:
            wins_count = sum(
                1 for other in alternatives
                if other != alt and win_matrix[alt][other] > win_matrix[other][alt]
            )
            ties = sum(
                0.5 for other in alternatives
                if other != alt and win_matrix[alt][other] == win_matrix[other][alt]
            )
            copeland[alt] = wins_count + ties

        winner = max(copeland, key=lambda a: copeland[a])
        max_copeland = max(copeland.values()) if copeland else 0.0
        consensus = max_copeland / (n - 1) if n > 1 else 1.0

        individual_votes: Dict[str, str] = {}
        for voter in voters:
            effective_prefs = [p for p in voter.preferences if p in alt_set]
            remaining = [a for a in alternatives if a not in effective_prefs]
            effective_prefs.extend(remaining)
            if effective_prefs:
                individual_votes[voter.id] = effective_prefs[0]

        dissenters = [
            voter.id
            for voter in voters
            if individual_votes.get(voter.id) != winner
        ]

        return GroupDecisionResult(
            selected_alternative=winner,
            consensus_level=consensus,
            individual_votes=individual_votes,
            dissenters=dissenters,
        )

    def compute_consensus(self, votes: Dict[str, str]) -> float:
        """Compute consensus level from a dict of voter_id -> vote.

        Returns a value in [0, 1] where 1 = unanimous.
        """
        if not votes:
            return 0.0
        counts = Counter(votes.values())
        most_common = counts.most_common(1)[0][1]
        return most_common / len(votes)

    def detect_voting_paradox(
        self,
        voters: List[Voter],
        alternatives: List[str],
    ) -> bool:
        """Detect Condorcet paradox (cyclical majority preferences).

        Returns True if a voting paradox is detected.
        """
        alt_set = set(alternatives)
        n = len(alternatives)
        if n < 3:
            return False

        # Build pairwise comparison
        win_matrix: Dict[str, Dict[str, int]] = {
            a: {b: 0 for b in alternatives} for a in alternatives
        }
        for voter in voters:
            effective_prefs = [p for p in voter.preferences if p in alt_set]
            remaining = [a for a in alternatives if a not in effective_prefs]
            effective_prefs.extend(remaining)
            for i, alt_i in enumerate(effective_prefs):
                for j, alt_j in enumerate(effective_prefs):
                    if i < j:
                        win_matrix[alt_i][alt_j] += 1
                    elif j < i:
                        win_matrix[alt_j][alt_i] += 1

        # Build majority preference graph
        majority_graph: Dict[str, Set[str]] = {a: set() for a in alternatives}
        for a in alternatives:
            for b in alternatives:
                if a == b:
                    continue
                if win_matrix[a][b] > win_matrix[b][a]:
                    majority_graph[a].add(b)

        # Detect cycles using DFS
        return self._has_cycle(majority_graph, alternatives)

    def weighted_voting(
        self,
        voters: List[Voter],
        alternatives: List[str],
    ) -> GroupDecisionResult:
        """Weighted plurality: each voter's weight multiplies their vote."""
        if not voters or not alternatives:
            return GroupDecisionResult()

        scores: Dict[str, float] = {a: 0.0 for a in alternatives}
        individual_votes: Dict[str, str] = {}

        alt_set = set(alternatives)
        for voter in voters:
            choice = None
            for pref in voter.preferences:
                if pref in alt_set:
                    choice = pref
                    break
            if choice is None and alternatives:
                choice = alternatives[0]
            if choice is not None:
                effective_weight = voter.weight * voter.reliability
                scores[choice] += effective_weight
                individual_votes[voter.id] = choice

        if not scores:
            return GroupDecisionResult()

        winner = max(scores, key=lambda a: scores[a])
        total_weight = sum(voter.weight * voter.reliability for voter in voters)
        winner_score = scores[winner]
        consensus = winner_score / total_weight if total_weight > 0 else 0.0

        dissenters = [
            voter.id
            for voter in voters
            if individual_votes.get(voter.id) != winner
        ]

        return GroupDecisionResult(
            selected_alternative=winner,
            consensus_level=consensus,
            individual_votes=individual_votes,
            dissenters=dissenters,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _has_cycle(
        graph: Dict[str, Set[str]],
        nodes: List[str],
    ) -> bool:
        """Detect if a directed graph has a cycle using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in nodes}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in graph.get(node, set()):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        for node in nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False
