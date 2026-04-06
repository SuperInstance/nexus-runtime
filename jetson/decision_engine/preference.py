"""Preference modeling: outranking, pairwise comparison, preference learning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Preference:
    """A single decision criterion preference."""
    criterion: str
    direction: str  # "min" or "max"
    importance: float = 1.0
    indifference_threshold: float = 0.0
    preference_threshold: float = 0.0


class PreferenceModel:
    """Models and reasons about decision-maker preferences."""

    def __init__(self) -> None:
        self._preferences: List[Preference] = []
        self._learned_weights: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_preference(self, preference: Preference) -> None:
        """Register a new preference criterion."""
        self._preferences.append(preference)

    @property
    def preferences(self) -> List[Preference]:
        return list(self._preferences)

    def compare_alternatives(
        self,
        alt_a: Dict[str, float],
        alt_b: Dict[str, float],
    ) -> Tuple[str, float]:
        """Compare two alternatives and return (preferred, strength).

        Returns the name of the preferred alternative ("a" or "b" or "tie")
        and a strength value in [0, 1].
        Uses the PROMETHEE-inspired concordance/discordance approach.
        """
        if not self._preferences:
            return ("tie", 0.0)

        concordance = 0.0
        discordance = 0.0
        total_weight = sum(p.importance for p in self._preferences)
        if total_weight == 0:
            return ("tie", 0.0)

        for pref in self._preferences:
            if pref.criterion not in alt_a or pref.criterion not in alt_b:
                continue
            val_a = alt_a[pref.criterion]
            val_b = alt_b[pref.criterion]
            diff = val_a - val_b  # positive means a > b

            # Concordance: how much a is preferred over b
            if pref.direction == "max":
                if diff > pref.preference_threshold:
                    c = 1.0
                elif diff > pref.indifference_threshold:
                    c = (diff - pref.indifference_threshold) / (
                        pref.preference_threshold - pref.indifference_threshold + 1e-12
                    )
                else:
                    c = 0.5  # indifference zone
            else:  # min
                if diff < -pref.preference_threshold:
                    c = 1.0
                elif diff < -pref.indifference_threshold:
                    c = (-diff - pref.indifference_threshold) / (
                        pref.preference_threshold - pref.indifference_threshold + 1e-12
                    )
                else:
                    c = 0.5

            concordance += pref.importance * c

            # Discordance: how much b is strongly better on this criterion
            if pref.direction == "max":
                d = max(0.0, -diff - pref.preference_threshold)
            else:
                d = max(0.0, diff - pref.preference_threshold)
            discordance = max(discordance, d)

        concordance /= total_weight

        # Net preference index
        # Check reverse concordance
        concordance_rev = 0.0
        for pref in self._preferences:
            if pref.criterion not in alt_a or pref.criterion not in alt_b:
                continue
            val_a = alt_a[pref.criterion]
            val_b = alt_b[pref.criterion]
            diff = val_b - val_a

            if pref.direction == "max":
                if diff > pref.preference_threshold:
                    c = 1.0
                elif diff > pref.indifference_threshold:
                    c = (diff - pref.indifference_threshold) / (
                        pref.preference_threshold - pref.indifference_threshold + 1e-12
                    )
                else:
                    c = 0.5
            else:
                if diff < -pref.preference_threshold:
                    c = 1.0
                elif diff < -pref.indifference_threshold:
                    c = (-diff - pref.indifference_threshold) / (
                        pref.preference_threshold - pref.indifference_threshold + 1e-12
                    )
                else:
                    c = 0.5
            concordance_rev += pref.importance * c
        concordance_rev /= total_weight

        net = concordance - concordance_rev
        strength = abs(net)

        if net > 1e-9:
            return ("a", strength)
        elif net < -1e-9:
            return ("b", strength)
        else:
            return ("tie", strength)

    def compute_outranking(
        self,
        alternatives: List[Dict[str, float]],
    ) -> List[List[float]]:
        """Compute the outranking matrix for a list of alternatives.

        Returns an n×n matrix where entry (i,j) is the credibility that
        alternative i outranks alternative j.
        """
        n = len(alternatives)
        matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
        total_weight = sum(p.importance for p in self._preferences)
        if total_weight == 0:
            return matrix

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                matrix[i][j] = self._outranking_credibility(
                    alternatives[i], alternatives[j]
                )
        return matrix

    def compute_ranking(
        self,
        outranking_matrix: List[List[float]],
    ) -> List[int]:
        """Compute a ranking from an outranking matrix using net flow scores.

        Returns a list of alternative indices ordered from best to worst.
        """
        n = len(outranking_matrix)
        if n == 0:
            return []
        net_flows: List[Tuple[float, int]] = []
        for i in range(n):
            leaving = sum(outranking_matrix[i][j] for j in range(n) if j != i)
            entering = sum(outranking_matrix[j][i] for j in range(n) if j != i)
            net_flows.append((leaving - entering, i))
        net_flows.sort(key=lambda x: -x[0])
        return [idx for _, idx in net_flows]

    def learn_preferences(
        self,
        pairs: List[Tuple[Dict[str, float], Dict[str, float]]],
    ) -> "PreferenceModel":
        """Learn preference weights from pairwise comparison data.

        Uses a simple frequency-based heuristic: weight each criterion
        proportionally to how often the winning alternative has a better value.
        """
        criteria: Set[str] = set()
        for a, b in pairs:
            criteria.update(a.keys())
            criteria.update(b.keys())

        counts: Dict[str, float] = {c: 0.0 for c in criteria}
        for a, b in pairs:
            # Determine which alternative is "preferred" by majority vote
            # across equal-weight criteria
            a_wins = 0
            b_wins = 0
            for c in criteria:
                va = a.get(c, 0.0)
                vb = b.get(c, 0.0)
                if va > vb:
                    a_wins += 1
                elif vb > va:
                    b_wins += 1

            winner = a if a_wins > b_wins else b
            loser = b if a_wins > b_wins else a

            for c in criteria:
                wv = winner.get(c, 0.0)
                lv = loser.get(c, 0.0)
                if wv > lv:
                    counts[c] += 1.0

        total = sum(counts.values()) or 1.0
        self._learned_weights = {c: counts[c] / total for c in criteria}

        # Rebuild preferences with learned weights
        self._preferences = []
        for c in sorted(criteria):
            self._preferences.append(
                Preference(
                    criterion=c,
                    direction="max",
                    importance=self._learned_weights[c],
                    indifference_threshold=0.0,
                    preference_threshold=0.0,
                )
            )
        return self

    def validate_preferences(
        self,
        preferences: Optional[List[Preference]] = None,
    ) -> float:
        """Check preference consistency, returning a score in [0, 1].

        Higher means more consistent (no cycles, reasonable thresholds).
        """
        prefs = preferences or self._preferences
        if not prefs:
            return 1.0

        score = 1.0

        # Check thresholds are non-negative
        for p in prefs:
            if p.indifference_threshold < 0:
                score -= 0.2
            if p.preference_threshold < p.indifference_threshold:
                score -= 0.2
            if p.importance < 0:
                score -= 0.3

        # Check weights sum to something reasonable
        total_w = sum(p.importance for p in prefs)
        if total_w <= 0:
            score -= 0.3

        # Check for duplicate criteria
        names = [p.criterion for p in prefs]
        if len(names) != len(set(names)):
            score -= 0.2

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _outranking_credibility(
        self,
        a: Dict[str, float],
        b: Dict[str, float],
    ) -> float:
        """Compute the credibility that a outranks b."""
        if not self._preferences:
            return 0.0

        total_weight = sum(p.importance for p in self._preferences)
        if total_weight == 0:
            return 0.0

        concordance = 0.0
        max_discordance = 0.0

        for pref in self._preferences:
            if pref.criterion not in a or pref.criterion not in b:
                continue
            val_a = a[pref.criterion]
            val_b = b[pref.criterion]
            diff = val_a - val_b

            if pref.direction == "max":
                if diff >= -pref.indifference_threshold:
                    c = 1.0
                elif diff >= -pref.preference_threshold:
                    c = (diff + pref.preference_threshold) / (
                        pref.preference_threshold - pref.indifference_threshold + 1e-12
                    )
                else:
                    c = 0.0
                d = max(0.0, -diff - pref.preference_threshold)
            else:  # min
                if diff <= pref.indifference_threshold:
                    c = 1.0
                elif diff <= pref.preference_threshold:
                    c = (pref.preference_threshold - diff) / (
                        pref.preference_threshold - pref.indifference_threshold + 1e-12
                    )
                else:
                    c = 0.0
                d = max(0.0, diff - pref.preference_threshold)

            concordance += pref.importance * c
            if d > max_discordance:
                max_discordance = d

        concordance /= total_weight
        # Veto effect from discordance
        credibility = concordance
        if max_discordance > 0:
            credibility = concordance * (1.0 - min(1.0, max_discordance))

        return max(0.0, min(1.0, credibility))
