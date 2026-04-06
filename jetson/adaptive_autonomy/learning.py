"""Autonomy level learning from experience — adaptive thresholds and recommendations."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from jetson.adaptive_autonomy.levels import AutonomyLevel


@dataclass
class TransitionExperience:
    """Recorded outcome of a single autonomy-level transition."""
    from_level: AutonomyLevel
    to_level: AutonomyLevel
    outcome: str             # success | failure | neutral
    satisfaction: float      # 0.0 – 1.0
    context_snapshot: Dict[str, Any] = field(default_factory=dict)


class AutonomyLearner:
    """Learns from transition outcomes to improve autonomy recommendations."""

    def __init__(self) -> None:
        self._experiences: List[TransitionExperience] = []
        # context_key -> (level, satisfaction_sum, count)
        self._context_level_stats: Dict[
            str, List[Tuple[AutonomyLevel, float]]
        ] = defaultdict(list)

    # ---- public API ----

    def record_experience(self, experience: TransitionExperience) -> None:
        """Store a transition experience for future learning."""
        self._experiences.append(experience)
        # Index by context key
        ctx = experience.context_snapshot
        for key, val in ctx.items():
            k = f"{key}={val}"
            self._context_level_stats[k].append(
                (experience.to_level, experience.satisfaction)
            )

    def compute_optimal_level(
        self, context: Dict[str, Any]
    ) -> Optional[AutonomyLevel]:
        """Recommend the best autonomy level given a context snapshot.

        Uses a simple majority-weighted vote from past experiences matching
        the current context keys.
        """
        if not self._experiences:
            return None

        level_scores: Dict[AutonomyLevel, float] = defaultdict(float)
        level_counts: Dict[AutonomyLevel, int] = defaultdict(int)

        for key, val in context.items():
            k = f"{key}={val}"
            entries = self._context_level_stats.get(k, [])
            for lv, sat in entries:
                level_scores[lv] += sat
                level_counts[lv] += 1

        if not level_scores:
            return None

        # Pick the level with the highest average satisfaction
        best_level: Optional[AutonomyLevel] = None
        best_avg = -1.0
        for lv in level_scores:
            avg = level_scores[lv] / level_counts[lv]
            if avg > best_avg:
                best_avg = avg
                best_level = lv
        return best_level

    def analyze_transition_satisfaction(
        self, experiences: Optional[List[TransitionExperience]] = None
    ) -> List[Dict[str, Any]]:
        """Rank transitions by average satisfaction.

        Returns list of dicts: ``from_level``, ``to_level``, ``count``,
        ``avg_satisfaction``, sorted best-first.
        """
        data = experiences if experiences is not None else self._experiences
        bucket: Dict[
            Tuple[AutonomyLevel, AutonomyLevel],
            List[float],
        ] = defaultdict(list)
        for exp in data:
            bucket[(exp.from_level, exp.to_level)].append(exp.satisfaction)

        results: List[Dict[str, Any]] = []
        for (fl, tl), sats in sorted(bucket.items()):
            results.append({
                "from_level": fl,
                "to_level": tl,
                "count": len(sats),
                "avg_satisfaction": round(statistics.mean(sats), 4),
            })

        results.sort(key=lambda r: r["avg_satisfaction"], reverse=True)
        return results

    def adapt_thresholds(
        self, experiences: Optional[List[TransitionExperience]] = None
    ) -> Dict[str, float]:
        """Compute suggested threshold adjustments from past experiences.

        Returns a dict with ``risk_up``, ``risk_down``, ``confidence_up``,
        ``confidence_down`` deltas.  Positive = make more autonomous.
        """
        data = experiences if experiences is not None else self._experiences
        if not data:
            return {
                "risk_up": 0.0,
                "risk_down": 0.0,
                "confidence_up": 0.0,
                "confidence_down": 0.0,
            }

        successes = [e for e in data if e.outcome == "success"]
        failures = [e for e in data if e.outcome == "failure"]

        # If mostly successful, we can be more autonomous (increase risk
        # tolerance, lower confidence threshold)
        total = len(data)
        success_rate = len(successes) / total if total else 0.0

        risk_delta = round((success_rate - 0.5) * 0.2, 4)
        confidence_delta = round((success_rate - 0.5) * 0.15, 4)

        return {
            "risk_up": max(risk_delta, 0.0),
            "risk_down": max(-risk_delta, 0.0),
            "confidence_up": max(confidence_delta, 0.0),
            "confidence_down": max(-confidence_delta, 0.0),
        }

    def predict_satisfaction(
        self,
        from_level: AutonomyLevel,
        to_level: AutonomyLevel,
        context: Dict[str, Any],
    ) -> float:
        """Predict satisfaction for a hypothetical transition.

        Returns 0.5 (neutral) if no matching data exists.
        """
        matching: List[float] = []
        for exp in self._experiences:
            if exp.from_level != from_level or exp.to_level != to_level:
                continue
            # Count context overlap
            overlap = sum(
                1 for k, v in context.items()
                if exp.context_snapshot.get(k) == v
            )
            if overlap > 0:
                matching.append(exp.satisfaction)

        if not matching:
            # Fallback: average of all transitions between these two levels
            all_between: List[float] = [
                e.satisfaction for e in self._experiences
                if e.from_level == from_level and e.to_level == to_level
            ]
            if all_between:
                return round(statistics.mean(all_between), 4)
            return 0.5

        return round(statistics.mean(matching), 4)

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics on recorded experiences."""
        if not self._experiences:
            return {
                "total_experiences": 0,
                "unique_transitions": 0,
                "avg_satisfaction": 0.0,
                "success_rate": 0.0,
                "most_common_transition": None,
                "best_transition": None,
                "worst_transition": None,
            }

        outcomes = [e.outcome for e in self._experiences]
        sats = [e.satisfaction for e in self._experiences]
        transitions = [
            (e.from_level, e.to_level) for e in self._experiences
        ]

        success_rate = outcomes.count("success") / len(outcomes)
        avg_sat = statistics.mean(sats)

        # Most common transition
        from collections import Counter
        tc = Counter(transitions)
        most_common = tc.most_common(1)[0][0] if tc else None

        # Best / worst by average satisfaction
        best_exp = max(self._experiences, key=lambda e: e.satisfaction)
        worst_exp = min(self._experiences, key=lambda e: e.satisfaction)

        return {
            "total_experiences": len(self._experiences),
            "unique_transitions": len(set(transitions)),
            "avg_satisfaction": round(avg_sat, 4),
            "success_rate": round(success_rate, 4),
            "most_common_transition": most_common,
            "best_transition": (best_exp.from_level, best_exp.to_level),
            "worst_transition": (worst_exp.from_level, worst_exp.to_level),
        }
