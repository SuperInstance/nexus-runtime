"""Reputation-weighted selection for marketplace bids."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ReputationScore:
    """Reputation score for a vessel."""
    vessel_id: str = ""
    score: float = 0.0
    history: List[float] = field(default_factory=list)
    rank: int = 0
    confidence: float = 0.0


@dataclass
class WeightedBid:
    """A bid adjusted by reputation weight."""
    bid_id: str = ""
    task_id: str = ""
    bidder_id: str = ""
    original_amount: float = 0.0
    weighted_amount: float = 0.0
    reputation_weight: float = 1.0


@dataclass
class RankedVessel:
    """A vessel in a ranked list."""
    vessel_id: str = ""
    score: float = 0.0
    rank: int = 0


class ReputationWeightedSelector:
    """Selects bids and ranks vessels using reputation weights."""

    def compute_reputation(
        self,
        trust_scores: List[float],
        task_history: Optional[List[Dict[str, Any]]] = None,
        reviews: Optional[List[float]] = None,
    ) -> ReputationScore:
        """Compute a reputation score from trust scores, history, and reviews."""
        task_history = task_history or []
        reviews = reviews or []

        if not trust_scores and not reviews:
            return ReputationScore(score=0.5, confidence=0.0)  # neutral default

        components: List[float] = []

        # Trust score component
        if trust_scores:
            avg_trust = sum(trust_scores) / len(trust_scores)
            components.append(avg_trust)

        # Task completion component
        if task_history:
            completed = sum(1 for t in task_history if t.get("completed", False))
            total = len(task_history)
            completion_rate = completed / total if total > 0 else 0.0
            components.append(completion_rate)

        # Review component
        if reviews:
            avg_review = sum(reviews) / len(reviews)
            components.append(avg_review)

        if not components:
            return ReputationScore(score=0.5, confidence=0.0)

        # Weighted average of components
        score = sum(components) / len(components)
        score = max(0.0, min(1.0, score))

        # Confidence based on data volume
        total_data_points = len(trust_scores) + len(task_history) + len(reviews)
        confidence = min(1.0, total_data_points / 10.0)  # Full confidence at 10+ data points

        return ReputationScore(
            score=round(score, 4),
            history=trust_scores,
            confidence=round(confidence, 4),
        )

    def weight_bids(
        self,
        bids: List[Any],
        reputations: Dict[str, ReputationScore],
    ) -> List[WeightedBid]:
        """Apply reputation weights to bids and return weighted rankings."""
        weighted: List[WeightedBid] = []
        for bid in bids:
            bidder_id = getattr(bid, "bidder_id", "")
            rep = reputations.get(bidder_id, ReputationScore(score=0.5))
            # Weighted amount = amount / reputation (higher rep = lower effective cost)
            rep_weight = rep.score if rep.score > 0 else 0.01
            weighted_amount = getattr(bid, "amount", 0.0) / rep_weight
            weighted.append(WeightedBid(
                bid_id=getattr(bid, "id", ""),
                task_id=getattr(bid, "task_id", ""),
                bidder_id=bidder_id,
                original_amount=getattr(bid, "amount", 0.0),
                weighted_amount=round(weighted_amount, 2),
                reputation_weight=rep_weight,
            ))

        weighted.sort(key=lambda w: w.weighted_amount)
        return weighted

    def update_reputation(
        self,
        reputation: ReputationScore,
        task_result: Dict[str, Any],
    ) -> ReputationScore:
        """Update reputation based on task result."""
        score = reputation.score
        history = list(reputation.history)

        # Performance score from task result
        performance = task_result.get("performance_score", 0.5)
        performance = max(0.0, min(1.0, performance))

        # Weighted moving average (new result weighted at 30%)
        alpha = 0.3
        new_score = alpha * performance + (1 - alpha) * score
        new_score = max(0.0, min(1.0, new_score))

        history.append(performance)

        new_confidence = min(1.0, len(history) / 10.0)

        return ReputationScore(
            vessel_id=reputation.vessel_id,
            score=round(new_score, 4),
            history=history,
            rank=reputation.rank,
            confidence=round(new_confidence, 4),
        )

    def compute_confidence(self, reputation_history: List[float]) -> float:
        """Compute confidence interval from reputation history."""
        if not reputation_history:
            return 0.0
        if len(reputation_history) == 1:
            return 0.5

        n = len(reputation_history)
        mean = sum(reputation_history) / n
        variance = sum((x - mean) ** 2 for x in reputation_history) / (n - 1) if n > 1 else 0.0
        std_dev = math.sqrt(variance)

        # Confidence: inversely proportional to variance, scaled by sample size
        # Using coefficient of variation
        if mean == 0:
            return 0.0
        cv = std_dev / mean  # coefficient of variation
        # Higher CV -> lower confidence
        confidence = 1.0 / (1.0 + cv)
        # Scale by sample size
        confidence *= min(1.0, n / 5.0)
        return round(max(0.0, min(1.0, confidence)), 4)

    def decay_reputation(self, score: float, time_since_last_activity: float) -> float:
        """Decay reputation score based on time since last activity (in days)."""
        # Exponential decay with half-life of 90 days
        half_life = 90.0
        decay_factor = math.exp(-0.693 * time_since_last_activity / half_life)
        decayed = score * decay_factor
        # Never decay below 0.1 (base trust level)
        return round(max(0.1, decayed), 4)

    def rank_vessels(
        self,
        vessels: List[Any],
        reputations: Dict[str, ReputationScore],
        task_requirements: Optional[Dict[str, Any]] = None,
    ) -> List[RankedVessel]:
        """Rank vessels by reputation and optionally task requirements."""
        task_requirements = task_requirements or {}
        ranked: List[RankedVessel] = []

        for vessel in vessels:
            vid = getattr(vessel, "vessel_id", "")
            rep = reputations.get(vid, ReputationScore(score=0.5))

            base_score = rep.score

            # Boost for matching task requirements
            boost = 0.0
            if "required_sensors" in task_requirements:
                req_sensors = set(task_requirements["required_sensors"])
                avail_sensors = set(getattr(vessel, "sensor_types", []))
                overlap = req_sensors & avail_sensors
                if req_sensors:
                    boost += 0.1 * (len(overlap) / len(req_sensors))

            if "min_speed" in task_requirements:
                min_speed = task_requirements["min_speed"]
                max_speed = getattr(vessel, "max_speed", 0.0)
                if max_speed >= min_speed:
                    boost += 0.1

            final_score = min(1.0, base_score + boost)
            ranked.append(RankedVessel(
                vessel_id=vid,
                score=round(final_score, 4),
                rank=0,
            ))

        # Assign ranks
        ranked.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1

        return ranked
