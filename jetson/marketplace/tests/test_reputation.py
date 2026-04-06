"""Tests for reputation module."""

import pytest

from jetson.marketplace.reputation import (
    ReputationScore, WeightedBid, RankedVessel, ReputationWeightedSelector,
)


class TestReputationScore:
    def test_default(self):
        r = ReputationScore()
        assert r.score == 0.0
        assert r.history == []
        assert r.rank == 0
        assert r.confidence == 0.0

    def test_custom(self):
        r = ReputationScore(vessel_id="v1", score=0.85, history=[0.7, 0.8, 0.9], rank=3, confidence=0.75)
        assert r.vessel_id == "v1"
        assert r.score == 0.85
        assert len(r.history) == 3
        assert r.rank == 3


class TestWeightedBid:
    def test_default(self):
        w = WeightedBid()
        assert w.original_amount == 0.0
        assert w.weighted_amount == 0.0
        assert w.reputation_weight == 1.0


class TestRankedVessel:
    def test_default(self):
        r = RankedVessel()
        assert r.vessel_id == ""
        assert r.rank == 0


class TestReputationWeightedSelector:
    def setup_method(self):
        self.selector = ReputationWeightedSelector()

    def test_compute_reputation_trust_only(self):
        result = self.selector.compute_reputation(trust_scores=[0.8, 0.9, 0.85])
        assert result.score == pytest.approx(0.85, abs=0.01)

    def test_compute_reputation_with_reviews(self):
        result = self.selector.compute_reputation(
            trust_scores=[0.8], reviews=[0.9, 0.7]
        )
        expected = (0.8 + 0.9 + 0.7) / 3
        assert result.score == pytest.approx(expected, abs=0.01)

    def test_compute_reputation_with_task_history(self):
        history = [
            {"completed": True},
            {"completed": True},
            {"completed": False},
            {"completed": True},
        ]
        result = self.selector.compute_reputation(trust_scores=[0.7], task_history=history)
        # trust avg = 0.7, completion = 0.75, avg = 0.725
        assert result.score == pytest.approx(0.725, abs=0.01)

    def test_compute_reputation_no_data(self):
        result = self.selector.compute_reputation(trust_scores=[])
        assert result.score == 0.5
        assert result.confidence == 0.0

    def test_compute_reputation_empty_lists(self):
        result = self.selector.compute_reputation(trust_scores=[], reviews=[])
        assert result.score == 0.5

    def test_compute_reputation_confidence_increases(self):
        r1 = self.selector.compute_reputation(trust_scores=[0.8])
        r2 = self.selector.compute_reputation(trust_scores=[0.8] * 10)
        assert r2.confidence > r1.confidence

    def test_compute_reputation_confidence_capped(self):
        r = self.selector.compute_reputation(trust_scores=[0.8] * 20)
        assert r.confidence <= 1.0

    def test_compute_reputation_score_capped_upper(self):
        r = self.selector.compute_reputation(trust_scores=[1.0], reviews=[1.0])
        assert r.score <= 1.0

    def test_compute_reputation_score_capped_lower(self):
        r = self.selector.compute_reputation(trust_scores=[0.0], reviews=[0.0])
        assert r.score >= 0.0

    def test_weight_bids_basic(self):
        class FakeBid:
            def __init__(self, bid_id, bidder_id, amount):
                self.id = bid_id
                self.bidder_id = bidder_id
                self.amount = amount
                self.task_id = "t1"

        bids = [FakeBid("b1", "v1", 1000.0), FakeBid("b2", "v2", 800.0)]
        reps = {"v1": ReputationScore(score=0.8), "v2": ReputationScore(score=0.5)}
        weighted = self.selector.weight_bids(bids, reps)
        assert len(weighted) == 2
        # v2 has lower amount but also lower reputation
        # weighted_amount = amount / reputation
        # v1: 1000/0.8 = 1250, v2: 800/0.5 = 1600
        assert weighted[0].bidder_id == "v1"

    def test_weight_bids_sorted(self):
        class FakeBid:
            def __init__(self, bid_id, bidder_id, amount):
                self.id = bid_id
                self.bidder_id = bidder_id
                self.amount = amount
                self.task_id = "t1"

        bids = [FakeBid("b1", "v3", 1200.0), FakeBid("b2", "v1", 500.0), FakeBid("b3", "v2", 800.0)]
        reps = {"v1": ReputationScore(score=1.0), "v2": ReputationScore(score=0.8), "v3": ReputationScore(score=0.6)}
        weighted = self.selector.weight_bids(bids, reps)
        amounts = [w.weighted_amount for w in weighted]
        assert amounts == sorted(amounts)

    def test_weight_bids_unknown_reputation(self):
        class FakeBid:
            def __init__(self, bid_id, bidder_id, amount):
                self.id = bid_id
                self.bidder_id = bidder_id
                self.amount = amount
                self.task_id = "t1"

        bids = [FakeBid("b1", "unknown", 500.0)]
        weighted = self.selector.weight_bids(bids, {})
        assert weighted[0].reputation_weight == 0.5

    def test_weight_bids_empty(self):
        weighted = self.selector.weight_bids([], {})
        assert weighted == []

    def test_update_reputation_improvement(self):
        rep = ReputationScore(vessel_id="v1", score=0.5, history=[0.5])
        updated = self.selector.update_reputation(rep, {"performance_score": 0.9})
        assert updated.score > 0.5
        assert len(updated.history) == 2

    def test_update_reputation_decline(self):
        rep = ReputationScore(vessel_id="v1", score=0.8, history=[0.8])
        updated = self.selector.update_reputation(rep, {"performance_score": 0.3})
        assert updated.score < 0.8

    def test_update_reputation_preserves_vessel_id(self):
        rep = ReputationScore(vessel_id="v1", score=0.7)
        updated = self.selector.update_reputation(rep, {"performance_score": 0.8})
        assert updated.vessel_id == "v1"

    def test_update_reputation_confidence_increases(self):
        rep = ReputationScore(vessel_id="v1", score=0.5, history=[])
        updated = self.selector.update_reputation(rep, {"performance_score": 0.7})
        assert updated.confidence > rep.confidence

    def test_update_reputation_score_capped(self):
        rep = ReputationScore(vessel_id="v1", score=0.99)
        updated = self.selector.update_reputation(rep, {"performance_score": 1.0})
        assert updated.score <= 1.0

    def test_compute_confidence_empty(self):
        assert self.selector.compute_confidence([]) == 0.0

    def test_compute_confidence_single_value(self):
        assert self.selector.compute_confidence([0.5]) == 0.5

    def test_compute_confidence_stable(self):
        stable = self.selector.compute_confidence([0.8, 0.8, 0.8, 0.8, 0.8])
        volatile = self.selector.compute_confidence([0.2, 0.9, 0.5, 1.0, 0.1])
        assert stable > volatile

    def test_compute_confidence_increasing_with_samples(self):
        c1 = self.selector.compute_confidence([0.8, 0.8])
        c2 = self.selector.compute_confidence([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
        assert c2 >= c1

    def test_compute_confidence_all_same(self):
        c = self.selector.compute_confidence([0.5, 0.5, 0.5])
        assert c > 0.0

    def test_decay_reputation_no_decay(self):
        decayed = self.selector.decay_reputation(0.9, 0.0)
        assert decayed == pytest.approx(0.9, abs=0.01)

    def test_decay_reputation_half_life(self):
        original = 0.9
        decayed = self.selector.decay_reputation(original, 90.0)
        assert decayed == pytest.approx(0.45, abs=0.01)

    def test_decay_reputation_long_inactive(self):
        decayed = self.selector.decay_reputation(0.9, 365.0)
        assert decayed < 0.9

    def test_decay_reputation_floor(self):
        decayed = self.selector.decay_reputation(0.2, 1000.0)
        assert decayed >= 0.1

    def test_rank_vessels_basic(self):
        class FakeVessel:
            def __init__(self, vid):
                self.vessel_id = vid
                self.sensor_types = []
                self.max_speed = 10.0

        vessels = [FakeVessel("v1"), FakeVessel("v2"), FakeVessel("v3")]
        reps = {
            "v1": ReputationScore(score=0.9),
            "v2": ReputationScore(score=0.7),
            "v3": ReputationScore(score=0.5),
        }
        ranked = self.selector.rank_vessels(vessels, reps)
        assert ranked[0].vessel_id == "v1"
        assert ranked[0].rank == 1
        assert ranked[2].rank == 3

    def test_rank_vessels_with_requirements(self):
        class FakeVessel:
            def __init__(self, vid, sensors):
                self.vessel_id = vid
                self.sensor_types = sensors
                self.max_speed = 10.0

        vessels = [
            FakeVessel("v1", ["sonar"]),
            FakeVessel("v2", ["sonar", "camera", "lidar"]),
        ]
        reps = {
            "v1": ReputationScore(score=0.9),
            "v2": ReputationScore(score=0.9),
        }
        ranked = self.selector.rank_vessels(vessels, reps, {"required_sensors": ["sonar", "camera"]})
        assert ranked[0].vessel_id == "v2"  # Boosted for matching sensors

    def test_rank_vessels_empty(self):
        ranked = self.selector.rank_vessels([], {})
        assert ranked == []

    def test_rank_vessels_no_reputation(self):
        class FakeVessel:
            vessel_id = "v1"
            sensor_types = []
            max_speed = 10.0

        ranked = self.selector.rank_vessels([FakeVessel()], {})
        assert len(ranked) == 1
        assert ranked[0].score == 0.5  # default

    def test_rank_vessels_speed_boost(self):
        class FakeVessel:
            def __init__(self, vid, speed):
                self.vessel_id = vid
                self.sensor_types = []
                self.max_speed = speed

        vessels = [FakeVessel("v1", 5.0), FakeVessel("v2", 20.0)]
        reps = {"v1": ReputationScore(score=0.5), "v2": ReputationScore(score=0.5)}
        ranked = self.selector.rank_vessels(vessels, reps, {"min_speed": 10})
        assert ranked[0].vessel_id == "v2"
