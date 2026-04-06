"""Tests for task_market module."""

import pytest
from datetime import datetime, timedelta

from jetson.marketplace.task_market import (
    TaskPost, Bid, MarketStatus, TaskMarket,
)


class TestTaskPost:
    def test_default_creation(self):
        t = TaskPost()
        assert t.description == ""
        assert t.reward == 0.0
        assert t.poster_id == ""
        assert t.status == MarketStatus.OPEN

    def test_custom_creation(self):
        deadline = datetime.utcnow() + timedelta(days=7)
        t = TaskPost(
            description="Survey task",
            reward=5000.0,
            poster_id="org-1",
            deadline=deadline,
            requirements={"min_speed": 10},
        )
        assert t.description == "Survey task"
        assert t.reward == 5000.0
        assert t.poster_id == "org-1"
        assert t.deadline == deadline
        assert t.requirements["min_speed"] == 10

    def test_auto_id(self):
        t1 = TaskPost()
        t2 = TaskPost()
        assert t1.id != t2.id

    def test_created_at(self):
        before = datetime.utcnow()
        t = TaskPost()
        after = datetime.utcnow()
        assert before <= t.created_at <= after

    def test_location_default(self):
        t = TaskPost()
        assert t.location == {"lat": 0.0, "lon": 0.0}

    def test_location_custom(self):
        t = TaskPost(location={"lat": 45.0, "lon": -70.0})
        assert t.location["lat"] == 45.0

    def test_metadata(self):
        t = TaskPost(metadata={"priority": "high"})
        assert t.metadata["priority"] == "high"


class TestBid:
    def test_default_creation(self):
        b = Bid()
        assert b.amount == 0.0
        assert b.bidder_id == ""
        assert b.accepted is False

    def test_custom_creation(self):
        eta = datetime.utcnow() + timedelta(hours=24)
        b = Bid(
            task_id="t1",
            bidder_id="v1",
            amount=3000.0,
            eta=eta,
            proposal="Fast delivery",
        )
        assert b.task_id == "t1"
        assert b.bidder_id == "v1"
        assert b.amount == 3000.0
        assert b.eta == eta

    def test_capability_scores(self):
        b = Bid(capability_scores={"speed": 0.9, "sensors": 0.8})
        assert b.capability_scores["speed"] == 0.9

    def test_auto_id(self):
        b1 = Bid()
        b2 = Bid()
        assert b1.id != b2.id


class TestMarketStatus:
    def test_enum_values(self):
        assert MarketStatus.OPEN.value == "OPEN"
        assert MarketStatus.CLOSED.value == "CLOSED"
        assert MarketStatus.PENDING_ASSIGNMENT.value == "PENDING_ASSIGNMENT"
        assert MarketStatus.COMPLETED.value == "COMPLETED"
        assert MarketStatus.CANCELLED.value == "CANCELLED"


class TestTaskMarket:
    def setup_method(self):
        self.market = TaskMarket()

    def test_post_task(self):
        t = TaskPost(description="Test task", reward=1000.0)
        task_id = self.market.post_task(t)
        assert task_id == t.id
        assert self.market.get_task(task_id) is not None

    def test_post_task_sets_open(self):
        t = TaskPost()
        self.market.post_task(t)
        assert t.status == MarketStatus.OPEN

    def test_post_multiple_tasks(self):
        t1 = TaskPost(description="Task 1")
        t2 = TaskPost(description="Task 2")
        id1 = self.market.post_task(t1)
        id2 = self.market.post_task(t2)
        assert id1 != id2
        assert len(self.market.get_all_tasks()) == 2

    def test_submit_bid(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        b = Bid(task_id=t.id, bidder_id="v1", amount=800.0)
        assert self.market.submit_bid(t.id, b) is True

    def test_submit_bid_rejects_nonexistent_task(self):
        b = Bid(task_id="fake", bidder_id="v1", amount=500.0)
        assert self.market.submit_bid("fake", b) is False

    def test_submit_bid_rejects_closed_task(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        self.market.close_market(t.id)
        b = Bid(task_id=t.id, bidder_id="v1", amount=500.0)
        assert self.market.submit_bid(t.id, b) is False

    def test_submit_bid_prevents_duplicate_bidder(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        b1 = Bid(task_id=t.id, bidder_id="v1", amount=800.0)
        b2 = Bid(task_id=t.id, bidder_id="v1", amount=700.0)
        self.market.submit_bid(t.id, b1)
        assert self.market.submit_bid(t.id, b2) is False

    def test_submit_bid_auto_sets_task_id(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        b = Bid(bidder_id="v1", amount=800.0)
        self.market.submit_bid(t.id, b)
        assert b.task_id == t.id

    def test_get_task_bids(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        b1 = Bid(task_id=t.id, bidder_id="v1", amount=800.0)
        b2 = Bid(task_id=t.id, bidder_id="v2", amount=900.0)
        self.market.submit_bid(t.id, b1)
        self.market.submit_bid(t.id, b2)
        bids = self.market.get_task_bids(t.id)
        assert len(bids) == 2

    def test_get_task_bids_empty(self):
        bids = self.market.get_task_bids("nonexistent")
        assert bids == []

    def test_close_market_selects_lowest(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        b1 = Bid(task_id=t.id, bidder_id="v1", amount=900.0)
        b2 = Bid(task_id=t.id, bidder_id="v2", amount=700.0)
        b3 = Bid(task_id=t.id, bidder_id="v3", amount=800.0)
        self.market.submit_bid(t.id, b1)
        self.market.submit_bid(t.id, b2)
        self.market.submit_bid(t.id, b3)
        winner = self.market.close_market(t.id)
        assert winner is not None
        assert winner.bidder_id == "v2"
        assert winner.accepted is True

    def test_close_market_no_bids(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        winner = self.market.close_market(t.id)
        assert winner is None
        assert self.market.get_task(t.id).status == MarketStatus.PENDING_ASSIGNMENT

    def test_close_market_nonexistent(self):
        assert self.market.close_market("fake") is None

    def test_close_market_already_closed(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        self.market.close_market(t.id)
        result = self.market.close_market(t.id)
        assert result is None

    def test_evaluate_bids(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        b1 = Bid(task_id=t.id, bidder_id="v1", amount=900.0, capability_scores={"speed": 0.8})
        b2 = Bid(task_id=t.id, bidder_id="v2", amount=700.0, capability_scores={"speed": 0.5})
        self.market.submit_bid(t.id, b1)
        self.market.submit_bid(t.id, b2)
        ranked = self.market.evaluate_bids(t.id)
        assert len(ranked) == 2

    def test_evaluate_bids_empty(self):
        t = TaskPost()
        self.market.post_task(t)
        ranked = self.market.evaluate_bids(t.id)
        assert ranked == []

    def test_evaluate_bids_custom_criteria(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        b1 = Bid(task_id=t.id, bidder_id="v1", amount=900.0, capability_scores={"speed": 1.0})
        b2 = Bid(task_id=t.id, bidder_id="v2", amount=700.0, capability_scores={"speed": 0.3})
        self.market.submit_bid(t.id, b1)
        self.market.submit_bid(t.id, b2)
        # Weight capability heavily
        ranked = self.market.evaluate_bids(t.id, {"cost": 0.1, "capability": 0.9, "eta": 0.0})
        assert ranked[0].bidder_id == "v1"  # Higher capability despite higher cost

    def test_cancel_task(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        self.market.cancel_task(t.id, "no longer needed")
        assert self.market.get_task(t.id).status == MarketStatus.CANCELLED
        assert self.market.get_task(t.id).metadata["cancel_reason"] == "no longer needed"

    def test_cancel_nonexistent(self):
        # Should not raise
        self.market.cancel_task("fake")

    def test_cancel_completed_task(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        t.status = MarketStatus.COMPLETED
        self.market.cancel_task(t.id)
        # Should remain completed
        assert t.status == MarketStatus.COMPLETED

    def test_get_open_tasks(self):
        t1 = TaskPost(description="Open task", reward=1000.0)
        t2 = TaskPost(description="Another open", reward=2000.0)
        self.market.post_task(t1)
        self.market.post_task(t2)
        open_tasks = self.market.get_open_tasks()
        assert len(open_tasks) == 2

    def test_get_open_tasks_excludes_closed(self):
        t1 = TaskPost(reward=1000.0)
        t2 = TaskPost(reward=2000.0)
        self.market.post_task(t1)
        self.market.post_task(t2)
        self.market.close_market(t1.id)
        open_tasks = self.market.get_open_tasks()
        assert len(open_tasks) == 1

    def test_get_open_tasks_excludes_cancelled(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        self.market.cancel_task(t.id)
        assert len(self.market.get_open_tasks()) == 0

    def test_get_task(self):
        t = TaskPost(reward=1000.0)
        tid = self.market.post_task(t)
        assert self.market.get_task(tid) is t
        assert self.market.get_task("fake") is None

    def test_get_all_tasks(self):
        t1 = TaskPost()
        t2 = TaskPost()
        self.market.post_task(t1)
        self.market.post_task(t2)
        assert len(self.market.get_all_tasks()) == 2

    def test_full_workflow(self):
        t = TaskPost(description="Survey", reward=5000.0, poster_id="org-1")
        tid = self.market.post_task(t)
        b1 = Bid(task_id=tid, bidder_id="v1", amount=4000.0)
        b2 = Bid(task_id=tid, bidder_id="v2", amount=3500.0)
        self.market.submit_bid(tid, b1)
        self.market.submit_bid(tid, b2)
        open = self.market.get_open_tasks()
        assert len(open) == 1
        winner = self.market.close_market(tid)
        assert winner.bidder_id == "v2"
        assert len(self.market.get_open_tasks()) == 0

    def test_evaluate_bids_with_eta(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        now = datetime.utcnow()
        b1 = Bid(task_id=t.id, bidder_id="v1", amount=800.0, eta=now + timedelta(hours=48))
        b2 = Bid(task_id=t.id, bidder_id="v2", amount=800.0, eta=now + timedelta(hours=12))
        self.market.submit_bid(t.id, b1)
        self.market.submit_bid(t.id, b2)
        ranked = self.market.evaluate_bids(t.id, {"cost": 0.0, "eta": 1.0, "capability": 0.0})
        assert ranked[0].bidder_id == "v2"

    def test_cancelled_task_rejects_bid(self):
        t = TaskPost(reward=1000.0)
        self.market.post_task(t)
        self.market.cancel_task(t.id)
        b = Bid(task_id=t.id, bidder_id="v1", amount=500.0)
        assert self.market.submit_bid(t.id, b) is False
