"""Tests for task_allocation.py — Task, ContractNetProtocol, AuctionEngine."""
import math, pytest
from jetson.swarm.task_allocation import (
    Task, TaskPriority, TaskType, TaskStatus, Bid, TaskAssignment,
    ContractNetProtocol, AuctionEngine
)

@pytest.fixture
def cnp():
    return ContractNetProtocol(bid_timeout=5.0, max_bids_per_task=10)

@pytest.fixture
def sample_task():
    return Task(id="t1", type=TaskType.PATROL, priority=TaskPriority.HIGH,
                location=(10, 20), reward=100.0)

@pytest.fixture
def sample_bid():
    return Bid(vessel_id="v1", task_id="t1", value=50.0, estimated_duration=5.0)

class TestTaskPriority:
    def test_ordering(self):
        assert TaskPriority.LOW.value < TaskPriority.MEDIUM.value
        assert TaskPriority.MEDIUM.value < TaskPriority.HIGH.value
        assert TaskPriority.HIGH.value < TaskPriority.CRITICAL.value
    def test_count(self): assert len(TaskPriority) == 4

class TestTaskType:
    def test_count(self): assert len(TaskType) == 7
    def test_has_patrol(self): assert TaskType.PATROL in TaskType

class TestTask:
    def test_create(self, sample_task):
        assert sample_task.id == "t1" and sample_task.type == TaskType.PATROL
    def test_default_status(self):
        t = Task(id="x", type=TaskType.SURVEY)
        assert t.status == TaskStatus.PENDING
    def test_distance_to(self):
        t = Task(id="t", type=TaskType.PATROL, location=(10, 0))
        assert t.distance_to(0, 0) == 10.0
    def test_distance_same(self):
        t = Task(id="t", type=TaskType.PATROL, location=(5, 5))
        assert t.distance_to(5, 5) == 0.0
    def test_requirements_default(self):
        t = Task(id="t", type=TaskType.PATROL)
        assert t.requirements == {}

class TestBid:
    def test_create(self, sample_bid):
        assert sample_bid.vessel_id == "v1" and sample_bid.value == 50.0

class TestCNPRegistration:
    def test_register(self, cnp):
        cnp.register_vessel("v1", {"speed": 5})
        assert "v1" in cnp.vessel_capabilities
    def test_unregister(self, cnp):
        cnp.register_vessel("v1", {"speed": 5})
        cnp.unregister_vessel("v1")
        assert "v1" not in cnp.vessel_capabilities
    def test_unregister_cancels_assignments(self, cnp, sample_task):
        cnp.register_vessel("v1", {"speed": 5})
        cnp.broadcast_task(sample_task)
        cnp.submit_bid(Bid("v1", "t1", 50.0))
        cnp.assign_task("t1")
        cnp.unregister_vessel("v1")
        assert "t1" not in cnp.assignments

class TestCNPBroadcast:
    def test_broadcast(self, cnp, sample_task):
        cnp.broadcast_task(sample_task)
        assert cnp.tasks["t1"].status == TaskStatus.BIDDING
    def test_broadcast_creates_bid_slot(self, cnp, sample_task):
        cnp.broadcast_task(sample_task)
        assert cnp.bids["t1"] == []

class TestCNPSubmitBid:
    def test_submit_valid(self, cnp, sample_task):
        cnp.register_vessel("v1", {"speed": 5})
        cnp.broadcast_task(sample_task)
        assert cnp.submit_bid(Bid("v1", "t1", 50.0)) is True
    def test_submit_unregistered(self, cnp, sample_task):
        cnp.broadcast_task(sample_task)
        assert cnp.submit_bid(Bid("v1", "t1", 50.0)) is False
    def test_submit_wrong_status(self, cnp, sample_task):
        cnp.register_vessel("v1", {"speed": 5})
        sample_task.status = TaskStatus.COMPLETED
        cnp.tasks["t1"] = sample_task
        assert cnp.submit_bid(Bid("v1", "t1", 50.0)) is False
    def test_submit_max_bids(self, cnp, sample_task):
        cnp.register_vessel("v1", {"speed": 5})
        cnp.broadcast_task(sample_task)
        cnp.max_bids_per_task = 2
        assert cnp.submit_bid(Bid("v1", "t1", 50.0)) is True
        assert cnp.submit_bid(Bid("v1", "t1", 60.0)) is True
        assert cnp.submit_bid(Bid("v1", "t1", 70.0)) is False

class TestCNPEvaluateBids:
    def test_best_bid(self, cnp, sample_task):
        cnp.register_vessel("v1", {"s": 5})
        cnp.register_vessel("v2", {"s": 5})
        cnp.broadcast_task(sample_task)
        cnp.submit_bid(Bid("v1", "t1", 80.0))
        cnp.submit_bid(Bid("v2", "t1", 30.0))
        best = cnp.evaluate_bids("t1")
        assert best.vessel_id == "v2"
    def test_no_bids(self, cnp, sample_task):
        cnp.broadcast_task(sample_task)
        assert cnp.evaluate_bids("t1") is None

class TestCNPAssign:
    def test_assign_winning(self, cnp, sample_task):
        cnp.register_vessel("v1", {"s": 5})
        cnp.broadcast_task(sample_task)
        cnp.submit_bid(Bid("v1", "t1", 50.0))
        a = cnp.assign_task("t1")
        assert a is not None and a.vessel_id == "v1"
    def test_assign_no_bids(self, cnp, sample_task):
        cnp.broadcast_task(sample_task)
        assert cnp.assign_task("t1") is None
    def test_assign_wrong_status(self, cnp, sample_task):
        cnp.tasks["t1"] = sample_task
        assert cnp.assign_task("t1") is None

class TestCNPProgress:
    def test_start(self, cnp, sample_task):
        cnp.register_vessel("v1", {"s": 5})
        cnp.broadcast_task(sample_task)
        cnp.submit_bid(Bid("v1", "t1", 50.0))
        cnp.assign_task("t1")
        assert cnp.start_task("t1") is True
    def test_start_wrong_status(self, cnp, sample_task):
        cnp.broadcast_task(sample_task)
        assert cnp.start_task("t1") is False
    def test_update_progress(self, cnp, sample_task):
        cnp.register_vessel("v1", {"s": 5})
        cnp.broadcast_task(sample_task)
        cnp.submit_bid(Bid("v1", "t1", 50.0))
        cnp.assign_task("t1")
        cnp.start_task("t1")
        assert cnp.update_progress("t1", 0.5) is True
        assert cnp.assignments["t1"].progress == 0.5
    def test_complete(self, cnp, sample_task):
        cnp.register_vessel("v1", {"s": 5})
        cnp.broadcast_task(sample_task)
        cnp.submit_bid(Bid("v1", "t1", 50.0))
        cnp.assign_task("t1"); cnp.start_task("t1")
        cnp.update_progress("t1", 1.0)
        assert cnp.tasks["t1"].status == TaskStatus.COMPLETED
    def test_clamp_progress(self, cnp, sample_task):
        cnp.register_vessel("v1", {"s": 5})
        cnp.broadcast_task(sample_task)
        cnp.submit_bid(Bid("v1", "t1", 50.0))
        cnp.assign_task("t1"); cnp.start_task("t1")
        cnp.update_progress("t1", 1.5)
        assert cnp.assignments["t1"].progress == 1.0

class TestCNPMonitor:
    def test_monitor_empty(self, cnp):
        assert cnp.monitor_progress() == {}
    def test_monitor_in_progress(self, cnp, sample_task):
        cnp.register_vessel("v1", {"s": 5})
        cnp.broadcast_task(sample_task)
        cnp.submit_bid(Bid("v1", "t1", 50.0))
        cnp.assign_task("t1"); cnp.start_task("t1")
        cnp.update_progress("t1", 0.3)
        m = cnp.monitor_progress()
        assert m["t1"] == 0.3

class TestCNPFailure:
    def test_handle_failure(self, cnp, sample_task):
        cnp.tasks["t1"] = sample_task
        assert cnp.handle_task_failure("t1") is True
        assert cnp.tasks["t1"].status == TaskStatus.FAILED
    def test_handle_nonexistent(self, cnp):
        assert cnp.handle_task_failure("zzz") is False
    def test_reassign_failed(self, cnp, sample_task):
        cnp.tasks["t1"] = sample_task
        cnp.handle_task_failure("t1")
        assert cnp.reassign_task("t1") is True
        assert cnp.tasks["t1"].status == TaskStatus.BIDDING
    def test_reassign_pending(self, cnp, sample_task):
        cnp.tasks["t1"] = sample_task  # status is PENDING
        assert cnp.reassign_task("t1") is True
        assert cnp.tasks["t1"].status.name == "BIDDING"
    def test_reassign_in_progress_fails(self, cnp, sample_task):
        sample_task.status = TaskStatus.IN_PROGRESS
        cnp.tasks["t1"] = sample_task
        assert cnp.reassign_task("t1") is False

class TestCNPGetters:
    def test_get_task(self, cnp, sample_task):
        cnp.broadcast_task(sample_task)
        assert cnp.get_task("t1").id == "t1"
    def test_get_nonexistent(self, cnp):
        assert cnp.get_task("zzz") is None
    def test_get_all(self, cnp, sample_task):
        cnp.broadcast_task(sample_task)
        assert len(cnp.get_all_tasks()) == 1
    def test_get_assignments(self, cnp, sample_task):
        cnp.register_vessel("v1", {"s": 5})
        cnp.broadcast_task(sample_task)
        cnp.submit_bid(Bid("v1", "t1", 50.0))
        cnp.assign_task("t1")
        assert len(cnp.get_assignments()) == 1
    def test_get_vessel_assignments(self, cnp, sample_task):
        cnp.register_vessel("v1", {"s": 5})
        cnp.broadcast_task(sample_task)
        cnp.submit_bid(Bid("v1", "t1", 50.0))
        cnp.assign_task("t1")
        assert len(cnp.get_vessel_assignments("v1")) == 1
        assert len(cnp.get_vessel_assignments("v2")) == 0

class TestAuctionEngine:
    @pytest.fixture
    def engine(self):
        return AuctionEngine(reserve_price=10.0)

    def test_create_auction(self, engine):
        engine.create_auction("a1", ["t1", "t2"])
        a = engine.get_auction("a1")
        assert a is not None and not a["closed"]

    def test_submit_valid_bid(self, engine):
        engine.create_auction("a1", ["t1"])
        bid = Bid("v1", "t1", value=20.0)
        assert engine.submit_bid("a1", bid) is True

    def test_submit_below_reserve(self, engine):
        engine.create_auction("a1", ["t1"])
        bid = Bid("v1", "t1", value=5.0)
        assert engine.submit_bid("a1", bid) is False

    def test_close_auction_winner(self, engine):
        engine.create_auction("a1", ["t1"])
        engine.submit_bid("a1", Bid("v1", "t1", 30.0))
        engine.submit_bid("a1", Bid("v2", "t1", 15.0))
        winner = engine.close_auction("a1")
        assert winner.vessel_id == "v2"

    def test_close_no_bids(self, engine):
        engine.create_auction("a1", ["t1"])
        assert engine.close_auction("a1") is None

    def test_close_nonexistent(self, engine):
        assert engine.close_auction("zzz") is None

    def test_is_closed(self, engine):
        engine.create_auction("a1", ["t1"])
        assert engine.is_closed("a1") is False
        engine.close_auction("a1")
        assert engine.is_closed("a1") is True

    def test_reserve_price(self, engine):
        engine.create_auction("a1", ["t1"], reserve=50.0)
        assert engine.reserve_prices("a1") == 50.0

    def test_set_reserve(self, engine):
        engine.create_auction("a1", ["t1"])
        assert engine.set_reserve_price("a1", 25.0) is True
        assert engine.reserve_prices("a1") == 25.0

    def test_set_reserve_closed_fails(self, engine):
        engine.create_auction("a1", ["t1"])
        engine.close_auction("a1")
        assert engine.set_reserve_price("a1", 99.0) is False

    def test_combinatorial(self, engine):
        # Non-conflicting bids: v1 covers t1, v2 covers t2
        bids = [
            Bid("v1", "t1", 30.0, capabilities={"task_ids": ["t1"]}),
            Bid("v2", "t2", 20.0, capabilities={"task_ids": ["t2"]}),
        ]
        winners = engine.combinatorial_auction("a1", bids)
        assert len(winners) == 2

    def test_combinatorial_conflict(self, engine):
        # Conflicting bids on t2: only first wins
        bids = [
            Bid("v1", "t1", 30.0, capabilities={"task_ids": ["t1", "t2"]}),
            Bid("v2", "t2", 20.0, capabilities={"task_ids": ["t2"]}),
        ]
        winners = engine.combinatorial_auction("a2", bids)
        assert len(winners) == 1  # v2 wins t2, v1 conflicts

    def test_get_result(self, engine):
        engine.create_auction("a1", ["t1"])
        engine.submit_bid("a1", Bid("v1", "t1", 20.0))
        engine.close_auction("a1")
        assert engine.get_result("a1").vessel_id == "v1"
    def test_get_result_none(self, engine):
        assert engine.get_result("zzz") is None
