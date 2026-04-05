"""
Deep unit tests for each CRDT solution.
Tests internal mechanics, edge cases, and correct CRDT semantics.
"""

import time
import copy
import pytest
from jetson.agent.fleet_sync.types import (
    FleetState, TaskItem, SkillVersion, VectorClock, TimestampedValue, SyncMetrics,
)
from jetson.agent.fleet_sync.solutions.git_sync import GitSync
from jetson.agent.fleet_sync.solutions.operation_crdt import OperationCRDT, Operation, OpType
from jetson.agent.fleet_sync.solutions.state_crdt import (
    StateCRDT, GCounter, PNCounter, LWWRegister, LWWElement,
)


# ==============================================================================
# VectorClock Tests
# ==============================================================================

class TestVectorClock:
    """Test vector clock implementation."""

    def test_create(self):
        vc = VectorClock("v0")
        assert vc.node_id == "v0"
        assert vc.clock == {}

    def test_increment(self):
        vc = VectorClock("v0")
        vc.increment()
        assert vc.clock["v0"] == 1
        vc.increment()
        assert vc.clock["v0"] == 2

    def test_merge(self):
        vc1 = VectorClock("v0")
        vc2 = VectorClock("v1")
        vc1.increment()
        vc1.increment()
        vc2.increment()
        vc1.merge(vc2)
        assert vc1.clock["v0"] == 2
        assert vc1.clock["v1"] == 1

    def test_happens_before(self):
        vc1 = VectorClock("v0")
        vc2 = VectorClock("v0")
        vc1.increment()
        vc2.increment()
        vc2.increment()
        assert vc1.happens_before(vc2) is True
        assert vc2.happens_before(vc1) is False

    def test_concurrent(self):
        vc1 = VectorClock("v0")
        vc2 = VectorClock("v1")
        vc1.increment()
        vc2.increment()
        assert vc1.happens_before(vc2) is None

    def test_copy(self):
        vc = VectorClock("v0")
        vc.increment()
        vc2 = vc.copy()
        vc2.increment()
        assert vc.clock["v0"] == 1
        assert vc2.clock["v0"] == 2

    def test_as_dict_from_dict(self):
        vc = VectorClock("v0")
        vc.increment()
        d = vc.as_dict()
        vc2 = VectorClock("v1")
        vc2.from_dict(d)
        assert vc2.clock["v0"] == 1

    def test_empty_happens_before(self):
        vc1 = VectorClock("v0")
        vc2 = VectorClock("v1")
        result = vc1.happens_before(vc2)
        assert result is None  # concurrent

    def test_equal_clocks(self):
        vc1 = VectorClock("v0")
        vc2 = VectorClock("v0")
        vc1.increment()
        vc2.increment()
        assert vc1.happens_before(vc2) is None


# ==============================================================================
# FleetState Tests
# ==============================================================================

class TestFleetState:
    """Test FleetState type."""

    def test_state_hash(self):
        s1 = FleetState("v0", trust_scores={"v1": 0.5})
        s2 = FleetState("v0", trust_scores={"v1": 0.5})
        assert s1.state_hash() == s2.state_hash()

    def test_state_hash_different(self):
        s1 = FleetState("v0", trust_scores={"v1": 0.5})
        s2 = FleetState("v0", trust_scores={"v1": 0.6})
        assert s1.state_hash() != s2.state_hash()

    def test_is_equivalent(self):
        s1 = FleetState("v0", trust_scores={"v1": 0.5})
        s2 = FleetState("v0", trust_scores={"v1": 0.5})
        assert s1.is_equivalent(s2)

    def test_not_equivalent_trust(self):
        s1 = FleetState("v0", trust_scores={"v1": 0.5})
        s2 = FleetState("v0", trust_scores={"v1": 0.6})
        assert not s1.is_equivalent(s2)

    def test_not_equivalent_tasks(self):
        s1 = FleetState("v0", task_queue=[TaskItem("t1", "desc", 5)])
        s2 = FleetState("v0", task_queue=[TaskItem("t1", "different", 5)])
        assert not s1.is_equivalent(s2)

    def test_not_equivalent_missing_trust_key(self):
        s1 = FleetState("v0", trust_scores={"v1": 0.5})
        s2 = FleetState("v0", trust_scores={"v2": 0.5})
        assert not s1.is_equivalent(s2)

    def test_not_equivalent_statuses(self):
        s1 = FleetState("v0", vessel_statuses={"v1": {"battery": 50}})
        s2 = FleetState("v0", vessel_statuses={"v1": {"battery": 60}})
        assert not s1.is_equivalent(s2)

    def test_not_equivalent_skills(self):
        s1 = FleetState("v0", skill_versions={"nav": SkillVersion("nav", 1, 0, 0)})
        s2 = FleetState("v0", skill_versions={"nav": SkillVersion("nav", 2, 0, 0)})
        assert not s1.is_equivalent(s2)

    def test_equivalent_tolerance(self):
        s1 = FleetState("v0", trust_scores={"v1": 0.50005})
        s2 = FleetState("v0", trust_scores={"v1": 0.50008})
        assert s1.is_equivalent(s2)  # Within 0.0001 tolerance


# ==============================================================================
# TimestampedValue Tests
# ==============================================================================

class TestTimestampedValue:
    """Test timestamped value."""

    def test_newer_timestamp(self):
        tv1 = TimestampedValue(1, 100.0, "v0")
        tv2 = TimestampedValue(2, 200.0, "v1")
        assert tv2.is_newer_than(tv1)

    def test_same_timestamp_vessel_tiebreak(self):
        tv1 = TimestampedValue(1, 100.0, "v0")
        tv2 = TimestampedValue(2, 100.0, "v1")
        assert tv2.is_newer_than(tv1)


# ==============================================================================
# GCounter Tests
# ==============================================================================

class TestGCounter:
    """Test grow-only counter."""

    def test_increment(self):
        gc = GCounter("v0")
        gc.increment("v0")
        gc.increment("v0")
        assert gc.value() == 2

    def test_increment_other(self):
        gc = GCounter("v0")
        gc.increment("v1", 5)
        assert gc.value() == 5

    def test_merge(self):
        gc1 = GCounter("v0")
        gc2 = GCounter("v0")
        gc1.increment("v0", 3)
        gc2.increment("v1", 5)
        merged = gc1.merge(gc2)
        assert merged.value() == 8

    def test_merge_max(self):
        gc1 = GCounter("v0")
        gc2 = GCounter("v0")
        gc1.increment("v0", 10)
        gc2.increment("v0", 5)
        merged = gc1.merge(gc2)
        assert merged.value() == 10  # max wins

    def test_copy(self):
        gc = GCounter("v0")
        gc.increment("v0", 3)
        gc2 = gc.copy()
        gc2.increment("v0", 2)
        assert gc.value() == 3
        assert gc2.value() == 5

    def test_empty_value(self):
        gc = GCounter("v0")
        assert gc.value() == 0


# ==============================================================================
# PNCounter Tests
# ==============================================================================

class TestPNCounter:
    """Test positive-negative counter."""

    def test_increment_decrement(self):
        pnc = PNCounter("v0")
        pnc.increment("v0")
        pnc.increment("v0")
        pnc.decrement("v0")
        assert pnc.p.value() == 2
        assert pnc.n.value() == 1

    def test_value(self):
        pnc = PNCounter("v0")
        pnc.increment("v0")
        pnc.increment("v0")
        pnc.decrement("v0")
        val = pnc.value({"v0": 1.0}, {"v0": 0.5})
        assert val == 0.5

    def test_merge(self):
        pnc1 = PNCounter("v0")
        pnc2 = PNCounter("v0")
        pnc1.increment("v0")
        pnc2.decrement("v0")
        merged = pnc1.merge(pnc2)
        assert merged.p.value() == 1
        assert merged.n.value() == 1

    def test_copy(self):
        pnc = PNCounter("v0")
        pnc.increment("v0")
        pnc2 = pnc.copy()
        pnc2.decrement("v0")
        assert pnc.p.value() == 1
        assert pnc.n.value() == 0
        assert pnc2.n.value() == 1

    def test_post_init(self):
        pnc = PNCounter("v0")
        assert pnc.p is not None
        assert pnc.n is not None


# ==============================================================================
# LWWRegister Tests
# ==============================================================================

class TestLWWRegister:
    """Test Last-Write-Wins register."""

    def test_set_newer(self):
        reg = LWWRegister(value=1, timestamp=100.0, vessel_id="v0")
        reg.set(2, 200.0, "v1")
        assert reg.value == 2

    def test_set_older_ignored(self):
        reg = LWWRegister(value=1, timestamp=200.0, vessel_id="v0")
        reg.set(2, 100.0, "v1")
        assert reg.value == 1

    def test_set_equal_timestamp_higher_vessel(self):
        reg = LWWRegister(value=1, timestamp=100.0, vessel_id="v0")
        reg.set(2, 100.0, "v1")
        assert reg.value == 2  # v1 > v0

    def test_merge_newer(self):
        reg1 = LWWRegister(value=1, timestamp=100.0, vessel_id="v0")
        reg2 = LWWRegister(value=2, timestamp=200.0, vessel_id="v1")
        merged = reg1.merge(reg2)
        assert merged.value == 2

    def test_merge_older(self):
        reg1 = LWWRegister(value=1, timestamp=200.0, vessel_id="v0")
        reg2 = LWWRegister(value=2, timestamp=100.0, vessel_id="v1")
        merged = reg1.merge(reg2)
        assert merged.value == 1

    def test_copy(self):
        reg = LWWRegister(value=42, timestamp=100.0, vessel_id="v0")
        reg2 = reg.copy()
        assert reg2.value == 42
        assert reg2.timestamp == 100.0
        assert reg2.vessel_id == "v0"


# ==============================================================================
# GitSync Unit Tests
# ==============================================================================

class TestGitSyncUnit:
    """Deep unit tests for GitSync."""

    def test_update_trust(self):
        gs = GitSync("v0")
        gs.update_trust("v1", 0.1)
        state = gs.get_state()
        assert state.trust_scores["v1"] == 0.6

    def test_trust_clamp_upper(self):
        gs = GitSync("v0")
        for _ in range(20):
            gs.update_trust("v1", 0.5)
        assert gs.get_state().trust_scores["v1"] <= 1.0

    def test_trust_clamp_lower(self):
        gs = GitSync("v0")
        for _ in range(20):
            gs.update_trust("v1", -0.5)
        assert gs.get_state().trust_scores["v1"] >= 0.0

    def test_add_task(self):
        gs = GitSync("v0")
        gs.add_task("t1", "Survey", 3)
        state = gs.get_state()
        assert len(state.task_queue) == 1
        assert state.task_queue[0].task_id == "t1"

    def test_update_task_status(self):
        gs = GitSync("v0")
        gs.add_task("t1", "Survey", 3)
        gs.update_task("t1", status="in_progress")
        assert gs.get_state().task_queue[0].status == "in_progress"

    def test_update_task_priority(self):
        gs = GitSync("v0")
        gs.add_task("t1", "Survey", 5)
        gs.update_task("t1", priority=1)
        assert gs.get_state().task_queue[0].priority == 1

    def test_update_vessel_status(self):
        gs = GitSync("v0")
        gs.update_vessel_status("v1", "battery", 85.0)
        assert gs.get_state().vessel_statuses["v1"]["battery"] == 85.0

    def test_update_skill_version(self):
        gs = GitSync("v0")
        gs.update_skill_version("nav", "1.2.3")
        assert gs.get_state().skill_versions["nav"].as_string() == "1.2.3"

    def test_skill_max_wins(self):
        gs = GitSync("v0")
        gs.update_skill_version("nav", "2.0.0")
        gs.update_skill_version("nav", "1.0.0")
        assert gs.get_state().skill_versions["nav"].as_string() == "2.0.0"

    def test_sync_two_vessels(self):
        v0 = GitSync("v0")
        v1 = GitSync("v1")
        v0.update_trust("v2", 0.1)
        v1.update_trust("v2", -0.05)
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        assert v0.get_state().is_equivalent(v1.get_state())

    def test_audit_trail(self):
        gs = GitSync("v0")
        gs.update_trust("v1", 0.1)
        gs.add_task("t1", "Survey", 3)
        trail = gs.get_audit_trail()
        assert len(trail) == 2
        assert trail[0]["type"] == "trust_update"
        assert trail[1]["type"] == "task_add"

    def test_sync_does_not_duplicate_tasks(self):
        v0 = GitSync("v0")
        v1 = GitSync("v1")
        v0.add_task("t1", "Survey", 3)
        v1.add_task("t1", "Survey", 3)
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        # Should have exactly one task
        assert len(v0.get_state().task_queue) == 1

    def test_multiple_sync_rounds(self):
        v0 = GitSync("v0")
        v1 = GitSync("v1")
        v0.update_trust("v2", 0.1)
        v1.update_trust("v2", -0.05)
        # Round 1
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        # Round 2
        v0.update_trust("v2", 0.02)
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        assert v0.get_state().is_equivalent(v1.get_state())


# ==============================================================================
# StateCRDT Unit Tests
# ==============================================================================

class TestStateCRDTUnit:
    """Deep unit tests for StateCRDT."""

    def test_trust_update(self):
        crdt = StateCRDT("v0")
        crdt.update_trust("v1", 0.1)
        assert crdt.get_state().trust_scores["v1"] == 0.6

    def test_trust_negative_delta(self):
        crdt = StateCRDT("v0")
        crdt.update_trust("v1", -0.1)
        assert crdt.get_state().trust_scores["v1"] == 0.4

    def test_trust_clamp(self):
        crdt = StateCRDT("v0")
        for _ in range(20):
            crdt.update_trust("v1", 0.5)
        assert crdt.get_state().trust_scores["v1"] <= 1.0

    def test_add_task(self):
        crdt = StateCRDT("v0")
        crdt.add_task("t1", "Survey", 3)
        assert len(crdt.get_state().task_queue) == 1

    def test_update_task(self):
        crdt = StateCRDT("v0")
        crdt.add_task("t1", "Survey", 5)
        crdt.update_task("t1", status="in_progress")
        assert crdt.get_state().task_queue[0].status == "in_progress"

    def test_status_update(self):
        crdt = StateCRDT("v0")
        crdt.update_vessel_status("v1", "battery", 85.0)
        assert crdt.get_state().vessel_statuses["v1"]["battery"] == 85.0

    def test_skill_update(self):
        crdt = StateCRDT("v0")
        crdt.update_skill_version("nav", "1.2.3")
        assert crdt.get_state().skill_versions["nav"].as_string() == "1.2.3"

    def test_sync_merge_trust(self):
        v0 = StateCRDT("v0")
        v1 = StateCRDT("v1")
        v0.update_trust("v2", 0.1)
        v1.update_trust("v2", -0.05)
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        assert v0.get_state().is_equivalent(v1.get_state())

    def test_sync_merge_tasks(self):
        v0 = StateCRDT("v0")
        v1 = StateCRDT("v1")
        v0.add_task("t1", "Task from v0", 3)
        v1.add_task("t2", "Task from v1", 5)
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        assert v0.get_state().is_equivalent(v1.get_state())
        task_ids = {t.task_id for t in v0.get_state().task_queue}
        assert "t1" in task_ids
        assert "t2" in task_ids

    def test_sync_merge_status(self):
        v0 = StateCRDT("v0")
        v1 = StateCRDT("v1")
        v0.update_vessel_status("v2", "battery", 80.0)
        v1.update_vessel_status("v2", "battery", 60.0)
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        assert v0.get_state().is_equivalent(v1.get_state())

    def test_sync_merge_skills(self):
        v0 = StateCRDT("v0")
        v1 = StateCRDT("v1")
        v0.update_skill_version("nav", "2.0.0")
        v1.update_skill_version("nav", "1.0.0")
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        assert v0.get_state().skill_versions["nav"].as_string() == "2.0.0"
        assert v1.get_state().skill_versions["nav"].as_string() == "2.0.0"

    def test_get_sync_payload_structure(self):
        crdt = StateCRDT("v0")
        crdt.update_trust("v1", 0.1)
        payload = crdt.get_sync_payload()
        assert "vessel_id" in payload
        assert "trust_crdt" in payload
        assert "task_crdt" in payload
        assert "status_crdt" in payload
        assert "skill_crdt" in payload

    def test_receive_sync_returns_conflicts(self):
        v0 = StateCRDT("v0")
        v1 = StateCRDT("v1")
        v0.update_trust("v2", 0.1)
        v1.update_trust("v2", -0.05)
        p01 = v0.get_sync_payload()
        conflicts = v1.receive_sync(p01, "v0")
        assert conflicts >= 0

    def test_tombstone_count(self):
        crdt = StateCRDT("v0")
        crdt.add_task("t1", "Task", 5)
        assert crdt.get_tombstone_count() == 0

    def test_get_memory_usage(self):
        crdt = StateCRDT("v0")
        crdt.update_trust("v1", 0.1)
        mem = crdt.get_memory_usage()
        assert mem > 0

    def test_materialize_tasks_sorting(self):
        crdt = StateCRDT("v0")
        crdt.add_task("t-low", "Low", 8)
        crdt.add_task("t-high", "High", 1)
        crdt.add_task("t-med", "Med", 4)
        tasks = crdt.get_state().task_queue
        assert tasks[0].task_id == "t-high"
        assert tasks[1].task_id == "t-med"
        assert tasks[2].task_id == "t-low"

    def test_empty_sync(self):
        v0 = StateCRDT("v0")
        v1 = StateCRDT("v1")
        conflicts = v0.receive_sync(v1.get_sync_payload(), "v1")
        assert conflicts == 0


# ==============================================================================
# OperationCRDT Unit Tests
# ==============================================================================

class TestOperationCRDTUnit:
    """Deep unit tests for OperationCRDT."""

    def test_trust_delta(self):
        crdt = OperationCRDT("v0")
        crdt.update_trust("v1", 0.1)
        assert crdt.get_state().trust_scores["v1"] == 0.6

    def test_trust_clamp(self):
        crdt = OperationCRDT("v0")
        for _ in range(20):
            crdt.update_trust("v1", 0.5)
        assert crdt.get_state().trust_scores["v1"] <= 1.0

    def test_trust_negative(self):
        crdt = OperationCRDT("v0")
        for _ in range(20):
            crdt.update_trust("v1", -0.5)
        assert crdt.get_state().trust_scores["v1"] >= 0.0

    def test_add_task(self):
        crdt = OperationCRDT("v0")
        crdt.add_task("t1", "Survey", 3)
        assert len(crdt.get_state().task_queue) == 1

    def test_update_task(self):
        crdt = OperationCRDT("v0")
        crdt.add_task("t1", "Survey", 5)
        crdt.update_task("t1", status="completed")
        assert crdt.get_state().task_queue[0].status == "completed"

    def test_status_update(self):
        crdt = OperationCRDT("v0")
        crdt.update_vessel_status("v1", "battery", 85.0)
        assert crdt.get_state().vessel_statuses["v1"]["battery"] == 85.0

    def test_skill_upgrade(self):
        crdt = OperationCRDT("v0")
        crdt.update_skill_version("nav", "1.2.3")
        assert crdt.get_state().skill_versions["nav"].as_string() == "1.2.3"

    def test_skill_no_downgrade(self):
        crdt = OperationCRDT("v0")
        crdt.update_skill_version("nav", "2.0.0")
        crdt.update_skill_version("nav", "1.0.0")
        assert crdt.get_state().skill_versions["nav"].as_string() == "2.0.0"

    def test_sync_payload_structure(self):
        crdt = OperationCRDT("v0")
        crdt.update_trust("v1", 0.1)
        payload = crdt.get_sync_payload()
        assert "vessel_id" in payload
        assert "vector_clock" in payload
        assert "operations" in payload

    def test_operation_deduplication(self):
        v0 = OperationCRDT("v0")
        v1 = OperationCRDT("v1")
        v0.update_trust("v2", 0.1)
        p01 = v0.get_sync_payload()
        # Apply same payload twice
        v1.receive_sync(p01, "v0")
        v1.receive_sync(p01, "v0")
        # Trust should only be applied once
        trust = v1.get_state().trust_scores.get("v2", 0.5)
        assert abs(trust - 0.6) < 0.01

    def test_compact_operation_log(self):
        crdt = OperationCRDT("v0", compact_threshold=5)
        for i in range(10):
            crdt.update_trust("v1", 0.01)
        initial = crdt.get_operation_count()
        crdt.compact_operation_log()
        assert crdt.get_operation_count() <= initial

    def test_get_operation_count(self):
        crdt = OperationCRDT("v0")
        assert crdt.get_operation_count() == 0
        crdt.update_trust("v1", 0.1)
        assert crdt.get_operation_count() == 1

    def test_vector_clock_advances(self):
        crdt = OperationCRDT("v0")
        crdt.update_trust("v1", 0.1)
        crdt.update_trust("v1", 0.1)
        payload = crdt.get_sync_payload()
        assert payload["vector_clock"]["v0"] >= 2

    def test_bidirectional_sync_trust(self):
        v0 = OperationCRDT("v0")
        v1 = OperationCRDT("v1")
        v0.update_trust("v2", 0.1)
        v1.update_trust("v2", -0.05)
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        # Trust deltas should be additive and converge
        assert v0.get_state().is_equivalent(v1.get_state())

    def test_multiple_trust_updates_converge(self):
        v0 = OperationCRDT("v0")
        v1 = OperationCRDT("v1")
        for i in range(10):
            v0.update_trust("v2", 0.01)
            v1.update_trust("v2", 0.01)
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        assert v0.get_state().is_equivalent(v1.get_state())

    def test_status_lww_ordering(self):
        v0 = OperationCRDT("v0")
        v1 = OperationCRDT("v1")
        v0.update_vessel_status("v2", "battery", 80.0)
        time.sleep(0.001)  # Ensure different timestamps
        v1.update_vessel_status("v2", "battery", 60.0)
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        assert v0.get_state().is_equivalent(v1.get_state())


# ==============================================================================
# TrustScore Tests
# ==============================================================================

class TestTrustScore:
    """Test TrustScore type."""

    def test_clamp(self):
        from jetson.agent.fleet_sync.types import TrustScore
        ts = TrustScore("v0", 1.5)
        ts.clamp()
        assert ts.score == 1.0

    def test_clamp_negative(self):
        from jetson.agent.fleet_sync.types import TrustScore
        ts = TrustScore("v0", -0.5)
        ts.clamp()
        assert ts.score == 0.0

    def test_clamp_valid(self):
        from jetson.agent.fleet_sync.types import TrustScore
        ts = TrustScore("v0", 0.7)
        ts.clamp()
        assert ts.score == 0.7


# ==============================================================================
# SkillVersion Tests
# ==============================================================================

class TestSkillVersion:
    """Test SkillVersion type."""

    def test_comparison(self):
        v1 = SkillVersion("nav", 1, 0, 0)
        v2 = SkillVersion("nav", 2, 0, 0)
        assert v2 > v1
        assert v1 < v2

    def test_equality(self):
        v1 = SkillVersion("nav", 1, 2, 3)
        v2 = SkillVersion("nav", 1, 2, 3)
        assert v1 == v2

    def test_inequality_different_types(self):
        v = SkillVersion("nav", 1, 0, 0)
        assert v != "nav 1.0.0"

    def test_as_tuple(self):
        v = SkillVersion("nav", 1, 2, 3)
        assert v.as_tuple() == (1, 2, 3)

    def test_as_string(self):
        v = SkillVersion("nav", 1, 2, 3)
        assert v.as_string() == "1.2.3"

    def test_hash(self):
        v1 = SkillVersion("nav", 1, 2, 3)
        v2 = SkillVersion("nav", 1, 2, 3)
        assert hash(v1) == hash(v2)


# ==============================================================================
# SyncMetrics Tests
# ==============================================================================

class TestSyncMetrics:
    """Test SyncMetrics type."""

    def test_defaults(self):
        m = SyncMetrics()
        assert m.convergence_correct is False
        assert m.data_loss_count == 0
        assert m.conflict_count == 0


# ==============================================================================
# TaskItem Tests
# ==============================================================================

class TestTaskItem:
    """Test TaskItem type."""

    def test_key(self):
        t = TaskItem("t1", "Survey", 5)
        assert t.key() == "t1"

    def test_defaults(self):
        t = TaskItem("t1", "Survey")
        assert t.priority == 5
        assert t.assigned_to == ""
        assert t.status == "pending"
