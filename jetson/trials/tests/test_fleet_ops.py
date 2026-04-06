"""Pytest integration tests for Fleet Operations cross-module trials.

Each test calls 2+ real fleet_coordination modules working together:
FleetManager, TaskOrchestrator, ConsensusProtocol, FleetCommunication.
"""
import pytest
import random

from jetson.fleet_coordination.fleet_manager import (
    FleetManager, VesselStatus, FleetState, AnomalyRecord,
)
from jetson.fleet_coordination.task_orchestration import (
    TaskOrchestrator, FleetTask, TaskType, TaskStatus, TaskRequirement,
    WorkloadAssignment,
)
from jetson.fleet_coordination.consensus import (
    ConsensusProtocol, Proposal, Vote, ConsensusResult,
)
from jetson.fleet_coordination.communication import (
    FleetCommunication, FleetMessage, MessageType, DeliveryStatus, LinkStatus,
    BroadcastResult,
)


# ─── Helpers ───────────────────────────────────────────────────────────────

def _vinfo(vid, pos=None, fuel=100.0, health=1.0, trust=1.0, available=True):
    """Build a vessel_info dict for FleetManager.register_vessel."""
    return {
        "vessel_id": vid,
        "position": pos or (0.0, 0.0),
        "heading": 0.0,
        "speed": 5.0,
        "fuel": fuel,
        "health": health,
        "trust_score": trust,
        "available": available,
    }


def _msg(src, tgt, mtype=MessageType.STATUS, payload=None, priority=0.7, ttl=10):
    """Build a FleetMessage."""
    return FleetMessage(source=src, target=tgt, type=mtype,
                        payload=payload, priority=priority, ttl=ttl)


def _task(tid=None, ttype=TaskType.PATROL, priority=0.5, status=None, progress=None):
    """Build a FleetTask."""
    kw = {"type": ttype, "priority": priority}
    if tid:
        kw["id"] = tid
    if status:
        kw["status"] = status
    if progress is not None:
        kw["progress"] = progress
    return FleetTask(**kw)


# ═══════════════════════════════════════════════════════════════════════════
# 1. FleetManager + TaskOrchestrator
# ═══════════════════════════════════════════════════════════════════════════

class TestFleetManagerTaskOrchestration:
    """Cross-module: FleetManager + TaskOrchestrator."""

    def setup_method(self):
        self.fm = FleetManager()
        self.to = TaskOrchestrator()
        self.fm.register_vessel(_vinfo("v1", (10, 20), health=1.0, fuel=90))
        self.fm.register_vessel(_vinfo("v2", (30, 40), health=0.9, fuel=80))
        self.fm.register_vessel(_vinfo("v3", (50, 60), health=0.8, fuel=70))

    def test_register_vessels(self):
        assert len(self.fm.get_all_vessels()) == 3
        assert self.fm.get_vessel("v1") is not None

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError):
            self.fm.register_vessel(_vinfo("v1"))

    def test_register_missing_id_raises(self):
        with pytest.raises(ValueError):
            self.fm.register_vessel({})

    def test_deregister_vessel(self):
        assert self.fm.deregister_vessel("v3") is True
        assert self.fm.get_vessel("v3") is None
        assert len(self.fm.get_all_vessels()) == 2

    def test_deregister_nonexistent(self):
        assert self.fm.deregister_vessel("ghost") is False

    def test_update_vessel_status(self):
        ok = self.fm.update_vessel_status("v1", {"fuel": 50.0, "health": 0.8})
        assert ok is True
        v = self.fm.get_vessel("v1")
        assert v.fuel == 50.0
        assert v.health == 0.8

    def test_update_nonexistent_vessel(self):
        assert self.fm.update_vessel_status("ghost", {"fuel": 50}) is False

    def test_update_fuel_clamped(self):
        self.fm.update_vessel_status("v1", {"fuel": 200.0})
        assert self.fm.get_vessel("v1").fuel == 100.0
        self.fm.update_vessel_status("v1", {"fuel": -10.0})
        assert self.fm.get_vessel("v1").fuel == 0.0

    def test_update_health_clamped(self):
        self.fm.update_vessel_status("v1", {"health": 5.0})
        assert self.fm.get_vessel("v1").health == 1.0
        self.fm.update_vessel_status("v1", {"health": -1.0})
        assert self.fm.get_vessel("v1").health == 0.0

    def test_submit_task(self):
        task = _task(ttype=TaskType.PATROL, priority=0.8)
        tid = self.to.submit_task(task)
        assert tid == task.id
        assert self.to.get_task_status(tid).status == TaskStatus.PENDING

    def test_submit_duplicate_task_raises(self):
        task = _task(tid="fixed_id")
        self.to.submit_task(task)
        with pytest.raises(ValueError):
            self.to.submit_task(_task(tid="fixed_id"))

    def test_assign_vessels(self):
        task = _task(ttype=TaskType.PATROL, priority=0.7)
        self.to.submit_task(task)
        ok = self.to.assign_vessels(task.id, ["v1", "v2"])
        assert ok is True
        assert task.status == TaskStatus.ASSIGNED

    def test_assign_to_completed_task_fails(self):
        task = _task()
        self.to.submit_task(task)
        task.status = TaskStatus.COMPLETED
        ok = self.to.assign_vessels(task.id, ["v1"])
        assert ok is False

    def test_assign_to_cancelled_task_fails(self):
        task = _task()
        self.to.submit_task(task)
        task.status = TaskStatus.CANCELLED
        ok = self.to.assign_vessels(task.id, ["v1"])
        assert ok is False

    def test_cancel_frees_vessels(self):
        task = _task()
        self.to.submit_task(task)
        self.to.assign_vessels(task.id, ["v1", "v2"])
        self.to.cancel_task(task.id)
        assert task.status == TaskStatus.CANCELLED
        assert self.to.get_active_task_count("v1") == 0
        assert self.to.get_active_task_count("v2") == 0

    def test_cancel_nonexistent(self):
        assert self.to.cancel_task("ghost") is False

    def test_reassign_task(self):
        task = _task()
        self.to.submit_task(task)
        self.to.assign_vessels(task.id, ["v1"])
        self.to.reassign_task(task.id, ["v2"])
        assert self.to.get_active_task_count("v2") == 1
        assert self.to.get_active_task_count("v1") == 0

    def test_reassign_clears_old(self):
        task = _task()
        self.to.submit_task(task)
        self.to.assign_vessels(task.id, ["v1", "v2"])
        self.to.reassign_task(task.id, ["v3"])
        assert self.to.get_active_task_count("v1") == 0
        assert self.to.get_active_task_count("v2") == 0
        assert self.to.get_active_task_count("v3") == 1

    def test_get_tasks_for_vessel(self):
        t1 = _task()
        t2 = _task()
        self.to.submit_task(t1)
        self.to.submit_task(t2)
        self.to.assign_vessels(t1.id, ["v1"])
        self.to.assign_vessels(t2.id, ["v1"])
        tasks = self.to.get_tasks_for_vessel("v1")
        assert len(tasks) == 2

    def test_get_tasks_for_idle_vessel(self):
        assert len(self.to.get_tasks_for_vessel("v1")) == 0

    def test_get_all_tasks(self):
        self.to.submit_task(_task())
        self.to.submit_task(_task())
        assert len(self.to.get_all_tasks()) == 2

    def test_workload_balance(self):
        for i in range(3):
            self.to.submit_task(_task(priority=0.5 + i * 0.1))
        assignments = self.to.balance_workload(
            self.to.get_all_tasks(), self.fm.get_available_vessels())
        assert isinstance(assignments, list)
        # Each assignment should reference valid tasks
        for a in assignments:
            assert a.task_id in [t.id for t in self.to.get_all_tasks()]

    def test_workload_balance_skips_completed(self):
        t1 = _task()
        self.to.submit_task(t1)
        t1.status = TaskStatus.COMPLETED
        assignments = self.to.balance_workload(
            self.to.get_all_tasks(), self.fm.get_available_vessels())
        assert all(a.task_id != t1.id for a in assignments)

    def test_dynamic_priority_uses_fleet_state(self):
        task = _task(ttype=TaskType.RESCUE, priority=0.9)
        self.to.submit_task(task)
        snap = self.fm.get_fleet_snapshot()
        prio = self.to.compute_task_priority(task, snap)
        assert prio >= 0.9  # rescue type boost

    def test_deadlock_vessel_contention(self):
        t1 = _task()
        t2 = _task()
        self.to.submit_task(t1)
        self.to.submit_task(t2)
        self.to.assign_vessels(t1.id, ["v1"])
        self.to.assign_vessels(t2.id, ["v1"])  # same vessel
        deadlocks = self.to.detect_deadlocks(self.to.get_all_tasks())
        assert any(d["type"] == "vessel_contention" for d in deadlocks)

    def test_deadlock_orphaned_task(self):
        task = _task()
        self.to.submit_task(task)
        task.status = TaskStatus.IN_PROGRESS
        # No vessels assigned to in-progress task
        deadlocks = self.to.detect_deadlocks(self.to.get_all_tasks())
        assert any(d["type"] == "orphaned_task" for d in deadlocks)

    def test_eta_estimation(self):
        task = _task(progress=0.0)
        vessels = [self.fm.get_vessel("v1"), self.fm.get_vessel("v2")]
        eta = self.to.estimate_completion(task, vessels)
        assert eta is not None
        assert eta > 0

    def test_eta_completed_task(self):
        task = _task(progress=1.0)
        vessels = [self.fm.get_vessel("v1")]
        assert self.to.estimate_completion(task, vessels) == 0.0

    def test_eta_no_vessels(self):
        task = _task(progress=0.0)
        assert self.to.estimate_completion(task, []) is None

    def test_fleet_health(self):
        h = self.fm.compute_fleet_health()
        assert 0 < h <= 1.0

    def test_empty_fleet_health(self):
        fm = FleetManager()
        assert fm.compute_fleet_health() == 0.0

    def test_fleet_snapshot(self):
        snap = self.fm.get_fleet_snapshot()
        assert isinstance(snap, FleetState)
        assert len(snap.vessels) == 3

    def test_available_vessels(self):
        avail = self.fm.get_available_vessels()
        assert len(avail) == 3
        self.fm.update_vessel_status("v1", {"available": False})
        assert len(self.fm.get_available_vessels()) == 2

    def test_task_with_requirements(self):
        task = FleetTask(
            type=TaskType.INSPECTION,
            requirements=[TaskRequirement(skill="camera", min_count=2)])
        self.to.submit_task(task)
        assert len(task.requirements) == 1
        assert task.requirements[0].skill == "camera"


# ═══════════════════════════════════════════════════════════════════════════
# 2. FleetManager + ConsensusProtocol
# ═══════════════════════════════════════════════════════════════════════════

class TestFleetManagerConsensus:
    """Cross-module: FleetManager + ConsensusProtocol."""

    def setup_method(self):
        self.fm = FleetManager()
        self.fm.register_vessel(_vinfo("v1", trust=0.95))
        self.fm.register_vessel(_vinfo("v2", trust=0.85))
        self.fm.register_vessel(_vinfo("v3", trust=0.75))
        self.cp = ConsensusProtocol()

    def test_raft_elect_returns_leader(self):
        leader, term = self.cp.raft_elect("v1", self.fm.get_all_vessels())
        assert leader in ("v1", "v2", "v3")
        assert term >= 1

    def test_raft_elect_increments_term(self):
        _, t1 = self.cp.raft_elect("v1", self.fm.get_all_vessels())
        _, t2 = self.cp.raft_elect("v1", self.fm.get_all_vessels())
        assert t2 == t1 + 1

    def test_raft_elect_empty_fleet(self):
        leader, term = self.cp.raft_elect("v1", [])
        assert leader is None
        assert term >= 1

    def test_raft_leader_set(self):
        leader, _ = self.cp.raft_elect("v1", self.fm.get_all_vessels())
        assert self.cp.leader_id == leader

    def test_raft_propose(self):
        leader, _ = self.cp.raft_elect("v1", self.fm.get_all_vessels())
        result = self.cp.raft_propose("go_north", leader, self.fm.get_all_vessels())
        assert isinstance(result, ConsensusResult)
        assert len(result.participating_nodes) > 0

    def test_raft_propose_single_vessel(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("solo", trust=1.0))
        cp = ConsensusProtocol()
        result = cp.raft_propose("go", "solo", fm.get_all_vessels())
        assert isinstance(result, ConsensusResult)
        assert result.consensus_reached is True

    def test_paxos_prepare(self):
        proposal = Proposal(proposer="v1", value="patrol", round_number=1)
        promises = self.cp.paxos_prepare(proposal, ["v1", "v2", "v3"])
        assert isinstance(promises, int)
        assert promises >= 0

    def test_paxos_prepare_empty_acceptors(self):
        p = Proposal(proposer="v1", value="x", round_number=1)
        assert self.cp.paxos_prepare(p, []) == 0

    def test_paxos_accept(self):
        p = Proposal(proposer="v1", value="patrol", round_number=1)
        promises = self.cp.paxos_prepare(p, ["v1", "v2"])
        accepted = self.cp.paxos_accept(p, promises)
        assert isinstance(accepted, bool)

    def test_paxos_accept_zero_promises(self):
        p = Proposal(proposer="v1", value="x", round_number=1)
        assert self.cp.paxos_accept(p, 0) is False

    def test_split_brain_two_partitions(self):
        assert self.cp.detect_split_brain([["v1", "v2"], ["v3"]]) is True

    def test_split_brain_single_partition(self):
        assert self.cp.detect_split_brain([["v1", "v2", "v3"]]) is False

    def test_split_brain_empty_partitions(self):
        assert self.cp.detect_split_brain([[], []]) is False

    def test_merkle_deterministic(self):
        h1 = self.cp.merkle_tree_hash({"x": 1, "y": 2})
        h2 = self.cp.merkle_tree_hash({"x": 1, "y": 2})
        assert h1 == h2

    def test_merkle_different_states(self):
        h1 = self.cp.merkle_tree_hash({"x": 1})
        h2 = self.cp.merkle_tree_hash({"x": 2})
        assert h1 != h2

    def test_merkle_empty_state(self):
        h = self.cp.merkle_tree_hash({})
        assert isinstance(h, str)
        assert len(h) > 0

    def test_merkle_nested_dict(self):
        h1 = self.cp.merkle_tree_hash({"v1": {"pos": [1, 2]}, "v2": {"pos": [3, 4]}})
        h2 = self.cp.merkle_tree_hash({"v2": {"pos": [3, 4]}, "v1": {"pos": [1, 2]}})
        assert h1 == h2  # sorted keys

    def test_consensus_log(self):
        self.cp.raft_elect("v1", self.fm.get_all_vessels())
        log = self.cp.log
        assert isinstance(log, list)

    def test_consensus_reset(self):
        self.cp.raft_elect("v1", self.fm.get_all_vessels())
        self.cp.reset()
        assert self.cp.current_term == 0
        assert self.cp.leader_id is None
        assert len(self.cp.log) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. FleetManager + FleetCommunication
# ═══════════════════════════════════════════════════════════════════════════

class TestFleetManagerCommunication:
    """Cross-module: FleetManager + FleetCommunication."""

    def setup_method(self):
        self.fm = FleetManager()
        self.fc = FleetCommunication()
        self.fm.register_vessel(_vinfo("v1", (0, 0)))
        self.fm.register_vessel(_vinfo("v2", (100, 100)))
        self.fm.register_vessel(_vinfo("v3", (200, 200)))

    def test_broadcast_to_fleet(self):
        msg = _msg("v1", "broadcast", MessageType.STATUS, "hello")
        vessels = self.fm.get_all_vessels()
        br = self.fc.broadcast("v1", msg, vessels)
        assert isinstance(br, BroadcastResult)
        total = len(br.reached_vessels) + len(br.failed_vessels)
        assert total == len(vessels) - 1  # exclude source

    def test_multicast_to_subset(self):
        msg = _msg("v1", "group", MessageType.COMMAND)
        mr = self.fc.multicast("v1", msg, ["v2", "v3"])
        assert isinstance(mr, BroadcastResult)
        total = len(mr.reached_vessels) + len(mr.failed_vessels)
        assert total == 2

    def test_relay_message(self):
        msg = _msg("v1", "v3", MessageType.DATA)
        ok = self.fc.relay_message(msg, ["v2", "v3"])
        assert isinstance(ok, bool)

    def test_relay_empty_route(self):
        msg = _msg("v1", "v2")
        assert self.fc.relay_message(msg, []) is False

    def test_send_message(self):
        msg = _msg("v1", "v2", MessageType.STATUS, "ping")
        status = self.fc.send(msg)
        assert isinstance(status, DeliveryStatus)

    def test_send_expired_ttl(self):
        msg = _msg("v1", "v2", ttl=0)
        status = self.fc.send(msg)
        assert status == DeliveryStatus.EXPIRED

    def test_send_high_priority(self):
        msg = _msg("v1", "v2", priority=1.0, ttl=5)
        status = self.fc.send(msg)
        assert isinstance(status, DeliveryStatus)

    def test_add_link(self):
        link = self.fc.add_link("v1", "v2", latency=10.0, bandwidth=50.0)
        assert isinstance(link, LinkStatus)
        assert link.active is True

    def test_get_link(self):
        self.fc.add_link("v1", "v2", latency=15.0)
        link = self.fc.get_link("v1", "v2")
        assert link is not None
        assert link.latency == 15.0

    def test_get_link_nonexistent(self):
        assert self.fc.get_link("v1", "v2") is None

    def test_remove_link(self):
        self.fc.add_link("v1", "v2")
        assert self.fc.remove_link("v1", "v2") is True
        assert self.fc.get_link("v1", "v2") is None

    def test_remove_nonexistent_link(self):
        assert self.fc.remove_link("v1", "v2") is False

    def test_network_health_all_good(self):
        self.fc.add_link("v1", "v2", latency=5.0, bandwidth=100.0, packet_loss=0.0)
        link = self.fc.get_link("v1", "v2")
        health = self.fc.estimate_network_health([link])
        assert health > 0.5

    def test_network_health_bad_link(self):
        self.fc.add_link("v1", "v2", latency=400.0, bandwidth=10.0, packet_loss=0.5)
        link = self.fc.get_link("v1", "v2")
        health = self.fc.estimate_network_health([link])
        assert health < 1.0

    def test_network_health_empty(self):
        assert self.fc.estimate_network_health([]) == 0.0

    def test_network_health_all_inactive(self):
        self.fc.add_link("v1", "v2")
        link = self.fc.get_link("v1", "v2")
        link.active = False
        assert self.fc.estimate_network_health([link]) == 0.0

    def test_routing_table(self):
        graph = {"v1": ["v2"], "v2": ["v1", "v3"], "v3": ["v2"]}
        rt = self.fc.compute_optimal_routes(graph)
        assert isinstance(rt, dict)
        assert "v1" in rt

    def test_routing_empty_graph(self):
        rt = self.fc.compute_optimal_routes({})
        assert rt == {}

    def test_message_log(self):
        self.fc.send(_msg("v1", "v2"))
        log = self.fc.get_message_log(10)
        assert len(log) >= 1

    def test_message_log_limit(self):
        for _ in range(5):
            self.fc.send(_msg("v1", "v2"))
        log = self.fc.get_message_log(2)
        assert len(log) <= 2

    def test_clear_log(self):
        self.fc.send(_msg("v1", "v2"))
        self.fc.clear_log()
        assert len(self.fc.get_message_log()) == 0

    def test_add_connection(self):
        ok = self.fm.add_connection("v1", "v2")
        assert ok is True

    def test_add_connection_nonexistent(self):
        assert self.fm.add_connection("v1", "ghost") is False

    def test_remove_connection(self):
        self.fm.add_connection("v1", "v2")
        assert self.fm.remove_connection("v1", "v2") is True

    def test_vessel_distance_to(self):
        v1 = self.fm.get_vessel("v1")
        v2 = self.fm.get_vessel("v2")
        d = v1.distance_to(v2)
        assert d > 0

    def test_vessel_bearing_to(self):
        v1 = self.fm.get_vessel("v1")
        v2 = self.fm.get_vessel("v2")
        b = v1.bearing_to(v2)
        assert 0 <= b < 360


# ═══════════════════════════════════════════════════════════════════════════
# 4. TaskOrchestrator + ConsensusProtocol
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskOrchestratorConsensus:
    """Cross-module: TaskOrchestrator + ConsensusProtocol."""

    def test_consensus_after_task_assignment(self):
        fm = FleetManager()
        for i in range(3):
            fm.register_vessel(_vinfo(f"v{i}", trust=0.8 + i * 0.05))
        to = TaskOrchestrator()
        cp = ConsensusProtocol()

        task = _task(ttype=TaskType.SURVEY, priority=0.8)
        to.submit_task(task)
        to.assign_vessels(task.id, ["v0", "v1"])

        # Consensus on task
        leader, _ = cp.raft_elect("v0", fm.get_all_vessels())
        result = cp.raft_propose(f"execute_{task.id}", leader, fm.get_all_vessels())
        assert isinstance(result, ConsensusResult)

    def test_paxos_for_task_priority(self):
        to = TaskOrchestrator()
        cp = ConsensusProtocol()

        task = _task(ttype=TaskType.RESCUE, priority=0.9)
        to.submit_task(task)

        proposal = Proposal(proposer="coordinator", value={"task": task.id, "priority": 0.9},
                            round_number=1)
        promises = cp.paxos_prepare(proposal, ["v1", "v2", "v3"])
        accepted = cp.paxos_accept(proposal, promises)
        assert isinstance(accepted, bool)

    def test_merkle_for_fleet_state_consistency(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("v1", (0, 0), fuel=80))
        fm.register_vessel(_vinfo("v2", (100, 100), fuel=60))

        snap1 = fm.get_fleet_snapshot()
        state1 = {
            "vessels": {v.vessel_id: {"fuel": v.fuel} for v in snap1.vessels}
        }

        fm.update_vessel_status("v1", {"fuel": 50.0})
        snap2 = fm.get_fleet_snapshot()
        state2 = {
            "vessels": {v.vessel_id: {"fuel": v.fuel} for v in snap2.vessels}
        }

        cp = ConsensusProtocol()
        h1 = cp.merkle_tree_hash(state1)
        h2 = cp.merkle_tree_hash(state2)
        # State changed -> hashes differ
        assert h1 != h2


# ═══════════════════════════════════════════════════════════════════════════
# 5. FleetCommunication + ConsensusProtocol
# ═══════════════════════════════════════════════════════════════════════════

class TestCommunicationConsensus:
    """Cross-module: FleetCommunication + ConsensusProtocol."""

    def test_broadcast_consensus_result(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("v1", trust=0.9))
        fm.register_vessel(_vinfo("v2", trust=0.8))
        fc = FleetCommunication()
        cp = ConsensusProtocol()

        leader, _ = cp.raft_elect("v1", fm.get_all_vessels())
        result = cp.raft_propose("route_change", leader, fm.get_all_vessels())

        # Broadcast consensus result
        msg = _msg("v1", "broadcast", MessageType.COORDINATION,
                   payload={"agreed": result.agreed_value})
        br = fc.broadcast("v1", msg, fm.get_all_vessels())
        assert isinstance(br, BroadcastResult)

    def test_send_leader_notification(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("v1"))
        fm.register_vessel(_vinfo("v2"))
        fc = FleetCommunication()
        cp = ConsensusProtocol()

        leader, _ = cp.raft_elect("v1", fm.get_all_vessels())
        msg = _msg("system", leader, MessageType.ALERT,
                   payload={"event": "new_leader", "leader": leader})
        status = fc.send(msg)
        assert isinstance(status, DeliveryStatus)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Anomaly Detection (FleetManager)
# ═══════════════════════════════════════════════════════════════════════════

class TestAnomalyDetection:
    """Cross-module anomaly detection in FleetManager."""

    def test_low_fuel_anomaly(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("vf", fuel=3.0))
        anomalies = fm.detect_anomalies()
        assert any(a.anomaly_type == "low_fuel" for a in anomalies)

    def test_health_degradation_anomaly(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("vh", health=0.2))
        anomalies = fm.detect_anomalies()
        assert any(a.anomaly_type == "health_degradation" for a in anomalies)

    def test_low_trust_anomaly(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("vt", trust=0.1))
        anomalies = fm.detect_anomalies()
        assert any(a.anomaly_type == "low_trust" for a in anomalies)

    def test_no_anomaly_nominal(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("ok", health=1.0, fuel=90.0, trust=0.9))
        anomalies = fm.detect_anomalies()
        assert not any(a.anomaly_type == "low_fuel" for a in anomalies)
        assert not any(a.anomaly_type == "health_degradation" for a in anomalies)
        assert not any(a.anomaly_type == "low_trust" for a in anomalies)

    def test_proximity_anomaly(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("pa", pos=(0.0, 0.0)))
        fm.register_vessel(_vinfo("pb", pos=(10.0, 10.0)))
        anomalies = fm.detect_anomalies()
        prox = [a for a in anomalies if a.anomaly_type == "proximity_warning"]
        assert len(prox) == 1

    def test_no_proximity_far_away(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("fa", pos=(0.0, 0.0)))
        fm.register_vessel(_vinfo("fb", pos=(200.0, 200.0)))
        anomalies = fm.detect_anomalies()
        assert not any(a.anomaly_type == "proximity_warning" for a in anomalies)

    def test_anomaly_history(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("ax", fuel=5.0))
        fm.detect_anomalies()
        hist = fm.get_anomaly_history()
        assert len(hist) > 0

    def test_anomaly_history_empty(self):
        fm = FleetManager()
        assert len(fm.get_anomaly_history()) == 0

    def test_multiple_anomaly_types(self):
        fm = FleetManager()
        fm.register_vessel(_vinfo("bad", fuel=5.0, health=0.2, trust=0.1))
        anomalies = fm.detect_anomalies()
        types = {a.anomaly_type for a in anomalies}
        assert "low_fuel" in types
        assert "health_degradation" in types
        assert "low_trust" in types


# ═══════════════════════════════════════════════════════════════════════════
# 7. End-to-End: All four modules
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndFleetOps:
    """Full end-to-end integration of all four fleet modules."""

    def test_full_cycle_register_assign_communicate(self):
        fm = FleetManager()
        to = TaskOrchestrator()
        cp = ConsensusProtocol()
        fc = FleetCommunication()

        # 1. Register fleet
        for i in range(3):
            fm.register_vessel(_vinfo(f"v{i}", trust=0.9 - i * 0.1))

        # 2. Elect leader
        leader, term = cp.raft_elect("v0", fm.get_all_vessels())
        assert leader is not None

        # 3. Create and assign task
        task = _task(ttype=TaskType.PATROL, priority=0.8)
        to.submit_task(task)
        to.assign_vessels(task.id, ["v0", "v1"])
        assert task.status == TaskStatus.ASSIGNED

        # 4. Communicate task assignment
        msg = _msg(leader, "broadcast", MessageType.COMMAND,
                   payload={"task": task.id, "action": "start"})
        br = fc.broadcast(leader, msg, fm.get_all_vessels())
        assert isinstance(br, BroadcastResult)

        # 5. Balance workload
        to.submit_task(_task(ttype=TaskType.SURVEY, priority=0.6))
        assignments = to.balance_workload(to.get_all_tasks(), fm.get_available_vessels())
        assert isinstance(assignments, list)

    def test_fleet_state_consensus_round_trip(self):
        fm = FleetManager()
        cp = ConsensusProtocol()
        fc = FleetCommunication()

        fm.register_vessel(_vinfo("v1", (10, 20), fuel=85))
        fm.register_vessel(_vinfo("v2", (30, 40), fuel=90))

        # Capture state hash
        snap = fm.get_fleet_snapshot()
        state = {
            "vessels": {v.vessel_id: {"fuel": v.fuel} for v in snap.vessels}
        }
        h1 = cp.merkle_tree_hash(state)

        # Simulate state change
        fm.update_vessel_status("v1", {"fuel": 50.0})
        snap2 = fm.get_fleet_snapshot()
        state2 = {
            "vessels": {v.vessel_id: {"fuel": v.fuel} for v in snap2.vessels}
        }
        h2 = cp.merkle_tree_hash(state2)
        assert h1 != h2

    def test_task_lifecycle_with_communication(self):
        fm = FleetManager()
        to = TaskOrchestrator()
        fc = FleetCommunication()

        fm.register_vessel(_vinfo("v1"))
        fm.register_vessel(_vinfo("v2"))

        task = _task(ttype=TaskType.DELIVERY, priority=0.7)
        to.submit_task(task)
        to.assign_vessels(task.id, ["v1"])

        # Notify vessel
        msg = _msg("ops", "v1", MessageType.COMMAND,
                   payload={"task_id": task.id, "status": "assigned"})
        status = fc.send(msg)
        assert isinstance(status, DeliveryStatus)

        # Cancel and notify
        to.cancel_task(task.id)
        msg2 = _msg("ops", "v1", MessageType.ALERT,
                    payload={"task_id": task.id, "status": "cancelled"})
        status2 = fc.send(msg2)
        assert isinstance(status2, DeliveryStatus)
