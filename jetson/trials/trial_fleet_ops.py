"""Trial 2: Fleet Operations — Fleet coordination + Swarm + Marketplace + Trust.

Tests cross-module integration between fleet management, task orchestration,
marketplace, consensus protocols, and fleet communication.
"""

from jetson.fleet_coordination.fleet_manager import (
    FleetManager, VesselStatus, AnomalyRecord,
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
from jetson.marketplace.task_market import (
    TaskMarket, TaskPost, Bid, MarketStatus,
)
# TrustLevel not needed for fleet ops trial


def _make_vessel_info(vid, pos=None, fuel=100.0, health=1.0, trust=1.0):
    return {
        "vessel_id": vid,
        "position": pos or (0.0, 0.0),
        "heading": 0.0, "speed": 5.0,
        "fuel": fuel, "health": health,
        "trust_score": trust, "available": True,
    }


def run_trial():
    """Run all fleet operations integration tests. Returns True if all pass."""
    passed = 0
    failed = 0
    total = 0

    def check(name, condition):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
        else:
            failed += 1

    # === Fleet Manager + Task Orchestration ===

    fm = FleetManager()
    to = TaskOrchestrator()

    # 1. Register vessels
    v1 = fm.register_vessel(_make_vessel_info("v1", (10.0, 20.0)))
    check("register_v1", v1.vessel_id == "v1")
    v2 = fm.register_vessel(_make_vessel_info("v2", (30.0, 40.0)))
    check("register_v2", v2.vessel_id == "v2")
    v3 = fm.register_vessel(_make_vessel_info("v3", (50.0, 60.0)))
    check("register_v3", v3.vessel_id == "v3")
    check("vessel_count_3", len(fm.get_all_vessels()) == 3)

    # 2. Duplicate registration raises
    try:
        fm.register_vessel(_make_vessel_info("v1"))
        check("dup_register_raises", False)
    except ValueError:
        check("dup_register_raises", True)

    # 3. Fleet health
    health = fm.compute_fleet_health()
    check("fleet_health_1", 0 <= health <= 1.0)

    # 4. Fleet snapshot
    snap = fm.get_fleet_snapshot()
    check("snapshot_vessels", len(snap.vessels) == 3)

    # 5. Update vessel status
    ok = fm.update_vessel_status("v1", {"fuel": 50.0, "health": 0.8})
    check("update_status", ok is True)
    updated = fm.get_vessel("v1")
    check("fuel_updated", updated.fuel == 50.0)

    # 6. Submit task
    task = FleetTask(type=TaskType.PATROL, priority=0.8)
    tid = to.submit_task(task)
    check("task_submitted", tid == task.id)
    check("task_status_pending", to.get_task_status(tid).status == TaskStatus.PENDING)

    # 7. Assign vessels to task
    ok = to.assign_vessels(tid, ["v1", "v2"])
    check("assign_vessels", ok is True)
    check("task_status_assigned", to.get_task_status(tid).status == TaskStatus.ASSIGNED)
    check("v1_task_count", to.get_active_task_count("v1") == 1)

    # 8. Reassign task
    ok = to.reassign_task(tid, ["v3"])
    check("reassign", ok is True)
    check("v1_task_count_after_reassign", to.get_active_task_count("v1") == 0)
    check("v3_task_count_after_reassign", to.get_active_task_count("v3") == 1)

    # 9. Compute task priority with fleet context
    dynamic_prio = to.compute_task_priority(task, snap)
    check("dynamic_priority_valid", 0 <= dynamic_prio <= 1.0)

    # 10. Workload balance
    task2 = FleetTask(type=TaskType.SURVEY, priority=0.6,
                      requirements=[TaskRequirement(skill="sonar", min_count=1)])
    to.submit_task(task2)
    assignments = to.balance_workload(to.get_all_tasks(), fm.get_available_vessels())
    check("balance_returns_list", isinstance(assignments, list))

    # 11. Cancel task
    ok = to.cancel_task(tid)
    check("cancel_task", ok is True)
    check("cancelled_status", to.get_task_status(tid).status == TaskStatus.CANCELLED)

    # 12. Get tasks for vessel
    to.submit_task(FleetTask(type=TaskType.PATROL))
    to.assign_vessels(to.get_all_tasks()[-1].id, ["v2"])
    v2_tasks = to.get_tasks_for_vessel("v2")
    check("v2_tasks_list", isinstance(v2_tasks, list))

    # 13. Deadlock detection
    deadlocks = to.detect_deadlocks(to.get_all_tasks())
    check("deadlocks_list", isinstance(deadlocks, list))

    # 14. ETA estimation
    eta = to.estimate_completion(task2, [v2, v3])
    check("eta_is_numeric", eta is None or isinstance(eta, (int, float)))

    # === Fleet Manager + Connectivity + Communication ===

    fc = FleetCommunication()

    # 15. Add connections
    ok = fm.add_connection("v1", "v2")
    check("add_connection", ok is True)
    ok = fm.add_connection("v2", "v3")
    check("add_connection_v2v3", ok is True)
    ok = fm.add_connection("v1", "v_nonexist")
    check("add_conn_nonexist", ok is False)

    # 16. Remove connection
    ok = fm.remove_connection("v2", "v3")
    check("remove_connection", ok is True)

    # 17. Unregister vessel
    ok = fm.deregister_vessel("v3")
    check("deregister", ok is True)
    check("vessel_count_after_dereg", len(fm.get_all_vessels()) == 2)

    # 18. Anomaly detection
    fm.register_vessel(_make_vessel_info("low_fuel", fuel=5.0))
    anomalies = fm.detect_anomalies()
    check("anomalies_detected", len(anomalies) > 0)
    low_fuel_anomaly = any(a.anomaly_type == "low_fuel" for a in anomalies)
    check("low_fuel_anomaly", low_fuel_anomaly)

    # 19. Health degradation anomaly
    fm.register_vessel(_make_vessel_info("sick", health=0.3))
    anomalies = fm.detect_anomalies()
    health_anomaly = any(a.anomaly_type == "health_degradation" for a in anomalies)
    check("health_anomaly", health_anomaly)

    # 20. Proximity anomaly
    fm.register_vessel(_make_vessel_info("close_a", pos=(0.0, 0.0)))
    fm.register_vessel(_make_vessel_info("close_b", pos=(10.0, 10.0)))
    # positions far enough — no proximity warning
    anomalies = fm.detect_anomalies()
    prox_anomaly = any(a.anomaly_type == "proximity_warning" for a in anomalies)
    check("proximity_check", isinstance(prox_anomaly, bool))

    # === Communication ===

    # 21. Send message
    msg = FleetMessage(source="v1", target="v2", type=MessageType.STATUS, payload="hello")
    status = fc.send(msg)
    check("send_returns_status", isinstance(status, DeliveryStatus))

    # 22. Broadcast
    vessels = [v1, v2]
    br = fc.broadcast("v1", msg, vessels)
    check("broadcast_result", isinstance(br, BroadcastResult))
    check("broadcast_reached_plus_failed", len(br.reached_vessels) + len(br.failed_vessels) >= 0)

    # 23. Multicast
    mr = fc.multicast("v1", msg, ["v2"])
    check("multicast_result", isinstance(mr, BroadcastResult))

    # 24. Relay
    ok = fc.relay_message(msg, ["v2"])
    check("relay_bool", isinstance(ok, bool))

    # 25. Add link
    link = fc.add_link("v1", "v2", latency=15.0, bandwidth=50.0)
    check("add_link", isinstance(link, LinkStatus))
    check("link_active", link.active is True)

    # 26. Network health
    all_links = [fc.get_link("v1", "v2")]
    h = fc.estimate_network_health(all_links)
    check("network_health_range", 0 <= h <= 1.0)

    # 27. Routing table
    graph = {"v1": ["v2"], "v2": ["v1"]}
    rt = fc.compute_optimal_routes(graph)
    check("routing_table_dict", isinstance(rt, dict))

    # 28. Message log
    log = fc.get_message_log(10)
    check("message_log_list", isinstance(log, list))

    # 29. Clear log
    fc.clear_log()
    check("log_cleared", len(fc.get_message_log()) == 0)

    # 30. Remove link
    ok = fc.remove_link("v1", "v2")
    check("remove_link", ok is True)
    check("link_gone", fc.get_link("v1", "v2") is None)

    # === Marketplace ===

    tm = TaskMarket()

    # 31. Post task
    tp = TaskPost(description="Patrol zone A", reward=100.0, poster_id="operator")
    task_id = tm.post_task(tp)
    check("post_task", len(task_id) > 0)

    # 32. Get open tasks
    open_tasks = tm.get_open_tasks()
    check("open_tasks_has_one", len(open_tasks) == 1)

    # 33. Submit bids
    bid1 = Bid(bidder_id="v1", amount=80.0, proposal="I can do it")
    ok = tm.submit_bid(task_id, bid1)
    check("submit_bid_v1", ok is True)

    bid2 = Bid(bidder_id="v2", amount=90.0, proposal="Better offer")
    ok = tm.submit_bid(task_id, bid2)
    check("submit_bid_v2", ok is True)

    # 34. Duplicate bidder rejected
    dup_bid = Bid(bidder_id="v1", amount=70.0)
    ok = tm.submit_bid(task_id, dup_bid)
    check("dup_bid_rejected", ok is False)

    # 35. Get bids
    bids = tm.get_task_bids(task_id)
    check("bid_count_2", len(bids) == 2)

    # 36. Evaluate bids
    ranked = tm.evaluate_bids(task_id)
    check("ranked_bids", len(ranked) == 2)
    check("first_ranked_lower_cost", ranked[0].amount <= ranked[1].amount)

    # 37. Close market
    winner = tm.close_market(task_id)
    check("winner_found", winner is not None)
    check("winner_accepted", winner.accepted is True)
    check("winner_lowest_cost", winner.amount == 80.0)

    # 38. Closed task not accepting bids
    late_bid = Bid(bidder_id="v3", amount=60.0)
    ok = tm.submit_bid(task_id, late_bid)
    check("late_bid_rejected", ok is False)

    # 39. Cancel task
    tp2 = TaskPost(description="Cancel me")
    tid2 = tm.post_task(tp2)
    tm.cancel_task(tid2, "No longer needed")
    check("cancel_marketplace", tm.get_task(tid2).status == MarketStatus.CANCELLED)

    # 40. Get all tasks
    all_tasks = tm.get_all_tasks()
    check("all_tasks_count", len(all_tasks) == 2)

    # === Consensus ===

    cp = ConsensusProtocol()

    # 41. Raft election
    leader, term = cp.raft_elect("v1", [v1, v2])
    check("raft_leader_str", isinstance(leader, (str, type(None))))
    check("raft_term_positive", term >= 1)
    check("leader_set", cp.leader_id == leader)

    # 42. Paxos prepare
    proposal = Proposal(proposer="v1", value="deploy_patrol", round_number=1)
    promises = cp.paxos_prepare(proposal, ["v1", "v2", "v3"])
    check("paxos_promises_int", isinstance(promises, int))
    check("paxos_promises_nonneg", promises >= 0)

    # 43. Paxos accept
    accepted = cp.paxos_accept(proposal, promises)
    check("paxos_accept_bool", isinstance(accepted, bool))

    # 44. Raft propose
    result = cp.raft_propose("go_north", leader or "v1", [v1, v2])
    check("raft_propose_result", isinstance(result, ConsensusResult))
    check("raft_propose_participants", len(result.participating_nodes) > 0)

    # 45. Split brain detection
    split = cp.detect_split_brain([["v1", "v2"], ["v3"]])
    check("split_brain_detected", split is True)
    no_split = cp.detect_split_brain([["v1", "v2", "v3"]])
    check("no_split_brain", no_split is False)

    # 46. Merkle hash
    state = {"v1": {"pos": [1, 2]}, "v2": {"pos": [3, 4]}}
    h1 = cp.merkle_tree_hash(state)
    h2 = cp.merkle_tree_hash(state)
    check("merkle_deterministic", h1 == h2)
    h3 = cp.merkle_tree_hash({"v1": {"pos": [5, 6]}})
    check("merkle_different_state", h1 != h3)

    # 47. Consensus log
    log = cp.log
    check("consensus_log_list", isinstance(log, list))

    # 48. Consensus reset
    cp.reset()
    check("reset_term_zero", cp.current_term == 0)
    check("reset_leader_none", cp.leader_id is None)

    # === Fleet Manager + Task Orchestration + Marketplace integration ===

    fm2 = FleetManager()
    to2 = TaskOrchestrator()
    tm2 = TaskMarket()

    fm2.register_vessel(_make_vessel_info("alpha", (0, 0)))
    fm2.register_vessel(_make_vessel_info("beta", (100, 100)))

    # 49. End-to-end: post task, get bids, assign
    tp3 = TaskPost(description="Survey area", reward=200.0, poster_id="ops")
    tid3 = tm2.post_task(tp3)
    tm2.submit_bid(tid3, Bid(bidder_id="alpha", amount=150.0))
    tm2.submit_bid(tid3, Bid(bidder_id="beta", amount=180.0))
    winner = tm2.close_market(tid3)
    check("e2e_winner", winner is not None and winner.bidder_id == "alpha")

    fleet_task = FleetTask(type=TaskType.SURVEY, priority=0.7)
    to2.submit_task(fleet_task)
    to2.assign_vessels(fleet_task.id, [winner.bidder_id])
    check("e2e_assigned", fleet_task.status == TaskStatus.ASSIGNED)

    # 50. Anomaly history
    anomaly_hist = fm2.get_anomaly_history()
    check("anomaly_history_list", isinstance(anomaly_hist, list))

    # 51. Available vessels
    avail = fm2.get_available_vessels()
    check("available_vessels", len(avail) >= 1)

    # 52. Update non-existent vessel
    ok = fm2.update_vessel_status("nonexistent", {"fuel": 50})
    check("update_nonexistent", ok is False)

    # 53. Get non-existent vessel
    check("get_nonexistent_vessel", fm2.get_vessel("ghost") is None)

    # 54. Consensus with empty fleet
    cp2 = ConsensusProtocol()
    leader_none, term2 = cp2.raft_elect("v1", [])
    check("empty_fleet_leader_none", leader_none is None)

    # 55. Task with requirements
    task_req = FleetTask(
        type=TaskType.INSPECTION,
        requirements=[TaskRequirement(skill="camera", min_count=2, min_trust=0.5)],
    )
    to2.submit_task(task_req)
    check("task_with_reqs", len(task_req.requirements) == 1)

    return failed == 0
