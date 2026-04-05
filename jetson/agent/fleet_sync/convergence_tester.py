"""
NEXUS CRDT Convergence Tester

Simulates multi-vessel fleets under realistic network conditions
and measures convergence properties.

Competes: state-based CRDT vs operation-based CRDT vs git-sync
across various network conditions (latency, loss, partitions).
"""

import time
import json
import random
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from .types import FleetState, TaskItem, SkillVersion, SyncMetrics
from .solutions.base import FleetSyncBase
from .solutions.git_sync import GitSync
from .solutions.operation_crdt import OperationCRDT
from .solutions.state_crdt import StateCRDT
from .network_simulator import (
    NetworkSimulator, NetworkConfig, NetworkCondition,
    PartitionManager, NetworkStats,
)


# State generation constants
SKILL_NAMES = ["navigation", "collision_avoidance", "sensor_fusion",
               "path_planning", "comms_relay"]
STATUS_KEYS = ["battery_level", "gps_fix", "water_temp", "speed_knots", "heading_deg"]
TASK_DESCRIPTIONS = [
    "Survey reef section A", "Monitor shipping lane", "Deploy sensor buoy",
    "Rescue beacon check", "Water quality sample", "Map seafloor grid 7",
    "Track whale migration", "Inspect underwater cable", "Weather station data",
    "Harbor patrol sweep", "Deploy ROV at waypoint", "Collect plankton sample",
]


@dataclass
class ConvergenceResult:
    """Results from a convergence test."""
    solution_name: str
    network_condition: str
    num_vessels: int
    ops_per_vessel: int
    converged: bool = False
    sync_rounds: int = 0
    messages_sent: int = 0
    messages_delivered: int = 0
    messages_lost: int = 0
    delivery_rate: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    conflict_count: int = 0
    data_loss_count: int = 0
    duration_ms: float = 0.0
    final_state_size_bytes: int = 0
    payload_size_bytes: int = 0
    unique_state_hashes: int = 0
    trust_converged: bool = False
    tasks_converged: bool = False
    skills_converged: bool = False
    statuses_converged: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solution_name": self.solution_name,
            "network_condition": self.network_condition,
            "num_vessels": self.num_vessels,
            "ops_per_vessel": self.ops_per_vessel,
            "converged": self.converged,
            "sync_rounds": self.sync_rounds,
            "messages_sent": self.messages_sent,
            "messages_delivered": self.messages_delivered,
            "messages_lost": self.messages_lost,
            "delivery_rate": round(self.delivery_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "conflict_count": self.conflict_count,
            "data_loss_count": self.data_loss_count,
            "duration_ms": round(self.duration_ms, 2),
            "final_state_size_bytes": self.final_state_size_bytes,
            "payload_size_bytes": self.payload_size_bytes,
            "unique_state_hashes": self.unique_state_hashes,
            "trust_converged": self.trust_converged,
            "tasks_converged": self.tasks_converged,
            "skills_converged": self.skills_converged,
            "statuses_converged": self.statuses_converged,
            "errors": self.errors,
        }


class ConvergenceTester:
    """
    Test CRDT convergence under various network conditions.

    Usage:
        tester = ConvergenceTester()
        result = tester.run_test(
            solution_class=StateCRDT,
            num_vessels=5,
            ops_per_vessel=20,
            network_condition=NetworkCondition.SATELLITE,
            seed=42,
        )
        print(result.to_dict())
    """

    def __init__(self):
        self.results: List[ConvergenceResult] = []

    def run_test(
        self,
        solution_class: Type[FleetSyncBase],
        num_vessels: int = 5,
        ops_per_vessel: int = 20,
        network_condition: NetworkCondition = NetworkCondition.PERFECT,
        seed: int = 42,
        max_sync_rounds: int = 30,
        use_network_sim: bool = False,
        partition_schedule: Optional[List] = None,
    ) -> ConvergenceResult:
        """Run a single convergence test."""
        start = time.time()
        vessel_ids = [f"vessel-{i}" for i in range(num_vessels)]
        result = ConvergenceResult(
            solution_name=solution_class.__name__,
            network_condition=network_condition.value,
            num_vessels=num_vessels,
            ops_per_vessel=ops_per_vessel,
        )

        try:
            # Initialize network simulator
            if use_network_sim:
                net_config = NetworkConfig.from_condition(network_condition)
                net_sim = NetworkSimulator(config=net_config, seed=seed)
                net_sim.add_vessels(vessel_ids)
                if partition_schedule:
                    for ps in partition_schedule:
                        net_sim.schedule_partition(**ps)
            else:
                net_sim = None

            # Create vessels
            random.seed(seed)
            vessels: Dict[str, FleetSyncBase] = {}
            for vid in vessel_ids:
                vessels[vid] = solution_class(vid)

            # Setup initial state via API
            for vid in vessel_ids:
                v = vessels[vid]
                for j, target in enumerate(vessel_ids):
                    if target != vid:
                        delta = round(0.3 + random.random() * 0.4 - 0.5, 3)
                        v.update_trust(target, delta)
                for skill in SKILL_NAMES:
                    ver = f"{random.randint(0,2)}.{random.randint(0,5)}.{random.randint(0,10)}"
                    v.update_skill_version(skill, ver)
                for t in range(2):
                    v.add_task(f"task-init-{vid}-{t}", f"Init task {t}", random.randint(1, 10))

            # Generate offline changes
            all_task_ids = set()
            for vid in vessel_ids:
                v = vessels[vid]
                for op_num in range(ops_per_vessel):
                    op_type = random.choice([
                        "trust_update", "trust_update", "task_add",
                        "task_update", "status_update", "skill_update",
                    ])
                    if op_type == "trust_update":
                        others = [x for x in vessel_ids if x != vid]
                        if others:
                            target = random.choice(others)
                            v.update_trust(target, round(random.uniform(-0.1, 0.1), 3))
                    elif op_type == "task_add":
                        tid = f"task-{vid}-{op_num}-{random.randint(100,999)}"
                        v.add_task(tid, random.choice(TASK_DESCRIPTIONS), random.randint(1, 10))
                        all_task_ids.add(tid)
                    elif op_type == "task_update":
                        tasks = v.get_state().task_queue
                        if tasks:
                            t = random.choice(tasks)
                            v.update_task(t.task_id, status=random.choice(["in_progress", "completed"]))
                    elif op_type == "status_update":
                        target = random.choice(vessel_ids)
                        key = random.choice(STATUS_KEYS)
                        if key in ("battery_level", "water_temp", "speed_knots", "heading_deg"):
                            value = round(random.uniform(0, 100), 1)
                        else:
                            value = random.choice(["3d", "2d", "none"])
                        v.update_vessel_status(target, key, value)
                    elif op_type == "skill_update":
                        skill = random.choice(SKILL_NAMES)
                        current = v.get_state().skill_versions.get(skill, SkillVersion(skill))
                        if random.random() < 0.7:
                            new_ver = f"{current.major}.{current.minor}.{current.patch + 1}"
                        else:
                            new_ver = f"{current.major}.{max(0, current.minor - 1)}.{current.patch}"
                            if new_ver == "0.0.0" and random.random() < 0.5:
                                new_ver = f"{current.major}.{current.minor}.{current.patch + 1}"
                        v.update_skill_version(skill, new_ver)

            # Sync rounds
            sync_rounds = 0
            converged = False

            while not converged and sync_rounds < max_sync_rounds:
                sync_rounds += 1

                # Pair up vessels
                pair_list = list(vessel_ids)
                random.shuffle(pair_list)

                for i in range(0, len(pair_list) - 1, 2):
                    v1_id = pair_list[i]
                    v2_id = pair_list[i + 1]

                    if net_sim is not None:
                        # Send through network simulator
                        net_sim.send(v1_id, v2_id, vessels[v1_id].get_sync_payload())
                        net_sim.send(v2_id, v1_id, vessels[v2_id].get_sync_payload())
                    else:
                        # Direct sync
                        p1 = vessels[v1_id].get_sync_payload()
                        p2 = vessels[v2_id].get_sync_payload()
                        vessels[v1_id].receive_sync(p2, v2_id)
                        vessels[v2_id].receive_sync(p1, v1_id)

                if net_sim is not None:
                    # Deliver messages
                    delivered = net_sim.drain()
                    for msg in delivered:
                        receiver = vessels.get(msg.receiver_id)
                        if receiver:
                            receiver.receive_sync(msg.payload, msg.sender_id)

                # Check convergence
                converged = self._check_convergence(vessels, vessel_ids)

            # Compute results
            final_states = {vid: vessels[vid].get_state() for vid in vessel_ids}
            hashes = {vid: s.state_hash() for vid, s in final_states.items()}

            result.converged = converged
            result.sync_rounds = sync_rounds
            result.duration_ms = (time.time() - start) * 1000
            result.unique_state_hashes = len(set(hashes.values()))

            if net_sim:
                stats = net_sim.get_stats()
                result.messages_sent = stats.messages_sent
                result.messages_delivered = stats.messages_delivered
                result.messages_lost = stats.messages_lost
                result.delivery_rate = stats.delivery_rate
                result.avg_latency_ms = stats.avg_latency_ms
                result.max_latency_ms = stats.max_latency_ms

            result.conflict_count = sum(
                v.metrics.conflict_count for v in vessels.values()
            )

            # Data loss
            final_tasks = set()
            for state in final_states.values():
                for task in state.task_queue:
                    final_tasks.add(task.task_id)
            result.data_loss_count = len(all_task_ids - final_tasks)

            # Per-domain convergence
            result.trust_converged = self._check_trust_convergence(final_states)
            result.tasks_converged = self._check_tasks_convergence(final_states)
            result.skills_converged = self._check_skills_convergence(final_states)
            result.statuses_converged = self._check_statuses_convergence(final_states)

            # State size
            v0_state = vessels[vessel_ids[0]].get_state()
            result.final_state_size_bytes = len(json.dumps({
                "trust_scores": {k: round(v, 6) for k, v in v0_state.trust_scores.items()},
                "tasks": len(v0_state.task_queue),
                "skills": {k: v.as_string() for k, v in v0_state.skill_versions.items()},
                "statuses": v0_state.vessel_statuses,
            }, default=str))

            v0_payload = vessels[vessel_ids[0]].get_sync_payload()
            result.payload_size_bytes = len(json.dumps(v0_payload, default=str))

        except Exception as e:
            result.errors.append(f"Test failed: {str(e)}")
            import traceback
            result.errors.append(traceback.format_exc())

        self.results.append(result)
        return result

    def run_competition(
        self,
        num_vessels: int = 5,
        ops_per_vessel: int = 20,
        network_condition: NetworkCondition = NetworkCondition.WIFI,
        seeds: List[int] = None,
        use_network_sim: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Run competition between all three solutions."""
        if seeds is None:
            seeds = [42, 123, 456, 789, 1000]

        solutions = [
            GitSync,
            OperationCRDT,
            StateCRDT,
        ]

        competition_results: Dict[str, Dict[str, Any]] = {}

        for sol_cls in solutions:
            name = sol_cls.__name__
            run_results = []

            for seed in seeds:
                r = self.run_test(
                    solution_class=sol_cls,
                    num_vessels=num_vessels,
                    ops_per_vessel=ops_per_vessel,
                    network_condition=network_condition,
                    seed=seed,
                    use_network_sim=use_network_sim,
                )
                run_results.append(r)

            competition_results[name] = {
                "convergence_rate": sum(1 for r in run_results if r.converged) / len(run_results),
                "avg_rounds": sum(r.sync_rounds for r in run_results) / len(run_results),
                "avg_conflicts": sum(r.conflict_count for r in run_results) / len(run_results),
                "avg_data_loss": sum(r.data_loss_count for r in run_results) / len(run_results),
                "avg_duration_ms": sum(r.duration_ms for r in run_results) / len(run_results),
                "avg_delivery_rate": sum(r.delivery_rate for r in run_results) / len(run_results),
                "avg_latency_ms": sum(r.avg_latency_ms for r in run_results) / len(run_results),
                "trust_convergence_rate": sum(1 for r in run_results if r.trust_converged) / len(run_results),
                "tasks_convergence_rate": sum(1 for r in run_results if r.tasks_converged) / len(run_results),
                "skills_convergence_rate": sum(1 for r in run_results if r.skills_converged) / len(run_results),
                "errors": [r.errors for r in run_results if r.errors],
            }

        return competition_results

    def _check_convergence(self, vessels: Dict[str, FleetSyncBase],
                           vessel_ids: List[str]) -> bool:
        if len(vessel_ids) < 2:
            return True
        states = [vessels[vid].get_state() for vid in vessel_ids]
        reference = states[0]
        return all(reference.is_equivalent(s) for s in states[1:])

    def _check_trust_convergence(self, final_states: Dict[str, FleetState]) -> bool:
        all_keys: Optional[set] = None
        all_values: Optional[Dict[str, float]] = None
        for state in final_states.values():
            keys = set(state.trust_scores.keys())
            if all_keys is None:
                all_keys = keys
            elif keys != all_keys:
                return False
        for key in all_keys or set():
            values = set()
            for state in final_states.values():
                val = state.trust_scores.get(key, 0.0)
                values.add(round(val, 3))
            if len(values) > 1:
                return False
        return True

    def _check_tasks_convergence(self, final_states: Dict[str, FleetState]) -> bool:
        all_ids: Optional[set] = None
        for state in final_states.values():
            ids = set(t.task_id for t in state.task_queue)
            if all_ids is None:
                all_ids = ids
            elif ids != all_ids:
                return False
        for tid in all_ids or set():
            statuses = set()
            for state in final_states.values():
                for t in state.task_queue:
                    if t.task_id == tid:
                        statuses.add(t.status)
            if len(statuses) > 1:
                return False
        return True

    def _check_skills_convergence(self, final_states: Dict[str, FleetState]) -> bool:
        for state in final_states.values():
            ref_keys = set(state.skill_versions.keys())
            other_keys = [set(s.skill_versions.keys()) for s in final_states.values()]
            if any(k != ref_keys for k in other_keys):
                return False
        for skill_name in list(final_states.values())[0].skill_versions:
            versions = set()
            for state in final_states.values():
                if skill_name in state.skill_versions:
                    versions.add(state.skill_versions[skill_name].as_string())
            if len(versions) > 1:
                return False
        return True

    def _check_statuses_convergence(self, final_states: Dict[str, FleetState]) -> bool:
        for state in final_states.values():
            ref_keys = set(state.vessel_statuses.keys())
            other_keys = [set(s.vessel_statuses.keys()) for s in final_states.values()]
            if any(k != ref_keys for k in other_keys):
                return False
        return True
