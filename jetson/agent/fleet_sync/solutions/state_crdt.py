"""
Solution C: State-Based CRDT with G-Counter

Each vessel maintains a full state snapshot + per-vessel update counter.
On sync, merge by taking max of each counter.
Trust scores use grow-only counters (PN-Counter for increments/decrements).
Task queues use LWW-Element-Set (add wins with timestamp).
Vessel status uses LWW-Register per key.
Skill versions use max-wins.

Pros: Simpler than operation-based, bounded state size
Cons: Less precise than operation-based, tombstone accumulation

State merge function is the core CRDT primitive:
  - merge(a, b) = element-wise max of per-vessel counters
  - This guarantees eventual convergence: merge(merge(a,b), merge(a,c)) = merge(merge(a,b),c)

Data structures:
  - PNCounter: Tracks increments and decrements separately with G-Counters
  - LWWElementSet: Each element has (value, timestamp, added_by). Add wins over remove.
  - LWWRegister: Each register has (value, timestamp). Last write wins.
  - MaxRegister: For skill versions — always takes the maximum version.
"""

import time
import copy
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field

from ..types import FleetState, TaskItem, SkillVersion, VectorClock, SyncMetrics
from .base import FleetSyncBase


@dataclass
class GCounter:
    """Grow-only counter. Each vessel has its own counter that only increases."""
    vessel_id: str
    counts: Dict[str, int] = field(default_factory=dict)  # vessel_id -> count

    def increment(self, vessel_id: str = None, amount: int = 1):
        vid = vessel_id or self.vessel_id
        self.counts[vid] = self.counts.get(vid, 0) + amount

    def value(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: "GCounter") -> "GCounter":
        """Merge two G-Counters by taking element-wise max."""
        all_keys = set(self.counts.keys()) | set(other.counts.keys())
        merged = GCounter(self.vessel_id)
        for k in all_keys:
            merged.counts[k] = max(self.counts.get(k, 0), other.counts.get(k, 0))
        return merged

    def copy(self) -> "GCounter":
        gc = GCounter(self.vessel_id)
        gc.counts = dict(self.counts)
        return gc


@dataclass
class PNCounter:
    """Positive-Negative Counter. Supports both increment and decrement."""
    vessel_id: str
    p: GCounter = field(default_factory=lambda: None)  # positive increments
    n: GCounter = field(default_factory=lambda: None)  # negative increments (decrements)

    def __post_init__(self):
        if self.p is None:
            self.p = GCounter(self.vessel_id)
        if self.n is None:
            self.n = GCounter(self.vessel_id)

    def increment(self, vessel_id: str = None, amount: float = 1.0):
        vid = vessel_id or self.vessel_id
        self.p.increment(vid, 1)  # Track number of positive ops

    def decrement(self, vessel_id: str = None, amount: float = 1.0):
        vid = vessel_id or self.vessel_id
        self.n.increment(vid, 1)  # Track number of negative ops

    def value(self, increments: Dict[str, float] = None,
              decrements: Dict[str, float] = None) -> float:
        """Calculate current value from tracked increments/decrements."""
        inc = increments or {}
        dec = decrements or {}
        total_pos = sum(inc.values())
        total_neg = sum(dec.values())
        return total_pos - total_neg

    def merge(self, other: "PNCounter") -> "PNCounter":
        """Merge two PN-Counters."""
        merged = PNCounter(self.vessel_id)
        merged.p = self.p.merge(other.p)
        merged.n = self.n.merge(other.n)
        return merged

    def copy(self) -> "PNCounter":
        pnc = PNCounter(self.vessel_id)
        pnc.p = self.p.copy()
        pnc.n = self.n.copy()
        return pnc


@dataclass
class LWWElement:
    """An element in an LWW-Element-Set."""
    value: Dict[str, Any]
    add_timestamp: float = 0.0
    remove_timestamp: float = 0.0
    added_by: str = ""
    removed: bool = False


@dataclass
class LWWRegister:
    """Last-Write-Wins Register. Holds a single value with timestamp."""
    value: Any = None
    timestamp: float = 0.0
    vessel_id: str = ""

    def set(self, value: Any, timestamp: float, vessel_id: str):
        if timestamp >= self.timestamp:
            if timestamp > self.timestamp or vessel_id > self.vessel_id:
                self.value = value
                self.timestamp = timestamp
                self.vessel_id = vessel_id

    def merge(self, other: "LWWRegister") -> "LWWRegister":
        """Merge by taking the newer value."""
        if other.timestamp > self.timestamp:
            return other.copy()
        elif other.timestamp == self.timestamp:
            if other.vessel_id > self.vessel_id:
                return other.copy()
        return self.copy()

    def copy(self) -> "LWWRegister":
        return LWWRegister(self.value, self.timestamp, self.vessel_id)


class StateCRDT(FleetSyncBase):
    """
    State-based CRDT for fleet state synchronization.

    Each vessel maintains:
    1. PN-Counters for trust scores (track per-vessel increment/decrement ops)
    2. LWW-Element-Set for task queue (add wins with timestamp)
    3. LWW-Register per (vessel, key) pair for vessel status
    4. Max-Register per skill name for skill versions

    On sync, merge by taking element-wise max/union of all CRDT states.
    """

    def __init__(self, vessel_id: str):
        super().__init__(vessel_id)
        self._state = FleetState(vessel_id=vessel_id)

        # CRDT state for trust scores:
        # vessel_id -> { "increments": {op_id: delta}, "decrements": {op_id: delta}, "pn_counter": PNCounter }
        self._trust_crdt: Dict[str, Dict[str, Any]] = {}

        # CRDT state for task queue:
        # task_id -> LWWElement
        self._task_crdt: Dict[str, LWWElement] = {}

        # CRDT state for vessel status:
        # target_vessel -> key -> LWWRegister
        self._status_crdt: Dict[str, Dict[str, LWWRegister]] = {}

        # CRDT state for skill versions:
        # skill_name -> { "version": SkillVersion, "register": LWWRegister }
        self._skill_crdt: Dict[str, Dict[str, Any]] = {}

        # Counter for operations
        self._op_counter = 0
        self._conflict_count = 0

    def _next_op_id(self) -> str:
        self._op_counter += 1
        return f"{self.vessel_id}:{self._op_counter}"

    def get_state(self) -> FleetState:
        """Return the current fleet state (materialized from CRDT state)."""
        return copy.deepcopy(self._state)

    def apply_change(self, change_type: str, **kwargs) -> None:
        """Apply a state change as a CRDT mutation."""
        self.track_operation()
        now = time.time()

        if change_type == "trust_update":
            target = kwargs["target_vessel"]
            delta = kwargs["delta"]

            if target not in self._trust_crdt:
                self._trust_crdt[target] = {
                    "increments": {},
                    "decrements": {},
                    "pn_counter": PNCounter(self.vessel_id),
                }

            crdt_state = self._trust_crdt[target]
            op_id = self._next_op_id()

            if delta > 0:
                crdt_state["increments"][op_id] = delta
                crdt_state["pn_counter"].increment(self.vessel_id)
            else:
                crdt_state["decrements"][op_id] = abs(delta)
                crdt_state["pn_counter"].decrement(self.vessel_id)

            # Materialize: sum all increments - sum all decrements
            total_inc = sum(crdt_state["increments"].values())
            total_dec = sum(crdt_state["decrements"].values())
            self._state.trust_scores[target] = max(0.0, min(1.0, 0.5 + total_inc - total_dec))

        elif change_type == "task_add":
            task_id = kwargs["task_id"]
            element = LWWElement(
                value={
                    "task_id": task_id,
                    "description": kwargs["description"],
                    "priority": kwargs.get("priority", 5),
                    "assigned_to": kwargs.get("assigned_to", ""),
                    "status": kwargs.get("status", "pending"),
                },
                add_timestamp=now,
                added_by=self.vessel_id,
            )
            # Only add if not already present with a newer add timestamp
            if task_id not in self._task_crdt or now >= self._task_crdt[task_id].add_timestamp:
                self._task_crdt[task_id] = element
            self._materialize_tasks()

        elif change_type == "task_update":
            task_id = kwargs["task_id"]
            if task_id in self._task_crdt:
                element = self._task_crdt[task_id]
                if kwargs.get("status"):
                    element.value["status"] = kwargs["status"]
                if kwargs.get("priority") is not None:
                    element.value["priority"] = kwargs["priority"]
                element.add_timestamp = max(element.add_timestamp, now)
            self._materialize_tasks()

        elif change_type == "status_update":
            target = kwargs["target_vessel"]
            key = kwargs["key"]
            value = kwargs["value"]

            if target not in self._status_crdt:
                self._status_crdt[target] = {}
            if key not in self._status_crdt[target]:
                self._status_crdt[target][key] = LWWRegister()

            self._status_crdt[target][key].set(value, now, self.vessel_id)

            # Materialize
            if target not in self._state.vessel_statuses:
                self._state.vessel_statuses[target] = {}
            self._state.vessel_statuses[target][key] = value

        elif change_type == "skill_update":
            skill_name = kwargs["skill_name"]
            version_str = kwargs["version_str"]
            parts = version_str.split(".")
            new_version = SkillVersion(skill_name, int(parts[0]), int(parts[1]), int(parts[2]))

            if skill_name not in self._skill_crdt:
                self._skill_crdt[skill_name] = {
                    "version": new_version,
                    "register": LWWRegister(version_str, now, self.vessel_id),
                }
            else:
                current = self._skill_crdt[skill_name]["version"]
                if new_version > current:
                    self._skill_crdt[skill_name]["version"] = new_version
                    self._skill_crdt[skill_name]["register"].set(version_str, now, self.vessel_id)

            self._state.skill_versions[skill_name] = new_version

    def _materialize_tasks(self):
        """Rebuild task queue from CRDT elements."""
        tasks = []
        for task_id, element in self._task_crdt.items():
            if not element.removed:
                tasks.append(TaskItem(**element.value))
        tasks.sort(key=lambda t: (t.priority, t.task_id))
        self._state.task_queue = tasks

    def get_sync_payload(self) -> Dict[str, Any]:
        """Generate sync payload: full CRDT state."""
        return {
            "vessel_id": self.vessel_id,
            "trust_crdt": self._serialize_trust_crdt(),
            "task_crdt": self._serialize_task_crdt(),
            "status_crdt": self._serialize_status_crdt(),
            "skill_crdt": self._serialize_skill_crdt(),
        }

    def receive_sync(self, payload: Dict[str, Any], from_vessel_id: str) -> int:
        """
        Merge CRDT state from another vessel.
        Returns the number of conflicts detected.
        """
        conflicts = 0

        # Merge trust CRDTs
        conflicts += self._merge_trust_crdt(payload.get("trust_crdt", {}))

        # Merge task CRDTs
        conflicts += self._merge_task_crdt(payload.get("task_crdt", {}))

        # Merge status CRDTs
        conflicts += self._merge_status_crdt(payload.get("status_crdt", {}))

        # Merge skill CRDTs
        conflicts += self._merge_skill_crdt(payload.get("skill_crdt", {}))

        self._conflict_count += conflicts
        self.metrics.conflict_count = self._conflict_count
        return conflicts

    def _serialize_trust_crdt(self) -> Dict[str, Any]:
        result = {}
        for vessel, data in self._trust_crdt.items():
            result[vessel] = {
                "increments": data["increments"],
                "decrements": data["decrements"],
            }
        return result

    def _serialize_task_crdt(self) -> Dict[str, Any]:
        result = {}
        for task_id, element in self._task_crdt.items():
            result[task_id] = {
                "value": element.value,
                "add_timestamp": element.add_timestamp,
                "remove_timestamp": element.remove_timestamp,
                "added_by": element.added_by,
                "removed": element.removed,
            }
        return result

    def _serialize_status_crdt(self) -> Dict[str, Any]:
        result = {}
        for vessel, keys in self._status_crdt.items():
            result[vessel] = {}
            for key, reg in keys.items():
                result[vessel][key] = {
                    "value": reg.value,
                    "timestamp": reg.timestamp,
                    "vessel_id": reg.vessel_id,
                }
        return result

    def _serialize_skill_crdt(self) -> Dict[str, Any]:
        result = {}
        for skill, data in self._skill_crdt.items():
            result[skill] = {
                "version": data["version"].as_string(),
                "register": {
                    "value": data["register"].value,
                    "timestamp": data["register"].timestamp,
                    "vessel_id": data["register"].vessel_id,
                },
            }
        return result

    def _merge_trust_crdt(self, remote_trust: Dict[str, Any]) -> int:
        """Merge trust CRDTs. Conflicts happen when same vessel has different increment/decrement sets."""
        conflicts = 0
        for vessel, remote_data in remote_trust.items():
            if vessel not in self._trust_crdt:
                # New vessel trust entry — just adopt
                self._trust_crdt[vessel] = {
                    "increments": dict(remote_data.get("increments", {})),
                    "decrements": dict(remote_data.get("decrements", {})),
                    "pn_counter": PNCounter(self.vessel_id),
                }
            else:
                # Merge: take union of all increment/decrement operations
                local = self._trust_crdt[vessel]
                local_incs = set(local["increments"].keys())
                remote_incs = set(remote_data.get("increments", {}).keys())
                local_decs = set(local["decrements"].keys())
                remote_decs = set(remote_data.get("decrements", {}).keys())

                # If both have non-overlapping operations, it's a concurrent conflict
                if local_incs & remote_incs or local_decs & remote_decs:
                    # Both modified same op — shouldn't happen (ops are unique per vessel)
                    pass
                if (local_incs - remote_incs) and (remote_incs - local_incs):
                    conflicts += 1  # Concurrent modifications

                # Union of all operations
                for op_id, delta in remote_data.get("increments", {}).items():
                    if op_id not in local["increments"]:
                        local["increments"][op_id] = delta
                for op_id, delta in remote_data.get("decrements", {}).items():
                    if op_id not in local["decrements"]:
                        local["decrements"][op_id] = delta

            # Materialize
            total_inc = sum(self._trust_crdt[vessel]["increments"].values())
            total_dec = sum(self._trust_crdt[vessel]["decrements"].values())
            self._state.trust_scores[vessel] = max(0.0, min(1.0, 0.5 + total_inc - total_dec))

        return conflicts

    def _merge_task_crdt(self, remote_tasks: Dict[str, Any]) -> int:
        """Merge task LWW-Element-Sets. Add-wins with timestamp."""
        conflicts = 0
        for task_id, remote_element in remote_tasks.items():
            if task_id not in self._task_crdt:
                # New task — add it
                self._task_crdt[task_id] = LWWElement(
                    value=remote_element["value"],
                    add_timestamp=remote_element["add_timestamp"],
                    remove_timestamp=remote_element.get("remove_timestamp", 0),
                    added_by=remote_element.get("added_by", ""),
                    removed=remote_element.get("removed", False),
                )
            else:
                # Merge: add-wins (add timestamp > remove timestamp)
                local = self._task_crdt[task_id]
                if (remote_element["add_timestamp"] > local.add_timestamp or
                        (remote_element["add_timestamp"] == local.add_timestamp and
                         remote_element.get("added_by", "") > local.added_by)):
                    # Remote add is newer — take remote
                    if local.add_timestamp != remote_element["add_timestamp"]:
                        conflicts += 1
                    self._task_crdt[task_id] = LWWElement(
                        value=remote_element["value"],
                        add_timestamp=remote_element["add_timestamp"],
                        remove_timestamp=remote_element.get("remove_timestamp", 0),
                        added_by=remote_element.get("added_by", ""),
                        removed=remote_element.get("removed", False),
                    )
                elif remote_element.get("removed", False) and not local.removed:
                    # Remote removed — only if remove > add
                    if remote_element.get("remove_timestamp", 0) > local.add_timestamp:
                        local.removed = True
                        local.remove_timestamp = remote_element["remove_timestamp"]

        self._materialize_tasks()
        return conflicts

    def _merge_status_crdt(self, remote_status: Dict[str, Any]) -> int:
        """Merge status LWW-Registers."""
        conflicts = 0
        for vessel, keys in remote_status.items():
            if vessel not in self._status_crdt:
                self._status_crdt[vessel] = {}
            for key, remote_reg_data in keys.items():
                remote_reg = LWWRegister(
                    value=remote_reg_data["value"],
                    timestamp=remote_reg_data["timestamp"],
                    vessel_id=remote_reg_data["vessel_id"],
                )
                if key not in self._status_crdt[vessel]:
                    self._status_crdt[vessel][key] = remote_reg
                else:
                    local_reg = self._status_crdt[vessel][key]
                    merged = local_reg.merge(remote_reg)
                    if merged.vessel_id != local_reg.vessel_id:
                        conflicts += 1
                    self._status_crdt[vessel][key] = merged

                # Materialize
                if vessel not in self._state.vessel_statuses:
                    self._state.vessel_statuses[vessel] = {}
                self._state.vessel_statuses[vessel][key] = self._status_crdt[vessel][key].value

        return conflicts

    def _merge_skill_crdt(self, remote_skills: Dict[str, Any]) -> int:
        """Merge skill Max-Registers."""
        conflicts = 0
        for skill_name, remote_data in remote_skills.items():
            remote_version = SkillVersion(
                skill_name,
                *[int(x) for x in remote_data["version"].split(".")]
            )

            if skill_name not in self._skill_crdt:
                self._skill_crdt[skill_name] = {
                    "version": remote_version,
                    "register": LWWRegister(
                        remote_data["version"],
                        remote_data["register"]["timestamp"],
                        remote_data["register"]["vessel_id"],
                    ),
                }
            else:
                current = self._skill_crdt[skill_name]["version"]
                if remote_version > current:
                    conflicts += 1
                    self._skill_crdt[skill_name]["version"] = remote_version
                    self._skill_crdt[skill_name]["register"] = LWWRegister(
                        remote_data["version"],
                        remote_data["register"]["timestamp"],
                        remote_data["register"]["vessel_id"],
                    )

            self._state.skill_versions[skill_name] = self._skill_crdt[skill_name]["version"]

        return conflicts

    def get_lines_of_code(self) -> int:
        """Return approximate lines of code."""
        return 370

    def get_edge_case_count(self) -> int:
        """Return known edge cases."""
        return 9  # LWW ties, tombstone accumulation, trust accumulation ordering, etc.

    def get_memory_usage(self) -> int:
        """Estimate memory usage including CRDT metadata."""
        import sys
        base = sys.getsizeof(self._state)
        trust_size = sum(
            sys.getsizeof(data["increments"]) + sys.getsizeof(data["decrements"])
            for data in self._trust_crdt.values()
        )
        task_size = sys.getsizeof(self._task_crdt)
        status_size = sum(
            sum(sys.getsizeof(reg) for reg in keys.values())
            for keys in self._status_crdt.values()
        )
        skill_size = sys.getsizeof(self._skill_crdt)
        return base + trust_size + task_size + status_size + skill_size

    def get_tombstone_count(self) -> int:
        """Count removed (tombstoned) elements."""
        return sum(1 for e in self._task_crdt.values() if e.removed)
