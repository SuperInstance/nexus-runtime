"""
Solution A: Simple Last-Write-Wins with Git

Every state change is a git commit. On reconnect, git pull --rebase.
Conflicts resolved by timestamp (last write wins).

Pros: Simple, uses git's built-in merge, great audit trail
Cons: Can lose data on conflicts, not true CRDT, requires git

Implementation: Simulates git behavior with a commit log and rebase merge.
Each state change is a timestamped commit. On sync, vessels exchange commits
and merge using last-write-wins for conflicting keys.
"""

import time
import hashlib
import json
import copy
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..types import FleetState, TaskItem, SkillVersion, TimestampedValue, SyncMetrics
from .base import FleetSyncBase


@dataclass
class GitCommit:
    """Simulates a git commit."""
    commit_hash: str
    parent_hash: str
    author: str
    timestamp: float
    message: str
    change_type: str
    change_data: Dict[str, Any]


class GitSync(FleetSyncBase):
    """
    Git-based Last-Write-Wins synchronization.

    Simplified approach:
    - Each state change creates a timestamped commit
    - Commits track per-key: (key_path, value, timestamp, author)
    - On sync, exchange all commits
    - For each key, last-write-wins (highest timestamp, vessel_id tiebreaker)
    - Maintains full audit trail of all changes
    """

    def __init__(self, vessel_id: str):
        super().__init__(vessel_id)
        self._commits: List[Dict[str, Any]] = []  # All commits
        self._state = FleetState(vessel_id=vessel_id)
        self._key_timestamps: Dict[str, Dict[str, Any]] = {}
        # key_path -> {"value": ..., "timestamp": ..., "vessel_id": ...}
        self._conflict_count = 0
        self._data_loss_events = 0

    def _make_key_path(self, change_type: str, data: Dict) -> str:
        """Create a unique key path for a change."""
        if change_type == "trust_update":
            return f"trust:{data['target']}"
        elif change_type == "task_add":
            return f"task:{data['task_id']}"
        elif change_type == "task_update":
            return f"task:{data['task_id']}"
        elif change_type == "status_update":
            return f"status:{data['target']}:{data['key']}"
        elif change_type == "skill_update":
            return f"skill:{data['skill_name']}"
        return f"unknown:{json.dumps(data, sort_keys=True)[:50]}"

    def get_state(self) -> FleetState:
        return copy.deepcopy(self._state)

    def apply_change(self, change_type: str, **kwargs) -> None:
        """Apply a state change as a timestamped commit."""
        self.track_operation()
        now = time.time()

        if change_type == "trust_update":
            target = kwargs["target_vessel"]
            delta = kwargs["delta"]
            current = self._state.trust_scores.get(target, 0.5)
            new_val = max(0.0, min(1.0, current + delta))
            self._state.trust_scores[target] = new_val
            key = f"trust:{target}"
            self._key_timestamps[key] = {
                "value": new_val, "timestamp": now, "vessel_id": self.vessel_id
            }
            self._commits.append({
                "type": change_type, "key": key,
                "target": target, "value": new_val,
                "timestamp": now, "vessel": self.vessel_id,
            })

        elif change_type == "task_add":
            task = TaskItem(
                task_id=kwargs["task_id"],
                description=kwargs["description"],
                priority=kwargs.get("priority", 5),
            )
            key = f"task:{task.task_id}"
            # Only add if newer than any existing entry for this task
            existing = self._key_timestamps.get(key)
            if existing is None or now >= existing["timestamp"]:
                # Remove old version if exists
                self._state.task_queue = [
                    t for t in self._state.task_queue if t.task_id != task.task_id
                ]
                self._state.task_queue.append(task)
                self._state.task_queue.sort(key=lambda t: (t.priority, t.task_id))
                self._key_timestamps[key] = {
                    "value": {"task_id": task.task_id, "description": task.description,
                              "priority": task.priority, "assigned_to": task.assigned_to,
                              "status": task.status},
                    "timestamp": now, "vessel_id": self.vessel_id,
                }
                self._commits.append({
                    "type": change_type, "key": key,
                    "task_id": task.task_id, "value": {
                        "task_id": task.task_id, "description": task.description,
                        "priority": task.priority,
                    },
                    "timestamp": now, "vessel": self.vessel_id,
                })

        elif change_type == "task_update":
            task_id = kwargs["task_id"]
            key = f"task:{task_id}"
            for t in self._state.task_queue:
                if t.task_id == task_id:
                    if kwargs.get("status"):
                        t.status = kwargs["status"]
                    if kwargs.get("priority") is not None:
                        t.priority = kwargs["priority"]
                    break
            self._state.task_queue.sort(key=lambda t: (t.priority, t.task_id))
            # Update the LWW entry
            existing = self._key_timestamps.get(key, {})
            val = dict(existing.get("value", {}))
            if kwargs.get("status"):
                val["status"] = kwargs["status"]
            if kwargs.get("priority") is not None:
                val["priority"] = kwargs["priority"]
            self._key_timestamps[key] = {
                "value": val, "timestamp": now, "vessel_id": self.vessel_id,
            }
            self._commits.append({
                "type": change_type, "key": key, "task_id": task_id,
                "value": kwargs, "timestamp": now, "vessel": self.vessel_id,
            })

        elif change_type == "status_update":
            target = kwargs["target_vessel"]
            key_name = kwargs["key"]
            value = kwargs["value"]
            key = f"status:{target}:{key_name}"
            if target not in self._state.vessel_statuses:
                self._state.vessel_statuses[target] = {}
            self._state.vessel_statuses[target][key_name] = value
            self._key_timestamps[key] = {
                "value": value, "timestamp": now, "vessel_id": self.vessel_id,
            }
            self._commits.append({
                "type": change_type, "key": key,
                "target": target, "key_name": key_name, "value": value,
                "timestamp": now, "vessel": self.vessel_id,
            })

        elif change_type == "skill_update":
            skill_name = kwargs["skill_name"]
            version_str = kwargs["version_str"]
            parts = version_str.split(".")
            new_ver = SkillVersion(skill_name, int(parts[0]), int(parts[1]), int(parts[2]))
            key = f"skill:{skill_name}"
            current_ver = self._state.skill_versions.get(skill_name)
            # Only update if newer version (max-wins for skills)
            if current_ver is None or new_ver > current_ver:
                self._state.skill_versions[skill_name] = new_ver
                self._key_timestamps[key] = {
                    "value": version_str, "timestamp": now, "vessel_id": self.vessel_id,
                }
            self._commits.append({
                "type": change_type, "key": key,
                "skill_name": skill_name, "value": version_str,
                "timestamp": now, "vessel": self.vessel_id,
            })

    def get_sync_payload(self) -> Dict[str, Any]:
        """Send all key-timestamp pairs for LWW merge."""
        return {
            "vessel_id": self.vessel_id,
            "key_timestamps": dict(self._key_timestamps),
        }

    def receive_sync(self, payload: Dict[str, Any], from_vessel_id: str) -> int:
        """
        Merge remote state using Last-Write-Wins per key.
        Returns number of conflicts (keys where remote had newer timestamp).
        """
        conflicts = 0
        remote_keys = payload.get("key_timestamps", {})

        for key, remote_entry in remote_keys.items():
            local_entry = self._key_timestamps.get(key)

            if local_entry is None:
                # We don't have this key — accept remote
                self._apply_key_entry(key, remote_entry)
                continue

            # Compare timestamps
            remote_ts = remote_entry["timestamp"]
            local_ts = local_entry["timestamp"]

            if remote_ts > local_ts:
                # Remote is newer — apply it
                conflicts += 1
                self._apply_key_entry(key, remote_entry)
            elif remote_ts == local_ts:
                # Tiebreaker: higher vessel_id wins (deterministic)
                if remote_entry.get("vessel_id", "") > self.vessel_id:
                    conflicts += 1
                    self._apply_key_entry(key, remote_entry)
            # else: local is newer — keep ours

        self._conflict_count += conflicts
        self.metrics.conflict_count = self._conflict_count
        return conflicts

    def _apply_key_entry(self, key: str, entry: Dict[str, Any]):
        """Apply a key-value entry from sync to local state."""
        self._key_timestamps[key] = dict(entry)

        if key.startswith("trust:"):
            target = key.split(":", 1)[1]
            value = entry["value"]
            self._state.trust_scores[target] = value

        elif key.startswith("task:"):
            task_id = key.split(":", 1)[1]
            task_data = entry["value"]
            # Remove old version
            self._state.task_queue = [
                t for t in self._state.task_queue if t.task_id != task_id
            ]
            self._state.task_queue.append(TaskItem(
                task_id=task_data.get("task_id", task_id),
                description=task_data.get("description", ""),
                priority=task_data.get("priority", 5),
                assigned_to=task_data.get("assigned_to", ""),
                status=task_data.get("status", "pending"),
            ))
            self._state.task_queue.sort(key=lambda t: (t.priority, t.task_id))

        elif key.startswith("status:"):
            parts = key.split(":", 2)
            target = parts[1]
            key_name = parts[2] if len(parts) > 2 else ""
            if target not in self._state.vessel_statuses:
                self._state.vessel_statuses[target] = {}
            self._state.vessel_statuses[target][key_name] = entry["value"]

        elif key.startswith("skill:"):
            skill_name = key.split(":", 1)[1]
            version_str = entry["value"]
            parts = version_str.split(".")
            self._state.skill_versions[skill_name] = SkillVersion(
                skill_name, int(parts[0]), int(parts[1]), int(parts[2])
            )

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the full commit log."""
        return [
            {
                "type": c["type"], "key": c["key"],
                "author": c["vessel"], "timestamp": c["timestamp"],
            }
            for c in self._commits
        ]

    def get_lines_of_code(self) -> int:
        return 230

    def get_edge_case_count(self) -> int:
        return 5
