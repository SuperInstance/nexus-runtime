"""
Task Allocation Module
======================
Implements Contract-Net Protocol (CNP) and auction-based task allocation
for distributing missions across a marine robot swarm.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskType(Enum):
    """Types of tasks a vessel can be assigned."""
    PATROL = auto()
    SURVEY = auto()
    INSPECTION = auto()
    INTERCEPTION = auto()
    SEARCH = auto()
    RELAY = auto()
    SAMPLE = auto()


class TaskStatus(Enum):
    """Lifecycle status of a task."""
    PENDING = auto()
    BIDDING = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class Task:
    """Represents a mission task for allocation."""
    id: str
    type: TaskType
    priority: TaskPriority = TaskPriority.MEDIUM
    location: Tuple[float, float] = (0.0, 0.0)
    deadline: Optional[float] = None
    requirements: Dict[str, Any] = field(default_factory=dict)
    reward: float = 100.0
    status: TaskStatus = TaskStatus.PENDING

    def distance_to(self, x: float, y: float) -> float:
        """Euclidean distance from this task's location to (x, y)."""
        return math.hypot(self.location[0] - x, self.location[1] - y)


@dataclass
class Bid:
    """A bid submitted by a vessel for a task."""
    vessel_id: str
    task_id: str
    value: float  # lower is better (cost / estimated completion time)
    estimated_duration: float = 0.0
    capabilities: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class TaskAssignment:
    """Record of a task-to-vessel assignment."""
    task_id: str
    vessel_id: str
    bid_value: float
    assigned_at: float = 0.0
    progress: float = 0.0  # 0..1


class ContractNetProtocol:
    """
    Contract-Net Protocol (CNP) implementation for task allocation.

    Broadcasts tasks, collects bids, evaluates them, assigns tasks,
    monitors progress, and handles failures.
    """

    def __init__(self, bid_timeout: float = 5.0, max_bids_per_task: int = 10):
        self.bid_timeout = bid_timeout
        self.max_bids_per_task = max_bids_per_task
        self.tasks: Dict[str, Task] = {}
        self.bids: Dict[str, List[Bid]] = {}  # task_id -> bids
        self.assignments: Dict[str, TaskAssignment] = {}  # task_id -> assignment
        self.vessel_capabilities: Dict[str, Dict[str, Any]] = {}
        self._bid_counter: Dict[str, int] = {}  # task_id -> count

    def register_vessel(self, vessel_id: str, capabilities: Dict[str, Any]) -> None:
        """Register a vessel with its capabilities."""
        self.vessel_capabilities[vessel_id] = capabilities

    def unregister_vessel(self, vessel_id: str) -> None:
        """Unregister a vessel, cancelling its assigned tasks."""
        self.vessel_capabilities.pop(vessel_id, None)
        to_cancel = [
            tid for tid, a in self.assignments.items() if a.vessel_id == vessel_id
        ]
        for tid in to_cancel:
            self._cancel_task(tid)

    def broadcast_task(self, task: Task) -> None:
        """Broadcast a task for bidding."""
        task.status = TaskStatus.BIDDING
        self.tasks[task.id] = task
        self.bids[task.id] = []
        self._bid_counter[task.id] = 0

    def submit_bid(self, bid: Bid) -> bool:
        """Submit a bid for a task. Returns True if accepted."""
        task = self.tasks.get(bid.task_id)
        if task is None or task.status != TaskStatus.BIDDING:
            return False
        if bid.vessel_id not in self.vessel_capabilities:
            return False
        if self._bid_counter.get(bid.task_id, 0) >= self.max_bids_per_task:
            return False
        bid.timestamp = time.time()
        self.bids[bid.task_id].append(bid)
        self._bid_counter[bid.task_id] = self._bid_counter.get(bid.task_id, 0) + 1
        return True

    def evaluate_bids(self, task_id: str) -> Optional[Bid]:
        """
        Evaluate bids for a task and return the winning bid (lowest value).
        Returns None if no bids available.
        """
        bids = self.bids.get(task_id, [])
        if not bids:
            return None
        # Sort by bid value (lower is better)
        bids.sort(key=lambda b: b.value)
        return bids[0]

    def assign_task(self, task_id: str) -> Optional[TaskAssignment]:
        """
        Assign a task to the winning bidder.
        Returns the assignment or None if bidding is not complete.
        """
        task = self.tasks.get(task_id)
        if task is None or task.status != TaskStatus.BIDDING:
            return None
        winning_bid = self.evaluate_bids(task_id)
        if winning_bid is None:
            task.status = TaskStatus.PENDING
            return None
        assignment = TaskAssignment(
            task_id=task_id,
            vessel_id=winning_bid.vessel_id,
            bid_value=winning_bid.value,
            assigned_at=time.time(),
        )
        self.assignments[task_id] = assignment
        task.status = TaskStatus.ASSIGNED
        return assignment

    def start_task(self, task_id: str) -> bool:
        """Mark an assigned task as in-progress."""
        task = self.tasks.get(task_id)
        if task is None or task.status != TaskStatus.ASSIGNED:
            return False
        task.status = TaskStatus.IN_PROGRESS
        return True

    def update_progress(self, task_id: str, progress: float) -> bool:
        """Update progress for an in-progress task (0.0 – 1.0)."""
        task = self.tasks.get(task_id)
        if task is None or task.status != TaskStatus.IN_PROGRESS:
            return False
        progress = max(0.0, min(1.0, progress))
        assignment = self.assignments.get(task_id)
        if assignment is not None:
            assignment.progress = progress
        if progress >= 1.0:
            task.status = TaskStatus.COMPLETED
        return True

    def monitor_progress(self) -> Dict[str, float]:
        """Return a snapshot of progress for all in-progress tasks."""
        return {
            tid: a.progress
            for tid, a in self.assignments.items()
            if self.tasks.get(tid, Task("", TaskType.PATROL)).status == TaskStatus.IN_PROGRESS
        }

    def handle_task_failure(self, task_id: str) -> bool:
        """
        Handle a failed task: mark as failed and optionally re-broadcast.
        Returns True if the task was found.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return False
        task.status = TaskStatus.FAILED
        # Remove assignment
        self.assignments.pop(task_id, None)
        return True

    def reassign_task(self, task_id: str) -> bool:
        """
        Re-broadcast a failed or pending task for re-bidding.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return False
        if task.status not in (TaskStatus.FAILED, TaskStatus.PENDING):
            return False
        self.bids[task_id] = []
        self._bid_counter[task_id] = 0
        task.status = TaskStatus.BIDDING
        return True

    def _cancel_task(self, task_id: str) -> None:
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
        self.assignments.pop(task_id, None)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        return list(self.tasks.values())

    def get_assignments(self) -> List[TaskAssignment]:
        return list(self.assignments.values())

    def get_vessel_assignments(self, vessel_id: str) -> List[TaskAssignment]:
        return [a for a in self.assignments.values() if a.vessel_id == vessel_id]


class AuctionEngine:
    """
    Auction-based task allocation supporting single-item, combinatorial,
    and reserve-price auctions.
    """

    def __init__(self, reserve_price: float = 0.0):
        self.reserve_price = reserve_price
        self._auctions: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, Optional[Bid]] = {}

    def create_auction(
        self,
        auction_id: str,
        task_ids: List[str],
        reserve: Optional[float] = None,
    ) -> None:
        """
        Create a new auction for one or more tasks.
        """
        self._auctions[auction_id] = {
            "task_ids": list(task_ids),
            "reserve": reserve if reserve is not None else self.reserve_price,
            "bids": [],
            "closed": False,
        }

    def submit_bid(self, auction_id: str, bid: Bid) -> bool:
        """Submit a combinatorial bid (bid covers multiple tasks)."""
        auction = self._auctions.get(auction_id)
        if auction is None or auction["closed"]:
            return False
        if bid.value < auction["reserve"]:
            return False
        auction["bids"].append(bid)
        return True

    def close_auction(self, auction_id: str) -> Optional[Bid]:
        """
        Close an auction and return the winning bid (lowest value).
        Returns None if no valid bids.
        """
        auction = self._auctions.get(auction_id)
        if auction is None:
            return None
        auction["closed"] = True
        bids: List[Bid] = auction["bids"]
        if not bids:
            self._results[auction_id] = None
            return None
        bids.sort(key=lambda b: b.value)
        winner = bids[0]
        self._results[auction_id] = winner
        return winner

    def combinatorial_auction(
        self,
        auction_id: str,
        bids: List[Bid],
        reserve: Optional[float] = None,
    ) -> List[Bid]:
        """
        Run a simple greedy combinatorial auction.

        Accepts bids that cover bundles of tasks.  Returns a list of
        winning (non-conflicting) bids sorted by value ascending.
        """
        auction = self._auctions.get(auction_id)
        effective_reserve = reserve or (auction["reserve"] if auction else self.reserve_price)
        valid = [b for b in bids if b.value >= effective_reserve]
        valid.sort(key=lambda b: b.value)

        winners: List[Bid] = []
        assigned_tasks: set = set()
        for bid in valid:
            bid_tasks = set(bid.capabilities.get("task_ids", [bid.task_id]))
            if not bid_tasks.intersection(assigned_tasks):
                winners.append(bid)
                assigned_tasks.update(bid_tasks)
        self._results[auction_id] = winners[0] if winners else None
        return winners

    def reserve_prices(self, auction_id: str) -> float:
        """Get the reserve price for an auction."""
        auction = self._auctions.get(auction_id)
        if auction is None:
            return self.reserve_price
        return auction["reserve"]

    def set_reserve_price(self, auction_id: str, price: float) -> bool:
        """Set the reserve price for an existing auction."""
        auction = self._auctions.get(auction_id)
        if auction is None or auction["closed"]:
            return False
        auction["reserve"] = price
        return True

    def get_auction(self, auction_id: str) -> Optional[Dict[str, Any]]:
        return self._auctions.get(auction_id)

    def get_result(self, auction_id: str) -> Optional[Bid]:
        return self._results.get(auction_id)

    def is_closed(self, auction_id: str) -> bool:
        auction = self._auctions.get(auction_id)
        if auction is None:
            return False
        return auction["closed"]
