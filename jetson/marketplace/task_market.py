"""Task posting and bidding for the fleet marketplace."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MarketStatus(Enum):
    """Status of a task in the marketplace."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING_ASSIGNMENT = "PENDING_ASSIGNMENT"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass
class TaskPost:
    """Represents a task posted to the marketplace."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    requirements: Dict[str, Any] = field(default_factory=dict)
    location: Dict[str, float] = field(default_factory=lambda: {"lat": 0.0, "lon": 0.0})
    deadline: Optional[datetime] = None
    reward: float = 0.0
    poster_id: str = ""
    status: MarketStatus = MarketStatus.OPEN
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bid:
    """Represents a bid on a marketplace task."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    bidder_id: str = ""
    amount: float = 0.0
    eta: Optional[datetime] = None
    capability_scores: Dict[str, float] = field(default_factory=dict)
    proposal: str = ""
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    accepted: bool = False


class TaskMarket:
    """Central marketplace for posting tasks and collecting bids."""

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskPost] = {}
        self._bids: Dict[str, List[Bid]] = {}  # task_id -> bids
        self._winning_bids: Dict[str, Bid] = {}  # task_id -> winning bid

    def post_task(self, task_post: TaskPost) -> str:
        """Post a new task to the marketplace. Returns the task id."""
        task_post.status = MarketStatus.OPEN
        self._tasks[task_post.id] = task_post
        self._bids[task_post.id] = []
        return task_post.id

    def submit_bid(self, task_id: str, bid: Bid) -> bool:
        """Submit a bid for a task. Returns True if accepted."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status != MarketStatus.OPEN:
            return False
        if bid.task_id != task_id:
            bid.task_id = task_id
        # Check if this bidder already bid
        for existing in self._bids[task_id]:
            if existing.bidder_id == bid.bidder_id:
                return False  # Duplicate bidder
        self._bids[task_id].append(bid)
        return True

    def close_market(self, task_id: str) -> Optional[Bid]:
        """Close the market for a task and return the winning bid."""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if task.status != MarketStatus.OPEN:
            return None
        bids = self._bids.get(task_id, [])
        if not bids:
            task.status = MarketStatus.PENDING_ASSIGNMENT
            return None
        # Select lowest cost bid
        winning = min(bids, key=lambda b: b.amount)
        winning.accepted = True
        self._winning_bids[task_id] = winning
        task.status = MarketStatus.CLOSED
        return winning

    def evaluate_bids(self, task_id: str, criteria: Optional[Dict[str, float]] = None) -> List[Bid]:
        """Evaluate and rank bids for a task based on criteria weights."""
        bids = self._bids.get(task_id, [])
        if not bids:
            return []
        if criteria is None:
            criteria = {"cost": 0.5, "eta": 0.3, "capability": 0.2}

        def score_bid(bid: Bid) -> float:
            # Cost component (lower is better)
            max_amount = max(b.amount for b in bids) if bids else 1.0
            min_amount = min(b.amount for b in bids) if bids else 0.0
            cost_range = max_amount - min_amount if max_amount != min_amount else 1.0
            cost_score = 1.0 - (bid.amount - min_amount) / cost_range

            # Capability component (higher is better)
            cap_scores = list(bid.capability_scores.values())
            cap_score = sum(cap_scores) / len(cap_scores) if cap_scores else 0.0

            # ETA component – if all bids have eta, closer is better
            eta_score = 1.0
            etas = [b.eta for b in bids if b.eta is not None]
            if etas and bid.eta is not None:
                earliest = min(etas)
                latest = max(etas)
                eta_range = (latest - earliest).total_seconds() if latest != earliest else 1.0
                if eta_range > 0:
                    eta_score = 1.0 - (bid.eta - earliest).total_seconds() / eta_range

            total = (
                criteria.get("cost", 0.5) * cost_score
                + criteria.get("capability", 0.2) * cap_score
                + criteria.get("eta", 0.3) * eta_score
            )
            return total

        ranked = sorted(bids, key=score_bid, reverse=True)
        return ranked

    def cancel_task(self, task_id: str, reason: str = "") -> None:
        """Cancel a task."""
        task = self._tasks.get(task_id)
        if task is None:
            return
        if task.status in (MarketStatus.COMPLETED, MarketStatus.CANCELLED):
            return
        task.status = MarketStatus.CANCELLED
        task.metadata["cancel_reason"] = reason

    def get_open_tasks(self) -> List[TaskPost]:
        """Return all open tasks."""
        return [t for t in self._tasks.values() if t.status == MarketStatus.OPEN]

    def get_task_bids(self, task_id: str) -> List[Bid]:
        """Return all bids for a given task."""
        return self._bids.get(task_id, [])

    def get_task(self, task_id: str) -> Optional[TaskPost]:
        """Get a task by id."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[TaskPost]:
        """Get all tasks."""
        return list(self._tasks.values())
