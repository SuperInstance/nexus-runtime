"""Quality of Service: priorities, acknowledgements, retry, throttling."""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, Dict, List
import time
import heapq


class QoSLevel(IntEnum):
    BEST_EFFORT = 0
    RELIABLE = 1
    REALTIME = 2
    CRITICAL = 3


@dataclass(order=True)
class QoSMessage:
    """A message with QoS metadata."""
    id: str
    payload: bytes
    priority: int = 0
    qos_level: QoSLevel = QoSLevel.BEST_EFFORT
    deadline: float = 0.0
    retries: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    _acked: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.deadline == 0.0:
            self.deadline = self.created_at + 30.0

    @property
    def effective_priority(self) -> float:
        """Priority score: higher QoS level and urgency = higher score."""
        urgency = max(0, self.deadline - time.time())
        return self.qos_level * 1000 + self.priority + urgency


class QoSManager:
    """Manages message queuing, delivery, acknowledgement, and throttling."""

    def __init__(self, max_queue_size: int = 10_000):
        self.max_queue_size = max_queue_size
        self._queue: Dict[str, QoSMessage] = {}
        self._pending: Dict[str, QoSMessage] = {}  # sent-but-unacked
        self._acked: set = set()
        self._dead: set = set()
        self._delivery_log: List[dict] = []

    # ------------------------------------------------------------------
    # Queue operations
    # ------------------------------------------------------------------
    def enqueue(self, msg: QoSMessage) -> bool:
        """Add *msg* to the queue. Returns False if queue full."""
        if msg.id in self._queue or msg.id in self._pending or msg.id in self._acked:
            return False
        if len(self._queue) >= self.max_queue_size:
            return False
        self._queue[msg.id] = msg
        return True

    def dequeue(self) -> Optional[QoSMessage]:
        """Remove and return the highest-priority message."""
        if not self._queue:
            return None
        best_id = max(self._queue, key=lambda mid: self._queue[mid].effective_priority)
        msg = self._queue.pop(best_id)
        self._pending[msg.id] = msg
        self._delivery_log.append({"msg_id": msg.id, "action": "dequeue", "ts": time.time()})
        return msg

    # ------------------------------------------------------------------
    # Ack / Retry
    # ------------------------------------------------------------------
    def ack(self, msg_id: str) -> bool:
        """Acknowledge a message. Returns True if it was pending."""
        if msg_id in self._pending:
            self._pending.pop(msg_id)
            self._acked.add(msg_id)
            self._delivery_log.append({"msg_id": msg_id, "action": "ack", "ts": time.time()})
            return True
        return False

    def retry(self, msg_id: str) -> Optional[QoSMessage]:
        """Retry a pending message. Increments retry count. Returns msg or None."""
        msg = self._pending.get(msg_id)
        if msg is None:
            return None
        msg.retries += 1
        if msg.retries > msg.max_retries:
            self._pending.pop(msg_id)
            self._dead.add(msg_id)
            self._delivery_log.append({"msg_id": msg_id, "action": "dead", "ts": time.time()})
            return None
        # Move back to queue
        self._pending.pop(msg_id)
        self._queue[msg_id] = msg
        self._delivery_log.append({"msg_id": msg_id, "action": "retry", "ts": time.time()})
        return msg

    def is_expired(self, msg: QoSMessage) -> bool:
        """Check if *msg* has passed its deadline."""
        return time.time() > msg.deadline

    # ------------------------------------------------------------------
    # Deadline-based prioritization
    # ------------------------------------------------------------------
    def prioritize_by_deadline(self, messages: List[QoSMessage]) -> List[QoSMessage]:
        """Sort messages by deadline (nearest first), then by QoS level."""
        return sorted(messages, key=lambda m: (m.deadline, -int(m.qos_level)))

    # ------------------------------------------------------------------
    # Throttling
    # ------------------------------------------------------------------
    def throttle_queue(self, queue: List[QoSMessage], max_rate: float) -> List[QoSMessage]:
        """Limit messages to *max_rate* per second.

        Simulates token-bucket: accepts messages if sending them doesn't
        exceed the rate, drops the rest.
        """
        if max_rate <= 0:
            return []
        if not queue:
            return []
        # Simple: take ceil(max_rate) messages per batch
        count = max(1, int(max_rate))
        # Sort by QoS level descending, then priority descending
        sorted_msgs = sorted(queue, key=lambda m: (-int(m.qos_level), -m.priority))
        return sorted_msgs[:count]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def compute_delivery_stats(self) -> dict:
        """Compute delivery success/failure statistics from the log."""
        total_sent = sum(1 for e in self._delivery_log if e["action"] == "dequeue")
        total_acked = sum(1 for e in self._delivery_log if e["action"] == "ack")
        total_retried = sum(1 for e in self._delivery_log if e["action"] == "retry")
        total_dead = sum(1 for e in self._delivery_log if e["action"] == "dead")
        success_rate = total_acked / total_sent if total_sent > 0 else 0.0
        return {
            "total_sent": total_sent,
            "total_acked": total_acked,
            "total_retried": total_retried,
            "total_dead": total_dead,
            "success_rate": success_rate,
            "queue_size": len(self._queue),
            "pending_size": len(self._pending),
        }
