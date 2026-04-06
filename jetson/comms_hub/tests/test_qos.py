"""Tests for jetson.comms_hub.qos."""

import time
import pytest
from jetson.comms_hub.qos import QoSLevel, QoSMessage, QoSManager


@pytest.fixture
def manager():
    return QoSManager()


@pytest.fixture
def sample_msg():
    return QoSMessage(id="msg-1", payload=b"data", priority=5, qos_level=QoSLevel.RELIABLE)


@pytest.fixture
def sample_msg_critical():
    return QoSMessage(id="msg-crit", payload=b"critical", priority=10, qos_level=QoSLevel.CRITICAL)


class TestQoSLevel:
    def test_enum_values(self):
        assert QoSLevel.BEST_EFFORT == 0
        assert QoSLevel.RELIABLE == 1
        assert QoSLevel.REALTIME == 2
        assert QoSLevel.CRITICAL == 3

    def test_ordering(self):
        assert QoSLevel.CRITICAL > QoSLevel.REALTIME
        assert QoSLevel.REALTIME > QoSLevel.RELIABLE
        assert QoSLevel.RELIABLE > QoSLevel.BEST_EFFORT

    def test_count(self):
        assert len(QoSLevel) == 4


class TestQoSMessage:
    def test_construction(self, sample_msg):
        assert sample_msg.id == "msg-1"
        assert sample_msg.payload == b"data"
        assert sample_msg.priority == 5
        assert sample_msg.qos_level == QoSLevel.RELIABLE
        assert sample_msg.retries == 0
        assert sample_msg.max_retries == 3
        assert sample_msg.created_at > 0

    def test_deadline_auto_set(self, sample_msg):
        assert sample_msg.deadline > sample_msg.created_at

    def test_effective_priority(self, sample_msg, sample_msg_critical):
        assert sample_msg_critical.effective_priority > sample_msg.effective_priority

    def test_effective_priority_components(self):
        msg = QoSMessage(
            id="m1",
            payload=b"x",
            priority=5,
            qos_level=QoSLevel.CRITICAL,
            deadline=time.time() + 100,
        )
        ep = msg.effective_priority
        # CRITICAL(3)*1000 + priority(5) + urgency(~100) = ~3105
        assert ep > 3000

    def test_default_qos_level(self):
        msg = QoSMessage(id="m", payload=b"d")
        assert msg.qos_level == QoSLevel.BEST_EFFORT


class TestEnqueueDequeue:
    def test_enqueue_dequeue(self, manager, sample_msg):
        assert manager.enqueue(sample_msg) is True
        msg = manager.dequeue()
        assert msg is not None
        assert msg.id == "msg-1"

    def test_dequeue_empty(self, manager):
        assert manager.dequeue() is None

    def test_enqueue_duplicate(self, manager, sample_msg):
        assert manager.enqueue(sample_msg) is True
        assert manager.enqueue(sample_msg) is False

    def test_enqueue_after_dequeue_fails(self, manager, sample_msg):
        manager.enqueue(sample_msg)
        manager.dequeue()
        assert manager.enqueue(sample_msg) is False

    def test_priority_ordering(self, manager):
        m1 = QoSMessage(id="low", payload=b"x", qos_level=QoSLevel.BEST_EFFORT, priority=1)
        m2 = QoSMessage(id="high", payload=b"x", qos_level=QoSLevel.CRITICAL, priority=10)
        manager.enqueue(m1)
        manager.enqueue(m2)
        first = manager.dequeue()
        assert first.id == "high"

    def test_queue_full(self, manager):
        manager = QoSManager(max_queue_size=2)
        m1 = QoSMessage(id="1", payload=b"x")
        m2 = QoSMessage(id="2", payload=b"x")
        m3 = QoSMessage(id="3", payload=b"x")
        assert manager.enqueue(m1) is True
        assert manager.enqueue(m2) is True
        assert manager.enqueue(m3) is False

    def test_fifo_same_priority(self, manager):
        m1 = QoSMessage(id="first", payload=b"x", qos_level=QoSLevel.BEST_EFFORT)
        m2 = QoSMessage(id="second", payload=b"x", qos_level=QoSLevel.BEST_EFFORT)
        manager.enqueue(m1)
        manager.enqueue(m2)
        # Both have same effective_priority components, dequeue whichever is max
        msg = manager.dequeue()
        assert msg.id in ("first", "second")


class TestAck:
    def test_ack_pending(self, manager, sample_msg):
        manager.enqueue(sample_msg)
        manager.dequeue()
        assert manager.ack("msg-1") is True

    def test_ack_not_pending(self, manager):
        assert manager.ack("nonexistent") is False

    def test_ack_already_acked(self, manager, sample_msg):
        manager.enqueue(sample_msg)
        manager.dequeue()
        assert manager.ack("msg-1") is True
        assert manager.ack("msg-1") is False

    def test_ack_removes_from_pending(self, manager, sample_msg):
        manager.enqueue(sample_msg)
        manager.dequeue()
        manager.ack("msg-1")
        stats = manager.compute_delivery_stats()
        assert stats["pending_size"] == 0


class TestRetry:
    def test_retry_pending(self, manager, sample_msg):
        manager.enqueue(sample_msg)
        manager.dequeue()
        result = manager.retry("msg-1")
        assert result is not None
        assert result.retries == 1

    def test_retry_not_pending(self, manager):
        assert manager.retry("nonexistent") is None

    def test_retry_moves_to_queue(self, manager, sample_msg):
        manager.enqueue(sample_msg)
        manager.dequeue()
        manager.retry("msg-1")
        stats = manager.compute_delivery_stats()
        assert stats["queue_size"] == 1
        assert stats["pending_size"] == 0

    def test_retry_exceeds_max(self, manager):
        msg = QoSMessage(id="m1", payload=b"x", max_retries=2)
        manager.enqueue(msg)
        manager.dequeue()
        manager.retry("m1")
        manager.dequeue()
        manager.retry("m1")
        manager.dequeue()
        # Third retry should fail (max_retries=2)
        result = manager.retry("m1")
        assert result is None

    def test_retry_dead_letter(self, manager):
        msg = QoSMessage(id="m1", payload=b"x", max_retries=1)
        manager.enqueue(msg)
        manager.dequeue()
        manager.retry("m1")
        manager.dequeue()
        result = manager.retry("m1")
        assert result is None
        stats = manager.compute_delivery_stats()
        assert stats["total_dead"] == 1


class TestIsExpired:
    def test_not_expired(self, manager, sample_msg):
        assert not manager.is_expired(sample_msg)

    def test_expired_message(self, manager):
        msg = QoSMessage(id="old", payload=b"x", deadline=time.time() - 1)
        assert manager.is_expired(msg)

    def test_just_in_time(self, manager):
        msg = QoSMessage(id="edge", payload=b"x", deadline=time.time() + 0.1)
        assert not manager.is_expired(msg)


class TestPrioritizeByDeadline:
    def test_sorts_by_deadline(self, manager):
        m1 = QoSMessage(id="urgent", payload=b"x", deadline=time.time() + 1)
        m2 = QoSMessage(id="relaxed", payload=b"x", deadline=time.time() + 100)
        result = manager.prioritize_by_deadline([m2, m1])
        assert result[0].id == "urgent"
        assert result[1].id == "relaxed"

    def test_same_deadline_qos_breaks_tie(self, manager):
        deadline = time.time() + 10
        m1 = QoSMessage(id="best", payload=b"x", deadline=deadline, qos_level=QoSLevel.BEST_EFFORT)
        m2 = QoSMessage(id="critical", payload=b"x", deadline=deadline, qos_level=QoSLevel.CRITICAL)
        result = manager.prioritize_by_deadline([m1, m2])
        assert result[0].id == "critical"

    def test_empty_list(self, manager):
        assert manager.prioritize_by_deadline([]) == []

    def test_single_message(self, manager):
        m = QoSMessage(id="only", payload=b"x")
        result = manager.prioritize_by_deadline([m])
        assert len(result) == 1


class TestThrottleQueue:
    def test_basic_throttle(self, manager):
        msgs = [QoSMessage(id=f"m{i}", payload=b"x") for i in range(10)]
        result = manager.throttle_queue(msgs, max_rate=3.0)
        assert len(result) <= 3

    def test_zero_rate(self, manager):
        msgs = [QoSMessage(id="m1", payload=b"x")]
        result = manager.throttle_queue(msgs, max_rate=0)
        assert result == []

    def test_high_rate_all_pass(self, manager):
        msgs = [QoSMessage(id=f"m{i}", payload=b"x") for i in range(5)]
        result = manager.throttle_queue(msgs, max_rate=100.0)
        assert len(result) == 5

    def test_qos_priority_kept(self, manager):
        best = QoSMessage(id="best", payload=b"x", qos_level=QoSLevel.BEST_EFFORT)
        critical = QoSMessage(id="crit", payload=b"x", qos_level=QoSLevel.CRITICAL)
        reliable = QoSMessage(id="rel", payload=b"x", qos_level=QoSLevel.RELIABLE)
        result = manager.throttle_queue([best, critical, reliable], max_rate=2.0)
        ids = [m.id for m in result]
        # Critical and Reliable should be in the top 2
        assert "crit" in ids
        assert "rel" in ids

    def test_empty_queue(self, manager):
        assert manager.throttle_queue([], max_rate=5.0) == []

    def test_negative_rate(self, manager):
        msgs = [QoSMessage(id="m1", payload=b"x")]
        result = manager.throttle_queue(msgs, max_rate=-1)
        assert result == []


class TestDeliveryStats:
    def test_empty_stats(self, manager):
        stats = manager.compute_delivery_stats()
        assert stats["total_sent"] == 0
        assert stats["total_acked"] == 0
        assert stats["total_retried"] == 0
        assert stats["total_dead"] == 0
        assert stats["success_rate"] == 0.0

    def test_stats_after_operations(self, manager):
        m1 = QoSMessage(id="m1", payload=b"x")
        m2 = QoSMessage(id="m2", payload=b"x")
        manager.enqueue(m1)
        manager.enqueue(m2)
        manager.dequeue()  # m1 sent
        manager.dequeue()  # m2 sent
        manager.ack("m1")
        stats = manager.compute_delivery_stats()
        assert stats["total_sent"] == 2
        assert stats["total_acked"] == 1
        assert stats["success_rate"] == 0.5

    def test_stats_with_retry_and_dead(self, manager):
        m = QoSMessage(id="m", payload=b"x", max_retries=0)
        manager.enqueue(m)
        manager.dequeue()
        result = manager.retry("m")
        assert result is None  # already dead
        stats = manager.compute_delivery_stats()
        assert stats["total_dead"] == 1
        assert stats["total_retried"] == 0

    def test_queue_and_pending_sizes(self, manager):
        m1 = QoSMessage(id="m1", payload=b"x")
        m2 = QoSMessage(id="m2", payload=b"x")
        manager.enqueue(m1)
        manager.enqueue(m2)
        manager.dequeue()
        stats = manager.compute_delivery_stats()
        assert stats["queue_size"] == 1
        assert stats["pending_size"] == 1
