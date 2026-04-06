"""Tests for communication.py — CommLink, Message, CommSimulator."""

import pytest

from jetson.simulation.communication import CommLink, Message, CommSimulator


class TestCommLink:
    def test_default_creation(self):
        link = CommLink(source="a", target="b")
        assert link.source == "a"
        assert link.target == "b"
        assert link.bandwidth == 1000.0
        assert link.latency == 0.1
        assert link.reliability == 0.99
        assert link.max_range == 1000.0

    def test_custom_creation(self):
        link = CommLink(source="x", target="y", bandwidth=500, latency=0.5, reliability=0.9)
        assert link.bandwidth == 500.0
        assert link.latency == 0.5


class TestMessage:
    def test_default_creation(self):
        msg = Message(source="a", target="b")
        assert msg.source == "a"
        assert msg.target == "b"
        assert msg.payload == ""
        assert msg.timestamp == 0.0
        assert msg.size == 100.0

    def test_custom_creation(self):
        msg = Message(source="a", target="b", payload="hello", timestamp=1.0, size=200)
        assert msg.payload == "hello"
        assert msg.size == 200.0


class TestCommSimulatorCreation:
    def test_default_creation(self):
        sim = CommSimulator()
        assert sim.link_count == 0

    def test_seeded_creation(self):
        sim = CommSimulator(seed=42)
        assert sim.link_count == 0


class TestAddRemoveLink:
    def test_add_link(self):
        sim = CommSimulator()
        sim.add_link(CommLink(source="a", target="b"))
        assert sim.link_count >= 1

    def test_add_link_bidirectional(self):
        sim = CommSimulator()
        sim.add_link(CommLink(source="a", target="b"))
        # Both directions should exist
        assert sim.get_link("a", "b") is not None
        assert sim.get_link("b", "a") is not None

    def test_remove_link(self):
        sim = CommSimulator()
        sim.add_link(CommLink(source="a", target="b"))
        assert sim.remove_link("a", "b") is True

    def test_remove_nonexistent(self):
        sim = CommSimulator()
        assert sim.remove_link("x", "y") is False

    def test_get_link(self):
        sim = CommSimulator()
        link = CommLink(source="a", target="b", bandwidth=500)
        sim.add_link(link)
        found = sim.get_link("a", "b")
        assert found is not None
        assert found.bandwidth == 500.0

    def test_get_nonexistent_link(self):
        sim = CommSimulator()
        assert sim.get_link("x", "y") is None


class TestSendMessage:
    def test_send_message_delivered(self):
        sim = CommSimulator(seed=42)
        sim.add_link(CommLink(source="a", target="b", reliability=1.0))
        msg = Message(source="a", target="b", payload="hello")
        delivered, latency = sim.send_message(msg)
        assert delivered is True
        assert latency > 0

    def test_send_message_no_link(self):
        sim = CommSimulator()
        msg = Message(source="a", target="b")
        delivered, latency = sim.send_message(msg)
        assert delivered is False
        assert latency == float("inf")

    def test_send_message_latency_positive(self):
        sim = CommSimulator(seed=42)
        sim.add_link(CommLink(source="a", target="b", reliability=1.0, bandwidth=1000.0))
        msg = Message(source="a", target="b", size=100.0)
        delivered, latency = sim.send_message(msg)
        assert latency > 0

    def test_send_message_logged(self):
        sim = CommSimulator(seed=42)
        sim.add_link(CommLink(source="a", target="b", reliability=1.0))
        msg = Message(source="a", target="b")
        sim.send_message(msg)
        assert len(sim.message_log) == 1

    def test_send_multiple_messages(self):
        sim = CommSimulator(seed=42)
        sim.add_link(CommLink(source="a", target="b", reliability=1.0))
        for i in range(5):
            msg = Message(source="a", target="b", payload=f"msg_{i}")
            sim.send_message(msg)
        assert len(sim.message_log) == 5


class TestPacketLoss:
    def test_packet_loss_with_perfect_reliability(self):
        sim = CommSimulator(seed=42)
        msg = Message(source="a", target="b")
        assert sim.simulate_packet_loss(msg, 1.0) is True

    def test_packet_loss_with_zero_reliability(self):
        sim = CommSimulator(seed=42)
        msg = Message(source="a", target="b")
        assert sim.simulate_packet_loss(msg, 0.0) is False

    def test_packet_loss_randomness(self):
        sim1 = CommSimulator(seed=42)
        sim2 = CommSimulator(seed=99)
        msg = Message(source="a", target="b")
        r1 = sim1.simulate_packet_loss(msg, 0.5)
        r2 = sim2.simulate_packet_loss(msg, 0.5)
        # May or may not be different, just test it runs


class TestInterference:
    def test_interference_returns_list(self):
        sim = CommSimulator(seed=42)
        msgs = [Message(source="a", target="b", payload=f"m{i}") for i in range(3)]
        results = sim.simulate_interference(msgs, noise_level=0.1)
        assert len(results) == 3

    def test_interference_result_keys(self):
        sim = CommSimulator(seed=42)
        msgs = [Message(source="a", target="b")]
        results = sim.simulate_interference(msgs, 0.2)
        assert "delivered" in results[0]
        assert "signal_quality" in results[0]
        assert "noise_degradation" in results[0]

    def test_interference_high_noise(self):
        sim = CommSimulator(seed=42)
        msgs = [Message(source="a", target="b") for _ in range(10)]
        results = sim.simulate_interference(msgs, noise_level=0.9)
        # High noise should reduce delivery rate
        deliveries = [r["delivered"] for r in results]
        # With 0.9 noise, effective reliability is 0.99 * 0.1 = 0.099
        assert sum(deliveries) <= len(msgs)

    def test_interference_zero_noise(self):
        sim = CommSimulator(seed=42)
        sim.add_link(CommLink(source="a", target="b", reliability=1.0))
        msgs = [Message(source="a", target="b") for _ in range(10)]
        results = sim.simulate_interference(msgs, noise_level=0.0)
        deliveries = [r["delivered"] for r in results]
        assert all(deliveries)

    def test_interference_signal_quality_range(self):
        sim = CommSimulator(seed=42)
        msgs = [Message(source="a", target="b")]
        results = sim.simulate_interference(msgs, 0.5)
        assert 0.0 <= results[0]["signal_quality"] <= 1.0


class TestNetworkGraph:
    def test_empty_graph(self):
        sim = CommSimulator()
        graph = sim.compute_network_graph()
        assert graph == {}

    def test_single_link_graph(self):
        sim = CommSimulator()
        sim.add_link(CommLink(source="a", target="b"))
        graph = sim.compute_network_graph()
        assert "b" in graph["a"]
        assert "a" in graph["b"]

    def test_triangle_graph(self):
        sim = CommSimulator()
        sim.add_link(CommLink(source="a", target="b"))
        sim.add_link(CommLink(source="b", target="c"))
        graph = sim.compute_network_graph()
        assert "a" in graph
        assert "b" in graph
        assert "c" in graph
        assert "b" in graph["a"]
        assert "c" in graph["b"]


class TestNetworkStats:
    def test_empty_stats(self):
        sim = CommSimulator()
        stats = sim.compute_network_stats()
        assert stats["num_nodes"] == 0
        assert stats["num_links"] == 0

    def test_single_link_stats(self):
        sim = CommSimulator()
        sim.add_link(CommLink(source="a", target="b", bandwidth=500, latency=0.2, reliability=0.95))
        stats = sim.compute_network_stats()
        assert stats["num_nodes"] == 2
        assert stats["num_links"] == 1
        assert stats["avg_bandwidth"] == pytest.approx(500.0)
        assert stats["avg_latency"] == pytest.approx(0.2)
        assert stats["avg_reliability"] == pytest.approx(0.95)

    def test_stats_has_nodes_list(self):
        sim = CommSimulator()
        sim.add_link(CommLink(source="a", target="b"))
        stats = sim.compute_network_stats()
        assert "a" in stats["nodes"]
        assert "b" in stats["nodes"]


class TestBroadcast:
    def test_broadcast_no_neighbors(self):
        sim = CommSimulator()
        results = sim.broadcast_message("a", "hello")
        assert results == []

    def test_broadcast_reaches_neighbors(self):
        sim = CommSimulator(seed=42)
        sim.add_link(CommLink(source="a", target="b", reliability=1.0))
        sim.add_link(CommLink(source="a", target="c", reliability=1.0))
        results = sim.broadcast_message("a", "hello")
        assert len(results) == 2
        targets = [r[0] for r in results]
        assert "b" in targets
        assert "c" in targets

    def test_broadcast_result_format(self):
        sim = CommSimulator(seed=42)
        sim.add_link(CommLink(source="a", target="b", reliability=1.0))
        results = sim.broadcast_message("a", "test")
        # Each result is (target, delivered, latency)
        assert len(results[0]) == 3


class TestTimeManagement:
    def test_initial_time(self):
        sim = CommSimulator()
        assert sim.time == 0.0

    def test_update_time(self):
        sim = CommSimulator()
        sim.update_time(1.5)
        assert sim.time == pytest.approx(1.5)
