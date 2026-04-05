"""
Tests for the Network Simulator module.
"""

import time
import pytest
from jetson.agent.fleet_sync.network_simulator import (
    NetworkSimulator, NetworkConfig, NetworkCondition,
    NetworkMessage, NetworkStats, PartitionManager, PartitionEvent,
)


# ==============================================================================
# NetworkConfig Tests
# ==============================================================================

class TestNetworkConfig:
    """Test network configuration."""

    def test_from_condition_perfect(self):
        cfg = NetworkConfig.from_condition(NetworkCondition.PERFECT)
        assert cfg.base_latency_ms == 0.0
        assert cfg.packet_loss_pct == 0.0
        assert cfg.duplicate_pct == 0.0

    def test_from_condition_wifi(self):
        cfg = NetworkConfig.from_condition(NetworkCondition.WIFI)
        assert cfg.base_latency_ms == 10.0
        assert cfg.packet_loss_pct == 0.1
        assert cfg.bandwidth_bytes_per_sec == 5e7

    def test_from_condition_satellite(self):
        cfg = NetworkConfig.from_condition(NetworkCondition.SATELLITE)
        assert cfg.base_latency_ms == 1200.0
        assert cfg.packet_loss_pct == 3.0
        assert cfg.bandwidth_bytes_per_sec == 5e5

    def test_from_condition_storm(self):
        cfg = NetworkConfig.from_condition(NetworkCondition.STORM)
        assert cfg.base_latency_ms == 2000.0
        assert cfg.packet_loss_pct == 40.0
        assert cfg.bandwidth_bytes_per_sec == 5e3

    def test_from_condition_radio(self):
        cfg = NetworkConfig.from_condition(NetworkCondition.RADIO)
        assert cfg.base_latency_ms == 100.0
        assert cfg.bandwidth_bytes_per_sec == 1e4

    def test_from_condition_cellular(self):
        cfg = NetworkConfig.from_condition(NetworkCondition.CELLULAR)
        assert cfg.base_latency_ms == 200.0
        assert cfg.latency_jitter_ms == 300.0

    def test_from_condition_bad_weather(self):
        cfg = NetworkConfig.from_condition(NetworkCondition.BAD_WEATHER)
        assert cfg.base_latency_ms == 1000.0
        assert cfg.latency_jitter_ms == 2000.0
        assert cfg.packet_loss_pct == 20.0

    def test_default_config(self):
        cfg = NetworkConfig()
        assert cfg.base_latency_ms == 10.0
        assert cfg.packet_loss_pct == 1.0
        assert cfg.max_retries == 3

    def test_all_conditions_have_profiles(self):
        for condition in NetworkCondition:
            cfg = NetworkConfig.from_condition(condition)
            assert isinstance(cfg, NetworkConfig)
            assert cfg.condition == condition


# ==============================================================================
# NetworkStats Tests
# ==============================================================================

class TestNetworkStats:
    """Test network statistics."""

    def test_empty_stats(self):
        stats = NetworkStats()
        assert stats.messages_sent == 0
        assert stats.delivery_rate == 1.0
        assert stats.avg_latency_ms == 0.0

    def test_delivery_rate(self):
        stats = NetworkStats(messages_sent=100, messages_delivered=90)
        assert stats.delivery_rate == 0.9

    def test_avg_latency(self):
        stats = NetworkStats(messages_delivered=4, total_latency_ms=100.0)
        assert stats.avg_latency_ms == 25.0

    def test_to_dict(self):
        stats = NetworkStats(messages_sent=10, messages_delivered=8, total_latency_ms=100.0)
        d = stats.to_dict()
        assert d["messages_sent"] == 10
        assert d["delivery_rate"] == 0.8

    def test_min_latency_initial(self):
        stats = NetworkStats()
        assert stats.min_latency_ms == float("inf")


# ==============================================================================
# PartitionManager Tests
# ==============================================================================

class TestPartitionManager:
    """Test network partition management."""

    def test_no_partitions(self):
        pm = PartitionManager()
        pm.no_partitions(["v0", "v1", "v2"])
        assert pm.can_communicate("v0", "v1")
        assert pm.can_communicate("v1", "v2")
        assert pm.can_communicate("v0", "v2")

    def test_split(self):
        pm = PartitionManager()
        pm.split(["v0", "v1", "v2", "v3"], {"v0", "v1"})
        assert pm.can_communicate("v0", "v1")
        assert pm.can_communicate("v2", "v3")
        assert not pm.can_communicate("v0", "v2")
        assert not pm.can_communicate("v1", "v3")

    def test_get_reachable(self):
        pm = PartitionManager()
        pm.split(["v0", "v1", "v2", "v3"], {"v0", "v1"})
        assert pm.get_reachable("v0") == {"v1"}
        assert pm.get_reachable("v2") == {"v3"}

    def test_get_disconnected_from(self):
        pm = PartitionManager()
        pm.split(["v0", "v1", "v2", "v3"], {"v0", "v1"})
        assert pm.get_disconnected_from("v0") == {"v2", "v3"}

    def test_set_groups(self):
        pm = PartitionManager()
        pm.set_groups([{"v0", "v1"}, {"v2"}, {"v3"}])
        assert pm.can_communicate("v0", "v1")
        assert not pm.can_communicate("v0", "v2")

    def test_three_way_partition(self):
        pm = PartitionManager()
        pm.set_groups([{"v0"}, {"v1"}, {"v2"}])
        assert not pm.can_communicate("v0", "v1")
        assert not pm.can_communicate("v1", "v2")
        assert not pm.can_communicate("v0", "v2")

    def test_heal_partition(self):
        pm = PartitionManager()
        pm.split(["v0", "v1", "v2"], {"v0"})
        assert not pm.can_communicate("v0", "v1")
        pm.no_partitions(["v0", "v1", "v2"])
        assert pm.can_communicate("v0", "v1")

    def test_empty_split(self):
        pm = PartitionManager()
        pm.split(["v0", "v1"], set())
        assert pm.can_communicate("v0", "v1")

    def test_full_split(self):
        pm = PartitionManager()
        pm.split(["v0", "v1"], {"v0", "v1"})
        assert pm.can_communicate("v0", "v1")


# ==============================================================================
# NetworkSimulator Tests
# ==============================================================================

class TestNetworkSimulator:
    """Test the network simulator."""

    def test_add_vessel(self):
        net = NetworkSimulator()
        net.add_vessel("v0")
        assert "v0" in net.vessel_ids

    def test_add_vessels(self):
        net = NetworkSimulator()
        net.add_vessels(["v0", "v1", "v2"])
        assert net.vessel_ids == ["v0", "v1", "v2"]

    def test_perfect_network_delivery(self):
        net = NetworkSimulator(config=NetworkConfig.from_condition(NetworkCondition.PERFECT), seed=42)
        net.add_vessels(["v0", "v1"])
        assert net.send("v0", "v1", {"test": 1})
        msgs = net.drain()
        assert len(msgs) == 1
        assert msgs[0].payload == {"test": 1}

    def test_send_unknown_vessel(self):
        net = NetworkSimulator()
        assert not net.send("unknown", "v1", {})

    def test_blocked_by_partition(self):
        net = NetworkSimulator(seed=42)
        net.add_vessels(["v0", "v1", "v2"])
        net.partition_manager.split(["v0", "v1", "v2"], {"v0"})
        assert not net.send("v0", "v1", {})
        assert net.send("v0", "v0", {})  # can send to self group
        stats = net.get_stats()
        assert stats.messages_lost >= 1

    def test_packet_loss_with_seed(self):
        net = NetworkSimulator(
            config=NetworkConfig(base_latency_ms=0, packet_loss_pct=100.0, max_retries=1),
            seed=42,
        )
        net.add_vessels(["v0", "v1"])
        # 100% loss should mean no deliveries
        for _ in range(20):
            net.send("v0", "v1", {"i": 1})
        msgs = net.drain()
        assert len(msgs) == 0
        stats = net.get_stats()
        assert stats.messages_sent == 20
        assert stats.messages_lost == 20

    def test_zero_loss_network(self):
        net = NetworkSimulator(
            config=NetworkConfig(base_latency_ms=0, packet_loss_pct=0.0),
            seed=42,
        )
        net.add_vessels(["v0", "v1"])
        for i in range(10):
            net.send("v0", "v1", {"i": i})
        msgs = net.drain()
        assert len(msgs) == 10
        assert net.get_stats().delivery_rate == 1.0

    def test_link_config_override(self):
        net = NetworkSimulator(
            config=NetworkConfig.from_condition(NetworkCondition.PERFECT),
            seed=42,
        )
        net.add_vessels(["v0", "v1"])
        net.set_link_config("v0", "v1", NetworkConfig(
            base_latency_ms=0, packet_loss_pct=100.0, max_retries=1
        ))
        net.send("v0", "v1", {"test": 1})
        msgs = net.drain()
        assert len(msgs) == 0

    def test_duplicate_detection(self):
        net = NetworkSimulator(
            config=NetworkConfig(base_latency_ms=0, duplicate_pct=100.0, packet_loss_pct=0.0),
            seed=42,
        )
        net.add_vessels(["v0", "v1"])
        net.send("v0", "v1", {"test": 1})
        msgs = net.drain()
        assert len(msgs) >= 2  # Original + duplicate
        stats = net.get_stats()
        assert stats.messages_duplicated >= 1

    def test_get_delivered_for(self):
        net = NetworkSimulator(
            config=NetworkConfig(base_latency_ms=0, packet_loss_pct=0.0),
            seed=42,
        )
        net.add_vessels(["v0", "v1", "v2"])
        net.send("v0", "v1", {"a": 1})
        net.send("v2", "v1", {"b": 2})
        net.drain()
        v1_msgs = net.get_delivered_for("v1")
        assert len(v1_msgs) == 2

    def test_clear_delivered(self):
        net = NetworkSimulator(
            config=NetworkConfig(base_latency_ms=0, packet_loss_pct=0.0),
            seed=42,
        )
        net.add_vessels(["v0", "v1"])
        net.send("v0", "v1", {"test": 1})
        net.drain()
        assert len(net._delivered) > 0
        net.clear_delivered()
        assert len(net._delivered) == 0

    def test_reset_stats(self):
        net = NetworkSimulator(seed=42)
        net.add_vessels(["v0", "v1"])
        net.send("v0", "v1", {"test": 1})
        net.drain()
        net.reset_stats()
        stats = net.get_stats()
        assert stats.messages_sent == 0
        assert stats.messages_delivered == 0

    def test_inflight_count(self):
        net = NetworkSimulator(
            config=NetworkConfig(base_latency_ms=5000, packet_loss_pct=0.0),
            seed=42,
        )
        net.add_vessels(["v0", "v1"])
        net.send("v0", "v1", {"test": 1})
        assert net.inflight_count == 1
        net.drain()
        assert net.inflight_count == 0

    def test_schedule_partition(self):
        net = NetworkSimulator(
            config=NetworkConfig(base_latency_ms=0, packet_loss_pct=0.0),
            seed=42,
        )
        net.add_vessels(["v0", "v1", "v2"])
        net.schedule_partition(0.5, [{"v0"}, {"v1", "v2"}], "split")
        # Before partition time
        net.tick(0.3)
        assert net.partition_manager.can_communicate("v0", "v1")
        # After partition time
        net.tick(0.6)
        assert not net.partition_manager.can_communicate("v0", "v1")

    def test_multiple_sequential_sends(self):
        net = NetworkSimulator(
            config=NetworkConfig(base_latency_ms=0, packet_loss_pct=0.0),
            seed=42,
        )
        net.add_vessels(["v0", "v1"])
        for i in range(50):
            net.send("v0", "v1", {"i": i})
        msgs = net.drain()
        assert len(msgs) == 50

    def test_bidirectional_send(self):
        net = NetworkSimulator(
            config=NetworkConfig(base_latency_ms=0, packet_loss_pct=0.0),
            seed=42,
        )
        net.add_vessels(["v0", "v1"])
        net.send("v0", "v1", {"from": "v0"})
        net.send("v1", "v0", {"from": "v1"})
        msgs = net.drain()
        assert len(msgs) == 2

    def test_message_size_tracking(self):
        net = NetworkSimulator(
            config=NetworkConfig(base_latency_ms=0, packet_loss_pct=0.0),
            seed=42,
        )
        net.add_vessels(["v0", "v1"])
        net.send("v0", "v1", {"data": "x" * 100})
        net.drain()
        stats = net.get_stats()
        assert stats.bytes_transmitted > 0

    def test_network_condition_enum_values(self):
        for cond in NetworkCondition:
            assert isinstance(cond.value, str)
            cfg = NetworkConfig.from_condition(cond)
            assert cfg.condition == cond

    def test_vessel_ids_sorted(self):
        net = NetworkSimulator()
        net.add_vessels(["v2", "v0", "v1"])
        assert net.vessel_ids == ["v0", "v1", "v2"]

    def test_self_send(self):
        net = NetworkSimulator(
            config=NetworkConfig(base_latency_ms=0, packet_loss_pct=0.0),
            seed=42,
        )
        net.add_vessels(["v0"])
        assert net.send("v0", "v0", {"self": True})
        msgs = net.drain()
        assert len(msgs) == 1
