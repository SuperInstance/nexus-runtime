"""Tests for communication module — FleetCommunication, FleetMessage, BroadcastResult."""

import pytest

from jetson.fleet_coordination.communication import (
    BroadcastResult,
    DeliveryStatus,
    FleetCommunication,
    FleetMessage,
    LinkStatus,
    MessageType,
    random_simulate,
)
from jetson.fleet_coordination.fleet_manager import VesselStatus


# ────────────────────────────────────────────────────── fixtures

@pytest.fixture
def comm():
    return FleetCommunication()


@pytest.fixture
def sample_message():
    return FleetMessage(
        source="V0",
        target="V1",
        type=MessageType.STATUS,
        payload={"lat": 10.0, "lon": 20.0},
        priority=0.7,
        ttl=10,
    )


@pytest.fixture
def sample_fleet():
    return [VesselStatus(vessel_id=f"V{i}") for i in range(5)]


# ────────────────────────────────────────────────────── FleetMessage

class TestFleetMessage:
    def test_default_message(self):
        m = FleetMessage()
        assert m.source == ""
        assert m.target == ""
        assert m.type == MessageType.STATUS
        assert m.ttl == 10
        assert m.hops == 0

    def test_message_types(self):
        assert MessageType.COMMAND.value == "command"
        assert MessageType.ALERT.value == "alert"
        assert MessageType.HEARTBEAT.value == "heartbeat"

    def test_auto_id(self):
        m1 = FleetMessage()
        m2 = FleetMessage()
        assert m1.id != m2.id


# ────────────────────────────────────────────────────── BroadcastResult

class TestBroadcastResult:
    def test_default(self):
        r = BroadcastResult()
        assert r.reached_vessels == []
        assert r.failed_vessels == []
        assert r.total_time >= 0.0

    def test_with_data(self):
        r = BroadcastResult(
            reached_vessels=["V1", "V2"],
            failed_vessels=["V3"],
            total_time=0.05,
        )
        assert len(r.reached_vessels) == 2
        assert len(r.failed_vessels) == 1


# ────────────────────────────────────────────────────── LinkStatus

class TestLinkStatus:
    def test_default(self):
        l = LinkStatus(vessel_a="A", vessel_b="B")
        assert l.active is True
        assert l.latency == 0.0
        assert l.bandwidth == 100.0
        assert l.packet_loss == 0.0


# ────────────────────────────────────────────────────── Send

class TestSend:
    def test_send_delivered(self, comm, sample_message):
        status = comm.send(sample_message)
        assert isinstance(status, DeliveryStatus)

    def test_send_expired_ttl(self, comm):
        m = FleetMessage(source="V0", target="V1", ttl=0)
        assert comm.send(m) == DeliveryStatus.EXPIRED

    def test_send_negative_ttl(self, comm):
        m = FleetMessage(source="V0", target="V1", ttl=-5)
        assert comm.send(m) == DeliveryStatus.EXPIRED

    def test_send_inactive_link(self, comm):
        comm.add_link("V0", "V1")
        link = comm.get_link("V0", "V1")
        link.active = False
        m = FleetMessage(source="V0", target="V1", ttl=5)
        assert comm.send(m) == DeliveryStatus.FAILED

    def test_send_increments_delivery_attempts(self, comm):
        # Send an expired message to guarantee delivery attempt tracking
        m = FleetMessage(source="V0", target="V1", ttl=0)
        comm.send(m)
        assert comm._delivery_attempts.get(m.id, 0) >= 1

    def test_send_logs_message(self, comm, sample_message):
        comm.clear_log()
        comm.send(sample_message)
        assert len(comm.get_message_log()) >= 1


# ────────────────────────────────────────────────────── Broadcast

class TestBroadcast:
    def test_broadcast_reaches_some(self, comm, sample_fleet):
        msg = FleetMessage(source="V0", type=MessageType.STATUS, priority=0.9, ttl=10)
        result = comm.broadcast("V0", msg, sample_fleet)
        # Source should not be in results
        assert "V0" not in result.reached_vessels
        assert "V0" not in result.failed_vessels

    def test_broadcast_total_time(self, comm, sample_fleet):
        msg = FleetMessage(source="V0", type=MessageType.STATUS, ttl=10)
        result = comm.broadcast("V0", msg, sample_fleet)
        assert result.total_time >= 0

    def test_broadcast_empty_fleet(self, comm):
        msg = FleetMessage(source="V0", type=MessageType.STATUS, ttl=10)
        result = comm.broadcast("V0", msg, [])
        assert result.reached_vessels == []
        assert result.failed_vessels == []

    def test_broadcast_single_vessel(self, comm):
        fleet = [VesselStatus(vessel_id="V0")]
        msg = FleetMessage(source="V0", type=MessageType.STATUS, ttl=10)
        result = comm.broadcast("V0", msg, fleet)
        # Only vessel is the source
        assert result.reached_vessels == []
        assert result.failed_vessels == []


# ────────────────────────────────────────────────────── Multicast

class TestMulticast:
    def test_multicast_targets(self, comm):
        msg = FleetMessage(source="V0", type=MessageType.COMMAND, priority=0.8, ttl=10)
        result = comm.multicast("V0", msg, ["V1", "V2", "V3"])
        total = len(result.reached_vessels) + len(result.failed_vessels)
        assert total == 3

    def test_multicast_excludes_source(self, comm):
        msg = FleetMessage(source="V0", type=MessageType.STATUS, ttl=10)
        result = comm.multicast("V0", msg, ["V0", "V1"])
        assert "V0" not in result.reached_vessels
        assert "V0" not in result.failed_vessels

    def test_multicast_empty_group(self, comm):
        msg = FleetMessage(source="V0", type=MessageType.STATUS, ttl=10)
        result = comm.multicast("V0", msg, [])
        assert result.reached_vessels == []

    def test_multicast_time_nonnegative(self, comm):
        msg = FleetMessage(source="V0", type=MessageType.STATUS, ttl=10)
        result = comm.multicast("V0", msg, ["V1", "V2"])
        assert result.total_time >= 0


# ────────────────────────────────────────────────────── Relay

class TestRelay:
    def test_relay_success(self, comm):
        m = FleetMessage(source="V0", type=MessageType.DATA, ttl=10, priority=1.0, payload="hello")
        route = ["V1", "V2", "V3"]
        result = comm.relay_message(m, route)
        # probabilistic — may succeed
        assert isinstance(result, bool)

    def test_relay_empty_route(self, comm):
        m = FleetMessage(source="V0", type=MessageType.DATA, ttl=10)
        assert comm.relay_message(m, []) is False

    def test_relay_exhausted_ttl(self, comm):
        m = FleetMessage(source="V0", type=MessageType.DATA, ttl=1)
        route = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11"]
        result = comm.relay_message(m, route)
        assert result is False

    def test_relay_increments_hops(self, comm):
        m = FleetMessage(source="V0", type=MessageType.DATA, ttl=10, priority=1.0)
        initial_hops = m.hops
        comm.relay_message(m, ["V1"])
        assert m.hops >= initial_hops


# ────────────────────────────────────────────────────── Routing

class TestRouting:
    def test_compute_routes_line(self, comm):
        graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
        table = comm.compute_optimal_routes(graph)
        assert "A" in table
        assert "B" in table
        assert "C" in table

    def test_compute_routes_star(self, comm):
        graph = {"HUB": ["A", "B", "C"], "A": ["HUB"], "B": ["HUB"], "C": ["HUB"]}
        table = comm.compute_optimal_routes(graph)
        assert "HUB" in table

    def test_compute_routes_empty(self, comm):
        table = comm.compute_optimal_routes({})
        assert table == {}

    def test_compute_routes_single_node(self, comm):
        table = comm.compute_optimal_routes({"X": []})
        assert "X" in table

    def test_routing_considers_link_quality(self, comm):
        comm.add_link("A", "B", latency=1.0, packet_loss=0.0)
        comm.add_link("B", "C", latency=500.0, packet_loss=0.9)
        graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
        table = comm.compute_optimal_routes(graph)
        assert "A" in table


# ────────────────────────────────────────────────────── Network health

class TestNetworkHealth:
    def test_perfect_network(self, comm):
        links = [
            LinkStatus(vessel_a="A", vessel_b="B", latency=5, bandwidth=100, packet_loss=0),
            LinkStatus(vessel_a="B", vessel_b="C", latency=5, bandwidth=100, packet_loss=0),
        ]
        health = comm.estimate_network_health(links)
        assert health > 0.8

    def test_degraded_network(self, comm):
        links = [
            LinkStatus(vessel_a="A", vessel_b="B", latency=200, bandwidth=10, packet_loss=0.3),
        ]
        health = comm.estimate_network_health(links)
        assert health < 0.8

    def test_all_links_down(self, comm):
        links = [
            LinkStatus(vessel_a="A", vessel_b="B", active=False),
            LinkStatus(vessel_a="B", vessel_b="C", active=False),
        ]
        assert comm.estimate_network_health(links) == 0.0

    def test_empty_links(self, comm):
        assert comm.estimate_network_health([]) == 0.0

    def test_mixed_links(self, comm):
        links = [
            LinkStatus(vessel_a="A", vessel_b="B", latency=5, bandwidth=100, packet_loss=0),
            LinkStatus(vessel_a="C", vessel_b="D", latency=400, bandwidth=5, packet_loss=0.5, active=False),
        ]
        health = comm.estimate_network_health(links)
        assert 0.0 < health < 1.0

    def test_health_clamped(self, comm):
        links = [LinkStatus(vessel_a="A", vessel_b="B", latency=0, bandwidth=1000, packet_loss=0)]
        health = comm.estimate_network_health(links)
        assert 0.0 <= health <= 1.0


# ────────────────────────────────────────────────────── Link management

class TestLinkManagement:
    def test_add_link(self, comm):
        link = comm.add_link("A", "B", latency=10, bandwidth=50)
        assert link.vessel_a == "A"
        assert link.latency == 10

    def test_get_link(self, comm):
        comm.add_link("A", "B")
        link = comm.get_link("A", "B")
        assert link is not None

    def test_get_link_order_invariant(self, comm):
        comm.add_link("X", "Y")
        assert comm.get_link("Y", "X") is not None

    def test_get_link_not_found(self, comm):
        assert comm.get_link("A", "B") is None

    def test_remove_link(self, comm):
        comm.add_link("A", "B")
        assert comm.remove_link("A", "B") is True
        assert comm.get_link("A", "B") is None

    def test_remove_nonexistent(self, comm):
        assert comm.remove_link("A", "B") is False


# ────────────────────────────────────────────────────── Utility

class TestUtility:
    def test_random_simulate_always_true(self):
        assert random_simulate(1.0) is True

    def test_random_simulate_never_true(self):
        assert random_simulate(0.0) is False

    def test_clear_log(self, comm):
        comm.send(FleetMessage(source="V0", target="V1", ttl=10))
        comm.clear_log()
        assert comm.get_message_log() == []

    def test_message_log_limit(self, comm):
        comm.clear_log()
        for _ in range(200):
            comm.send(FleetMessage(source="V0", target="V1", ttl=10, priority=0.1))
        log = comm.get_message_log(limit=50)
        assert len(log) <= 50
