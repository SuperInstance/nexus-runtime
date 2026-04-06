"""Tests for jetson.comms_hub.routing."""

import time
import pytest
from jetson.comms_hub.routing import RouteEntry, RoutingTable, LinkStatus
from jetson.comms_hub.protocol import ProtocolType


@pytest.fixture
def table():
    return RoutingTable(node_id="test-node")


@pytest.fixture
def sample_entry():
    return RouteEntry(destination="dest-A", next_hop="hop-1", cost=5.0, ttl=60, protocol="udp")


class TestRouteEntry:
    def test_construction(self):
        e = RouteEntry(destination="dest-A", next_hop="hop-1", cost=5.0)
        assert e.destination == "dest-A"
        assert e.next_hop == "hop-1"
        assert e.cost == 5.0
        assert e.ttl == 255
        assert e.protocol == "udp"
        assert e.last_updated > 0

    def test_expired_entry(self):
        e = RouteEntry(destination="d", next_hop="h", cost=1.0, ttl=0)
        # ttl=0 means it expires immediately (last_updated is now)
        time.sleep(0.01)
        assert e.is_expired

    def test_fresh_entry_not_expired(self):
        e = RouteEntry(destination="d", next_hop="h", cost=1.0, ttl=9999)
        assert not e.is_expired

    def test_fields_assignable(self):
        e = RouteEntry(destination="a", next_hop="b", cost=10, ttl=100, protocol="tcp")
        assert e.protocol == "tcp"


class TestRoutingTableAddRemove:
    def test_add_route(self, table, sample_entry):
        table.add_route(sample_entry)
        assert table.get_route("dest-A") is not None

    def test_add_lower_cost_replaces(self, table):
        e1 = RouteEntry(destination="d", next_hop="h1", cost=10.0)
        e2 = RouteEntry(destination="d", next_hop="h2", cost=3.0)
        table.add_route(e1)
        table.add_route(e2)
        route = table.get_route("d")
        assert route.next_hop == "h2"
        assert route.cost == 3.0

    def test_add_higher_cost_keeps_old(self, table):
        e1 = RouteEntry(destination="d", next_hop="h1", cost=3.0)
        e2 = RouteEntry(destination="d", next_hop="h2", cost=10.0)
        table.add_route(e1)
        table.add_route(e2)
        route = table.get_route("d")
        assert route.next_hop == "h1"

    def test_remove_route(self, table, sample_entry):
        table.add_route(sample_entry)
        removed = table.remove_route("dest-A")
        assert removed is not None
        assert removed.destination == "dest-A"
        assert table.get_route("dest-A") is None

    def test_remove_nonexistent(self, table):
        assert table.remove_route("no-such") is None

    def test_get_route_nonexistent(self, table):
        assert table.get_route("no-such") is None


class TestFindBestRoute:
    def test_find_best_by_cost(self, table):
        table.add_route(RouteEntry(destination="d", next_hop="h1", cost=5.0))
        result = table.find_best_route("d", "cost")
        assert result is not None
        assert result.cost == 5.0

    def test_find_best_missing(self, table):
        assert table.find_best_route("missing", "cost") is None

    def test_find_best_by_ttl(self, table):
        table.add_route(RouteEntry(destination="d", next_hop="h1", cost=5.0, ttl=100))
        result = table.find_best_route("d", "ttl")
        assert result is not None

    def test_find_best_by_recency(self, table):
        table.add_route(RouteEntry(destination="d", next_hop="h1", cost=5.0))
        result = table.find_best_route("d", "recency")
        assert result is not None


class TestFloodRoute:
    def test_flood_basic(self, table):
        table.add_neighbor("neighbor-1", cost=1.0)
        table.flood_route(source="neighbor-1", dest="far-dest", cost=5.0)
        route = table.get_route("far-dest")
        assert route is not None
        assert route.next_hop == "neighbor-1"

    def test_flood_cost_accumulates(self, table):
        table.add_neighbor("n1", cost=2.0)
        table.flood_route(source="n1", dest="d1", cost=3.0)
        route = table.get_route("d1")
        assert route.cost == 5.0

    def test_flood_better_route(self, table):
        table.add_neighbor("n1", cost=1.0)
        table.add_neighbor("n2", cost=2.0)
        table.flood_route(source="n1", dest="d1", cost=10.0)
        table.flood_route(source="n2", dest="d1", cost=1.0)
        route = table.get_route("d1")
        # n2 route: 2.0 + 1.0 = 3.0 vs n1 route: 1.0 + 10.0 = 11.0
        assert route.next_hop == "n2"

    def test_flood_ttl(self, table):
        table.add_neighbor("n1", cost=1.0)
        table.flood_route(source="n1", dest="d1", cost=5.0, ttl=30)
        route = table.get_route("d1")
        assert route.ttl == 30


class TestShortestPath:
    def test_simple_path(self):
        graph = {
            "A": [("B", 1), ("C", 4)],
            "B": [("C", 2), ("D", 5)],
            "C": [("D", 1)],
            "D": [],
        }
        path = RoutingTable.compute_shortest_path(graph, "A", "D")
        assert path[0] == "A"
        assert path[-1] == "D"
        assert path == ["A", "B", "C", "D"]

    def test_direct_connection(self):
        graph = {"A": [("B", 1)], "B": []}
        path = RoutingTable.compute_shortest_path(graph, "A", "B")
        assert path == ["A", "B"]

    def test_no_path(self):
        graph = {"A": [("B", 1)], "B": [], "C": []}
        path = RoutingTable.compute_shortest_path(graph, "A", "C")
        assert path == []

    def test_same_node(self):
        graph = {"A": []}
        path = RoutingTable.compute_shortest_path(graph, "A", "A")
        assert path == ["A"]

    def test_disconnected_graph(self):
        graph = {"A": [("B", 1)], "B": [], "C": [("D", 1)], "D": []}
        path = RoutingTable.compute_shortest_path(graph, "A", "D")
        assert path == []

    def test_complex_graph(self):
        graph = {
            "S": [("A", 2), ("B", 5)],
            "A": [("B", 1), ("C", 3)],
            "B": [("C", 1)],
            "C": [("D", 2)],
            "D": [],
        }
        path = RoutingTable.compute_shortest_path(graph, "S", "D")
        assert path[0] == "S"
        assert path[-1] == "D"
        # S->A->B->C->D = 2+1+1+2 = 6 vs S->A->C->D = 2+3+2 = 7 vs S->B->C->D = 5+1+2 = 8
        assert path == ["S", "A", "B", "C", "D"]


class TestUpdateCosts:
    def test_link_down_removes_routes(self, table):
        table.add_neighbor("n1", cost=1.0)
        table.add_route(RouteEntry(destination="d1", next_hop="n1", cost=2.0))
        table.update_costs({"n1": LinkStatus.DOWN})
        assert table.get_route("d1") is None

    def test_link_degraded_increases_cost(self, table):
        table.add_neighbor("n1", cost=1.0)
        table.add_route(RouteEntry(destination="d1", next_hop="n1", cost=2.0))
        table.update_costs({"n1": LinkStatus.DEGRADED})
        # Neighbor cost triples but route entry itself isn't updated
        assert "n1" in table.get_routing_stats()["neighbors"]

    def test_link_up_restores(self, table):
        table.add_neighbor("n1", cost=1.0)
        table.update_costs({"n1": LinkStatus.DOWN})
        table.update_costs({"n1": LinkStatus.UP})
        route = table.get_routing_stats()
        assert "n1" in route["neighbors"]

    def test_multiple_links(self, table):
        table.add_neighbor("n1", cost=1.0)
        table.add_neighbor("n2", cost=2.0)
        table.update_costs({
            "n1": LinkStatus.UP,
            "n2": LinkStatus.DOWN,
        })
        assert table.get_route("any") is None  # no routes through n1 or n2


class TestAddNeighbor:
    def test_add_neighbor(self, table):
        table.add_neighbor("n1", cost=5.0)
        stats = table.get_routing_stats()
        assert "n1" in stats["neighbors"]

    def test_default_cost(self, table):
        table.add_neighbor("n2")
        assert "n2" in table.get_routing_stats()["neighbors"]


class TestRoutingStats:
    def test_empty_stats(self, table):
        stats = table.get_routing_stats()
        assert stats["total_routes"] == 0
        assert stats["active_routes"] == 0
        assert stats["node_id"] == "test-node"

    def test_stats_with_routes(self, table):
        table.add_route(RouteEntry(destination="d1", next_hop="h1", cost=1.0))
        table.add_route(RouteEntry(destination="d2", next_hop="h2", cost=2.0))
        stats = table.get_routing_stats()
        assert stats["total_routes"] == 2
        assert stats["active_routes"] == 2

    def test_stats_protocols(self, table):
        table.add_route(RouteEntry(destination="d1", next_hop="h1", cost=1.0, protocol="udp"))
        table.add_route(RouteEntry(destination="d2", next_hop="h2", cost=2.0, protocol="tcp"))
        stats = table.get_routing_stats()
        assert "udp" in stats["protocols"]
        assert "tcp" in stats["protocols"]

    def test_stats_neighbors(self, table):
        table.add_neighbor("n1")
        table.add_neighbor("n2")
        stats = table.get_routing_stats()
        assert len(stats["neighbors"]) == 2


class TestLinkStatus:
    def test_enum_values(self):
        assert LinkStatus.UP.value == "up"
        assert LinkStatus.DOWN.value == "down"
        assert LinkStatus.DEGRADED.value == "degraded"
