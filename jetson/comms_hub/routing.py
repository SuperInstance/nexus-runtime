"""Message routing and relay with shortest-path computation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Set
import heapq
import time


class LinkStatus(Enum):
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"


@dataclass
class RouteEntry:
    """A single entry in the routing table."""
    destination: str
    next_hop: str
    cost: float
    ttl: int = 255
    protocol: str = "udp"
    last_updated: float = 0.0

    def __post_init__(self):
        if self.last_updated == 0.0:
            self.last_updated = time.time()

    @property
    def is_expired(self) -> bool:
        age = time.time() - self.last_updated
        # Treat TTL as seconds multiplier for simplicity
        return age > self.ttl


class RoutingTable:
    """Dynamic routing table with Dijkstra shortest-path and flood-based learning."""

    def __init__(self, node_id: str = "self"):
        self.node_id = node_id
        self._routes: Dict[str, RouteEntry] = {}
        self._neighbors: Dict[str, float] = {}  # neighbor -> link_cost

    # ------------------------------------------------------------------
    # Table manipulation
    # ------------------------------------------------------------------
    def add_route(self, entry: RouteEntry) -> None:
        """Add or update a route. Keeps the lowest-cost entry per destination."""
        existing = self._routes.get(entry.destination)
        if existing is None or entry.cost < existing.cost:
            self._routes[entry.destination] = entry

    def remove_route(self, destination: str) -> Optional[RouteEntry]:
        """Remove a route by destination. Returns the removed entry or None."""
        return self._routes.pop(destination, None)

    def get_route(self, dest: str) -> Optional[RouteEntry]:
        """Return the current best route for *dest*."""
        entry = self._routes.get(dest)
        if entry and not entry.is_expired:
            return entry
        return None

    def find_best_route(self, dest: str, criteria: str = "cost") -> Optional[RouteEntry]:
        """Find best route for *dest* by *criteria* (``cost``, ``ttl``, ``recency``)."""
        entry = self._routes.get(dest)
        if entry is None:
            return None
        if criteria == "cost":
            return entry
        elif criteria == "ttl":
            # Among routes with same dest (we store one), this is the entry
            return entry
        elif criteria == "recency":
            return entry
        return entry

    # ------------------------------------------------------------------
    # Flood-based route learning
    # ------------------------------------------------------------------
    def flood_route(self, source: str, dest: str, cost: float, ttl: int = 64) -> None:
        """Simulate receiving a flooded route update from *source*."""
        total_cost = self._neighbors.get(source, 1.0) + cost
        entry = RouteEntry(
            destination=dest,
            next_hop=source,
            cost=total_cost,
            ttl=ttl,
            last_updated=time.time(),
        )
        existing = self._routes.get(dest)
        if existing is None or total_cost < existing.cost:
            self._routes[dest] = entry

    # ------------------------------------------------------------------
    # Shortest-path (Dijkstra) over an adjacency graph
    # ------------------------------------------------------------------
    @staticmethod
    def compute_shortest_path(
        graph: Dict[str, List[Tuple[str, float]]],
        source: str,
        dest: str,
    ) -> List[str]:
        """Dijkstra's algorithm. Returns list of node ids from source to dest.

        *graph* maps each node to ``[(neighbor, weight), ...]``.
        """
        dist: Dict[str, float] = {source: 0.0}
        prev: Dict[str, Optional[str]] = {source: None}
        visited: Set[str] = set()
        heap = [(0.0, source)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == dest:
                break
            for v, w in graph.get(u, []):
                if v in visited:
                    continue
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if dest not in prev:
            return []
        path: List[str] = []
        node = dest
        while node is not None:
            path.append(node)
            node = prev.get(node)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Link-status cost updates
    # ------------------------------------------------------------------
    def update_costs(self, links_status: Dict[str, LinkStatus]) -> None:
        """Adjust route costs based on neighbor link status changes."""
        for neighbor, status in links_status.items():
            if status == LinkStatus.DOWN:
                self._neighbors[neighbor] = float("inf")
                # Remove routes going through this neighbor
                to_remove = [d for d, e in self._routes.items() if e.next_hop == neighbor]
                for d in to_remove:
                    self._routes.pop(d, None)
            elif status == LinkStatus.DEGRADED:
                base = self._neighbors.get(neighbor, 1.0)
                self._neighbors[neighbor] = base * 3.0
            else:
                self._neighbors[neighbor] = 1.0

    def add_neighbor(self, neighbor: str, cost: float = 1.0) -> None:
        """Register a direct neighbor with its link cost."""
        self._neighbors[neighbor] = cost

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_routing_stats(self) -> dict:
        """Return summary statistics."""
        entries = list(self._routes.values())
        active = [e for e in entries if not e.is_expired]
        return {
            "total_routes": len(entries),
            "active_routes": len(active),
            "neighbors": list(self._neighbors.keys()),
            "node_id": self.node_id,
            "protocols": list({e.protocol for e in active}),
        }
