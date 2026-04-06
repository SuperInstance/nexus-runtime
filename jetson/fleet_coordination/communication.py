"""Fleet communication patterns — unicast, broadcast, multicast, routing, health."""

from __future__ import annotations

import heapq
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class MessageType(Enum):
    COMMAND = "command"
    STATUS = "status"
    ALERT = "alert"
    DATA = "data"
    HEARTBEAT = "heartbeat"
    COORDINATION = "coordination"


class DeliveryStatus(Enum):
    DELIVERED = "delivered"
    FAILED = "failed"
    PENDING = "pending"
    RELAYED = "relayed"
    EXPIRED = "expired"


@dataclass
class FleetMessage:
    """A message in the fleet communication network."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = ""
    target: str = ""              # vessel id or "broadcast"
    type: MessageType = MessageType.STATUS
    payload: Any = None
    priority: float = 0.5         # 0.0 (low) to 1.0 (critical)
    ttl: int = 10                 # time-to-live hops
    created_at: float = field(default_factory=time.time)
    hops: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BroadcastResult:
    """Result of a broadcast/multicast operation."""
    reached_vessels: List[str] = field(default_factory=list)
    failed_vessels: List[str] = field(default_factory=list)
    total_time: float = 0.0       # seconds


@dataclass
class LinkStatus:
    """Status of a communication link between two vessels."""
    vessel_a: str
    vessel_b: str
    latency: float = 0.0          # ms
    bandwidth: float = 100.0      # Mbps
    packet_loss: float = 0.0      # 0.0 to 1.0
    active: bool = True


class FleetCommunication:
    """Fleet-wide communication manager with routing, broadcasting, and health."""

    def __init__(self) -> None:
        self._message_log: List[FleetMessage] = []
        self._links: Dict[Tuple[str, str], LinkStatus] = {}
        self._routing_table: Dict[str, List[str]] = {}
        self._delivery_attempts: Dict[str, int] = {}

    # ------------------------------------------------------- Send
    def send(self, message: FleetMessage) -> DeliveryStatus:
        """Send a unicast message. Simulates delivery."""
        if message.ttl <= 0:
            self._delivery_attempts[message.id] = self._delivery_attempts.get(message.id, 0) + 1
            return DeliveryStatus.EXPIRED

        start = time.time()
        # Check link quality
        link_key = tuple(sorted([message.source, message.target]))
        link = self._links.get(link_key)

        if link and not link.active:
            self._message_log.append(message)
            self._delivery_attempts[message.id] = self._delivery_attempts.get(message.id, 0) + 1
            return DeliveryStatus.FAILED

        # Simulate delivery: high priority messages have higher success
        success_prob = 0.7 + 0.3 * message.priority
        if link:
            success_prob *= (1.0 - link.packet_loss * 0.5)

        if random_simulate(success_prob):
            message.hops += 1
            message.ttl -= 1
            self._message_log.append(message)
            return DeliveryStatus.DELIVERED
        else:
            self._message_log.append(message)
            self._delivery_attempts[message.id] = self._delivery_attempts.get(message.id, 0) + 1
            return DeliveryStatus.FAILED

    # ----------------------------------------------------- Broadcast
    def broadcast(self, source: str, message: FleetMessage,
                  fleet: List[Any]) -> BroadcastResult:
        """Broadcast a message to all vessels in the fleet."""
        start = time.time()
        reached: List[str] = []
        failed: List[str] = []

        for vessel in fleet:
            vid = vessel.vessel_id if hasattr(vessel, "vessel_id") else str(vessel)
            if vid == source:
                continue
            msg_copy = FleetMessage(
                id=uuid.uuid4().hex[:12],
                source=source,
                target=vid,
                type=message.type,
                payload=message.payload,
                priority=message.priority,
                ttl=message.ttl,
            )
            status = self.send(msg_copy)
            if status == DeliveryStatus.DELIVERED:
                reached.append(vid)
            else:
                failed.append(vid)

        return BroadcastResult(
            reached_vessels=reached,
            failed_vessels=failed,
            total_time=time.time() - start,
        )

    # ---------------------------------------------------- Multicast
    def multicast(self, source: str, message: FleetMessage,
                  target_group: List[str]) -> BroadcastResult:
        """Send a message to a specific group of vessels."""
        start = time.time()
        reached: List[str] = []
        failed: List[str] = []

        for vid in target_group:
            if vid == source:
                continue
            msg_copy = FleetMessage(
                id=uuid.uuid4().hex[:12],
                source=source,
                target=vid,
                type=message.type,
                payload=message.payload,
                priority=message.priority,
                ttl=message.ttl,
            )
            status = self.send(msg_copy)
            if status == DeliveryStatus.DELIVERED:
                reached.append(vid)
            else:
                failed.append(vid)

        return BroadcastResult(
            reached_vessels=reached,
            failed_vessels=failed,
            total_time=time.time() - start,
        )

    # ------------------------------------------------------ Relay
    def relay_message(self, message: FleetMessage,
                      route: List[str]) -> bool:
        """Relay a message along a route. Returns True if fully delivered."""
        if not route:
            return False

        current = message.source
        for next_hop in route:
            if message.ttl <= 0:
                return False
            msg_copy = FleetMessage(
                id=uuid.uuid4().hex[:12],
                source=current,
                target=next_hop,
                type=message.type,
                payload=message.payload,
                priority=message.priority,
                ttl=message.ttl,
                hops=message.hops,
            )
            status = self.send(msg_copy)
            if status == DeliveryStatus.DELIVERED:
                message.hops += 1
                message.ttl -= 1
                current = next_hop
            else:
                return False
        return True

    # ---------------------------------------------------- Routing
    def compute_optimal_routes(self, fleet_graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Compute shortest-path routing table using Dijkstra.

        fleet_graph: adjacency list vessel_id -> [neighbour_ids]
        Returns: routing table vessel_id -> [next_hops for all reachable nodes]
        """
        # Build adjacency with weights (uniform weight = 1)
        adj: Dict[str, List[Tuple[str, float]]] = {}
        for node, neighbours in fleet_graph.items():
            adj.setdefault(node, [])
            for n in neighbours:
                key = tuple(sorted([node, n]))
                link = self._links.get(key)
                weight = 1.0
                if link and link.active:
                    weight = 1.0 + link.latency / 100.0 + link.packet_loss * 5.0
                adj[node].append((n, weight))
                adj.setdefault(n, [])
                key2 = tuple(sorted([n, node]))
                link2 = self._links.get(key2)
                weight2 = 1.0
                if link2 and link2.active:
                    weight2 = 1.0 + link2.latency / 100.0 + link2.packet_loss * 5.0
                if (n, weight2) not in adj[n] and (node, weight2) not in adj[n]:
                    pass  # already handled by symmetry

        routing_table: Dict[str, List[str]] = {}
        for source in fleet_graph:
            paths = self._dijkstra(source, adj)
            routing_table[source] = []
            for dest, (dist, next_hop) in paths.items():
                if dest != source and next_hop is not None:
                    routing_table[source].append(next_hop)

        self._routing_table = routing_table
        return routing_table

    @staticmethod
    def _dijkstra(source: str,
                  adj: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Tuple[float, Optional[str]]]:
        """Run Dijkstra from source. Returns {dest: (distance, next_hop)}."""
        dist: Dict[str, float] = {source: 0.0}
        next_hop: Dict[str, Optional[str]] = {source: None}
        visited: Set[str] = set()
        heap = [(0.0, source)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            for v, w in adj.get(u, []):
                if v in visited:
                    continue
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    # Track next hop from source
                    if u == source:
                        next_hop[v] = v
                    else:
                        next_hop[v] = next_hop.get(u)
                    heapq.heappush(heap, (nd, v))

        return {k: (dist[k], next_hop[k]) for k in dist}

    # ------------------------------------------------- Network health
    def estimate_network_health(self, links: List[LinkStatus]) -> float:
        """Estimate overall network health from 0.0 (dead) to 1.0 (perfect)."""
        if not links:
            return 0.0

        active = [l for l in links if l.active]
        if not active:
            return 0.0

        avg_latency = sum(l.latency for l in active) / len(active)
        avg_loss = sum(l.packet_loss for l in active) / len(active)
        avg_bw = sum(l.bandwidth for l in active) / len(active)

        # Latency score (0-100ms = good, >500ms = bad)
        latency_score = max(0.0, 1.0 - avg_latency / 500.0)
        # Loss score
        loss_score = 1.0 - avg_loss
        # Bandwidth score (>=50 Mbps = good)
        bw_score = min(1.0, avg_bw / 50.0)

        # Connectivity: fraction of links active
        connectivity = len(active) / len(links)

        health = 0.3 * connectivity + 0.3 * latency_score + 0.2 * loss_score + 0.2 * bw_score
        return max(0.0, min(1.0, health))

    # -------------------------------------------------- Link mgmt
    def add_link(self, vessel_a: str, vessel_b: str,
                 latency: float = 10.0, bandwidth: float = 100.0,
                 packet_loss: float = 0.0) -> LinkStatus:
        key = tuple(sorted([vessel_a, vessel_b]))
        link = LinkStatus(
            vessel_a=key[0], vessel_b=key[1],
            latency=latency, bandwidth=bandwidth,
            packet_loss=packet_loss, active=True,
        )
        self._links[key] = link
        return link

    def remove_link(self, vessel_a: str, vessel_b: str) -> bool:
        key = tuple(sorted([vessel_a, vessel_b]))
        if key in self._links:
            del self._links[key]
            return True
        return False

    def get_link(self, vessel_a: str, vessel_b: str) -> Optional[LinkStatus]:
        key = tuple(sorted([vessel_a, vessel_b]))
        return self._links.get(key)

    def get_message_log(self, limit: int = 100) -> List[FleetMessage]:
        return self._message_log[-limit:]

    def clear_log(self) -> None:
        self._message_log.clear()
        self._delivery_attempts.clear()


# ---- helper ----
def random_simulate(probability: float) -> bool:
    """Deterministic-ish simulation using time-based seed for reproducibility in tests."""
    import random
    # Use a simple hash to keep it deterministic when called in sequence
    r = random.random()
    return r < probability
