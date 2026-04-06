"""Communication simulation for NEXUS marine robotics.

Models acoustic/radio links with bandwidth constraints, latency,
packet loss, interference, and network topology analysis.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .world import Vector3


@dataclass
class CommLink:
    """A communication link between two nodes."""

    source: str
    target: str
    bandwidth: float = 1000.0  # bytes/sec
    latency: float = 0.1  # seconds
    reliability: float = 0.99  # 0.0-1.0
    max_range: float = 1000.0  # meters


@dataclass
class Message:
    """A message to be transmitted."""

    source: str
    target: str
    payload: str = ""
    timestamp: float = 0.0
    size: float = 100.0  # bytes


class CommSimulator:
    """Simulates marine communication with realistic impairments."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._links: Dict[Tuple[str, str], CommLink] = {}
        self._messages_log: List[dict] = []
        self._time: float = 0.0

    def add_link(self, link: CommLink) -> None:
        key = (link.source, link.target)
        self._links[key] = link
        # Also add reverse link if not present
        rev_key = (link.target, link.source)
        if rev_key not in self._links:
            self._links[rev_key] = CommLink(
                source=link.target,
                target=link.source,
                bandwidth=link.bandwidth,
                latency=link.latency,
                reliability=link.reliability,
                max_range=link.max_range,
            )

    def remove_link(self, source: str, target: str) -> bool:
        key = (source, target)
        if key in self._links:
            del self._links[key]
            return True
        return False

    def get_link(self, source: str, target: str) -> Optional[CommLink]:
        return self._links.get((source, target))

    def send_message(self, msg: Message) -> Tuple[bool, float]:
        """Attempt to send a message. Returns (delivered, actual_latency)."""
        link = self._links.get((msg.source, msg.target))
        if link is None:
            return False, float("inf")

        # Check range (if positions are known)
        if not self.simulate_packet_loss(msg, link.reliability):
            return False, 0.0

        # Compute transmission time based on bandwidth
        transmission_time = msg.size / link.bandwidth if link.bandwidth > 0 else float("inf")
        # Add jitter to latency
        jitter = self._rng.gauss(0.0, link.latency * 0.1)
        actual_latency = link.latency + abs(jitter) + transmission_time

        self._messages_log.append({
            "source": msg.source,
            "target": msg.target,
            "timestamp": self._time,
            "size": msg.size,
            "delivered": True,
            "latency": actual_latency,
        })

        return True, actual_latency

    def simulate_packet_loss(self, msg: Message, reliability: float) -> bool:
        """Returns True if packet survives (not lost)."""
        return self._rng.random() < reliability

    def simulate_interference(
        self, msgs: List[Message], noise_level: float = 0.1
    ) -> List[dict]:
        """Simulate interference effects on a batch of messages.

        Returns list of dicts with delivery status and degraded quality.
        """
        results = []
        for msg in msgs:
            link = self._links.get((msg.source, msg.target))
            effective_reliability = (link.reliability if link else 0.9) * (1.0 - noise_level)
            effective_reliability = max(0.0, min(1.0, effective_reliability))
            delivered = self._rng.random() < effective_reliability
            # Signal quality decreases with noise
            signal_quality = max(0.0, 1.0 - noise_level - self._rng.random() * noise_level)
            results.append({
                "source": msg.source,
                "target": msg.target,
                "delivered": delivered,
                "signal_quality": signal_quality,
                "noise_degradation": noise_level,
                "message_size": msg.size,
            })
        return results

    def compute_network_graph(self) -> Dict[str, List[str]]:
        """Compute adjacency graph from all links."""
        graph: Dict[str, List[str]] = {}
        for (src, tgt) in self._links.keys():
            if src not in graph:
                graph[src] = []
            if tgt not in graph[src]:
                graph[src].append(tgt)
            if tgt not in graph:
                graph[tgt] = []
            if src not in graph[tgt]:
                graph[tgt].append(src)
        return graph

    def compute_network_stats(self) -> dict:
        """Compute statistics about the communication network."""
        graph = self.compute_network_graph()
        nodes = set(graph.keys())
        link_count = len(self._links) // 2  # Account for bidirectional
        avg_bandwidth = 0.0
        avg_latency = 0.0
        avg_reliability = 0.0
        if self._links:
            seen = set()
            total_bw = 0.0
            total_lat = 0.0
            total_rel = 0.0
            count = 0
            for link in self._links.values():
                canonical = tuple(sorted((link.source, link.target)))
                if canonical not in seen:
                    seen.add(canonical)
                    total_bw += link.bandwidth
                    total_lat += link.latency
                    total_rel += link.reliability
                    count += 1
            if count > 0:
                avg_bandwidth = total_bw / count
                avg_latency = total_lat / count
                avg_reliability = total_rel / count

        return {
            "num_nodes": len(nodes),
            "num_links": link_count,
            "avg_bandwidth": avg_bandwidth,
            "avg_latency": avg_latency,
            "avg_reliability": avg_reliability,
            "nodes": list(nodes),
        }

    def broadcast_message(self, source: str, payload: str, timestamp: float = 0.0) -> List[Tuple[str, bool, float]]:
        """Broadcast a message to all connected neighbors.

        Returns list of (target, delivered, latency).
        """
        graph = self.compute_network_graph()
        neighbors = graph.get(source, [])
        results = []
        for neighbor in neighbors:
            msg = Message(
                source=source,
                target=neighbor,
                payload=payload,
                timestamp=timestamp,
            )
            delivered, latency = self.send_message(msg)
            results.append((neighbor, delivered, latency))
        return results

    def update_time(self, dt: float) -> None:
        self._time += dt

    @property
    def time(self) -> float:
        return self._time

    @property
    def link_count(self) -> int:
        return len(self._links)

    @property
    def message_log(self) -> List[dict]:
        return list(self._messages_log)
