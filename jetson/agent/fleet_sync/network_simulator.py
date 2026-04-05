"""
NEXUS Fleet Network Simulator

Simulates realistic marine fleet network conditions:
- Configurable latency (0-5000ms for satellite links)
- Packet loss (0-50% for rough seas)
- Partition/reconnection (vessels lose contact, then reconnect)
- Bandwidth limits (low-bandwidth satellite vs high-bandwidth WiFi)
- Message ordering (in-order, out-of-order, duplicated messages)

Designed for testing CRDT convergence under adverse conditions.
"""

import time
import random
import heapq
import copy
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict


class NetworkCondition(Enum):
    """Predefined network condition profiles."""
    PERFECT = "perfect"
    WIFI = "wifi"
    RADIO = "radio"
    CELLULAR = "cellular"
    SATELLITE = "satellite"
    BAD_WEATHER = "bad_weather"
    STORM = "storm"


@dataclass
class NetworkConfig:
    """Configuration for network simulation."""
    condition: NetworkCondition = NetworkCondition.WIFI
    base_latency_ms: float = 10.0
    latency_jitter_ms: float = 5.0
    packet_loss_pct: float = 1.0
    bandwidth_bytes_per_sec: float = 1e6
    duplicate_pct: float = 0.0
    out_of_order_pct: float = 0.0
    max_retries: int = 3

    @classmethod
    def from_condition(cls, condition: NetworkCondition) -> "NetworkConfig":
        profiles = {
            NetworkCondition.PERFECT: cls(
                condition=condition,
                base_latency_ms=0.0, latency_jitter_ms=0.0,
                packet_loss_pct=0.0, bandwidth_bytes_per_sec=1e9,
                duplicate_pct=0.0, out_of_order_pct=0.0,
            ),
            NetworkCondition.WIFI: cls(
                condition=condition,
                base_latency_ms=10.0, latency_jitter_ms=10.0,
                packet_loss_pct=0.1, bandwidth_bytes_per_sec=5e7,
                duplicate_pct=0.0, out_of_order_pct=0.5,
            ),
            NetworkCondition.RADIO: cls(
                condition=condition,
                base_latency_ms=100.0, latency_jitter_ms=100.0,
                packet_loss_pct=2.0, bandwidth_bytes_per_sec=1e4,
                duplicate_pct=0.5, out_of_order_pct=2.0,
            ),
            NetworkCondition.CELLULAR: cls(
                condition=condition,
                base_latency_ms=200.0, latency_jitter_ms=300.0,
                packet_loss_pct=5.0, bandwidth_bytes_per_sec=5e6,
                duplicate_pct=0.1, out_of_order_pct=1.0,
            ),
            NetworkCondition.SATELLITE: cls(
                condition=condition,
                base_latency_ms=1200.0, latency_jitter_ms=600.0,
                packet_loss_pct=3.0, bandwidth_bytes_per_sec=5e5,
                duplicate_pct=0.2, out_of_order_pct=0.5,
            ),
            NetworkCondition.BAD_WEATHER: cls(
                condition=condition,
                base_latency_ms=1000.0, latency_jitter_ms=2000.0,
                packet_loss_pct=20.0, bandwidth_bytes_per_sec=1e4,
                duplicate_pct=1.0, out_of_order_pct=5.0,
            ),
            NetworkCondition.STORM: cls(
                condition=condition,
                base_latency_ms=2000.0, latency_jitter_ms=3000.0,
                packet_loss_pct=40.0, bandwidth_bytes_per_sec=5e3,
                duplicate_pct=2.0, out_of_order_pct=10.0,
            ),
        }
        return profiles[condition]


@dataclass
class NetworkMessage:
    """A message in transit through the network."""
    msg_id: int
    sender_id: str
    receiver_id: str
    payload: Dict[str, Any]
    send_time: float
    deliver_time: float
    delivery_attempts: int = 0
    size_bytes: int = 0
    seq_num: int = 0

    def __lt__(self, other: "NetworkMessage"):
        return self.deliver_time < other.deliver_time


@dataclass
class NetworkStats:
    """Statistics from network simulation."""
    messages_sent: int = 0
    messages_delivered: int = 0
    messages_lost: int = 0
    messages_duplicated: int = 0
    messages_out_of_order: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    bandwidth_throttle_count: int = 0
    bytes_transmitted: int = 0

    @property
    def delivery_rate(self) -> float:
        if self.messages_sent == 0:
            return 1.0
        return self.messages_delivered / self.messages_sent

    @property
    def avg_latency_ms(self) -> float:
        if self.messages_delivered == 0:
            return 0.0
        return self.total_latency_ms / self.messages_delivered

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages_sent": self.messages_sent,
            "messages_delivered": self.messages_delivered,
            "messages_lost": self.messages_lost,
            "messages_duplicated": self.messages_duplicated,
            "messages_out_of_order": self.messages_out_of_order,
            "delivery_rate": round(self.delivery_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float("inf") else 0.0,
            "bytes_transmitted": self.bytes_transmitted,
        }


class PartitionManager:
    """Manages network partitions."""

    def __init__(self):
        self.groups: List[Set[str]] = []

    def set_groups(self, groups: List[Set[str]]):
        self.groups = [set(g) for g in groups]

    def no_partitions(self, vessel_ids: List[str]):
        self.groups = [set(vessel_ids)]

    def split(self, vessel_ids: List[str], group_a: Set[str]):
        group_b = set(vessel_ids) - group_a
        if group_a and group_b:
            self.groups = [group_a, group_b]
        else:
            self.groups = [set(vessel_ids)]

    def can_communicate(self, sender: str, receiver: str) -> bool:
        for group in self.groups:
            if sender in group and receiver in group:
                return True
        return False

    def get_reachable(self, vessel_id: str) -> Set[str]:
        for group in self.groups:
            if vessel_id in group:
                return group - {vessel_id}
        return set()

    def get_disconnected_from(self, vessel_id: str) -> Set[str]:
        reachable = self.get_reachable(vessel_id)
        all_vessels = set()
        for group in self.groups:
            all_vessels |= group
        return all_vessels - reachable - {vessel_id}


@dataclass
class PartitionEvent:
    sim_time: float
    groups: List[Set[str]]
    description: str = ""


class NetworkSimulator:
    """
    Simulates marine fleet network conditions for CRDT testing.

    Usage:
        sim = NetworkSimulator(config=NetworkConfig.from_condition(NetworkCondition.SATELLITE))
        sim.add_vessel("vessel-0")
        sim.add_vessel("vessel-1")
        sim.send("vessel-0", "vessel-1", payload)
        messages = sim.drain()
        print(sim.get_stats())
    """

    def __init__(self, config: NetworkConfig = None, seed: int = 42):
        self.config = config or NetworkConfig()
        self._rng = random.Random(seed)
        self._msg_counter = 0
        self._seq_counters: Dict[str, int] = defaultdict(int)
        self._inflight: List[Tuple[float, int, NetworkMessage]] = []
        self._delivered: List[NetworkMessage] = []
        self._vessels: Set[str] = set()
        self.partition_manager = PartitionManager()
        self._partition_schedule: List[PartitionEvent] = []
        self._applied_partitions: List[PartitionEvent] = []
        self.stats = NetworkStats()
        self._link_configs: Dict[Tuple[str, str], NetworkConfig] = {}

    def add_vessel(self, vessel_id: str):
        self._vessels.add(vessel_id)
        self.partition_manager.no_partitions(list(self._vessels))

    def add_vessels(self, vessel_ids: List[str]):
        for vid in vessel_ids:
            self._vessels.add(vid)
        self.partition_manager.no_partitions(list(self._vessels))

    def set_link_config(self, sender: str, receiver: str, config: NetworkConfig):
        self._link_configs[(sender, receiver)] = config

    def schedule_partition(self, sim_time: float, groups: List[Set[str]],
                           description: str = ""):
        event = PartitionEvent(sim_time=sim_time, groups=groups, description=description)
        self._partition_schedule.append(event)
        self._partition_schedule.sort(key=lambda e: e.sim_time)

    def send(self, sender_id: str, receiver_id: str,
             payload: Dict[str, Any]) -> bool:
        if sender_id not in self._vessels or receiver_id not in self._vessels:
            return False

        if not self.partition_manager.can_communicate(sender_id, receiver_id):
            self.stats.messages_lost += 1
            self.stats.messages_sent += 1
            return False

        self.stats.messages_sent += 1
        self._msg_counter += 1
        self._seq_counters[sender_id] += 1

        payload_size = len(json.dumps(payload, default=str))
        self.stats.bytes_transmitted += payload_size

        effective_bw = self._get_effective_bw(sender_id, receiver_id)
        if payload_size > effective_bw * 0.1:
            self.stats.bandwidth_throttle_count += 1

        latency = self._calculate_latency(sender_id, receiver_id)
        deliver_time = time.time() + latency / 1000.0

        msg = NetworkMessage(
            msg_id=self._msg_counter,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload=copy.deepcopy(payload),
            send_time=time.time(),
            deliver_time=deliver_time,
            size_bytes=payload_size,
            seq_num=self._seq_counters[sender_id],
        )

        loss_pct = self._get_effective_loss(sender_id, receiver_id)
        if self._rng.random() * 100 < loss_pct:
            self.stats.messages_lost += 1
            for attempt in range(self.config.max_retries - 1):
                if self._rng.random() * 100 >= loss_pct:
                    msg.delivery_attempts = attempt + 1
                    msg.deliver_time = time.time() + (latency * (attempt + 2)) / 1000.0
                    heapq.heappush(self._inflight, (msg.deliver_time, msg.msg_id, msg))
                    return True
            return False

        dup_pct = self._get_effective_dup(sender_id, receiver_id)
        if self._rng.random() * 100 < dup_pct:
            self.stats.messages_duplicated += 1
            heapq.heappush(self._inflight, (msg.deliver_time, msg.msg_id, msg))
            dup_msg = copy.deepcopy(msg)
            dup_msg.msg_id = self._msg_counter + 100000
            dup_msg.deliver_time += self._rng.uniform(0.001, 0.05)
            heapq.heappush(self._inflight, (dup_msg.deliver_time, dup_msg.msg_id, dup_msg))
            return True

        ooo_pct = self._get_effective_ooo(sender_id, receiver_id)
        if self._rng.random() * 100 < ooo_pct:
            msg.deliver_time -= self._rng.uniform(0.001, 0.01)

        heapq.heappush(self._inflight, (msg.deliver_time, msg.msg_id, msg))
        return True

    def tick(self, sim_time: float = 0.0) -> List[NetworkMessage]:
        current_time = time.time()
        delivered = []
        last_seq: Dict[str, int] = defaultdict(int)

        remaining = []
        for event in self._partition_schedule:
            if event.sim_time <= sim_time:
                self.partition_manager.set_groups(event.groups)
                self._applied_partitions.append(event)
            else:
                remaining.append(event)
        self._partition_schedule = remaining

        while self._inflight:
            deliver_time, msg_id, msg = self._inflight[0]
            if deliver_time > current_time + sim_time:
                break
            heapq.heappop(self._inflight)

            latency = (msg.deliver_time - msg.send_time) * 1000.0
            self.stats.messages_delivered += 1
            self.stats.total_latency_ms += latency
            self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency)
            self.stats.min_latency_ms = min(self.stats.min_latency_ms, latency)

            if msg.seq_num < last_seq[msg.sender_id]:
                self.stats.messages_out_of_order += 1
            last_seq[msg.sender_id] = max(last_seq[msg.sender_id], msg.seq_num)

            delivered.append(msg)
            self._delivered.append(msg)

        return delivered

    def drain(self) -> List[NetworkMessage]:
        delivered = []
        last_seq: Dict[str, int] = defaultdict(int)

        while self._inflight:
            _, msg_id, msg = heapq.heappop(self._inflight)

            latency = (msg.deliver_time - msg.send_time) * 1000.0
            self.stats.messages_delivered += 1
            self.stats.total_latency_ms += latency
            self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency)
            self.stats.min_latency_ms = min(self.stats.min_latency_ms, latency)

            if msg.seq_num < last_seq[msg.sender_id]:
                self.stats.messages_out_of_order += 1
            last_seq[msg.sender_id] = max(last_seq[msg.sender_id], msg.seq_num)

            delivered.append(msg)
            self._delivered.append(msg)

        return delivered

    def get_stats(self) -> NetworkStats:
        return self.stats

    def reset_stats(self):
        self.stats = NetworkStats()

    def _calculate_latency(self, sender: str, receiver: str) -> float:
        config = self._link_configs.get((sender, receiver), self.config)
        jitter = self._rng.uniform(0, config.latency_jitter_ms)
        return config.base_latency_ms + jitter

    def _get_effective_loss(self, sender: str, receiver: str) -> float:
        config = self._link_configs.get((sender, receiver), self.config)
        return config.packet_loss_pct

    def _get_effective_bw(self, sender: str, receiver: str) -> float:
        config = self._link_configs.get((sender, receiver), self.config)
        return config.bandwidth_bytes_per_sec

    def _get_effective_dup(self, sender: str, receiver: str) -> float:
        config = self._link_configs.get((sender, receiver), self.config)
        return config.duplicate_pct

    def _get_effective_ooo(self, sender: str, receiver: str) -> float:
        config = self._link_configs.get((sender, receiver), self.config)
        return config.out_of_order_pct

    @property
    def vessel_ids(self) -> List[str]:
        return sorted(self._vessels)

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    def get_delivered_for(self, vessel_id: str) -> List[NetworkMessage]:
        return [m for m in self._delivered if m.receiver_id == vessel_id]

    def clear_delivered(self):
        self._delivered.clear()
