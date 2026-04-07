"""NEXUS Marine Robotics Platform - WiFi Mesh Networking.

Defines mesh networking configuration types for ESP32-based nodes in the
NEXUS fleet. Supports star, tree, and chain topologies with configurable
topology parameters, data-rate tuning, and per-node runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MeshType(str, Enum):
    """Supported mesh network topologies."""

    STAR = "star"
    TREE = "tree"
    CHAIN = "chain"


class NodeRole(str, Enum):
    """Role a node plays within the mesh network."""

    COORDINATOR = "coordinator"
    ROUTER = "router"
    END_DEVICE = "end_device"


# ---------------------------------------------------------------------------
# Mesh network configuration
# ---------------------------------------------------------------------------

@dataclass
class MeshConfig:
    """Parameters that define the mesh network itself.

    Attributes:
        mesh_type: Topology style (star, tree, or chain).
        max_nodes: Maximum number of nodes allowed in the mesh.
        channel: WiFi channel for the mesh (1-13).
        encryption: Whether link-layer encryption is enabled.
        ssid_prefix: Prefix used to build the mesh SSID.
        password: Pre-shared key for the mesh network (empty = open).
    """

    mesh_type: MeshType = MeshType.TREE
    max_nodes: int = 20
    channel: int = 6
    encryption: bool = True
    ssid_prefix: str = "NEXUS-MESH"
    password: str = ""


# ---------------------------------------------------------------------------
# Mesh topology tuning
# ---------------------------------------------------------------------------

@dataclass
class MeshTopology:
    """Runtime topology management parameters.

    Attributes:
        auto_heal: Automatically re-route when a parent node disappears.
        max_hops: Maximum number of hops from coordinator to end device.
        parent_timeout_s: Seconds before a child considers its parent lost.
        retry_count: Number of join / route-repair retries before giving up.
    """

    auto_heal: bool = True
    max_hops: int = 5
    parent_timeout_s: int = 60
    retry_count: int = 3


# ---------------------------------------------------------------------------
# Data-rate / QoS configuration
# ---------------------------------------------------------------------------

@dataclass
class DataRateConfig:
    """Sensor data throughput and command latency tuning.

    Attributes:
        sensor_hz: Target sensor data publishing frequency (Hz).
        cmd_latency_ms: Maximum acceptable command round-trip latency (ms).
        heartbeat_s: Interval between mesh heartbeat beacons (seconds).
        max_payload_bytes: Maximum application payload per frame (bytes).
    """

    sensor_hz: int = 10
    cmd_latency_ms: int = 100
    heartbeat_s: int = 5
    max_payload_bytes: int = 256


# ---------------------------------------------------------------------------
# Per-node runtime info
# ---------------------------------------------------------------------------

@dataclass
class MeshNodeInfo:
    """Runtime state for a single node in the mesh.

    Attributes:
        node_id: Unique identifier for this node.
        role: The node's current role in the mesh.
        parent_id: Node ID of the parent (empty if coordinator).
        rssi_dbm: Signal strength of link to parent (dBm).
        uptime_s: Seconds since the node joined the mesh.
    """

    node_id: str = ""
    role: NodeRole = NodeRole.END_DEVICE
    parent_id: str = ""
    rssi_dbm: int = 0
    uptime_s: int = 0
