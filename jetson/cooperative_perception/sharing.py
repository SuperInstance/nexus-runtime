"""Perception data sharing protocol for cooperative maritime perception.

Handles message creation, serialization, compression, priority computation,
and relevance filtering for inter-vessel perception sharing.
"""

from __future__ import annotations

import json
import math
import time
import zlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Any


@dataclass
class PerceivedObject:
    """A single object detected by a vessel's perception system."""
    id: str
    type: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    size: Tuple[float, float, float]
    confidence: float
    source_vessel: str
    timestamp: float

    def distance_to(self, other_position: Tuple[float, float, float]) -> float:
        """Euclidean distance to another 3D position."""
        dx = self.position[0] - other_position[0]
        dy = self.position[1] - other_position[1]
        dz = self.position[2] - other_position[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def speed(self) -> float:
        """Compute speed magnitude from velocity vector."""
        return math.sqrt(
            self.velocity[0] ** 2 + self.velocity[1] ** 2 + self.velocity[2] ** 2
        )


@dataclass
class PerceptionMessage:
    """A perception sharing message exchanged between vessels."""
    sender_id: str
    timestamp: float
    objects: List[PerceivedObject]
    sensor_type: str
    position: Tuple[float, float, float]
    confidence: float
    fields_of_view: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def age(self, now: Optional[float] = None) -> float:
        """Seconds elapsed since this message was created."""
        if now is None:
            now = time.time()
        return now - self.timestamp

    def object_count(self) -> int:
        """Number of perceived objects in this message."""
        return len(self.objects)


class PerceptionSharer:
    """Creates, serializes, compresses and filters perception sharing messages."""

    def __init__(self, vessel_id: str, sensor_type: str = "lidar"):
        self.vessel_id = vessel_id
        self.sensor_type = sensor_type

    def create_message(
        self,
        vessel_state: dict,
        detected_objects: List[PerceivedObject],
    ) -> PerceptionMessage:
        """Create a perception message from vessel state and detections.

        Args:
            vessel_state: dict with 'position' (x,y,z) and optionally 'confidence'.
            detected_objects: list of PerceivedObject instances.

        Returns:
            A populated PerceptionMessage.
        """
        position = tuple(vessel_state.get("position", (0.0, 0.0, 0.0)))
        confidence = vessel_state.get("confidence", 1.0)
        fovs = vessel_state.get("fields_of_view", [])
        return PerceptionMessage(
            sender_id=self.vessel_id,
            timestamp=time.time(),
            objects=detected_objects,
            sensor_type=self.sensor_type,
            position=position,
            confidence=confidence,
            fields_of_view=fovs,
        )

    def serialize_message(self, msg: PerceptionMessage) -> bytes:
        """Serialize a PerceptionMessage to bytes (JSON + utf-8)."""
        payload = {
            "sender_id": msg.sender_id,
            "timestamp": msg.timestamp,
            "sensor_type": msg.sensor_type,
            "position": list(msg.position),
            "confidence": msg.confidence,
            "fields_of_view": [list(f) for f in msg.fields_of_view],
            "objects": [
                {
                    "id": o.id,
                    "type": o.type,
                    "position": list(o.position),
                    "velocity": list(o.velocity),
                    "size": list(o.size),
                    "confidence": o.confidence,
                    "source_vessel": o.source_vessel,
                    "timestamp": o.timestamp,
                }
                for o in msg.objects
            ],
        }
        return json.dumps(payload).encode("utf-8")

    def deserialize_message(self, data: bytes) -> PerceptionMessage:
        """Deserialize bytes back into a PerceptionMessage."""
        payload = json.loads(data.decode("utf-8"))
        objects = [
            PerceivedObject(
                id=o["id"],
                type=o["type"],
                position=tuple(o["position"]),
                velocity=tuple(o["velocity"]),
                size=tuple(o["size"]),
                confidence=o["confidence"],
                source_vessel=o["source_vessel"],
                timestamp=o["timestamp"],
            )
            for o in payload["objects"]
        ]
        fovs = [tuple(f) for f in payload.get("fields_of_view", [])]
        return PerceptionMessage(
            sender_id=payload["sender_id"],
            timestamp=payload["timestamp"],
            objects=objects,
            sensor_type=payload["sensor_type"],
            position=tuple(payload["position"]),
            confidence=payload["confidence"],
            fields_of_view=fovs,
        )

    def compress_for_bandwidth(self, message: PerceptionMessage, max_bytes: int) -> bytes:
        """Compress a perception message to fit within max_bytes.

        Uses zlib compression. If the result still exceeds max_bytes,
        progressively drops the lowest-confidence objects until it fits.

        Args:
            message: the PerceptionMessage to compress.
            max_bytes: maximum allowed byte size.

        Returns:
            Compressed bytes payload.
        """
        data = self.serialize_message(message)
        compressed = zlib.compress(data, level=9)

        if len(compressed) <= max_bytes:
            return compressed

        # Progressive object pruning
        msg_copy = PerceptionMessage(
            sender_id=message.sender_id,
            timestamp=message.timestamp,
            objects=sorted(message.objects, key=lambda o: o.confidence, reverse=True),
            sensor_type=message.sensor_type,
            position=message.position,
            confidence=message.confidence,
            fields_of_view=message.fields_of_view,
        )

        for i in range(len(msg_copy.objects), 0, -1):
            msg_copy.objects = msg_copy.objects[:i]
            data = self.serialize_message(msg_copy)
            compressed = zlib.compress(data, level=9)
            if len(compressed) <= max_bytes:
                return compressed

        # Even empty message; return header only
        minimal = PerceptionMessage(
            sender_id=message.sender_id,
            timestamp=message.timestamp,
            objects=[],
            sensor_type=message.sensor_type,
            position=message.position,
            confidence=message.confidence,
            fields_of_view=[],
        )
        data = self.serialize_message(minimal)
        return zlib.compress(data, level=9)

    def decompress_message(self, data: bytes) -> PerceptionMessage:
        """Decompress a compressed payload back to a PerceptionMessage."""
        decompressed = zlib.decompress(data)
        return self.deserialize_message(decompressed)

    def compute_message_priority(self, message: PerceptionMessage) -> float:
        """Compute sharing priority for a message (0-1, higher = more urgent).

        Factors: object count, average confidence, message freshness, and
        presence of high-speed objects.
        """
        now = time.time()
        age = now - message.timestamp
        freshness = max(0.0, 1.0 - age / 60.0)  # 0-1 over 60s window

        if not message.objects:
            return freshness * 0.1

        avg_conf = sum(o.confidence for o in message.objects) / len(message.objects)
        max_speed = max(o.speed() for o in message.objects)
        speed_factor = min(1.0, max_speed / 20.0)  # 0-1, normalized to 20 m/s
        count_factor = min(1.0, len(message.objects) / 20.0)

        priority = 0.3 * freshness + 0.25 * avg_conf + 0.25 * speed_factor + 0.2 * count_factor
        return max(0.0, min(1.0, priority))

    def filter_by_relevance(
        self,
        message: PerceptionMessage,
        receiver_position: Tuple[float, float, float],
        max_range: float,
    ) -> PerceptionMessage:
        """Filter objects to those within max_range of receiver_position.

        Also drops the message's own fields_of_view to save bandwidth.
        """
        relevant = [
            o for o in message.objects
            if o.distance_to(receiver_position) <= max_range
        ]
        return PerceptionMessage(
            sender_id=message.sender_id,
            timestamp=message.timestamp,
            objects=relevant,
            sensor_type=message.sensor_type,
            position=message.position,
            confidence=message.confidence,
            fields_of_view=[],  # stripped for relevance filter
        )
