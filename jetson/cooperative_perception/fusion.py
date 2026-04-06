"""Multi-vessel perception fusion.

Fuses observations from multiple vessels into a unified perception picture,
handling observation association, conflict resolution, and track history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class FusedObject:
    """An object resulting from fusing multiple observations."""
    id: str
    type: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    confidence: float
    sources: List[str]
    fusion_method: str = "weighted_average"


@dataclass
class FusionResult:
    """Result of a fusion operation."""
    fused_objects: List[FusedObject]
    conflicts: List[Dict[str, Any]]
    new_objects: List[FusedObject]
    lost_objects: List[str]


class PerceptionFusion:
    """Fuses perception observations from multiple vessels."""

    def __init__(self, association_threshold: float = 15.0, conflict_threshold: float = 5.0):
        self.association_threshold = association_threshold
        self.conflict_threshold = conflict_threshold
        self._known_objects: Dict[str, FusedObject] = {}

    def fuse_observations(
        self, observations_from_vessels: Dict[str, List[dict]]
    ) -> FusionResult:
        """Fuse observations from multiple vessels.

        Args:
            observations_from_vessels: mapping of vessel_id -> list of observation dicts.
                Each dict has 'id', 'type', 'position' (x,y,z), 'velocity' (x,y,z),
                'confidence', 'timestamp'.

        Returns:
            FusionResult with fused objects, conflicts, new, and lost objects.
        """
        all_objects: List[Tuple[str, dict]] = []
        for vessel_id, observations in observations_from_vessels.items():
            for obs in observations:
                all_objects.append((vessel_id, obs))

        # Group by object id across vessels
        groups: Dict[str, List[Tuple[str, dict]]] = {}
        for vessel_id, obs in all_objects:
            oid = obs["id"]
            groups.setdefault(oid, []).append((vessel_id, obs))

        fused_objects: List[FusedObject] = []
        conflicts: List[Dict[str, Any]] = []
        new_object_ids = set()

        for oid, group in groups.items():
            vessels_in_group = [v for v, _ in group]
            if len(group) == 1:
                v, obs = group[0]
                positions = [obs["position"]]
                confidences = [obs["confidence"]]
                fused_pos = self.compute_fused_position(positions, confidences)
                fused_vel = self.compute_fused_velocity(
                    [obs["velocity"]], confidences
                )
                fused_conf = obs["confidence"]
                fo = FusedObject(
                    id=oid,
                    type=obs["type"],
                    position=fused_pos,
                    velocity=fused_vel,
                    confidence=fused_conf,
                    sources=vessels_in_group,
                    fusion_method="single_source",
                )
            else:
                positions = [obs["position"] for _, obs in group]
                velocities = [obs["velocity"] for _, obs in group]
                confidences = [obs["confidence"] for _, obs in group]
                types = [obs["type"] for _, obs in group]

                # Check type consistency
                if len(set(types)) > 1:
                    conflicts.append({
                        "object_id": oid,
                        "vessels": vessels_in_group,
                        "types": list(set(types)),
                        "reason": "type_mismatch",
                    })

                # Check position consistency
                fused_pos = self.compute_fused_position(positions, confidences)
                pos_spread = self._position_spread(positions)
                if pos_spread > self.conflict_threshold:
                    conflicts.append({
                        "object_id": oid,
                        "vessels": vessels_in_group,
                        "spread": pos_spread,
                        "reason": "position_conflict",
                    })

                fused_vel = self.compute_fused_velocity(velocities, confidences)
                avg_conf = sum(confidences) / len(confidences)
                # Boost confidence for multi-vessel corroboration
                fused_conf = min(1.0, avg_conf * (1.0 + 0.1 * (len(group) - 1)))

                fo = FusedObject(
                    id=oid,
                    type=types[0],
                    position=fused_pos,
                    velocity=fused_vel,
                    confidence=fused_conf,
                    sources=vessels_in_group,
                    fusion_method="weighted_average",
                )

            fused_objects.append(fo)
            if oid not in self._known_objects:
                new_object_ids.add(oid)

        # Detect lost objects
        current_ids = set(groups.keys())
        lost_ids = list(set(self._known_objects.keys()) - current_ids)

        # Update known objects
        for fo in fused_objects:
            self._known_objects[fo.id] = fo

        new_objects = [fo for fo in fused_objects if fo.id in new_object_ids]

        return FusionResult(
            fused_objects=fused_objects,
            conflicts=conflicts,
            new_objects=new_objects,
            lost_objects=lost_ids,
        )

    def associate_observations(
        self, obs_a: List[dict], obs_b: List[dict]
    ) -> List[Tuple[int, int]]:
        """Associate observations from two lists by proximity.

        Args:
            obs_a: first list of observation dicts with 'id', 'position'.
            obs_b: second list of observation dicts with 'id', 'position'.

        Returns:
            List of (index_a, index_b) pairs for matched observations.
        """
        matches: List[Tuple[int, int]] = []
        used_b: set = set()

        for i, a in enumerate(obs_a):
            best_j = -1
            best_dist = float("inf")
            for j, b in enumerate(obs_b):
                if j in used_b:
                    continue
                dist = self._distance3d(a["position"], b["position"])
                if dist < self.association_threshold and dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j >= 0:
                matches.append((i, best_j))
                used_b.add(best_j)

        return matches

    def resolve_conflicts(
        self, conflicting_observations: List[dict]
    ) -> List[dict]:
        """Resolve conflicting observations by majority vote on type and
        confidence-weighted position/velocity.

        Args:
            conflicting_observations: list of dicts with 'position', 'velocity',
                'confidence', 'type', 'source_vessel'.

        Returns:
            A single-element list with the resolved observation.
        """
        if not conflicting_observations:
            return []

        # Weighted type vote
        type_votes: Dict[str, float] = {}
        for obs in conflicting_observations:
            t = obs["type"]
            type_votes[t] = type_votes.get(t, 0) + obs["confidence"]

        resolved_type = max(type_votes, key=type_votes.get)

        # Confidence-weighted position
        positions = [obs["position"] for obs in conflicting_observations]
        velocities = [obs["velocity"] for obs in conflicting_observations]
        confidences = [obs["confidence"] for obs in conflicting_observations]
        resolved_pos = self.compute_fused_position(positions, confidences)
        resolved_vel = self.compute_fused_velocity(velocities, confidences)
        resolved_conf = min(1.0, sum(confidences) / len(confidences) * 1.1)

        return [{
            "id": conflicting_observations[0].get("id", "resolved"),
            "type": resolved_type,
            "position": resolved_pos,
            "velocity": resolved_vel,
            "confidence": resolved_conf,
            "source_vessel": "fusion",
        }]

    def compute_fused_position(
        self,
        positions: List[Tuple[float, float, float]],
        confidences: List[float],
    ) -> Tuple[float, float, float]:
        """Compute confidence-weighted average position.

        Args:
            positions: list of (x, y, z) positions.
            confidences: parallel list of confidence values.

        Returns:
            Weighted average (x, y, z).
        """
        if not positions:
            return (0.0, 0.0, 0.0)
        total_w = sum(confidences)
        if total_w == 0:
            # Uniform average
            n = len(positions)
            return (
                sum(p[0] for p in positions) / n,
                sum(p[1] for p in positions) / n,
                sum(p[2] for p in positions) / n,
            )
        x = sum(p[0] * w for p, w in zip(positions, confidences)) / total_w
        y = sum(p[1] * w for p, w in zip(positions, confidences)) / total_w
        z = sum(p[2] * w for p, w in zip(positions, confidences)) / total_w
        return (x, y, z)

    def compute_fused_velocity(
        self,
        velocities: List[Tuple[float, float, float]],
        confidences: List[float],
    ) -> Tuple[float, float, float]:
        """Compute confidence-weighted average velocity.

        Args:
            velocities: list of (vx, vy, vz) velocity vectors.
            confidences: parallel list of confidence values.

        Returns:
            Weighted average (vx, vy, vz).
        """
        if not velocities:
            return (0.0, 0.0, 0.0)
        total_w = sum(confidences)
        if total_w == 0:
            n = len(velocities)
            return (
                sum(v[0] for v in velocities) / n,
                sum(v[1] for v in velocities) / n,
                sum(v[2] for v in velocities) / n,
            )
        vx = sum(v[0] * w for v, w in zip(velocities, confidences)) / total_w
        vy = sum(v[1] * w for v, w in zip(velocities, confidences)) / total_w
        vz = sum(v[2] * w for v, w in zip(velocities, confidences)) / total_w
        return (vx, vy, vz)

    def track_object_history(
        self, object_id: str, observations: List[dict]
    ) -> List[dict]:
        """Build a track history from a series of observations for an object.

        Args:
            object_id: the object identifier.
            observations: chronological list of observation dicts.

        Returns:
            List of track state dicts with 'object_id', 'timestamp', 'position',
            'velocity', 'confidence'.
        """
        track = []
        for obs in observations:
            track.append({
                "object_id": object_id,
                "timestamp": obs.get("timestamp", 0.0),
                "position": obs.get("position", (0.0, 0.0, 0.0)),
                "velocity": obs.get("velocity", (0.0, 0.0, 0.0)),
                "confidence": obs.get("confidence", 0.0),
            })
        return track

    def _distance3d(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _position_spread(
        self, positions: List[Tuple[float, float, float]]
    ) -> float:
        """Max pairwise distance among positions."""
        if len(positions) < 2:
            return 0.0
        max_dist = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = self._distance3d(positions[i], positions[j])
                if d > max_dist:
                    max_dist = d
        return max_dist
