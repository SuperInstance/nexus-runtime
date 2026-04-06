"""Cooperative object tracking across multiple vessels.

Manages cooperative tracks that combine observations from multiple vessels,
including track creation, prediction, association, merging, and cleanup.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class CooperativeTrack:
    """A cooperative track maintained across multiple vessels."""
    id: str
    state_history: List[dict]
    predicted_state: Optional[dict] = None
    contributing_vessels: List[str] = field(default_factory=list)
    last_update: float = 0.0
    quality: float = 1.0

    def latest_state(self) -> Optional[dict]:
        """Return the most recent state in history."""
        if self.state_history:
            return self.state_history[-1]
        return None

    def vessel_count(self) -> int:
        """Number of unique contributing vessels."""
        return len(set(self.contributing_vessels))

    def age(self, now: Optional[float] = None) -> float:
        """Seconds since last update."""
        if now is None:
            now = time.time()
        return now - self.last_update


@dataclass
class TrackAssociation:
    """Association between a local and remote track."""
    local_track_id: str
    remote_track_id: str
    association_confidence: float
    vessel_ids: List[str] = field(default_factory=list)


class CooperativeTracker:
    """Tracks objects cooperatively across multiple vessels."""

    def __init__(self, association_threshold: float = 12.0):
        self.association_threshold = association_threshold
        self._tracks: Dict[str, CooperativeTrack] = {}
        self._track_counter: int = 0

    def create_track(self, initial_observation: dict) -> str:
        """Create a new cooperative track from an initial observation.

        Args:
            initial_observation: dict with 'position', 'velocity', 'confidence',
                'timestamp', 'source_vessel', optionally 'object_type', 'object_id'.

        Returns:
            The newly created track_id.
        """
        self._track_counter += 1
        track_id = f"ctrack_{self._track_counter}_{uuid.uuid4().hex[:8]}"
        state = {
            "position": initial_observation.get("position", (0.0, 0.0, 0.0)),
            "velocity": initial_observation.get("velocity", (0.0, 0.0, 0.0)),
            "confidence": initial_observation.get("confidence", 1.0),
            "timestamp": initial_observation.get("timestamp", time.time()),
        }
        vessel = initial_observation.get("source_vessel", "unknown")
        track = CooperativeTrack(
            id=track_id,
            state_history=[state],
            predicted_state=None,
            contributing_vessels=[vessel],
            last_update=initial_observation.get("timestamp", time.time()),
            quality=initial_observation.get("confidence", 1.0),
        )
        self._tracks[track_id] = track
        return track_id

    def update_track(self, track_id: str, observation: dict) -> CooperativeTrack:
        """Update an existing track with a new observation.

        Args:
            track_id: the track to update.
            observation: new observation dict.

        Returns:
            The updated CooperativeTrack.

        Raises:
            KeyError: if track_id is not found.
        """
        if track_id not in self._tracks:
            raise KeyError(f"Track {track_id} not found")

        track = self._tracks[track_id]
        state = {
            "position": observation.get("position", (0.0, 0.0, 0.0)),
            "velocity": observation.get("velocity", (0.0, 0.0, 0.0)),
            "confidence": observation.get("confidence", 1.0),
            "timestamp": observation.get("timestamp", time.time()),
        }
        track.state_history.append(state)
        track.last_update = observation.get("timestamp", time.time())
        track.predicted_state = None  # Clear prediction on update

        vessel = observation.get("source_vessel", "unknown")
        if vessel not in track.contributing_vessels:
            track.contributing_vessels.append(vessel)

        # Update quality based on recency and confidence
        if len(track.state_history) >= 2:
            prev = track.state_history[-2]
            pos_change = self._distance3d(
                prev["position"], state["position"]
            )
            # Excessive jump reduces quality
            speed = state["velocity"]
            speed_mag = math.sqrt(
                speed[0] ** 2 + speed[1] ** 2 + speed[2] ** 2
            )
            dt = max(0.001, state["timestamp"] - prev["timestamp"])
            expected = speed_mag * dt
            if expected > 0:
                consistency = max(0.0, 1.0 - abs(pos_change - expected) / max(expected, 1.0))
            else:
                consistency = 1.0 if pos_change < 1.0 else 0.5
            track.quality = min(1.0, state["confidence"] * consistency)
        else:
            track.quality = state["confidence"]

        return track

    def predict_tracks(self, dt: float) -> Dict[str, dict]:
        """Predict all track states dt seconds into the future.

        Uses constant-velocity (linear) prediction.

        Args:
            dt: time delta in seconds.

        Returns:
            Dict mapping track_id -> predicted state dict.
        """
        predicted = {}
        for tid, track in self._tracks.items():
            latest = track.latest_state()
            if latest is None:
                continue
            pred_pos = (
                latest["position"][0] + latest["velocity"][0] * dt,
                latest["position"][1] + latest["velocity"][1] * dt,
                latest["position"][2] + latest["velocity"][2] * dt,
            )
            pred_state = {
                "track_id": tid,
                "position": pred_pos,
                "velocity": latest["velocity"],
                "confidence": latest["confidence"] * max(0.0, 1.0 - dt / 30.0),
                "timestamp": latest["timestamp"] + dt,
                "method": "linear_prediction",
            }
            track.predicted_state = pred_state
            predicted[tid] = pred_state
        return predicted

    def associate_local_remote(
        self,
        local_tracks: List[CooperativeTrack],
        remote_tracks: List[CooperativeTrack],
    ) -> List[TrackAssociation]:
        """Associate local and remote tracks by predicted position proximity.

        Args:
            local_tracks: list of local CooperativeTrack instances.
            remote_tracks: list of remote CooperativeTrack instances.

        Returns:
            List of TrackAssociation objects for matched pairs.
        """
        associations: List[TrackAssociation] = []
        used_remote: set = set()

        for lt in local_tracks:
            lt_state = lt.latest_state()
            if lt_state is None:
                continue
            lt_pos = lt_state["position"]

            best_rt = None
            best_dist = float("inf")
            for rt in remote_tracks:
                if rt.id in used_remote:
                    continue
                rt_state = rt.latest_state()
                if rt_state is None:
                    continue
                dist = self._distance3d(lt_pos, rt_state["position"])
                if dist < self.association_threshold and dist < best_dist:
                    best_dist = dist
                    best_rt = rt

            if best_rt is not None:
                conf = max(0.0, 1.0 - best_dist / self.association_threshold)
                all_vessels = list(set(
                    lt.contributing_vessels + best_rt.contributing_vessels
                ))
                associations.append(TrackAssociation(
                    local_track_id=lt.id,
                    remote_track_id=best_rt.id,
                    association_confidence=conf,
                    vessel_ids=all_vessels,
                ))
                used_remote.add(best_rt.id)

        return associations

    def merge_tracks(
        self, track_a: str, track_b: str
    ) -> CooperativeTrack:
        """Merge two tracks into one, keeping the longer history.

        Args:
            track_a: first track id.
            track_b: second track id.

        Returns:
            The merged CooperativeTrack with a new id.

        Raises:
            KeyError: if either track_id is not found.
        """
        if track_a not in self._tracks:
            raise KeyError(f"Track {track_a} not found")
        if track_b not in self._tracks:
            raise KeyError(f"Track {track_b} not found")

        ta = self._tracks[track_a]
        tb = self._tracks[track_b]

        # Keep the track with longer history as base
        if len(ta.state_history) >= len(tb.state_history):
            base, other = ta, tb
        else:
            base, other = tb, ta

        merged_vessels = list(set(base.contributing_vessels + other.contributing_vessels))
        merged_quality = (base.quality + other.quality) / 2.0

        self._track_counter += 1
        merged_id = f"ctrack_{self._track_counter}_{uuid.uuid4().hex[:8]}"
        merged = CooperativeTrack(
            id=merged_id,
            state_history=list(base.state_history),
            predicted_state=None,
            contributing_vessels=merged_vessels,
            last_update=max(base.last_update, other.last_update),
            quality=merged_quality,
        )

        # Remove old tracks, add merged
        del self._tracks[track_a]
        del self._tracks[track_b]
        self._tracks[merged_id] = merged

        return merged

    def delete_stale_tracks(self, max_age: float) -> int:
        """Delete tracks that have not been updated within max_age seconds.

        Args:
            max_age: maximum age in seconds before a track is considered stale.

        Returns:
            Number of deleted tracks.
        """
        now = time.time()
        stale_ids = [
            tid for tid, track in self._tracks.items()
            if track.age(now) > max_age
        ]
        for tid in stale_ids:
            del self._tracks[tid]
        return len(stale_ids)

    def get_track(self, track_id: str) -> Optional[CooperativeTrack]:
        """Get a track by ID, or None if not found."""
        return self._tracks.get(track_id)

    def list_tracks(self) -> List[str]:
        """List all active track IDs."""
        return list(self._tracks.keys())

    def _distance3d(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)
