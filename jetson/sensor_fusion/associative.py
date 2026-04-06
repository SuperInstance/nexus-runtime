"""Multi-sensor data association for tracking.

Pure Python — no external dependencies.
Implements nearest-neighbor, global nearest-neighbor (greedy), gating,
track lifecycle management, and cost matrix computation.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Track:
    """A tracked object with state estimate and lifecycle counters."""
    id: int
    state: List[float]
    covariance: List[List[float]] = field(default_factory=lambda: [[1.0]])
    last_update: float = 0.0
    hits: int = 1
    misses: int = 0


@dataclass
class Association:
    """Result of associating a measurement to a track."""
    measurement_id: int
    track_id: int
    distance: float = 0.0
    gated: bool = False


class DataAssociator:
    """Multi-sensor data association engine.

    Supports nearest-neighbor, global nearest-neighbor (greedy Hungarian
    approximation), gating, and track lifecycle management.
    """

    def __init__(self, gate_threshold: float = 9.0) -> None:
        """gate_threshold : Mahalanobis-distance squared gating threshold."""
        self.gate_threshold = gate_threshold
        self._next_track_id = 1

    # -- public API --------------------------------------------------------

    def compute_cost_matrix(
        self,
        tracks: List[Track],
        measurements: List[List[float]],
    ) -> List[List[float]]:
        """Compute Euclidean distance cost matrix between tracks and measurements.

        Returns matrix of shape (len(tracks), len(measurements)).
        """
        n_tracks = len(tracks)
        n_meas = len(measurements)
        cost = [[0.0] * n_meas for _ in range(n_tracks)]
        for i, trk in enumerate(tracks):
            for j, m in enumerate(measurements):
                cost[i][j] = self._euclidean(trk.state, m)
        return cost

    def nearest_neighbor(
        self,
        predicted_tracks: List[Track],
        measurements: List[List[float]],
    ) -> List[Association]:
        """Associate each measurement to its nearest track.

        Each measurement is assigned to at most one track (first-come basis).
        """
        associations: List[Association] = []
        assigned_tracks: set = set()

        for j, m in enumerate(measurements):
            best_dist = float('inf')
            best_i = -1
            for i, trk in enumerate(predicted_tracks):
                if i in assigned_tracks:
                    continue
                d = self._euclidean(trk.state, m)
                if d < best_dist:
                    best_dist = d
                    best_i = i
            if best_i >= 0:
                assigned_tracks.add(best_i)
                gated = best_dist > math.sqrt(self.gate_threshold)
                associations.append(Association(
                    measurement_id=j,
                    track_id=predicted_tracks[best_i].id,
                    distance=best_dist,
                    gated=gated,
                ))
        return associations

    def global_nearest_neighbor(
        self,
        tracks: List[Track],
        measurements: List[List[float]],
        cost_matrix: Optional[List[List[float]]] = None,
    ) -> List[Association]:
        """Global nearest-neighbor association using greedy assignment.

        Iteratively picks the smallest cost in the matrix, assigns that
        (track, measurement) pair, and removes the row and column.
        """
        if cost_matrix is None:
            cost_matrix = self.compute_cost_matrix(tracks, measurements)

        n_t = len(tracks)
        n_m = len(measurements)
        if n_t == 0 or n_m == 0:
            return []

        # Build list of (cost, track_idx, meas_idx) sorted ascending
        candidates = []
        for i in range(n_t):
            for j in range(n_m):
                candidates.append((cost_matrix[i][j], i, j))
        candidates.sort(key=lambda x: x[0])

        assigned_tracks: set = set()
        assigned_meas: set = set()
        associations: List[Association] = []

        for cost, ti, mj in candidates:
            if ti in assigned_tracks or mj in assigned_meas:
                continue
            assigned_tracks.add(ti)
            assigned_meas.add(mj)
            gated = cost > math.sqrt(self.gate_threshold)
            associations.append(Association(
                measurement_id=mj,
                track_id=tracks[ti].id,
                distance=cost,
                gated=gated,
            ))

        return associations

    def gated_association(
        self,
        tracks: List[Track],
        measurements: List[List[float]],
        gate_threshold: Optional[float] = None,
    ) -> List[Association]:
        """Filter associations based on gating threshold.

        Returns only associations whose distance is within the gate.
        """
        threshold = gate_threshold if gate_threshold is not None else math.sqrt(self.gate_threshold)
        cost_matrix = self.compute_cost_matrix(tracks, measurements)
        filtered: List[Association] = []

        for i, trk in enumerate(tracks):
            for j, m in enumerate(measurements):
                d = cost_matrix[i][j]
                if d <= threshold:
                    filtered.append(Association(
                        measurement_id=j,
                        track_id=trk.id,
                        distance=d,
                        gated=False,
                    ))
        return filtered

    def create_new_tracks(
        self,
        unassociated_measurements: List[List[float]],
    ) -> List[Track]:
        """Create new tracks from unassociated measurements."""
        new_tracks = []
        for m in unassociated_measurements:
            tid = self._next_track_id
            self._next_track_id += 1
            dim = len(m)
            cov = [[1.0] * dim for _ in range(dim)]
            for k in range(dim):
                cov[k][k] = 10.0  # High initial uncertainty
            new_tracks.append(Track(
                id=tid,
                state=m[:],
                covariance=cov,
                last_update=0.0,
                hits=1,
                misses=0,
            ))
        return new_tracks

    def delete_lost_tracks(
        self,
        tracks: List[Track],
        max_misses: int = 3,
    ) -> List[Track]:
        """Remove tracks that have exceeded max_misses consecutive misses."""
        return [t for t in tracks if t.misses <= max_misses]

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _euclidean(a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
