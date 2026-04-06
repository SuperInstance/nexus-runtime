"""State mirror: real vessel <-> digital twin synchronization."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict

from .physics import VesselState, Force, VesselPhysics


class SyncStatus(Enum):
    """Synchronization status between real vessel and digital twin."""
    SYNCED = "synced"
    DRIFTING = "drifting"
    DIVERGED = "diverged"
    UNKNOWN = "unknown"


@dataclass
class MirrorConfig:
    """Configuration for state mirroring."""
    sync_interval: float = 0.1         # seconds between syncs
    drift_threshold: float = 0.5       # meters - below this is synced
    divergence_threshold: float = 5.0   # meters - above this is diverged
    smoothing: float = 0.3             # exponential smoothing factor (0..1)


@dataclass
class SyncRecord:
    """Single sync history entry."""
    timestamp: float
    real_state: VesselState
    twin_state: VesselState
    drift: float


class StateMirror:
    """Maintains synchronization between the real vessel and its digital twin."""

    def __init__(self, config: MirrorConfig = None, physics: VesselPhysics = None):
        self.config = config or MirrorConfig()
        self.physics = physics or VesselPhysics()
        self._real_history: List[Tuple[float, VesselState]] = []
        self._twin_history: List[Tuple[float, VesselState]] = []
        self._last_sync_time: float = -1.0
        self._current_twin: Optional[VesselState] = None
        self._current_real: Optional[VesselState] = None
        self._smoothed_state: Optional[VesselState] = None
        self._last_forces: List[Force] = []
        self._sync_records: List[SyncRecord] = []

    def update_twin(self, real_state: VesselState, timestamp: float) -> None:
        """Update twin state with real vessel state (sync point)."""
        if self._current_twin is not None:
            alpha = self.config.smoothing
            self._smoothed_state = self._interpolate(
                self._current_twin, real_state, alpha
            )
            self._current_twin = self._smoothed_state.copy()
        else:
            self._current_twin = real_state.copy()
            self._smoothed_state = real_state.copy()

        self._current_real = real_state.copy()
        self._last_sync_time = timestamp
        self._real_history.append((timestamp, real_state.copy()))
        self._twin_history.append((timestamp, self._current_twin.copy()))

        # Trim history to last 1000 entries
        if len(self._real_history) > 1000:
            self._real_history = self._real_history[-1000:]
            self._twin_history = self._twin_history[-1000:]

    def get_twin_state(self, timestamp: float) -> VesselState:
        """Get twin state at a given timestamp (interpolated)."""
        if self._current_twin is None:
            return VesselState()

        # Find bracketing history entries
        t_prev = None
        s_prev = None
        for t, s in self._twin_history:
            if t <= timestamp:
                t_prev = t
                s_prev = s
            else:
                break

        if t_prev is None:
            return self._twin_history[0][1].copy() if self._twin_history else VesselState()

        if t_prev == timestamp:
            return s_prev.copy()

        # Find next entry
        for t, s in self._twin_history:
            if t > timestamp:
                dt = t - t_prev
                if dt > 0:
                    alpha = (timestamp - t_prev) / dt
                    return self._interpolate(s_prev, s, alpha)
                return s_prev.copy()

        # After last entry: return latest
        return self._twin_history[-1][1].copy()

    def compute_drift(self, real_state: VesselState, twin_state: VesselState) -> float:
        """Compute drift magnitude between real and twin states.
        Uses 3D Euclidean distance."""
        dx = real_state.x - twin_state.x
        dy = real_state.y - twin_state.y
        dz = real_state.z - twin_state.z
        position_drift = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Also consider velocity drift
        dvx = real_state.vx - twin_state.vx
        dvy = real_state.vy - twin_state.vy
        dvz = real_state.vz - twin_state.vz
        velocity_drift = math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)

        # Weighted combination (position matters more)
        return position_drift + 0.1 * velocity_drift

    def get_sync_status(self) -> SyncStatus:
        """Get current synchronization status."""
        if self._current_real is None or self._current_twin is None:
            return SyncStatus.UNKNOWN

        drift = self.compute_drift(self._current_real, self._current_twin)

        if drift < self.config.drift_threshold:
            return SyncStatus.SYNCED
        elif drift < self.config.divergence_threshold:
            return SyncStatus.DRIFTING
        else:
            return SyncStatus.DIVERGED

    def interpolate_state(self, t1: VesselState, t2: VesselState,
                          alpha: float) -> VesselState:
        """Linear interpolation between two states. alpha=0 gives t1, alpha=1 gives t2."""
        return self._interpolate(t1, t2, alpha)

    def _interpolate(self, t1: VesselState, t2: VesselState,
                     alpha: float) -> VesselState:
        """Internal interpolation."""
        alpha = max(0.0, min(1.0, alpha))
        return VesselState(
            x=t1.x + alpha * (t2.x - t1.x),
            y=t1.y + alpha * (t2.y - t1.y),
            z=t1.z + alpha * (t2.z - t1.z),
            vx=t1.vx + alpha * (t2.vx - t1.vx),
            vy=t1.vy + alpha * (t2.vy - t1.vy),
            vz=t1.vz + alpha * (t2.vz - t1.vz),
            roll=t1.roll + alpha * (t2.roll - t1.roll),
            pitch=t1.pitch + alpha * (t2.pitch - t1.pitch),
            yaw=t1.yaw + alpha * (t2.yaw - t1.yaw),
            wx=t1.wx + alpha * (t2.wx - t1.wx),
            wy=t1.wy + alpha * (t2.wy - t1.wy),
            wz=t1.wz + alpha * (t2.wz - t1.wz),
        )

    def predict_state(self, current: VesselState, dt: float,
                      forces: List[Force]) -> VesselState:
        """Predict future state by advancing the physics simulation."""
        return self.physics.update_state(current, forces, dt)

    def sync_history(self, duration: float) -> List[SyncRecord]:
        """Get sync history for the last `duration` seconds."""
        if not self._sync_records:
            # Build from real/twin history
            cutoff = self._sync_records[-1][0] - duration if self._sync_records else 0
        cutoff_time = self._real_history[-1][0] - duration if self._real_history else 0
        records = []
        for i in range(min(len(self._real_history), len(self._twin_history))):
            ts, real_s = self._real_history[i]
            _, twin_s = self._twin_history[i]
            if ts >= cutoff_time:
                drift = self.compute_drift(real_s, twin_s)
                records.append(SyncRecord(
                    timestamp=ts,
                    real_state=real_s,
                    twin_state=twin_s,
                    drift=drift,
                ))
        return records

    def force_resync(self, real_state: VesselState, timestamp: float) -> None:
        """Force immediate resynchronization without smoothing."""
        self._current_twin = real_state.copy()
        self._current_real = real_state.copy()
        self._smoothed_state = real_state.copy()
        self._last_sync_time = timestamp
        self._real_history.append((timestamp, real_state.copy()))
        self._twin_history.append((timestamp, real_state.copy()))

    def record_sync(self, real_state: VesselState, twin_state: VesselState,
                    timestamp: float) -> None:
        """Manually record a sync event."""
        drift = self.compute_drift(real_state, twin_state)
        self._sync_records.append(SyncRecord(
            timestamp=timestamp,
            real_state=real_state,
            twin_state=twin_state,
            drift=drift,
        ))

    def get_last_sync_time(self) -> float:
        """Get timestamp of last synchronization."""
        return self._last_sync_time

    def get_history_length(self) -> int:
        """Get number of history entries."""
        return len(self._real_history)

    def clear_history(self) -> None:
        """Clear all history."""
        self._real_history.clear()
        self._twin_history.clear()
        self._sync_records.clear()

    def get_drift_history(self) -> List[float]:
        """Get list of drift values over time."""
        return [r.drift for r in self._sync_records]
