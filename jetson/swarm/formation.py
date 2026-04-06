"""
Formation Control Module
========================
Manages vessel formation patterns for marine swarm operations.
Supports LINE, WEDGE, CIRCLE, GRID, and V_SHAPE formations with
dynamic reconfiguration and missing vessel handling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class FormationType(Enum):
    """Supported formation types for marine swarm operations."""
    LINE = auto()
    WEDGE = auto()
    CIRCLE = auto()
    GRID = auto()
    V_SHAPE = auto()


@dataclass(frozen=True)
class VesselState:
    """Immutable vessel state snapshot."""
    vessel_id: str
    x: float
    y: float
    heading: float = 0.0
    speed: float = 0.0
    active: bool = True


@dataclass
class FormationPosition:
    """Target position for a vessel within a formation."""
    vessel_id: str
    target_x: float
    target_y: float
    slot_index: int
    formation_type: FormationType = FormationType.LINE


@dataclass
class FormationError:
    """Error metrics for a vessel's deviation from its formation slot."""
    vessel_id: str
    position_error: float
    heading_error: float
    is_acceptable: bool


class FormationController:
    """
    Computes and manages formation positions for a swarm of marine vessels.

    Handles formation generation, vessel-to-slot assignment, error computation,
    and dynamic reconfiguration when vessels join/leave.
    """

    def __init__(
        self,
        formation_type: FormationType = FormationType.LINE,
        spacing: float = 10.0,
        heading: float = 0.0,
        center: Tuple[float, float] = (0.0, 0.0),
        error_threshold: float = 2.0,
    ):
        self.formation_type = formation_type
        self.spacing = spacing
        self.heading = heading
        self.center = center
        self.error_threshold = error_threshold
        self._current_assignments: Dict[str, FormationPosition] = {}
        self._formation_history: List[FormationType] = []
        self._keepalive_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_formation_positions(
        self, count: int, center: Optional[Tuple[float, float]] = None
    ) -> List[Tuple[float, float]]:
        """
        Compute *count* ideal formation positions around *center* (or self.center).

        Returns a list of (x, y) tuples in formation order.
        """
        cx, cy = center if center is not None else self.center
        if count <= 0:
            return []

        dispatch = {
            FormationType.LINE: self._line_positions,
            FormationType.WEDGE: self._wedge_positions,
            FormationType.CIRCLE: self._circle_positions,
            FormationType.GRID: self._grid_positions,
            FormationType.V_SHAPE: self._v_shape_positions,
        }
        return dispatch[self.formation_type](count, cx, cy)

    def assign_positions(
        self,
        vessels: List[VesselState],
        positions: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, FormationPosition]:
        """
        Assign formation slots to active vessels using greedy nearest-slot.

        Missing (inactive) vessels are skipped.  If *positions* is None the
        method recomputes them from the current formation type and vessel count.
        """
        active = [v for v in vessels if v.active]
        if not active:
            self._current_assignments = {}
            return {}

        if positions is None:
            positions = self.compute_formation_positions(len(active))

        # Greedy nearest-slot assignment (O(n*m), fine for swarm sizes)
        assigned: Dict[str, FormationPosition] = {}
        used_slots: set = set()

        # Sort vessels by id for deterministic tie-breaking
        for vessel in sorted(active, key=lambda v: v.vessel_id):
            best_dist = float("inf")
            best_idx = -1
            for idx, (px, py) in enumerate(positions):
                if idx in used_slots:
                    continue
                d = math.hypot(vessel.x - px, vessel.y - py)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_idx >= 0:
                used_slots.add(best_idx)
                px, py = positions[best_idx]
                assigned[vessel.vessel_id] = FormationPosition(
                    vessel_id=vessel.vessel_id,
                    target_x=px,
                    target_y=py,
                    slot_index=best_idx,
                    formation_type=self.formation_type,
                )

        self._current_assignments = assigned
        return assigned

    def compute_formation_errors(
        self, vessels: List[VesselState]
    ) -> List[FormationError]:
        """
        Compute deviation of each active vessel from its assigned slot.

        Vessels without an assignment receive an infinite position error.
        """
        errors: List[FormationError] = []
        for v in vessels:
            if not v.active:
                continue
            assignment = self._current_assignments.get(v.vessel_id)
            if assignment is None:
                errors.append(FormationError(
                    vessel_id=v.vessel_id,
                    position_error=float("inf"),
                    heading_error=float("inf"),
                    is_acceptable=False,
                ))
                continue
            pos_err = math.hypot(v.x - assignment.target_x, v.y - assignment.target_y)
            heading_err = abs(self._angle_diff(v.heading, self.heading))
            errors.append(FormationError(
                vessel_id=v.vessel_id,
                position_error=pos_err,
                heading_error=heading_err,
                is_acceptable=pos_err <= self.error_threshold,
            ))
        return errors

    def formation_keepalive(self, vessels: List[VesselState]) -> bool:
        """
        Check whether the formation is still viable given the current vessels.
        Returns True if the formation should be maintained.
        Increments an internal keepalive counter.
        """
        self._keepalive_counter += 1
        active = [v for v in vessels if v.active]
        if len(active) < 2:
            return False
        # Check that at least half the assigned vessels are within threshold
        errors = self.compute_formation_errors(active)
        if not errors:
            return True
        acceptable = sum(1 for e in errors if e.is_acceptable)
        return acceptable >= len(errors) / 2

    def set_formation_type(self, formation_type: FormationType) -> None:
        """Switch to a different formation type."""
        self._formation_history.append(self.formation_type)
        self.formation_type = formation_type

    def get_formation_history(self) -> List[FormationType]:
        """Return list of previously used formation types."""
        return list(self._formation_history)

    def reconfigure(self, vessels: List[VesselState]) -> Dict[str, FormationPosition]:
        """
        Reconfigure formation for the current set of active vessels.
        Automatically picks a suitable formation if too few vessels.
        """
        active = [v for v in vessels if v.active]
        n = len(active)
        if n <= 1:
            self._current_assignments = {}
            return {}
        if n == 2:
            if self.formation_type not in (FormationType.LINE, FormationType.WEDGE):
                self.set_formation_type(FormationType.LINE)
        return self.assign_positions(active)

    # ------------------------------------------------------------------
    # Internal formation generators
    # ------------------------------------------------------------------

    def _rotate(self, x: float, y: float, angle: float) -> Tuple[float, float]:
        """Rotate (x, y) around origin by *angle* radians."""
        c, s = math.cos(angle), math.sin(angle)
        return x * c - y * s, x * s + y * c

    def _line_positions(
        self, count: int, cx: float, cy: float
    ) -> List[Tuple[float, float]]:
        positions = []
        half = (count - 1) * self.spacing / 2.0
        for i in range(count):
            lx = -half + i * self.spacing
            ly = 0.0
            rx, ry = self._rotate(lx, ly, self.heading)
            positions.append((cx + rx, cy + ry))
        return positions

    def _wedge_positions(
        self, count: int, cx: float, cy: float
    ) -> List[Tuple[float, float]]:
        positions = [(cx, cy)]  # Lead vessel at center
        for i in range(1, count):
            row = int(math.ceil(i / 2))
            side = 1 if i % 2 == 1 else -1
            lx = -row * self.spacing
            ly = side * row * self.spacing * 0.5
            rx, ry = self._rotate(lx, ly, self.heading)
            positions.append((cx + rx, cy + ry))
        return positions

    def _circle_positions(
        self, count: int, cx: float, cy: float
    ) -> List[Tuple[float, float]]:
        positions = []
        if count == 1:
            return [(cx, cy)]
        radius = self.spacing * count / (2 * math.pi)
        for i in range(count):
            angle = 2 * math.pi * i / count + self.heading
            positions.append((cx + radius * math.cos(angle),
                              cy + radius * math.sin(angle)))
        return positions

    def _grid_positions(
        self, count: int, cx: float, cy: float
    ) -> List[Tuple[float, float]]:
        cols = max(1, math.ceil(math.sqrt(count)))
        rows = max(1, math.ceil(count / cols))
        positions = []
        half_x = (cols - 1) * self.spacing / 2.0
        half_y = (rows - 1) * self.spacing / 2.0
        for i in range(count):
            r, c = divmod(i, cols)
            lx = -half_x + c * self.spacing
            ly = -half_y + r * self.spacing
            rx, ry = self._rotate(lx, ly, self.heading)
            positions.append((cx + rx, cy + ry))
        return positions

    def _v_shape_positions(
        self, count: int, cx: float, cy: float
    ) -> List[Tuple[float, float]]:
        positions = [(cx, cy)]  # Lead at tip
        for i in range(1, count):
            side = 1 if i % 2 == 1 else -1
            depth = (i + 1) // 2
            lx = -depth * self.spacing
            ly = side * depth * self.spacing * 0.6
            rx, ry = self._rotate(lx, ly, self.heading)
            positions.append((cx + rx, cy + ry))
        return positions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Smallest signed difference between two angles (radians)."""
        d = (a - b) % (2 * math.pi)
        if d > math.pi:
            d -= 2 * math.pi
        return d
