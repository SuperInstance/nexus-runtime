"""
Constraint handling — generic and marine-specific constraints.

Pure Python — math, dataclasses, enum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ConstraintKind(Enum):
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    BOUND_LOWER = "bound_lower"
    BOUND_UPPER = "bound_upper"


@dataclass
class Constraint:
    name: str = ""
    constraint_type: ConstraintKind = ConstraintKind.INEQUALITY
    bounds: Tuple[float, float] = (0.0, float("inf"))
    active: bool = True
    index: int = 0


@dataclass
class ConstraintSet:
    constraints: List[Constraint] = field(default_factory=list)
    dimension: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist2(ax, ay, bx, by):
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx * dx + dy * dy)


def _point_in_polygon(px, py, polygon):
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# ConstraintHandler
# ---------------------------------------------------------------------------

class ConstraintHandler:
    """Generic constraint management and projection."""

    def __init__(self, dimension: int = 0):
        self._constraints: List[Constraint] = []
        self._next_id = 0
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def add_constraint(
        self,
        name: str,
        ctype: ConstraintKind,
        bounds: Tuple[float, float],
    ) -> int:
        cid = self._next_id
        self._next_id += 1
        c = Constraint(name=name, constraint_type=ctype, bounds=bounds,
                       active=True, index=cid)
        self._constraints.append(c)
        return cid

    @property
    def constraints(self) -> List[Constraint]:
        return list(self._constraints)

    def check_constraints(
        self, variables: List[float]
    ) -> List[dict]:
        """Return list of violations (dictionaries with name, index, magnitude)."""
        violations = []
        for c in self._constraints:
            if not c.active:
                continue
            lo, hi = c.bounds
            if c.index < len(variables):
                val = variables[c.index]
                if val < lo:
                    violations.append({
                        "name": c.name, "index": c.index,
                        "magnitude": lo - val, "type": "below_lower",
                    })
                if val > hi and hi != float("inf"):
                    violations.append({
                        "name": c.name, "index": c.index,
                        "magnitude": val - hi, "type": "above_upper",
                    })
        return violations

    def project_to_feasible(
        self,
        variables: List[float],
        constraints: Optional[List[Constraint]] = None,
    ) -> List[float]:
        """Clamp variables to respect their bounds."""
        proj = list(variables)
        cs = constraints or self._constraints
        for c in cs:
            if not c.active:
                continue
            if c.index < len(proj):
                lo, hi = c.bounds
                proj[c.index] = max(proj[c.index], lo)
                if hi != float("inf"):
                    proj[c.index] = min(proj[c.index], hi)
        return proj

    def compute_constraint_margin(
        self,
        variables: List[float],
        constraint: Constraint,
    ) -> float:
        """Distance of variable to nearest bound."""
        if constraint.index >= len(variables):
            return float("inf")
        val = variables[constraint.index]
        lo, hi = constraint.bounds
        margin_lo = val - lo
        margin_hi = (hi - val) if hi != float("inf") else float("inf")
        return min(margin_lo, margin_hi)

    def soft_constraint_penalty(
        self,
        variables: List[float],
        constraint: Constraint,
        weight: float,
    ) -> float:
        """Quadratic penalty for constraint violation."""
        if constraint.index >= len(variables):
            return 0.0
        val = variables[constraint.index]
        lo, hi = constraint.bounds
        penalty = 0.0
        if val < lo:
            penalty += weight * (lo - val) ** 2
        if hi != float("inf") and val > hi:
            penalty += weight * (val - hi) ** 2
        return penalty

    def generate_constraints(self, config: dict) -> ConstraintSet:
        """
        Generate constraints from a configuration dict.
        Expected keys: "dimension", "bounds" (list of (lo, hi) tuples).
        """
        dim = config.get("dimension", self._dimension)
        bounds_list = config.get("bounds", [])
        cs = []
        for i, (lo, hi) in enumerate(bounds_list):
            cs.append(Constraint(
                name=f"var_{i}",
                constraint_type=ConstraintKind.BOUND_LOWER,
                bounds=(lo, hi),
                active=True,
                index=i,
            ))
        return ConstraintSet(constraints=cs, dimension=dim)


# ---------------------------------------------------------------------------
# MarineConstraints
# ---------------------------------------------------------------------------

@dataclass
class VesselState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0


class MarineConstraints:
    """Marine-vessel-specific constraint generators."""

    def __init__(self):
        self.handler = ConstraintHandler()

    def actuator_limits(
        self,
        thrust_range: Tuple[float, float],
        rudder_range: Tuple[float, float],
    ) -> ConstraintSet:
        """Bounds on thrust and rudder angle."""
        cs = [
            Constraint(name="thrust", constraint_type=ConstraintKind.BOUND_LOWER,
                       bounds=thrust_range, index=0),
            Constraint(name="rudder", constraint_type=ConstraintKind.BOUND_LOWER,
                       bounds=rudder_range, index=1),
        ]
        return ConstraintSet(constraints=cs, dimension=2)

    def safety_zone(
        self,
        own_position: Tuple[float, float],
        obstacle_positions: List[Tuple[float, float]],
        min_distance: float,
    ) -> ConstraintSet:
        """Safety-distance constraints to obstacles."""
        cs = []
        for i, (ox, oy) in enumerate(obstacle_positions):
            dist = _dist2(own_position[0], own_position[1], ox, oy)
            margin = dist - min_distance
            cs.append(Constraint(
                name=f"obstacle_{i}",
                constraint_type=ConstraintKind.INEQUALITY,
                bounds=(0.0, float("inf")),
                active=margin < 0,
                index=i,
            ))
        return ConstraintSet(constraints=cs,
                             dimension=len(obstacle_positions))

    def geofence(
        self,
        boundary_polygon: List[Tuple[float, float]],
    ) -> ConstraintSet:
        """Geofence constraint (keep inside polygon)."""
        # We model as a single active/inactive constraint
        centroid_x = sum(p[0] for p in boundary_polygon) / max(len(boundary_polygon), 1)
        centroid_y = sum(p[1] for p in boundary_polygon) / max(len(boundary_polygon), 1)
        inside = _point_in_polygon(centroid_x, centroid_y, boundary_polygon)
        cs = [Constraint(
            name="geofence",
            constraint_type=ConstraintKind.INEQUALITY,
            bounds=(0.0, float("inf")),
            active=inside,
            index=0,
        )]
        return ConstraintSet(constraints=cs, dimension=2)

    def speed_limit(self, max_speed: float) -> ConstraintSet:
        """Speed constraint."""
        cs = [Constraint(
            name="speed",
            constraint_type=ConstraintKind.BOUND_UPPER,
            bounds=(0.0, max_speed),
            index=0,
        )]
        return ConstraintSet(constraints=cs, dimension=1)

    def colregs_rules(
        self,
        own_vessel: VesselState,
        other_vessels: List[VesselState],
    ) -> ConstraintSet:
        """COLREGs-inspired rules (simplified)."""
        cs = []
        for i, other in enumerate(other_vessels):
            dist = _dist2(own_vessel.x, own_vessel.y, other.x, other.y)
            # Rule: keep distance > 50m
            safe = dist > 50.0
            cs.append(Constraint(
                name=f"colreg_{i}",
                constraint_type=ConstraintKind.INEQUALITY,
                bounds=(50.0, float("inf")),
                active=not safe,
                index=i,
            ))
        return ConstraintSet(constraints=cs,
                             dimension=len(other_vessels))
