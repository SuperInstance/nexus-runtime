"""NEXUS Autonomous Navigation Package.

Provides marine navigation capabilities including waypoint management,
path following, collision avoidance, geospatial calculations,
autopilot control, and situational awareness.
"""

from .geospatial import Coordinate, GeoCalculator
from .waypoint import Waypoint, WaypointManager
from .path_follower import CrossTrackError, PathFollower
from .collision import (
    CollisionAvoidance, CollisionThreat, Severity, VesselState,
)
from .pilot import Autopilot, PilotCommand, PilotMode, VesselStateInternal
from .situational import (
    Contact, ContactType, SituationalAwareness, SituationReport,
    Weather, WeatherCondition,
)

__all__ = [
    # Geospatial
    'Coordinate',
    'GeoCalculator',
    # Waypoint
    'Waypoint',
    'WaypointManager',
    # Path Following
    'CrossTrackError',
    'PathFollower',
    # Collision Avoidance
    'CollisionAvoidance',
    'CollisionThreat',
    'Severity',
    'VesselState',
    # Pilot
    'Autopilot',
    'PilotCommand',
    'PilotMode',
    'VesselStateInternal',
    # Situational Awareness
    'Contact',
    'ContactType',
    'SituationalAwareness',
    'SituationReport',
    'Weather',
    'WeatherCondition',
]
