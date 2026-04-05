"""NEXUS Dead Reckoning Engine.

Sensor fusion, position estimation, Kalman filtering, intention
broadcasting, and waypoint navigation for marine robotics.
"""

from .navigation import NavigationMath
from .ins import INSIntegrator, IntegrationMethod
from .kalman import KalmanApproach, KalmanFilter2D
from .compass import CompassFusion
from .position_estimator import PositionEstimator, SensorQuality
from .intention import (
    IntentionBroadcaster,
    IntentionMessage,
    CollisionAssessment,
    CPAAlgorithm,
    DRMessageType,
    DR_MSG_TYPE_INFO,
)
from .waypoint import WaypointNavigator, WaypointStatus

__all__ = [
    "NavigationMath",
    "INSIntegrator",
    "IntegrationMethod",
    "KalmanApproach",
    "KalmanFilter2D",
    "CompassFusion",
    "PositionEstimator",
    "SensorQuality",
    "IntentionBroadcaster",
    "IntentionMessage",
    "CollisionAssessment",
    "CPAAlgorithm",
    "DRMessageType",
    "DR_MSG_TYPE_INFO",
    "WaypointNavigator",
    "WaypointStatus",
]
