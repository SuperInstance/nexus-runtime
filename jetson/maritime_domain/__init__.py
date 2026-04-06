"""
Maritime Domain Awareness Module — Phase 6 Round 6

Provides AIS processing, traffic pattern analysis, maritime zone management,
weather impact assessment, and maritime situational awareness capabilities.
"""

from .ais import AISMessage, AISDecoder
from .traffic import TrafficPattern, TrafficAnalyzer
from .zoning import MaritimeZone, ZoneType, ZoneManager
from .weather import WeatherCondition, WeatherImpact, WeatherAssessor
from .situational_awareness import (
    MaritimePicture, ThreatAssessment, MaritimeSituationalAwareness
)

__all__ = [
    "AISMessage", "AISDecoder",
    "TrafficPattern", "TrafficAnalyzer",
    "MaritimeZone", "ZoneType", "ZoneManager",
    "WeatherCondition", "WeatherImpact", "WeatherAssessor",
    "MaritimePicture", "ThreatAssessment", "MaritimeSituationalAwareness",
]
