"""Situational awareness for autonomous marine navigation.

Provides contact tracking, prediction, classification, threat assessment,
and comprehensive situation reporting for autonomous vessels.
"""

from dataclasses import dataclass, field
from enum import Enum
from math import cos, radians, sin, sqrt
from typing import Dict, List, Optional, Tuple

from .geospatial import Coordinate, GeoCalculator
from .collision import CollisionAvoidance, CollisionThreat, Severity, VesselState


class ContactType(Enum):
    """Types of maritime contacts."""
    UNKNOWN = 0
    VESSEL = 1
    BUOY = 2
    LAND = 3
    PLATFORM = 4
    DEBRIS = 5
    AIS_TARGET = 6


class WeatherCondition(Enum):
    """Weather condition categories."""
    CALM = 0
    MODERATE = 1
    ROUGH = 2
    STORM = 3
    HURRICANE = 4


@dataclass
class Weather:
    """Current weather conditions."""
    wind_speed: float = 0.0        # m/s
    wind_direction: float = 0.0    # degrees from north
    wave_height: float = 0.0       # meters
    visibility: float = 10000.0    # meters
    current_speed: float = 0.0     # m/s
    current_direction: float = 0.0 # degrees from north
    condition: WeatherCondition = WeatherCondition.CALM

    @staticmethod
    def classify(wind_speed: float, wave_height: float) -> WeatherCondition:
        """Classify weather conditions from wind and wave parameters."""
        if wind_speed < 3.0 and wave_height < 0.3:
            return WeatherCondition.CALM
        if wind_speed < 8.0 and wave_height < 1.0:
            return WeatherCondition.MODERATE
        if wind_speed < 14.0 and wave_height < 2.5:
            return WeatherCondition.ROUGH
        if wind_speed < 25.0 and wave_height < 5.0:
            return WeatherCondition.STORM
        return WeatherCondition.HURRICANE


@dataclass
class Contact:
    """A detected maritime contact."""
    id: str
    position: Coordinate
    velocity: Tuple[float, float]  # (east_mps, north_mps)
    heading: float = 0.0           # degrees
    contact_type: ContactType = ContactType.UNKNOWN
    distance: float = 0.0          # meters from own vessel
    bearing: float = 0.0           # degrees from own vessel
    last_updated: float = 0.0      # timestamp
    confidence: float = 1.0        # tracking confidence [0, 1]


@dataclass
class SituationReport:
    """Comprehensive situational awareness report."""
    own_vessel: VesselState
    contacts: List[Contact]
    threats: List[CollisionThreat]
    weather: Weather
    overall_risk: float = 0.0      # [0, 1]
    timestamp: float = 0.0


class SituationalAwareness:
    """Situational awareness system for maritime navigation."""

    def __init__(self, max_contacts: int = 100, track_timeout: float = 300.0):
        self._contacts: Dict[str, Contact] = {}
        self._contact_history: Dict[str, List[Tuple[float, Coordinate]]] = {}
        self.max_contacts = max_contacts
        self.track_timeout = track_timeout
        self._collision_avoidance = CollisionAvoidance()

    def update_contacts(self, sensor_readings: List[dict]) -> List[Contact]:
        """Update contacts from sensor readings.

        Args:
            sensor_readings: List of dicts with keys:
                id, latitude, longitude, speed, heading (optional)

        Returns:
            List of updated/added contacts.
        """
        updated = []
        for reading in sensor_readings:
            contact_id = reading['id']
            coord = Coordinate(
                latitude=reading['latitude'],
                longitude=reading['longitude'],
            )
            speed = reading.get('speed', 0.0)
            heading = reading.get('heading', 0.0)
            velocity = self._speed_heading_to_velocity(speed, heading)

            if contact_id in self._contacts:
                existing = self._contacts[contact_id]
                existing.position = coord
                existing.velocity = velocity
                existing.heading = heading
                existing.last_updated = reading.get('timestamp', 0.0)
                updated.append(existing)
            else:
                if len(self._contacts) >= self.max_contacts:
                    # Remove oldest contact
                    oldest_id = min(
                        self._contacts, key=lambda k: self._contacts[k].last_updated
                    )
                    del self._contacts[oldest_id]
                    if oldest_id in self._contact_history:
                        del self._contact_history[oldest_id]

                contact = Contact(
                    id=contact_id,
                    position=coord,
                    velocity=velocity,
                    heading=heading,
                    contact_type=ContactType.UNKNOWN,
                    last_updated=reading.get('timestamp', 0.0),
                )
                self._contacts[contact_id] = contact
                self._contact_history[contact_id] = []
                updated.append(contact)

        return updated

    def track_contact(
        self, contact_id: str, position: Coordinate, timestamp: float
    ) -> None:
        """Record a position observation for a contact.

        Maintains a position history for tracking and prediction.
        """
        if contact_id not in self._contact_history:
            self._contact_history[contact_id] = []
        history = self._contact_history[contact_id]
        history.append((timestamp, position))
        # Keep only last 100 entries
        if len(history) > 100:
            self._contact_history[contact_id] = history[-100:]

    def predict_contact_positions(
        self, contacts: List[Contact], dt: float
    ) -> Dict[str, Coordinate]:
        """Predict future positions of contacts.

        Args:
            contacts: List of contacts to predict for.
            dt: Time delta in seconds.

        Returns:
            Dict mapping contact ID to predicted Coordinate.
        """
        predictions = {}
        for contact in contacts:
            east_mps, north_mps = contact.velocity
            lat_rad = radians(contact.position.latitude)
            dlat = north_mps * dt / 110540.0
            dlon = east_mps * dt / (111320.0 * cos(lat_rad))
            predicted = Coordinate(
                latitude=contact.position.latitude + dlat,
                longitude=contact.position.longitude + dlon,
            )
            predictions[contact.id] = predicted
        return predictions

    def compute_situation_report(
        self,
        own: VesselState,
        contacts: List[Contact],
        weather: Weather,
        timestamp: float = 0.0,
    ) -> SituationReport:
        """Compute a comprehensive situation report.

        Analyzes all contacts, detects threats, and assesses overall risk.
        """
        # Convert contacts to VesselState for collision analysis
        other_vessels = []
        for contact in contacts:
            speed = sqrt(contact.velocity[0] ** 2 + contact.velocity[1] ** 2)
            other_vessels.append(VesselState(
                position=contact.position,
                speed=speed,
                heading=contact.heading,
                vessel_id=contact.id,
            ))

        threats = self._collision_avoidance.detect_threats(own, other_vessels)
        overall_risk = self.assess_overall_risk(threats, weather, own)

        return SituationReport(
            own_vessel=own,
            contacts=contacts,
            threats=threats,
            weather=weather,
            overall_risk=overall_risk,
            timestamp=timestamp,
        )

    def classify_contact(self, contact: Contact) -> ContactType:
        """Classify a contact based on its movement characteristics.

        Static objects -> BUOY/LAND/PLATFORM
        Slow moving -> DEBRIS
        Moderate speed -> VESSEL
        """
        speed = sqrt(contact.velocity[0] ** 2 + contact.velocity[1] ** 2)

        if speed < 0.1:
            # Stationary object
            if contact.distance > 5000:
                return ContactType.LAND
            elif contact.distance > 500:
                return ContactType.PLATFORM
            else:
                return ContactType.BUOY
        elif speed < 0.5:
            return ContactType.DEBRIS
        elif speed < 0.3:
            return ContactType.BUOY  # Slightly drifting buoy
        else:
            return ContactType.VESSEL

    def assess_overall_risk(
        self,
        threats: List[CollisionThreat],
        weather: Weather,
        own: VesselState
    ) -> float:
        """Assess overall navigation risk from threats and environment.

        Returns a risk score [0, 1].
        """
        if not threats:
            threat_risk = 0.0
        else:
            max_severity = max(t.severity.value for t in threats)
            threat_risk = max_severity / 4.0  # Normalize CRITICAL=4 -> 1.0

        # Weather risk factor
        weather_risk = {
            WeatherCondition.CALM: 0.0,
            WeatherCondition.MODERATE: 0.1,
            WeatherCondition.ROUGH: 0.3,
            WeatherCondition.STORM: 0.5,
            WeatherCondition.HURRICANE: 0.8,
        }.get(weather.condition, 0.0)

        # Visibility risk
        vis_risk = 0.0
        if weather.visibility < 200:
            vis_risk = 0.5
        elif weather.visibility < 1000:
            vis_risk = 0.3
        elif weather.visibility < 5000:
            vis_risk = 0.1

        # Own vessel speed risk (faster = less reaction time)
        speed_risk = min(0.2, own.speed / 20.0)

        overall = min(1.0, 0.5 * threat_risk + 0.2 * weather_risk + 0.15 * vis_risk + 0.15 * speed_risk)
        return overall

    def get_contact(self, contact_id: str) -> Optional[Contact]:
        """Retrieve a tracked contact by ID."""
        return self._contacts.get(contact_id)

    def get_all_contacts(self) -> List[Contact]:
        """Get all tracked contacts."""
        return list(self._contacts.values())

    def remove_stale_contacts(self, current_time: float) -> List[str]:
        """Remove contacts that haven't been updated recently.

        Returns list of removed contact IDs.
        """
        stale = []
        for cid, contact in list(self._contacts.items()):
            if current_time - contact.last_updated > self.track_timeout:
                stale.append(cid)
                del self._contacts[cid]
                if cid in self._contact_history:
                    del self._contact_history[cid]
        return stale

    def get_contact_count(self) -> int:
        """Return number of tracked contacts."""
        return len(self._contacts)

    @staticmethod
    def _speed_heading_to_velocity(speed: float, heading: float) -> Tuple[float, float]:
        """Convert speed and heading to velocity components."""
        heading_rad = radians(heading)
        vx = speed * sin(heading_rad)
        vy = speed * cos(heading_rad)
        return (vx, vy)
