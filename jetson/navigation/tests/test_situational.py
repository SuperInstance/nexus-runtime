"""Tests for situational awareness module."""

import pytest

from jetson.navigation.geospatial import Coordinate
from jetson.navigation.collision import CollisionThreat, Severity, VesselState
from jetson.navigation.situational import (
    Contact, ContactType, SituationalAwareness, SituationReport,
    Weather, WeatherCondition,
)


class TestContactType:
    def test_type_values(self):
        assert ContactType.UNKNOWN.value == 0
        assert ContactType.VESSEL.value == 1
        assert ContactType.BUOY.value == 2
        assert ContactType.LAND.value == 3
        assert ContactType.PLATFORM.value == 4
        assert ContactType.DEBRIS.value == 5
        assert ContactType.AIS_TARGET.value == 6

    def test_all_types_exist(self):
        types = list(ContactType)
        assert len(types) == 7


class TestWeatherCondition:
    def test_condition_values(self):
        assert WeatherCondition.CALM.value == 0
        assert WeatherCondition.MODERATE.value == 1
        assert WeatherCondition.ROUGH.value == 2
        assert WeatherCondition.STORM.value == 3
        assert WeatherCondition.HURRICANE.value == 4

    def test_all_conditions(self):
        conditions = list(WeatherCondition)
        assert len(conditions) == 5


class TestWeather:
    def test_default_weather(self):
        w = Weather()
        assert w.wind_speed == 0.0
        assert w.visibility == 10000.0
        assert w.condition == WeatherCondition.CALM

    def test_classify_calm(self):
        assert Weather.classify(2.0, 0.2) == WeatherCondition.CALM

    def test_classify_moderate(self):
        assert Weather.classify(5.0, 0.5) == WeatherCondition.MODERATE

    def test_classify_rough(self):
        assert Weather.classify(10.0, 1.5) == WeatherCondition.ROUGH

    def test_classify_storm(self):
        assert Weather.classify(20.0, 3.0) == WeatherCondition.STORM

    def test_classify_hurricane(self):
        assert Weather.classify(30.0, 6.0) == WeatherCondition.HURRICANE

    def test_classify_boundary_calm_moderate(self):
        assert Weather.classify(3.0, 0.5) == WeatherCondition.MODERATE

    def test_classify_boundary_moderate_rough(self):
        assert Weather.classify(8.0, 1.5) == WeatherCondition.ROUGH

    def test_classify_high_wind_low_wave(self):
        """High wind with low waves -> hurricane by wind."""
        assert Weather.classify(26.0, 2.0) == WeatherCondition.HURRICANE


class TestContact:
    def test_create_contact(self):
        c = Contact(
            id="c1",
            position=Coordinate(latitude=0, longitude=0),
            velocity=(1.0, 0.0),
        )
        assert c.id == "c1"
        assert c.heading == 0.0
        assert c.contact_type == ContactType.UNKNOWN
        assert c.confidence == 1.0

    def test_contact_defaults(self):
        c = Contact(id="c1", position=Coordinate(0, 0), velocity=(0, 0))
        assert c.distance == 0.0
        assert c.last_updated == 0.0


class TestSituationReport:
    def test_create_report(self):
        own = VesselState(
            position=Coordinate(0, 0), speed=2.0, heading=90.0, vessel_id="own"
        )
        weather = Weather()
        report = SituationReport(
            own_vessel=own,
            contacts=[],
            threats=[],
            weather=weather,
            overall_risk=0.1,
            timestamp=100.0,
        )
        assert report.overall_risk == 0.1
        assert report.timestamp == 100.0
        assert len(report.contacts) == 0
        assert len(report.threats) == 0

    def test_default_timestamp(self):
        own = VesselState(
            position=Coordinate(0, 0), speed=2.0, heading=90.0
        )
        report = SituationReport(
            own_vessel=own, contacts=[], threats=[],
            weather=Weather(),
        )
        assert report.timestamp == 0.0


class TestSituationalAwarenessInit:
    def test_default_init(self):
        sa = SituationalAwareness()
        assert sa.get_contact_count() == 0
        assert sa.max_contacts == 100

    def test_custom_max_contacts(self):
        sa = SituationalAwareness(max_contacts=50)
        assert sa.max_contacts == 50

    def test_custom_timeout(self):
        sa = SituationalAwareness(track_timeout=600.0)
        assert sa.track_timeout == 600.0


class TestUpdateContacts:
    def test_add_new_contact(self):
        sa = SituationalAwareness()
        readings = [{'id': 'c1', 'latitude': 0.0, 'longitude': 0.0}]
        updated = sa.update_contacts(readings)
        assert len(updated) == 1
        assert sa.get_contact_count() == 1

    def test_update_existing_contact(self):
        sa = SituationalAwareness()
        sa.update_contacts([{'id': 'c1', 'latitude': 0.0, 'longitude': 0.0}])
        sa.update_contacts([{'id': 'c1', 'latitude': 1.0, 'longitude': 1.0}])
        assert sa.get_contact_count() == 1
        contact = sa.get_contact('c1')
        assert contact.position.latitude == pytest.approx(1.0)

    def test_add_multiple_contacts(self):
        sa = SituationalAwareness()
        readings = [
            {'id': f'c{i}', 'latitude': float(i), 'longitude': float(i)}
            for i in range(5)
        ]
        updated = sa.update_contacts(readings)
        assert len(updated) == 5
        assert sa.get_contact_count() == 5

    def test_max_contacts_limit(self):
        sa = SituationalAwareness(max_contacts=3)
        readings = [
            {'id': f'c{i}', 'latitude': float(i), 'longitude': float(i),
             'timestamp': float(i)}
            for i in range(5)
        ]
        # First add 3, then add 2 more; should evict oldest
        sa.update_contacts(readings[:3])
        sa.update_contacts(readings[3:])
        assert sa.get_contact_count() == 3

    def test_speed_heading_in_reading(self):
        sa = SituationalAwareness()
        sa.update_contacts([{
            'id': 'c1', 'latitude': 0.0, 'longitude': 0.0,
            'speed': 5.0, 'heading': 90.0,
        }])
        contact = sa.get_contact('c1')
        assert abs(contact.velocity[0]) > 0  # East component

    def test_timestamp_stored(self):
        sa = SituationalAwareness()
        sa.update_contacts([{
            'id': 'c1', 'latitude': 0.0, 'longitude': 0.0,
            'timestamp': 1234.5,
        }])
        contact = sa.get_contact('c1')
        assert contact.last_updated == 1234.5


class TestTrackContact:
    def test_track_new_contact(self):
        sa = SituationalAwareness()
        pos = Coordinate(latitude=0, longitude=0)
        sa.track_contact('c1', pos, 1.0)
        sa.track_contact('c1', pos, 2.0)
        # History should exist
        assert 'c1' in sa._contact_history

    def test_track_unknown_contact(self):
        sa = SituationalAwareness()
        pos = Coordinate(latitude=0, longitude=0)
        sa.track_contact('unknown', pos, 1.0)
        # Should create history entry
        assert 'unknown' in sa._contact_history

    def test_history_limit(self):
        sa = SituationalAwareness()
        for i in range(150):
            pos = Coordinate(latitude=float(i) * 0.001, longitude=0)
            sa.track_contact('c1', pos, float(i))
        # Should be limited to 100 entries
        assert len(sa._contact_history['c1']) == 100


class TestPredictContactPositions:
    def test_no_movement(self):
        sa = SituationalAwareness()
        contacts = [Contact(id='c1', position=Coordinate(0, 0), velocity=(0, 0))]
        predictions = sa.predict_contact_positions(contacts, 60.0)
        assert predictions['c1'].latitude == pytest.approx(0.0, abs=1e-10)
        assert predictions['c1'].longitude == pytest.approx(0.0, abs=1e-10)

    def test_north_movement(self):
        sa = SituationalAwareness()
        contacts = [Contact(id='c1', position=Coordinate(0, 0), velocity=(0, 10.0))]
        predictions = sa.predict_contact_positions(contacts, 60.0)
        assert predictions['c1'].latitude > 0.0

    def test_east_movement(self):
        sa = SituationalAwareness()
        contacts = [Contact(id='c1', position=Coordinate(0, 0), velocity=(10.0, 0))]
        predictions = sa.predict_contact_positions(contacts, 60.0)
        assert predictions['c1'].longitude > 0.0

    def test_multiple_contacts(self):
        sa = SituationalAwareness()
        contacts = [
            Contact(id=f'c{i}', position=Coordinate(0, 0),
                    velocity=(float(i), 0))
            for i in range(5)
        ]
        predictions = sa.predict_contact_positions(contacts, 60.0)
        assert len(predictions) == 5

    def test_zero_dt(self):
        sa = SituationalAwareness()
        contacts = [Contact(id='c1', position=Coordinate(0, 0), velocity=(10, 10))]
        predictions = sa.predict_contact_positions(contacts, 0.0)
        assert predictions['c1'].latitude == pytest.approx(0.0, abs=1e-10)


class TestClassifyContact:
    def test_stationary_near(self):
        sa = SituationalAwareness()
        c = Contact(id='c1', position=Coordinate(0, 0), velocity=(0, 0), distance=100)
        assert sa.classify_contact(c) == ContactType.BUOY

    def test_stationary_far(self):
        sa = SituationalAwareness()
        c = Contact(id='c1', position=Coordinate(0, 0), velocity=(0, 0), distance=10000)
        assert sa.classify_contact(c) == ContactType.LAND

    def test_stationary_medium_range(self):
        sa = SituationalAwareness()
        c = Contact(id='c1', position=Coordinate(0, 0), velocity=(0, 0), distance=2000)
        assert sa.classify_contact(c) == ContactType.PLATFORM

    def test_moving_vessel(self):
        sa = SituationalAwareness()
        c = Contact(id='c1', position=Coordinate(0, 0), velocity=(5.0, 5.0))
        speed = (c.velocity[0]**2 + c.velocity[1]**2) ** 0.5
        assert speed > 0.5
        assert sa.classify_contact(c) == ContactType.VESSEL

    def test_slow_moving(self):
        sa = SituationalAwareness()
        c = Contact(id='c1', position=Coordinate(0, 0), velocity=(0.2, 0.1))
        assert sa.classify_contact(c) in (ContactType.DEBRIS, ContactType.BUOY)


class TestAssessOverallRisk:
    def test_no_threats_calm(self):
        sa = SituationalAwareness()
        weather = Weather()
        own = VesselState(position=Coordinate(0, 0), speed=2.0, heading=0.0)
        risk = sa.assess_overall_risk([], weather, own)
        assert risk == pytest.approx(0.015, abs=0.001)

    def test_high_severity_threat(self):
        sa = SituationalAwareness()
        weather = Weather()
        own = VesselState(position=Coordinate(0, 0), speed=2.0, heading=0.0)
        threat = CollisionThreat(
            vessel_id="t", position=Coordinate(0, 0),
            velocity=(0, 0), distance=50,
            tcpa=10, dcpa=5, severity=Severity.CRITICAL,
        )
        risk = sa.assess_overall_risk([threat], weather, own)
        assert risk > 0.3

    def test_storm_weather(self):
        sa = SituationalAwareness()
        weather = Weather(wind_speed=20.0, wave_height=3.0,
                          condition=WeatherCondition.STORM)
        own = VesselState(position=Coordinate(0, 0), speed=5.0, heading=0.0)
        risk = sa.assess_overall_risk([], weather, own)
        assert risk > 0.07

    def test_poor_visibility(self):
        sa = SituationalAwareness()
        weather = Weather(visibility=100.0)
        own = VesselState(position=Coordinate(0, 0), speed=2.0, heading=0.0)
        risk = sa.assess_overall_risk([], weather, own)
        assert risk > 0.07

    def test_high_speed_risk(self):
        sa = SituationalAwareness()
        weather = Weather()
        own_slow = VesselState(position=Coordinate(0, 0), speed=1.0, heading=0.0)
        own_fast = VesselState(position=Coordinate(0, 0), speed=15.0, heading=0.0)
        r1 = sa.assess_overall_risk([], weather, own_slow)
        r2 = sa.assess_overall_risk([], weather, own_fast)
        assert r2 > r1

    def test_risk_bounded(self):
        sa = SituationalAwareness()
        weather = Weather(wind_speed=30, wave_height=6, condition=WeatherCondition.HURRICANE, visibility=50)
        own = VesselState(position=Coordinate(0, 0), speed=20.0, heading=0.0)
        threats = [
            CollisionThreat(
                vessel_id="t", position=Coordinate(0, 0),
                velocity=(0, 0), distance=10,
                tcpa=1, dcpa=1, severity=Severity.CRITICAL,
            )
        ]
        risk = sa.assess_overall_risk(threats, weather, own)
        assert 0.0 <= risk <= 1.0


class TestComputeSituationReport:
    def test_empty_contacts(self):
        sa = SituationalAwareness()
        own = VesselState(position=Coordinate(0, 0), speed=2.0, heading=90.0, vessel_id="own")
        weather = Weather()
        report = sa.compute_situation_report(own, [], weather, 100.0)
        assert isinstance(report, SituationReport)
        assert report.timestamp == 100.0
        assert len(report.contacts) == 0

    def test_with_contacts(self):
        sa = SituationalAwareness()
        own = VesselState(position=Coordinate(0, 0), speed=2.0, heading=90.0, vessel_id="own")
        contacts = [
            Contact(id='c1', position=Coordinate(0.01, 0.01), velocity=(5, 5)),
        ]
        weather = Weather()
        report = sa.compute_situation_report(own, contacts, weather)
        assert isinstance(report, SituationReport)
        assert len(report.contacts) == 1

    def test_default_timestamp(self):
        sa = SituationalAwareness()
        own = VesselState(position=Coordinate(0, 0), speed=2.0, heading=90.0)
        report = sa.compute_situation_report(own, [], Weather())
        assert report.timestamp == 0.0

    def test_overall_risk_computed(self):
        sa = SituationalAwareness()
        own = VesselState(position=Coordinate(0, 0), speed=2.0, heading=90.0)
        report = sa.compute_situation_report(own, [], Weather())
        assert 0.0 <= report.overall_risk <= 1.0


class TestGetContact:
    def test_get_existing(self):
        sa = SituationalAwareness()
        sa.update_contacts([{'id': 'c1', 'latitude': 0, 'longitude': 0}])
        c = sa.get_contact('c1')
        assert c is not None
        assert c.id == 'c1'

    def test_get_missing(self):
        sa = SituationalAwareness()
        assert sa.get_contact('missing') is None


class TestGetAllContacts:
    def test_empty(self):
        sa = SituationalAwareness()
        assert sa.get_all_contacts() == []

    def test_returns_all(self):
        sa = SituationalAwareness()
        sa.update_contacts([
            {'id': 'c1', 'latitude': 0, 'longitude': 0},
            {'id': 'c2', 'latitude': 1, 'longitude': 1},
        ])
        contacts = sa.get_all_contacts()
        assert len(contacts) == 2


class TestRemoveStaleContacts:
    def test_remove_stale(self):
        sa = SituationalAwareness(track_timeout=10.0)
        sa.update_contacts([{
            'id': 'c1', 'latitude': 0, 'longitude': 0, 'timestamp': 0.0,
        }])
        removed = sa.remove_stale_contacts(20.0)
        assert 'c1' in removed
        assert sa.get_contact_count() == 0

    def test_keep_fresh(self):
        sa = SituationalAwareness(track_timeout=100.0)
        sa.update_contacts([{
            'id': 'c1', 'latitude': 0, 'longitude': 0, 'timestamp': 0.0,
        }])
        removed = sa.remove_stale_contacts(10.0)
        assert len(removed) == 0
        assert sa.get_contact_count() == 1

    def test_empty_no_error(self):
        sa = SituationalAwareness()
        removed = sa.remove_stale_contacts(100.0)
        assert removed == []

    def test_mixed_fresh_stale(self):
        sa = SituationalAwareness(track_timeout=10.0)
        sa.update_contacts([
            {'id': 'fresh', 'latitude': 0, 'longitude': 0, 'timestamp': 5.0},
            {'id': 'stale', 'latitude': 1, 'longitude': 1, 'timestamp': 0.0},
        ])
        removed = sa.remove_stale_contacts(15.0)
        assert 'stale' in removed
        assert 'fresh' not in removed
        assert sa.get_contact_count() == 1
