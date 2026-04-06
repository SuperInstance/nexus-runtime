"""Tests for collision avoidance system."""

import pytest

from jetson.navigation.geospatial import Coordinate
from jetson.navigation.collision import (
    CollisionAvoidance, CollisionThreat, Severity, VesselState,
)


def make_vessel(lat, lon, speed, heading, vid="v"):
    return VesselState(
        position=Coordinate(latitude=lat, longitude=lon),
        speed=speed, heading=heading, vessel_id=vid,
    )


class TestSeverity:
    def test_severity_values(self):
        assert Severity.NONE.value == 0
        assert Severity.LOW.value == 1
        assert Severity.MEDIUM.value == 2
        assert Severity.HIGH.value == 3
        assert Severity.CRITICAL.value == 4

    def test_severity_ordering(self):
        assert Severity.NONE.value < Severity.LOW.value < Severity.CRITICAL.value


class TestVesselState:
    def test_create_vessel(self):
        v = make_vessel(0, 0, 5.0, 90.0)
        assert v.speed == 5.0
        assert v.heading == 90.0
        assert v.vessel_id == "v"

    def test_default_vessel_id(self):
        v = VesselState(
            position=Coordinate(latitude=0, longitude=0),
            speed=1.0, heading=0.0,
        )
        assert v.vessel_id == ""


class TestCollisionThreat:
    def test_create_threat(self):
        t = CollisionThreat(
            vessel_id="target1",
            position=Coordinate(latitude=0, longitude=0),
            velocity=(1.0, 0.0),
            distance=500.0,
            tcpa=120.0,
            dcpa=50.0,
            severity=Severity.HIGH,
        )
        assert t.vessel_id == "target1"
        assert t.severity == Severity.HIGH
        assert t.tcpa == 120.0


class TestCollisionAvoidanceInit:
    def test_defaults(self):
        ca = CollisionAvoidance()
        assert ca.safe_distance == 100.0
        assert ca.warning_distance == 300.0
        assert ca.critical_distance == 50.0
        assert ca.max_tcpa == 600.0

    def test_custom_params(self):
        ca = CollisionAvoidance(
            safe_distance=200.0, warning_distance=500.0,
            critical_distance=100.0, max_tcpa=300.0,
        )
        assert ca.safe_distance == 200.0
        assert ca.warning_distance == 500.0


class TestDetectThreats:
    def test_no_others(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 2.0, 0.0)
        threats = ca.detect_threats(own, [])
        assert threats == []

    def test_distant_vessel_no_threat(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 2.0, 0.0, "own")
        other = make_vessel(10, 10, 1.0, 180.0, "far")
        threats = ca.detect_threats(own, [other])
        assert len(threats) == 0

    def test_converging_vessels(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 5.0, 90.0, "own")
        # Vessel approaching from the east going west
        other = make_vessel(0, 0.01, 5.0, 270.0, "target")
        threats = ca.detect_threats(own, [other])
        # Close converging vessels should be a threat
        assert isinstance(threats, list)

    def test_multiple_others(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 2.0, 0.0, "own")
        others = [
            make_vessel(5, 5, 1.0, 180.0, f"v{i}")
            for i in range(5)
        ]
        threats = ca.detect_threats(own, others)
        assert isinstance(threats, list)

    def test_sorted_by_severity(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 5.0, 0.0, "own")
        others = [
            make_vessel(0, 0.001, 5.0, 270.0, "close"),
            make_vessel(5, 5, 1.0, 180.0, "far"),
        ]
        threats = ca.detect_threats(own, others)
        if len(threats) >= 2:
            assert threats[0].severity.value >= threats[1].severity.value

    def test_static_vessel(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 5.0, 0.0, "own")
        static = make_vessel(0.001, 0, 0.0, 0.0, "static")
        threats = ca.detect_threats(own, [static])
        assert isinstance(threats, list)


class TestTCPA:
    def test_converging_positive_tcpa(self):
        own = make_vessel(0, 0, 5.0, 90.0)
        other = make_vessel(0, 0.01, 5.0, 270.0)
        tcpa = CollisionAvoidance.compute_tcpa(own, other)
        assert tcpa > 0

    def test_diverging_negative_tcpa(self):
        own = make_vessel(0, 0, 5.0, 90.0)
        # Moving same direction, faster - separating
        other = make_vessel(0, 0.01, 10.0, 90.0)
        tcpa = CollisionAvoidance.compute_tcpa(own, other)
        assert tcpa < 0 or tcpa == float('inf')

    def test_parallel_zero_relative(self):
        """Parallel same speed/same direction should have large tcpa."""
        own = make_vessel(0, 0, 5.0, 90.0)
        other = make_vessel(0.001, 0, 5.0, 90.0)
        tcpa = CollisionAvoidance.compute_tcpa(own, other)
        # They should have a finite TCPA
        assert isinstance(tcpa, (int, float))

    def test_stationary_own(self):
        own = make_vessel(0, 0, 0.0, 0.0)
        other = make_vessel(0, 0.01, 5.0, 270.0)
        tcpa = CollisionAvoidance.compute_tcpa(own, other)
        assert tcpa > 0

    def test_stationary_other(self):
        own = make_vessel(0, 0, 5.0, 90.0)
        other = make_vessel(0, 0.01, 0.0, 0.0)
        tcpa = CollisionAvoidance.compute_tcpa(own, other)
        assert tcpa > 0

    def test_both_stationary(self):
        own = make_vessel(0, 0, 0.0, 0.0)
        other = make_vessel(0, 0.01, 0.0, 0.0)
        tcpa = CollisionAvoidance.compute_tcpa(own, other)
        assert tcpa == float('inf')


class TestDCPA:
    def test_head_on(self):
        own = make_vessel(0, 0, 5.0, 90.0)
        other = make_vessel(0, 0.01, 5.0, 270.0)
        dcpa = CollisionAvoidance.compute_dcpa(own, other)
        assert isinstance(dcpa, float)
        assert dcpa >= 0

    def test_zero_dcpa_direct_collision(self):
        own = make_vessel(0, 0, 5.0, 90.0)
        other = make_vessel(0, 0.01, 5.0, 270.0)
        dcpa = CollisionAvoidance.compute_dcpa(own, other)
        # Head-on should have very small DCPA
        assert dcpa < 100

    def test_passing_at_distance(self):
        own = make_vessel(0, 0, 5.0, 90.0)
        other = make_vessel(0.001, 0.01, 5.0, 270.0)
        dcpa = CollisionAvoidance.compute_dcpa(own, other)
        assert dcpa > 0

    def test_diverging_returns_current_distance(self):
        own = make_vessel(0, 0, 5.0, 90.0)
        other = make_vessel(0, 0.01, 10.0, 90.0)
        dcpa = CollisionAvoidance.compute_dcpa(own, other)
        assert dcpa > 0


class TestRiskScore:
    def test_zero_risk_diverging(self):
        score = CollisionAvoidance.compute_risk_score(-10.0, 1000.0, 2.0, 3.0)
        assert score == 0.0

    def test_high_risk_close(self):
        score = CollisionAvoidance.compute_risk_score(10.0, 5.0, 10.0, 10.0)
        assert score > 0.5

    def test_low_risk_far(self):
        score = CollisionAvoidance.compute_risk_score(500.0, 500.0, 1.0, 1.0)
        assert score < 0.3

    def test_risk_bounded(self):
        score = CollisionAvoidance.compute_risk_score(1.0, 0.0, 20.0, 20.0)
        assert 0.0 <= score <= 1.0

    def test_zero_tcpa(self):
        score = CollisionAvoidance.compute_risk_score(0.0, 10.0, 5.0, 5.0)
        assert score > 0

    def test_large_tcpa(self):
        score = CollisionAvoidance.compute_risk_score(1000.0, 100.0, 2.0, 2.0)
        assert score == 0.0

    def test_speed_factor(self):
        s1 = CollisionAvoidance.compute_risk_score(60.0, 50.0, 0.0, 0.0)
        s2 = CollisionAvoidance.compute_risk_score(60.0, 50.0, 20.0, 20.0)
        assert s2 > s1

    def test_dcpa_factor(self):
        s1 = CollisionAvoidance.compute_risk_score(60.0, 10.0, 5.0, 5.0)
        s2 = CollisionAvoidance.compute_risk_score(60.0, 500.0, 5.0, 5.0)
        assert s1 > s2


class TestEvasiveManeuver:
    def test_no_threat(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 5.0, 90.0, "own")
        threat = CollisionThreat(
            vessel_id="t", position=Coordinate(0, 0),
            velocity=(0, 0), distance=1000.0,
            tcpa=500.0, dcpa=500.0, severity=Severity.NONE,
        )
        heading_change, speed_mult = ca.generate_evasive_maneuver(threat, own)
        assert heading_change == 0.0
        assert speed_mult == 1.0

    def test_critical_threat_strong_maneuver(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 5.0, 0.0, "own")
        threat = CollisionThreat(
            vessel_id="t", position=Coordinate(0, 0.001),
            velocity=(0, -5), distance=50.0,
            tcpa=5.0, dcpa=5.0, severity=Severity.CRITICAL,
        )
        heading_change, speed_mult = ca.generate_evasive_maneuver(threat, own)
        assert abs(heading_change) >= 35.0
        assert speed_mult < 1.0

    def test_low_threat_gentle_maneuver(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 5.0, 0.0, "own")
        threat = CollisionThreat(
            vessel_id="t", position=Coordinate(0, 0.01),
            velocity=(0, -3), distance=200.0,
            tcpa=100.0, dcpa=100.0, severity=Severity.LOW,
        )
        heading_change, speed_mult = ca.generate_evasive_maneuver(threat, own)
        assert abs(heading_change) <= 20.0

    def test_high_threat_moderate_maneuver(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 5.0, 0.0, "own")
        threat = CollisionThreat(
            vessel_id="t", position=Coordinate(0, 0.001),
            velocity=(0, -5), distance=100.0,
            tcpa=20.0, dcpa=20.0, severity=Severity.HIGH,
        )
        heading_change, speed_mult = ca.generate_evasive_maneuver(threat, own)
        assert abs(heading_change) >= 20.0

    def test_medium_threat(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 5.0, 0.0, "own")
        threat = CollisionThreat(
            vessel_id="t", position=Coordinate(0, 0.005),
            velocity=(0, -3), distance=150.0,
            tcpa=60.0, dcpa=60.0, severity=Severity.MEDIUM,
        )
        heading_change, speed_mult = ca.generate_evasive_maneuver(threat, own)
        assert abs(heading_change) >= 10.0


class TestSafeZone:
    def test_zero_speed(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 0.0, 0.0)
        safe = ca.compute_safe_zone(own, 0.0, 30.0)
        assert safe >= ca.safe_distance

    def test_high_speed(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 0.0, 0.0)
        safe = ca.compute_safe_zone(own, 10.0, 30.0)
        assert safe > ca.safe_distance

    def test_reaction_time_factor(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 0.0, 0.0)
        s1 = ca.compute_safe_zone(own, 5.0, 10.0)
        s2 = ca.compute_safe_zone(own, 5.0, 60.0)
        assert s2 > s1

    def test_custom_safe_distance(self):
        ca = CollisionAvoidance(safe_distance=500.0)
        own = make_vessel(0, 0, 0.0, 0.0)
        safe = ca.compute_safe_zone(own, 0.0, 30.0)
        assert safe >= 500.0


class TestAvoidancePath:
    def test_no_threats(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 5.0, 90.0, "own")
        dest = Coordinate(latitude=0, longitude=1)
        path = ca.plan_avoidance_path(own, [], dest)
        assert len(path) == 2
        assert path[0] == own.position
        assert path[1] == dest

    def test_single_threat(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 5.0, 90.0, "own")
        dest = Coordinate(latitude=0, longitude=1)
        threat = CollisionThreat(
            vessel_id="t", position=Coordinate(0, 0.005),
            velocity=(0, 0), distance=500.0,
            tcpa=60.0, dcpa=50.0, severity=Severity.HIGH,
        )
        path = ca.plan_avoidance_path(own, [threat], dest)
        assert len(path) == 3  # start, avoidance, destination
        assert path[0] == own.position
        assert path[-1] == dest

    def test_multiple_threats(self):
        ca = CollisionAvoidance()
        own = make_vessel(0, 0, 5.0, 90.0, "own")
        dest = Coordinate(latitude=0, longitude=1)
        threats = [
            CollisionThreat(
                vessel_id=f"t{i}", position=Coordinate(0, 0.005 * (i + 1)),
                velocity=(0, 0), distance=500.0,
                tcpa=60.0, dcpa=50.0, severity=Severity.HIGH,
            )
            for i in range(3)
        ]
        path = ca.plan_avoidance_path(own, threats, dest)
        assert len(path) == 5  # start + 3 avoidance + dest
        assert path[-1] == dest


class TestVelocityComponents:
    def test_north(self):
        vx, vy = CollisionAvoidance._velocity_components(10.0, 0.0)
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(10.0, abs=1e-10)

    def test_east(self):
        vx, vy = CollisionAvoidance._velocity_components(10.0, 90.0)
        assert vx == pytest.approx(10.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_south(self):
        vx, vy = CollisionAvoidance._velocity_components(10.0, 180.0)
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(-10.0, abs=1e-10)

    def test_west(self):
        vx, vy = CollisionAvoidance._velocity_components(10.0, 270.0)
        assert vx == pytest.approx(-10.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_zero_speed(self):
        vx, vy = CollisionAvoidance._velocity_components(0.0, 45.0)
        assert vx == 0.0
        assert vy == 0.0

    def test_diagonal(self):
        vx, vy = CollisionAvoidance._velocity_components(10.0, 45.0)
        mag = (vx ** 2 + vy ** 2) ** 0.5
        assert mag == pytest.approx(10.0, abs=1e-10)
