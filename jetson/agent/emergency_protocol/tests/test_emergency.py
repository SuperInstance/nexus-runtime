"""Comprehensive tests for NEXUS Emergency Protocol Bridge.

Tests all detectors, response levels, incident tracking, de-escalation,
fleet alerts, and the main EmergencyProtocol orchestration.

Run with:
    cd /tmp/nexus-runtime && python -m pytest jetson/agent/emergency_protocol/tests/ -v
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.emergency_protocol.protocol import (
    DEFAULT_THRESHOLDS,
    DeescalationResult,
    EmergencyAssessment,
    EmergencyLevel,
    EmergencyProtocol,
    EscalationResult,
    Incident,
    IncidentCategory,
    generate_incident_id,
)
from agent.emergency_protocol.detectors import (
    CommunicationLossDetector,
    MissionTimeoutDetector,
    SafetyViolationDetector,
    SensorFailureDetector,
    TrustCollapseDetector,
)
from agent.emergency_protocol.response import EmergencyResponder


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def now():
    """Current timestamp."""
    return time.time()


@pytest.fixture
def healthy_vessel_state(now):
    """Vessel state with all systems nominal."""
    return {
        "safety_state": {"state": "NOMINAL", "e_stop": False},
        "last_comm_time": now,
        "mission_start": now - 60,
        "expected_duration": 300.0,
    }


@pytest.fixture
def healthy_trust_scores():
    """Trust scores with all subsystems healthy."""
    return {
        "navigation": 0.90,
        "steering": 0.85,
        "engine": 0.88,
        "sensors": 0.92,
    }


@pytest.fixture
def healthy_sensor_status(now):
    """Sensor status with all sensors reading nominal."""
    return {
        "sensors": [
            {"id": "gps", "value": 12.34, "timestamp": now - 1,
             "min_value": -90.0, "max_value": 90.0, "quality": 0.95},
            {"id": "imu", "value": 0.01, "timestamp": now - 1,
             "min_value": -2.0, "max_value": 2.0, "quality": 0.98},
            {"id": "batt_v", "value": 12.6, "timestamp": now - 1,
             "min_value": 10.0, "max_value": 14.8, "quality": 1.0},
        ],
    }


@pytest.fixture
def protocol():
    """Create a basic EmergencyProtocol instance."""
    return EmergencyProtocol(vessel_id="test-vessel")


@pytest.fixture
def responder():
    """Create an EmergencyResponder without bridge."""
    return EmergencyResponder(vessel_id="test-vessel")


@pytest.fixture
def sample_incident(now):
    """Create a sample YELLOW-level incident."""
    return Incident(
        id="INC-TEST-001",
        level="YELLOW",
        category="SENSOR",
        description="Sensor degradation detected",
        timestamp=now,
    )


@pytest.fixture
def sample_red_incident(now):
    """Create a sample RED-level incident."""
    return Incident(
        id="INC-TEST-RED",
        level="RED",
        category="SAFETY",
        description="E-Stop triggered",
        timestamp=now,
    )


# ═══════════════════════════════════════════════════════════════════
# 1. Protocol Module Tests
# ═══════════════════════════════════════════════════════════════════

class TestEmergencyLevel:
    """Tests for the EmergencyLevel enum."""

    def test_level_values(self):
        assert EmergencyLevel.GREEN.value == "GREEN"
        assert EmergencyLevel.YELLOW.value == "YELLOW"
        assert EmergencyLevel.ORANGE.value == "ORANGE"
        assert EmergencyLevel.RED.value == "RED"


class TestIncidentCategory:
    """Tests for the IncidentCategory enum."""

    def test_category_values(self):
        assert IncidentCategory.SENSOR.value == "SENSOR"
        assert IncidentCategory.TRUST.value == "TRUST"
        assert IncidentCategory.SAFETY.value == "SAFETY"
        assert IncidentCategory.COMMUNICATION.value == "COMMUNICATION"
        assert IncidentCategory.MISSION.value == "MISSION"


class TestIncident:
    """Tests for the Incident dataclass."""

    def test_incident_creation(self, now):
        inc = Incident(
            id="INC-001",
            level="YELLOW",
            category="SENSOR",
            description="Test incident",
            timestamp=now,
        )
        assert inc.id == "INC-001"
        assert inc.level == "YELLOW"
        assert inc.category == "SENSOR"
        assert inc.description == "Test incident"
        assert inc.resolution is None
        assert inc.auto_actions_taken == []
        assert inc.vessel_state == {}

    def test_incident_with_optional_fields(self, now):
        inc = Incident(
            id="INC-002",
            level="RED",
            category="SAFETY",
            description="E-Stop",
            timestamp=now,
            vessel_state={"mode": "autonomous"},
            trust_scores={"nav": 0.5},
            auto_actions_taken=["halt_ops"],
        )
        assert inc.vessel_state == {"mode": "autonomous"}
        assert inc.trust_scores == {"nav": 0.5}
        assert inc.auto_actions_taken == ["halt_ops"]

    def test_incident_resolution(self, now):
        inc = Incident(
            id="INC-003",
            level="ORANGE",
            category="TRUST",
            description="Trust degradation",
            timestamp=now,
        )
        assert inc.resolution is None
        inc.resolution = "Sensor recalibrated"
        inc.resolved_at = now + 60
        assert inc.resolution == "Sensor recalibrated"
        assert inc.resolved_at is not None


class TestGenerateIncidentId:
    """Tests for generate_incident_id helper."""

    def test_unique_ids(self):
        ids = [generate_incident_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_id_format(self):
        inc_id = generate_incident_id()
        assert inc_id.startswith("INC-")
        assert len(inc_id) > 20  # Should have date + uuid


class TestEmergencyAssessment:
    """Tests for the EmergencyAssessment dataclass."""

    def test_level_changed_true(self):
        assessment = EmergencyAssessment(
            vessel_id="v1",
            timestamp=time.time(),
            previous_level="GREEN",
            current_level="RED",
        )
        assert assessment.level_changed is True

    def test_level_changed_false(self):
        assessment = EmergencyAssessment(
            vessel_id="v1",
            timestamp=time.time(),
            previous_level="ORANGE",
            current_level="ORANGE",
        )
        assert assessment.level_changed is False

    def test_escalated_true(self):
        assessment = EmergencyAssessment(
            vessel_id="v1",
            timestamp=time.time(),
            previous_level="GREEN",
            current_level="YELLOW",
        )
        assert assessment.escalated is True

    def test_escalated_false(self):
        assessment = EmergencyAssessment(
            vessel_id="v1",
            timestamp=time.time(),
            previous_level="ORANGE",
            current_level="GREEN",
        )
        assert assessment.escalated is False

    def test_escalated_same_level(self):
        assessment = EmergencyAssessment(
            vessel_id="v1",
            timestamp=time.time(),
            previous_level="YELLOW",
            current_level="YELLOW",
        )
        assert assessment.escalated is False


# ═══════════════════════════════════════════════════════════════════
# 2. SensorFailureDetector Tests
# ═══════════════════════════════════════════════════════════════════

class TestSensorFailureDetector:
    """Tests for the SensorFailureDetector."""

    def test_healthy_sensors(self, healthy_sensor_status):
        detector = SensorFailureDetector()
        incidents = detector.detect(healthy_sensor_status, DEFAULT_THRESHOLDS)
        assert len(incidents) == 0

    def test_offline_sensor(self, now):
        detector = SensorFailureDetector()
        sensor_status = {
            "sensors": [
                {"id": "gps", "value": None, "timestamp": now - 1},
            ],
        }
        incidents = detector.detect(sensor_status, DEFAULT_THRESHOLDS)
        assert len(incidents) == 1
        assert incidents[0].level == "ORANGE"
        assert incidents[0].category == "SENSOR"
        assert "offline" in incidents[0].description.lower()

    def test_stale_sensor_data(self, now):
        detector = SensorFailureDetector()
        sensor_status = {
            "sensors": [
                {"id": "imu", "value": 0.01, "timestamp": now - 60},
            ],
        }
        incidents = detector.detect(
            sensor_status,
            {"sensor_stale_seconds": 30.0},
        )
        assert len(incidents) == 1
        assert incidents[0].level == "YELLOW"
        assert "stale" in incidents[0].description.lower()

    def test_out_of_range_low(self, now):
        detector = SensorFailureDetector()
        sensor_status = {
            "sensors": [
                {"id": "batt", "value": 5.0, "timestamp": now - 1,
                 "min_value": 10.0, "max_value": 14.8},
            ],
        }
        incidents = detector.detect(sensor_status, DEFAULT_THRESHOLDS)
        assert len(incidents) == 1
        assert incidents[0].level == "YELLOW"
        assert "below range" in incidents[0].description.lower()

    def test_out_of_range_high(self, now):
        detector = SensorFailureDetector()
        sensor_status = {
            "sensors": [
                {"id": "temp", "value": 150.0, "timestamp": now - 1,
                 "min_value": -20.0, "max_value": 80.0},
            ],
        }
        incidents = detector.detect(sensor_status, DEFAULT_THRESHOLDS)
        assert len(incidents) == 1
        assert incidents[0].level == "YELLOW"
        assert "above range" in incidents[0].description.lower()

    def test_low_quality_sensor(self, now):
        detector = SensorFailureDetector()
        sensor_status = {
            "sensors": [
                {"id": "sonar", "value": 3.5, "timestamp": now - 1,
                 "quality": 0.1},
            ],
        }
        incidents = detector.detect(sensor_status, DEFAULT_THRESHOLDS)
        assert len(incidents) == 1
        assert incidents[0].level == "YELLOW"
        assert "quality" in incidents[0].description.lower()

    def test_multiple_sensor_issues(self, now):
        detector = SensorFailureDetector()
        sensor_status = {
            "sensors": [
                {"id": "gps", "value": None, "timestamp": now - 1},
                {"id": "imu", "value": 0.01, "timestamp": now - 60},
                {"id": "batt", "value": 5.0, "timestamp": now - 1,
                 "min_value": 10.0, "max_value": 14.8},
            ],
        }
        incidents = detector.detect(
            sensor_status,
            {"sensor_stale_seconds": 30.0},
        )
        assert len(incidents) >= 3  # At least offline, stale, and out-of-range

    def test_empty_sensor_list(self):
        detector = SensorFailureDetector()
        incidents = detector.detect({"sensors": []}, DEFAULT_THRESHOLDS)
        assert len(incidents) == 0


# ═══════════════════════════════════════════════════════════════════
# 3. TrustCollapseDetector Tests
# ═══════════════════════════════════════════════════════════════════

class TestTrustCollapseDetector:
    """Tests for the TrustCollapseDetector."""

    def test_healthy_trust(self, healthy_trust_scores):
        detector = TrustCollapseDetector()
        incidents = detector.detect(healthy_trust_scores, [])
        assert len(incidents) == 0

    def test_yellow_trust(self):
        detector = TrustCollapseDetector()
        trust_scores = {"navigation": 0.55}
        incidents = detector.detect(trust_scores, [])
        assert len(incidents) == 1
        assert incidents[0].level == "YELLOW"
        assert incidents[0].category == "TRUST"

    def test_orange_trust(self):
        detector = TrustCollapseDetector()
        trust_scores = {"engine": 0.30}
        incidents = detector.detect(trust_scores, [])
        assert len(incidents) == 1
        assert incidents[0].level == "ORANGE"
        assert incidents[0].category == "TRUST"

    def test_red_trust(self):
        detector = TrustCollapseDetector()
        trust_scores = {"steering": 0.10}
        incidents = detector.detect(trust_scores, [])
        assert len(incidents) == 1
        assert incidents[0].level == "RED"
        assert incidents[0].category == "TRUST"

    def test_multiple_degraded_subsystems(self):
        detector = TrustCollapseDetector()
        trust_scores = {
            "nav": 0.50,
            "steer": 0.45,
            "engine": 0.40,
            "payload": 0.55,
        }
        incidents = detector.detect(trust_scores, [])
        levels = [inc.level for inc in incidents]
        # Should have YELLOW per-subsystem + ORANGE for multiple degraded
        assert "YELLOW" in levels
        assert "ORANGE" in levels

    def test_mixed_trust_levels(self):
        detector = TrustCollapseDetector()
        trust_scores = {
            "nav": 0.90,
            "engine": 0.10,  # RED
            "steer": 0.30,  # ORANGE
        }
        incidents = detector.detect(trust_scores, [])
        levels = [inc.level for inc in incidents]
        assert "RED" in levels
        assert "ORANGE" in levels

    def test_empty_trust_scores(self):
        detector = TrustCollapseDetector()
        incidents = detector.detect({}, [])
        assert len(incidents) == 0

    def test_non_numeric_trust_scores(self):
        detector = TrustCollapseDetector()
        trust_scores = {"nav": "bad", "engine": None}
        incidents = detector.detect(trust_scores, [])
        assert len(incidents) == 0


# ═══════════════════════════════════════════════════════════════════
# 4. CommunicationLossDetector Tests
# ═══════════════════════════════════════════════════════════════════

class TestCommunicationLossDetector:
    """Tests for the CommunicationLossDetector."""

    def test_recent_comm(self):
        detector = CommunicationLossDetector()
        incidents = detector.detect(time.time(), DEFAULT_THRESHOLDS)
        assert len(incidents) == 0

    def test_comm_timeout(self):
        detector = CommunicationLossDetector()
        last_comm = time.time() - 180  # 3 minutes ago
        incidents = detector.detect(last_comm, DEFAULT_THRESHOLDS)
        assert len(incidents) == 1
        assert incidents[0].level == "ORANGE"
        assert incidents[0].category == "COMMUNICATION"
        assert "timeout" in incidents[0].description.lower()

    def test_comm_dead(self):
        detector = CommunicationLossDetector()
        last_comm = time.time() - 400  # ~7 minutes ago
        incidents = detector.detect(last_comm, DEFAULT_THRESHOLDS)
        assert len(incidents) == 1
        assert incidents[0].level == "RED"
        assert "dead" in incidents[0].description.lower()

    def test_custom_thresholds(self):
        detector = CommunicationLossDetector()
        last_comm = time.time() - 30
        # With very short timeout, 30s should trigger ORANGE
        config = {"comm_timeout_seconds": 20.0, "comm_dead_seconds": 60.0}
        incidents = detector.detect(last_comm, config)
        assert len(incidents) == 1
        assert incidents[0].level == "ORANGE"


# ═══════════════════════════════════════════════════════════════════
# 5. SafetyViolationDetector Tests
# ═══════════════════════════════════════════════════════════════════

class TestSafetyViolationDetector:
    """Tests for the SafetyViolationDetector."""

    def test_nominal_state(self):
        detector = SafetyViolationDetector()
        incidents = detector.detect({"state": "NOMINAL"})
        assert len(incidents) == 0

    def test_empty_state(self):
        detector = SafetyViolationDetector()
        incidents = detector.detect({})
        assert len(incidents) == 0

    def test_e_stop_triggered(self):
        detector = SafetyViolationDetector()
        incidents = detector.detect({"state": "NOMINAL", "e_stop": True})
        assert len(incidents) == 1
        assert incidents[0].level == "RED"
        assert incidents[0].category == "SAFETY"
        assert "e-stop" in incidents[0].description.lower()

    def test_fault_state(self):
        detector = SafetyViolationDetector()
        incidents = detector.detect({"state": "FAULT"})
        assert len(incidents) == 1
        assert incidents[0].level == "RED"
        assert "FAULT" in incidents[0].description

    def test_critical_state(self):
        detector = SafetyViolationDetector()
        incidents = detector.detect({"state": "CRITICAL"})
        assert len(incidents) == 1
        assert incidents[0].level == "RED"

    def test_warning_state(self):
        detector = SafetyViolationDetector()
        incidents = detector.detect({
            "state": "WARNING",
            "violation_type": "actuator_overrange",
        })
        assert len(incidents) == 1
        assert incidents[0].level == "ORANGE"
        assert "actuator_overrange" in incidents[0].description

    def test_watchdog_triggered(self):
        detector = SafetyViolationDetector()
        incidents = detector.detect({
            "state": "NOMINAL",
            "watchdog_triggered": True,
        })
        assert len(incidents) == 1
        assert incidents[0].level == "RED"
        assert "watchdog" in incidents[0].description.lower()

    def test_e_stop_no_double_count(self):
        detector = SafetyViolationDetector()
        # E_STOP state + e_stop flag should produce only 1 RED incident
        incidents = detector.detect({"state": "E_STOP", "e_stop": True})
        assert len(incidents) == 1


# ═══════════════════════════════════════════════════════════════════
# 6. MissionTimeoutDetector Tests
# ═══════════════════════════════════════════════════════════════════

class TestMissionTimeoutDetector:
    """Tests for the MissionTimeoutDetector."""

    def test_mission_on_time(self):
        detector = MissionTimeoutDetector()
        mission_start = time.time() - 60
        expected_duration = 300.0
        incidents = detector.detect(mission_start, expected_duration)
        assert len(incidents) == 0

    def test_mission_timeout(self):
        detector = MissionTimeoutDetector()
        mission_start = time.time() - 600  # 10 minutes ago
        expected_duration = 300.0  # Expected 5 minutes
        incidents = detector.detect(mission_start, expected_duration)
        assert len(incidents) == 1
        assert incidents[0].level == "ORANGE"
        assert incidents[0].category == "MISSION"
        assert "timeout" in incidents[0].description.lower()

    def test_mission_just_over_expected(self):
        detector = MissionTimeoutDetector()
        mission_start = time.time() - 350  # Slightly over 5 minutes
        expected_duration = 300.0
        incidents = detector.detect(mission_start, expected_duration)
        # 350 < 300 * 1.5 = 450, so should not trigger
        assert len(incidents) == 0

    def test_infinite_expected_duration(self):
        detector = MissionTimeoutDetector()
        incidents = detector.detect(
            time.time() - 10000, float("inf")
        )
        assert len(incidents) == 0

    def test_zero_expected_duration(self):
        detector = MissionTimeoutDetector()
        incidents = detector.detect(time.time() - 100, 0)
        assert len(incidents) == 0


# ═══════════════════════════════════════════════════════════════════
# 7. EmergencyResponder Tests
# ═══════════════════════════════════════════════════════════════════

class TestEmergencyResponder:
    """Tests for the EmergencyResponder."""

    def test_respond_yellow(self, sample_incident, responder):
        result = responder.respond_yellow(sample_incident)
        assert result["level"] == "YELLOW"
        assert "log_incident" in result["actions"]
        assert "increase_monitoring" in result["actions"]
        assert result["incident_id"] == sample_incident.id

    def test_respond_yellow_with_bridge(self, sample_incident):
        mock_bridge = MagicMock()
        responder = EmergencyResponder(vessel_id="test", bridge=mock_bridge)
        result = responder.respond_yellow(sample_incident)
        mock_bridge.record_trust_event.assert_called_once()
        assert "trust_event_recorded" in result["actions"]

    def test_respond_orange(self, now):
        incident = Incident(
            id="INC-ORANGE-001",
            level="ORANGE",
            category="TRUST",
            description="Trust degradation",
            timestamp=now,
        )
        responder = EmergencyResponder(vessel_id="test")
        result = responder.respond_orange(incident)
        assert result["level"] == "ORANGE"
        assert "alert_captain" in result["actions"]
        assert "reduce_autonomy" in result["actions"]
        assert "prepare_contingency" in result["actions"]

    def test_respond_orange_with_bridge(self, now):
        incident = Incident(
            id="INC-ORANGE-002",
            level="ORANGE",
            category="TRUST",
            description="Trust degradation",
            timestamp=now,
        )
        mock_bridge = MagicMock()
        mock_bridge.record_trust_event.return_value = MagicMock(committed=True)
        mock_bridge.report_safety_event.return_value = MagicMock(committed=True)
        responder = EmergencyResponder(vessel_id="test", bridge=mock_bridge)
        result = responder.respond_orange(incident)
        assert result["trust_reduced"] is True
        assert result["fleet_notified"] is True

    def test_respond_red(self, sample_red_incident, responder):
        result = responder.respond_red(sample_red_incident)
        assert result["level"] == "RED"
        assert "halt_autonomous_ops" in result["actions"]
        assert "set_actuators_safe" in result["actions"]
        assert "create_github_issue" in result["actions"]
        assert "reduce_trust" in result["actions"]
        assert "notify_fleet" in result["actions"]
        assert "start_watchdog" in result["actions"]

    def test_respond_red_with_bridge(self, sample_red_incident):
        mock_bridge = MagicMock()
        mock_bridge.report_safety_event.return_value = MagicMock(
            committed=True,
            commit_hash="abc123",
            issue_created=True,
            issue_number=42,
        )
        mock_bridge.record_trust_event.return_value = MagicMock(committed=True)
        responder = EmergencyResponder(vessel_id="test", bridge=mock_bridge)
        result = responder.respond_red(sample_red_incident)
        assert result["commit_hash"] == "abc123"
        assert result["issue_created"] is True
        assert result["issue_number"] == 42
        assert result["trust_reduced"] is True
        assert result["fleet_notified"] is True

    def test_create_fleet_alert(self, sample_incident, responder):
        alert = responder.create_fleet_alert(sample_incident)
        assert alert["alert_id"].startswith("FALERT-")
        assert alert["vessel_id"] == "test-vessel"
        assert alert["incident_id"] == sample_incident.id
        assert alert["incident_level"] == "YELLOW"
        assert alert["requires_fleet_action"] is False
        assert "recommended_actions" in alert

    def test_create_fleet_alert_red(self, sample_red_incident, responder):
        alert = responder.create_fleet_alert(sample_red_incident)
        assert alert["requires_fleet_action"] is True
        assert alert["autonomy_level"] == "L0"
        actions = alert["recommended_actions"]
        assert "dispatch_assessment_team" in actions

    def test_create_fleet_alert_orange(self, now):
        incident = Incident(
            id="INC-O-001", level="ORANGE", category="TRUST",
            description="Trust issue", timestamp=now,
        )
        responder = EmergencyResponder(vessel_id="test")
        alert = responder.create_fleet_alert(incident)
        assert alert["requires_fleet_action"] is True
        assert alert["autonomy_level"] == "L1"


# ═══════════════════════════════════════════════════════════════════
# 8. EmergencyProtocol Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestEmergencyProtocol:
    """Integration tests for the main EmergencyProtocol class."""

    def test_initial_state(self, protocol):
        assert protocol.current_level == "GREEN"
        assert protocol.incident_count == 0
        assert len(protocol.incident_history) == 0
        assert protocol.monitoring_multiplier == 1.0

    def test_assess_healthy_state(
        self, protocol, healthy_vessel_state, healthy_trust_scores,
        healthy_sensor_status,
    ):
        assessment = protocol.assess(
            healthy_vessel_state, healthy_trust_scores, healthy_sensor_status
        )
        assert assessment.current_level == "GREEN"
        assert assessment.vessel_id == "test-vessel"
        assert len(assessment.incidents_detected) == 0
        assert "nominal" in assessment.reason.lower()

    def test_assess_sensor_failure(self, protocol, now):
        vessel_state = {
            "safety_state": {"state": "NOMINAL"},
            "last_comm_time": now,
        }
        trust_scores = {"nav": 0.90}
        sensor_status = {
            "sensors": [
                {"id": "gps", "value": None, "timestamp": now - 1},
            ],
        }
        assessment = protocol.assess(vessel_state, trust_scores, sensor_status)
        assert assessment.current_level == "ORANGE"
        assert len(assessment.incidents_detected) >= 1
        sensor_incidents = [
            i for i in assessment.incidents_detected
            if i.category == "SENSOR"
        ]
        assert len(sensor_incidents) >= 1

    def test_assess_trust_collapse(self, protocol, now):
        vessel_state = {
            "safety_state": {"state": "NOMINAL"},
            "last_comm_time": now,
        }
        trust_scores = {"steering": 0.10}  # RED threshold
        sensor_status = {"sensors": []}
        assessment = protocol.assess(vessel_state, trust_scores, sensor_status)
        assert assessment.current_level == "RED"
        assert assessment.escalated is True

    def test_assess_communication_loss(self, protocol, now):
        vessel_state = {
            "safety_state": {"state": "NOMINAL"},
            "last_comm_time": now - 400,  # Dead comms
        }
        trust_scores = {"nav": 0.90}
        sensor_status = {"sensors": []}
        assessment = protocol.assess(vessel_state, trust_scores, sensor_status)
        assert assessment.current_level == "RED"
        comm_incidents = [
            i for i in assessment.incidents_detected
            if i.category == "COMMUNICATION"
        ]
        assert len(comm_incidents) >= 1

    def test_assess_safety_violation(self, protocol, now):
        vessel_state = {
            "safety_state": {"state": "FAULT"},
            "last_comm_time": now,
        }
        trust_scores = {"nav": 0.90}
        sensor_status = {"sensors": []}
        assessment = protocol.assess(vessel_state, trust_scores, sensor_status)
        assert assessment.current_level == "RED"
        safety_incidents = [
            i for i in assessment.incidents_detected
            if i.category == "SAFETY"
        ]
        assert len(safety_incidents) >= 1

    def test_assess_mission_timeout(self, protocol, now):
        vessel_state = {
            "safety_state": {"state": "NOMINAL"},
            "last_comm_time": now,
            "mission_start": now - 600,
            "expected_duration": 300.0,
        }
        trust_scores = {"nav": 0.90}
        sensor_status = {"sensors": []}
        assessment = protocol.assess(vessel_state, trust_scores, sensor_status)
        assert assessment.current_level == "ORANGE"
        mission_incidents = [
            i for i in assessment.incidents_detected
            if i.category == "MISSION"
        ]
        assert len(mission_incidents) >= 1

    def test_monitoring_multiplier_increases(self, protocol, now):
        """Monitoring multiplier should increase during emergencies."""
        vessel_state = {
            "safety_state": {"state": "WARNING"},
            "last_comm_time": now,
        }
        trust_scores = {"nav": 0.55}  # YELLOW
        sensor_status = {"sensors": []}
        protocol.assess(vessel_state, trust_scores, sensor_status)
        assert protocol.monitoring_multiplier >= 2.0

    def test_monitoring_multiplier_red(self, protocol, now):
        """RED level should set monitoring multiplier to 4x."""
        vessel_state = {
            "safety_state": {"state": "FAULT"},
            "last_comm_time": now,
        }
        trust_scores = {"nav": 0.90}
        sensor_status = {"sensors": []}
        protocol.assess(vessel_state, trust_scores, sensor_status)
        assert protocol.monitoring_multiplier == 4.0

    def test_monitoring_multiplier_resets_on_green(self, protocol, now):
        """GREEN level should reset monitoring multiplier to 1x."""
        # First induce an emergency
        vessel_state_bad = {
            "safety_state": {"state": "WARNING"},
            "last_comm_time": now,
        }
        protocol.assess(vessel_state_bad, {"nav": 0.50}, {"sensors": []})
        assert protocol.monitoring_multiplier > 1.0

        # Deescalate all incidents
        for inc in list(protocol.incident_history):
            if inc.resolution is None:
                protocol.deescalate(inc.id, "resolved")

        # Re-assess with healthy state — all incidents resolved, should go GREEN
        assessment = protocol.assess(
            {
                "safety_state": {"state": "NOMINAL"},
                "last_comm_time": now,
            },
            {"nav": 0.90},
            {"sensors": []},
        )
        assert assessment.current_level == "GREEN"
        assert protocol.monitoring_multiplier == 1.0

    def test_incidents_accumulate(self, protocol, now):
        """Multiple assessments should accumulate incidents."""
        vessel_state = {
            "safety_state": {"state": "WARNING"},
            "last_comm_time": now,
        }
        protocol.assess(vessel_state, {"nav": 0.50}, {"sensors": []})
        count_after_first = protocol.incident_count
        protocol.assess(vessel_state, {"nav": 0.50}, {"sensors": []})
        assert protocol.incident_count > count_after_first


class TestEmergencyProtocolEscalation:
    """Tests for incident escalation."""

    def test_escalate_yellow(self, protocol, sample_incident):
        result = protocol.escalate(sample_incident)
        assert isinstance(result, EscalationResult)
        assert result.incident_id == sample_incident.id
        assert result.level == "YELLOW"
        assert len(sample_incident.auto_actions_taken) > 0

    def test_escalate_orange(self, protocol, now):
        incident = Incident(
            id="INC-ESC-O", level="ORANGE", category="TRUST",
            description="Trust degradation", timestamp=now,
        )
        result = protocol.escalate(incident)
        assert isinstance(result, EscalationResult)
        assert result.level == "ORANGE"
        assert len(incident.auto_actions_taken) > 0

    def test_escalate_red(self, protocol, sample_red_incident):
        result = protocol.escalate(sample_red_incident)
        assert isinstance(result, EscalationResult)
        assert result.level == "RED"
        assert len(sample_red_incident.auto_actions_taken) > 0
        assert "halt_autonomous_ops" in sample_red_incident.auto_actions_taken

    def test_escalate_red_with_bridge(self, sample_red_incident):
        mock_bridge = MagicMock()
        mock_bridge.report_safety_event.return_value = MagicMock(
            committed=True, commit_hash="def456",
            issue_created=False, issue_number=0,
        )
        mock_bridge.record_trust_event.return_value = MagicMock(committed=True)
        protocol = EmergencyProtocol(
            vessel_id="test", bridge=mock_bridge
        )
        result = protocol.escalate(sample_red_incident)
        assert result.commit_hash == "def456"
        assert result.issue_created is False


class TestEmergencyProtocolDeescalation:
    """Tests for incident de-escalation."""

    def test_deescalate_unknown_incident(self, protocol):
        result = protocol.deescalate("INC-NONEXISTENT", "fixed")
        assert isinstance(result, DeescalationResult)
        assert "not found" in result.error.lower()

    def test_deescalate_resolves_incident(self, protocol, now):
        # First create an incident
        vessel_state = {
            "safety_state": {"state": "WARNING"},
            "last_comm_time": now,
        }
        assessment = protocol.assess(vessel_state, {"nav": 0.50}, {"sensors": []})
        assert len(assessment.incidents_detected) >= 1

        # Resolve the first incident
        incident = assessment.incidents_detected[0]
        result = protocol.deescalate(incident.id, "Sensor recalibrated")
        assert isinstance(result, DeescalationResult)
        assert result.incident_id == incident.id
        assert incident.resolution == "Sensor recalibrated"
        assert incident.resolved_at is not None

    def test_deescalate_to_green(self, protocol, now):
        """Resolving all incidents should return to GREEN."""
        # Create incidents
        vessel_state = {
            "safety_state": {"state": "WARNING"},
            "last_comm_time": now,
        }
        protocol.assess(vessel_state, {"nav": 0.50}, {"sensors": []})
        assert protocol.current_level != "GREEN"

        # Resolve all incidents
        for inc in list(protocol.incident_history):
            if inc.resolution is None:
                protocol.deescalate(inc.id, "resolved")

        # Should be back to GREEN since all incidents resolved
        assert protocol.current_level == "GREEN"

    def test_deescalate_partial(self, protocol, now):
        """Resolving some incidents should reduce level but not to GREEN."""
        # Create multiple incidents
        vessel_state = {
            "safety_state": {"state": "WARNING"},
            "last_comm_time": now,
        }
        protocol.assess(vessel_state, {"nav": 0.50}, {"sensors": []})
        protocol.assess(vessel_state, {"nav": 0.40}, {"sensors": []})

        unresolved_before = sum(
            1 for inc in protocol.incident_history if inc.resolution is None
        )
        assert unresolved_before > 1

        # Resolve one
        unresolved_incidents = [
            inc for inc in protocol.incident_history if inc.resolution is None
        ]
        protocol.deescalate(unresolved_incidents[0].id, "partial fix")

        # Should still be elevated (not GREEN)
        # Level should be from remaining unresolved incidents
        assert protocol.current_level != "GREEN"


class TestEmergencyProtocolReport:
    """Tests for incident report generation."""

    def test_empty_report(self, protocol):
        report = protocol.get_incident_report()
        assert report["vessel_id"] == "test-vessel"
        assert report["current_level"] == "GREEN"
        assert report["incident_count"] == 0
        assert report["incidents"] == []

    def test_report_with_incidents(self, protocol, now):
        vessel_state = {
            "safety_state": {"state": "WARNING"},
            "last_comm_time": now,
        }
        protocol.assess(vessel_state, {"nav": 0.50}, {"sensors": []})

        report = protocol.get_incident_report()
        assert report["incident_count"] > 0
        assert len(report["incidents"]) > 0

        # Check incident fields
        incident_report = report["incidents"][0]
        assert "id" in incident_report
        assert "level" in incident_report
        assert "category" in incident_report
        assert "description" in incident_report
        assert "timestamp" in incident_report
        assert "timestamp_iso" in incident_report
        assert "resolved" in incident_report

    def test_report_resolved_incidents(self, protocol, now):
        vessel_state = {
            "safety_state": {"state": "WARNING"},
            "last_comm_time": now,
        }
        assessment = protocol.assess(vessel_state, {"nav": 0.50}, {"sensors": []})
        incident = assessment.incidents_detected[0]
        protocol.deescalate(incident.id, "fixed")

        report = protocol.get_incident_report()
        resolved = [i for i in report["incidents"] if i["resolved"]]
        assert len(resolved) >= 1
        assert resolved[0]["resolution"] == "fixed"


class TestDefaultThresholds:
    """Tests for default threshold values."""

    def test_trust_thresholds(self):
        assert DEFAULT_THRESHOLDS["trust_yellow"] > DEFAULT_THRESHOLDS["trust_orange"]
        assert DEFAULT_THRESHOLDS["trust_orange"] > DEFAULT_THRESHOLDS["trust_red"]

    def test_comm_thresholds(self):
        assert DEFAULT_THRESHOLDS["comm_dead_seconds"] > DEFAULT_THRESHOLDS["comm_timeout_seconds"]

    def test_custom_thresholds(self, protocol):
        custom = {"trust_yellow": 0.80, "trust_red": 0.30}
        protocol2 = EmergencyProtocol(vessel_id="v2", thresholds=custom)
        assert protocol2.thresholds["trust_yellow"] == 0.80
        assert protocol2.thresholds["trust_red"] == 0.30
        # Other thresholds should still have defaults
        assert "comm_timeout_seconds" in protocol2.thresholds
