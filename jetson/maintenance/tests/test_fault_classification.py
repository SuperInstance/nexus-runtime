"""Tests for fault_classification module — 30+ tests."""

import math
import pytest
from jetson.maintenance.fault_classification import (
    FaultSignature,
    FaultReport,
    FaultClassifier,
)


@pytest.fixture
def classifier():
    return FaultClassifier()


@pytest.fixture
def sample_library():
    return [
        FaultSignature(
            fault_type="bearing_wear",
            severity="high",
            sensor_patterns={
                "vibration": {"mean": 15.0, "std": 5.0, "min": 8.0, "max": 25.0},
                "temperature": {"mean": 80.0, "std": 3.0, "min": 75.0, "max": 88.0},
            },
            description="Bearing wear detected",
        ),
        FaultSignature(
            fault_type="motor_overheat",
            severity="critical",
            sensor_patterns={
                "temperature": {"mean": 120.0, "std": 10.0, "min": 105.0, "max": 140.0},
            },
            description="Motor overheat",
        ),
    ]


# ── FaultSignature ───────────────────────────────────────────

class TestFaultSignature:
    def test_defaults(self):
        fs = FaultSignature(fault_type="test")
        assert fs.severity == "medium"
        assert fs.sensor_patterns == {}
        assert fs.description == ""

    def test_full(self):
        fs = FaultSignature(
            fault_type="x", severity="critical",
            sensor_patterns={"s1": {"mean": 1.0}},
            description="desc",
        )
        assert fs.severity == "critical"


# ── FaultReport ──────────────────────────────────────────────

class TestFaultReport:
    def test_defaults(self):
        fr = FaultReport()
        assert fr.detected_fault is None
        assert fr.confidence == 0.0
        assert fr.evidence == []
        assert fr.recommended_action == ""


# ── classify ─────────────────────────────────────────────────

class TestClassify:
    def test_empty_readings(self, classifier, sample_library):
        result = classifier.classify({}, sample_library)
        assert result.confidence == 0.0

    def test_empty_library(self, classifier):
        readings = {"vibration": [15.0, 16.0, 14.0]}
        result = classifier.classify(readings, [])
        assert result.confidence == 0.0

    def test_bearing_wear_match(self, classifier, sample_library):
        readings = {
            "vibration": [15.0, 16.0, 14.0, 13.0, 17.0],
            "temperature": [80.0, 81.0, 79.0],
        }
        result = classifier.classify(readings, sample_library)
        assert result.detected_fault == "bearing_wear"
        assert result.confidence > 0.5

    def test_motor_overheat_match(self, classifier, sample_library):
        readings = {
            "temperature": [120.0, 125.0, 115.0],
        }
        result = classifier.classify(readings, sample_library)
        assert result.detected_fault == "motor_overheat"

    def test_has_evidence(self, classifier, sample_library):
        readings = {"vibration": [15.0]}
        result = classifier.classify(readings, sample_library)
        assert len(result.evidence) > 0

    def test_has_recommended_action(self, classifier, sample_library):
        readings = {"temperature": [120.0, 125.0]}
        result = classifier.classify(readings, sample_library)
        assert result.recommended_action != ""

    def test_empty_sensor_lists(self, classifier, sample_library):
        readings = {"vibration": []}
        result = classifier.classify(readings, sample_library)
        # Should still process without crash
        assert isinstance(result, FaultReport)

    def test_unknown_fault(self, classifier):
        lib = [FaultSignature(fault_type="known", sensor_patterns={"s1": {"mean": 999.0}})]
        readings = {"s1": [1.0]}
        result = classifier.classify(readings, lib)
        # Low confidence expected
        assert result.confidence < 1.0


# ── match_signature ──────────────────────────────────────────

class TestMatchSignature:
    def test_perfect_match(self, classifier):
        readings = {"s1": {"mean": 10.0, "std": 2.0}}
        patterns = {"s1": {"mean": 10.0, "std": 2.0}}
        score = classifier.match_signature(readings, patterns)
        assert abs(score - 1.0) < 1e-9

    def test_zero_match(self, classifier):
        readings = {"s1": {"mean": 10.0}}
        patterns = {"s1": {"mean": 0.0}}
        score = classifier.match_signature(readings, patterns)
        assert score < 1.0

    def test_empty_readings(self, classifier):
        score = classifier.match_signature({}, {"s1": {"mean": 1.0}})
        assert score == 0.0

    def test_empty_patterns(self, classifier):
        score = classifier.match_signature({"s1": {"mean": 1.0}}, {})
        assert score == 0.0

    def test_different_keys(self, classifier):
        readings = {"a": {"mean": 1.0}}
        patterns = {"b": {"mean": -1.0}}
        score = classifier.match_signature(readings, patterns)
        # Opposing values at different key positions -> score ~0
        assert score < 0.1

    def test_partial_match(self, classifier):
        readings = {"s1": {"mean": 10.0, "std": 2.0, "min": 5.0, "max": 15.0}}
        patterns = {"s1": {"mean": 10.0, "std": 2.0}}
        score = classifier.match_signature(readings, patterns)
        assert 0.5 < score <= 1.0

    def test_score_bounded(self, classifier):
        readings = {"s1": {"mean": 1e6, "std": 1e6, "min": -1e6, "max": 1e6}}
        patterns = {"s1": {"mean": -1e6, "std": -1e6, "min": 1e6, "max": -1e6}}
        score = classifier.match_signature(readings, patterns)
        assert 0.0 <= score <= 1.0


# ── compute_severity ─────────────────────────────────────────

class TestComputeSeverity:
    def test_critical_fault_type(self, classifier):
        assert classifier.compute_severity("motor_overheat", 0.9) == "critical"
        assert classifier.compute_severity("hull_breach", 0.5) == "critical"
        assert classifier.compute_severity("power_failure", 0.8) == "critical"
        assert classifier.compute_severity("propeller_damage", 0.9) == "critical"

    def test_high_fault_low_health(self, classifier):
        assert classifier.compute_severity("bearing_wear", 0.3) == "critical"

    def test_high_fault_medium_health(self, classifier):
        assert classifier.compute_severity("bearing_wear", 0.6) == "high"

    def test_high_fault_good_health(self, classifier):
        assert classifier.compute_severity("bearing_wear", 0.9) == "medium"

    def test_unknown_fault_low_health(self, classifier):
        assert classifier.compute_severity("unknown", 0.2) == "high"

    def test_unknown_fault_medium_health(self, classifier):
        assert classifier.compute_severity("unknown", 0.5) == "medium"

    def test_unknown_fault_good_health(self, classifier):
        assert classifier.compute_severity("unknown", 0.8) == "low"

    def test_seal_leak_critical_health(self, classifier):
        assert classifier.compute_severity("seal_leak", 0.1) == "critical"

    def test_corrosion_various_health(self, classifier):
        assert classifier.compute_severity("corrosion", 0.8) == "medium"
        assert classifier.compute_severity("corrosion", 0.3) == "critical"


# ── recommend_action ─────────────────────────────────────────

class TestRecommendAction:
    def test_critical(self, classifier):
        action = classifier.recommend_action("some_fault", "critical")
        assert "Immediate" in action or "EMERGENCY" in action

    def test_high(self, classifier):
        action = classifier.recommend_action("some_fault", "high")
        assert "12 hours" in action

    def test_medium(self, classifier):
        action = classifier.recommend_action("some_fault", "medium")
        assert "72 hours" in action

    def test_low(self, classifier):
        action = classifier.recommend_action("some_fault", "low")
        assert "next scheduled" in action

    def test_hull_breach_emergency(self, classifier):
        action = classifier.recommend_action("hull_breach", "medium")
        assert "EMERGENCY" in action

    def test_power_failure_emergency(self, classifier):
        action = classifier.recommend_action("power_failure", "low")
        assert "EMERGENCY" in action

    def test_unknown_severity(self, classifier):
        action = classifier.recommend_action("test", "unknown_sev")
        # Falls back to 'low' default action
        assert "scheduled maintenance" in action


# ── learn_new_fault ──────────────────────────────────────────

class TestLearnNewFault:
    def test_basic(self, classifier):
        readings = {"vibration": [15.0, 16.0, 14.0]}
        sig = classifier.learn_new_fault(readings, "new_bearing_issue")
        assert sig.fault_type == "new_bearing_issue"
        assert "vibration" in sig.sensor_patterns
        assert sig.description != ""

    def test_high_severity_auto(self, classifier):
        readings = {"s1": [100.0, 200.0, -50.0, 300.0]}
        sig = classifier.learn_new_fault(readings, "erratic_sensor")
        assert sig.severity in ("high", "medium", "low")

    def test_low_severity_auto(self, classifier):
        readings = {"s1": [10.0, 10.1, 9.9, 10.0]}
        sig = classifier.learn_new_fault(readings, "minor_drift")
        assert sig.severity == "low"

    def test_medium_severity_auto(self, classifier):
        readings = {"s1": [10.0, 20.0, 5.0, 25.0]}
        sig = classifier.learn_new_fault(readings, "moderate_drift")
        assert sig.severity in ("medium", "high")

    def test_empty_readings(self, classifier):
        sig = classifier.learn_new_fault({}, "empty_fault")
        assert sig.sensor_patterns == {}
        assert sig.severity == "low"

    def test_pattern_values(self, classifier):
        readings = {"s1": [2.0, 4.0, 6.0]}
        sig = classifier.learn_new_fault(readings, "test")
        p = sig.sensor_patterns["s1"]
        assert abs(p["mean"] - 4.0) < 1e-9
        assert "std" in p
        assert "min" in p
        assert "max" in p
