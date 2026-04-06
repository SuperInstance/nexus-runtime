"""Tests for mission monitoring module."""

import time
import pytest
from jetson.mission.planner import MissionPlan, MissionPhase, MissionObjective
from jetson.mission.monitoring import (
    AlertLevel,
    TrendDirection,
    ProgressMetric,
    MissionAlert,
    MissionStatus,
    DeviationReport,
    StatusReport,
    ResourceWarning,
    MissionMonitor,
)


class TestAlertLevel:
    def test_info(self):
        assert AlertLevel.INFO.value == "info"

    def test_warning(self):
        assert AlertLevel.WARNING.value == "warning"

    def test_critical(self):
        assert AlertLevel.CRITICAL.value == "critical"

    def test_all_levels(self):
        values = {a.value for a in AlertLevel}
        assert values == {"info", "warning", "critical"}


class TestTrendDirection:
    def test_improving(self):
        assert TrendDirection.IMPROVING.value == "improving"

    def test_stable(self):
        assert TrendDirection.STABLE.value == "stable"

    def test_declining(self):
        assert TrendDirection.DECLINING.value == "declining"

    def test_unknown(self):
        assert TrendDirection.UNKNOWN.value == "unknown"


class TestProgressMetric:
    def test_default_creation(self):
        m = ProgressMetric()
        assert m.name == ""
        assert m.value == 0.0
        assert m.target == 100.0
        assert m.trend == TrendDirection.UNKNOWN

    def test_update_value(self):
        m = ProgressMetric(name="depth")
        m.update(50.0)
        assert m.value == 50.0
        assert len(m.history) == 1

    def test_update_increasing(self):
        m = ProgressMetric(name="progress")
        m.update(10.0)
        m.update(20.0)
        assert m.trend == TrendDirection.IMPROVING

    def test_update_decreasing(self):
        m = ProgressMetric(name="battery")
        m.update(90.0)
        m.update(80.0)
        assert m.trend == TrendDirection.DECLINING

    def test_update_stable(self):
        m = ProgressMetric(name="temp")
        m.update(20.0)
        m.update(20.0)
        assert m.trend == TrendDirection.STABLE

    def test_get_progress_percentage(self):
        m = ProgressMetric(value=75.0, target=100.0)
        assert m.get_progress_percentage() == 75.0

    def test_get_progress_percentage_zero_target(self):
        m = ProgressMetric(value=0.0, target=0.0)
        assert m.get_progress_percentage() == 100.0

    def test_get_progress_percentage_capped(self):
        m = ProgressMetric(value=150.0, target=100.0)
        assert m.get_progress_percentage() == 100.0


class TestMissionAlert:
    def test_default_creation(self):
        a = MissionAlert(level=AlertLevel.INFO, message="test")
        assert a.level == AlertLevel.INFO
        assert a.message == "test"
        assert a.acknowledged is False

    def test_custom_creation(self):
        a = MissionAlert(
            level=AlertLevel.CRITICAL, message="danger",
            source="sensor", acknowledged=True,
        )
        assert a.source == "sensor"
        assert a.acknowledged is True


class TestMissionStatus:
    def test_default_creation(self):
        s = MissionStatus()
        assert s.mission_id == ""
        assert s.state == "idle"
        assert s.progress == 0.0

    def test_updated_at(self):
        before = time.time()
        s = MissionStatus()
        after = time.time()
        assert before <= s.updated_at <= after


class TestDeviationReport:
    def test_default_creation(self):
        d = DeviationReport()
        assert d.magnitude == 0.0
        assert d.severity == 0.0

    def test_custom_creation(self):
        d = DeviationReport(
            phase="p1", deviation_type="overrun",
            expected=10.0, actual=15.0, magnitude=0.5, severity=0.5,
        )
        assert d.deviation_type == "overrun"


class TestStatusReport:
    def test_default_creation(self):
        r = StatusReport()
        assert r.mission_id == ""
        assert r.progress == 0.0
        assert r.recommendations == []


class TestResourceWarning:
    def test_default_creation(self):
        w = ResourceWarning()
        assert w.resource == ""
        assert w.percent_used == 0.0

    def test_custom_creation(self):
        w = ResourceWarning(
            resource="battery", current_usage=85.0,
            limit=100.0, percent_used=85.0,
            severity=AlertLevel.WARNING,
        )
        assert w.resource == "battery"
        assert w.severity == AlertLevel.WARNING


class TestMissionMonitor:
    def setup_method(self):
        self.monitor = MissionMonitor()

    def test_register_mission(self):
        status = self.monitor.register_mission("m1")
        assert status.mission_id == "m1"

    def test_register_mission_with_plan(self):
        plan = MissionPlan(
            name="test",
            phases=[MissionPhase(name="p1"), MissionPhase(name="p2")],
        )
        status = self.monitor.register_mission("m1", plan)
        assert status.mission_id == "m1"

    def test_update_progress(self):
        self.monitor.register_mission("m1")
        metric = ProgressMetric(name="coverage", value=50.0, target=100.0)
        status = self.monitor.update_progress("m1", metric)
        assert "coverage" in status.metrics
        assert status.metrics["coverage"].value == 50.0

    def test_update_progress_auto_register(self):
        metric = ProgressMetric(name="speed", value=5.0)
        status = self.monitor.update_progress("m_new", metric)
        assert "speed" in status.metrics

    def test_update_multiple_metrics(self):
        self.monitor.register_mission("m1")
        self.monitor.update_progress("m1", ProgressMetric(name="a", value=80.0))
        self.monitor.update_progress("m1", ProgressMetric(name="b", value=60.0))
        status = self.monitor.get_status("m1")
        assert len(status.metrics) == 2

    def test_check_objectives(self):
        plan = MissionPlan(
            objectives=[
                MissionObjective(id="o1", name="survey", type="survey"),
                MissionObjective(id="o2", name="sample", type="sample"),
            ],
            phases=[MissionPhase(name="p1"), MissionPhase(name="p2")],
        )
        statuses = self.monitor.check_objectives(plan, 50.0)
        assert len(statuses) == 2

    def test_detect_deviations_overrun(self):
        plan = MissionPlan(
            phases=[MissionPhase(name="p1", duration=100.0)],
        )
        actual = {"p1": 130.0}
        devs = self.monitor.detect_deviations(plan, actual)
        assert len(devs) == 1
        assert devs[0].deviation_type == "overrun"

    def test_detect_deviations_underrun(self):
        plan = MissionPlan(
            phases=[MissionPhase(name="p1", duration=100.0)],
        )
        actual = {"p1": 60.0}
        devs = self.monitor.detect_deviations(plan, actual)
        assert len(devs) == 1
        assert devs[0].deviation_type == "underrun"

    def test_detect_deviations_none(self):
        plan = MissionPlan(
            phases=[MissionPhase(name="p1", duration=100.0)],
        )
        devs = self.monitor.detect_deviations(plan, {"p1": 100.0})
        assert len(devs) == 0

    def test_detect_deviations_no_actual(self):
        plan = MissionPlan(phases=[MissionPhase(name="p1", duration=100.0)])
        devs = self.monitor.detect_deviations(plan, {})
        assert len(devs) == 0

    def test_detect_deviations_no_phases(self):
        plan = MissionPlan(phases=[])
        devs = self.monitor.detect_deviations(plan, {"p1": 100.0})
        assert len(devs) == 0

    def test_estimate_completion(self):
        self.monitor.register_mission("m1")
        time.sleep(0.05)
        self.monitor.update_progress("m1", ProgressMetric(name="p", value=50.0))
        eta = self.monitor.estimate_completion("m1")
        assert eta is not None
        assert eta > time.time()

    def test_estimate_completion_unknown(self):
        eta = self.monitor.estimate_completion("nonexistent")
        assert eta is None

    def test_estimate_completion_no_progress(self):
        self.monitor.register_mission("m1")
        eta = self.monitor.estimate_completion("m1")
        assert eta is None

    def test_generate_status_report(self):
        self.monitor.register_mission("m1")
        report = self.monitor.generate_status_report("m1")
        assert report.mission_id == "m1"
        assert report.progress == 0.0

    def test_generate_status_report_unknown(self):
        report = self.monitor.generate_status_report("unknown")
        assert report.state == "unknown"

    def test_generate_status_report_recommendations(self):
        self.monitor.register_mission("m1")
        self.monitor.update_progress("m1", ProgressMetric(name="p", value=10.0))
        report = self.monitor.generate_status_report("m1")
        assert len(report.recommendations) > 0

    def test_set_resource_limit(self):
        self.monitor.set_resource_limit("battery", 100.0)
        warnings = self.monitor.check_resource_status("m1", {"battery": 75.0})
        assert len(warnings) == 1
        assert warnings[0].percent_used == 75.0

    def test_check_resource_critical(self):
        self.monitor.set_resource_limit("energy", 100.0)
        warnings = self.monitor.check_resource_status("m1", {"energy": 95.0})
        assert len(warnings) == 1
        assert warnings[0].severity == AlertLevel.CRITICAL

    def test_check_resource_normal(self):
        self.monitor.set_resource_limit("cpu", 100.0)
        warnings = self.monitor.check_resource_status("m1", {"cpu": 30.0})
        assert len(warnings) == 0

    def test_check_resource_no_limit(self):
        warnings = self.monitor.check_resource_status("m1", {"cpu": 99.0})
        assert len(warnings) == 0

    def test_compute_mission_efficiency(self):
        self.monitor.register_mission("m1")
        self.monitor.update_progress("m1", ProgressMetric(name="p", value=80.0))
        eff = self.monitor.compute_mission_efficiency("m1")
        assert 0.0 <= eff <= 1.0

    def test_compute_mission_efficiency_unknown(self):
        eff = self.monitor.compute_mission_efficiency("nonexistent")
        assert eff == 0.0

    def test_compute_efficiency_with_alerts(self):
        self.monitor.register_mission("m1")
        self.monitor.update_progress("m1", ProgressMetric(name="p", value=80.0))
        self.monitor.add_alert("m1", MissionAlert(AlertLevel.WARNING, "low battery"))
        self.monitor.add_alert("m1", MissionAlert(AlertLevel.WARNING, "high temp"))
        eff = self.monitor.compute_mission_efficiency("m1")
        # Alerts reduce efficiency
        assert eff <= 0.8

    def test_add_alert(self):
        self.monitor.register_mission("m1")
        alert = MissionAlert(AlertLevel.INFO, "test alert")
        self.monitor.add_alert("m1", alert)
        alerts = self.monitor.get_alerts("m1")
        assert len(alerts) == 1

    def test_get_alerts_filtered(self):
        self.monitor.register_mission("m1")
        self.monitor.add_alert("m1", MissionAlert(AlertLevel.INFO, "info"))
        self.monitor.add_alert("m1", MissionAlert(AlertLevel.CRITICAL, "crit"))
        crit = self.monitor.get_alerts("m1", AlertLevel.CRITICAL)
        assert len(crit) == 1

    def test_acknowledge_alert(self):
        self.monitor.register_mission("m1")
        self.monitor.add_alert("m1", MissionAlert(AlertLevel.WARNING, "warn"))
        result = self.monitor.acknowledge_alert("m1", 0)
        assert result is True
        assert self.monitor.get_alerts("m1")[0].acknowledged is True

    def test_acknowledge_alert_invalid(self):
        result = self.monitor.acknowledge_alert("m1", 99)
        assert result is False

    def test_get_status(self):
        self.monitor.register_mission("m1")
        status = self.monitor.get_status("m1")
        assert status is not None
        assert status.mission_id == "m1"

    def test_get_status_unknown(self):
        assert self.monitor.get_status("unknown") is None

    def test_get_metric_history(self):
        self.monitor.register_mission("m1")
        m1 = ProgressMetric(name="temp", value=20.0)
        m2 = ProgressMetric(name="temp", value=21.0)
        self.monitor.update_progress("m1", m1)
        self.monitor.update_progress("m1", m2)
        history = self.monitor.get_metric_history("m1", "temp")
        assert len(history) == 2

    def test_get_metric_history_empty(self):
        history = self.monitor.get_metric_history("m1", "nonexistent")
        assert history == []

    def test_unregister_mission(self):
        self.monitor.register_mission("m1")
        assert self.monitor.unregister_mission("m1") is True
        assert self.monitor.get_status("m1") is None

    def test_unregister_mission_nonexistent(self):
        assert self.monitor.unregister_mission("unknown") is False

    def test_get_all_missions(self):
        self.monitor.register_mission("m1")
        self.monitor.register_mission("m2")
        ids = self.monitor.get_all_missions()
        assert set(ids) == {"m1", "m2"}

    def test_status_report_active_alerts(self):
        self.monitor.register_mission("m1")
        self.monitor.add_alert("m1", MissionAlert(AlertLevel.WARNING, "warn"))
        self.monitor.add_alert("m1", MissionAlert(AlertLevel.CRITICAL, "crit"))
        report = self.monitor.generate_status_report("m1")
        assert report.active_alerts == 2

    def test_status_report_late_stage(self):
        self.monitor.register_mission("m1")
        self.monitor.update_progress("m1", ProgressMetric(name="p", value=90.0))
        report = self.monitor.generate_status_report("m1")
        assert any("completion" in r.lower() for r in report.recommendations)

    def test_multiple_missions_independent(self):
        self.monitor.register_mission("m1")
        self.monitor.register_mission("m2")
        self.monitor.update_progress("m1", ProgressMetric(name="p", value=50.0))
        s1 = self.monitor.get_status("m1")
        s2 = self.monitor.get_status("m2")
        assert s1.progress > s2.progress
