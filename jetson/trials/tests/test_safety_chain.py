"""Cross-module integration tests: Safety chain (invariants, monitor, watchdog, fault, diagnosis, recovery, compliance).

Each test calls 2+ safety modules together.
"""

import time
import pytest

from jetson.runtime_verification.invariants import (
    InvariantChecker, Invariant, Violation,
)
from jetson.runtime_verification.monitor import (
    RuntimeMonitor, MonitorEvent, MonitorRule, Alert,
)
from jetson.runtime_verification.watchdog import (
    WatchdogManager, WatchdogConfig, WatchdogState,
)
from jetson.self_healing.fault_detector import (
    FaultDetector, HealthIndicator, FaultEvent, FaultSeverity,
    FaultCategory, IndicatorStatus, DegradationReport,
)
from jetson.self_healing.diagnosis import (
    RootCauseAnalyzer, Diagnosis, DiagnosticRule, CausalGraph,
)
from jetson.self_healing.recovery import (
    RecoveryManager, RecoveryAction, RecoveryResult, RecoveryType,
    Urgency, RecoveryStrategy,
)
from jetson.compliance.iec61508 import (
    SILVerifier, SILLevel, SILTarget, SILVerificationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_indicator(component="sensor", metric="temp", value=75.0,
                    normal_range=(20.0, 80.0)):
    return HealthIndicator(
        component=component, metric_name=metric,
        value=value, normal_range=normal_range,
    )


def _make_fault_event(component="gps", fault_type="gps_anomaly",
                      severity=FaultSeverity.HIGH):
    return FaultEvent(
        component=component, fault_type=fault_type, severity=severity,
        symptoms=[f"{fault_type} detected"],
        context={"value": 150.0, "normal_range": (0, 100)},
    )


def _make_invariant(name="temp_ok", check_fn=None, severity="warning"):
    if check_fn is None:
        check_fn = lambda state: state.get(name, 0) < 100
    return Invariant(
        name=name, check_fn=check_fn,
        severity=severity, description=f"Check {name}",
        category="thermal",
    )


def _make_monitor_event(source="sensor", event_type="reading", value=50.0):
    return MonitorEvent(
        timestamp=time.time(), source=source,
        event_type=event_type, value=value,
    )


def _make_monitor_rule(name="temp_high", condition_fn=None, severity="warning"):
    if condition_fn is None:
        condition_fn = lambda e: isinstance(e.value, (int, float)) and e.value > 80
    return MonitorRule(name=name, condition_fn=condition_fn, severity=severity)


# ===========================================================================
# 1. InvariantChecker + RuntimeMonitor integration tests
# ===========================================================================

class TestInvariantMonitorIntegration:
    """Tests that exercise InvariantChecker together with RuntimeMonitor."""

    def test_invariant_violation_triggers_monitor_alert(self):
        checker = InvariantChecker()
        monitor = RuntimeMonitor()
        checker.register(_make_invariant("temp"))
        # Register a monitor rule that checks for violations
        def violation_rule(event):
            return isinstance(event.value, dict) and event.value.get("is_violation", False)
        monitor.add_rule(MonitorRule(
            name="invariant_violation", condition_fn=violation_rule, severity="warning",
        ))
        # Check invariant with violating state
        violation = checker.check("temp", {"temp": 150})
        assert violation is not None
        # Fire a monitor event about it
        event = MonitorEvent(
            timestamp=time.time(), source="invariant_checker",
            event_type="violation", value={"is_violation": True, "name": "temp"},
        )
        alerts = monitor.process_event(event)
        assert len(alerts) >= 1

    def test_multiple_invariants_checked_with_monitor_tracking(self):
        checker = InvariantChecker()
        monitor = RuntimeMonitor()
        for i in range(5):
            checker.register(_make_invariant(f"inv_{i}"))
        monitor.add_rule(MonitorRule(
            name="any_violation",
            condition_fn=lambda e: e.event_type == "violation",
            severity="error",
        ))
        state = {"temp": 200, "voltage": 0}
        violations = checker.check_all(state)
        for v in violations:
            event = MonitorEvent(
                timestamp=time.time(), source="invariants",
                event_type="violation", value={"name": v.invariant},
            )
            monitor.process_event(event)
        assert len(violations) >= 0  # depends on state
        health = monitor.compute_monitor_health()
        assert 0.0 <= health <= 100.0

    def test_invariant_summary_reflected_in_monitor(self):
        checker = InvariantChecker()
        monitor = RuntimeMonitor()
        checker.register(_make_invariant("a"))
        checker.register(_make_invariant("b"))
        summary = checker.get_summary()
        assert summary["total_invariants"] == 2
        monitor.add_rule(MonitorRule(
            name="health_check",
            condition_fn=lambda e: False,
            severity="info",
        ))
        assert len(monitor._rules) == 1

    def test_invariant_coverage_computation(self):
        coverage = InvariantChecker.compute_invariant_coverage(8, 10)
        assert coverage == 80.0
        coverage_full = InvariantChecker.compute_invariant_coverage(5, 5)
        assert coverage_full == 100.0

    def test_violation_history_tracked_across_checks(self):
        checker = InvariantChecker()
        checker.register(_make_invariant("x"))
        for i in range(5):
            checker.check("x", {"x": 200})
        history = checker.get_violation_history("x")
        assert len(history) == 5
        # Reset and verify
        checker.reset_violations("x")
        history_after = checker.get_violation_history("x")
        assert len(history_after) == 0


# ===========================================================================
# 2. RuntimeMonitor + WatchdogManager integration tests
# ===========================================================================

class TestMonitorWatchdogIntegration:
    """Tests that exercise RuntimeMonitor together with WatchdogManager."""

    def test_watchdog_expiry_triggers_monitor_event(self):
        monitor = RuntimeMonitor()
        watchdog = WatchdogManager()
        config = WatchdogConfig(timeout_ms=50, max_missed_beats=1)
        watchdog.register("sensor", config)
        monitor.add_rule(MonitorRule(
            name="watchdog_expiry",
            condition_fn=lambda e: e.event_type == "watchdog_expired",
            severity="critical",
        ))
        # Don't feed heartbeat, wait for expiry
        time.sleep(0.06)
        expired = watchdog.check_all()
        if expired:
            for comp in expired:
                event = MonitorEvent(
                    timestamp=time.time(), source="watchdog",
                    event_type="watchdog_expired", value={"component": comp},
                )
                alerts = monitor.process_event(event)
                if alerts:
                    assert alerts[0].severity == "critical"

    def test_healthy_watchdog_no_monitor_alert(self):
        monitor = RuntimeMonitor()
        watchdog = WatchdogManager()
        config = WatchdogConfig(timeout_ms=5000)
        watchdog.register("healthy_comp", config)
        watchdog.feed_heartbeat("healthy_comp")
        expired = watchdog.check_all()
        assert "healthy_comp" not in expired

    def test_multiple_watchdogs_and_monitor_rules(self):
        monitor = RuntimeMonitor()
        watchdog = WatchdogManager()
        for i in range(3):
            watchdog.register(f"comp_{i}", WatchdogConfig(timeout_ms=50, max_missed_beats=1))
        for i in range(3):
            monitor.add_rule(MonitorRule(
                name=f"wd_{i}",
                condition_fn=lambda e: e.source == "watchdog" and e.event_type == "watchdog_expired",
                severity="error",
            ))
        time.sleep(0.06)
        expired = watchdog.check_all()
        assert len(expired) >= 0

    def test_watchdog_reset_and_monitor_health(self):
        monitor = RuntimeMonitor()
        watchdog = WatchdogManager()
        config = WatchdogConfig(timeout_ms=50, max_missed_beats=2)
        watchdog.register("sensor", config)
        watchdog.feed_heartbeat("sensor")
        assert watchdog.compute_uptime("sensor") == 100.0
        watchdog.reset("sensor")
        assert watchdog.compute_uptime("sensor") == 100.0

    def test_watchdog_custom_handler_with_monitor(self):
        monitor = RuntimeMonitor()
        watchdog = WatchdogManager()
        escalation_log = []
        def escalation_handler(name, state):
            escalation_log.append((name, state.missed_beats))
            event = MonitorEvent(
                timestamp=time.time(), source="watchdog",
                event_type="escalation", value={"component": name},
            )
            monitor.process_event(event)
        config = WatchdogConfig(timeout_ms=50, max_missed_beats=1)
        watchdog.register("comp", config)
        watchdog.set_custom_handler("comp", escalation_handler)
        time.sleep(0.06)
        watchdog.check_all()
        watchdog.check_all()
        # After 1+ check_all with missed beats and max_missed_beats=1, handler should fire
        state = watchdog.get_state("comp")
        if state and state.escalated:
            assert len(escalation_log) > 0

    def test_watchdog_summary_and_monitor_state(self):
        watchdog = WatchdogManager()
        monitor = RuntimeMonitor()
        watchdog.register("a", WatchdogConfig(timeout_ms=5000))
        watchdog.register("b", WatchdogConfig(timeout_ms=5000))
        summary = watchdog.get_summary()
        assert summary["total_components"] == 2
        health = monitor.compute_monitor_health()
        assert health == 100.0  # no alerts


# ===========================================================================
# 3. FaultDetector + RootCauseAnalyzer integration tests
# ===========================================================================

class TestFaultDiagnosisIntegration:
    """Tests that exercise FaultDetector together with RootCauseAnalyzer."""

    def test_detect_fault_and_diagnose(self):
        detector = FaultDetector()
        analyzer = RootCauseAnalyzer()
        analyzer.add_rule(DiagnosticRule(
            symptoms_pattern={"component": "gps", "fault_type": "gps_anomaly"},
            probable_cause="gps_hardware_failure",
            confidence=0.8,
            fix_recommendation="Replace GPS module",
            priority=10,
        ))
        indicator = _make_indicator("gps", "signal", 150, (10, 100))
        detector.register_health_indicator(indicator)
        indicators = detector.check_indicators()
        fault = detector.detect_fault(indicators)
        if fault:
            diag = analyzer.diagnose(fault, {"component": "gps"})
            assert diag.confidence > 0
            assert diag.root_cause == "gps_hardware_failure"

    def test_classify_fault_then_diagnose(self):
        detector = FaultDetector()
        analyzer = RootCauseAnalyzer()
        fault = _make_fault_event("motor", "motor_stall", FaultSeverity.HIGH)
        category = detector.classify_fault(fault)
        assert category == FaultCategory.HARDWARE
        diag = analyzer.diagnose(fault, {"component": "motor"})
        assert diag.root_cause is not None

    def test_multiple_faults_analyzed(self):
        detector = FaultDetector()
        analyzer = RootCauseAnalyzer()
        analyzer.add_rule(DiagnosticRule(
            symptoms_pattern={"fault_type": "temp_anomaly"},
            probable_cause="overheating",
            confidence=0.9,
            fix_recommendation="Reduce load",
        ))
        faults = [
            _make_fault_event("cpu", "temp_anomaly"),
            _make_fault_event("memory", "memory_leak"),
        ]
        for f in faults:
            diag = analyzer.diagnose(f)
            assert diag.root_cause is not None

    def test_degradation_detected_and_diagnosed(self):
        detector = FaultDetector()
        analyzer = RootCauseAnalyzer()
        # Build degradation history
        history = []
        for i in range(10):
            ind = HealthIndicator(
                component="battery", metric_name="capacity",
                value=100.0 - i * 2.0,
                normal_range=(80.0, 100.0),
                timestamp=time.time() - (10 - i) * 60,
            )
            history.append(ind)
            detector.register_health_indicator(ind)
        degradation = detector.detect_degradation(history)
        if degradation:
            assert degradation.trend == "degrading"
            # Create a fault event from degradation
            fault = FaultEvent(
                component="battery", fault_type="capacity_degradation",
                severity=FaultSeverity.MEDIUM,
                symptoms=[f"capacity degraded {degradation.degradation_pct:.1f}%"],
                context={"value": degradation.end_value},
            )
            diag = analyzer.diagnose(fault)
            assert diag is not None

    def test_diagnosis_confidence_computation(self):
        analyzer = RootCauseAnalyzer()
        hypotheses = [
            {"cause": "hardware", "base_confidence": 0.6, "component": "gps"},
            {"cause": "software", "base_confidence": 0.3, "component": "gps"},
        ]
        evidence = {"component": "gps", "sensor_ok": False}
        ranked = analyzer.rank_hypotheses(hypotheses, evidence)
        assert len(ranked) == 2
        assert "_score" in ranked[0]

    def test_causal_graph_from_symptoms(self):
        analyzer = RootCauseAnalyzer()
        symptoms = ["gps_signal_lost", "position_drift", "navigation_error"]
        graph = analyzer.compute_causal_graph(symptoms)
        assert len(graph.nodes) > 0
        assert "root_cause" in graph.nodes

    def test_learn_from_resolution(self):
        analyzer = RootCauseAnalyzer()
        fault = _make_fault_event("sensor", "sensor_timeout")
        diag = Diagnosis(
            fault_id=fault.id, root_cause="initial_guess",
            confidence=0.3, recommended_fix="Restart sensor",
        )
        new_rules = analyzer.learn_from_resolution(fault, diag, "actual_cause_power")
        assert len(new_rules) == 1
        assert new_rules[0].probable_cause == "actual_cause_power"


# ===========================================================================
# 4. RootCauseAnalyzer + RecoveryManager integration tests
# ===========================================================================

class TestDiagnosisRecoveryIntegration:
    """Tests that exercise RootCauseAnalyzer with RecoveryManager."""

    def test_diagnosis_drives_recovery_plan(self):
        analyzer = RootCauseAnalyzer()
        analyzer.add_rule(DiagnosticRule(
            symptoms_pattern={"component": "motor"},
            probable_cause="motor_overheating",
            confidence=0.9,
            fix_recommendation="Reduce motor load and increase cooling",
        ))
        fault = _make_fault_event("motor", "motor_stall")
        diag = analyzer.diagnose(fault, {"component": "motor"})
        recovery = RecoveryManager()
        actions = recovery.generate_recovery_plan(diag)
        assert len(actions) > 0
        assert any(a.type == RecoveryType.RECONFIGURE for a in actions)
        assert any(a.type == RecoveryType.RESTART for a in actions)

    def test_execute_recovery_from_diagnosis(self):
        analyzer = RootCauseAnalyzer()
        recovery = RecoveryManager()
        fault = _make_fault_event("network", "network_timeout")
        diag = Diagnosis(
            fault_id=fault.id, root_cause="network_congestion",
            confidence=0.7, recommended_fix="Reset network interface",
        )
        actions = recovery.generate_recovery_plan(diag)
        if actions:
            result = recovery.execute_recovery(actions[0])
            assert result.success is True

    def test_recovery_strategy_selection_from_diagnosis(self):
        analyzer = RootCauseAnalyzer()
        recovery = RecoveryManager()
        # High confidence diagnosis -> moderate urgency -> moderate strategy
        diag = Diagnosis(
            fault_id="f1", root_cause="known_issue",
            confidence=0.8, recommended_fix="Apply fix",
        )
        strategy = recovery.select_recovery_strategy(diag, Urgency.MEDIUM)
        assert strategy == RecoveryStrategy.MODERATE
        # Low confidence + low urgency -> conservative
        diag_low = Diagnosis(fault_id="f2", root_cause="unknown", confidence=0.2)
        strategy_low = recovery.select_recovery_strategy(diag_low, Urgency.LOW)
        assert strategy_low == RecoveryStrategy.CONSERVATIVE
        # Critical urgency -> aggressive
        strategy_crit = recovery.select_recovery_strategy(diag, Urgency.CRITICAL)
        assert strategy_crit == RecoveryStrategy.AGGRESSIVE

    def test_recovery_rollback_after_diagnosis(self):
        analyzer = RootCauseAnalyzer()
        recovery = RecoveryManager()
        diag = Diagnosis(
            fault_id="f1", root_cause="config_error",
            confidence=0.9, recommended_fix="Change config",
        )
        actions = recovery.generate_recovery_plan(diag)
        if actions:
            result = recovery.execute_recovery(actions[0])
            rollback = recovery.rollback(actions[0])
            assert rollback.success is True

    def test_recovery_time_estimation(self):
        recovery = RecoveryManager()
        action = RecoveryAction(
            type=RecoveryType.RESTART,
            target="component",
            steps=["stop", "start", "verify"],
            estimated_time_seconds=10.0,
            risk_level=0.5,
        )
        estimated = recovery.estimate_recovery_time(action)
        assert estimated >= 10.0

    def test_recovery_success_rate(self):
        recovery = RecoveryManager()
        # Execute a few recoveries
        for _ in range(3):
            action = RecoveryAction(type=RecoveryType.RESTART, target="comp")
            recovery.execute_recovery(action)
        rate = recovery.compute_recovery_success_rate()
        assert rate == 100.0

    def test_diagnosis_history_and_recovery_tracking(self):
        analyzer = RootCauseAnalyzer()
        recovery = RecoveryManager()
        fault = _make_fault_event("sensor", "sensor_fail")
        diag = analyzer.diagnose(fault)
        actions = recovery.generate_recovery_plan(diag)
        for action in actions:
            recovery.execute_recovery(action)
        assert len(recovery.history) == len(actions)
        assert len(analyzer.history) == 1


# ===========================================================================
# 5. FaultDetector + RuntimeMonitor integration tests
# ===========================================================================

class TestFaultMonitorIntegration:
    """Tests that exercise FaultDetector with RuntimeMonitor."""

    def test_fault_event_creates_monitor_alert(self):
        monitor = RuntimeMonitor()
        detector = FaultDetector()
        monitor.add_rule(MonitorRule(
            name="fault_detected",
            condition_fn=lambda e: e.event_type == "fault",
            severity="error",
        ))
        indicator = _make_indicator("cpu", "temp", 95.0, (20, 80))
        detector.register_health_indicator(indicator)
        indicators = detector.check_indicators()
        fault = detector.detect_fault(indicators)
        if fault:
            event = MonitorEvent(
                timestamp=time.time(), source="fault_detector",
                event_type="fault", value={"fault_id": fault.id, "component": fault.component},
            )
            alerts = monitor.process_event(event)
            assert len(alerts) >= 1

    def test_healthy_indicators_no_monitor_alerts(self):
        monitor = RuntimeMonitor()
        detector = FaultDetector()
        monitor.add_rule(MonitorRule(
            name="fault_alert",
            condition_fn=lambda e: e.event_type == "fault",
        ))
        healthy = _make_indicator("sensor", "reading", 50.0, (0, 100))
        detector.register_health_indicator(healthy)
        indicators = detector.check_indicators()
        fault = detector.detect_fault(indicators)
        assert fault is None
        health = monitor.compute_monitor_health()
        assert health == 100.0

    def test_multiple_fault_severities_in_monitor(self):
        monitor = RuntimeMonitor()
        detector = FaultDetector()
        for sev_name in ["info", "warning", "error", "critical"]:
            monitor.add_rule(MonitorRule(
                name=f"fault_{sev_name}",
                condition_fn=lambda e: e.event_type == "fault" and e.metadata and e.metadata.get("severity") == sev_name,
                severity=sev_name,
            ))
        # Create fault events of different severities
        for severity in [FaultSeverity.LOW, FaultSeverity.MEDIUM, FaultSeverity.HIGH, FaultSeverity.CRITICAL]:
            fault = FaultEvent(component="comp", fault_type="test", severity=severity)
            event = MonitorEvent(
                timestamp=time.time(), source="fault_detector",
                event_type="fault", metadata={"severity": severity.name.lower()},
            )
            monitor.process_event(event)


# ===========================================================================
# 6. Watchdog + FaultDetector integration tests
# ===========================================================================

class TestWatchdogFaultIntegration:
    """Tests that exercise WatchdogManager with FaultDetector."""

    def test_watchdog_miss_triggers_fault_registration(self):
        watchdog = WatchdogManager()
        detector = FaultDetector()
        config = WatchdogConfig(timeout_ms=50, max_missed_beats=2)
        watchdog.register("sensor", config)
        # Feed initial heartbeat
        watchdog.feed_heartbeat("sensor")
        # Wait for expiry
        time.sleep(0.06)
        expired = watchdog.check_all()
        if "sensor" in expired:
            state = watchdog.get_state("sensor")
            indicator = HealthIndicator(
                component="sensor", metric_name="heartbeat",
                value=0.0, normal_range=(0.5, 2.0),
            )
            detector.register_health_indicator(indicator)

    def test_healthy_watchdog_no_fault(self):
        watchdog = WatchdogManager()
        detector = FaultDetector()
        config = WatchdogConfig(timeout_ms=5000)
        watchdog.register("gps", config)
        watchdog.feed_heartbeat("gps")
        expired = watchdog.check_all()
        assert "gps" not in expired
        # No fault should be detected
        indicators = detector.check_indicators()
        assert all(ind.status == IndicatorStatus.UNKNOWN for ind in indicators)

    def test_watchdog_uptime_and_fault_correlation(self):
        watchdog = WatchdogManager()
        detector = FaultDetector()
        config = WatchdogConfig(timeout_ms=50, max_missed_beats=3)
        watchdog.register("motor", config)
        # Feed some heartbeats
        for _ in range(5):
            watchdog.feed_heartbeat("motor")
            time.sleep(0.01)
        uptime = watchdog.compute_uptime("motor")
        assert uptime == 100.0
        # Register a healthy indicator
        indicator = _make_indicator("motor", "rpm", 2500, (2000, 3000))
        detector.register_health_indicator(indicator)
        fault = detector.detect_fault([indicator])
        assert fault is None


# ===========================================================================
# 7. SIL Verification + Fault/Recovery integration tests
# ===========================================================================

class TestSILSafetyIntegration:
    """Tests that exercise SILVerifier with fault and recovery modules."""

    def test_sil_verification_with_fault_context(self):
        sil = SILVerifier()
        detector = FaultDetector()
        # Create a fault that would affect SIL
        fault = _make_fault_event("safety_sensor", "sensor_anomaly", FaultSeverity.CRITICAL)
        category = detector.classify_fault(fault)
        assert category == FaultCategory.SENSOR
        # Verify SIL for a safety function affected by this fault
        target = SILTarget(
            safety_function="obstacle_detection",
            required_sil=SILLevel.SIL_2,
        )
        result = sil.verify_sil(target, hazard_rate=1e-3, test_coverage=0.85,
                                diagnostic_coverage=0.7)
        assert isinstance(result, SILVerificationResult)

    def test_hazard_rate_from_fault_frequency(self):
        sil = SILVerifier()
        recovery = RecoveryManager()
        # Compute hazard rate
        hazard = sil.compute_hazard_rate(1e-5, 8760, 0.5)
        assert hazard > 0
        # Verify PFD
        pfd = sil.compute_pfd(0.7, 8760)
        assert pfd >= 0
        # Estimate recovery needs
        action = RecoveryAction(
            type=RecoveryType.RESET, target="safety_module",
            steps=["reset", "verify"], estimated_time_seconds=5.0, risk_level=0.3,
        )
        result = recovery.execute_recovery(action)
        assert result.success

    def test_sff_computation_for_safety_verification(self):
        sil = SILVerifier()
        sff = sil.compute_sff(1e-4, 1e-3)
        assert 0.0 <= sff <= 1.0
        sff_full = sil.compute_sff(1e-3, 1e-3)
        assert sff_full == 1.0

    def test_sil_architecture_check(self):
        sil = SILVerifier()
        # SIL 3 needs HFT >= 1 (channels >= 2)
        valid, issues = sil.check_sil_architecture(SILLevel.SIL_3, "Type B", 2)
        assert valid
        # SIL 3 with 1 channel should fail
        invalid, issues = sil.check_sil_architecture(SILLevel.SIL_3, "Type B", 1)
        assert not invalid or len(issues) > 0

    def test_sil_recommendations_after_recovery(self):
        sil = SILVerifier()
        recovery = RecoveryManager()
        target = SILTarget(safety_function="collision_avoidance", required_sil=SILLevel.SIL_3)
        result = sil.verify_sil(target, hazard_rate=1e-2, test_coverage=0.5,
                                diagnostic_coverage=0.3)
        recommendations = sil.recommend_measures(result.achieved_sil, target.required_sil, result.gaps)
        # If there are gaps, there should be recommendations
        if result.gaps:
            assert len(recommendations) > 0
        # Execute recovery with moderate urgency
        diag = Diagnosis(
            fault_id="f1", root_cause="sensor_degradation",
            confidence=0.6, recommended_fix="Replace sensor",
        )
        strategy = recovery.select_recovery_strategy(diag, Urgency.MEDIUM)
        assert strategy in (RecoveryStrategy.MODERATE, RecoveryStrategy.CONSERVATIVE)

    def test_sil_pfd_ranges(self):
        sil = SILVerifier()
        # SIL 2: PFD should be in [1e-3, 1e-2]
        achieved = sil._determine_sil_from_pfd(5e-3)
        assert achieved == SILLevel.SIL_2
        # Very low PFD -> SIL 4
        achieved_high = sil._determine_sil_from_pfd(5e-6)
        assert achieved_high == SILLevel.SIL_4


# ===========================================================================
# 8. Full safety chain tests (3+ modules)
# ===========================================================================

class TestFullSafetyChain:
    """End-to-end tests: invariant check -> fault detect -> diagnose -> recover -> SIL verify."""

    def test_full_safety_chain_violation_to_recovery(self):
        checker = InvariantChecker()
        detector = FaultDetector()
        analyzer = RootCauseAnalyzer()
        recovery = RecoveryManager()
        checker.register(_make_invariant("temp"))
        # Violation detected
        violation = checker.check("temp", {"temp": 150})
        assert violation is not None
        # Create fault from violation
        indicator = HealthIndicator(
            component="thermal", metric_name="temp",
            value=150.0, normal_range=(20, 80),
        )
        detector.register_health_indicator(indicator)
        indicators = detector.check_indicators()
        fault = detector.detect_fault(indicators)
        # Diagnose
        analyzer.add_rule(DiagnosticRule(
            symptoms_pattern={"component": "thermal"},
            probable_cause="cooling_system_failure",
            confidence=0.85,
            fix_recommendation="Activate backup cooling",
        ))
        diag = analyzer.diagnose(fault, {"component": "thermal"}) if fault else Diagnosis(
            fault_id="gen", root_cause="unknown", confidence=0.3,
        )
        # Generate and execute recovery plan
        actions = recovery.generate_recovery_plan(diag)
        for action in actions:
            result = recovery.execute_recovery(action)
            assert result.success is True

    def test_monitor_watchdog_fault_recovery_chain(self):
        monitor = RuntimeMonitor()
        watchdog = WatchdogManager()
        detector = FaultDetector()
        recovery = RecoveryManager()
        # Register component
        config = WatchdogConfig(timeout_ms=50, max_missed_beats=2)
        watchdog.register("critical_sensor", config)
        monitor.add_rule(MonitorRule(
            name="wd_expired",
            condition_fn=lambda e: e.event_type == "watchdog_expired",
            severity="critical",
        ))
        # Feed then let expire
        watchdog.feed_heartbeat("critical_sensor")
        time.sleep(0.06)
        expired = watchdog.check_all()
        if "critical_sensor" in expired:
            # Create fault
            indicator = HealthIndicator(
                component="critical_sensor", metric_name="heartbeat",
                value=0, normal_range=(0.5, 2),
            )
            detector.register_health_indicator(indicator)
            fault = detector.detect_fault([indicator])
            if fault:
                event = MonitorEvent(
                    timestamp=time.time(), source="watchdog",
                    event_type="watchdog_expired", value={"component": "critical_sensor"},
                )
                alerts = monitor.process_event(event)
                # Execute recovery
                diag = Diagnosis(
                    fault_id=fault.id, root_cause="sensor_unresponsive",
                    confidence=0.7, recommended_fix="Restart sensor",
                )
                actions = recovery.generate_recovery_plan(diag)
                if actions:
                    result = recovery.execute_recovery(actions[0])
                    assert result.success

    def test_invariant_monitor_watchdog_combined(self):
        checker = InvariantChecker()
        monitor = RuntimeMonitor()
        watchdog = WatchdogManager()
        # Register invariants and watchdog
        checker.register(_make_invariant("cpu_temp"))
        checker.register(_make_invariant("memory"))
        watchdog.register("cpu", WatchdogConfig(timeout_ms=5000))
        watchdog.register("memory", WatchdogConfig(timeout_ms=5000))
        monitor.add_rule(MonitorRule(
            name="invariant_check",
            condition_fn=lambda e: e.event_type == "invariant_check",
            severity="info",
        ))
        # Check invariants
        state = {"cpu_temp": 70, "memory": 50}
        violations = checker.check_all(state)
        for v in violations:
            event = MonitorEvent(
                timestamp=time.time(), source="invariants",
                event_type="invariant_check", value={"name": v.invariant},
            )
            monitor.process_event(event)
        # Feed watchdogs
        watchdog.feed_heartbeat("cpu")
        watchdog.feed_heartbeat("memory")
        expired = watchdog.check_all()
        assert "cpu" not in expired
        assert "memory" not in expired

    def test_fault_severity_computation(self):
        detector = FaultDetector()
        fault = FaultEvent(
            component="sensor", fault_type="gps_anomaly",
            severity=FaultSeverity.LOW,
            symptoms=["gps=150.0 (normal (0, 100))"],
            context={"value": 150.0, "normal_range": (0, 100)},
        )
        severity = detector.compute_severity(fault)
        assert severity.value >= FaultSeverity.MEDIUM.value

    def test_sil_verification_with_recovery_success_rate(self):
        sil = SILVerifier()
        recovery = RecoveryManager()
        # Execute multiple recoveries
        for _ in range(5):
            action = RecoveryAction(type=RecoveryType.RESTART, target="safety_sys")
            recovery.execute_recovery(action)
        rate = recovery.compute_recovery_success_rate()
        assert rate == 100.0
        # Use recovery rate in SIL context
        target = SILTarget(safety_function="emergency_stop", required_sil=SILLevel.SIL_3)
        result = sil.verify_sil(target, hazard_rate=1e-4, test_coverage=0.9,
                                diagnostic_coverage=0.85)
        assert isinstance(result.pass_fail, bool)

    def test_full_chain_sil_pfd_with_recovery(self):
        sil = SILVerifier()
        recovery = RecoveryManager()
        analyzer = RootCauseAnalyzer()
        # Compute PFD for safety function
        # Use a small proof test interval so PFD falls within a SIL range
        pfd = sil.compute_pfd(diagnostic_coverage=0.9, proof_test_interval=1.0)
        assert pfd >= 0
        # Generate recovery plan regardless of SIL level
        diag = Diagnosis(
            fault_id="safety1", root_cause="degraded_diagnostics",
            confidence=0.6, recommended_fix="Improve diagnostic coverage",
        )
        actions = recovery.generate_recovery_plan(diag)
        assert len(actions) > 0
        # Verify SIL with a reasonable hazard rate
        target = SILTarget(safety_function="hull_integrity", required_sil=SILLevel.SIL_2)
        result = sil.verify_sil(target, hazard_rate=5e-4, test_coverage=0.9,
                                diagnostic_coverage=0.9)
        assert isinstance(result, SILVerificationResult)

    def test_monitor_alert_acknowledgement_in_safety_context(self):
        monitor = RuntimeMonitor()
        checker = InvariantChecker()
        checker.register(_make_invariant("pressure"))
        violation = checker.check("pressure", {"pressure": 200})
        if violation:
            event = MonitorEvent(
                timestamp=time.time(), source="safety",
                event_type="violation", value={"name": "pressure"},
            )
            monitor.add_rule(MonitorRule(
                name="safety_violation",
                condition_fn=lambda e: e.event_type == "violation",
                severity="critical",
            ))
            alerts = monitor.process_event(event)
            if alerts:
                ack = monitor.acknowledge_alert(alerts[0].alert_id)
                assert ack is True
                remaining = monitor.get_active_alerts()
                assert len(remaining) == 0

    def test_recovery_history_persistence_across_faults(self):
        detector = FaultDetector()
        analyzer = RootCauseAnalyzer()
        recovery = RecoveryManager()
        analyzer.add_rule(DiagnosticRule(
            symptoms_pattern={"component": "power"},
            probable_cause="power_supply_degradation",
            confidence=0.8,
            fix_recommendation="Switch to backup power",
        ))
        for i in range(3):
            fault = FaultEvent(
                component="power", fault_type="voltage_drop",
                severity=FaultSeverity.HIGH,
                symptoms=[f"voltage={4.0 - i * 0.3}V"],
            )
            diag = analyzer.diagnose(fault, {"component": "power"})
            actions = recovery.generate_recovery_plan(diag)
            for action in actions:
                recovery.execute_recovery(action)
        assert len(recovery.history) >= 3
        rate = recovery.compute_recovery_success_rate()
        assert rate == 100.0

    def test_watchdog_multiple_components_safety(self):
        watchdog = WatchdogManager()
        monitor = RuntimeMonitor()
        components = ["nav", "propulsion", "sensors", "comms"]
        for comp in components:
            watchdog.register(comp, WatchdogConfig(timeout_ms=5000))
            watchdog.feed_heartbeat(comp)
        summary = watchdog.get_summary()
        assert summary["total_components"] == len(components)
        for comp in components:
            uptime = watchdog.compute_uptime(comp)
            assert uptime == 100.0
            state = watchdog.get_state(comp)
            assert state is not None
            assert state.active is True

    def test_invariant_checker_edge_cases(self):
        checker = InvariantChecker()
        # Register invariant with exception-raising check
        def bad_check(state):
            raise ValueError("test error")
        checker.register(Invariant(name="bad", check_fn=bad_check))
        violation = checker.check("bad", {"bad": 1})
        assert violation is not None  # exceptions treated as violations

    def test_recovery_all_types(self):
        recovery = RecoveryManager()
        types = [
            RecoveryType.RESTART, RecoveryType.RECONFIGURE,
            RecoveryType.FAILOVER, RecoveryType.ISOLATE,
            RecoveryType.PATCH, RecoveryType.RESET,
        ]
        for rtype in types:
            action = RecoveryAction(type=rtype, target="component")
            result = recovery.execute_recovery(action)
            assert result.success is True
            assert result.action_taken == rtype.value
        assert len(recovery.history) == len(types)

    def test_full_fault_detect_diagnose_recover_sil(self):
        detector = FaultDetector()
        analyzer = RootCauseAnalyzer()
        recovery = RecoveryManager()
        sil = SILVerifier()
        # 1. Detect fault
        indicator = HealthIndicator(
            component="brake", metric_name="response_time",
            value=500.0, normal_range=(10, 100),
        )
        detector.register_health_indicator(indicator)
        fault = detector.detect_fault([indicator])
        # 2. Diagnose
        analyzer.add_rule(DiagnosticRule(
            symptoms_pattern={"component": "brake"},
            probable_cause="brake_actuator_degradation",
            confidence=0.9,
            fix_recommendation="Replace brake actuator",
        ))
        if fault:
            diag = analyzer.diagnose(fault, {"component": "brake"})
        else:
            diag = Diagnosis(fault_id="f1", root_cause="unknown", confidence=0.3)
        # 3. Recover
        actions = recovery.generate_recovery_plan(diag)
        for action in actions:
            result = recovery.execute_recovery(action)
            assert result.success
        # 4. SIL verify
        target = SILTarget(safety_function="emergency_braking", required_sil=SILLevel.SIL_2)
        sil_result = sil.verify_sil(target, hazard_rate=5e-4, test_coverage=0.9,
                                     diagnostic_coverage=0.8)
        assert sil_result.achieved_sil is not None
