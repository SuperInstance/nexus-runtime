"""NEXUS Jetson tests - Trust engine tests.

20+ tests covering INCREMENTS trust algorithm.
"""

import math
import time

import pytest

from trust.increments import (
    IncrementTrustEngine,
    TrustEvent,
    TrustParams,
    TrustUpdateResult,
    SubsystemTrust,
    SUBSYSTEMS,
)
from trust.events import (
    EventDefinition,
    classify_event,
    get_bad_events,
    get_good_events,
    get_neutral_events,
)
from trust.levels import (
    AUTONOMY_LEVELS,
    AutonomyLevel,
    can_promote,
    get_level_definition,
)


@pytest.fixture
def engine() -> IncrementTrustEngine:
    e = IncrementTrustEngine()
    e.register_all_subsystems()
    return e


@pytest.fixture
def engine_single() -> IncrementTrustEngine:
    e = IncrementTrustEngine()
    e.register_subsystem("steering")
    return e


def _make_good_event(quality: float = 0.8, ts: float = 0.0) -> TrustEvent:
    return TrustEvent.good("reflex_completed", quality, timestamp=ts, subsystem="steering")


def _make_bad_event(severity: float = 0.6, ts: float = 0.0) -> TrustEvent:
    return TrustEvent.bad("reflex_error", severity, timestamp=ts, subsystem="steering")


# ===================================================================
# Test 1: Parameter Validation
# ===================================================================

class TestIncrementsParams:
    def test_default_params(self) -> None:
        params = TrustParams()
        assert params.alpha_gain == 0.002
        assert params.alpha_loss == 0.05
        assert params.alpha_decay == 0.0001
        assert params.t_floor == 0.2
        assert params.quality_cap == 10
        assert params.severity_exponent == 1.0
        assert params.streak_bonus == 0.00005
        assert params.min_events_for_gain == 1
        assert params.n_penalty_slope == 0.1
        assert params.reset_grace_hours == 24.0
        assert params.promotion_cooldown_hours == 72.0

    def test_parameter_invariants(self) -> None:
        p = TrustParams()
        assert p.alpha_loss / p.alpha_gain == 25.0
        assert 0.0 <= p.t_floor <= 1.0
        assert p.alpha_gain > 0
        assert p.alpha_loss > p.alpha_gain
        assert p.quality_cap > 0


# ===================================================================
# Test 2: Engine Creation
# ===================================================================

class TestEngineCreation:
    def test_engine_creation(self) -> None:
        engine = IncrementTrustEngine()
        assert engine is not None

    def test_register_subsystem(self) -> None:
        engine = IncrementTrustEngine()
        engine.register_subsystem("steering")
        assert engine.get_trust("steering") == 0.0

    def test_register_all_subsystems(self) -> None:
        engine = IncrementTrustEngine()
        engine.register_all_subsystems()
        scores = engine.get_all_scores()
        assert len(scores) == 5
        for name in SUBSYSTEMS:
            assert name in scores


# ===================================================================
# Test 3: Ideal Trust Growth
# ===================================================================

class TestTrustGrowth:
    def test_single_window_gain(self, engine_single) -> None:
        result = engine_single.evaluate_window("steering", [_make_good_event()])
        assert result.delta > 0
        assert result.new_score > 0.0
        assert result.branch == "gain"

    def test_ideal_growth_trajectory(self, engine_single) -> None:
        """658 windows of quality=1.0 events should reach >= 0.79."""
        event = _make_good_event(quality=1.0)
        for _ in range(658):
            engine_single.evaluate_window("steering", [event])
        score = engine_single.get_trust_score("steering")
        assert score >= 0.79, f"Expected >= 0.79 after 658 windows, got {score:.6f}"


# ===================================================================
# Test 4: Rapid Trust Loss
# ===================================================================

class TestTrustLoss:
    def test_rapid_loss_from_090(self, engine_single) -> None:
        """From 0.90, severity=1.0 events should drive trust to near floor in ~29 windows."""
        engine_single.subsystems["steering"].trust_score = 0.90
        engine_single.subsystems["steering"].consecutive_clean_windows = 0

        bad = _make_bad_event(severity=1.0)
        for _ in range(29):
            engine_single.evaluate_window("steering", [bad])

        score = engine_single.get_trust_score("steering")
        assert score <= 0.21, f"Expected <= 0.21 after 29 severity-1.0 hits, got {score:.6f}"
        assert score >= engine_single.params.t_floor

    def test_penalty_branch_ignores_good(self, engine_single) -> None:
        """Penalty branch: good events are ignored when bad events exist."""
        # Build trust to ~0.30
        event = _make_good_event(quality=1.0)
        for _ in range(300):
            engine_single.evaluate_window("steering", [event])
        pre_score = engine_single.get_trust_score("steering")
        assert pre_score > 0.2

        # Mixed events: bad + good -> penalty branch
        mixed = [_make_bad_event(0.5), _make_good_event(1.0)]
        result = engine_single.evaluate_window("steering", mixed)
        assert result.branch == "penalty"
        assert result.delta < 0
        assert result.new_score < pre_score


# ===================================================================
# Test 5: Agent Code Half-Rate
# ===================================================================

class TestAgentCodeMultiplier:
    def test_agent_code_half_gain_direct(self, engine_single) -> None:
        """Agent code compute_delta gives exactly 0.5x gain (no streak)."""
        event = _make_good_event(quality=1.0)
        d_normal, b = engine_single.compute_delta(0.5, [event], consecutive_clean=0, is_agent_code=False)
        d_agent, b2 = engine_single.compute_delta(0.5, [event], consecutive_clean=0, is_agent_code=True)
        assert b == "gain"
        assert b2 == "gain"
        ratio = d_agent / d_normal if d_normal > 0 else 0
        assert ratio == 0.5, f"Agent/normal delta ratio should be 0.5, got {ratio}"

    def test_agent_code_accumulates_less(self, engine_single) -> None:
        """Over many windows, agent code accumulates less trust."""
        event = _make_good_event(quality=1.0)

        for _ in range(200):
            engine_single.evaluate_window("steering", [event])
        normal_score = engine_single.get_trust_score("steering")

        engine_single.subsystems["steering"].trust_score = 0.0
        engine_single.subsystems["steering"].consecutive_clean_windows = 0
        engine_single.subsystems["steering"].clean_windows = 0
        engine_single.subsystems["steering"].total_windows = 0
        engine_single.subsystems["steering"].total_observation_hours = 0.0
        engine_single.subsystems["steering"].last_reset_time = 0.0
        for _ in range(200):
            engine_single.evaluate_window("steering", [event], is_agent_code=True)
        agent_score = engine_single.get_trust_score("steering")

        assert agent_score < normal_score, (
            f"Agent ({agent_score:.6f}) should be less than normal ({normal_score:.6f})"
        )


# ===================================================================
# Test 6: Per-Subsystem Independence
# ===================================================================

class TestSubsystemIndependence:
    def test_independent_subsystems(self, engine) -> None:
        good_nav = TrustEvent.good("reflex_completed", 0.9, subsystem="navigation")
        bad_steer = TrustEvent.bad("reflex_error", 0.8, subsystem="steering")

        for _ in range(100):
            engine.evaluate_window("navigation", [good_nav])
            engine.evaluate_window("steering", [bad_steer])

        nav_score = engine.get_trust_score("navigation")
        steer_score = engine.get_trust_score("steering")
        assert nav_score > 0.1
        assert steer_score <= engine.params.t_floor

    def test_concurrent_tracking(self, engine) -> None:
        for name in SUBSYSTEMS:
            event = TrustEvent.good("reflex_completed", 0.9, subsystem=name)
            engine.evaluate_window(name, [event])

        scores = engine.get_all_scores()
        for name in SUBSYSTEMS:
            assert scores[name].trust_score > 0.0


# ===================================================================
# Test 7: Event Classification
# ===================================================================

class TestEventClassification:
    def test_all_15_events_exist(self) -> None:
        from trust.events import EVENT_DEFINITIONS
        assert len(EVENT_DEFINITIONS) == 15

    def test_good_events(self) -> None:
        goods = get_good_events()
        assert len(goods) == 5
        for name in goods:
            defn = classify_event(name)
            assert defn is not None
            assert defn.category == "good"
            assert defn.quality > 0.0

    def test_bad_events(self) -> None:
        bads = get_bad_events()
        assert len(bads) == 7
        for name in bads:
            defn = classify_event(name)
            assert defn is not None
            assert defn.category == "bad"
            assert defn.severity > 0.0

    def test_neutral_events(self) -> None:
        neutrals = get_neutral_events()
        assert len(neutrals) == 3
        for name in neutrals:
            defn = classify_event(name)
            assert defn is not None
            assert defn.category == "neutral"

    def test_unknown_event(self) -> None:
        assert classify_event("nonexistent_event") is None

    def test_event_severity_ordering(self) -> None:
        bads = get_bad_events()
        severities = [classify_event(n).severity for n in bads]
        for s in severities:
            assert 0.0 < s <= 1.0

    def test_all_good_event_names(self) -> None:
        for name in ["heartbeat_ok", "sensor_valid", "reflex_completed", "actuator_nominal", "command_ack"]:
            assert classify_event(name) is not None

    def test_all_bad_event_names(self) -> None:
        for name in ["heartbeat_missed", "sensor_invalid", "reflex_error",
                      "actuator_overrange", "trust_violation", "safety_trigger",
                      "communication_timeout"]:
            assert classify_event(name) is not None

    def test_all_neutral_event_names(self) -> None:
        for name in ["system_boot", "parameter_change", "calibration_complete"]:
            assert classify_event(name) is not None


# ===================================================================
# Test 8: Autonomy Levels
# ===================================================================

class TestAutonomyLevels:
    def test_all_6_levels_defined(self) -> None:
        for level in range(6):
            defn = get_level_definition(level)
            assert defn is not None
            assert defn.level == level

    def test_no_promotion_at_zero(self) -> None:
        assert can_promote(0.0, 0, 0.0, 0) == 0

    def test_level_thresholds(self) -> None:
        assert AUTONOMY_LEVELS[1].trust_threshold == 0.20
        assert AUTONOMY_LEVELS[2].trust_threshold == 0.40
        assert AUTONOMY_LEVELS[3].trust_threshold == 0.60
        assert AUTONOMY_LEVELS[4].trust_threshold == 0.80
        assert AUTONOMY_LEVELS[5].trust_threshold == 0.95

    def test_can_promote_checks_hours(self) -> None:
        assert can_promote(0.50, 0, 0.0, 0) == 0
        assert can_promote(0.50, 0, 48.0, 100) >= 2


# ===================================================================
# Test 9: Demotion Rules
# ===================================================================

class TestDemotion:
    def test_immediate_demotion_on_bad(self, engine_single) -> None:
        engine_single.subsystems["steering"].autonomy_level = 3
        engine_single.subsystems["steering"].trust_score = 0.85
        engine_single.subsystems["steering"].clean_windows = 200
        engine_single.subsystems["steering"].total_observation_hours = 500.0

        result = engine_single.evaluate_window("steering", [_make_bad_event(0.3)])
        assert result.new_level == 2

    def test_severity_08_demotes_2(self, engine_single) -> None:
        engine_single.subsystems["steering"].autonomy_level = 4
        engine_single.subsystems["steering"].trust_score = 0.85
        result = engine_single.evaluate_window("steering", [_make_bad_event(0.8)])
        assert result.new_level <= 2

    def test_severity_10_demotes_to_l0(self, engine_single) -> None:
        engine_single.subsystems["steering"].autonomy_level = 5
        engine_single.subsystems["steering"].trust_score = 0.99
        result = engine_single.evaluate_window("steering", [_make_bad_event(1.0)])
        assert result.new_level == 0


# ===================================================================
# Test 10: Decay Branch
# ===================================================================

class TestDecay:
    def test_decay_no_events(self, engine_single) -> None:
        event = _make_good_event(quality=1.0)
        for _ in range(200):
            engine_single.evaluate_window("steering", [event])
        score_before = engine_single.get_trust_score("steering")
        assert score_before > engine_single.params.t_floor

        neutral = TrustEvent("system_boot", quality=0.0, severity=0.0,
                            timestamp=0.0, subsystem="steering", is_bad=False)
        result = engine_single.evaluate_window("steering", [neutral])
        assert result.branch == "decay"
        assert result.delta < 0
        assert result.new_score < score_before

    def test_decay_floor_clamp(self, engine_single) -> None:
        engine_single.subsystems["steering"].trust_score = engine_single.params.t_floor

        neutral = TrustEvent("system_boot", quality=0.0, severity=0.0,
                            timestamp=0.0, subsystem="steering", is_bad=False)
        result = engine_single.evaluate_window("steering", [neutral])
        assert result.new_score >= engine_single.params.t_floor


# ===================================================================
# Test 11: Streak Bonus
# ===================================================================

class TestStreakBonus:
    def test_streak_bonus_increases_delta(self, engine_single) -> None:
        event = _make_good_event(quality=1.0)
        d0, _ = engine_single.compute_delta(0.5, [event], consecutive_clean=0)
        d24, _ = engine_single.compute_delta(0.5, [event], consecutive_clean=24)
        assert d24 > d0, f"Streak bonus should increase delta: {d24} vs {d0}"

    def test_streak_caps_at_24(self, engine_single) -> None:
        d24, _ = engine_single.compute_delta(0.5, [_make_good_event()], consecutive_clean=24)
        d48, _ = engine_single.compute_delta(0.5, [_make_good_event()], consecutive_clean=48)
        assert abs(d24 - d48) < 1e-12, "Streak should cap at 24"


# ===================================================================
# Test 12: Quality Cap
# ===================================================================

class TestQualityCap:
    def test_quality_cap(self, engine_single) -> None:
        params = TrustParams(quality_cap=5)
        engine_single.params = params

        events_5 = [_make_good_event(1.0) for _ in range(5)]
        result_5 = engine_single.evaluate_window("steering", events_5)

        engine_single.subsystems["steering"].trust_score = 0.0
        engine_single.subsystems["steering"].consecutive_clean_windows = 0
        engine_single.subsystems["steering"].clean_windows = 0
        engine_single.subsystems["steering"].total_windows = 0
        engine_single.subsystems["steering"].total_observation_hours = 0.0
        engine_single.subsystems["steering"].last_reset_time = 0.0

        events_20 = [_make_good_event(1.0) for _ in range(20)]
        result_20 = engine_single.evaluate_window("steering", events_20)

        assert result_5.branch == "gain"
        assert result_20.branch == "gain"
        assert abs(result_5.delta - result_20.delta) < 1e-10


# ===================================================================
# Test 13: Should Allow Deploy
# ===================================================================

class TestDeployCheck:
    def test_deploy_blocked_low_trust(self, engine_single) -> None:
        assert not engine_single.should_allow_deploy("steering", 0.5)

    def test_deploy_allowed_high_trust(self, engine_single) -> None:
        engine_single.subsystems["steering"].trust_score = 0.9
        assert engine_single.should_allow_deploy("steering", 0.5)

    def test_deploy_exact_threshold(self, engine_single) -> None:
        engine_single.subsystems["steering"].trust_score = 0.5
        assert engine_single.should_allow_deploy("steering", 0.5)
        assert not engine_single.should_allow_deploy("steering", 0.51)


# ===================================================================
# Test 14: Reset Grace Period
# ===================================================================

class TestResetGrace:
    def test_reset_within_grace_ignored(self, engine_single) -> None:
        engine_single.subsystems["steering"].trust_score = 0.8
        engine_single.subsystems["steering"].last_reset_time = time.time()
        engine_single.reset_subsystem("steering", "full")
        assert engine_single.get_trust_score("steering") == 0.8

    def test_reset_after_grace(self, engine_single) -> None:
        engine_single.subsystems["steering"].trust_score = 0.8
        engine_single.subsystems["steering"].last_reset_time = time.time() - 86400
        engine_single.reset_subsystem("steering", "full")
        assert engine_single.get_trust_score("steering") == 0.0


# ===================================================================
# Test 17: n_penalty_slope
# ===================================================================

class TestNPenalty:
    def test_multiple_bad_events_amplify(self, engine_single) -> None:
        engine_single.subsystems["steering"].trust_score = 0.8

        single_bad = engine_single.evaluate_window("steering", [_make_bad_event(0.5)])
        engine_single.subsystems["steering"].trust_score = 0.8
        triple_bad = engine_single.evaluate_window(
            "steering", [_make_bad_event(0.5), _make_bad_event(0.5), _make_bad_event(0.5)]
        )

        assert abs(triple_bad.delta) > abs(single_bad.delta)


# ===================================================================
# Test 18: Severity Exponent
# ===================================================================

class TestSeverityExponent:
    def test_exponent_amplifies_spread(self) -> None:
        """Exponent > 1 amplifies the penalty spread between severity levels."""
        e1 = IncrementTrustEngine(TrustParams(severity_exponent=1.0))
        e1.register_subsystem("steering")

        e2 = IncrementTrustEngine(TrustParams(severity_exponent=2.0))
        e2.register_subsystem("steering")

        # Penalties at severity 0.8 vs 0.4
        d_exp1_high, _ = e1.compute_delta(0.8, [TrustEvent.bad("x", 0.8, subsystem="steering")], 0)
        d_exp1_low, _ = e1.compute_delta(0.8, [TrustEvent.bad("x", 0.4, subsystem="steering")], 0)
        ratio_exp1 = abs(d_exp1_high / d_exp1_low) if d_exp1_low != 0 else 0

        d_exp2_high, _ = e2.compute_delta(0.8, [TrustEvent.bad("x", 0.8, subsystem="steering")], 0)
        d_exp2_low, _ = e2.compute_delta(0.8, [TrustEvent.bad("x", 0.4, subsystem="steering")], 0)
        ratio_exp2 = abs(d_exp2_high / d_exp2_low) if d_exp2_low != 0 else 0

        # With exponent=2, the spread ratio should be larger
        assert ratio_exp2 > ratio_exp1, (
            f"Exp=2 spread ({ratio_exp2:.2f}) should be > Exp=1 spread ({ratio_exp1:.2f})"
        )


# ===================================================================
# Test 19: Concurrent Subsystem Tracking
# ===================================================================

class TestConcurrentTracking:
    def test_five_subsystems_independent(self, engine) -> None:
        for i, name in enumerate(SUBSYSTEMS):
            quality = 0.5 + i * 0.1
            event = TrustEvent.good("reflex_completed", quality, subsystem=name)
            for _ in range(50):
                engine.evaluate_window(name, [event])

        scores = engine.get_all_scores()
        prev = -1.0
        for name in SUBSYSTEMS:
            s = scores[name].trust_score
            assert s >= prev, f"{name} trust {s} should be >= previous {prev}"
            prev = s


# ===================================================================
# Test 20: Full Simulation
# ===================================================================

class TestFullSimulation:
    def test_growth_then_shock(self, engine_single) -> None:
        good = _make_good_event(1.0)
        bad = _make_bad_event(0.8)

        for _ in range(200):
            engine_single.evaluate_window("steering", [good])
        peak = engine_single.get_trust_score("steering")
        assert peak > 0.1, "Trust should grow after 200 good windows"

        for _ in range(10):
            engine_single.evaluate_window("steering", [bad])
        trough = engine_single.get_trust_score("steering")
        assert trough < peak, "Trust should drop after bad events"

    def test_score_never_exceeds_1(self, engine_single) -> None:
        event = _make_good_event(1.0)
        for _ in range(5000):
            engine_single.evaluate_window("steering", [event])
            score = engine_single.get_trust_score("steering")
            assert 0.0 <= score <= 1.0

    def test_initial_trust_is_zero(self, engine_single) -> None:
        assert engine_single.get_trust_score("steering") == 0.0

    def test_initial_autonomy_is_zero(self, engine_single) -> None:
        assert engine_single.get_autonomy_level("steering") == 0

    def test_full_growth_trajectory(self, engine_single) -> None:
        event = _make_good_event(1.0)
        prev = 0.0
        for i in range(100):
            engine_single.evaluate_window("steering", [event])
            curr = engine_single.get_trust_score("steering")
            assert curr >= prev, f"Trust should not decrease: {curr} < {prev} at window {i}"
            prev = curr
        assert prev > 0.0
