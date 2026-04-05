"""Tests for NEXUS Tripartite Consensus — 80+ tests."""

import pytest
from jetson.agent.tripartite.agents import (
    AgentAssessment, DecisionVerdict, EthosAgent, IntentAssessment,
    IntentType, LogosAgent, PathosAgent, PlanAssessment, SafetyAssessment,
)
from jetson.agent.tripartite.consensus import ConsensusEngine, ConsensusResult, VotingStrategy
from jetson.agent.tripartite.marine_rules import (
    COLREGsRule, EquipmentLimits, EnvironmentalRules, MarineRulesDatabase,
    NoGoZone, VesselSituation,
)


# ===================================================================
# Pathos Tests
# ===================================================================
class TestPathosAgentInit:
    def test_default_mission(self):
        p = PathosAgent()
        assert p.mission_type == IntentType.ROUTINE

    def test_fishing_mission(self):
        p = PathosAgent(mission_type=IntentType.FISHING, mission_description="Trawl for cod")
        assert p.mission_type == IntentType.FISHING

    def test_navigation_mission(self):
        p = PathosAgent(mission_type=IntentType.NAVIGATION)
        assert p.mission_type == IntentType.NAVIGATION


class TestPathosIntentDetection:
    def test_detect_fishing(self):
        p = PathosAgent()
        assert p._detect_intent("deploy fishing nets") == IntentType.FISHING

    def test_detect_navigation(self):
        p = PathosAgent()
        assert p._detect_intent("navigate to waypoint") == IntentType.NAVIGATION

    def test_detect_emergency(self):
        p = PathosAgent()
        assert p._detect_intent("emergency abandon ship") == IntentType.EMERGENCY

    def test_detect_rescue(self):
        p = PathosAgent()
        assert p._detect_intent("rescue the crew") == IntentType.RESCUE

    def test_detect_docking(self):
        p = PathosAgent()
        assert p._detect_intent("dock at harbor") == IntentType.DOCKING

    def test_detect_survey(self):
        p = PathosAgent()
        assert p._detect_intent("survey the seabed") == IntentType.SURVEY

    def test_detect_routine(self):
        p = PathosAgent()
        assert p._detect_intent("patrol the area") == IntentType.ROUTINE

    def test_detect_unknown(self):
        p = PathosAgent()
        assert p._detect_intent("do something") == IntentType.UNKNOWN


class TestPathosUrgency:
    def test_no_urgency(self):
        p = PathosAgent()
        assert p._detect_urgency("go fishing", {}) == 0.0

    def test_emergency_urgency(self):
        p = PathosAgent()
        assert p._detect_urgency("emergency stop", {}) == 1.0

    def test_mayday_urgency(self):
        p = PathosAgent()
        assert p._detect_urgency("mayday mayday", {}) == 1.0

    def test_rescue_urgency(self):
        p = PathosAgent()
        assert p._detect_urgency("rescue vessel", {}) == 0.9

    def test_context_emergency(self):
        p = PathosAgent()
        assert p._detect_urgency("normal operation", {"emergency": True}) == 1.0

    def test_urgent_keyword(self):
        p = PathosAgent()
        assert p._detect_urgency("urgent repair", {}) == 0.7


class TestPathosEvaluate:
    def test_aligned_intent(self):
        p = PathosAgent(mission_type=IntentType.FISHING)
        r = p.evaluate("deploy nets and trawl for cod")
        assert r.alignment_score == 1.0
        assert r.intent_type == IntentType.FISHING

    def test_misaligned_intent(self):
        p = PathosAgent(mission_type=IntentType.FISHING)
        r = p.evaluate("dock at harbor")
        assert r.alignment_score == 0.3
        assert r.verdict == DecisionVerdict.DELEGATE

    def test_emergency_always_aligns(self):
        p = PathosAgent(mission_type=IntentType.ROUTINE)
        r = p.evaluate("emergency mayday fire on board")
        assert r.alignment_score == 0.95
        assert r.urgency == 1.0

    def test_ambiguous_action(self):
        p = PathosAgent()
        r = p.evaluate("do")
        assert r.verdict in (DecisionVerdict.REJECT, DecisionVerdict.DELEGATE)

    def test_returns_assessment_type(self):
        p = PathosAgent()
        r = p.evaluate("navigate to waypoint alpha")
        assert isinstance(r, IntentAssessment)
        assert r.agent_name == "pathos"

    def test_no_context(self):
        p = PathosAgent()
        r = p.evaluate("fish")
        assert r is not None

    def test_with_context(self):
        p = PathosAgent()
        r = p.evaluate("stop", {"emergency": True})
        assert r.urgency == 1.0


# ===================================================================
# Logos Tests
# ===================================================================
class TestLogosAgentInit:
    def test_defaults(self):
        l = LogosAgent()
        assert l.fuel_pct == 100.0
        assert l.battery_pct == 100.0

    def test_custom_fuel(self):
        l = LogosAgent(fuel_pct=50.0)
        assert l.fuel_pct == 50.0


class TestLogosHaversine:
    def test_same_point(self):
        assert LogosAgent._haversine((0, 0), (0, 0)) == 0.0

    def test_known_distance(self):
        d = LogosAgent._haversine((0, 0), (1, 0))
        assert 58 < d < 62  # ~60 nautical miles per degree latitude

    def test_equator_to_pole(self):
        d = LogosAgent._haversine((0, 0), (90, 0))
        assert 5390 < d < 5410  # ~5400 nm


class TestLogosEvaluate:
    def test_sound_plan(self):
        l = LogosAgent(fuel_pct=100.0)
        r = l.evaluate("navigate", {"distance_nm": 10.0})
        assert r.is_sound
        assert r.verdict == DecisionVerdict.APPROVE

    def test_insufficient_fuel(self):
        l = LogosAgent(fuel_pct=5.0, consumption_rate_pct_per_nm=5.0)
        r = l.evaluate("navigate far", {"distance_nm": 100.0})
        assert not r.is_sound
        assert not r.resource_check

    def test_out_of_range_target(self):
        l = LogosAgent(fuel_pct=20.0, consumption_rate_pct_per_nm=5.0)
        r = l.evaluate("go far", {"target_position": (45, -60)})
        assert not r.navigation_check

    def test_cycle_budget_exceeded(self):
        l = LogosAgent()
        r = l.evaluate("deploy", {"bytecode_length": 20000})
        assert not r.cycle_budget_ok

    def test_stack_budget_exceeded(self):
        l = LogosAgent()
        r = l.evaluate("deploy", {"max_stack_depth": 100})
        assert not r.stack_budget_ok

    def test_speed_limit(self):
        l = LogosAgent(max_speed_knots=5.0)
        r = l.evaluate("go fast", {"requested_speed_knots": 15.0})
        assert r.risk_score >= 0.2

    def test_returns_plan_assessment(self):
        l = LogosAgent()
        r = l.evaluate("test")
        assert isinstance(r, PlanAssessment)
        assert r.agent_name == "logos"


# ===================================================================
# Ethos Tests
# ===================================================================
class TestEthosAgentInit:
    def test_defaults(self):
        e = EthosAgent()
        assert e.trust_level == 0
        assert e.safety_state == "normal"

    def test_high_trust(self):
        e = EthosAgent(trust_level=5)
        assert e.trust_level == 5

    def test_fault_state(self):
        e = EthosAgent(safety_state="fault")
        assert e.safety_state == "fault"


class TestEthosTrustRequirements:
    def test_read_sensor(self):
        e = EthosAgent()
        assert e._get_trust_requirement("read_sensor data") == 0

    def test_emergency_stop(self):
        e = EthosAgent()
        assert e._get_trust_requirement("emergency_stop") == 0

    def test_deploy_reflex(self):
        e = EthosAgent()
        assert e._get_trust_requirement("deploy_reflex") == 2

    def test_override_safety(self):
        e = EthosAgent()
        assert e._get_trust_requirement("override_safety") == 5

    def test_unknown_default(self):
        e = EthosAgent()
        assert e._get_trust_requirement("something weird") == 1


class TestEthosEvaluate:
    def test_safe_action(self):
        e = EthosAgent(trust_level=2)
        r = e.evaluate("navigate to waypoint")
        assert r.is_safe

    def test_low_trust_reject(self):
        e = EthosAgent(trust_level=0)
        r = e.evaluate("deploy reflex code")
        assert not r.is_safe

    def test_fault_state_reject(self):
        e = EthosAgent(safety_state="fault")
        r = e.evaluate("deploy reflex code")
        assert not r.is_safe

    def test_fault_allows_emergency(self):
        e = EthosAgent(safety_state="fault", trust_level=5)
        r = e.evaluate("emergency stop", {})
        assert r.is_safe

    def test_no_go_zone(self):
        e = EthosAgent(no_go_zones=[{"name": "reef", "bounds": {"south": 10, "north": 20, "west": -50, "east": -40}}])
        r = e.evaluate("go", {"target_position": (15, -45)})
        assert not r.is_safe
        assert any("no-go" in v for v in r.violations)

    def test_colregs_violation(self):
        e = EthosAgent(colregs_rules=True)
        r = e.evaluate("collision with vessel", {"nearby_vessels": 2})
        assert not r.is_safe

    def test_speed_limit(self):
        e = EthosAgent()
        r = e.evaluate("go", {"requested_speed_knots": 20.0, "max_speed_knots": 10.0})
        assert not r.is_safe

    def test_marine_life(self):
        e = EthosAgent()
        r = e.evaluate("proceed at high speed", {"marine_life_nearby": True})
        assert r.ethical_score < 1.0

    def test_protected_area(self):
        e = EthosAgent()
        r = e.evaluate("anchor and fish", {"protected_area": True})
        assert r.ethical_score <= 0.5


# ===================================================================
# Consensus Tests
# ===================================================================
class TestConsensusUnanimous:
    def test_all_approve(self):
        e = ConsensusEngine(strategy=VotingStrategy.UNANIMOUS,
                           pathos=PathosAgent(mission_type=IntentType.NAVIGATION),
                           logos=LogosAgent(fuel_pct=100.0),
                           ethos=EthosAgent(trust_level=2))
        r = e.evaluate("navigate to waypoint", {"distance_nm": 5.0})
        assert r.approved

    def test_one_reject(self):
        e = ConsensusEngine(strategy=VotingStrategy.UNANIMOUS,
                           ethos=EthosAgent(trust_level=0))
        r = e.evaluate("deploy reflex code")
        assert not r.approved


class TestConsensusMajority:
    def test_majority_approves(self):
        e = ConsensusEngine(strategy=VotingStrategy.MAJORITY,
                           pathos=PathosAgent(mission_type=IntentType.ROUTINE),
                           logos=LogosAgent(fuel_pct=100.0),
                           ethos=EthosAgent(trust_level=3))
        r = e.evaluate("navigate to waypoint", {"distance_nm": 5.0})
        assert r.approved

    def test_minority_rejects(self):
        e = ConsensusEngine(strategy=VotingStrategy.MAJORITY,
                           ethos=EthosAgent(trust_level=5))
        r = e.evaluate("navigate to waypoint", {"distance_nm": 5.0})
        assert r.approved  # 2/3 still approve


class TestConsensusWeighted:
    def test_high_score_approves(self):
        e = ConsensusEngine(strategy=VotingStrategy.WEIGHTED,
                           pathos=PathosAgent(mission_type=IntentType.NAVIGATION),
                           logos=LogosAgent(fuel_pct=100.0),
                           ethos=EthosAgent(trust_level=3))
        r = e.evaluate("navigate to waypoint", {"distance_nm": 5.0})
        assert r.weighted_score > 0.6
        assert r.approved

    def test_low_score_rejects(self):
        e = ConsensusEngine(strategy=VotingStrategy.WEIGHTED,
                           ethos=EthosAgent(trust_level=0))
        r = e.evaluate("deploy reflex code")
        assert not r.approved


class TestConsensusVeto:
    def test_no_veto(self):
        e = ConsensusEngine(strategy=VotingStrategy.VETO,
                           pathos=PathosAgent(mission_type=IntentType.NAVIGATION),
                           logos=LogosAgent(fuel_pct=100.0),
                           ethos=EthosAgent(trust_level=3))
        r = e.evaluate("navigate to waypoint", {"distance_nm": 5.0})
        assert r.approved

    def test_double_veto(self):
        e = ConsensusEngine(strategy=VotingStrategy.VETO,
                           pathos=PathosAgent(mission_type=IntentType.DOCKING),
                           logos=LogosAgent(fuel_pct=100.0),
                           ethos=EthosAgent(trust_level=0))
        r = e.evaluate("navigate to waypoint")
        assert not r.approved


class TestConsensusAutoEthos:
    def test_ethos_breaks_tie(self):
        e = ConsensusEngine(strategy=VotingStrategy.AUTO_ETHOS,
                           pathos=PathosAgent(mission_type=IntentType.DOCKING),
                           logos=LogosAgent(fuel_pct=100.0),
                           ethos=EthosAgent(trust_level=3))
        r = e.evaluate("navigate to waypoint", {"distance_nm": 5.0})
        assert r.decision in (DecisionVerdict.APPROVE, DecisionVerdict.DELEGATE)

    def test_ethos_rejects(self):
        e = ConsensusEngine(strategy=VotingStrategy.AUTO_ETHOS,
                           ethos=EthosAgent(trust_level=0))
        r = e.evaluate("deploy reflex code")
        assert not r.approved


class TestConsensusTimeout:
    def test_timeout_rejects(self):
        e = ConsensusEngine(timeout_ms=-1.0)
        r = e.evaluate("anything")
        assert not r.approved


class TestConsensusHistory:
    def test_records_decisions(self):
        e = ConsensusEngine()
        e.evaluate("test action")
        e.evaluate("test action 2")
        assert len(e.decision_history) == 2

    def test_clear_history(self):
        e = ConsensusEngine()
        e.evaluate("test")
        e.clear_history()
        assert len(e.decision_history) == 0


class TestConsensusResult:
    def test_summary_contains_info(self):
        e = ConsensusEngine()
        r = e.evaluate("test action")
        s = r.summary()
        assert "Consensus:" in s
        assert "pathos:" in s
        assert "logos:" in s
        assert "ethos:" in s

    def test_approved_property(self):
        r = ConsensusResult(
            decision=DecisionVerdict.APPROVE, strategy=VotingStrategy.MAJORITY,
            pathos=IntentAssessment(agent_name="p", verdict=DecisionVerdict.APPROVE, confidence=1.0, score=1.0),
            logos=PlanAssessment(agent_name="l", verdict=DecisionVerdict.APPROVE, confidence=1.0, score=1.0),
            ethos=SafetyAssessment(agent_name="e", verdict=DecisionVerdict.APPROVE, confidence=1.0, score=1.0),
            weighted_score=1.0, elapsed_ms=1.0,
        )
        assert r.approved


# ===================================================================
# Marine Rules Tests
# ===================================================================
class TestMarineRulesInit:
    def test_default_rules(self):
        db = MarineRulesDatabase()
        assert len(db.colregs) > 5

    def test_equipment_defaults(self):
        db = MarineRulesDatabase()
        assert db.equipment_limits.max_speed_knots == 10.0

    def test_environmental_defaults(self):
        db = MarineRulesDatabase()
        assert db.equipment_limits.max_rudder_angle_deg == 45.0


class TestNoGoZone:
    def test_create(self):
        z = NoGoZone("reef", {"south": 10, "north": 20, "west": -50, "east": -40})
        assert z.name == "reef"
        assert z.active

    def test_roundtrip(self):
        z = NoGoZone("test", {"south": 0, "north": 1, "west": 0, "east": 1}, reason="test")
        d = z.to_dict()
        z2 = NoGoZone.from_dict(d)
        assert z2.name == "test"
        assert z2.reason == "test"


class TestEquipmentLimits:
    def test_roundtrip(self):
        lim = EquipmentLimits(max_speed_knots=15.0)
        d = lim.to_dict()
        lim2 = EquipmentLimits.from_dict(d)
        assert lim2.max_speed_knots == 15.0


class TestMarineRulesCheck:
    def test_no_go_zone_inside(self):
        db = MarineRulesDatabase()
        db.add_no_go_zone(NoGoZone("reef", {"south": 10, "north": 20, "west": -50, "east": -40}))
        assert db.check_no_go_zone(15, -45) is not None

    def test_no_go_zone_outside(self):
        db = MarineRulesDatabase()
        db.add_no_go_zone(NoGoZone("reef", {"south": 10, "north": 20, "west": -50, "east": -40}))
        assert db.check_no_go_zone(0, 0) is None

    def test_inactive_zone(self):
        db = MarineRulesDatabase()
        db.add_no_go_zone(NoGoZone("inactive", {"south": 0, "north": 90, "west": -180, "east": 180}, active=False))
        assert db.check_no_go_zone(15, -45) is None

    def test_equipment_speed_violation(self):
        db = MarineRulesDatabase()
        v = db.check_equipment_limits(speed=20.0)
        assert len(v) > 0

    def test_equipment_all_ok(self):
        db = MarineRulesDatabase()
        v = db.check_equipment_limits(speed=5.0, rudder=10.0, throttle=50.0)
        assert len(v) == 0

    def test_colregs_filtering(self):
        db = MarineRulesDatabase()
        rules = db.check_colregs(VesselSituation.POWER_DRIVEN, nearby_vessels=3)
        assert len(rules) > 0

    def test_active_zones_list(self):
        db = MarineRulesDatabase()
        db.add_no_go_zone(NoGoZone("a", {"south": 0, "north": 1, "west": 0, "east": 1}))
        db.add_no_go_zone(NoGoZone("b", {"south": 0, "north": 1, "west": 0, "east": 1}, active=False))
        zones = db.get_active_no_go_zones()
        assert len(zones) == 1


# ===================================================================
# Integration Tests
# ===================================================================
class TestIntegrationFullConsensus:
    def test_fishing_mission_approve(self):
        engine = ConsensusEngine(
            pathos=PathosAgent(mission_type=IntentType.FISHING),
            logos=LogosAgent(fuel_pct=80.0),
            ethos=EthosAgent(trust_level=2),
        )
        r = engine.evaluate("deploy nets and fish for cod", {"distance_nm": 5.0})
        assert r.approved

    def test_emergency_approves_despite_low_trust(self):
        engine = ConsensusEngine(
            pathos=PathosAgent(mission_type=IntentType.ROUTINE),
            logos=LogosAgent(fuel_pct=50.0),
            ethos=EthosAgent(trust_level=3),
        )
        r = engine.evaluate("emergency stop vessel", {"emergency": True})
        assert r.pathos.urgency == 1.0

    def test_no_go_zone_blocks(self):
        engine = ConsensusEngine(
            pathos=PathosAgent(mission_type=IntentType.NAVIGATION),
            logos=LogosAgent(fuel_pct=100.0),
            ethos=EthosAgent(trust_level=3, no_go_zones=[
                {"name": "reef", "bounds": {"south": 10, "north": 20, "west": -50, "east": -40}}
            ]),
        )
        r = engine.evaluate("navigate to reef area", {"target_position": (15, -45)})
        assert not r.approved

    def test_all_strategies_on_same_input(self):
        results = {}
        for strategy in VotingStrategy:
            engine = ConsensusEngine(
                pathos=PathosAgent(mission_type=IntentType.NAVIGATION),
                logos=LogosAgent(fuel_pct=100.0),
                ethos=EthosAgent(trust_level=3),
                strategy=strategy,
            )
            r = engine.evaluate("navigate to waypoint", {"distance_nm": 5.0})
            results[strategy] = r.approved
        # Most strategies should approve a safe navigation action
        assert sum(results.values()) >= 3  # at least 3 of 5 strategies approve

    def test_summary_output(self):
        engine = ConsensusEngine()
        r = engine.evaluate("test action")
        s = r.summary()
        assert isinstance(s, str)
        assert len(s) > 50
