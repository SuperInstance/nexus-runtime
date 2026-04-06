"""Tests for CommandValidator — syntax, semantics, safety, permissions, risk, alternatives."""

import pytest
from jetson.nl_commands.intent import Intent, IntentType
from jetson.nl_commands.executor import Command
from jetson.nl_commands.validation import (
    ValidationResult, RiskLevel, SafeAlternative, CommandValidator,
)


def _make_command(itype: IntentType, params=None, slots=None, text=""):
    intent = Intent(type=itype, slots=slots or {}, confidence=0.9, raw_text=text)
    return Command(intent=intent, parameters=params or {})


@pytest.fixture
def validator():
    return CommandValidator()


# ===================================================================
# ValidationResult dataclass
# ===================================================================

class TestValidationResult:
    def test_valid_result(self):
        r = ValidationResult(valid=True)
        assert r.valid
        assert r.errors == []
        assert r.warnings == []
        assert r.safe_to_execute

    def test_invalid_result(self):
        r = ValidationResult(valid=False, errors=["bad"], safe_to_execute=False)
        assert not r.valid
        assert not r.safe_to_execute

    def test_warnings_do_not_affect_validity(self):
        r = ValidationResult(valid=True, warnings=["caution"])
        assert r.valid
        assert r.safe_to_execute


# ===================================================================
# RiskLevel enum
# ===================================================================

class TestRiskLevel:
    def test_risk_levels(self):
        levels = list(RiskLevel)
        assert len(levels) == 5
        assert RiskLevel.NONE.value == 0
        assert RiskLevel.CRITICAL.value == 4

    def test_ordering(self):
        assert RiskLevel.NONE.value < RiskLevel.LOW.value
        assert RiskLevel.LOW.value < RiskLevel.MEDIUM.value
        assert RiskLevel.MEDIUM.value < RiskLevel.HIGH.value
        assert RiskLevel.HIGH.value < RiskLevel.CRITICAL.value


# ===================================================================
# SafeAlternative dataclass
# ===================================================================

class TestSafeAlternative:
    def test_alternative_creation(self):
        alt = SafeAlternative(
            command_text="station keep",
            description="Hold position",
            risk_level=RiskLevel.LOW,
            confidence=0.9,
        )
        assert alt.command_text == "station keep"
        assert alt.risk_level == RiskLevel.LOW


# ===================================================================
# CommandValidator.validate_syntax
# ===================================================================

class TestValidateSyntax:
    def test_valid_command(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        result = validator.validate_syntax(cmd)
        assert result.valid

    def test_none_command(self, validator):
        result = validator.validate_syntax(None)
        assert not result.valid

    def test_none_intent(self, validator):
        cmd = Command(intent=None)
        result = validator.validate_syntax(cmd)
        assert not result.valid

    def test_unknown_intent(self, validator):
        cmd = _make_command(IntentType.UNKNOWN)
        result = validator.validate_syntax(cmd)
        assert not result.valid

    def test_zero_timestamp(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        cmd.timestamp = 0
        result = validator.validate_syntax(cmd)
        assert not result.valid

    def test_negative_timestamp(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        cmd.timestamp = -1
        result = validator.validate_syntax(cmd)
        assert not result.valid

    def test_empty_text_warning(self, validator):
        cmd = _make_command(IntentType.NAVIGATE, text="   ")
        result = validator.validate_syntax(cmd)
        assert result.valid  # Still syntactically valid
        assert any("empty" in w.lower() for w in result.warnings)

    def test_all_intent_types_valid(self, validator):
        for itype in IntentType:
            if itype == IntentType.UNKNOWN:
                continue
            cmd = _make_command(itype)
            result = validator.validate_syntax(cmd)
            assert result.valid, f"Failed for {itype}"


# ===================================================================
# CommandValidator.validate_semantics
# ===================================================================

class TestValidateSemantics:
    def test_valid_navigation(self, validator):
        cmd = _make_command(IntentType.NAVIGATE, params={"destination": "alpha"})
        state = {"battery_level": 100, "engine_on": True, "gps_fix": True}
        result = validator.validate_semantics(cmd, state)
        assert result.valid

    def test_low_battery_blocks_navigation(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        state = {"battery_level": 5, "engine_on": True, "gps_fix": True}
        result = validator.validate_semantics(cmd, state)
        assert not result.valid
        assert any("battery" in e.lower() for e in result.errors)

    def test_engine_off_blocks_navigation(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        state = {"battery_level": 100, "engine_on": False, "gps_fix": True}
        result = validator.validate_semantics(cmd, state)
        assert not result.valid

    def test_no_gps_blocks_navigation(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        state = {"battery_level": 100, "engine_on": True, "gps_fix": False}
        result = validator.validate_semantics(cmd, state)
        assert not result.valid

    def test_emergency_allowed_low_battery(self, validator):
        cmd = _make_command(IntentType.EMERGENCY_STOP)
        state = {"battery_level": 5, "engine_on": True, "gps_fix": True}
        result = validator.validate_semantics(cmd, state)
        assert result.valid  # emergency stop is always allowed

    def test_return_home_gps_warning(self, validator):
        cmd = _make_command(IntentType.RETURN_HOME)
        state = {"battery_level": 100, "engine_on": True, "gps_fix": False}
        result = validator.validate_semantics(cmd, state)
        # Should warn but still be valid
        assert result.valid

    def test_speed_exceeds_harbor_limit(self, validator):
        cmd = _make_command(IntentType.SET_SPEED, params={"speed": 10}, slots={"speed": 10})
        state = {"battery_level": 100, "engine_on": True, "mode": "harbor"}
        result = validator.validate_semantics(cmd, state)
        assert not result.valid
        assert any("speed" in e.lower() or "limit" in e.lower() for e in result.errors)

    def test_speed_within_open_water(self, validator):
        cmd = _make_command(IntentType.SET_SPEED, params={"speed": 15}, slots={"speed": 15})
        state = {"battery_level": 100, "engine_on": True, "mode": "open_water"}
        result = validator.validate_semantics(cmd, state)
        assert result.valid

    def test_heading_out_of_range(self, validator):
        cmd = _make_command(IntentType.SET_HEADING, params={"heading_degrees": 400}, slots={"heading_degrees": 400})
        state = {"battery_level": 100, "engine_on": True}
        result = validator.validate_semantics(cmd, state)
        assert not result.valid

    def test_heading_valid(self, validator):
        cmd = _make_command(IntentType.SET_HEADING, params={"heading_degrees": 180}, slots={"heading_degrees": 180})
        state = {"battery_level": 100, "engine_on": True}
        result = validator.validate_semantics(cmd, state)
        assert result.valid

    def test_low_battery_warning(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        state = {"battery_level": 20, "engine_on": True, "gps_fix": True}
        result = validator.validate_semantics(cmd, state)
        assert result.valid
        assert any("battery" in w.lower() for w in result.warnings)

    def test_already_at_destination(self, validator):
        cmd = _make_command(IntentType.NAVIGATE, params={"destination": "home"})
        state = {
            "battery_level": 100, "engine_on": True, "gps_fix": True,
            "position": {"name": "home"},
        }
        result = validator.validate_semantics(cmd, state)
        assert not result.valid
        assert any("already" in e.lower() for e in result.errors)

    def test_depth_warning(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        state = {
            "battery_level": 100, "engine_on": True, "gps_fix": True,
            "depth": 0.3, "min_safe_depth": 0.5,
        }
        result = validator.validate_semantics(cmd, state)
        assert any("depth" in w.lower() for w in result.warnings)

    def test_none_command(self, validator):
        result = validator.validate_semantics(None, {})
        assert not result.valid


# ===================================================================
# CommandValidator.validate_safety
# ===================================================================

class TestValidateSafety:
    def test_valid_with_default_rules(self, validator):
        cmd = _make_command(IntentType.QUERY_STATUS)
        result = validator.validate_safety(cmd)
        assert result.valid

    def test_custom_safety_rule_error(self, validator):
        rule = {
            "name": "test_rule",
            "intent_type": IntentType.NAVIGATE,
            "severity": "error",
            "message": "Navigation blocked by test rule",
        }
        cmd = _make_command(IntentType.NAVIGATE)
        result = validator.validate_safety(cmd, [rule])
        assert not result.valid
        assert any("blocked" in e for e in result.errors)

    def test_custom_safety_rule_warning(self, validator):
        rule = {
            "name": "test_warning",
            "intent_type": IntentType.PATROL,
            "severity": "warning",
            "message": "Caution: test warning",
        }
        cmd = _make_command(IntentType.PATROL)
        result = validator.validate_safety(cmd, [rule])
        assert result.valid  # warnings don't block
        assert any("caution" in w.lower() for w in result.warnings)

    def test_rule_mismatched_intent(self, validator):
        rule = {
            "name": "nav_only_rule",
            "intent_type": IntentType.NAVIGATE,
            "severity": "error",
            "message": "Blocked",
        }
        cmd = _make_command(IntentType.SURVEY)
        result = validator.validate_safety(cmd, [rule])
        assert result.valid

    def test_none_command(self, validator):
        result = validator.validate_safety(None)
        assert not result.valid

    def test_multiple_rules(self, validator):
        rules = [
            {"name": "r1", "intent_type": IntentType.NAVIGATE, "severity": "warning", "message": "w1"},
            {"name": "r2", "intent_type": IntentType.NAVIGATE, "severity": "error", "message": "e1"},
        ]
        cmd = _make_command(IntentType.NAVIGATE)
        result = validator.validate_safety(cmd, rules)
        assert not result.valid
        assert len(result.errors) >= 1
        assert len(result.warnings) >= 1

    def test_default_rules_loaded(self, validator):
        rules = validator.get_safety_rules()
        assert len(rules) > 0


# ===================================================================
# CommandValidator.check_permissions
# ===================================================================

class TestCheckPermissions:
    def test_admin_all_access(self, validator):
        cmd = _make_command(IntentType.CONFIGURE)
        allowed, msg = validator.check_permissions(cmd, "admin")
        assert allowed

    def test_guest_limited_access(self, validator):
        cmd = _make_command(IntentType.CONFIGURE)
        allowed, msg = validator.check_permissions(cmd, "guest")
        assert not allowed

    def test_emergency_stop_all_roles(self, validator):
        cmd = _make_command(IntentType.EMERGENCY_STOP)
        for role in ["admin", "operator", "observer", "guest"]:
            allowed, _ = validator.check_permissions(cmd, role)
            assert allowed, f"Emergency stop should be allowed for {role}"

    def test_query_status_all_roles(self, validator):
        cmd = _make_command(IntentType.QUERY_STATUS)
        for role in ["admin", "operator", "observer", "guest"]:
            allowed, _ = validator.check_permissions(cmd, role)
            assert allowed, f"Query status should be allowed for {role}"

    def test_operator_can_navigate(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        allowed, _ = validator.check_permissions(cmd, "operator")
        assert allowed

    def test_observer_cannot_navigate(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        allowed, _ = validator.check_permissions(cmd, "observer")
        assert not allowed

    def test_unknown_role(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        allowed, _ = validator.check_permissions(cmd, "hacker")
        assert not allowed

    def test_none_command(self, validator):
        allowed, _ = validator.check_permissions(None, "admin")
        assert not allowed

    def test_message_content(self, validator):
        cmd = _make_command(IntentType.CONFIGURE)
        _, msg = validator.check_permissions(cmd, "guest")
        assert "denied" in msg.lower() or "permission" in msg.lower()

    def test_return_home_all_roles(self, validator):
        cmd = _make_command(IntentType.RETURN_HOME)
        for role in ["admin", "operator", "observer", "guest"]:
            allowed, _ = validator.check_permissions(cmd, role)
            assert allowed


# ===================================================================
# CommandValidator.estimate_risk
# ===================================================================

class TestEstimateRisk:
    def test_low_risk_query(self, validator):
        cmd = _make_command(IntentType.QUERY_STATUS)
        risk = validator.estimate_risk(cmd, {"sea_state": 0, "visibility": 10})
        assert risk == RiskLevel.LOW

    def test_high_risk_bad_conditions(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        conditions = {
            "sea_state": 5,
            "visibility": 0.5,
            "traffic_density": "critical",
            "distance_to_shore": 0.3,
            "battery_level": 10,
            "is_night": True,
        }
        risk = validator.estimate_risk(cmd, conditions)
        assert risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_calm_conditions_low_risk(self, validator):
        cmd = _make_command(IntentType.STATION_KEEP)
        conditions = {
            "sea_state": 0,
            "visibility": 10,
            "traffic_density": "low",
            "distance_to_shore": 10,
            "battery_level": 100,
            "is_night": False,
        }
        risk = validator.estimate_risk(cmd, conditions)
        assert risk == RiskLevel.LOW

    def test_medium_risk_navigation(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        conditions = {"sea_state": 2, "visibility": 5, "traffic_density": "medium"}
        risk = validator.estimate_risk(cmd, conditions)
        assert risk in (RiskLevel.MEDIUM, RiskLevel.LOW, RiskLevel.HIGH)

    def test_none_command(self, validator):
        risk = validator.estimate_risk(None, {})
        assert risk == RiskLevel.CRITICAL

    def test_unknown_intent_high_risk(self, validator):
        cmd = _make_command(IntentType.UNKNOWN)
        risk = validator.estimate_risk(cmd, {"sea_state": 0})
        assert risk == RiskLevel.CRITICAL

    def test_night_adds_risk(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        day = validator.estimate_risk(cmd, {"sea_state": 1, "visibility": 5, "is_night": False})
        night = validator.estimate_risk(cmd, {"sea_state": 1, "visibility": 5, "is_night": True})
        assert night.value >= day.value

    def test_low_battery_adds_risk(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        good = validator.estimate_risk(cmd, {"sea_state": 1, "visibility": 5, "battery_level": 90})
        low = validator.estimate_risk(cmd, {"sea_state": 1, "visibility": 5, "battery_level": 10})
        assert low.value >= good.value


# ===================================================================
# CommandValidator.compute_safe_alternatives
# ===================================================================

class TestComputeSafeAlternatives:
    def test_navigate_alternatives(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        alts = validator.compute_safe_alternatives(cmd)
        assert len(alts) > 0
        types = [a.command_text for a in alts]
        assert any("station keep" in t or "return home" in t for t in types)

    def test_patrol_alternatives(self, validator):
        cmd = _make_command(IntentType.PATROL)
        alts = validator.compute_safe_alternatives(cmd)
        assert len(alts) > 0

    def test_survey_alternatives(self, validator):
        cmd = _make_command(IntentType.SURVEY)
        alts = validator.compute_safe_alternatives(cmd)
        assert len(alts) > 0

    def test_configure_alternatives(self, validator):
        cmd = _make_command(IntentType.CONFIGURE)
        alts = validator.compute_safe_alternatives(cmd)
        assert len(alts) > 0

    def test_none_command(self, validator):
        alts = validator.compute_safe_alternatives(None)
        assert alts == []

    def test_alternatives_have_descriptions(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        alts = validator.compute_safe_alternatives(cmd)
        for alt in alts:
            assert alt.description != ""

    def test_alternatives_low_risk(self, validator):
        cmd = _make_command(IntentType.NAVIGATE)
        alts = validator.compute_safe_alternatives(cmd)
        for alt in alts:
            assert alt.risk_level in (RiskLevel.NONE, RiskLevel.LOW)

    def test_high_speed_alternative(self, validator):
        cmd = _make_command(IntentType.SET_SPEED, params={"speed": 20}, slots={"speed": 20})
        alts = validator.compute_safe_alternatives(cmd)
        assert len(alts) > 0


# ===================================================================
# CommandValidator.add_safety_rule / get_safety_rules
# ===================================================================

class TestSafetyRulesManagement:
    def test_add_rule(self, validator):
        initial_count = len(validator.get_safety_rules())
        validator.add_safety_rule({
            "name": "custom_rule",
            "intent_type": IntentType.NAVIGATE,
            "severity": "warning",
            "message": "Custom safety rule",
        })
        assert len(validator.get_safety_rules()) == initial_count + 1

    def test_get_safety_rules_returns_list(self, validator):
        rules = validator.get_safety_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_get_safety_rules_is_copy(self, validator):
        rules1 = validator.get_safety_rules()
        rules1.append({"fake": True})
        rules2 = validator.get_safety_rules()
        assert len(rules2) == len(rules1) - 1  # shouldn't contain the fake
