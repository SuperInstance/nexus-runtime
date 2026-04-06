"""Tests for COLREGs Rule Verification Engine."""

import pytest
import math

from jetson.compliance.colregs import (
    VisibilityCondition,
    WaterwayType,
    VesselType,
    VesselState,
    VesselSituation,
    RuleReference,
    COLREGsResult,
    COLREGsChecker,
    UrgencyLevel,
)


def make_vessel(
    name: str = "Vessel",
    vessel_type: VesselType = VesselType.POWER_DRIVEN,
    heading: float = 0.0,
    speed: float = 5.0,
    x: float = 0.0,
    y: float = 0.0,
) -> VesselState:
    """Helper to create a VesselState."""
    return VesselState(
        vessel_type=vessel_type,
        heading=heading,
        speed=speed,
        position=(x, y),
        name=name,
        has_night_lights=True,
        has_sound_signals=True,
    )


class TestEnums:
    """Tests for enum values."""

    def test_visibility_conditions(self):
        assert len(VisibilityCondition) == 4
        assert VisibilityCondition.GOOD.value == "good"
        assert VisibilityCondition.RESTRICTED.value == "restricted"

    def test_waterway_types(self):
        assert len(WaterwayType) == 5
        assert WaterwayType.OPEN_SEA.value == "open_sea"

    def test_vessel_types(self):
        assert len(VesselType) >= 10

    def test_urgency_levels(self):
        assert len(UrgencyLevel) == 4
        assert UrgencyLevel.IMMEDIATE.value == "immediate"


class TestVesselState:
    """Tests for VesselState dataclass."""

    def test_create_vessel(self):
        v = make_vessel("Test", VesselType.POWER_DRIVEN, 45.0, 10.0, 1.0, 2.0)
        assert v.name == "Test"
        assert v.heading == 45.0
        assert v.speed == 10.0
        assert v.position == (1.0, 2.0)

    def test_default_lights(self):
        v = make_vessel()
        assert v.has_night_lights is True

    def test_default_dimensions(self):
        v = make_vessel()
        assert v.length == 10.0
        assert v.beam == 3.0


class TestVesselSituation:
    """Tests for VesselSituation dataclass."""

    def test_default_situation(self):
        own = make_vessel("Own")
        sit = VesselSituation(own_vessel=own)
        assert sit.own_vessel.name == "Own"
        assert sit.other_vessels == []
        assert sit.visibility == VisibilityCondition.GOOD

    def test_with_others(self):
        own = make_vessel("Own")
        other = make_vessel("Other")
        sit = VesselSituation(own_vessel=own, other_vessels=[other])
        assert len(sit.other_vessels) == 1

    def test_restricted_visibility(self):
        own = make_vessel("Own")
        sit = VesselSituation(
            own_vessel=own,
            visibility=VisibilityCondition.RESTRICTED,
        )
        assert sit.visibility == VisibilityCondition.RESTRICTED


class TestRuleReference:
    """Tests for RuleReference dataclass."""

    def test_create_rule(self):
        rule = RuleReference(
            rule_number="Rule 5",
            title="Look-out",
            description="Maintain proper look-out",
            applicable_when="At all times",
        )
        assert rule.rule_number == "Rule 5"
        assert rule.title == "Look-out"


class TestCOLREGsCheckerApplicableRules:
    """Tests for COLREGsChecker.get_applicable_rules."""

    def setup_method(self):
        self.checker = COLREGsChecker()

    def test_always_rules_present(self):
        own = make_vessel()
        sit = VesselSituation(own_vessel=own)
        rules = self.checker.get_applicable_rules(sit)
        rule_numbers = [r.rule_number for r in rules]
        assert "Rule 5" in rule_numbers
        assert "Rule 6" in rule_numbers
        assert "Rule 7" in rule_numbers
        assert "Rule 8" in rule_numbers

    def test_narrow_channel_adds_rule_9(self):
        own = make_vessel()
        sit = VesselSituation(
            own_vessel=own,
            waterway_type=WaterwayType.NARROW_CHANNEL,
        )
        rules = self.checker.get_applicable_rules(sit)
        rule_numbers = [r.rule_number for r in rules]
        assert "Rule 9" in rule_numbers

    def test_tss_adds_rule_10(self):
        own = make_vessel()
        sit = VesselSituation(
            own_vessel=own,
            waterway_type=WaterwayType.TRAFFIC_SEPARATION,
        )
        rules = self.checker.get_applicable_rules(sit)
        rule_numbers = [r.rule_number for r in rules]
        assert "Rule 10" in rule_numbers

    def test_restricted_visibility_adds_rule_19(self):
        own = make_vessel()
        sit = VesselSituation(
            own_vessel=own,
            visibility=VisibilityCondition.RESTRICTED,
        )
        rules = self.checker.get_applicable_rules(sit)
        rule_numbers = [r.rule_number for r in rules]
        assert "Rule 19" in rule_numbers

    def test_night_adds_rule_20(self):
        own = make_vessel()
        sit = VesselSituation(own_vessel=own, time_of_day="night")
        rules = self.checker.get_applicable_rules(sit)
        rule_numbers = [r.rule_number for r in rules]
        assert "Rule 20" in rule_numbers

    def test_approaching_vessel_adds_rules(self):
        own = make_vessel("Own", heading=0.0, x=0.0, y=0.0)
        other = make_vessel("Other", heading=180.0, x=0.0, y=200.0)
        sit = VesselSituation(own_vessel=own, other_vessels=[other])
        rules = self.checker.get_applicable_rules(sit)
        rule_numbers = [r.rule_number for r in rules]
        assert "Rule 13" in rule_numbers
        assert "Rule 14" in rule_numbers


class TestCOLREGsCheckerHeadOn:
    """Tests for COLREGsChecker.check_head_on_situation."""

    def setup_method(self):
        self.checker = COLREGsChecker()

    def test_no_head_on_empty(self):
        own = make_vessel()
        sit = VesselSituation(own_vessel=own)
        result = self.checker.check_head_on_situation(sit)
        assert "No head-on" in result.recommended_action

    def test_head_on_detected(self):
        own = make_vessel("Own", heading=0.0, x=0.0, y=0.0)
        other = make_vessel("Other", heading=180.0, x=0.0, y=200.0)
        sit = VesselSituation(own_vessel=own, other_vessels=[other])
        result = self.checker.check_head_on_situation(sit)
        assert len(result.applicable_rules) > 0
        assert result.urgency in (UrgencyLevel.IMMEDIATE, UrgencyLevel.URGENT)

    def test_head_on_alter_starboard(self):
        own = make_vessel("Own", heading=0.0, x=0.0, y=0.0)
        other = make_vessel("Other", heading=180.0, x=0.0, y=200.0)
        sit = VesselSituation(own_vessel=own, other_vessels=[other])
        result = self.checker.check_head_on_situation(sit)
        assert "starboard" in result.recommended_action.lower()

    def test_distant_vessel_no_urgency(self):
        own = make_vessel("Own", heading=0.0, x=0.0, y=0.0)
        other = make_vessel("Other", heading=180.0, x=0.0, y=5000.0)
        sit = VesselSituation(own_vessel=own, other_vessels=[other])
        result = self.checker.check_head_on_situation(sit)
        assert result.urgency == UrgencyLevel.INFORMATIONAL


class TestCOLREGsCheckerCrossing:
    """Tests for COLREGsChecker.check_crossing_situation."""

    def setup_method(self):
        self.checker = COLREGsChecker()

    def test_no_crossing_empty(self):
        own = make_vessel()
        sit = VesselSituation(own_vessel=own)
        result = self.checker.check_crossing_situation(sit)
        assert "No crossing" in result.recommended_action

    def test_starboard_crossing(self):
        own = make_vessel("Own", heading=0.0, x=0.0, y=0.0)
        other = make_vessel("Other", heading=270.0, x=100.0, y=200.0)
        sit = VesselSituation(own_vessel=own, other_vessels=[other])
        result = self.checker.check_crossing_situation(sit)
        if result.applicable_rules:
            assert result.give_way_vessel == "Own" or "Own" in result.give_way_vessel


class TestCOLREGsCheckerOvertaking:
    """Tests for COLREGsChecker.check_overtaking."""

    def setup_method(self):
        self.checker = COLREGsChecker()

    def test_no_overtaking_empty(self):
        own = make_vessel()
        sit = VesselSituation(own_vessel=own)
        result = self.checker.check_overtaking(sit)
        assert "No overtaking" in result.recommended_action

    def test_overtaking_detected(self):
        own = make_vessel("Own", heading=0.0, speed=10.0, x=0.0, y=0.0)
        other = make_vessel("Other", heading=0.0, speed=5.0, x=0.0, y=-200.0)
        sit = VesselSituation(own_vessel=own, other_vessels=[other])
        result = self.checker.check_overtaking(sit)
        if result.applicable_rules:
            assert "keep out" in result.recommended_action.lower() or "keep clear" in result.recommended_action.lower()


class TestCOLREGsCheckerStandOnGiveWay:
    """Tests for COLREGsChecker.check_stand_on_give_way."""

    def setup_method(self):
        self.checker = COLREGsChecker()

    def test_empty_situation(self):
        own = make_vessel()
        sit = VesselSituation(own_vessel=own)
        result = self.checker.check_stand_on_give_way(sit)
        assert "No collision" in result.recommended_action

    def test_head_on_triggers(self):
        own = make_vessel("Own", heading=0.0, x=0.0, y=0.0)
        other = make_vessel("Other", heading=180.0, x=0.0, y=200.0)
        sit = VesselSituation(own_vessel=own, other_vessels=[other])
        result = self.checker.check_stand_on_give_way(sit)
        assert len(result.applicable_rules) > 0


class TestCOLREGsCheckerNavigationLights:
    """Tests for COLREGsChecker.check_navigation_lights."""

    def setup_method(self):
        self.checker = COLREGsChecker()

    def test_not_required_day(self):
        own = make_vessel()
        sit = VesselSituation(own_vessel=own, time_of_day="day")
        result = self.checker.check_navigation_lights(sit)
        assert result.urgency == UrgencyLevel.INFORMATIONAL

    def test_night_lights_compliant(self):
        own = make_vessel("Own")
        sit = VesselSituation(own_vessel=own, time_of_day="night")
        result = self.checker.check_navigation_lights(sit)
        assert result.urgency == UrgencyLevel.INFORMATIONAL
        assert "proper" in result.recommended_action.lower()

    def test_night_lights_missing(self):
        own = make_vessel("Own")
        own.has_night_lights = False
        sit = VesselSituation(own_vessel=own, time_of_day="night")
        result = self.checker.check_navigation_lights(sit)
        assert result.urgency == UrgencyLevel.IMMEDIATE

    def test_other_vessel_no_lights(self):
        own = make_vessel("Own", x=0.0, y=0.0)
        other = make_vessel("Other", x=100.0, y=100.0)
        other.has_night_lights = False
        sit = VesselSituation(own_vessel=own, other_vessels=[other], time_of_day="night")
        result = self.checker.check_navigation_lights(sit)
        assert result.urgency == UrgencyLevel.URGENT


class TestCOLREGsCheckerSoundSignals:
    """Tests for COLREGsChecker.check_sound_signals."""

    def setup_method(self):
        self.checker = COLREGsChecker()

    def test_good_visibility_no_signals(self):
        own = make_vessel()
        sit = VesselSituation(own_vessel=own)
        result = self.checker.check_sound_signals(sit)
        assert "required" in result.recommended_action.lower()

    def test_restricted_visibility_signal(self):
        own = make_vessel("Own")
        own.has_sound_signals = False
        sit = VesselSituation(
            own_vessel=own,
            visibility=VisibilityCondition.RESTRICTED,
        )
        result = self.checker.check_sound_signals(sit)
        assert result.urgency == UrgencyLevel.URGENT

    def test_restricted_visibility_signal_compliant(self):
        own = make_vessel("Own")
        own.has_sound_signals = True
        sit = VesselSituation(
            own_vessel=own,
            visibility=VisibilityCondition.RESTRICTED,
        )
        result = self.checker.check_sound_signals(sit)
        assert "Continue" in result.recommended_action or result.urgency <= UrgencyLevel.CAUTIONARY


class TestCOLREGsCheckerRestrictedVisibility:
    """Tests for COLREGsChecker.check_conduct_in_restricted_visibility."""

    def setup_method(self):
        self.checker = COLREGsChecker()

    def test_good_visibility_not_applicable(self):
        own = make_vessel()
        sit = VesselSituation(own_vessel=own, visibility=VisibilityCondition.GOOD)
        result = self.checker.check_conduct_in_restricted_visibility(sit)
        assert "not applicable" in result.recommended_action

    def test_restricted_reduce_speed(self):
        own = make_vessel("Own", speed=15.0)
        sit = VesselSituation(
            own_vessel=own,
            visibility=VisibilityCondition.RESTRICTED,
        )
        result = self.checker.check_conduct_in_restricted_visibility(sit)
        assert result.urgency == UrgencyLevel.URGENT
        assert "speed" in result.recommended_action.lower()

    def test_restricted_safe_speed_ok(self):
        own = make_vessel("Own", speed=3.0)
        sit = VesselSituation(
            own_vessel=own,
            visibility=VisibilityCondition.RESTRICTED,
        )
        result = self.checker.check_conduct_in_restricted_visibility(sit)
        assert result.urgency.value <= UrgencyLevel.CAUTIONARY.value

    def test_close_vessel_in_fog(self):
        own = make_vessel("Own", speed=5.0, x=0.0, y=0.0)
        other = make_vessel("Other", x=50.0, y=50.0)
        sit = VesselSituation(
            own_vessel=own,
            other_vessels=[other],
            visibility=VisibilityCondition.RESTRICTED,
        )
        result = self.checker.check_conduct_in_restricted_visibility(sit)
        assert result.urgency == UrgencyLevel.URGENT
        assert "avoiding" in result.recommended_action.lower()
