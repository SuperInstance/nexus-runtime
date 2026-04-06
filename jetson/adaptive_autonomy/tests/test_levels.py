"""Tests for jetson.adaptive_autonomy.levels  — AutonomyLevel, LevelCapabilities, AutonomyLevelManager."""

import pytest

from jetson.adaptive_autonomy.levels import (
    AutonomyLevel,
    LevelCapabilities,
    AutonomyLevelManager,
)


# ── AutonomyLevel enum ────────────────────────────────────────────

class TestAutonomyLevel:
    def test_enum_has_six_levels(self):
        assert len(AutonomyLevel) == 6

    def test_manual_is_zero(self):
        assert AutonomyLevel.MANUAL == 0

    def test_autonomous_is_five(self):
        assert AutonomyLevel.AUTONOMOUS == 5

    def test_ordering_manual_lt_autonomous(self):
        assert AutonomyLevel.MANUAL < AutonomyLevel.AUTONOMOUS

    def test_str_returns_name(self):
        assert str(AutonomyLevel.SEMI_AUTO) == "SEMI_AUTO"

    def test_int_values_sequential(self):
        for i, lv in enumerate(AutonomyLevel):
            assert lv.value == i

    def test_equality(self):
        assert AutonomyLevel.FULL_AUTO == AutonomyLevel.FULL_AUTO

    def test_inequality(self):
        assert AutonomyLevel.MANUAL != AutonomyLevel.ASSISTED

    def test_all_names_unique(self):
        names = [lv.name for lv in AutonomyLevel]
        assert len(names) == len(set(names))

    def test_iterable(self):
        levels = list(AutonomyLevel)
        assert len(levels) == 6


# ── LevelCapabilities ─────────────────────────────────────────────

class TestLevelCapabilities:
    def test_defaults(self):
        caps = LevelCapabilities()
        assert caps.allowed_operations == []
        assert caps.required_human_approval == []
        assert caps.max_risk_tolerance == 0.0
        assert caps.decision_authority == 0.0

    def test_custom_values(self):
        caps = LevelCapabilities(
            allowed_operations=["nav"],
            required_human_approval=["critical"],
            max_risk_tolerance=0.5,
            decision_authority=0.75,
        )
        assert caps.allowed_operations == ["nav"]
        assert caps.max_risk_tolerance == 0.5

    def test_mutable_operations(self):
        caps = LevelCapabilities(allowed_operations=["a"])
        caps.allowed_operations.append("b")
        assert "b" in caps.allowed_operations


# ── AutonomyLevelManager ──────────────────────────────────────────

class TestAutonomyLevelManager:

    @pytest.fixture()
    def mgr(self):
        return AutonomyLevelManager()

    # get_capabilities
    def test_get_capabilities_returns_capabilities(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.MANUAL)
        assert isinstance(caps, LevelCapabilities)

    def test_manual_capabilities_read_sensors(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.MANUAL)
        assert "read_sensors" in caps.allowed_operations

    def test_autonomous_has_full_control(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.AUTONOMOUS)
        assert "full_control" in caps.allowed_operations

    def test_manual_no_full_control(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.MANUAL)
        assert "full_control" not in caps.allowed_operations

    def test_semi_auto_has_adjust_parameters(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.SEMI_AUTO)
        assert "adjust_parameters" in caps.allowed_operations

    def test_assisted_no_navigate(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.ASSISTED)
        assert "navigate" not in caps.allowed_operations

    # is_operation_allowed
    def test_is_operation_allowed_true(self, mgr):
        assert mgr.is_operation_allowed("log_data", AutonomyLevel.MANUAL)

    def test_is_operation_allowed_false(self, mgr):
        assert not mgr.is_operation_allowed("navigate", AutonomyLevel.MANUAL)

    def test_is_operation_allowed_autonomous(self, mgr):
        assert mgr.is_operation_allowed("self_diagnose", AutonomyLevel.AUTONOMOUS)

    def test_is_operation_allowed_unknown(self, mgr):
        assert not mgr.is_operation_allowed("nonexistent", AutonomyLevel.MANUAL)

    # get_max_risk_tolerance
    def test_manual_risk_zero(self, mgr):
        assert mgr.get_max_risk_tolerance(AutonomyLevel.MANUAL) == 0.0

    def test_autonomous_risk_high(self, mgr):
        assert mgr.get_max_risk_tolerance(AutonomyLevel.AUTONOMOUS) == 0.9

    def test_risk_increases_with_level(self, mgr):
        for prev, cur in zip(AutonomyLevel, list(AutonomyLevel)[1:]):
            assert mgr.get_max_risk_tolerance(prev) <= mgr.get_max_risk_tolerance(cur)

    # compute_decision_authority
    def test_manual_authority_zero_pct(self, mgr):
        assert mgr.compute_decision_authority(AutonomyLevel.MANUAL) == 0.0

    def test_autonomous_authority_100_pct(self, mgr):
        assert mgr.compute_decision_authority(AutonomyLevel.AUTONOMOUS) == 100.0

    def test_authority_returns_percentage(self, mgr):
        val = mgr.compute_decision_authority(AutonomyLevel.SEMI_AUTO)
        assert 0.0 <= val <= 100.0

    # list_level_operations
    def test_list_operations_returns_list(self, mgr):
        ops = mgr.list_level_operations(AutonomyLevel.MANUAL)
        assert isinstance(ops, list)

    def test_list_operations_copy(self, mgr):
        ops1 = mgr.list_level_operations(AutonomyLevel.MANUAL)
        ops2 = mgr.list_level_operations(AutonomyLevel.MANUAL)
        assert ops1 == ops2
        assert ops1 is not ops2

    def test_autonomous_more_ops_than_manual(self, mgr):
        assert (
            len(mgr.list_level_operations(AutonomyLevel.AUTONOMOUS))
            > len(mgr.list_level_operations(AutonomyLevel.MANUAL))
        )

    # compare_levels
    def test_compare_higher(self, mgr):
        assert mgr.compare_levels(AutonomyLevel.FULL_AUTO, AutonomyLevel.MANUAL) == "higher"

    def test_compare_lower(self, mgr):
        assert mgr.compare_levels(AutonomyLevel.MANUAL, AutonomyLevel.FULL_AUTO) == "lower"

    def test_compare_equal(self, mgr):
        assert mgr.compare_levels(AutonomyLevel.SEMI_AUTO, AutonomyLevel.SEMI_AUTO) == "equal"

    def test_compare_all_levels_ordered(self, mgr):
        levels = list(AutonomyLevel)
        for i in range(len(levels) - 1):
            assert mgr.compare_levels(levels[i + 1], levels[i]) == "higher"

    # required_human_approval
    def test_autonomous_no_approval_required(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.AUTONOMOUS)
        assert caps.required_human_approval == []

    def test_manual_all_approval_required(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.MANUAL)
        assert "all" in caps.required_human_approval

    def test_assisted_all_approval_required(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.ASSISTED)
        assert "all" in caps.required_human_approval

    def test_full_auto_only_critical_approval(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.FULL_AUTO)
        assert "critical" in caps.required_human_approval
        assert "hazardous" not in caps.required_human_approval

    def test_auto_with_supervision_has_plan_path(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.AUTO_WITH_SUPERVISION)
        assert "plan_path" in caps.allowed_operations

    def test_auto_with_supervision_hazardous_approval(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.AUTO_WITH_SUPERVISION)
        assert "hazardous" in caps.required_human_approval

    def test_all_levels_have_display_info(self, mgr):
        for lv in AutonomyLevel:
            caps = mgr.get_capabilities(lv)
            assert "display_info" in caps.allowed_operations

    def test_authority_increases_with_level(self, mgr):
        levels = list(AutonomyLevel)
        for prev, cur in zip(levels, levels[1:]):
            prev_auth = mgr.compute_decision_authority(prev)
            cur_auth = mgr.compute_decision_authority(cur)
            assert cur_auth >= prev_auth

    def test_semi_auto_has_execute_approved(self, mgr):
        caps = mgr.get_capabilities(AutonomyLevel.SEMI_AUTO)
        assert "execute_approved_actions" in caps.allowed_operations

    def test_assisted_risk_0_1(self, mgr):
        assert mgr.get_max_risk_tolerance(AutonomyLevel.ASSISTED) == 0.1

    def test_level_capabilities_dataclass_repr(self):
        caps = LevelCapabilities(max_risk_tolerance=0.5)
        assert "max_risk_tolerance=0.5" in repr(caps)

    def test_get_capabilities_idempotent(self, mgr):
        c1 = mgr.get_capabilities(AutonomyLevel.ASSISTED)
        c2 = mgr.get_capabilities(AutonomyLevel.ASSISTED)
        assert c1 is c2  # class-level cache
