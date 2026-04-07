"""Tests for the NEXUS AAB Roles module (15+ tests)."""

import pytest

from nexus.aab.roles import Role, RoleType, RoleRegistry, RoleAssignment


@pytest.fixture
def registry():
    return RoleRegistry()


class TestRoleType:
    def test_role_count(self):
        assert len(RoleType) == 12

    def test_key_roles_exist(self):
        assert RoleType.PILOT.value == "pilot"
        assert RoleType.NAVIGATOR.value == "navigator"
        assert RoleType.SAFETY_OFFICER.value == "safety_officer"
        assert RoleType.MISSION_COMMANDER.value == "mission_commander"


class TestRole:
    def test_create_role(self):
        role = Role(role_type=RoleType.PILOT, required_capabilities={"navigation": 0.8})
        assert role.name == "Pilot"
        assert role.max_assignees == 1

    def test_custom_name(self):
        role = Role(role_type=RoleType.PILOT, name="Chief Pilot")
        assert role.name == "Chief Pilot"

    def test_matches_capabilities(self):
        role = Role(role_type=RoleType.PILOT, required_capabilities={"navigation": 0.8})
        agent_caps = {"navigation": 0.9, "sensing": 0.5}
        score = role.matches_capabilities(agent_caps)
        assert score > 0.0

    def test_perfect_match(self):
        role = Role(role_type=RoleType.PILOT, required_capabilities={"navigation": 0.8})
        agent_caps = {"navigation": 0.8}
        assert role.matches_capabilities(agent_caps) == 1.0

    def test_no_match(self):
        role = Role(role_type=RoleType.PILOT, required_capabilities={"navigation": 0.9})
        agent_caps = {"navigation": 0.1}
        score = role.matches_capabilities(agent_caps)
        assert score < 0.5


class TestRoleRegistry:
    def test_default_roles(self, registry):
        roles = registry.list_roles()
        assert len(roles) == 12

    def test_get_role(self, registry):
        role = registry.get_role(RoleType.PILOT)
        assert role is not None
        assert role.role_type == RoleType.PILOT

    def test_define_custom_role(self, registry):
        custom = Role(role_type=RoleType.PILOT, required_capabilities={"custom": 0.9})
        registry.define_role(custom)
        role = registry.get_role(RoleType.PILOT)
        assert "custom" in role.required_capabilities

    def test_assign_role(self, registry):
        assignment = registry.assign_role(RoleType.PILOT, "AUV-001")
        assert assignment is not None
        assert assignment.agent_id == "AUV-001"
        assert assignment.active is True

    def test_singleton_role_limit(self, registry):
        registry.assign_role(RoleType.NAVIGATOR, "AUV-001")
        # Navigator is singleton — second assignment should fail
        result = registry.assign_role(RoleType.NAVIGATOR, "AUV-002")
        assert result is None

    def test_release_role(self, registry):
        registry.assign_role(RoleType.PILOT, "AUV-001")
        assert registry.release_role("AUV-001", RoleType.PILOT) is True

    def test_release_nonexistent(self, registry):
        assert registry.release_role("NOPE", RoleType.PILOT) is False

    def test_release_all_roles(self, registry):
        registry.assign_role(RoleType.PILOT, "AUV-001")
        registry.assign_role(RoleType.SENSOR_OPERATOR, "AUV-001")
        count = registry.release_all_roles("AUV-001")
        assert count == 2

    def test_get_agent_roles(self, registry):
        registry.assign_role(RoleType.PILOT, "AUV-001")
        registry.assign_role(RoleType.SENSOR_OPERATOR, "AUV-001")
        roles = registry.get_agent_roles("AUV-001")
        assert len(roles) == 2

    def test_get_assignments_for_role(self, registry):
        registry.assign_role(RoleType.PILOT, "AUV-001")
        registry.assign_role(RoleType.PILOT, "AUV-002")
        assignments = registry.get_assignments_for_role(RoleType.PILOT)
        assert len(assignments) == 2

    def test_find_best_candidate(self, registry):
        candidates = {
            "AUV-001": {"navigation": 0.95, "speed": 0.7},
            "AUV-002": {"navigation": 0.5, "speed": 0.3},
        }
        result = registry.find_best_candidate(RoleType.PILOT, candidates)
        assert result is not None
        agent_id, score = result
        assert agent_id == "AUV-001"
        assert score > 0.5

    def test_find_best_excludes_assigned(self, registry):
        registry.assign_role(RoleType.PILOT, "AUV-001")
        candidates = {
            "AUV-001": {"navigation": 0.99},
            "AUV-002": {"navigation": 0.5},
        }
        result = registry.find_best_candidate(RoleType.PILOT, candidates)
        assert result is not None
        assert result[0] == "AUV-002"

    def test_role_rotation(self, registry):
        registry.assign_role(RoleType.NAVIGATOR, "AUV-001")
        assert registry.rotate_role(RoleType.NAVIGATOR, "AUV-002") is True
        assignments = registry.get_assignments_for_role(RoleType.NAVIGATOR)
        active = [a for a in assignments if a.active]
        assert len(active) == 1
        assert active[0].agent_id == "AUV-002"

    def test_rotation_history(self, registry):
        registry.assign_role(RoleType.NAVIGATOR, "AUV-001")
        registry.rotate_role(RoleType.NAVIGATOR, "AUV-002")
        history = registry.get_rotation_history()
        assert len(history) == 1

    def test_max_assignees(self, registry):
        # PILOT has max_assignees=5 by default
        for i in range(5):
            result = registry.assign_role(RoleType.PILOT, f"AUV-{i:03d}")
            assert result is not None
        result = registry.assign_role(RoleType.PILOT, "AUV-999")
        assert result is None
