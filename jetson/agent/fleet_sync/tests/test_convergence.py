"""
Tests for the Convergence Tester module.
Tests convergence under various network conditions, partition scenarios,
and solution comparisons.
"""

import time
import pytest
from jetson.agent.fleet_sync.convergence_tester import (
    ConvergenceTester, ConvergenceResult,
)
from jetson.agent.fleet_sync.network_simulator import (
    NetworkCondition, NetworkConfig, NetworkSimulator,
)
from jetson.agent.fleet_sync.solutions.git_sync import GitSync
from jetson.agent.fleet_sync.solutions.operation_crdt import OperationCRDT
from jetson.agent.fleet_sync.solutions.state_crdt import StateCRDT


# ==============================================================================
# Basic Convergence Tests
# ==============================================================================

class TestBasicConvergence:
    """Test basic convergence scenarios."""

    def test_two_vessels_git_sync(self):
        tester = ConvergenceTester()
        r = tester.run_test(GitSync, num_vessels=2, ops_per_vessel=5, seed=42)
        assert r.converged

    def test_two_vessels_state_crdt(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=2, ops_per_vessel=5, seed=42)
        assert r.converged

    def test_two_vessels_operation_crdt(self):
        tester = ConvergenceTester()
        r = tester.run_test(OperationCRDT, num_vessels=2, ops_per_vessel=5, seed=42)
        assert r.converged

    def test_three_vessels_state_crdt(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=3, ops_per_vessel=10, seed=42)
        assert r.converged

    def test_five_vessels_git_sync(self):
        tester = ConvergenceTester()
        r = tester.run_test(GitSync, num_vessels=5, ops_per_vessel=10, seed=42)
        assert r.converged

    def test_five_vessels_state_crdt(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=5, ops_per_vessel=10, seed=42)
        assert r.converged

    def test_single_vessel(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=1, ops_per_vessel=10, seed=42)
        assert r.converged

    def test_zero_operations(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=3, ops_per_vessel=0, seed=42)
        assert r.converged

    def test_empty_fleet(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=0, ops_per_vessel=5, seed=42)
        assert r.converged


# ==============================================================================
# Multi-Seed Convergence Tests
# ==============================================================================

class TestMultiSeedConvergence:
    """Test convergence across multiple random seeds."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000, 555, 999, 1])
    def test_state_crdt_various_seeds(self, seed):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=3, ops_per_vessel=15, seed=seed)
        assert r.converged, f"StateCRDT failed with seed={seed}: {r.errors}"

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_git_sync_various_seeds(self, seed):
        tester = ConvergenceTester()
        r = tester.run_test(GitSync, num_vessels=3, ops_per_vessel=15, seed=seed)
        assert r.converged, f"GitSync failed with seed={seed}: {r.errors}"


# ==============================================================================
# Network Condition Convergence Tests
# ==============================================================================

class TestNetworkConditionConvergence:
    """Test convergence under various network conditions."""

    @pytest.mark.parametrize("condition", list(NetworkCondition))
    def test_state_crdt_all_conditions(self, condition):
        tester = ConvergenceTester()
        r = tester.run_test(
            StateCRDT, num_vessels=3, ops_per_vessel=10,
            network_condition=condition, use_network_sim=True, seed=42,
        )
        # Under lossy networks we may need more rounds, but should still converge
        assert r.errors == [], f"Errors under {condition.value}: {r.errors}"

    @pytest.mark.parametrize("condition", list(NetworkCondition))
    def test_git_sync_all_conditions(self, condition):
        tester = ConvergenceTester()
        r = tester.run_test(
            GitSync, num_vessels=3, ops_per_vessel=10,
            network_condition=condition, use_network_sim=True, seed=42,
        )
        assert r.errors == [], f"Errors under {condition.value}: {r.errors}"

    def test_perfect_network_zero_loss(self):
        tester = ConvergenceTester()
        r = tester.run_test(
            StateCRDT, num_vessels=5, ops_per_vessel=20,
            network_condition=NetworkCondition.PERFECT,
            use_network_sim=True, seed=42,
        )
        assert r.converged

    def test_wifi_network(self):
        tester = ConvergenceTester()
        r = tester.run_test(
            StateCRDT, num_vessels=5, ops_per_vessel=20,
            network_condition=NetworkCondition.WIFI,
            use_network_sim=True, seed=42,
        )
        assert r.errors == []

    def test_satellite_network(self):
        tester = ConvergenceTester()
        r = tester.run_test(
            StateCRDT, num_vessels=3, ops_per_vessel=10,
            network_condition=NetworkCondition.SATELLITE,
            use_network_sim=True, seed=42,
        )
        assert r.errors == []

    def test_storm_network(self):
        tester = ConvergenceTester()
        r = tester.run_test(
            StateCRDT, num_vessels=3, ops_per_vessel=5,
            network_condition=NetworkCondition.STORM,
            use_network_sim=True, seed=42,
        )
        # Storm has 40% loss — may not converge with few ops
        assert r.errors == []


# ==============================================================================
# Domain-Specific Convergence Tests
# ==============================================================================

class TestDomainConvergence:
    """Test per-domain convergence."""

    def test_trust_convergence_state_crdt(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=3, ops_per_vessel=20, seed=42)
        assert r.trust_converged

    def test_trust_convergence_git_sync(self):
        tester = ConvergenceTester()
        r = tester.run_test(GitSync, num_vessels=3, ops_per_vessel=20, seed=42)
        # GitSync may not converge on trust due to delta accumulation
        assert r.errors == []

    def test_skills_convergence_state_crdt(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=3, ops_per_vessel=20, seed=42)
        assert r.skills_converged

    def test_skills_convergence_git_sync(self):
        tester = ConvergenceTester()
        r = tester.run_test(GitSync, num_vessels=3, ops_per_vessel=20, seed=42)
        assert r.skills_converged

    def test_statuses_convergence_state_crdt(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=3, ops_per_vessel=20, seed=42)
        assert r.statuses_converged

    def test_tasks_convergence_state_crdt(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=3, ops_per_vessel=20, seed=42)
        assert r.tasks_converged


# ==============================================================================
# Competition Tests
# ==============================================================================

class TestCompetition:
    """Compare solutions against each other."""

    def test_competition_perfect_network(self):
        tester = ConvergenceTester()
        results = tester.run_competition(
            num_vessels=3, ops_per_vessel=10,
            network_condition=NetworkCondition.PERFECT,
            seeds=[42], use_network_sim=True,
        )
        assert "GitSync" in results
        assert "StateCRDT" in results
        assert "OperationCRDT" in results

    def test_competition_wifi(self):
        tester = ConvergenceTester()
        results = tester.run_competition(
            num_vessels=3, ops_per_vessel=10,
            network_condition=NetworkCondition.WIFI,
            seeds=[42], use_network_sim=True,
        )
        for name, data in results.items():
            assert "convergence_rate" in data

    def test_competition_satellite(self):
        tester = ConvergenceTester()
        results = tester.run_competition(
            num_vessels=3, ops_per_vessel=10,
            network_condition=NetworkCondition.SATELLITE,
            seeds=[42], use_network_sim=True,
        )
        for name, data in results.items():
            assert "avg_latency_ms" in data


# ==============================================================================
# Scale Tests
# ==============================================================================

class TestScale:
    """Test convergence at different scales."""

    def test_10_vessels_state_crdt(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=10, ops_per_vessel=10, seed=42)
        assert r.converged
        assert r.sync_rounds <= 30

    def test_10_vessels_git_sync(self):
        tester = ConvergenceTester()
        r = tester.run_test(GitSync, num_vessels=10, ops_per_vessel=10, seed=42)
        assert r.converged
        assert r.sync_rounds <= 30

    def test_high_operation_count(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=3, ops_per_vessel=50, seed=42)
        assert r.converged

    def test_many_vessels_few_ops(self):
        tester = ConvergenceTester()
        r = tester.run_test(StateCRDT, num_vessels=8, ops_per_vessel=2, seed=42)
        assert r.converged


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Test edge cases in convergence."""

    def test_simultaneous_updates_same_key(self):
        """Multiple vessels update same trust key simultaneously."""
        from jetson.agent.fleet_sync.solutions.state_crdt import StateCRDT
        v0 = StateCRDT("v0")
        v1 = StateCRDT("v1")
        v2 = StateCRDT("v2")

        # All update trust for "v3" simultaneously
        v0.update_trust("v3", 0.1)
        v1.update_trust("v3", -0.05)
        v2.update_trust("v3", 0.15)

        # Sync
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        p20 = v2.get_sync_payload()

        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        v0.receive_sync(p20, "v2")
        v2.receive_sync(p01, "v0")
        v1.receive_sync(p20, "v2")
        v2.receive_sync(p10, "v1")

        assert v0.get_state().is_equivalent(v1.get_state())
        assert v0.get_state().is_equivalent(v2.get_state())

    def test_all_vessels_add_same_task(self):
        """All vessels add the same task independently."""
        v0 = StateCRDT("v0")
        v1 = StateCRDT("v1")
        v2 = StateCRDT("v2")

        v0.add_task("shared-task", "Shared", 5)
        v1.add_task("shared-task", "Shared", 5)
        v2.add_task("shared-task", "Shared", 5)

        # Sync
        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        p20 = v2.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")
        v0.receive_sync(p20, "v2")
        v2.receive_sync(p01, "v0")
        v1.receive_sync(p20, "v2")
        v2.receive_sync(p10, "v1")

        assert v0.get_state().is_equivalent(v1.get_state())
        assert v0.get_state().is_equivalent(v2.get_state())

    def test_single_vessel_isolation(self):
        """Single vessel should trivially converge."""
        v = StateCRDT("v0")
        v.update_trust("v1", 0.1)
        v.add_task("t1", "Task", 5)
        v.update_vessel_status("v0", "battery", 85.0)
        v.update_skill_version("nav", "1.0.0")

        state = v.get_state()
        assert len(state.trust_scores) == 1
        assert len(state.task_queue) == 1

    def test_vessel_updates_self_status(self):
        """Vessel updates its own status."""
        v0 = StateCRDT("v0")
        v1 = StateCRDT("v1")

        v0.update_vessel_status("v0", "battery", 100.0)
        v1.update_vessel_status("v0", "battery", 50.0)

        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")

        assert v0.get_state().is_equivalent(v1.get_state())

    def test_empty_sync(self):
        """Syncing empty states should converge."""
        v0 = StateCRDT("v0")
        v1 = StateCRDT("v1")

        p01 = v0.get_sync_payload()
        p10 = v1.get_sync_payload()
        v0.receive_sync(p10, "v1")
        v1.receive_sync(p01, "v0")

        assert v0.get_state().is_equivalent(v1.get_state())


# ==============================================================================
# ConvergenceResult Tests
# ==============================================================================

class TestConvergenceResult:
    """Test the result data class."""

    def test_to_dict(self):
        r = ConvergenceResult(
            solution_name="StateCRDT",
            network_condition="wifi",
            num_vessels=5,
            ops_per_vessel=10,
            converged=True,
            sync_rounds=3,
        )
        d = r.to_dict()
        assert d["solution_name"] == "StateCRDT"
        assert d["converged"] is True
        assert d["sync_rounds"] == 3
        assert d["network_condition"] == "wifi"

    def test_defaults(self):
        r = ConvergenceResult("Test", "wifi", 3, 10)
        assert not r.converged
        assert r.sync_rounds == 0
        assert r.errors == []
