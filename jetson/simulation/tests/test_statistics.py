"""Tests for statistics.py — MonteCarloResult, SimulationRun, SimulationStatistics."""

import math
import pytest

from jetson.simulation.statistics import MonteCarloResult, SimulationRun, SimulationStatistics


class TestMonteCarloResult:
    def test_default_creation(self):
        r = MonteCarloResult(metric_name="test")
        assert r.metric_name == "test"
        assert r.mean == 0.0
        assert r.std_dev == 0.0
        assert r.confidence_interval == (0.0, 0.0)
        assert r.samples == 0

    def test_custom_creation(self):
        r = MonteCarloResult(metric_name="m1", mean=5.0, std_dev=1.0, confidence_interval=(3.0, 7.0), samples=100)
        assert r.mean == 5.0
        assert r.samples == 100


class TestSimulationRun:
    def test_default_creation(self):
        run = SimulationRun(run_id=1)
        assert run.run_id == 1
        assert run.config == {}
        assert run.results == {}
        assert run.metrics == {}

    def test_custom_creation(self):
        run = SimulationRun(run_id=5, config={"key": "val"}, results={"metric": 3.14})
        assert run.config["key"] == "val"
        assert run.results["metric"] == 3.14


class TestSimulationStatisticsCreation:
    def test_default_creation(self):
        stats = SimulationStatistics()
        assert stats.run_count == 0

    def test_seeded_creation(self):
        stats = SimulationStatistics(seed=42)
        assert stats.run_count == 0


class TestConfidenceInterval:
    def test_ci_single_sample(self):
        stats = SimulationStatistics()
        ci = stats.compute_confidence_interval([5.0])
        assert ci == (5.0, 5.0)

    def test_ci_empty_samples(self):
        stats = SimulationStatistics()
        ci = stats.compute_confidence_interval([])
        assert ci == (0.0, 0.0)

    def test_ci_two_identical_samples(self):
        stats = SimulationStatistics()
        ci = stats.compute_confidence_interval([10.0, 10.0])
        assert ci[0] <= 10.0 <= ci[1]

    def test_ci_95_percent(self):
        stats = SimulationStatistics()
        samples = list(range(100))
        ci = stats.compute_confidence_interval(samples, confidence=0.95)
        mean = sum(samples) / len(samples)
        assert ci[0] < mean < ci[1]

    def test_ci_99_percent_wider(self):
        stats = SimulationStatistics()
        samples = [float(i) for i in range(100)]
        ci_95 = stats.compute_confidence_interval(samples, confidence=0.95)
        ci_99 = stats.compute_confidence_interval(samples, confidence=0.99)
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        assert width_99 > width_95

    def test_ci_large_sample_narrower(self):
        stats = SimulationStatistics()
        small = [float(i) for i in range(10)]
        large = [float(i) for i in range(1000)]
        ci_small = stats.compute_confidence_interval(small)
        ci_large = stats.compute_confidence_interval(large)
        # Larger sample should give narrower CI relative to range
        mean_s = sum(small) / len(small)
        mean_l = sum(large) / len(large)
        rel_small = (ci_small[1] - ci_small[0]) / (mean_s + 1)
        rel_large = (ci_large[1] - ci_large[0]) / (mean_l + 1)
        assert rel_large < rel_small


class TestPercentiles:
    def test_percentiles_basic(self):
        stats = SimulationStatistics()
        samples = [float(i) for i in range(101)]  # 0 to 100
        pcts = stats.compute_percentiles(samples, [25, 50, 75])
        assert pcts[50] == pytest.approx(50.0, abs=2.0)
        assert pcts[25] < pcts[50] < pcts[75]

    def test_percentiles_empty(self):
        stats = SimulationStatistics()
        pcts = stats.compute_percentiles([], [50])
        assert pcts[50] == 0.0

    def test_percentiles_single(self):
        stats = SimulationStatistics()
        pcts = stats.compute_percentiles([42.0], [50])
        assert pcts[50] == pytest.approx(42.0)

    def test_percentiles_100(self):
        stats = SimulationStatistics()
        pcts = stats.compute_percentiles([1, 2, 3, 4, 5], [0, 100])
        assert pcts[0] == pytest.approx(1.0)
        assert pcts[100] == pytest.approx(5.0)

    def test_percentiles_custom(self):
        stats = SimulationStatistics()
        samples = [float(i) for i in range(100)]
        pcts = stats.compute_percentiles(samples, [10, 30, 70, 90])
        assert len(pcts) == 4
        assert pcts[10] < pcts[30] < pcts[70] < pcts[90]


class TestCompareDistributions:
    def test_identical_distributions(self):
        stats = SimulationStatistics()
        samples = [float(i) for i in range(100)]
        stat, p_val = stats.compare_distributions(samples, samples)
        assert p_val > 0.05  # Should not be significant

    def test_different_distributions(self):
        stats = SimulationStatistics()
        a = [float(i) for i in range(50)]
        b = [float(i + 50) for i in range(50)]
        stat, p_val = stats.compare_distributions(a, b)
        assert p_val < 0.05  # Should be significant

    def test_empty_samples(self):
        stats = SimulationStatistics()
        stat, p_val = stats.compare_distributions([], [1.0, 2.0])
        assert stat == 0.0
        assert p_val == 1.0

    def test_single_samples(self):
        stats = SimulationStatistics()
        stat, p_val = stats.compare_distributions([1.0], [2.0])
        # Should return something without error
        assert isinstance(stat, float)
        assert isinstance(p_val, float)

    def test_returns_tuple(self):
        stats = SimulationStatistics()
        result = stats.compare_distributions([1, 2, 3], [4, 5, 6])
        assert len(result) == 2


class TestReliability:
    def test_perfect_reliability(self):
        stats = SimulationStatistics()
        runs = [{"success": True} for _ in range(10)]
        rel = stats.compute_reliability(runs, lambda r: r["success"])
        assert rel == pytest.approx(1.0)

    def test_zero_reliability(self):
        stats = SimulationStatistics()
        runs = [{"success": False} for _ in range(10)]
        rel = stats.compute_reliability(runs, lambda r: r["success"])
        assert rel == pytest.approx(0.0)

    def test_partial_reliability(self):
        stats = SimulationStatistics()
        runs = [{"success": i < 7} for i in range(10)]
        rel = stats.compute_reliability(runs, lambda r: r["success"])
        assert rel == pytest.approx(0.7)

    def test_empty_runs(self):
        stats = SimulationStatistics()
        rel = stats.compute_reliability([], lambda r: True)
        assert rel == 0.0

    def test_complex_criteria(self):
        stats = SimulationStatistics()
        runs = [{"score": i} for i in range(10)]
        rel = stats.compute_reliability(runs, lambda r: r["score"] >= 5)
        assert rel == pytest.approx(0.5)


class TestMonteCarlo:
    def test_basic_monte_carlo(self):
        stats = SimulationStatistics(seed=42)
        def scenario():
            return {"value": 10.0}
        results = stats.run_monte_carlo(scenario, num_runs=50)
        assert len(results) == 1
        assert results[0].metric_name == "value"
        assert results[0].mean == pytest.approx(10.0)
        assert results[0].samples == 50

    def test_monte_carlo_multiple_metrics(self):
        stats = SimulationStatistics(seed=42)
        def scenario():
            return {"a": 1.0, "b": 2.0}
        results = stats.run_monte_carlo(scenario, num_runs=20)
        assert len(results) == 2
        names = {r.metric_name for r in results}
        assert "a" in names
        assert "b" in names

    def test_monte_carlo_random(self):
        stats = SimulationStatistics(seed=42)
        import random as _random
        rng = _random.Random(42)
        def scenario():
            return {"val": rng.gauss(0, 1)}
        results = stats.run_monte_carlo(scenario, num_runs=100)
        assert results[0].std_dev > 0.0

    def test_monte_carlo_creates_runs(self):
        stats = SimulationStatistics(seed=42)
        def scenario():
            return {"x": 1.0}
        stats.run_monte_carlo(scenario, num_runs=10)
        assert stats.run_count == 10

    def test_monte_carlo_with_config(self):
        stats = SimulationStatistics(seed=42)
        def scenario():
            return {"y": 5.0}
        results = stats.run_monte_carlo(scenario, num_runs=5, config={"key": "val"})
        assert stats.runs[0].config == {"key": "val"}


class TestGenerateReport:
    def test_empty_report(self):
        stats = SimulationStatistics()
        report = stats.generate_report([])
        assert report["total_metrics"] == 0
        assert report["metrics"] == []

    def test_report_with_results(self):
        stats = SimulationStatistics()
        results = [
            MonteCarloResult(metric_name="m1", mean=5.0, std_dev=1.0, confidence_interval=(3.0, 7.0), samples=100),
            MonteCarloResult(metric_name="m2", mean=10.0, std_dev=2.0, confidence_interval=(6.0, 14.0), samples=100),
        ]
        report = stats.generate_report(results)
        assert report["total_metrics"] == 2
        assert report["total_samples"] == 200
        assert len(report["metrics"]) == 2

    def test_report_has_summary(self):
        stats = SimulationStatistics()
        results = [MonteCarloResult(metric_name="m", mean=5.0, std_dev=1.0, confidence_interval=(4.0, 6.0), samples=50)]
        report = stats.generate_report(results)
        assert "summary" in report
        assert isinstance(report["summary"], str)

    def test_report_metric_details(self):
        stats = SimulationStatistics()
        results = [MonteCarloResult(metric_name="test", mean=5.0, std_dev=1.0, confidence_interval=(4.0, 6.0), samples=50)]
        report = stats.generate_report(results)
        metric = report["metrics"][0]
        assert metric["name"] == "test"
        assert metric["margin_of_error"] == pytest.approx(1.0)
        assert "confidence_interval" in metric


class TestRunManagement:
    def test_add_run(self):
        stats = SimulationStatistics()
        run = SimulationRun(run_id=1)
        stats.add_run(run)
        assert stats.run_count == 1

    def test_clear_runs(self):
        stats = SimulationStatistics()
        stats.add_run(SimulationRun(run_id=1))
        stats.clear_runs()
        assert stats.run_count == 0

    def test_runs_property(self):
        stats = SimulationStatistics()
        stats.add_run(SimulationRun(run_id=1))
        stats.add_run(SimulationRun(run_id=2))
        runs = stats.runs
        assert len(runs) == 2
        assert runs[0].run_id == 1

    def test_runs_list_copy(self):
        stats = SimulationStatistics()
        stats.add_run(SimulationRun(run_id=1))
        runs1 = stats.runs
        runs2 = stats.runs
        assert runs1 is not runs2  # Should be copies


class TestMeanStdDev:
    def test_mean_helper(self):
        assert SimulationStatistics._mean([1, 2, 3, 4, 5]) == 3.0

    def test_mean_empty(self):
        assert SimulationStatistics._mean([]) == 0.0

    def test_std_dev_helper(self):
        # Known: std_dev of [2,4,4,4,5,5,7,9] = 2.0
        result = SimulationStatistics._std_dev([2, 4, 4, 4, 5, 5, 7, 9])
        assert result == pytest.approx(2.138, abs=0.01)

    def test_std_dev_single(self):
        assert SimulationStatistics._std_dev([5.0]) == 0.0

    def test_std_dev_empty(self):
        assert SimulationStatistics._std_dev([]) == 0.0

    def test_std_dev_zero_variance(self):
        assert SimulationStatistics._std_dev([3.0, 3.0, 3.0]) == pytest.approx(0.0)


class TestNormalSurvival:
    def test_zero_z(self):
        stats = SimulationStatistics()
        p = stats._normal_survival(0.0)
        assert p == pytest.approx(1.0)

    def test_large_z(self):
        stats = SimulationStatistics()
        p = stats._normal_survival(5.0)
        assert p < 0.01

    def test_negative_z(self):
        stats = SimulationStatistics()
        p = stats._normal_survival(-1.0)
        assert p == 1.0
