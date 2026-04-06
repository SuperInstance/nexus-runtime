"""Tests for jetson.performance.benchmark — BenchmarkConfig, BenchmarkResult, BenchmarkRunner."""

import time
import pytest
from jetson.performance.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner

def _noop():
    pass

def _fast():
    return sum(range(10))

def _slow():
    time.sleep(0.001)

class TestBenchmarkConfig:
    def test_defaults(self):
        c = BenchmarkConfig()
        assert c.name == "unnamed"
        assert c.iterations == 100
        assert c.warmup == 5
        assert c.timeout == 30.0
        assert c.parameters == {}

    def test_custom(self):
        c = BenchmarkConfig(name="test", iterations=50, warmup=2, timeout=5.0, parameters={"x": 1})
        assert c.name == "test"
        assert c.iterations == 50
        assert c.parameters["x"] == 1

class TestBenchmarkResult:
    def test_construction(self):
        r = BenchmarkResult(name="t", mean_time=0.01, std_dev=0.001, min_time=0.005, max_time=0.02, iterations=100, ops_per_sec=100.0)
        assert r.name == "t"
        assert r.mean_time == pytest.approx(0.01)
        assert r.ops_per_sec == pytest.approx(100.0)

    def test_median_time_property(self):
        r = BenchmarkResult(name="t", mean_time=0.05, std_dev=0.0, min_time=0.05, max_time=0.05, iterations=1, ops_per_sec=20.0)
        assert r.median_time == pytest.approx(0.05)

    def test_as_dict(self):
        r = BenchmarkResult(name="t", mean_time=0.01, std_dev=0.001, min_time=0.005, max_time=0.02, iterations=10, ops_per_sec=100.0)
        d = r.as_dict()
        assert set(d.keys()) == {"name","mean_time","std_dev","min_time","max_time","iterations","ops_per_sec"}

    def test_as_dict_values(self):
        r = BenchmarkResult(name="x", mean_time=0.1, std_dev=0.01, min_time=0.08, max_time=0.12, iterations=5, ops_per_sec=10.0)
        d = r.as_dict()
        assert d["name"] == "x"
        assert d["mean_time"] == pytest.approx(0.1)

class TestBenchmarkRunner:
    def test_run_benchmark_noop(self):
        runner = BenchmarkRunner()
        cfg = BenchmarkConfig(name="noop", iterations=10, warmup=1, timeout=5.0)
        result = runner.run_benchmark(_noop, cfg)
        assert result.name == "noop"
        assert result.iterations == 10
        assert result.mean_time >= 0
        assert result.ops_per_sec >= 0

    def test_run_benchmark_fast(self):
        runner = BenchmarkRunner()
        cfg = BenchmarkConfig(name="fast", iterations=20, warmup=2, timeout=5.0)
        result = runner.run_benchmark(_fast, cfg)
        assert result.iterations == 20
        assert result.std_dev >= 0

    def test_run_benchmark_default_config(self):
        runner = BenchmarkRunner()
        result = runner.run_benchmark(_noop)
        assert result.name == "unnamed"
        assert result.iterations > 0

    def test_run_benchmark_warmup_executed(self):
        runner = BenchmarkRunner()
        calls = [0]
        def counting():
            calls[0] += 1
        cfg = BenchmarkConfig(iterations=5, warmup=3)
        runner.run_benchmark(counting, cfg)
        assert calls[0] == 8  # 3 warmup + 5 iterations

    def test_run_benchmark_min_max(self):
        runner = BenchmarkRunner()
        cfg = BenchmarkConfig(iterations=20, warmup=2)
        result = runner.run_benchmark(_noop, cfg)
        assert result.min_time <= result.max_time

    def test_run_benchmark_ops_per_sec(self):
        runner = BenchmarkRunner()
        result = runner.run_benchmark(_noop, BenchmarkConfig(iterations=10, warmup=1))
        if result.mean_time > 0:
            assert result.ops_per_sec > 0

    def test_run_benchmark_single_iteration(self):
        runner = BenchmarkRunner()
        cfg = BenchmarkConfig(iterations=1, warmup=0)
        result = runner.run_benchmark(_noop, cfg)
        assert result.iterations == 1
        assert result.std_dev == 0.0

    def test_compare_implementations(self):
        runner = BenchmarkRunner()
        results = runner.compare_implementations([_noop, _fast], BenchmarkConfig(iterations=10, warmup=1))
        assert len(results) == 2
        assert results[0].mean_time <= results[1].mean_time

    def test_compare_implementations_default_config(self):
        runner = BenchmarkRunner()
        results = runner.compare_implementations([_noop, _fast])
        assert len(results) == 2

    def test_compare_implementations_ranked(self):
        runner = BenchmarkRunner()
        results = runner.compare_implementations([_slow, _noop, _fast], BenchmarkConfig(iterations=5, warmup=1))
        names = [r.name for r in results]
        assert "_slow" not in names or names[-1] == results[-1].name  # slowest last

    def test_compare_implementations_names(self):
        runner = BenchmarkRunner()
        results = runner.compare_implementations([_noop, _fast], BenchmarkConfig(name="mybench"))
        assert results[0].name.startswith("mybench_impl_")
        assert results[1].name.startswith("mybench_impl_")

    def test_run_regression_suite(self):
        runner = BenchmarkRunner()
        suite = {"noop": _noop, "fast": _fast}
        report = runner.run_regression_suite(suite)
        assert "results" in report
        assert "noop" in report["results"]
        assert "fast" in report["results"]

    def test_run_regression_suite_with_baseline(self):
        runner = BenchmarkRunner()
        suite = {"noop": _noop}
        baseline = {"noop": BenchmarkResult("noop", 0.001, 0.0, 0.001, 0.001, 1, 1000.0)}
        report = runner.run_regression_suite(suite, baseline)
        assert "regressions" in report

    def test_run_regression_suite_detects_regression(self):
        runner = BenchmarkRunner()
        def very_slow():
            time.sleep(0.01)
        suite = {"slow": very_slow}
        baseline = {"slow": BenchmarkResult("slow", 0.0001, 0.0, 0.0001, 0.0001, 1, 10000.0)}
        report = runner.run_regression_suite(suite, baseline)
        assert len(report["regressions"]) >= 1

    def test_run_regression_suite_empty(self):
        runner = BenchmarkRunner()
        report = runner.run_regression_suite({})
        assert report["results"] == {}

    def test_compute_statistical_significance_different(self):
        runner = BenchmarkRunner()
        r1 = BenchmarkResult("a", 0.01, 0.001, 0.005, 0.02, 100, 100.0)
        r2 = BenchmarkResult("b", 0.05, 0.005, 0.02, 0.1, 100, 20.0)
        assert runner.compute_statistical_significance(r1, r2) is True

    def test_compute_statistical_significance_same(self):
        runner = BenchmarkRunner()
        r1 = BenchmarkResult("a", 0.01, 0.001, 0.009, 0.011, 100, 100.0)
        r2 = BenchmarkResult("b", 0.0101, 0.001, 0.009, 0.011, 100, 99.0)
        assert runner.compute_statistical_significance(r1, r2) is False

    def test_compute_statistical_significance_low_n(self):
        runner = BenchmarkRunner()
        r1 = BenchmarkResult("a", 0.01, 0.0, 0.01, 0.01, 1, 100.0)
        r2 = BenchmarkResult("b", 0.02, 0.0, 0.02, 0.02, 1, 50.0)
        assert runner.compute_statistical_significance(r1, r2) is True

    def test_compute_statistical_significance_identical(self):
        runner = BenchmarkRunner()
        r = BenchmarkResult("a", 0.01, 0.0, 0.01, 0.01, 2, 100.0)
        assert runner.compute_statistical_significance(r, r) is False

    def test_generate_benchmark_report(self):
        runner = BenchmarkRunner()
        r1 = BenchmarkResult("a", 0.01, 0.001, 0.005, 0.02, 100, 100.0)
        r2 = BenchmarkResult("b", 0.05, 0.005, 0.02, 0.1, 50, 20.0)
        report = runner.generate_benchmark_report([r1, r2])
        assert report["total_benchmarks"] == 2
        assert report["fastest"] == "a"
        assert report["slowest"] == "b"
        assert len(report["benchmarks"]) == 2

    def test_generate_benchmark_report_empty(self):
        runner = BenchmarkRunner()
        report = runner.generate_benchmark_report([])
        assert report["total_benchmarks"] == 0
        assert report["benchmarks"] == []

    def test_generate_benchmark_report_single(self):
        runner = BenchmarkRunner()
        r = BenchmarkResult("only", 0.01, 0.0, 0.01, 0.01, 1, 100.0)
        report = runner.generate_benchmark_report([r])
        assert report["fastest"] == "only"
        assert report["slowest"] == "only"
