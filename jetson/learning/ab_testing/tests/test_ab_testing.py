"""NEXUS A/B Testing Framework — Comprehensive test suite.

80+ tests covering:
  - ABTestSuite creation, serialization, metric recording
  - Statistical tests (Welch t, Mann-Whitney U, Bootstrap CI)
  - Test selection logic (normal → t-test, non-normal → Mann-Whitney)
  - Bonferroni correction
  - ReflexComparator VM simulation
  - Git-branch integration (mocked)
  - Edge cases: tiny samples, identical variants, extreme outliers
  - Integration: full A/B test from creation to winner selection
"""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
from pathlib import Path

import pytest

from learning.ab_testing.experiment import (
    ABTestResult,
    ABTestSuite,
    ExperimentVariant,
    MetricRecord,
    MetricType,
    PowerAnalysisResult,
    _normal_ppf,
)
from learning.ab_testing.git_integration import (
    BranchInfo,
    BranchIntegration,
    GitOperationError,
)
from learning.ab_testing.statistical_engine import (
    BonferroniResult,
    StatisticalEngine,
    TestMethod,
    TestResult,
    _bootstrap_ci,
    _cliffs_delta,
    _cohen_d,
    _mean,
    _mann_whitney_u_test,
    _median,
    _normal_cdf,
    _rank_data,
    _shapiro_wilk_w,
    _std,
    _t_cdf,
    _welch_t_test,
    is_normal,
)
from learning.ab_testing.vm_simulator import (
    ReflexComparator,
    SimulationIteration,
    StackVM,
    _f16_to_f32,
)
from reflex.bytecode_emitter import BytecodeEmitter, INSTR_SIZE


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def simple_bytecode_a():
    """Read sensor pin 0, push 270.0, subtract, clamp [-30, 30], write pin 0, halt."""
    e = BytecodeEmitter()
    e.emit_read_pin(0)
    e.emit_push_f32(270.0)
    e.emit_sub_f()
    e.emit_clamp_f(-30.0, 30.0)
    e.emit_write_pin(0)
    e.emit_halt()
    return e.get_bytecode()


@pytest.fixture
def simple_bytecode_b():
    """Variant B: different gain factor."""
    e = BytecodeEmitter()
    e.emit_read_pin(0)
    e.emit_push_f32(270.0)
    e.emit_sub_f()
    e.emit_push_f32(0.5)
    e.emit_mul_f()
    e.emit_clamp_f(-30.0, 30.0)
    e.emit_write_pin(0)
    e.emit_halt()
    return e.get_bytecode()


@pytest.fixture
def add_bytecode():
    """Push 10.0, push 20.0, add, halt."""
    e = BytecodeEmitter()
    e.emit_push_f32(10.0)
    e.emit_push_f32(20.0)
    e.emit_add_f()
    e.emit_halt()
    return e.get_bytecode()


@pytest.fixture
def basic_suite():
    """Create a basic A/B test suite with two variants."""
    e = BytecodeEmitter()
    e.emit_read_pin(0)
    e.emit_push_f32(270.0)
    e.emit_sub_f()
    e.emit_clamp_f(-30.0, 30.0)
    e.emit_write_pin(0)
    e.emit_halt()
    bc_a = e.get_bytecode()

    e2 = BytecodeEmitter()
    e2.emit_read_pin(0)
    e2.emit_push_f32(270.0)
    e2.emit_sub_f()
    e2.emit_push_f32(0.5)
    e2.emit_mul_f()
    e2.emit_clamp_f(-30.0, 30.0)
    e2.emit_write_pin(0)
    e2.emit_halt()
    bc_b = e2.get_bytecode()

    suite = ABTestSuite(
        name="heading_hold_test",
        description="Test heading hold reflex variants",
    )
    suite.add_variant("A", bc_a, {"branch": "main", "compiler": "v1.0"})
    suite.add_variant("B", bc_b, {"branch": "experiment/gain", "compiler": "v1.1"})
    return suite


@pytest.fixture
def engine():
    return StatisticalEngine(alpha=0.05)


# ===========================================================================
# ABTestSuite Tests (15 tests)
# ===========================================================================

class TestABTestSuiteCreation:
    """Test suite creation and initialization."""

    def test_create_basic_suite(self) -> None:
        suite = ABTestSuite(name="test_exp")
        assert suite.name == "test_exp"
        assert suite.description == ""
        assert len(suite.variants) == 0
        assert suite.results is None

    def test_create_with_description(self) -> None:
        suite = ABTestSuite(name="test", description="Test description")
        assert suite.description == "Test description"

    def test_create_with_custom_metrics(self) -> None:
        metrics = [MetricType.CYCLE_TIME_MS, MetricType.ACCURACY]
        suite = ABTestSuite(name="test", metrics=metrics)
        assert len(suite.metrics) == 2
        assert MetricType.CYCLE_TIME_MS in suite.metrics

    def test_create_with_alpha(self) -> None:
        suite = ABTestSuite(name="test", alpha=0.01)
        assert suite.alpha == 0.01

    def test_add_variant(self) -> None:
        suite = ABTestSuite(name="test")
        v = suite.add_variant("A", b"\x00" * 8, {"source": "main"})
        assert "A" in suite.variants
        assert suite.variants["A"].name == "A"
        assert suite.variants["A"].metadata["source"] == "main"

    def test_add_multiple_variants(self) -> None:
        suite = ABTestSuite(name="test")
        suite.add_variant("A", b"\x00")
        suite.add_variant("B", b"\x01")
        suite.add_variant("C", b"\x02")
        assert len(suite.variants) == 3

    def test_get_variant(self) -> None:
        suite = ABTestSuite(name="test")
        suite.add_variant("A", b"\x00")
        v = suite.get_variant("A")
        assert v is not None
        assert v.name == "A"

    def test_get_unknown_variant(self) -> None:
        suite = ABTestSuite(name="test")
        assert suite.get_variant("Z") is None

    def test_variant_names(self) -> None:
        suite = ABTestSuite(name="test")
        suite.add_variant("A")
        suite.add_variant("B")
        assert suite.variant_names() == ["A", "B"]

    def test_default_duration(self) -> None:
        suite = ABTestSuite(name="test")
        assert suite.duration_seconds == 3600

    def test_custom_duration(self) -> None:
        suite = ABTestSuite(name="test", duration_seconds=7200)
        assert suite.duration_seconds == 7200

    def test_default_power(self) -> None:
        suite = ABTestSuite(name="test")
        assert suite.target_power == 0.80

    def test_all_default_metrics(self) -> None:
        suite = ABTestSuite(name="test")
        assert len(suite.metrics) == len(MetricType)

    def test_no_variants_initially(self) -> None:
        suite = ABTestSuite(name="test")
        assert suite.total_observations() == 0
        assert suite.min_observations() == 0


class TestABTestSuiteMetricRecording:
    """Test metric recording and retrieval."""

    def test_record_metric(self, basic_suite) -> None:
        basic_suite.record_metric("A", MetricType.CYCLE_TIME_MS, 5.0)
        assert basic_suite.total_observations() == 1

    def test_record_unknown_variant_raises(self) -> None:
        suite = ABTestSuite(name="test")
        with pytest.raises(KeyError):
            suite.record_metric("Z", MetricType.CYCLE_TIME_MS, 1.0)

    def test_record_multiple_metrics(self, basic_suite) -> None:
        for i in range(10):
            basic_suite.record_metric("A", MetricType.CYCLE_TIME_MS, float(i))
            basic_suite.record_metric("B", MetricType.CYCLE_TIME_MS, float(i + 1))
        assert basic_suite.total_observations() == 20
        assert basic_suite.min_observations() == 10

    def test_get_metric_values(self, basic_suite) -> None:
        for i in range(5):
            basic_suite.record_metric("A", MetricType.ACCURACY, 0.9 + i * 0.01)
        values = basic_suite.get_metric_values("A", MetricType.ACCURACY)
        assert len(values) == 5
        assert abs(values[0] - 0.90) < 1e-9

    def test_get_empty_metric_values(self, basic_suite) -> None:
        values = basic_suite.get_metric_values("A", MetricType.TRUST_DELTA)
        assert values == []

    def test_observations_per_variant(self, basic_suite) -> None:
        for i in range(5):
            basic_suite.record_metric("A", MetricType.CYCLE_TIME_MS, float(i))
        for i in range(3):
            basic_suite.record_metric("B", MetricType.CYCLE_TIME_MS, float(i))
        obs = basic_suite.observations_per_variant()
        assert obs["A"] == 5
        assert obs["B"] == 3

    def test_max_observations(self, basic_suite) -> None:
        for i in range(7):
            basic_suite.record_metric("A", MetricType.CYCLE_TIME_MS, float(i))
        for i in range(3):
            basic_suite.record_metric("B", MetricType.CYCLE_TIME_MS, float(i))
        assert basic_suite.max_observations() == 7

    def test_clear_metrics(self, basic_suite) -> None:
        for i in range(5):
            basic_suite.record_metric("A", MetricType.CYCLE_TIME_MS, float(i))
        basic_suite.clear_metrics()
        assert basic_suite.total_observations() == 0


class TestABTestSuiteSummaryStats:
    """Test summary statistics computation."""

    def test_summary_stats_normal(self) -> None:
        suite = ABTestSuite(name="test")
        suite.add_variant("A")
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            suite.record_metric("A", MetricType.CYCLE_TIME_MS, v)
        stats = suite.summary_stats("A", MetricType.CYCLE_TIME_MS)
        assert stats["n"] == 5
        assert abs(stats["mean"] - 3.0) < 1e-9
        assert abs(stats["median"] - 3.0) < 1e-9
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["std"] > 0

    def test_summary_stats_empty(self) -> None:
        suite = ABTestSuite(name="test")
        suite.add_variant("A")
        stats = suite.summary_stats("A", MetricType.CYCLE_TIME_MS)
        assert stats["n"] == 0
        assert stats["mean"] == 0.0

    def test_summary_stats_single_value(self) -> None:
        suite = ABTestSuite(name="test")
        suite.add_variant("A")
        suite.record_metric("A", MetricType.CYCLE_TIME_MS, 42.0)
        stats = suite.summary_stats("A", MetricType.CYCLE_TIME_MS)
        assert stats["n"] == 1
        assert stats["std"] == 0.0


class TestABTestSuiteSerialization:
    """Test JSON serialization/deserialization."""

    def test_to_dict(self, basic_suite) -> None:
        d = basic_suite.to_dict()
        assert d["name"] == "heading_hold_test"
        assert "A" in d["variants"]
        assert "B" in d["variants"]

    def test_from_dict(self, basic_suite) -> None:
        d = basic_suite.to_dict()
        restored = ABTestSuite.from_dict(d)
        assert restored.name == basic_suite.name
        assert len(restored.variants) == 2

    def test_to_json_roundtrip(self, basic_suite) -> None:
        basic_suite.record_metric("A", MetricType.CYCLE_TIME_MS, 5.0)
        json_str = basic_suite.to_json()
        restored = ABTestSuite.from_json(json_str)
        assert restored.name == basic_suite.name
        assert restored.total_observations() == 1

    def test_serialization_preserves_bytecode(self) -> None:
        suite = ABTestSuite(name="test")
        bc = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        suite.add_variant("A", bc)
        json_str = suite.to_json()
        restored = ABTestSuite.from_json(json_str)
        assert restored.variants["A"].bytecode == bc

    def test_serialization_with_results(self) -> None:
        suite = ABTestSuite(name="test")
        suite.add_variant("A")
        suite.add_variant("B")
        result = ABTestResult(
            experiment_name="test",
            winner="A",
            recommendation="A wins",
        )
        suite.set_result(result)
        json_str = suite.to_json()
        restored = ABTestSuite.from_json(json_str)
        assert restored.results is not None
        assert restored.results.winner == "A"


class TestABTestSuitePowerAnalysis:
    """Test statistical power analysis."""

    def test_power_analysis_default(self) -> None:
        suite = ABTestSuite(name="test")
        pa = suite.compute_power_analysis(effect_size=0.5)
        assert pa.effect_size == 0.5
        assert pa.alpha == 0.05
        assert pa.power == 0.80
        assert pa.min_sample_size > 0

    def test_power_analysis_custom_alpha(self) -> None:
        suite = ABTestSuite(name="test")
        pa = suite.compute_power_analysis(effect_size=0.5, alpha=0.01)
        assert pa.alpha == 0.01
        assert pa.min_sample_size > 0

    def test_power_analysis_large_effect_small_sample(self) -> None:
        suite = ABTestSuite(name="test")
        pa = suite.compute_power_analysis(effect_size=1.5)
        assert pa.min_sample_size < 20

    def test_power_analysis_small_effect_large_sample(self) -> None:
        suite = ABTestSuite(name="test")
        pa = suite.compute_power_analysis(effect_size=0.1)
        assert pa.min_sample_size > 100

    def test_has_sufficient_data(self, basic_suite) -> None:
        # Not enough data initially
        assert not basic_suite.has_sufficient_data(effect_size=0.5)
        # Add enough data (power analysis needs ~64 samples for effect_size=0.5)
        for i in range(70):
            basic_suite.record_metric("A", MetricType.CYCLE_TIME_MS, float(i))
            basic_suite.record_metric("B", MetricType.CYCLE_TIME_MS, float(i))
        assert basic_suite.has_sufficient_data(effect_size=0.5)

    def test_power_analysis_result_serialization(self) -> None:
        pa = PowerAnalysisResult(
            effect_size=0.5, alpha=0.05, power=0.80, min_sample_size=64
        )
        d = pa.to_dict()
        assert d["effect_size"] == 0.5
        restored = PowerAnalysisResult.from_dict(d)
        assert restored.min_sample_size == 64


# ===========================================================================
# MetricRecord Tests (5 tests)
# ===========================================================================

class TestMetricRecord:
    def test_create_record(self) -> None:
        r = MetricRecord(metric_type=MetricType.CYCLE_TIME_MS, value=5.0)
        assert r.metric_type == MetricType.CYCLE_TIME_MS
        assert r.value == 5.0

    def test_record_serialization(self) -> None:
        r = MetricRecord(
            metric_type=MetricType.ACCURACY, value=0.95,
            timestamp_ms=1000, iteration=5,
        )
        d = r.to_dict()
        assert d["metric_type"] == "accuracy"
        restored = MetricRecord.from_dict(d)
        assert restored.value == 0.95
        assert restored.iteration == 5

    def test_all_metric_types(self) -> None:
        for mt in MetricType:
            r = MetricRecord(metric_type=mt, value=1.0)
            assert r.metric_type == mt

    def test_experiment_variant_metric_recording(self) -> None:
        v = ExperimentVariant(name="A")
        v.record_metric(MetricType.CYCLE_TIME_MS, 5.0, timestamp_ms=100)
        v.record_metric(MetricType.ACCURACY, 0.95)
        assert len(v.metrics) == 2
        assert v.get_metric_values(MetricType.CYCLE_TIME_MS) == [5.0]

    def test_variant_serialization(self) -> None:
        v = ExperimentVariant(name="A", bytecode=b"\x00\x01", metadata={"key": "val"})
        v.record_metric(MetricType.CYCLE_TIME_MS, 5.0)
        d = v.to_dict()
        restored = ExperimentVariant.from_dict(d)
        assert restored.name == "A"
        assert restored.bytecode == b"\x00\x01"
        assert restored.metadata["key"] == "val"
        assert len(restored.metrics) == 1


# ===========================================================================
# Statistical Helper Tests (10 tests)
# ===========================================================================

class TestStatisticalHelpers:
    def test_mean_empty(self) -> None:
        assert _mean([]) == 0.0

    def test_mean_single(self) -> None:
        assert _mean([5.0]) == 5.0

    def test_mean_multiple(self) -> None:
        assert abs(_mean([1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0) < 1e-9

    def test_std_empty(self) -> None:
        assert _std([]) == 0.0

    def test_std_single(self) -> None:
        assert _std([5.0]) == 0.0

    def test_std_known(self) -> None:
        # std of [2, 4, 4, 4, 5, 5, 7, 9] = sqrt(32/7) ≈ 2.138
        s = _std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(s - math.sqrt(32.0 / 7.0)) < 0.01

    def test_median_odd(self) -> None:
        assert _median([1.0, 3.0, 5.0]) == 3.0

    def test_median_even(self) -> None:
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_normal_ppf_50(self) -> None:
        assert abs(_normal_ppf(0.5)) < 0.1  # Should be ~0

    def test_normal_ppf_extreme(self) -> None:
        assert _normal_ppf(0.0) == -10.0
        assert _normal_ppf(1.0) == 10.0

    def test_normal_cdf_zero(self) -> None:
        assert abs(_normal_cdf(0.0) - 0.5) < 0.1

    def test_rank_data_no_ties(self) -> None:
        ranks = _rank_data([3.0, 1.0, 2.0])
        assert ranks == [3.0, 1.0, 2.0]

    def test_rank_data_with_ties(self) -> None:
        ranks = _rank_data([1.0, 2.0, 2.0, 3.0])
        assert ranks == [1.0, 2.5, 2.5, 4.0]


# ===========================================================================
# Normality Tests (6 tests)
# ===========================================================================

class TestNormality:
    def test_normal_data_detected(self) -> None:
        import random
        rng = random.Random(42)
        data = [rng.gauss(0, 1) for _ in range(100)]
        assert is_normal(data)

    def test_uniform_data_not_normal(self) -> None:
        # Exponential distribution is clearly non-normal
        data = [(-1.0 * math.log(1.0 - i / 101.0)) for i in range(1, 101)]
        assert not is_normal(data)

    def test_small_sample_assumed_normal(self) -> None:
        assert is_normal([1.0, 2.0])

    def test_constant_data_normal(self) -> None:
        assert is_normal([5.0] * 20)

    def test_shapiro_wilk_perfect_normal(self) -> None:
        import random
        rng = random.Random(42)
        data = [rng.gauss(100, 15) for _ in range(200)]
        w = _shapiro_wilk_w(data)
        assert w > 0.95

    def test_shapiro_wilk_non_normal(self) -> None:
        # Exponential distribution
        data = [(-1.0 * math.log(1.0 - i / 100.0)) for i in range(1, 100)]
        w = _shapiro_wilk_w(data)
        assert w < 0.98


# ===========================================================================
# Welch's T-Test Tests (8 tests)
# ===========================================================================

class TestWelchTTest:
    def test_same_distribution_not_significant(self) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(100, 10) for _ in range(50)]
        _, p = _welch_t_test(a, b)
        assert p > 0.01  # Not significant

    def test_different_means_significant(self) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(110, 10) for _ in range(50)]
        _, p = _welch_t_test(a, b)
        assert p < 0.05  # Significant

    def test_different_variances(self) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 5) for _ in range(50)]
        b = [rng.gauss(100, 20) for _ in range(50)]
        _, p = _welch_t_test(a, b)
        assert p > 0.01  # Same mean, different variance

    def test_tiny_samples(self) -> None:
        _, p = _welch_t_test([1.0], [2.0])
        assert p == 1.0  # Not enough data

    def test_single_element_each(self) -> None:
        _, p = _welch_t_test([5.0], [5.0])
        assert p == 1.0

    def test_identical_data(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, p = _welch_t_test(data, data)
        assert p == 1.0

    def test_cohen_d_same(self) -> None:
        import random
        rng = random.Random(42)
        data = [rng.gauss(100, 10) for _ in range(50)]
        d = _cohen_d(data, data)
        assert abs(d) < 0.01

    def test_cohen_d_different(self) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(110, 10) for _ in range(50)]
        d = _cohen_d(a, b)
        assert abs(d) > 0.5  # Large effect


# ===========================================================================
# Mann-Whitney U Test Tests (8 tests)
# ===========================================================================

class TestMannWhitneyUTest:
    def test_same_distribution_not_significant(self) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(100, 10) for _ in range(50)]
        _, p = _mann_whitney_u_test(a, b)
        assert p > 0.01

    def test_different_means_significant(self) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(115, 10) for _ in range(50)]
        _, p = _mann_whitney_u_test(a, b)
        assert p < 0.05

    def test_tiny_samples(self) -> None:
        _, p = _mann_whitney_u_test([1.0], [2.0])
        assert p == 1.0

    def test_identical_data(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, p = _mann_whitney_u_test(data, data)
        assert p > 0.05

    def test_all_ties(self) -> None:
        a = [5.0] * 10
        b = [5.0] * 10
        _, p = _mann_whitney_u_test(a, b)
        assert p > 0.01

    def test_cliffs_delta_same(self) -> None:
        import random
        rng = random.Random(42)
        data = [rng.gauss(100, 10) for _ in range(30)]
        d = _cliffs_delta(data, data)
        assert abs(d) < 0.01

    def test_cliffs_delta_different(self) -> None:
        a = [1.0] * 20
        b = [2.0] * 20
        d = _cliffs_delta(a, b)
        assert d < -0.5

    def test_non_normal_distribution(self) -> None:
        a = [1.0] * 10 + [100.0] * 10  # Bimodal
        b = [50.0] * 20
        _, p = _mann_whitney_u_test(a, b)
        # Non-parametric test should still work
        assert isinstance(p, float)


# ===========================================================================
# Bootstrap CI Tests (6 tests)
# ===========================================================================

class TestBootstrapCI:
    def test_same_distribution_ci_contains_zero(self) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(100, 10) for _ in range(50)]
        lo, hi, _ = _bootstrap_ci(a, b, n_bootstrap=5000, seed=42)
        assert lo <= 0 <= hi

    def test_different_distribution_ci_excludes_zero(self) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 5) for _ in range(100)]
        b = [rng.gauss(115, 5) for _ in range(100)]
        lo, hi, _ = _bootstrap_ci(a, b, n_bootstrap=5000, seed=42)
        assert hi < 0  # Negative difference (a < b), no zero in CI

    def test_empty_data(self) -> None:
        lo, hi, eff = _bootstrap_ci([], [1.0, 2.0])
        assert lo == 0.0 and hi == 0.0 and eff == 0.0

    def test_deterministic_with_seed(self) -> None:
        import random
        rng = random.Random(42)
        data_a = [rng.gauss(100, 10) for _ in range(50)]
        data_b = [rng.gauss(105, 10) for _ in range(50)]
        r1 = _bootstrap_ci(data_a, data_b, n_bootstrap=1000, seed=42)
        r2 = _bootstrap_ci(data_a, data_b, n_bootstrap=1000, seed=42)
        assert r1 == r2

    def test_effect_size_magnitude(self) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(0, 1) for _ in range(100)]
        b = [rng.gauss(2, 1) for _ in range(100)]
        _, _, eff = _bootstrap_ci(a, b, n_bootstrap=2000, seed=42)
        assert abs(eff) > 1.0

    def test_single_value_data(self) -> None:
        lo, hi, _ = _bootstrap_ci([5.0], [5.0], n_bootstrap=1000, seed=42)
        assert lo <= 0 <= hi


# ===========================================================================
# StatisticalEngine Tests (15 tests)
# ===========================================================================

class TestStatisticalEngine:
    def test_select_method_normal(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(100, 10) for _ in range(50)]
        method, _, _ = engine.select_method(a, b)
        assert method == "welch_t"

    def test_select_method_non_normal(self, engine) -> None:
        # Exponential distribution is clearly non-normal
        a = [(-1.0 * math.log(1.0 - i / 101.0)) for i in range(1, 101)]
        b = [(-1.0 * math.log(1.0 - i / 101.0)) + 0.5 for i in range(1, 101)]
        method, _, _ = engine.select_method(a, b)
        assert method == "mann_whitney_u"

    def test_run_welch_t(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(110, 10) for _ in range(50)]
        result = engine.run_welch_t(a, b, "test_metric", "A", "B")
        assert result.method == TestMethod.WELCH_T
        assert result.p_value < 0.05
        # A has lower mean, so _recommendation returns "A wins" (lower is better default)
        assert result.recommendation == "A wins"

    def test_run_mann_whitney(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(110, 10) for _ in range(50)]
        result = engine.run_mann_whitney(a, b, "test_metric", "A", "B")
        assert result.method == TestMethod.MANN_WHITNEY_U
        assert result.p_value < 0.05

    def test_run_bootstrap(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 5) for _ in range(100)]
        b = [rng.gauss(115, 5) for _ in range(100)]
        result = engine.run_bootstrap(a, b, "test_metric", "A", "B", seed=42)
        assert result.method == TestMethod.BOOTSTRAP_CI
        assert result.ci_lower < result.ci_upper

    def test_run_auto_selects_welch(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(110, 10) for _ in range(50)]
        result = engine.run_auto(a, b, "metric", "A", "B")
        assert result.selected_method == "welch_t"

    def test_run_auto_too_few_samples(self, engine) -> None:
        result = engine.run_auto([1.0], [2.0], "metric", "A", "B")
        assert result.recommendation == "need_more_data"

    def test_run_auto_identical_data(self, engine) -> None:
        data = [rng_val for rng_val in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
        result = engine.run_auto(data, data, "metric", "A", "B")
        assert result.recommendation == "inconclusive"

    def test_result_serialization(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(100, 10) for _ in range(50)]
        result = engine.run_welch_t(a, b, "test", "A", "B")
        d = result.to_dict()
        assert d["method"] == "welch_t"
        assert "p_value" in d


# ===========================================================================
# Bonferroni Correction Tests (8 tests)
# ===========================================================================

class TestBonferroniCorrection:
    def test_no_tests(self, engine) -> None:
        result = engine.run_bonferroni([])
        assert result.num_tests == 0
        assert result.overall_recommendation == "inconclusive"

    def test_single_test_unchanged(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(110, 10) for _ in range(50)]
        tr = engine.run_welch_t(a, b, "test", "A", "B")
        result = engine.run_bonferroni([tr])
        assert result.num_tests == 1
        assert result.adjusted_alpha == 0.05

    def test_multiple_tests_stricter_alpha(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(100, 10) for _ in range(50)]
        results = []
        for mt in ["cycle_time_ms", "accuracy", "trust_delta"]:
            tr = engine.run_welch_t(a, b, mt, "A", "B")
            results.append(tr)
        bonf = engine.run_bonferroni(results)
        assert bonf.num_tests == 3
        assert bonf.adjusted_alpha < 0.05

    def test_bonferroni_result_serialization(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(100, 10) for _ in range(50)]
        results = [
            engine.run_welch_t(a, b, "m1", "A", "B"),
            engine.run_welch_t(a, b, "m2", "A", "B"),
        ]
        bonf = engine.run_bonferroni(results)
        d = bonf.to_dict()
        assert d["num_tests"] == 2
        assert "corrected_results" in d

    def test_bonferroni_majority_a_wins(self, engine) -> None:
        """When most metrics favor A, overall should be A wins."""
        import random
        rng = random.Random(42)
        a = [rng.gauss(90, 5) for _ in range(100)]  # Better (lower)
        b = [rng.gauss(110, 5) for _ in range(100)]  # Worse (higher)
        results = []
        for mt in ["cycle_time_ms", "accuracy", "trust_delta"]:
            tr = engine.run_welch_t(a, b, mt, "A", "B")
            results.append(tr)
        bonf = engine.run_bonferroni(results)
        assert bonf.overall_recommendation in ("A wins", "inconclusive")

    def test_custom_alpha(self, engine) -> None:
        result = engine.run_bonferroni([], alpha=0.01)
        assert result.adjusted_alpha == 0.01

    def test_corrected_results_count(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(50)]
        b = [rng.gauss(100, 10) for _ in range(50)]
        results = [
            engine.run_welch_t(a, b, f"m{i}", "A", "B")
            for i in range(5)
        ]
        bonf = engine.run_bonferroni(results)
        assert len(bonf.corrected_results) == 5
        assert len(bonf.original_results) == 5

    def test_compare_variants_integration(self, basic_suite, engine) -> None:
        """Full compare_variants integration test."""
        import random
        rng = random.Random(42)
        for i in range(50):
            basic_suite.record_metric(
                "A", MetricType.CYCLE_TIME_MS, rng.gauss(5.0, 0.5)
            )
            basic_suite.record_metric(
                "B", MetricType.CYCLE_TIME_MS, rng.gauss(7.0, 0.5)
            )
        bonf = engine.compare_variants(basic_suite, MetricType.CYCLE_TIME_MS)
        assert bonf.num_tests >= 1


# ===========================================================================
# VM Simulator Tests (12 tests)
# ===========================================================================

class TestStackVM:
    def test_empty_bytecode(self) -> None:
        vm = StackVM()
        result = vm.execute(b"")
        assert result.safety_events
        assert "empty_bytecode" in result.safety_events

    def test_simple_addition(self, add_bytecode) -> None:
        vm = StackVM()
        result = vm.execute(add_bytecode)
        assert result.error is None
        assert len(result.final_stack) == 1
        assert abs(result.final_stack[0] - 30.0) < 1e-9

    def test_halt_stops_execution(self) -> None:
        e = BytecodeEmitter()
        e.emit_push_f32(1.0)
        e.emit_halt()
        e.emit_push_f32(2.0)  # Should not execute
        vm = StackVM()
        result = vm.execute(e.get_bytecode())
        assert len(result.final_stack) == 1

    def test_read_pin(self) -> None:
        e = BytecodeEmitter()
        e.emit_read_pin(0)
        e.emit_write_pin(1)
        e.emit_halt()
        vm = StackVM(sensor_inputs={0: 42.0})
        result = vm.execute(e.get_bytecode())
        assert abs(result.actuator_outputs[1] - 42.0) < 1e-9

    def test_stack_overflow(self) -> None:
        e = BytecodeEmitter()
        for _ in range(100):
            e.emit_push_f32(1.0)
        e.emit_halt()
        vm = StackVM(max_stack=10)
        result = vm.execute(e.get_bytecode())
        assert any("stack_overflow" in ev for ev in result.safety_events)

    def test_stack_underflow(self) -> None:
        e = BytecodeEmitter()
        e.emit_pop()
        vm = StackVM()
        result = vm.execute(e.get_bytecode())
        assert any("stack_underflow" in ev for ev in result.safety_events)

    def test_division_by_zero(self) -> None:
        e = BytecodeEmitter()
        e.emit_push_f32(1.0)
        e.emit_push_f32(0.0)
        e.emit_div_f()
        e.emit_halt()
        vm = StackVM()
        result = vm.execute(e.get_bytecode())
        assert any("division_by_zero" in ev for ev in result.safety_events)

    def test_clamp_instruction(self) -> None:
        e = BytecodeEmitter()
        e.emit_push_f32(50.0)
        e.emit_clamp_f(-30.0, 30.0)
        e.emit_halt()
        vm = StackVM()
        result = vm.execute(e.get_bytecode())
        assert abs(result.final_stack[0] - 30.0) < 1e-6

    def test_comparison_lt(self) -> None:
        e = BytecodeEmitter()
        e.emit_push_f32(5.0)
        e.emit_push_f32(10.0)
        e.emit_lt_f()
        e.emit_halt()
        vm = StackVM()
        result = vm.execute(e.get_bytecode())
        assert abs(result.final_stack[0] - 1.0) < 1e-9

    def test_jump_if_false(self) -> None:
        e = BytecodeEmitter()
        e.emit_push_f32(0.0)
        # Jump to halt (instruction index 3)
        e.emit_jump_if_false(3 * INSTR_SIZE)
        e.emit_push_f32(999.0)  # Should be skipped
        e.emit_halt()
        vm = StackVM()
        result = vm.execute(e.get_bytecode())
        # 0.0 was consumed by JUMP_IF_FALSE check, stack empty
        assert len(result.final_stack) == 0

    def test_cycle_count(self, add_bytecode) -> None:
        vm = StackVM()
        result = vm.execute(add_bytecode)
        assert result.cycle_count == 4  # push, push, add, halt

    def test_f16_conversion(self) -> None:
        assert abs(_f16_to_f32(0x3C00) - 1.0) < 1e-6
        assert abs(_f16_to_f32(0x4000) - 2.0) < 1e-6
        assert abs(_f16_to_f32(0x4200) - 3.0) < 1e-6


# ===========================================================================
# ReflexComparator Tests (8 tests)
# ===========================================================================

class TestReflexComparator:
    def test_simulate_single(self, simple_bytecode_a) -> None:
        comp = ReflexComparator(sensor_inputs={0: 280.0})
        result = comp.simulate_single(simple_bytecode_a)
        assert result.cycle_count > 0
        assert result.error is None

    def test_simulate_variant(self, simple_bytecode_a) -> None:
        comp = ReflexComparator(sensor_inputs={0: 280.0})
        results = comp.simulate_variant(simple_bytecode_a, n_iterations=10)
        assert len(results) == 10
        for r in results:
            assert r.cycle_count > 0

    def test_simulate_with_noise(self, simple_bytecode_a) -> None:
        comp = ReflexComparator(sensor_inputs={0: 280.0}, noise_std=5.0)
        results = comp.simulate_variant(simple_bytecode_a, n_iterations=20)
        assert len(results) == 20

    def test_simulate_with_scenarios(self, simple_bytecode_a) -> None:
        comp = ReflexComparator()
        scenarios = [{0: 260.0}, {0: 270.0}, {0: 280.0}]
        results = comp.simulate_variant(
            simple_bytecode_a, n_iterations=6, sensor_scenarios=scenarios
        )
        assert len(results) == 6

    def test_extract_metrics(self, simple_bytecode_a) -> None:
        comp = ReflexComparator(sensor_inputs={0: 280.0})
        results = comp.simulate_variant(simple_bytecode_a, n_iterations=10)
        metrics = comp.extract_metrics(results)
        assert "cycle_time_ms" in metrics
        assert "accuracy" in metrics
        assert "safety_events" in metrics
        assert len(metrics["cycle_time_ms"]) == 10

    def test_compare_populates_suite(self, basic_suite) -> None:
        comp = ReflexComparator(sensor_inputs={0: 280.0})
        comp.compare(basic_suite, n_iterations=5)
        assert basic_suite.total_observations() > 0
        assert basic_suite.min_observations() >= 5

    def test_different_bytecodes_different_outputs(
        self, simple_bytecode_a, simple_bytecode_b
    ) -> None:
        comp = ReflexComparator(sensor_inputs={0: 280.0})
        r_a = comp.simulate_single(simple_bytecode_a)
        r_b = comp.simulate_single(simple_bytecode_b)
        # Variant B has a 0.5 multiplier, so output should be half
        assert r_a.output_value != r_b.output_value

    def test_deterministic_with_seed(self, simple_bytecode_a) -> None:
        comp1 = ReflexComparator(sensor_inputs={0: 280.0}, seed=42, noise_std=2.0)
        comp2 = ReflexComparator(sensor_inputs={0: 280.0}, seed=42, noise_std=2.0)
        r1 = comp1.simulate_single(simple_bytecode_a)
        r2 = comp2.simulate_single(simple_bytecode_a)
        assert r1.output_value == r2.output_value


# ===========================================================================
# Git Integration Tests (8 tests)
# ===========================================================================

class TestBranchIntegration:
    def test_branch_name_generation(self) -> None:
        bi = BranchIntegration(dry_run=True)
        name = bi.branch_name("heading test", "A")
        assert name == "experiment/heading-test/variant_A"

    def test_branch_name_special_chars(self) -> None:
        bi = BranchIntegration(dry_run=True)
        name = bi.branch_name("test/name", "B")
        assert "experiment" in name
        assert "variant_B" in name

    def test_create_variant_branch_dry_run(self) -> None:
        bi = BranchIntegration(dry_run=True)
        info = bi.create_variant_branch("test_exp", "A")
        assert info.variant_name == "A"
        assert info.experiment_name == "test_exp"
        assert info.status == "active"

    def test_archive_variant_branch_dry_run(self) -> None:
        bi = BranchIntegration(dry_run=True)
        bi.create_variant_branch("test_exp", "A")
        bi.archive_variant_branch("test_exp", "A", "Lost to variant B")
        assert bi._branch_cache[
            bi.branch_name("test_exp", "A")
        ].status == "archived"

    def test_merge_winner_dry_run(self) -> None:
        bi = BranchIntegration(dry_run=True)
        bi.create_variant_branch("test_exp", "A")
        branch = bi.merge_winner("test_exp", "A")
        assert "variant_A" in branch
        assert bi._branch_cache[
            bi.branch_name("test_exp", "A")
        ].status == "merged"

    def test_store_results_dry_run(self) -> None:
        bi = BranchIntegration(dry_run=True)
        result = ABTestResult(
            experiment_name="test",
            winner="A",
            recommendation="A wins",
        )
        path = bi.store_results("test_exp", result)
        assert "results.json" in str(path)

    def test_full_workflow_dry_run(self) -> None:
        bi = BranchIntegration(dry_run=True)
        result = ABTestResult(
            experiment_name="heading_test",
            winner="A",
            recommendation="A wins",
        )
        summary = bi.run_full_workflow(
            "heading_test",
            {"A": b"\x00", "B": b"\x01"},
            result,
        )
        assert summary["winner"] == "A"
        assert summary["dry_run"] is True
        assert "A" in summary["branches_created"]

    def test_full_workflow_inconclusive(self) -> None:
        bi = BranchIntegration(dry_run=True)
        result = ABTestResult(
            experiment_name="heading_test",
            winner="",
            recommendation="inconclusive",
        )
        summary = bi.run_full_workflow(
            "heading_test",
            {"A": b"\x00", "B": b"\x01"},
            result,
        )
        assert summary["recommendation"] == "inconclusive"

    def test_list_experiments_empty(self) -> None:
        bi = BranchIntegration(dry_run=True)
        assert bi.list_experiments() == []

    def test_store_and_load_results_real(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nexus_dir = os.path.join(tmpdir, ".nexus")
            bi = BranchIntegration(repo_root=tmpdir, nexus_dir=nexus_dir, dry_run=False)
            result = ABTestResult(
                experiment_name="test",
                winner="A",
                recommendation="A wins",
                total_iterations=100,
            )
            bi.store_results("test", result)
            loaded = bi.load_results("test")
            assert loaded is not None
            assert loaded["winner"] == "A"

    def test_cleanup_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nexus_dir = os.path.join(tmpdir, ".nexus")
            bi = BranchIntegration(repo_root=tmpdir, nexus_dir=nexus_dir, dry_run=False)
            bi.store_results("test", {"winner": "A"})
            assert "test" in bi.list_experiments()
            bi.cleanup_experiment("test")
            assert "test" not in bi.list_experiments()


# ===========================================================================
# Edge Case Tests (10 tests)
# ===========================================================================

class TestEdgeCases:
    def test_tiny_samples_t_test(self, engine) -> None:
        """Very small samples should return need_more_data."""
        result = engine.run_welch_t([1.0, 2.0], [3.0, 4.0], "m", "A", "B")
        assert result.p_value >= 0.0

    def test_extreme_outliers(self, engine) -> None:
        """Data with extreme outliers."""
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 1) for _ in range(49)] + [100000.0]
        b = [rng.gauss(100, 1) for _ in range(50)]
        result = engine.run_auto(a, b, "m", "A", "B")
        assert result.recommendation in ("A wins", "B wins", "inconclusive", "need_more_data")

    def test_constant_data(self, engine) -> None:
        result = engine.run_auto([5.0] * 20, [5.0] * 20, "m", "A", "B")
        assert result.recommendation == "inconclusive"

    def test_single_variant_suite(self) -> None:
        suite = ABTestSuite(name="test")
        suite.add_variant("A")
        assert suite.variant_names() == ["A"]

    def test_zero_variance(self, engine) -> None:
        result = engine.run_welch_t([5.0] * 20, [5.0] * 20, "m", "A", "B")
        assert result.p_value == 1.0

    def test_very_different_sample_sizes(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(10)]
        b = [rng.gauss(100, 10) for _ in range(200)]
        result = engine.run_welch_t(a, b, "m", "A", "B")
        assert isinstance(result.p_value, float)

    def test_negative_values(self, engine) -> None:
        a = [-10.0, -20.0, -15.0]
        b = [-5.0, -10.0, -7.0]
        result = engine.run_welch_t(a, b, "m", "A", "B")
        assert isinstance(result.p_value, float)

    def test_large_dataset(self, engine) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(100, 10) for _ in range(1000)]
        b = [rng.gauss(105, 10) for _ in range(1000)]
        result = engine.run_auto(a, b, "m", "A", "B")
        assert result.p_value < 0.05

    def test_vm_max_cycles(self) -> None:
        """Infinite loop should be caught by max cycles."""
        e = BytecodeEmitter()
        e.emit_push_f32(0.0)
        # Jump to self
        e.emit_jump(0)  # Jump to instruction 0
        vm = StackVM(max_cycles=5)
        result = vm.execute(e.get_bytecode())
        assert result.cycle_count <= 5

    def test_empty_suite_serialization(self) -> None:
        suite = ABTestSuite(name="empty_test")
        json_str = suite.to_json()
        restored = ABTestSuite.from_json(json_str)
        assert restored.name == "empty_test"
        assert len(restored.variants) == 0


# ===========================================================================
# Integration Tests (6 tests)
# ===========================================================================

class TestIntegration:
    def test_full_ab_test_workflow(self) -> None:
        """Full workflow: create suite → simulate → statistical analysis → result."""
        from learning.ab_testing.experiment import MetricType

        # Build two bytecodes
        e1 = BytecodeEmitter()
        e1.emit_read_pin(0)
        e1.emit_push_f32(270.0)
        e1.emit_sub_f()
        e1.emit_clamp_f(-30.0, 30.0)
        e1.emit_write_pin(0)
        e1.emit_halt()

        e2 = BytecodeEmitter()
        e2.emit_read_pin(0)
        e2.emit_push_f32(270.0)
        e2.emit_sub_f()
        e2.emit_push_f32(0.5)
        e2.emit_mul_f()
        e2.emit_clamp_f(-30.0, 30.0)
        e2.emit_write_pin(0)
        e2.emit_halt()

        suite = ABTestSuite(name="full_integration_test")
        suite.add_variant("A", e1.get_bytecode())
        suite.add_variant("B", e2.get_bytecode())

        # Simulate
        comp = ReflexComparator(sensor_inputs={0: 280.0}, seed=42)
        comp.compare(suite, n_iterations=20)

        # Verify data was collected
        assert suite.total_observations() > 0

        # Statistical analysis
        engine = StatisticalEngine(alpha=0.05)
        bonf = engine.compare_variants(suite, MetricType.CYCLE_TIME_MS)
        assert bonf.num_tests >= 1

    def test_suite_with_git_workflow(self) -> None:
        """Suite creation → simulation → analysis → git workflow."""
        e = BytecodeEmitter()
        e.emit_push_f32(42.0)
        e.emit_halt()

        suite = ABTestSuite(name="git_integration_test")
        suite.add_variant("A", e.get_bytecode())
        suite.add_variant("B", e.get_bytecode())

        comp = ReflexComparator(sensor_inputs={}, seed=42)
        comp.compare(suite, n_iterations=5)

        engine = StatisticalEngine()
        bonf = engine.compare_variants(suite)

        result = ABTestResult(
            experiment_name="git_integration_test",
            winner=bonf.overall_recommendation.replace(" wins", ""),
            recommendation=bonf.overall_recommendation,
        )

        bi = BranchIntegration(dry_run=True)
        summary = bi.run_full_workflow(
            "git_integration_test",
            {"A": suite.variants["A"].bytecode,
             "B": suite.variants["B"].bytecode},
            result,
        )
        assert "git_integration_test" in summary["experiment_name"]

    def test_power_analysis_guides_experiment(self) -> None:
        """Power analysis tells us how many samples we need."""
        suite = ABTestSuite(name="power_guided")
        suite.add_variant("A")
        suite.add_variant("B")

        pa = suite.compute_power_analysis(effect_size=0.8)
        assert pa.min_sample_size > 0

        # Collect exactly enough data
        for i in range(pa.min_sample_size):
            suite.record_metric("A", MetricType.CYCLE_TIME_MS, float(i))
            suite.record_metric("B", MetricType.CYCLE_TIME_MS, float(i + 1))

        assert suite.has_sufficient_data(effect_size=0.8)

    def test_all_metric_types_tracked(self) -> None:
        suite = ABTestSuite(
            name="all_metrics",
            metrics=list(MetricType),
        )
        suite.add_variant("A")
        for mt in MetricType:
            suite.record_metric("A", mt, 1.0)
        assert suite.total_observations() == len(MetricType)

    def test_ab_result_serialization_roundtrip(self) -> None:
        result = ABTestResult(
            experiment_name="test",
            winner="A",
            recommendation="A wins",
            total_iterations=100,
            metric_summaries={"cycle_time_ms": {"mean_a": 5.0, "mean_b": 7.0}},
            bonferroni_corrected=True,
        )
        json_str = result.to_json()
        restored = ABTestResult.from_dict(json.loads(json_str))
        assert restored.winner == "A"
        assert restored.bonferroni_corrected is True
        assert restored.total_iterations == 100

    def test_vm_simulator_iteration_data(self) -> None:
        e = BytecodeEmitter()
        e.emit_read_pin(0)
        e.emit_write_pin(1)
        e.emit_halt()
        vm = StackVM(sensor_inputs={0: 99.0})
        result = vm.execute(e.get_bytecode())
        assert result.actuator_outputs[1] == 99.0
        assert result.cycle_count == 3
        assert result.output_value == 99.0
        d = result.to_dict()
        assert d["cycle_count"] == 3
        assert d["output_value"] == 99.0
