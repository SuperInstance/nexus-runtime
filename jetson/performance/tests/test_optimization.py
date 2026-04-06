"""Tests for jetson.performance.optimization — OptimizationSuggestion, CodeAnalyzer."""

import time
import pytest
from jetson.performance.optimization import (
    OptimizationSuggestion, CodeAnalyzer, ComplexityClass, ComplexityEstimate, math_log_safe,
)

def _linear_fn(data):
    return [x * 2 for x in data]

def _constant_fn(data):
    return 42

class TestOptimizationSuggestion:
    def test_construction(self):
        s = OptimizationSuggestion(type="hotspot", location="fn", description="Slow", estimated_improvement="50%", complexity="medium")
        assert s.type == "hotspot"
        assert s.location == "fn"

    def test_as_dict(self):
        s = OptimizationSuggestion(type="x", location="y", description="z", estimated_improvement="w", complexity="low")
        d = s.as_dict()
        assert set(d.keys()) == {"type","location","description","estimated_improvement","complexity"}

    def test_default_complexity(self):
        s = OptimizationSuggestion(type="a", location="b", description="c", estimated_improvement="d")
        assert s.complexity == "low"

class TestCodeAnalyzer:
    def test_analyze_complexity_linear(self):
        analyzer = CodeAnalyzer()
        inputs = [list(range(n)) for n in [10, 20, 40, 80, 160]]
        tc, sc = analyzer.analyze_complexity(_linear_fn, inputs)
        assert isinstance(tc, str)
        assert isinstance(sc, str)

    def test_analyze_complexity_constant(self):
        analyzer = CodeAnalyzer()
        inputs = [list(range(n)) for n in [10, 20, 40, 80]]
        tc, sc = analyzer.analyze_complexity(_constant_fn, inputs)
        assert isinstance(tc, str)

    def test_analyze_complexity_empty_inputs(self):
        analyzer = CodeAnalyzer()
        tc, sc = analyzer.analyze_complexity(_constant_fn, [])
        assert tc == "Unknown"

    def test_suggest_optimizations_hotspot(self):
        analyzer = CodeAnalyzer()
        profile = {"entries": [{"function_name": "slow_fn", "total_time": 5.0, "call_count": 1, "avg_time": 5.0, "max_time": 5.0}]}
        suggestions = analyzer.suggest_optimizations(profile)
        types = [s.type for s in suggestions]
        assert "hotspot" in types

    def test_suggest_optimizations_high_call_count(self):
        analyzer = CodeAnalyzer()
        profile = {"entries": [{"function_name": "tiny_fn", "total_time": 0.001, "call_count": 5000, "avg_time": 0.0000002, "max_time": 0.001}]}
        suggestions = analyzer.suggest_optimizations(profile)
        types = [s.type for s in suggestions]
        assert "micro_optimization" in types

    def test_suggest_optimizations_variance(self):
        analyzer = CodeAnalyzer()
        profile = {"entries": [{"function_name": "var_fn", "total_time": 0.5, "call_count": 10, "avg_time": 0.05, "max_time": 2.0}]}
        suggestions = analyzer.suggest_optimizations(profile)
        types = [s.type for s in suggestions]
        assert "variance" in types

    def test_suggest_optimizations_empty(self):
        analyzer = CodeAnalyzer()
        suggestions = analyzer.suggest_optimizations({"entries": []})
        assert len(suggestions) == 1
        assert suggestions[0].type == "info"

    def test_suggest_optimizations_no_entries_key(self):
        analyzer = CodeAnalyzer()
        suggestions = analyzer.suggest_optimizations({})
        assert len(suggestions) >= 1

    def test_compute_big_o_constant(self):
        result = CodeAnalyzer.compute_big_o_sequence([100, 200, 400, 800], [0.01, 0.01, 0.01, 0.01])
        assert result == "O(1)"

    def test_compute_big_o_linear(self):
        result = CodeAnalyzer.compute_big_o_sequence([100, 200, 400, 800], [0.01, 0.02, 0.04, 0.08])
        assert result == "O(n)"

    def test_compute_big_o_quadratic(self):
        result = CodeAnalyzer.compute_big_o_sequence([100, 200, 300], [0.01, 0.04, 0.09])
        assert "O(n²)" in result or "O(n)" in result

    def test_compute_big_o_insufficient(self):
        assert CodeAnalyzer.compute_big_o_sequence([100], [0.01]) == "Unknown"
        assert CodeAnalyzer.compute_big_o_sequence([], []) == "Unknown"

    def test_compute_big_o_zero_sizes(self):
        assert CodeAnalyzer.compute_big_o_sequence([0, 0], [0.01, 0.01]) == "Unknown"

    def test_compute_big_o_zero_times(self):
        assert CodeAnalyzer.compute_big_o_sequence([10, 20], [0.0, 0.0]) == "Unknown"

    def test_detect_memory_leaks_no_leak(self):
        history = [{"memory_used": 100}, {"memory_used": 101}, {"memory_used": 99}]
        result = CodeAnalyzer.detect_memory_leaks(history)
        assert result["leak_detected"] is False or result["severity"] == "low"

    def test_detect_memory_leaks_clear_leak(self):
        history = [{"memory_used": float(100 + i * 5)} for i in range(20)]
        result = CodeAnalyzer.detect_memory_leaks(history)
        assert result["leak_detected"] is True
        assert "slope" in result

    def test_detect_memory_leaks_insufficient_data(self):
        result = CodeAnalyzer.detect_memory_leaks([{"memory_used": 50}])
        assert result["leak_detected"] is False

    def test_detect_memory_leaks_empty(self):
        result = CodeAnalyzer.detect_memory_leaks([])
        assert result["leak_detected"] is False

    def test_detect_memory_leaks_memory_mb_key(self):
        history = [{"memory_mb": 100}, {"memory_mb": 100}, {"memory_mb": 100}]
        result = CodeAnalyzer.detect_memory_leaks(history)
        assert "details" in result

    def test_compute_cache_miss_rate(self):
        stats = {"hits": 80, "misses": 20}
        assert CodeAnalyzer.compute_cache_miss_rate(stats) == pytest.approx(0.2)

    def test_compute_cache_miss_rate_zero(self):
        assert CodeAnalyzer.compute_cache_miss_rate({"hits": 0, "misses": 0}) == 0.0

    def test_compute_cache_miss_rate_all_hits(self):
        assert CodeAnalyzer.compute_cache_miss_rate({"hits": 100, "misses": 0}) == 0.0

    def test_compute_cache_miss_rate_all_misses(self):
        assert CodeAnalyzer.compute_cache_miss_rate({"hits": 0, "misses": 50}) == pytest.approx(1.0)

    def test_compute_cache_miss_rate_empty(self):
        assert CodeAnalyzer.compute_cache_miss_rate({}) == 0.0

    def test_analyze_algorithmic_bottleneck(self):
        profile = {"entries": [{"function_name": "main_loop", "total_time": 5.0}, {"function_name": "setup", "total_time": 0.1}]}
        result = CodeAnalyzer.analyze_algorithmic_bottleneck(profile)
        assert result["bottleneck"] == "main_loop"
        assert result["percentage"] > 90

    def test_analyze_algorithmic_bottleneck_empty(self):
        result = CodeAnalyzer.analyze_algorithmic_bottleneck({"entries": []})
        assert result["bottleneck"] is None

    def test_analyze_algorithmic_bottleneck_empty_dict(self):
        result = CodeAnalyzer.analyze_algorithmic_bottleneck({})
        assert result["bottleneck"] is None

class TestMathLogSafe:
    def test_positive(self):
        assert math_log_safe(10.0) == pytest.approx(2.302585093, abs=0.01)

    def test_zero(self):
        assert math_log_safe(0.0) == 0.0

    def test_negative(self):
        assert math_log_safe(-5.0) == 0.0
