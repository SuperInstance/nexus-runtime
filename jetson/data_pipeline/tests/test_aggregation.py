"""Tests for aggregation.py — AggregationType, TimeWindow, TimeSeriesAggregator."""

import math

import pytest

from jetson.data_pipeline.aggregation import (
    AggregationType,
    TimeWindow,
    TimeSeriesAggregator,
)


# ── AggregationType enum ───────────────────────────────────────

class TestAggregationType:

    def test_all_values(self):
        expected = {"MEAN", "MIN", "MAX", "SUM", "COUNT", "STD_DEV", "MEDIAN", "FIRST", "LAST"}
        actual = {e.name for e in AggregationType}
        assert actual == expected

    def test_value_property(self):
        assert AggregationType.MEAN.value == "mean"
        assert AggregationType.COUNT.value == "count"


# ── TimeWindow dataclass ───────────────────────────────────────

class TestTimeWindow:

    def test_create(self):
        tw = TimeWindow(start=0.0, end=10.0,
                        aggregation_type=AggregationType.MEAN, value=5.0)
        assert tw.start == 0.0
        assert tw.end == 10.0
        assert tw.aggregation_type == AggregationType.MEAN
        assert tw.value == 5.0


# ── aggregate ──────────────────────────────────────────────────

class TestAggregate:

    def setup_method(self):
        self.agg = TimeSeriesAggregator()
        self.series = [(float(i), float(i)) for i in range(20)]

    def test_empty_series(self):
        assert self.agg.aggregate([], 5.0, AggregationType.MEAN) == []

    def test_mean(self):
        windows = self.agg.aggregate(self.series, 5.0, AggregationType.MEAN)
        assert len(windows) == 4
        assert windows[0].value == 2.0  # mean of [0,1,2,3,4]

    def test_min(self):
        windows = self.agg.aggregate(self.series, 10.0, AggregationType.MIN)
        assert windows[0].value == 0.0
        assert windows[1].value == 10.0

    def test_max(self):
        windows = self.agg.aggregate(self.series, 10.0, AggregationType.MAX)
        assert windows[0].value == 9.0
        assert windows[1].value == 19.0

    def test_sum(self):
        windows = self.agg.aggregate(self.series, 5.0, AggregationType.SUM)
        assert windows[0].value == 10.0  # 0+1+2+3+4

    def test_count(self):
        windows = self.agg.aggregate(self.series, 5.0, AggregationType.COUNT)
        assert all(w.value == 5 for w in windows)

    def test_std_dev(self):
        windows = self.agg.aggregate(self.series, 5.0, AggregationType.STD_DEV)
        # std of [0,1,2,3,4] = sqrt(10) ≈ 1.581
        assert abs(windows[0].value - math.sqrt(10 / 4)) < 1e-6

    def test_median_odd(self):
        windows = self.agg.aggregate(self.series, 5.0, AggregationType.MEDIAN)
        assert windows[0].value == 2.0  # median of [0,1,2,3,4]

    def test_median_even(self):
        s = [(float(i), float(i)) for i in range(4)]
        windows = self.agg.aggregate(s, 4.0, AggregationType.MEDIAN)
        assert windows[0].value == 1.5  # median of [0,1,2,3]

    def test_first(self):
        windows = self.agg.aggregate(self.series, 10.0, AggregationType.FIRST)
        assert windows[0].value == 0.0
        assert windows[1].value == 10.0

    def test_last(self):
        windows = self.agg.aggregate(self.series, 10.0, AggregationType.LAST)
        assert windows[0].value == 9.0
        assert windows[1].value == 19.0

    def test_window_boundaries(self):
        windows = self.agg.aggregate(self.series, 5.0, AggregationType.MEAN)
        assert windows[0].start == 0.0
        assert windows[0].end == 5.0
        assert windows[1].start == 5.0

    def test_unsorted_input(self):
        unsorted = [(5.0, 5.0), (1.0, 1.0), (3.0, 3.0)]
        windows = self.agg.aggregate(unsorted, 3.0, AggregationType.MEAN)
        assert len(windows) >= 1


# ── aggregate_multiple ─────────────────────────────────────────

class TestAggregateMultiple:

    def test_two_types(self):
        agg = TimeSeriesAggregator()
        series = [(float(i), float(i)) for i in range(10)]
        result = agg.aggregate_multiple(
            series, 5.0,
            [AggregationType.MEAN, AggregationType.MAX],
        )
        assert AggregationType.MEAN in result
        assert AggregationType.MAX in result
        assert len(result[AggregationType.MEAN]) == 2

    def test_single_type(self):
        agg = TimeSeriesAggregator()
        series = [(0.0, 10.0), (1.0, 20.0)]
        result = agg.aggregate_multiple(series, 5.0, [AggregationType.SUM])
        assert result[AggregationType.SUM][0].value == 30.0

    def test_empty_series(self):
        agg = TimeSeriesAggregator()
        result = agg.aggregate_multiple([], 5.0, [AggregationType.MEAN])
        assert result[AggregationType.MEAN] == []


# ── downsampling ───────────────────────────────────────────────

class TestDownsampling:

    def test_basic_downsample(self):
        agg = TimeSeriesAggregator()
        series = [(float(i), float(i)) for i in range(100)]
        down = agg.downsampling(series, target_points=10, method=AggregationType.MEAN)
        assert len(down) <= 11  # approximate

    def test_target_larger_than_series(self):
        agg = TimeSeriesAggregator()
        series = [(1.0, 1.0), (2.0, 2.0)]
        down = agg.downsampling(series, target_points=10)
        assert down == series

    def test_empty_series(self):
        agg = TimeSeriesAggregator()
        assert agg.downsampling([], 5) == []

    def test_zero_target(self):
        agg = TimeSeriesAggregator()
        assert agg.downsampling([(1.0, 1.0)], 0) == []

    def test_downsample_min(self):
        agg = TimeSeriesAggregator()
        series = [(float(i), float(i)) for i in range(20)]
        down = agg.downsampling(series, 4, method=AggregationType.MIN)
        assert len(down) <= 5


# ── rolling computation ────────────────────────────────────────

class TestRolling:

    def test_rolling_mean(self):
        agg = TimeSeriesAggregator()
        series = [(float(i), float(i)) for i in range(6)]
        result = agg.compute_rolling(series, window_size=3, agg_type=AggregationType.MEAN)
        assert len(result) == 6
        # i=0: [0] -> 0, i=1: [0,1] -> 0.5, i=2: [0,1,2] -> 1
        assert result[0][1] == 0.0
        assert abs(result[1][1] - 0.5) < 1e-9
        assert abs(result[2][1] - 1.0) < 1e-9

    def test_rolling_sum(self):
        agg = TimeSeriesAggregator()
        series = [(float(i), float(i)) for i in range(4)]
        result = agg.compute_rolling(series, 2, AggregationType.SUM)
        assert result[0][1] == 0  # [0]
        assert result[1][1] == 1  # [0,1]
        assert result[2][1] == 3  # [1,2]
        assert result[3][1] == 5  # [2,3]

    def test_rolling_empty(self):
        agg = TimeSeriesAggregator()
        assert agg.compute_rolling([], 3) == []

    def test_rolling_window_larger_than_series(self):
        agg = TimeSeriesAggregator()
        series = [(1.0, 5.0), (2.0, 10.0)]
        result = agg.compute_rolling(series, 10, AggregationType.MAX)
        assert result[0][1] == 5.0
        assert result[1][1] == 10.0

    def test_rolling_zero_window(self):
        agg = TimeSeriesAggregator()
        assert agg.compute_rolling([(1.0, 1.0)], 0) == []


# ── outlier detection ──────────────────────────────────────────

class TestOutlierDetection:

    def test_iqr_finds_outlier(self):
        agg = TimeSeriesAggregator()
        series = [(float(i), v) for i, v in enumerate([1, 2, 3, 4, 5, 100])]
        outliers = agg.detect_outliers(series, method="iqr", threshold=1.5)
        assert 5 in outliers  # 100 is an outlier

    def test_iqr_no_outlier(self):
        agg = TimeSeriesAggregator()
        series = [(float(i), float(i)) for i in range(20)]
        outliers = agg.detect_outliers(series, method="iqr", threshold=3.0)
        assert len(outliers) == 0

    def test_zscore_outlier(self):
        agg = TimeSeriesAggregator()
        series = [(float(i), v) for i, v in enumerate([1, 2, 3, 4, 5, 6, 100])]
        outliers = agg.detect_outliers(series, method="zscore", threshold=2.0)
        assert 6 in outliers

    def test_mad_outlier(self):
        agg = TimeSeriesAggregator()
        series = [(float(i), v) for i, v in enumerate([1, 2, 3, 4, 5, 6, 100])]
        outliers = agg.detect_outliers(series, method="mad", threshold=3.0)
        assert 6 in outliers

    def test_too_few_points(self):
        agg = TimeSeriesAggregator()
        assert agg.detect_outliers([(0, 1)], "iqr") == []

    def test_unknown_method_raises(self):
        agg = TimeSeriesAggregator()
        with pytest.raises(ValueError):
            agg.detect_outliers([(0, 1), (1, 2)], method="nonexistent")


# ── derivative ─────────────────────────────────────────────────

class TestDerivative:

    def test_constant_slope(self):
        agg = TimeSeriesAggregator()
        series = [(float(i), float(i * 2)) for i in range(5)]
        deriv = agg.compute_derivative(series)
        assert all(abs(d[1] - 2.0) < 1e-9 for d in deriv)

    def test_custom_dt(self):
        agg = TimeSeriesAggregator()
        series = [(0.0, 0.0), (1.0, 10.0), (2.0, 20.0)]
        deriv = agg.compute_derivative(series, dt=2.0)
        assert abs(deriv[0][1] - 5.0) < 1e-9
        assert abs(deriv[1][1] - 5.0) < 1e-9

    def test_single_point(self):
        agg = TimeSeriesAggregator()
        assert agg.compute_derivative([(0, 5)]) == []

    def test_empty_series(self):
        agg = TimeSeriesAggregator()
        assert agg.compute_derivative([]) == []

    def test_zero_dt_skipped(self):
        agg = TimeSeriesAggregator()
        series = [(1.0, 5.0), (1.0, 10.0)]  # same timestamp
        deriv = agg.compute_derivative(series)
        assert len(deriv) == 0
