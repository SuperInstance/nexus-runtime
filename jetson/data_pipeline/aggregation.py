"""Time-series aggregation — enums, dataclasses, and aggregator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AggregationType(Enum):
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    STD_DEV = "std_dev"
    MEDIAN = "median"
    FIRST = "first"
    LAST = "last"


@dataclass
class TimeWindow:
    """Result of an aggregation over a time window."""
    start: float
    end: float
    aggregation_type: AggregationType
    value: Any


class TimeSeriesAggregator:
    """Pure-Python time-series aggregation utilities."""

    # ── core aggregation ───────────────────────────────────────

    def aggregate(self, series: List[Tuple[float, float]],
                  window_seconds: float,
                  agg_type: AggregationType) -> List[TimeWindow]:
        """Aggregate a (timestamp, value) series into fixed-length windows."""
        if not series:
            return []
        series_sorted = sorted(series, key=lambda p: p[0])
        start = series_sorted[0][0]
        end = series_sorted[-1][0]
        windows: List[TimeWindow] = []
        win_start = start
        while win_start < end:
            win_end = win_start + window_seconds
            values = [v for ts, v in series_sorted if win_start <= ts < win_end]
            agg_value = self._compute_agg(values, agg_type)
            windows.append(TimeWindow(
                start=win_start, end=win_end,
                aggregation_type=agg_type, value=agg_value,
            ))
            win_start = win_end
        return windows

    def aggregate_multiple(self, series: List[Tuple[float, float]],
                           window_seconds: float,
                           agg_types: List[AggregationType]) -> Dict[AggregationType, List[TimeWindow]]:
        """Aggregate using multiple aggregation types simultaneously."""
        result: Dict[AggregationType, List[TimeWindow]] = {}
        for at in agg_types:
            result[at] = self.aggregate(series, window_seconds, at)
        return result

    # ── downsampling ───────────────────────────────────────────

    def downsampling(self, series: List[Tuple[float, float]],
                     target_points: int,
                     method: AggregationType = AggregationType.MEAN) -> List[Tuple[float, float]]:
        """Downsample a series to approximately *target_points* points."""
        if not series or target_points <= 0:
            return []
        if target_points >= len(series):
            return list(series)
        n = len(series)
        step = max(1, n / target_points)
        result: List[Tuple[float, float]] = []
        i = 0.0
        while int(i) < n:
            batch_start = int(i)
            batch_end = min(n, int(i + step))
            batch_end = max(batch_end, batch_start + 1)
            window = series[batch_start:batch_end]
            values = [v for _, v in window]
            agg = self._compute_agg(values, method)
            ts = window[len(window) // 2][0]
            result.append((ts, agg))
            i += step
        return result

    # ── rolling computation ────────────────────────────────────

    def compute_rolling(self, series: List[Tuple[float, float]],
                        window_size: int,
                        agg_type: AggregationType = AggregationType.MEAN) -> List[Tuple[float, float]]:
        """Compute rolling aggregation over *window_size* points."""
        if not series or window_size <= 0:
            return []
        result: List[Tuple[float, float]] = []
        for i in range(len(series)):
            start_idx = max(0, i - window_size + 1)
            window = [v for _, v in series[start_idx:i + 1]]
            agg = self._compute_agg(window, agg_type)
            result.append((series[i][0], agg))
        return result

    # ── outlier detection ──────────────────────────────────────

    def detect_outliers(self, series: List[Tuple[float, float]],
                        method: str = "iqr",
                        threshold: float = 1.5) -> List[int]:
        """Return indices of outlier points.

        *method*:
        - ``"iqr"``: Inter-Quartile Range (threshold = IQR multiplier).
        - ``"zscore"``: Z-score (threshold = absolute z threshold).
        - ``"mad"``: Median Absolute Deviation (threshold = MAD multiplier).
        """
        if len(series) < 2:
            return []
        values = [v for _, v in series]
        if method == "iqr":
            return self._outliers_iqr(values, threshold)
        elif method == "zscore":
            return self._outliers_zscore(values, threshold)
        elif method == "mad":
            return self._outliers_mad(values, threshold)
        else:
            raise ValueError(f"Unknown outlier method: {method}")

    # ── derivative ─────────────────────────────────────────────

    def compute_derivative(self, series: List[Tuple[float, float]],
                           dt: Optional[float] = None) -> List[Tuple[float, float]]:
        """Compute numerical derivative (dv/dt).

        If *dt* is provided it overrides the timestamp delta; otherwise
        the actual time difference between consecutive points is used.
        """
        if len(series) < 2:
            return []
        result: List[Tuple[float, float]] = []
        for i in range(1, len(series)):
            t0, v0 = series[i - 1]
            t1, v1 = series[i]
            delta_t = dt if dt is not None else (t1 - t0)
            if delta_t == 0:
                continue
            result.append((t1, (v1 - v0) / delta_t))
        return result

    # ── private helpers ────────────────────────────────────────

    @staticmethod
    def _compute_agg(values: List[float], agg_type: AggregationType) -> Any:
        if not values:
            return None
        if agg_type == AggregationType.MEAN:
            return sum(values) / len(values)
        if agg_type == AggregationType.MIN:
            return min(values)
        if agg_type == AggregationType.MAX:
            return max(values)
        if agg_type == AggregationType.SUM:
            return sum(values)
        if agg_type == AggregationType.COUNT:
            return len(values)
        if agg_type == AggregationType.STD_DEV:
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            return math.sqrt(var)
        if agg_type == AggregationType.MEDIAN:
            s = sorted(values)
            n = len(s)
            mid = n // 2
            if n % 2 == 0:
                return (s[mid - 1] + s[mid]) / 2.0
            return s[mid]
        if agg_type == AggregationType.FIRST:
            return values[0]
        if agg_type == AggregationType.LAST:
            return values[-1]
        return None

    @staticmethod
    def _percentile(sorted_vals: List[float], p: float) -> float:
        """Linear-interpolation percentile on *already sorted* list."""
        n = len(sorted_vals)
        k = (n - 1) * p
        f = int(k)
        c = f + 1
        if c >= n:
            return sorted_vals[-1]
        d = k - f
        return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])

    def _outliers_iqr(self, values: List[float], threshold: float) -> List[int]:
        s = sorted(values)
        q1 = self._percentile(s, 0.25)
        q3 = self._percentile(s, 0.75)
        iqr = q3 - q1
        lo = q1 - threshold * iqr
        hi = q3 + threshold * iqr
        return [i for i, v in enumerate(values) if v < lo or v > hi]

    def _outliers_zscore(self, values: List[float], threshold: float) -> List[int]:
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0
        if std == 0:
            return []
        return [i for i, v in enumerate(values) if abs((v - mean) / std) > threshold]

    def _outliers_mad(self, values: List[float], threshold: float) -> List[int]:
        s = sorted(values)
        median = self._percentile(s, 0.5)
        mad_vals = sorted(abs(v - median) for v in values)
        mad = self._percentile(mad_vals, 0.5)
        if mad == 0:
            return []
        return [i for i, v in enumerate(values) if abs(v - median) / mad > threshold]
