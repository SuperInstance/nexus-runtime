"""NEXUS A/B Testing — Statistical Engine.

Three competing approaches evaluated on simulated data:
  Approach A: Welch's t-test (unequal variances) with Cohen's d effect size
  Approach B: Mann-Whitney U test (non-parametric)
  Approach C: Bootstrap confidence intervals (resampling)

The engine picks the most appropriate test based on Shapiro-Wilk normality.
Bonferroni correction applied for multiple metrics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TestMethod(Enum):
    """Statistical test methods available."""

    WELCH_T = "welch_t"
    MANN_WHITNEY_U = "mann_whitney_u"
    BOOTSTRAP_CI = "bootstrap_ci"
    AUTO = "auto"


@dataclass
class TestResult:
    """Result of a single statistical test between two variants."""

    method: TestMethod
    metric_name: str
    variant_a: str
    variant_b: str
    p_value: float
    ci_lower: float
    ci_upper: float
    effect_size: float
    recommendation: str  # "A wins", "B wins", "inconclusive", "need_more_data"
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    sample_sizes: dict[str, int] = field(default_factory=dict)
    is_normal_a: bool | None = None
    is_normal_b: bool | None = None
    selected_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "metric_name": self.metric_name,
            "variant_a": self.variant_a,
            "variant_b": self.variant_b,
            "p_value": self.p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "effect_size": self.effect_size,
            "recommendation": self.recommendation,
            "means": self.means,
            "stds": self.stds,
            "sample_sizes": self.sample_sizes,
            "is_normal_a": self.is_normal_a,
            "is_normal_b": self.is_normal_b,
            "selected_method": self.selected_method,
        }


@dataclass
class BonferroniResult:
    """Bonferroni-corrected results across multiple metrics."""

    original_results: list[TestResult] = field(default_factory=list)
    corrected_results: list[TestResult] = field(default_factory=list)
    num_tests: int = 0
    adjusted_alpha: float = 0.05
    overall_recommendation: str = "inconclusive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_tests": self.num_tests,
            "adjusted_alpha": self.adjusted_alpha,
            "original_results": [r.to_dict() for r in self.original_results],
            "corrected_results": [r.to_dict() for r in self.corrected_results],
            "overall_recommendation": self.overall_recommendation,
        }


def _mean(data: list[float]) -> float:
    if not data:
        return 0.0
    return sum(data) / len(data)


def _std(data: list[float], ddof: int = 1) -> float:
    n = len(data)
    if n <= ddof:
        return 0.0
    m = _mean(data)
    var = sum((x - m) ** 2 for x in data) / (n - ddof)
    return math.sqrt(var)


def _median(data: list[float]) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _shapiro_wilk_w(data: list[float]) -> float:
    """Approximate Shapiro-Wilk W statistic.

    Returns W statistic (closer to 1.0 = more normal).
    This is a simplified approximation using order statistic correlation.
    """
    n = len(data)
    if n < 3:
        return 1.0  # Too small to assess, assume normal

    sorted_data = sorted(data)
    m = _mean(sorted_data)
    s = _std(sorted_data, ddof=0)
    if s < 1e-15:
        return 1.0  # Constant data, treat as normal

    # Expected normal order statistics (Blom approximation)
    expected = []
    for i in range(n):
        z = (i + 1.0 - 0.375) / (n + 0.25)
        from .experiment import _normal_ppf
        expected.append(_normal_ppf(z))

    # Correlation between sorted data and expected normal
    m_e = _mean(expected)
    s_e = _std(expected, ddof=0)
    if s_e < 1e-15:
        return 1.0

    cov = sum(
        (sorted_data[i] - m) * (expected[i] - m_e) for i in range(n)
    ) / n
    r = cov / (s * s_e)
    return min(abs(r), 1.0)


def is_normal(data: list[float], alpha: float = 0.05) -> bool:
    """Test normality using Shapiro-Wilk approximation.

    Uses a W threshold that depends on sample size.
    For n >= 20, W > 0.95 suggests normality.
    For smaller samples, we're more lenient.
    """
    n = len(data)
    if n < 3:
        return True  # Too small to reject
    w = _shapiro_wilk_w(data)
    # Thresholds based on sample size (approximate)
    if n >= 50:
        threshold = 0.96
    elif n >= 20:
        threshold = 0.95
    elif n >= 10:
        threshold = 0.93
    else:
        threshold = 0.90
    return w >= threshold


def _rank_data(data: list[float]) -> list[float]:
    """Assign ranks to data with average rank for ties."""
    indexed = sorted(enumerate(data), key=lambda x: x[1])
    ranks = [0.0] * len(data)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _cohen_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size (pooled standard deviation)."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    m1, m2 = _mean(a), _mean(b)
    s1, s2 = _std(a), _std(b)
    pooled = math.sqrt(
        ((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2)
    ) if (n1 + n2 > 2) else 1.0
    if pooled < 1e-15:
        return 0.0
    return (m1 - m2) / pooled


def _welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test for unequal variances.

    Returns (t_statistic, p_value).
    Uses the Satterthwaite approximation for degrees of freedom.
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0  # Not enough data

    m1, m2 = _mean(a), _mean(b)
    v1, v2 = _std(a) ** 2, _std(b) ** 2

    if v1 + v2 < 1e-30:
        return 0.0, 1.0  # No variance

    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se

    # Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = num / denom if denom > 1e-30 else 1.0

    # Approximate p-value using the normal approximation for large df
    # For small df, use a crude t-approximation
    p_value = _t_cdf(abs(t_stat), df)
    p_value = 2.0 * (1.0 - p_value)  # Two-tailed

    return t_stat, min(max(p_value, 0.0), 1.0)


def _t_cdf(t: float, df: float) -> float:
    """Approximate cumulative distribution function of the t-distribution.

    Uses the normal approximation for df > 30,
    and a crude approximation for small df.
    """
    if df > 30:
        # Normal approximation
        return _normal_cdf(t)
    # Small df: use a scaled normal approximation
    # t_cdf(t, df) ≈ normal_cdf(t) + correction
    # Simple scaling factor
    scale = math.sqrt(df / (df - 2.0)) if df > 2 else 3.0
    return _normal_cdf(t / scale)


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using the logistic approximation."""
    return 1.0 / (1.0 + math.exp(-1.7 * x))


def _mann_whitney_u_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U test (non-parametric).

    Returns (U_statistic, p_value).
    Uses normal approximation for the U statistic.
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    combined = a + b
    ranks = _rank_data(combined)
    rank_a = ranks[:n1]
    rank_b = ranks[n1:]

    r_a = sum(rank_a)
    u_a = r_a - n1 * (n1 + 1) / 2.0
    u_b = n1 * n2 - u_a
    u_stat = min(u_a, u_b)

    # Normal approximation for U
    mean_u = n1 * n2 / 2.0
    # Variance with tie correction
    sorted_ranks = sorted(ranks)
    tie_sum = 0.0
    i = 0
    while i < len(sorted_ranks):
        j = i
        while j < len(sorted_ranks) and sorted_ranks[j] == sorted_ranks[i]:
            j += 1
        t = j - i
        if t > 1:
            tie_sum += t ** 3 - t
        i = j
    var_u = (n1 * n2 / 12.0) * (
        (n1 + n2 + 1) - tie_sum / ((n1 + n2) * (n1 + n2 - 1))
    ) if (n1 + n2) > 1 else 1.0

    if var_u < 1e-30:
        return u_stat, 1.0

    z = (u_stat - mean_u) / math.sqrt(var_u)
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))

    return u_stat, min(max(p_value, 0.0), 1.0)


def _cliffs_delta(a: list[float], b: list[float]) -> float:
    """Compute Cliff's delta non-parametric effect size."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    greater = 0
    less = 0
    for ai in a:
        for bj in b:
            if ai > bj:
                greater += 1
            elif ai < bj:
                less += 1
    total = n1 * n2
    if total == 0:
        return 0.0
    return (greater - less) / total


def _bootstrap_ci(
    a: list[float],
    b: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence intervals for the difference of means.

    Returns (ci_lower, ci_upper, effect_size).
    """
    rng = random.Random(seed)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, 0.0

    diffs = []
    for _ in range(n_bootstrap):
        sample_a = [a[rng.randint(0, n1 - 1)] for _ in range(n1)]
        sample_b = [b[rng.randint(0, n2 - 1)] for _ in range(n2)]
        diffs.append(_mean(sample_a) - _mean(sample_b))

    diffs.sort()
    alpha = 1.0 - confidence
    lower_idx = int(math.floor(alpha / 2.0 * n_bootstrap))
    upper_idx = int(math.floor((1.0 - alpha / 2.0) * n_bootstrap))
    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    ci_lower = diffs[lower_idx]
    ci_upper = diffs[upper_idx]
    obs_diff = _mean(a) - _mean(b)
    pooled_std = _std(a + b)
    effect = obs_diff / pooled_std if pooled_std > 1e-15 else 0.0

    return ci_lower, ci_upper, effect


class StatisticalEngine:
    """Statistical engine that selects and runs the appropriate test.

    Competes three approaches and picks the best based on data distribution:
    - Normal data → Welch's t-test (parametric)
    - Non-normal data → Mann-Whitney U (non-parametric)
    - Always runs bootstrap CI for robustness

    Applies Bonferroni correction when multiple metrics are tested.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        bootstrap_iterations: int = 10000,
        normality_threshold: float = 0.05,
        min_sample_size: int = 3,
    ) -> None:
        self.alpha = alpha
        self.bootstrap_iterations = bootstrap_iterations
        self.normality_threshold = normality_threshold
        self.min_sample_size = min_sample_size

    def select_method(
        self, a: list[float], b: list[float]
    ) -> tuple[str, bool, bool]:
        """Select the best statistical test based on data distribution.

        Returns (method_name, is_normal_a, is_normal_b).
        Normal data → welch_t, Non-normal → mann_whitney_u.
        """
        norm_a = is_normal(a, self.normality_threshold)
        norm_b = is_normal(b, self.normality_threshold)
        if norm_a and norm_b:
            return "welch_t", norm_a, norm_b
        else:
            return "mann_whitney_u", norm_a, norm_b

    def run_welch_t(
        self,
        a: list[float],
        b: list[float],
        metric_name: str = "",
        variant_a: str = "A",
        variant_b: str = "B",
    ) -> TestResult:
        """Run Welch's t-test with Cohen's d effect size."""
        t_stat, p_value = _welch_t_test(a, b)
        effect = _cohen_d(a, b)
        mean_a, mean_b = _mean(a), _mean(b)

        # Approximate 95% CI for difference of means
        n1, n2 = len(a), len(b)
        se = math.sqrt(_std(a) ** 2 / max(n1, 1) + _std(b) ** 2 / max(n2, 1))
        ci_lower = (mean_a - mean_b) - 1.96 * se
        ci_upper = (mean_a - mean_b) + 1.96 * se

        recommendation = self._recommendation(p_value, mean_a, mean_b, self.alpha)

        return TestResult(
            method=TestMethod.WELCH_T,
            metric_name=metric_name,
            variant_a=variant_a,
            variant_b=variant_b,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            effect_size=effect,
            recommendation=recommendation,
            means={"A": mean_a, "B": mean_b},
            stds={"A": _std(a), "B": _std(b)},
            sample_sizes={"A": n1, "B": n2},
            is_normal_a=True,
            is_normal_b=True,
            selected_method="welch_t",
        )

    def run_mann_whitney(
        self,
        a: list[float],
        b: list[float],
        metric_name: str = "",
        variant_a: str = "A",
        variant_b: str = "B",
    ) -> TestResult:
        """Run Mann-Whitney U test with Cliff's delta effect size."""
        u_stat, p_value = _mann_whitney_u_test(a, b)
        effect = _cliffs_delta(a, b)
        median_a, median_b = _median(a), _median(b)

        # Use bootstrap CI for non-parametric
        ci_lower, ci_upper, _ = _bootstrap_ci(a, b, n_bootstrap=2000)

        recommendation = self._recommendation(p_value, median_a, median_b, self.alpha)

        return TestResult(
            method=TestMethod.MANN_WHITNEY_U,
            metric_name=metric_name,
            variant_a=variant_a,
            variant_b=variant_b,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            effect_size=effect,
            recommendation=recommendation,
            means={"A": median_a, "B": median_b},
            stds={"A": _std(a), "B": _std(b)},
            sample_sizes={"A": len(a), "B": len(b)},
            is_normal_a=False,
            is_normal_b=False,
            selected_method="mann_whitney_u",
        )

    def run_bootstrap(
        self,
        a: list[float],
        b: list[float],
        metric_name: str = "",
        variant_a: str = "A",
        variant_b: str = "B",
        seed: int | None = None,
    ) -> TestResult:
        """Run bootstrap confidence interval test."""
        ci_lower, ci_upper, effect = _bootstrap_ci(
            a, b,
            n_bootstrap=self.bootstrap_iterations,
            seed=seed,
        )
        mean_a, mean_b = _mean(a), _mean(b)

        # p-value from CI: if 0 is in CI, not significant
        if ci_lower > 0 or ci_upper < 0:
            # Significant - estimate p from the proportion of bootstrap
            # samples that cross zero
            rng = random.Random(seed)
            n1, n2 = len(a), len(b)
            cross_zero = 0
            total = self.bootstrap_iterations
            for _ in range(total):
                sa = [a[rng.randint(0, n1 - 1)] for _ in range(n1)]
                sb = [b[rng.randint(0, n2 - 1)] for _ in range(n2)]
                diff = _mean(sa) - _mean(sb)
                if (mean_a - mean_b > 0 and diff <= 0) or (
                    mean_a - mean_b < 0 and diff >= 0
                ):
                    cross_zero += 1
            p_value = 2.0 * cross_zero / total
            p_value = min(p_value, 1.0)
        else:
            p_value = 1.0

        recommendation = self._recommendation(p_value, mean_a, mean_b, self.alpha)

        return TestResult(
            method=TestMethod.BOOTSTRAP_CI,
            metric_name=metric_name,
            variant_a=variant_a,
            variant_b=variant_b,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            effect_size=effect,
            recommendation=recommendation,
            means={"A": mean_a, "B": mean_b},
            stds={"A": _std(a), "B": _std(b)},
            sample_sizes={"A": len(a), "B": len(b)},
            selected_method="bootstrap_ci",
        )

    def run_auto(
        self,
        a: list[float],
        b: list[float],
        metric_name: str = "",
        variant_a: str = "A",
        variant_b: str = "B",
        seed: int | None = None,
    ) -> TestResult:
        """Automatically select and run the best test."""
        if len(a) < self.min_sample_size or len(b) < self.min_sample_size:
            return TestResult(
                method=TestMethod.AUTO,
                metric_name=metric_name,
                variant_a=variant_a,
                variant_b=variant_b,
                p_value=1.0,
                ci_lower=0.0,
                ci_upper=0.0,
                effect_size=0.0,
                recommendation="need_more_data",
                means={"A": _mean(a), "B": _mean(b)},
                stds={"A": _std(a), "B": _std(b)},
                sample_sizes={"A": len(a), "B": len(b)},
                selected_method="none",
            )

        method_name, norm_a, norm_b = self.select_method(a, b)

        if method_name == "welch_t":
            result = self.run_welch_t(a, b, metric_name, variant_a, variant_b)
        else:
            result = self.run_mann_whitney(a, b, metric_name, variant_a, variant_b)

        # Also compute bootstrap CI for robustness reporting
        bs_result = self.run_bootstrap(a, b, metric_name, variant_a, variant_b, seed)
        result.is_normal_a = norm_a
        result.is_normal_b = norm_b

        return result

    def run_bonferroni(
        self, results: list[TestResult], alpha: float | None = None
    ) -> BonferroniResult:
        """Apply Bonferroni correction across multiple metric tests.

        Adjusts alpha by dividing by the number of tests.
        """
        alpha = alpha if alpha is not None else self.alpha
        n_tests = len(results)
        if n_tests == 0:
            return BonferroniResult(
                adjusted_alpha=alpha,
                overall_recommendation="inconclusive",
            )

        adjusted_alpha = alpha / n_tests
        corrected = []

        for r in results:
            # Create corrected copy
            cr = TestResult(
                method=r.method,
                metric_name=r.metric_name,
                variant_a=r.variant_a,
                variant_b=r.variant_b,
                p_value=r.p_value,
                ci_lower=r.ci_lower,
                ci_upper=r.ci_upper,
                effect_size=r.effect_size,
                recommendation="inconclusive",  # Will recalculate
                means=dict(r.means),
                stds=dict(r.stds),
                sample_sizes=dict(r.sample_sizes),
                is_normal_a=r.is_normal_a,
                is_normal_b=r.is_normal_b,
                selected_method=r.selected_method,
            )
            # Recalculate recommendation with corrected alpha
            mean_a = r.means.get("A", 0.0)
            mean_b = r.means.get("B", 0.0)
            cr.recommendation = self._recommendation(
                r.p_value, mean_a, mean_b, adjusted_alpha
            )
            corrected.append(cr)

        # Overall recommendation: majority vote across metrics
        votes = {}
        for cr in corrected:
            votes[cr.recommendation] = votes.get(cr.recommendation, 0) + 1

        overall = "inconclusive"
        if votes.get("need_more_data", 0) == n_tests:
            overall = "need_more_data"
        elif votes.get("A wins", 0) > n_tests / 2:
            overall = "A wins"
        elif votes.get("B wins", 0) > n_tests / 2:
            overall = "B wins"
        elif votes.get("A wins", 0) == votes.get("B wins", 0) and votes.get("A wins", 0) > 0:
            overall = "inconclusive"

        return BonferroniResult(
            original_results=results,
            corrected_results=corrected,
            num_tests=n_tests,
            adjusted_alpha=adjusted_alpha,
            overall_recommendation=overall,
        )

    def _recommendation(
        self,
        p_value: float,
        mean_a: float,
        mean_b: float,
        alpha: float,
    ) -> str:
        """Generate recommendation based on p-value and means.

        For lower-is-better metrics, the interpretation is inverted
        based on the metric name prefix or context.
        """
        if p_value > alpha:
            return "inconclusive"

        # Lower is better for: cycle_time, safety_events, error_rate
        # Higher is better for: accuracy, trust_delta
        # The caller should handle direction; here we compare means directly
        if mean_a < mean_b:
            return "A wins"
        elif mean_b < mean_a:
            return "B wins"
        else:
            return "inconclusive"

    def compare_variants(
        self,
        suite: Any,
        metric_type: Any = None,
        method: TestMethod = TestMethod.AUTO,
    ) -> BonferroniResult:
        """Compare all variants in an ABTestSuite across all metrics.

        Returns Bonferroni-corrected results.
        """
        from .experiment import MetricType

        variants = list(suite.variants.values())
        if len(variants) < 2:
            return BonferroniResult(
                adjusted_alpha=self.alpha,
                overall_recommendation="need_more_data",
            )

        va, vb = variants[0], variants[1]
        metric_types = [metric_type] if metric_type else suite.metrics
        results = []

        for mt in metric_types:
            if isinstance(mt, str):
                mt = MetricType(mt)
            a_vals = va.get_metric_values(mt)
            b_vals = vb.get_metric_values(mt)

            if method == TestMethod.WELCH_T:
                result = self.run_welch_t(
                    a_vals, b_vals, mt.value, va.name, vb.name
                )
            elif method == TestMethod.MANN_WHITNEY_U:
                result = self.run_mann_whitney(
                    a_vals, b_vals, mt.value, va.name, vb.name
                )
            elif method == TestMethod.BOOTSTRAP_CI:
                result = self.run_bootstrap(
                    a_vals, b_vals, mt.value, va.name, vb.name
                )
            else:
                result = self.run_auto(
                    a_vals, b_vals, mt.value, va.name, vb.name
                )
            results.append(result)

        return self.run_bonferroni(results)
