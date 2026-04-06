"""Tests for IEC 61508 SIL Verification Engine."""

import pytest
import math
from dataclasses import dataclass

from jetson.compliance.iec61508 import (
    SILLevel,
    SILTarget,
    SILVerificationResult,
    SILVerifier,
)


class TestSILLevel:
    """Tests for SILLevel enum."""

    def test_sil_1_value(self):
        assert SILLevel.SIL_1.value == 1

    def test_sil_2_value(self):
        assert SILLevel.SIL_2.value == 2

    def test_sil_3_value(self):
        assert SILLevel.SIL_3.value == 3

    def test_sil_4_value(self):
        assert SILLevel.SIL_4.value == 4

    def test_sil_ordering(self):
        assert SILLevel.SIL_1.value < SILLevel.SIL_2.value < SILLevel.SIL_3.value < SILLevel.SIL_4.value

    def test_all_levels_defined(self):
        levels = list(SILLevel)
        assert len(levels) == 4


class TestSILTarget:
    """Tests for SILTarget dataclass."""

    def test_create_target(self):
        target = SILTarget(
            safety_function="collision_avoidance",
            required_sil=SILLevel.SIL_2,
        )
        assert target.safety_function == "collision_avoidance"
        assert target.required_sil == SILLevel.SIL_2

    def test_default_performance(self):
        target = SILTarget(
            safety_function="emergency_stop",
            required_sil=SILLevel.SIL_3,
        )
        assert target.current_performance == 0.0

    def test_custom_performance(self):
        target = SILTarget(
            safety_function="obstacle_detection",
            required_sil=SILLevel.SIL_1,
            current_performance=0.85,
        )
        assert target.current_performance == 0.85


class TestSILVerificationResult:
    """Tests for SILVerificationResult dataclass."""

    def test_default_result(self):
        result = SILVerificationResult(target=SILTarget("test", SILLevel.SIL_1))
        assert result.achieved_sil is None
        assert result.pass_fail is False
        assert result.gaps == []
        assert result.recommendations == []

    def test_pass_result(self):
        target = SILTarget("test", SILLevel.SIL_1)
        result = SILVerificationResult(
            target=target,
            achieved_sil=SILLevel.SIL_1,
            pass_fail=True,
        )
        assert result.pass_fail is True


class TestSILVerifierComputeHazardRate:
    """Tests for SILVerifier.compute_hazard_rate."""

    def setup_method(self):
        self.verifier = SILVerifier()

    def test_basic_computation(self):
        hr = self.verifier.compute_hazard_rate(1e-5, 8760, 1.0)
        assert hr == pytest.approx(1e-5 * 8760)

    def test_with_probability(self):
        hr = self.verifier.compute_hazard_rate(1e-5, 8760, 0.5)
        assert hr == pytest.approx(1e-5 * 8760 * 0.5)

    def test_zero_failure_rate(self):
        hr = self.verifier.compute_hazard_rate(0.0, 8760, 1.0)
        assert hr == 0.0

    def test_zero_exposure(self):
        hr = self.verifier.compute_hazard_rate(1e-5, 0.0, 1.0)
        assert hr == 0.0

    def test_zero_probability(self):
        hr = self.verifier.compute_hazard_rate(1e-5, 8760, 0.0)
        assert hr == 0.0

    def test_negative_failure_rate_raises(self):
        with pytest.raises(ValueError, match="failure_rate must be non-negative"):
            self.verifier.compute_hazard_rate(-1e-5, 8760)

    def test_negative_exposure_raises(self):
        with pytest.raises(ValueError, match="exposure_time must be non-negative"):
            self.verifier.compute_hazard_rate(1e-5, -1)

    def test_probability_above_one_raises(self):
        with pytest.raises(ValueError, match="probability must be between 0 and 1"):
            self.verifier.compute_hazard_rate(1e-5, 8760, 1.5)

    def test_probability_below_zero_raises(self):
        with pytest.raises(ValueError, match="probability must be between 0 and 1"):
            self.verifier.compute_hazard_rate(1e-5, 8760, -0.1)


class TestSILVerifierComputePFD:
    """Tests for SILVerifier.compute_pfd."""

    def setup_method(self):
        self.verifier = SILVerifier()

    def test_full_diagnostic_coverage(self):
        pfd = self.verifier.compute_pfd(1.0, 8760, 0.0)
        assert pfd == 0.0

    def test_no_diagnostic_coverage(self):
        pfd = self.verifier.compute_pfd(0.0, 8760, 0.0)
        assert pfd == pytest.approx(4380.0)

    def test_partial_coverage(self):
        pfd = self.verifier.compute_pfd(0.9, 8760, 0.0)
        assert pfd == pytest.approx(438.0)

    def test_with_beta_factor(self):
        pfd = self.verifier.compute_pfd(0.9, 8760, 0.01)
        assert pfd == pytest.approx(438.01)

    def test_dc_negative_raises(self):
        with pytest.raises(ValueError):
            self.verifier.compute_pfd(-0.1, 8760)

    def test_dc_above_one_raises(self):
        with pytest.raises(ValueError):
            self.verifier.compute_pfd(1.1, 8760)

    def test_zero_interval_raises(self):
        with pytest.raises(ValueError):
            self.verifier.compute_pfd(0.9, 0)

    def test_negative_interval_raises(self):
        with pytest.raises(ValueError):
            self.verifier.compute_pfd(0.9, -1)

    def test_beta_above_one_raises(self):
        with pytest.raises(ValueError):
            self.verifier.compute_pfd(0.9, 8760, 1.1)

    def test_pfd_never_negative(self):
        pfd = self.verifier.compute_pfd(0.99, 1, 0.0)
        assert pfd >= 0.0


class TestSILVerifierComputeSFF:
    """Tests for SILVerifier.compute_sff."""

    def setup_method(self):
        self.verifier = SILVerifier()

    def test_all_safe(self):
        sff = self.verifier.compute_sff(10.0, 10.0)
        assert sff == pytest.approx(1.0)

    def test_half_safe(self):
        sff = self.verifier.compute_sff(5.0, 10.0)
        assert sff == pytest.approx(0.5)

    def test_no_safe(self):
        sff = self.verifier.compute_sff(0.0, 10.0)
        assert sff == pytest.approx(0.0)

    def test_zero_total_raises(self):
        with pytest.raises(ValueError):
            self.verifier.compute_sff(5.0, 0.0)

    def test_negative_safe_raises(self):
        with pytest.raises(ValueError):
            self.verifier.compute_sff(-1.0, 10.0)

    def test_safe_clamped_to_total(self):
        sff = self.verifier.compute_sff(15.0, 10.0)
        assert sff == pytest.approx(1.0)


class TestSILVerifierCheckArchitecture:
    """Tests for SILVerifier.check_sil_architecture."""

    def setup_method(self):
        self.verifier = SILVerifier()

    def test_sil1_type_a_single_channel(self):
        valid, issues = self.verifier.check_sil_architecture(SILLevel.SIL_1, "Type A", 1)
        assert valid is True
        assert issues == []

    def test_sil2_single_channel_insufficient_hft(self):
        valid, issues = self.verifier.check_sil_architecture(SILLevel.SIL_2, "Type B", 1)
        assert valid is False
        assert any("hardware fault tolerance" in i for i in issues)

    def test_sil2_dual_channel(self):
        valid, issues = self.verifier.check_sil_architecture(SILLevel.SIL_2, "Type B", 2)
        assert valid is True

    def test_sil3_dual_channel(self):
        valid, issues = self.verifier.check_sil_architecture(SILLevel.SIL_3, "Type B", 2)
        assert valid is True

    def test_sil4_triple_channel(self):
        valid, issues = self.verifier.check_sil_architecture(SILLevel.SIL_4, "Type B", 3)
        assert valid is True

    def test_zero_channels(self):
        valid, issues = self.verifier.check_sil_architecture(SILLevel.SIL_1, "Type A", 0)
        assert valid is False

    def test_unknown_architecture_type(self):
        valid, issues = self.verifier.check_sil_architecture(SILLevel.SIL_1, "Type C", 1)
        assert valid is False
        assert any("Unknown architecture" in i for i in issues)


class TestSILVerifierVerifySIL:
    """Tests for SILVerifier.verify_sil."""

    def setup_method(self):
        self.verifier = SILVerifier()

    def test_achieves_target(self):
        target = SILTarget("test", SILLevel.SIL_1)
        result = self.verifier.verify_sil(target, 5e-2, 0.95, 0.95)
        assert result.achieved_sil == SILLevel.SIL_1
        assert result.pass_fail is True

    def test_fails_target(self):
        target = SILTarget("test", SILLevel.SIL_3)
        result = self.verifier.verify_sil(target, 5e-2, 0.5, 0.3)
        assert result.pass_fail is False
        assert len(result.gaps) > 0

    def test_sil_4_with_good_coverage(self):
        target = SILTarget("test", SILLevel.SIL_4)
        result = self.verifier.verify_sil(target, 5e-6, 0.99, 0.995)
        assert result.achieved_sil == SILLevel.SIL_4

    def test_has_recommendations_when_failing(self):
        target = SILTarget("test", SILLevel.SIL_3)
        result = self.verifier.verify_sil(target, 5e-2, 0.5, 0.3)
        assert len(result.recommendations) > 0

    def test_no_recommendations_when_passing(self):
        target = SILTarget("test", SILLevel.SIL_1)
        result = self.verifier.verify_sil(target, 5e-2, 0.95, 0.95)
        assert len(result.recommendations) == 0

    def test_very_low_hazard_rate(self):
        target = SILTarget("test", SILLevel.SIL_3)
        result = self.verifier.verify_sil(target, 1e-7, 0.95, 0.99)
        assert result.achieved_sil is not None


class TestSILVerifierRecommendMeasures:
    """Tests for SILVerifier.recommend_measures."""

    def setup_method(self):
        self.verifier = SILVerifier()

    def test_same_sil_no_gaps(self):
        measures = self.verifier.recommend_measures(SILLevel.SIL_2, SILLevel.SIL_2, [])
        assert measures == []

    def test_current_none_returns_basic(self):
        measures = self.verifier.recommend_measures(None, SILLevel.SIL_1, [])
        assert len(measures) >= 1

    def test_gap_of_one_has_measures(self):
        measures = self.verifier.recommend_measures(SILLevel.SIL_1, SILLevel.SIL_2, [])
        assert len(measures) >= 3

    def test_gap_of_two_has_redesign(self):
        measures = self.verifier.recommend_measures(SILLevel.SIL_1, SILLevel.SIL_3, [])
        assert any("redesign" in m.lower() for m in measures)

    def test_diagnostic_gap_in_recommendations(self):
        gaps = ["Diagnostic coverage 50% below required 60%"]
        measures = self.verifier.recommend_measures(SILLevel.SIL_1, SILLevel.SIL_2, gaps)
        assert any("diagnostic" in m.lower() for m in measures)

    def test_test_coverage_gap(self):
        gaps = ["Test coverage 60% below recommended 80%"]
        measures = self.verifier.recommend_measures(SILLevel.SIL_1, SILLevel.SIL_2, gaps)
        assert any("test" in m.lower() for m in measures)

    def test_hft_gap(self):
        gaps = ["Insufficient hardware fault tolerance"]
        measures = self.verifier.recommend_measures(SILLevel.SIL_1, SILLevel.SIL_2, gaps)
        assert any("redundant" in m.lower() or "channel" in m.lower() for m in measures)
