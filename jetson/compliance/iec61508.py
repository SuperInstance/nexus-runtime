"""IEC 61508 SIL (Safety Integrity Level) Verification Engine.

Implements safety integrity level verification per IEC 61508 for
marine robotics safety functions, including hazard rate computation,
PFD calculation, SFF computation, architecture checks, and
recommendation generation.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
import math


class SILLevel(Enum):
    """Safety Integrity Levels per IEC 61508."""
    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    SIL_4 = 4


@dataclass
class SILTarget:
    """Target SIL for a safety function."""
    safety_function: str
    required_sil: SILLevel
    current_performance: float = 0.0


@dataclass
class SILVerificationResult:
    """Result of SIL verification."""
    target: SILTarget
    achieved_sil: Optional[SILLevel] = None
    pass_fail: bool = False
    gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# IEC 61508 Part 1, Table 2 / Part 3, Table 2
# PFD ranges for low demand mode and PFH ranges for high/continuous demand mode
SIL_PFD_RANGES = {
    SILLevel.SIL_1: (1e-2, 1e-1),
    SILLevel.SIL_2: (1e-3, 1e-2),
    SILLevel.SIL_3: (1e-4, 1e-3),
    SILLevel.SIL_4: (1e-5, 1e-4),
}

# SFF requirements per architecture type per SIL level
SFF_REQUIREMENTS = {
    SILLevel.SIL_1: {"Type A": 0.60, "Type B": 0.90},
    SILLevel.SIL_2: {"Type A": 0.90, "Type B": 0.99},
    SILLevel.SIL_3: {"Type A": 0.99, "Type B": 0.99},
    SILLevel.SIL_4: {"Type A": 0.99, "Type B": 0.99},
}

# Minimum hardware fault tolerance (HFT) per SIL for 1ooN architectures
HFT_REQUIREMENTS = {
    SILLevel.SIL_1: 0,
    SILLevel.SIL_2: 1,
    SILLevel.SIL_3: 1,
    SILLevel.SIL_4: 2,
}

# Diagnostic coverage thresholds
DC_THRESHOLDS = {
    SILLevel.SIL_1: 0.0,
    SILLevel.SIL_2: 0.60,
    SILLevel.SIL_3: 0.90,
    SILLevel.SIL_4: 0.99,
}


class SILVerifier:
    """IEC 61508 SIL verification engine."""

    def compute_hazard_rate(
        self,
        failure_rate: float,
        exposure_time: float,
        probability: float = 1.0,
    ) -> float:
        """Compute hazard rate (risk) as failure_rate * exposure_time * probability.

        Args:
            failure_rate: Failure rate per hour (e.g., 1e-5)
            exposure_time: Time exposed in hours
            probability: Probability of the hazardous event (0-1)

        Returns:
            Hazard rate (annual average frequency of dangerous failure)
        """
        if failure_rate < 0:
            raise ValueError("failure_rate must be non-negative")
        if exposure_time < 0:
            raise ValueError("exposure_time must be non-negative")
        if not (0.0 <= probability <= 1.0):
            raise ValueError("probability must be between 0 and 1")
        return failure_rate * exposure_time * probability

    def compute_pfd(
        self,
        diagnostic_coverage: float,
        proof_test_interval: float,
        beta_factor: float = 0.0,
    ) -> float:
        """Compute Probability of Failure on Demand (PFD).

        PFD = (1 - DC) * TI / 2 + beta_factor

        Args:
            diagnostic_coverage: Diagnostic coverage (0-1)
            proof_test_interval: Proof test interval in hours
            beta_factor: Common cause factor (0-1)

        Returns:
            PFD value
        """
        if not (0.0 <= diagnostic_coverage <= 1.0):
            raise ValueError("diagnostic_coverage must be between 0 and 1")
        if proof_test_interval <= 0:
            raise ValueError("proof_test_interval must be positive")
        if not (0.0 <= beta_factor <= 1.0):
            raise ValueError("beta_factor must be between 0 and 1")

        undetected_fraction = 1.0 - diagnostic_coverage
        pfd = undetected_fraction * proof_test_interval / 2.0 + beta_factor
        return max(0.0, pfd)

    def compute_sff(
        self,
        safe_failure_rate: float,
        total_failure_rate: float,
    ) -> float:
        """Compute Safe Failure Fraction (SFF).

        SFF = (safe failure rate + detected dangerous failure rate) / total failure rate
        Simplified: SFF = safe_failure_rate / total_failure_rate when no
        dangerous detected rate is separately given (assumes all safe are detected).

        Args:
            safe_failure_rate: Rate of safe failures per hour
            total_failure_rate: Total failure rate per hour

        Returns:
            SFF as a fraction (0-1)
        """
        if total_failure_rate <= 0:
            raise ValueError("total_failure_rate must be positive")
        if safe_failure_rate < 0:
            raise ValueError("safe_failure_rate must be non-negative")
        if safe_failure_rate > total_failure_rate:
            safe_failure_rate = total_failure_rate
        return safe_failure_rate / total_failure_rate

    def _determine_sil_from_pfd(self, pfd: float) -> Optional[SILLevel]:
        """Determine SIL level from PFD value."""
        if pfd <= 0:
            return None
        for sil in reversed(SILLevel):
            low, high = SIL_PFD_RANGES[sil]
            if pfd <= high:
                return sil
        return None

    def _determine_sil_from_pfda(self, pfda: float) -> Optional[SILLevel]:
        """Determine SIL level from PFH (average probability) value.

        Uses PFD ranges as proxy since PFH ranges are comparable in scale.
        """
        return self._determine_sil_from_pfd(pfda)

    def check_sil_architecture(
        self,
        sil_level: SILLevel,
        architecture_type: str,
        channels: int,
    ) -> Tuple[bool, List[str]]:
        """Check if the architecture meets SIL requirements.

        Args:
            sil_level: Target SIL level
            architecture_type: "Type A" (well-known) or "Type B" (complex)
            channels: Number of channels (for kooN architecture)

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        if channels < 1:
            issues.append("At least 1 channel is required")
            return False, issues

        required_hft = HFT_REQUIREMENTS.get(sil_level, 0)
        actual_hft = max(0, channels - 1)

        if actual_hft < required_hft:
            issues.append(
                f"Insufficient hardware fault tolerance: "
                f"need HFT>={required_hft}, have HFT={actual_hft} "
                f"({channels} channels)"
            )

        if architecture_type not in ("Type A", "Type B"):
            issues.append(f"Unknown architecture type: {architecture_type}")
            return False, issues

        return len(issues) == 0, issues

    def verify_sil(
        self,
        target: SILTarget,
        hazard_rate: float,
        test_coverage: float,
        diagnostic_coverage: float,
    ) -> SILVerificationResult:
        """Verify if current performance achieves the target SIL.

        Args:
            target: SIL target specification
            hazard_rate: Current system hazard rate
            test_coverage: Test coverage (0-1)
            diagnostic_coverage: Diagnostic coverage (0-1)

        Returns:
            SILVerificationResult with detailed analysis
        """
        gaps = []
        achieved_sil = None

        # Determine achieved SIL from hazard rate
        achieved_sil = self._determine_sil_from_pfd(hazard_rate)

        # Check diagnostic coverage
        required_dc = DC_THRESHOLDS.get(target.required_sil, 0.0)
        if diagnostic_coverage < required_dc:
            gaps.append(
                f"Diagnostic coverage {diagnostic_coverage:.2%} below "
                f"required {required_dc:.0%} for {target.required_sil.name}"
            )

        # Check test coverage
        if test_coverage < 0.80:
            gaps.append(
                f"Test coverage {test_coverage:.2%} below recommended 80%"
            )

        # Architecture check
        arch_valid, arch_issues = self.check_sil_architecture(
            target.required_sil, "Type B", 1
        )
        if not arch_valid:
            gaps.extend(arch_issues)

        # Determine pass/fail
        pass_fail = False
        if achieved_sil is not None:
            pass_fail = achieved_sil.value >= target.required_sil.value and len(gaps) == 0

        # Generate recommendations
        recommendations = self.recommend_measures(
            achieved_sil, target.required_sil, gaps
        )

        return SILVerificationResult(
            target=target,
            achieved_sil=achieved_sil,
            pass_fail=pass_fail,
            gaps=gaps,
            recommendations=recommendations,
        )

    def recommend_measures(
        self,
        current_sil: Optional[SILLevel],
        target_sil: SILLevel,
        gaps: List[str],
    ) -> List[str]:
        """Generate improvement recommendations to reach target SIL.

        Args:
            current_sil: Currently achieved SIL level
            target_sil: Required target SIL level
            gaps: List of identified gaps

        Returns:
            List of recommended measures
        """
        measures = []

        if current_sil is None:
            measures.append(
                "Implement basic safety instrumentation to achieve at least SIL 1"
            )
            return measures

        sil_gap = target_sil.value - current_sil.value
        if sil_gap <= 0 and not gaps:
            return measures

        if sil_gap >= 2:
            measures.append(
                "Consider full safety system redesign with higher integrity components"
            )

        if sil_gap >= 1:
            measures.append(
                "Increase diagnostic coverage through additional self-tests "
                "and built-in diagnostics"
            )
            measures.append(
                "Implement redundant channels with diverse technology "
                "(hardware fault tolerance)"
            )
            measures.append(
                "Improve proof test coverage and reduce proof test intervals"
            )

        for gap in gaps:
            if "Diagnostic coverage" in gap:
                measures.append(
                    "Enhance diagnostic coverage using advanced fault detection "
                    "techniques (watchdog timers, checksums, range checks)"
                )
            elif "Test coverage" in gap:
                measures.append(
                    "Increase test coverage through additional unit tests, "
                    "integration tests, and formal verification"
                )
            elif "hardware fault tolerance" in gap:
                measures.append(
                    "Add redundant channels or use higher-reliability components "
                    "to increase HFT"
                )

        if sil_gap > 0:
            measures.append(
                "Implement systematic capability improvements: "
                "formal methods for SIL 3+, IEC 61508-3 compliant lifecycle"
            )

        return measures
