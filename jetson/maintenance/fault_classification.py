"""
Anomaly-based fault classification for marine robotics.

Signature matching, severity computation, action recommendation,
and online fault-library learning. Pure Python – no external deps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FaultSignature:
    """Template signature for a known fault type."""
    fault_type: str
    severity: str = "medium"  # low, medium, high, critical
    sensor_patterns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    description: str = ""


@dataclass
class FaultReport:
    """Output of the fault classification pipeline."""
    detected_fault: Optional[str] = None
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    recommended_action: str = ""


class FaultClassifier:
    """Classify equipment faults from anomaly readings."""

    # ── public API ──────────────────────────────────────────────

    def classify(
        self,
        anomaly_readings: Dict[str, List[float]],
        fault_library: List[FaultSignature],
    ) -> FaultReport:
        """Match anomaly readings against the fault library.

        Parameters
        ----------
        anomaly_readings : dict
            ``{sensor_id: [values]}`` — recent anomalous readings.
        fault_library : list[FaultSignature]
            Known fault templates.
        """
        if not anomaly_readings or not fault_library:
            return FaultReport(confidence=0.0)

        # Compute per-sensor summary stats
        summaries: Dict[str, Dict[str, float]] = {}
        for sid, vals in anomaly_readings.items():
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) or 1e-9
            summaries[sid] = {"mean": mean, "std": std, "min": min(vals), "max": max(vals)}

        best_match: Optional[FaultSignature] = None
        best_score = 0.0
        all_evidence: List[str] = []

        for sig in fault_library:
            score = self.match_signature(summaries, sig.sensor_patterns)
            all_evidence.append(
                f"{sig.fault_type}: similarity={score:.4f}"
            )
            if score > best_score:
                best_score = score
                best_match = sig

        if best_match is None:
            return FaultReport(confidence=0.0, evidence=all_evidence)

        severity = best_match.severity
        action = self.recommend_action(best_match.fault_type, severity)

        return FaultReport(
            detected_fault=best_match.fault_type,
            confidence=best_score,
            evidence=all_evidence,
            recommended_action=action,
        )

    def match_signature(
        self,
        readings_summary: Dict[str, Dict[str, float]],
        signature_patterns: Dict[str, Dict[str, float]],
    ) -> float:
        """Cosine-similarity between observed summary and signature pattern."""
        # Flatten both to vectors
        def flatten(d: Dict[str, Dict[str, float]]) -> List[float]:
            vals: List[float] = []
            for sid in sorted(d.keys()):
                for key in sorted(d[sid].keys()):
                    vals.append(d[sid][key])
            return vals

        obs = flatten(readings_summary)
        ref = flatten(signature_patterns)

        if not obs or not ref:
            return 0.0

        # Pad shorter vector with zeros
        max_len = max(len(obs), len(ref))
        obs += [0.0] * (max_len - len(obs))
        ref += [0.0] * (max_len - len(ref))

        dot = sum(a * b for a, b in zip(obs, ref))
        mag_obs = math.sqrt(sum(a * a for a in obs)) or 1e-9
        mag_ref = math.sqrt(sum(b * b for b in ref)) or 1e-9

        return max(0.0, min(1.0, dot / (mag_obs * mag_ref)))

    def compute_severity(
        self,
        fault_type: str,
        equipment_health: float,
    ) -> str:
        """Return severity level considering both fault type and current health.

        Parameters
        ----------
        fault_type : str
            E.g. ``"bearing_wear"``, ``"motor_overheat"``, etc.
        equipment_health : float
            Current health score (0–1).
        """
        # Critical faults regardless of health
        critical_faults = {"motor_overheat", "hull_breach", "power_failure", "propeller_damage"}
        high_faults = {"bearing_wear", "seal_leak", "corrosion"}

        if fault_type in critical_faults:
            return "critical"
        if fault_type in high_faults:
            if equipment_health < 0.4:
                return "critical"
            if equipment_health < 0.7:
                return "high"
            return "medium"

        # Default fault
        if equipment_health < 0.3:
            return "high"
        if equipment_health < 0.6:
            return "medium"
        return "low"

    def recommend_action(self, fault: str, severity: str) -> str:
        """Return a human-readable action recommendation."""
        actions: Dict[str, Dict[str, str]] = {
            "critical": {
                "default": "Immediate shutdown and emergency repair required.",
            },
            "high": {
                "default": "Schedule repair within 12 hours; restrict operations.",
            },
            "medium": {
                "default": "Plan maintenance within 72 hours; monitor closely.",
            },
            "low": {
                "default": "Log fault; include in next scheduled maintenance.",
            },
        }

        severity_actions = actions.get(severity, actions["low"])

        # Override for known critical faults
        if fault in ("hull_breach", "power_failure"):
            return "EMERGENCY: Immediate surface and secure vessel."

        return severity_actions.get("default", "Monitor and reassess.")

    def learn_new_fault(
        self,
        readings: Dict[str, List[float]],
        label: str,
    ) -> FaultSignature:
        """Learn a new fault signature from labelled anomaly readings."""
        patterns: Dict[str, Dict[str, float]] = {}
        for sid, vals in readings.items():
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) or 1e-9
            patterns[sid] = {"mean": mean, "std": std, "min": min(vals), "max": max(vals)}

        # Infer severity from magnitude of deviations
        avg_std = (
            sum(p["std"] for p in patterns.values()) / len(patterns)
            if patterns else 0.0
        )
        if avg_std > 15.0:
            severity = "high"
        elif avg_std > 5.0:
            severity = "medium"
        else:
            severity = "low"

        return FaultSignature(
            fault_type=label,
            severity=severity,
            sensor_patterns=patterns,
            description=f"Auto-learned fault signature for '{label}'",
        )
