"""Attack detection — spoofed sensor data, command injection detection."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InjectionType(Enum):
    NONE = "none"
    SQL = "sql"
    SHELL = "shell"
    BYTECODE = "bytecode"
    LFI = "lfi"
    XSS = "xss"
    COMMAND = "command"


@dataclass
class AnomalyRecord:
    timestamp: float
    sensor_id: str
    anomaly_type: str
    severity: Severity
    value: float
    expected_range: Tuple[float, float]


@dataclass
class AttackSignature:
    name: str
    pattern_description: str
    detection_threshold: float
    severity: Severity


@dataclass
class InjectionMatch:
    pattern: str
    matched_text: str
    position: int
    injection_type: InjectionType


class SensorAnomalyDetector:
    """Detect anomalies in sensor readings: range violations, rate spikes, consistency issues, jamming, spoofing."""

    def __init__(self) -> None:
        self._history: Dict[str, List[float]] = {}
        self._anomaly_log: List[AnomalyRecord] = []

    def check_range(
        self,
        sensor_id: str,
        value: float,
        valid_range: Tuple[float, float],
        timestamp: Optional[float] = None,
    ) -> Optional[AnomalyRecord]:
        lo, hi = valid_range
        ts = timestamp if timestamp is not None else time.time()
        if value < lo or value > hi:
            rec = AnomalyRecord(
                timestamp=ts,
                sensor_id=sensor_id,
                anomaly_type="range_violation",
                severity=Severity.HIGH,
                value=value,
                expected_range=valid_range,
            )
            self._anomaly_log.append(rec)
            return rec
        return None

    def check_rate(
        self,
        sensor_id: str,
        value: float,
        prev_value: float,
        max_rate: float,
        timestamp: Optional[float] = None,
    ) -> Optional[AnomalyRecord]:
        ts = timestamp if timestamp is not None else time.time()
        rate = abs(value - prev_value)
        if rate > max_rate:
            rec = AnomalyRecord(
                timestamp=ts,
                sensor_id=sensor_id,
                anomaly_type="rate_violation",
                severity=Severity.MEDIUM,
                value=rate,
                expected_range=(0.0, max_rate),
            )
            self._anomaly_log.append(rec)
            return rec
        return None

    def check_consistency(
        self,
        sensor_id: str,
        value: float,
        correlated_sensors: Dict[str, float],
        tolerance: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> Optional[AnomalyRecord]:
        ts = timestamp if timestamp is not None else time.time()
        for cid, cval in correlated_sensors.items():
            diff = abs(value - cval)
            if diff > tolerance:
                rec = AnomalyRecord(
                    timestamp=ts,
                    sensor_id=sensor_id,
                    anomaly_type="consistency_violation",
                    severity=Severity.MEDIUM,
                    value=diff,
                    expected_range=(0.0, tolerance),
                )
                self._anomaly_log.append(rec)
                return rec
        return None

    def detect_jamming(self, signal_strength: float, noise_floor: float, threshold: float = 10.0) -> bool:
        """Return True when signal-to-noise ratio is below threshold (jamming likely)."""
        if noise_floor <= 0:
            return signal_strength > 0
        snr = signal_strength / noise_floor
        return snr < threshold

    def detect_spoofing(
        self,
        sensor_readings: Dict[str, float],
        physical_model: Dict[str, Tuple[float, float]],
    ) -> float:
        """Return spoofing probability in [0, 1]."""
        if not sensor_readings or not physical_model:
            return 0.0
        violations = 0
        total = 0
        for sid, val in sensor_readings.items():
            if sid in physical_model:
                total += 1
                lo, hi = physical_model[sid]
                if val < lo or val > hi:
                    violations += 1
        if total == 0:
            return 0.0
        return violations / total

    def get_anomaly_log(self) -> List[AnomalyRecord]:
        return list(self._anomaly_log)

    def clear_log(self) -> None:
        self._anomaly_log.clear()


class CommandInjector:
    """Scan and sanitize commands for injection attacks."""

    _SQL_PATTERNS = [
        (r"(?i)(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE)\b.*)", InjectionType.SQL),
        (r"(;|\-\-|/\*|\*/)", InjectionType.SQL),
    ]
    _SHELL_PATTERNS = [
        (r"(`[^`]*`)", InjectionType.SHELL),
        (r"(\$\([^)]*\))", InjectionType.SHELL),
        (r"(;.*\b(rm|chmod|chown|wget|curl|nc|bash|sh)\b)", InjectionType.SHELL),
    ]
    _BYTECODE_PATTERNS = [
        (r"(__import__|exec|eval|compile|open|globals|locals)\s*\(", InjectionType.BYTECODE),
        (r"(\\\\x[0-9a-fA-F]{2})", InjectionType.BYTECODE),
    ]
    _LFI_PATTERNS = [
        (r"(\.\./)", InjectionType.LFI),
        (r"(/etc/passwd)", InjectionType.LFI),
        (r"(/proc/self/)", InjectionType.LFI),
    ]
    _COMMAND_PATTERNS = [
        (r"(\|)", InjectionType.COMMAND),
        (r"(>\s*/)", InjectionType.COMMAND),
        (r"(&&|\|\|)", InjectionType.COMMAND),
    ]
    _ALL_PATTERNS = (
        _SQL_PATTERNS + _SHELL_PATTERNS + _BYTECODE_PATTERNS
        + _LFI_PATTERNS + _COMMAND_PATTERNS
    )

    def __init__(self) -> None:
        self._compiled = [(re.compile(p), t) for p, t in self._ALL_PATTERNS]

    def scan_command(self, command: str, forbidden_patterns: Optional[List[str]] = None) -> List[InjectionMatch]:
        matches: List[InjectionMatch] = []
        for regex, inj_type in self._compiled:
            for m in regex.finditer(command):
                matches.append(InjectionMatch(
                    pattern=regex.pattern,
                    matched_text=m.group(),
                    position=m.start(),
                    injection_type=inj_type,
                ))
        if forbidden_patterns:
            for fp in forbidden_patterns:
                idx = command.find(fp)
                if idx >= 0:
                    matches.append(InjectionMatch(
                        pattern=fp,
                        matched_text=fp,
                        position=idx,
                        injection_type=InjectionType.COMMAND,
                    ))
        return matches

    def classify_injection(self, command: str) -> InjectionType:
        matches = self.scan_command(command)
        if not matches:
            return InjectionType.NONE
        severity_order = [
            InjectionType.SQL, InjectionType.BYTECODE,
            InjectionType.SHELL, InjectionType.LFI,
            InjectionType.COMMAND, InjectionType.XSS,
        ]
        for prio in severity_order:
            for m in matches:
                if m.injection_type == prio:
                    return prio
        return matches[0].injection_type

    def sanitize_command(self, command: str, allowed_ops: Optional[List[str]] = None) -> str:
        result = command
        # Strip any detected dangerous patterns
        patterns_to_strip = [
            (r";.*$", ""),
            (r"--.*$", ""),
            (r"/\*.*?\*/", ""),
            (r"`[^`]*`", ""),
            (r"\$\([^)]*\)", ""),
            (r"\.\./", ""),
            (r"(?i)\b(SELECT|INSERT|UPDATE|DROP|DELETE|TRUNCATE)\b", "[REDACTED]"),
        ]
        for pat, repl in patterns_to_strip:
            result = re.sub(pat, repl, result)
        # If allowed_ops provided, keep only those tokens
        if allowed_ops is not None:
            tokens = result.split()
            filtered = [t for t in tokens if any(ao in t for ao in allowed_ops)]
            result = " ".join(filtered) if filtered else ""
        return result.strip()

    def validate_command_structure(self, command: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = schema.get("required_keys", [])
        allowed_keys = schema.get("allowed_keys", None)
        max_length = schema.get("max_length", 1024)
        result: Dict[str, Any] = {"valid": True, "errors": []}

        if len(command) > max_length:
            result["valid"] = False
            result["errors"].append(f"Command length {len(command)} exceeds max {max_length}")

        # Check required keys appear in command string
        for rk in required_keys:
            if rk not in command:
                result["valid"] = False
                result["errors"].append(f"Missing required key: {rk}")

        # Check for forbidden keys
        forbidden = schema.get("forbidden_keys", [])
        for fk in forbidden:
            if fk in command:
                result["valid"] = False
                result["errors"].append(f"Forbidden key present: {fk}")

        # If allowed_keys, only those may appear
        if allowed_keys is not None:
            tokens = set(command.split())
            allowed_set = set(allowed_keys)
            for t in tokens:
                if t not in allowed_set:
                    result["valid"] = False
                    result["errors"].append(f"Disallowed token: {t}")

        return result
