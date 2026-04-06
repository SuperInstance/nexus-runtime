"""Decision logging and audit trail for NEXUS explainable AI.

Records all autonomous decisions with full context, enables querying,
anomaly detection, and statistics computation. Pure Python, no external deps.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import time
import uuid


@dataclass
class DecisionRecord:
    """Single autonomous decision record."""
    timestamp: float
    decision_type: str
    input_state: Dict[str, Any]
    output_action: Dict[str, Any]
    confidence: float
    reasoning: str
    model_version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp <= 0:
            self.timestamp = time.time()
        self.confidence = max(0.0, min(1.0, self.confidence))


class DecisionLog:
    """Decision log with querying, anomaly detection, and export capabilities."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._records: List[DecisionRecord] = []
        self._entity_chains: Dict[str, List[DecisionRecord]] = defaultdict(list)

    def log_decision(self, record: DecisionRecord) -> None:
        """Log a decision record. Evicts oldest when max_size exceeded."""
        self._records.append(record)
        # Track entity chains if entity_id in metadata
        entity_id = record.metadata.get("entity_id")
        if entity_id:
            self._entity_chains[entity_id].append(record)
        # Evict oldest if needed
        while len(self._records) > self.max_size:
            removed = self._records.pop(0)
            rid = removed.metadata.get("entity_id")
            if rid and rid in self._entity_chains:
                self._entity_chains[rid] = [
                    r for r in self._entity_chains[rid] if r.timestamp != removed.timestamp
                ]

    def query_by_type(
        self,
        decision_type: str,
        limit: Optional[int] = None,
    ) -> List[DecisionRecord]:
        """Query decisions by type, most recent first."""
        matching = [r for r in self._records if r.decision_type == decision_type]
        matching.sort(key=lambda r: r.timestamp, reverse=True)
        if limit is not None:
            matching = matching[:limit]
        return matching

    def query_by_time(
        self,
        start: float,
        end: float,
    ) -> List[DecisionRecord]:
        """Query decisions within a time range."""
        return [r for r in self._records if start <= r.timestamp <= end]

    def get_decision_chain(
        self,
        entity_id: str,
    ) -> List[DecisionRecord]:
        """Get chronological decision chain for an entity."""
        chain = self._entity_chains.get(entity_id, [])
        return sorted(chain, key=lambda r: r.timestamp)

    def compute_decision_frequency(
        self,
        decision_type: str,
    ) -> float:
        """Compute decisions per second for a given type over the log duration."""
        records = [r for r in self._records if r.decision_type == decision_type]
        if not records:
            return 0.0
        timestamps = [r.timestamp for r in records]
        duration = max(timestamps) - min(timestamps)
        if duration <= 0:
            return float(len(records))
        return len(records) / duration

    def detect_anomalies(
        self,
        decisions: Optional[List[DecisionRecord]] = None,
    ) -> List[DecisionRecord]:
        """Detect anomalous decisions based on confidence and frequency heuristics."""
        if decisions is None:
            decisions = self._records
        if not decisions:
            return []

        # Compute mean and std of confidence
        confidences = [r.confidence for r in decisions]
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        std_conf = math.sqrt(variance) if variance > 0 else 0.01

        # Compute type frequencies
        type_counts = Counter(r.decision_type for r in decisions)
        mean_freq = sum(type_counts.values()) / len(type_counts) if type_counts else 1
        freq_threshold = mean_freq * 3  # 3x average frequency

        anomalies = []
        for record in decisions:
            is_anomaly = False
            # Low confidence anomaly (< 2 std below mean)
            if record.confidence < mean_conf - 2 * std_conf:
                is_anomaly = True
            # High frequency anomaly
            if type_counts.get(record.decision_type, 0) > freq_threshold:
                is_anomaly = True
            if is_anomaly:
                anomalies.append(record)
        return anomalies

    def export_log(self, format: str = "json") -> str:
        """Export log in specified format ('json' or 'text')."""
        if format == "json":
            entries = []
            for r in self._records:
                entries.append({
                    "timestamp": r.timestamp,
                    "decision_type": r.decision_type,
                    "input_state": r.input_state,
                    "output_action": r.output_action,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "model_version": r.model_version,
                    "metadata": r.metadata,
                })
            return json.dumps(entries, indent=2, default=str)
        elif format == "text":
            lines = []
            for r in self._records:
                lines.append(
                    f"[{r.timestamp:.3f}] {r.decision_type} "
                    f"(conf={r.confidence:.3f}, v={r.model_version}): {r.reasoning}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def compute_statistics(
        self,
        period: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute summary statistics, optionally filtered to a time period."""
        now = time.time()
        if period is not None:
            records = [r for r in self._records if now - r.timestamp <= period]
        else:
            records = self._records

        if not records:
            return {
                "total_decisions": 0,
                "avg_confidence": 0.0,
                "decision_types": {},
                "time_range": None,
                "model_versions": {},
            }

        confidences = [r.confidence for r in records]
        type_counts = Counter(r.decision_type for r in records)
        version_counts = Counter(r.model_version for r in records)
        timestamps = [r.timestamp for r in records]

        return {
            "total_decisions": len(records),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "std_confidence": math.sqrt(
                sum((c - sum(confidences) / len(confidences)) ** 2 for c in confidences) / len(confidences)
            ) if len(confidences) > 1 else 0.0,
            "decision_types": dict(type_counts),
            "time_range": (min(timestamps), max(timestamps)),
            "model_versions": dict(version_counts),
        }

    @property
    def size(self) -> int:
        """Number of records in the log."""
        return len(self._records)

    @property
    def records(self) -> List[DecisionRecord]:
        """All records (read-only copy)."""
        return list(self._records)
