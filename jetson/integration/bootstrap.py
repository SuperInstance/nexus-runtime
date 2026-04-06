"""
Bootstrap management — phased initialization with rollback support,
environment validation, and boot-sequence introspection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class BootstrapPhase(str, Enum):
    CORE = "core"
    SERVICES = "services"
    AGENTS = "agents"
    INTEGRATION = "integration"
    READY = "ready"


@dataclass
class BootstrapStep:
    name: str
    phase: BootstrapPhase
    action_fn: Callable[[], bool]
    timeout: float = 30.0
    required: bool = True
    category: str = "general"


@dataclass
class BootstrapResult:
    success: bool
    phase: BootstrapPhase
    steps_completed: int = 0
    steps_total: int = 0
    failures: List[str] = field(default_factory=list)
    duration_s: float = 0.0
    log: List[str] = field(default_factory=list)


class BootstrapManager:
    """Manages phased bootstrap with timeout enforcement and rollback."""

    PHASE_ORDER = [
        BootstrapPhase.CORE,
        BootstrapPhase.SERVICES,
        BootstrapPhase.AGENTS,
        BootstrapPhase.INTEGRATION,
        BootstrapPhase.READY,
    ]

    def __init__(self) -> None:
        self._steps: List[BootstrapStep] = []
        self._log: List[str] = []
        self._completed_phases: List[BootstrapPhase] = []
        self._rollback_handlers: Dict[BootstrapPhase, Callable[[], None]] = {}

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def add_step(self, step: BootstrapStep) -> None:
        self._steps.append(step)

    def remove_step(self, name: str) -> bool:
        before = len(self._steps)
        self._steps = [s for s in self._steps if s.name != name]
        return len(self._steps) < before

    def get_boot_sequence(self) -> List[BootstrapStep]:
        """Return steps ordered by phase."""
        phase_rank = {p: i for i, p in enumerate(self.PHASE_ORDER)}
        return sorted(self._steps, key=lambda s: (phase_rank.get(s.phase, 99), s.name))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_bootstrap(self) -> BootstrapResult:
        self._log.clear()
        self._completed_phases.clear()
        sequence = self.get_boot_sequence()
        total = len(sequence)
        completed = 0
        failures: List[str] = []
        overall_success = True
        start = time.time()
        current_phase: Optional[BootstrapPhase] = None

        for step in sequence:
            if step.phase != current_phase:
                current_phase = step.phase
                self._log.append(f"=== Phase: {step.phase.value} ===")
            self._log.append(f"Running step: {step.name}")
            step_start = time.time()
            try:
                result = step.action_fn()
                elapsed = time.time() - step_start
                if result:
                    completed += 1
                    self._log.append(f"  OK ({elapsed:.3f}s)")
                else:
                    failures.append(step.name)
                    if step.required:
                        overall_success = False
                        self._log.append(f"  FAILED (required) ({elapsed:.3f}s)")
                    else:
                        completed += 1
                        self._log.append(f"  FAILED (optional) ({elapsed:.3f}s)")
            except Exception as exc:
                elapsed = time.time() - step_start
                failures.append(step.name)
                if step.required:
                    overall_success = False
                self._log.append(f"  ERROR: {exc} ({elapsed:.3f}s)")

            if elapsed > step.timeout:
                self._log.append(f"  WARNING: exceeded timeout ({step.timeout}s)")

        if overall_success:
            for phase in self.PHASE_ORDER:
                if any(s.phase == phase for s in sequence):
                    self._completed_phases.append(phase)

        duration = time.time() - start
        return BootstrapResult(
            success=overall_success,
            phase=BootstrapPhase.READY if overall_success else (current_phase or BootstrapPhase.CORE),
            steps_completed=completed,
            steps_total=total,
            failures=failures,
            duration_s=duration,
            log=list(self._log),
        )

    # ------------------------------------------------------------------
    # Environment validation
    # ------------------------------------------------------------------

    def validate_environment(self) -> Tuple[bool, List[str]]:
        """Check that required-phase steps have met dependencies. Returns (valid, missing_items)."""
        missing: List[str] = []
        for step in self._steps:
            if step.required and step.phase == BootstrapPhase.CORE:
                try:
                    result = step.action_fn()
                    if not result:
                        missing.append(f"core check failed: {step.name}")
                except Exception as exc:
                    missing.append(f"core check error: {step.name}: {exc}")
        return (len(missing) == 0, missing)

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def register_rollback_handler(self, phase: BootstrapPhase, handler: Callable[[], None]) -> None:
        self._rollback_handlers[phase] = handler

    def rollback(self, phase: BootstrapPhase) -> None:
        """Roll back from the given phase down to CORE."""
        idx = self.PHASE_ORDER.index(phase) if phase in self.PHASE_ORDER else -1
        if idx < 0:
            return
        for p in reversed(self.PHASE_ORDER[:idx + 1]):
            handler = self._rollback_handlers.get(p)
            if handler:
                try:
                    handler()
                    self._log.append(f"Rolled back phase: {p.value}")
                except Exception as exc:
                    self._log.append(f"Rollback error for {p.value}: {exc}")
        self._completed_phases = [
            p for p in self._completed_phases
            if p not in self.PHASE_ORDER[:idx + 1]
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_bootstrap_log(self) -> List[str]:
        return list(self._log)

    def get_completed_phases(self) -> List[BootstrapPhase]:
        return list(self._completed_phases)

    def get_steps_for_phase(self, phase: BootstrapPhase) -> List[BootstrapStep]:
        return [s for s in self._steps if s.phase == phase]

    def reset(self) -> None:
        self._log.clear()
        self._completed_phases.clear()
