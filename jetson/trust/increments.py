"""NEXUS Trust Engine - INCREMENTS trust algorithm.

12 parameters. 6 autonomy levels. 25:1 loss-to-gain ratio.
Gain time constant: ~658 windows (27.4 days).
Loss time constant: ~29 windows (1.2 days).

Three-branch delta computation:
  Branch 1 — Net Positive: trust increases
  Branch 2 — Penalty: trust decreases proportionally to severity
  Branch 3 — Decay: trust decays toward floor when idle
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TrustParams:
    """INCREMENTS algorithm parameters."""

    alpha_gain: float = 0.002
    alpha_loss: float = 0.05
    alpha_decay: float = 0.0001
    t_floor: float = 0.2
    quality_cap: int = 10
    evaluation_window_hours: float = 1.0
    severity_exponent: float = 1.0
    streak_bonus: float = 0.00005
    min_events_for_gain: int = 1
    n_penalty_slope: float = 0.1
    reset_grace_hours: float = 24.0
    promotion_cooldown_hours: float = 72.0


@dataclass
class TrustEvent:
    """A single trust-related event."""

    event_type: str
    quality: float = 0.0
    severity: float = 0.0
    timestamp: float = 0.0
    subsystem: str = "navigation"
    is_bad: bool = False

    @classmethod
    def good(cls, event_type: str, quality: float, timestamp: float = 0.0,
             subsystem: str = "navigation") -> TrustEvent:
        return cls(event_type=event_type, quality=quality, severity=0.0,
                   timestamp=timestamp, subsystem=subsystem, is_bad=False)

    @classmethod
    def bad(cls, event_type: str, severity: float, timestamp: float = 0.0,
            subsystem: str = "navigation") -> TrustEvent:
        return cls(event_type=event_type, quality=0.0, severity=severity,
                   timestamp=timestamp, subsystem=subsystem, is_bad=True)


@dataclass
class SubsystemTrust:
    """Trust state for a single subsystem."""

    subsystem: str
    trust_score: float = 0.0
    autonomy_level: int = 0
    consecutive_clean_windows: int = 0
    total_windows: int = 0
    clean_windows: int = 0
    total_observation_hours: float = 0.0
    last_promotion_time: float = 0.0
    last_reset_time: float = 0.0
    max_severity_history: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class TrustUpdateResult:
    """Result of a trust evaluation window."""

    subsystem: str
    old_score: float
    new_score: float
    delta: float
    branch: str
    n_good: int = 0
    n_bad: int = 0
    max_severity: float = 0.0
    old_level: int = 0
    new_level: int = 0
    reason: str = ""


SUBSYSTEMS = ["steering", "engine", "navigation", "payload", "communication"]


class IncrementTrustEngine:
    """INCREMENTS trust algorithm implementation."""

    def __init__(self, params: TrustParams | None = None) -> None:
        self.params = params or TrustParams()
        self.subsystems: dict[str, SubsystemTrust] = {}

    def register_subsystem(self, name: str) -> None:
        self.subsystems[name] = SubsystemTrust(subsystem=name)

    def register_all_subsystems(self) -> None:
        for name in SUBSYSTEMS:
            self.register_subsystem(name)

    def reset_subsystem(self, subsystem: str, reset_type: str = "full") -> None:
        st = self.subsystems.get(subsystem)
        if st is None:
            return
        now = time.time()
        if reset_type == "full" and st.last_reset_time > 0:
            hours_since_reset = (now - st.last_reset_time) / 3600.0
            if hours_since_reset < self.params.reset_grace_hours:
                return
        st.trust_score = 0.0
        st.autonomy_level = 0
        st.consecutive_clean_windows = 0
        st.total_windows = 0
        st.clean_windows = 0
        st.total_observation_hours = 0.0
        st.last_promotion_time = 0.0
        st.last_reset_time = now
        st.max_severity_history.clear()

    def compute_delta(
        self,
        prev_trust: float,
        events: list[TrustEvent],
        consecutive_clean: int,
        is_agent_code: bool = False,
    ) -> tuple[float, str]:
        """Compute trust delta using three-branch algorithm."""
        p = self.params
        good_events = [e for e in events if not e.is_bad and e.quality > 0.0]
        bad_events = [e for e in events if e.is_bad]
        n_good = len(good_events)
        n_bad = len(bad_events)

        # Branch 1: Net Positive
        if n_bad == 0 and n_good >= p.min_events_for_gain:
            avg_quality = sum(e.quality for e in good_events) / n_good
            capped_n_good = min(n_good, p.quality_cap)
            alpha_effective = p.alpha_gain * (0.5 if is_agent_code else 1.0)
            delta_T = (
                alpha_effective * (1.0 - prev_trust) * avg_quality
                * (capped_n_good / p.quality_cap)
                + p.streak_bonus * min(consecutive_clean, 24)
            )
            return delta_T, "gain"

        # Branch 2: Penalty
        elif n_bad > 0:
            max_severity = max(e.severity for e in bad_events)
            n_penalty = 1.0 + p.n_penalty_slope * (n_bad - 1)
            delta_T = (
                -p.alpha_loss * prev_trust
                * (max_severity ** p.severity_exponent)
                * n_penalty
            )
            return delta_T, "penalty"

        # Branch 3: Decay (no qualifying events)
        else:
            delta_T = -p.alpha_decay * (prev_trust - p.t_floor)
            return delta_T, "decay"

    def evaluate_window(
        self,
        subsystem: str,
        events: list[TrustEvent],
        is_agent_code: bool = False,
    ) -> TrustUpdateResult:
        st = self.subsystems.get(subsystem)
        if st is None:
            st = SubsystemTrust(subsystem=subsystem)
            self.subsystems[subsystem] = st

        old_score = st.trust_score
        old_level = st.autonomy_level

        delta, branch = self.compute_delta(
            old_score, events, st.consecutive_clean_windows, is_agent_code
        )

        new_score = old_score + delta

        # Clamp: floor applies to penalty and decay; gain starts from 0
        if branch == "gain":
            new_score = max(new_score, 0.0)
        else:
            new_score = max(new_score, self.params.t_floor)
        new_score = min(new_score, 1.0)

        st.trust_score = new_score

        bad_events = [e for e in events if e.is_bad]
        good_events = [e for e in events if not e.is_bad and e.quality > 0.0]
        n_good = len(good_events)
        n_bad = len(bad_events)
        max_severity = max((e.severity for e in bad_events), default=0.0)

        if bad_events:
            now = time.time()
            st.max_severity_history.append((max_severity, now))
            if len(st.max_severity_history) > 1000:
                st.max_severity_history = st.max_severity_history[-500:]

        st.total_windows += 1
        st.total_observation_hours += self.params.evaluation_window_hours

        if n_bad == 0:
            st.clean_windows += 1
            st.consecutive_clean_windows += 1
        else:
            st.consecutive_clean_windows = 0

        # Demotion (immediate)
        demotion_delta = 0
        if n_bad > 0:
            if max_severity >= 1.0:
                demotion_delta = st.autonomy_level
                st.autonomy_level = 0
            elif max_severity >= 0.8:
                demotion_delta = min(2, st.autonomy_level)
                st.autonomy_level = max(0, st.autonomy_level - 2)
            else:
                demotion_delta = 1
                st.autonomy_level = max(0, st.autonomy_level - 1)

        # Promotion
        new_level = st.autonomy_level
        if n_bad == 0 and demotion_delta == 0:
            now = time.time()
            promotion_candidate = self._check_promotion(st, now)
            if promotion_candidate > st.autonomy_level:
                st.autonomy_level = promotion_candidate
                st.last_promotion_time = now
                new_level = promotion_candidate

        reason = f"Branch={branch}, delta={delta:+.6f}"
        if demotion_delta > 0:
            reason += f", demoted_by={demotion_delta}"

        return TrustUpdateResult(
            subsystem=subsystem,
            old_score=old_score,
            new_score=new_score,
            delta=delta,
            branch=branch,
            n_good=n_good,
            n_bad=n_bad,
            max_severity=max_severity,
            old_level=old_level,
            new_level=new_level,
            reason=reason,
        )

    def _check_promotion(self, st: SubsystemTrust, now: float) -> int:
        from trust.levels import AUTONOMY_LEVELS
        current = st.autonomy_level
        highest = current
        for lvl in range(current + 1, 6):
            defn = AUTONOMY_LEVELS[lvl]
            if defn.trust_threshold is None:
                break
            if (st.trust_score >= defn.trust_threshold
                    and st.total_observation_hours >= defn.min_observation_hours
                    and st.clean_windows >= defn.min_clean_windows):
                if st.last_promotion_time > 0:
                    hours_since = (now - st.last_promotion_time) / 3600.0
                    if hours_since < self.params.promotion_cooldown_hours:
                        break
                highest = lvl
            else:
                break
        return highest

    def record_event(self, event: TrustEvent) -> None:
        self.evaluate_window(event.subsystem, [event])

    def get_trust_score(self, subsystem: str) -> float:
        st = self.subsystems.get(subsystem)
        return st.trust_score if st else 0.0

    def get_trust(self, subsystem: str) -> float:
        return self.get_trust_score(subsystem)

    def get_autonomy_level(self, subsystem: str) -> int:
        st = self.subsystems.get(subsystem)
        return st.autonomy_level if st else 0

    def should_allow_deploy(self, subsystem: str, trust_required: float) -> bool:
        return self.get_trust_score(subsystem) >= trust_required

    def get_all_scores(self) -> dict[str, SubsystemTrust]:
        return dict(self.subsystems)

    def get_subsystem(self, subsystem: str) -> SubsystemTrust | None:
        return self.subsystems.get(subsystem)


# Backward compatibility
IncrementsParams = TrustParams
IncrementsEngine = IncrementTrustEngine
