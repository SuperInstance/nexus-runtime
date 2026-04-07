"""NEXUS Edge Heartbeat — Main Heartbeat Loop.

5-phase heartbeat cycle for NEXUS edge vessel.
    PERCEIVE -> THINK -> ACT -> REMEMBER -> NOTIFY

Each cycle processes one mission from .agent/next.
Connects NEXUS edge runtime to git-agent coordination.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any

from agent.edge_heartbeat.config import HeartbeatConfig, load_config
from agent.edge_heartbeat.mission_runner import (
    Mission,
    MissionRunner,
    MissionStatus,
    MissionResult,
    MissionType,
    append_done,
    pop_next_mission,
    read_next_queue,
)
from agent.edge_heartbeat.vessel_state import VesselState, VesselStateManager


class HeartbeatPhase(str, Enum):
    """The 5 phases of a heartbeat cycle."""

    PERCEIVE = "PERCEIVE"
    THINK = "THINK"
    ACT = "ACT"
    REMEMBER = "REMEMBER"
    NOTIFY = "NOTIFY"


@dataclass
class PhaseResult:
    """Result of a single heartbeat phase."""

    phase: HeartbeatPhase
    success: bool = True
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class HeartbeatResult:
    """Result of a complete heartbeat cycle."""

    cycle_number: int = 0
    vessel_id: str = ""
    phases: list[PhaseResult] = field(default_factory=list)
    mission_executed: bool = False
    mission_type: str | None = None
    total_duration: float = 0.0
    error: str = ""

    @property
    def success(self) -> bool:
        return all(p.success for p in self.phases)

    @property
    def failed_phase(self) -> str | None:
        for p in self.phases:
            if not p.success:
                return p.phase.value
        return None

    def phase_result(self, phase: HeartbeatPhase) -> PhaseResult | None:
        for p in self.phases:
            if p.phase == phase:
                return p
        return None


logger = logging.getLogger("nexus.heartbeat")


class EdgeHeartbeat:
    """5-phase heartbeat cycle for NEXUS edge vessel.

    PERCEIVE: Read vessel state (sensors, trust, pending missions)
    THINK: Process current mission (determine action needed)
    ACT: Execute action (deploy bytecode, record telemetry, etc.)
    REMEMBER: Commit results to .agent/done
    NOTIFY: Update vessel status

    Usage:
        hb = EdgeHeartbeat("/path/to/config.json")
        result = hb.run_once()
        # or
        hb.run_forever(interval_seconds=300)
    """

    def __init__(self, config_path: str | None = None, config: HeartbeatConfig | None = None) -> None:
        """Initialize the heartbeat system.

        Args:
            config_path: Path to JSON configuration file.
            config: Pre-loaded HeartbeatConfig. If provided, config_path is ignored.
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            self.config = HeartbeatConfig()

        # Core subsystems
        self.state_manager = VesselStateManager(vessel_id=self.config.vessel_id)
        self.mission_runner = MissionRunner(repo_path=self.config.repo_path)

        # Paths
        self.repo_path = Path(self.config.repo_path)
        self.agent_dir = self.repo_path / self.config.agent_dir_name

        # State
        self._cycle_count = 0
        self._running = False
        self._logger = logging.getLogger(f"nexus.heartbeat.{self.config.vessel_id}")

        # Ensure .agent directory structure
        self._ensure_agent_dirs()

    def _ensure_agent_dirs(self) -> None:
        """Create .agent directory structure if it doesn't exist."""
        self.agent_dir.mkdir(parents=True, exist_ok=True)

        next_file = self.agent_dir / self.config.next_file_name
        if not next_file.exists():
            next_file.touch()

        done_file = self.agent_dir / self.config.done_file_name
        if not done_file.exists():
            done_file.touch()

    def run_once(self) -> HeartbeatResult:
        """Execute one complete heartbeat cycle.

        5 phases:
            1. PERCEIVE: Read vessel state (sensors, trust, pending missions)
            2. THINK: Process current mission (determine action needed)
            3. ACT: Execute action (deploy bytecode, record telemetry, etc.)
            4. REMEMBER: Commit results to .agent/done
            5. NOTIFY: Update vessel status

        Returns:
            HeartbeatResult with what was done in each phase.
        """
        self._cycle_count += 1
        cycle_start = time.time()

        result = HeartbeatResult(
            cycle_number=self._cycle_count,
            vessel_id=self.config.vessel_id,
        )

        # Phase 1: PERCEIVE
        perceive_result = self._phase_perceive()
        result.phases.append(perceive_result)

        if not perceive_result.success:
            result.error = f"PERCEIVE failed: {perceive_result.message}"
            result.total_duration = time.time() - cycle_start
            # Fill remaining phases as skipped
            for phase in [HeartbeatPhase.THINK, HeartbeatPhase.ACT,
                          HeartbeatPhase.REMEMBER, HeartbeatPhase.NOTIFY]:
                result.phases.append(PhaseResult(
                    phase=phase, success=True,
                    message="Skipped due to earlier failure",
                ))
            return result

        # Phase 2: THINK
        think_result = self._phase_think(perceive_result.data)
        result.phases.append(think_result)

        if not think_result.success:
            result.error = f"THINK failed: {think_result.message}"
            result.total_duration = time.time() - cycle_start
            for phase in [HeartbeatPhase.ACT, HeartbeatPhase.REMEMBER, HeartbeatPhase.NOTIFY]:
                result.phases.append(PhaseResult(
                    phase=phase, success=True,
                    message="Skipped due to earlier failure",
                ))
            return result

        # Phase 3: ACT
        act_result = self._phase_act(think_result.data)
        result.phases.append(act_result)

        # Phase 4: REMEMBER
        remember_result = self._phase_remember(think_result.data, act_result)
        result.phases.append(remember_result)

        # Phase 5: NOTIFY
        notify_result = self._phase_notify()
        result.phases.append(notify_result)

        result.mission_executed = act_result.data.get("executed", False)
        result.mission_type = act_result.data.get("mission_type")
        result.total_duration = time.time() - cycle_start

        return result

    def _phase_perceive(self) -> PhaseResult:
        """PERCEIVE: Read vessel state.

        Gathers sensor readings, trust scores, and pending missions
        from .agent/next. Updates VesselStateManager.
        """
        start = time.time()
        try:
            # Read pending missions
            pending = read_next_queue(self.agent_dir)

            # Update vessel state
            self.state_manager.set_pending_missions(pending)
            self.state_manager.record_heartbeat()

            # Build perceive data
            perceive_data: dict[str, Any] = {
                "pending_missions": pending,
                "vessel_state": self.state_manager.get_status_report(),
            }

            # Read current mission if available
            if pending:
                perceive_data["current_mission_line"] = pending[0]

            duration = time.time() - start
            self._logger.debug("PERCEIVE: %d pending missions in %.3fs", len(pending), duration)

            return PhaseResult(
                phase=HeartbeatPhase.PERCEIVE,
                success=True,
                message=f"Perceived {len(pending)} pending missions",
                data=perceive_data,
                duration_seconds=duration,
            )
        except Exception as exc:
            self.state_manager.record_error()
            return PhaseResult(
                phase=HeartbeatPhase.PERCEIVE,
                success=False,
                message=str(exc),
                duration_seconds=time.time() - start,
            )

    def _phase_think(self, perceive_data: dict[str, Any]) -> PhaseResult:
        """THINK: Process current mission and determine action.

        Parses the top mission from the queue, validates it,
        and prepares execution parameters.
        """
        start = time.time()
        try:
            pending = perceive_data.get("pending_missions", [])
            current_line = perceive_data.get("current_mission_line")

            think_data: dict[str, Any] = {
                "action": "idle",
                "mission": None,
                "should_execute": False,
            }

            if current_line:
                # Parse the mission
                mission = self.mission_runner.load_mission(current_line)
                think_data["mission"] = mission
                think_data["should_execute"] = True
                think_data["action"] = f"execute_{mission.mission_type.value}"

                self.state_manager.set_current_mission(mission.name)

                message = f"Will execute: {mission.mission_type.value} - {mission.name}"
            else:
                message = "No pending missions, idle cycle"
                self.state_manager.set_current_mission(None)

            duration = time.time() - start
            self._logger.debug("THINK: %s (%.3fs)", message, duration)

            return PhaseResult(
                phase=HeartbeatPhase.THINK,
                success=True,
                message=message,
                data=think_data,
                duration_seconds=duration,
            )
        except Exception as exc:
            # Pop unparseable mission from queue to avoid blocking
            if current_line:
                popped = pop_next_mission(self.agent_dir)
                if popped:
                    # Record the bad mission in done log
                    bad_mission = Mission(
                        raw_line=popped,
                        mission_type=MissionType.REPORT,  # fallback
                        params={},
                        description="",
                    )
                    bad_result = MissionResult(
                        mission=bad_mission,
                        status=MissionStatus.FAILED,
                        error=str(exc),
                    )
                    try:
                        append_done(self.agent_dir, popped, bad_result)
                    except Exception as e:
                        logger.warning("Failed to write mission result to done queue: %s", e)
            self.state_manager.record_error()
            # Don't fail the phase — skip bad mission and continue idle
            duration = time.time() - start
            return PhaseResult(
                phase=HeartbeatPhase.THINK,
                success=True,
                message=f"Skipped unparseable mission: {exc}",
                data={
                    "action": "idle",
                    "mission": None,
                    "should_execute": False,
                },
                duration_seconds=duration,
            )

    def _phase_act(self, think_data: dict[str, Any]) -> PhaseResult:
        """ACT: Execute the mission.

        Runs the mission executor and collects results.
        """
        start = time.time()
        act_data: dict[str, Any] = {
            "executed": False,
            "mission_type": None,
            "mission_result": None,
        }

        mission = think_data.get("mission")

        if mission is None or not think_data.get("should_execute", False):
            duration = time.time() - start
            return PhaseResult(
                phase=HeartbeatPhase.ACT,
                success=True,
                message="No mission to execute",
                data=act_data,
                duration_seconds=duration,
            )

        try:
            # Pop mission from queue before execution
            mission_line = pop_next_mission(self.agent_dir)

            if mission_line is None:
                return PhaseResult(
                    phase=HeartbeatPhase.ACT,
                    success=True,
                    message="Queue empty (race condition)",
                    data=act_data,
                    duration_seconds=time.time() - start,
                )

            # Execute
            result = self.mission_runner.execute_mission(mission)

            act_data["executed"] = True
            act_data["mission_type"] = mission.mission_type.value
            act_data["mission_result"] = result
            act_data["original_line"] = mission_line

            # Record in state
            self.state_manager.record_mission_complete(success=result.status == MissionStatus.SUCCESS)

            if result.status == MissionStatus.FAILED:
                self.state_manager.record_error()

            duration = time.time() - start
            status_msg = result.status.value

            self._logger.debug(
                "ACT: %s -> %s (%.3fs)", mission.mission_type.value, status_msg, duration
            )

            return PhaseResult(
                phase=HeartbeatPhase.ACT,
                success=result.status != MissionStatus.FAILED,
                message=f"Mission {status_msg}: {mission.name}",
                data=act_data,
                duration_seconds=duration,
            )
        except Exception as exc:
            self.state_manager.record_error()
            return PhaseResult(
                phase=HeartbeatPhase.ACT,
                success=False,
                message=f"Execution error: {exc}",
                data=act_data,
                duration_seconds=time.time() - start,
            )

    def _phase_remember(self, think_data: dict[str, Any], act_result: PhaseResult) -> PhaseResult:
        """REMEMBER: Commit results to .agent/done.

        Appends the mission result to the done log with timestamp.
        """
        start = time.time()
        try:
            act_data = act_result.data
            mission = think_data.get("mission")
            original_line = act_data.get("original_line")
            mission_result = act_data.get("mission_result")

            if mission and original_line and mission_result:
                append_done(self.agent_dir, original_line, mission_result)
                message = f"Recorded mission to done: {mission.name}"
            else:
                message = "No mission result to record"

            duration = time.time() - start
            self._logger.debug("REMEMBER: %s (%.3fs)", message, duration)

            return PhaseResult(
                phase=HeartbeatPhase.REMEMBER,
                success=True,
                message=message,
                duration_seconds=duration,
            )
        except Exception as exc:
            return PhaseResult(
                phase=HeartbeatPhase.REMEMBER,
                success=False,
                message=f"Remember error: {exc}",
                duration_seconds=time.time() - start,
            )

    def _phase_notify(self) -> PhaseResult:
        """NOTIFY: Update vessel status.

        Generates a status report and updates vessel state.
        In production, this would also push to GitHub / notify fleet.
        """
        start = time.time()
        try:
            report = self.state_manager.get_status_report()
            self.state_manager.set_current_mission(None)

            duration = time.time() - start
            self._logger.debug(
                "NOTIFY: %s (errors=%d) (%.3fs)",
                report["status_level"],
                self.state_manager.state.error_count,
                duration,
            )

            return PhaseResult(
                phase=HeartbeatPhase.NOTIFY,
                success=True,
                message=f"Status: {report['status_level']}",
                data={"status_report": report},
                duration_seconds=duration,
            )
        except Exception as exc:
            return PhaseResult(
                phase=HeartbeatPhase.NOTIFY,
                success=False,
                message=f"Notify error: {exc}",
                duration_seconds=time.time() - start,
            )

    def run_forever(self, interval_seconds: int | None = None) -> None:
        """Run heartbeat loop forever.

        Args:
            interval_seconds: Time between cycles. Defaults to config.heartbeat_interval.
        """
        interval = interval_seconds or self.config.heartbeat_interval
        self._running = True
        self._logger.info(
            "Heartbeat started: vessel=%s, interval=%ds",
            self.config.vessel_id,
            interval,
        )

        while self._running:
            try:
                result = self.run_once()
                self._logger.info(
                    "Cycle %d: success=%s, mission=%s, duration=%.1fs",
                    result.cycle_number,
                    result.success,
                    result.mission_type or "none",
                    result.total_duration,
                )
            except Exception as exc:
                self._logger.error("Heartbeat cycle error: %s", exc)

            time.sleep(interval)

    def stop(self) -> None:
        """Signal the heartbeat loop to stop after the current cycle."""
        self._running = False
        self._logger.info("Heartbeat stop requested")

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def vessel_state(self) -> VesselState:
        return self.state_manager.state

    def get_status(self) -> dict:
        """Get current heartbeat system status."""
        return {
            "vessel_id": self.config.vessel_id,
            "cycle_count": self._cycle_count,
            "running": self._running,
            "interval": self.config.heartbeat_interval,
            "agent_dir": str(self.agent_dir),
            "pending_missions": len(read_next_queue(self.agent_dir)),
        }
