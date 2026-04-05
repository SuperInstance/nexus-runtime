"""NEXUS Orchestrator — Central coordinator wiring all modules together.

The main entry point for the NEXUS edge agent on Jetson. Connects:
  - Edge Heartbeat (mission execution cycle)
  - git-agent Bridge (fleet coordination)
  - Rosetta Stone (intent translation)
  - Skill System (behavior loading)
  - Safety Validator (bytecode verification)
  - Trust Engine (autonomy gating)
  - Emergency Protocol (safety monitoring)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from trust.increments import (
    IncrementTrustEngine,
    TrustParams,
    TrustEvent,
    SUBSYSTEMS,
)
from trust.attestation import TrustAttestation
from trust.propagation import TrustPropagator

from agent.rosetta_stone import RosettaStone, TranslationResult
from agent.skill_system import SkillCartridge, SkillRegistry
from agent.skill_system.cartridge_builder import CartridgeBuilder
from agent.emergency_protocol import (
    EmergencyProtocol,
    Incident,
    IncidentCategory,
    generate_incident_id,
)

from core.safety_validator import BytecodeSafetyPipeline, SafetyReport

from agent.nexus_orchestrator.system_status import SystemStatus, StatusAggregator
from agent.nexus_orchestrator.simulation import MissionSimulator, SimulationResult

logger = logging.getLogger("nexus.orchestrator")


# ── Result types ──────────────────────────────────────────────────

@dataclass
class CommandResult:
    """Result of processing a natural language command."""

    success: bool
    command: str = ""
    bytecode: bytes = b""
    bytecode_hash: str = ""
    intent_text: str = ""
    safety_passed: bool = False
    trust_allowed: bool = False
    deployed: bool = False
    commit_hash: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    translation_result: TranslationResult | None = None
    safety_report: SafetyReport | None = None
    deployment_status: str = ""


@dataclass
class SkillLoadResult:
    """Result of loading a skill cartridge."""

    success: bool
    skill_name: str = ""
    skill_id: str = ""
    trust_allowed: bool = False
    safety_passed: bool = False
    error: str = ""


@dataclass
class EmergencyResult:
    """Result of processing an emergency incident."""

    success: bool
    incident_id: str = ""
    level: str = ""
    category: str = ""
    actions_taken: list[str] = field(default_factory=list)
    escalated: bool = False
    error: str = ""


@dataclass
class ComparisonResult:
    """Result of comparing two bytecode programs."""

    verdict: str = ""
    differences: list[str] = field(default_factory=list)
    result_a: SimulationResult = field(default_factory=SimulationResult)
    result_b: SimulationResult = field(default_factory=SimulationResult)


# ── NexusOrchestrator ─────────────────────────────────────────────

class NexusOrchestrator:
    """Central coordinator that wires all NEXUS modules together.

    This is the main entry point for the NEXUS edge agent on Jetson.
    It connects all subsystems and provides a unified API for:
      - Natural language command processing
      - Skill loading and management
      - Emergency handling
      - Mission simulation
      - System status reporting
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the orchestrator and all subsystems.

        Args:
            config_path: Path to a JSON configuration file. If None,
                         uses sensible defaults for a test/development setup.
        """
        # Load configuration
        self._config: dict[str, Any] = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self._config = json.load(f)

        self.vessel_id = self._config.get("vessel_id", "nexus-vessel-001")
        self._start_time = time.time()
        self._running = False
        self._deployment_count = 0

        # ── Trust Engine ──
        trust_params = TrustParams()
        if "trust" in self._config:
            for k, v in self._config["trust"].items():
                if hasattr(trust_params, k):
                    setattr(trust_params, k, v)
        self.trust_engine = IncrementTrustEngine(params=trust_params)
        self.trust_engine.register_all_subsystems()
        self.trust_attestation = TrustAttestation()
        self.trust_propagator = TrustPropagator()

        # ── Safety Validator ──
        safety_config = self._config.get("safety", {})
        self.safety_pipeline = BytecodeSafetyPipeline(
            trust_level=5,  # Start at L5 for orchestrator-level validation
            safety_config=safety_config,
        )

        # ── Rosetta Stone (intent translation) ──
        self.rosetta = RosettaStone(trust_level=5, optimize=True)

        # ── Skill System ──
        self.skill_registry = SkillRegistry()
        self._cartridge_builder = CartridgeBuilder()

        # ── Emergency Protocol ──
        self.emergency_protocol = EmergencyProtocol(vessel_id=self.vessel_id)

        # ── Mission Simulator ──
        self.simulator = MissionSimulator()

        # ── Status Aggregator ──
        self.status_aggregator = StatusAggregator()

        # ── Heartbeat & Bridge (initialized lazily) ──
        self.heartbeat = None
        self.bridge = None
        self._heartbeat_config = None
        self._heartbeat_repo_path = None

    def init_heartbeat(self, repo_path: str | None = None) -> None:
        """Initialize the heartbeat system with a repo path.

        Args:
            repo_path: Path to the vessel git repo. If None, uses
                       the config or a temp directory for testing.
        """
        from agent.edge_heartbeat.config import HeartbeatConfig
        from agent.edge_heartbeat.heartbeat import EdgeHeartbeat

        path = repo_path or self._config.get("repo_path", "")
        if not path:
            import tempfile
            path = tempfile.mkdtemp(prefix="nexus-hb-")

        hb_config = HeartbeatConfig(
            vessel_id=self.vessel_id,
            repo_path=path,
            heartbeat_interval=self._config.get("heartbeat_interval", 300),
        )
        self._heartbeat_config = hb_config
        self._heartbeat_repo_path = path
        self.heartbeat = EdgeHeartbeat(config=hb_config)

        # Sync trust scores to vessel state
        self._sync_trust_to_heartbeat()

    def init_bridge(self, repo_path: str | None = None) -> None:
        """Initialize the git-agent bridge.

        Args:
            repo_path: Path to the vessel git repo. Creates one if needed.
                         If None, creates a separate temp directory from heartbeat.
        """
        from agent.nexus_bridge.bridge import NexusBridge

        path = repo_path or self._config.get("repo_path", "")
        if not path:
            import tempfile
            path = tempfile.mkdtemp(prefix="nexus-bridge-")

        self.bridge = NexusBridge(
            vessel_id=self.vessel_id,
            repo_path=path,
        )

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the orchestrator: begin heartbeat cycle."""
        self._running = True
        logger.info(
            "NexusOrchestrator started: vessel=%s, subsystems=%d",
            self.vessel_id, len(self.trust_engine.subsystems),
        )

    def stop(self) -> None:
        """Graceful shutdown of the orchestrator."""
        self._running = False
        if self.heartbeat is not None:
            self.heartbeat.stop()
        logger.info("NexusOrchestrator stopped: vessel=%s", self.vessel_id)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    # ── Natural Language Command Pipeline ──────────────────────────

    def process_natural_language_command(
        self, command: str
    ) -> CommandResult:
        """Full pipeline: text → Rosetta Stone → safety validate → deploy.

        Example: "navigate to waypoint 100,200"
        1. Parse intent via Rosetta Stone
        2. Compile to bytecode
        3. Validate bytecode (safety pipeline)
        4. Check trust level permits this operation
        5. Deploy via bridge (git commit + PR)
        6. Return result with deployment status

        Args:
            command: Human-readable natural language command.

        Returns:
            CommandResult with full pipeline status.
        """
        result = CommandResult(success=False, command=command)

        # Step 1: Translate via Rosetta Stone
        translation = self.rosetta.translate(command)
        result.translation_result = translation

        if not translation.success:
            result.errors = translation.errors
            result.warnings = translation.warnings
            result.intent_text = command
            return result

        result.bytecode = translation.bytecode or b""
        result.intent_text = command

        if translation.intent:
            result.intent_text = translation.intent.raw

        # Compute bytecode hash
        if result.bytecode:
            result.bytecode_hash = hashlib.sha256(result.bytecode).hexdigest()[:16]

        # Step 2: Safety validation
        safety_report = self.safety_pipeline.validate(result.bytecode)
        result.safety_report = safety_report
        result.safety_passed = safety_report.overall_passed

        if not safety_report.overall_passed:
            result.errors.extend(
                v.description for v in safety_report.violations
                if v.severity == "error"
            )
            result.warnings.extend(
                v.description for v in safety_report.violations
                if v.severity == "warning"
            )

        # Step 3: Trust check — only required for actual deployment
        avg_trust = self._get_average_trust()
        # Trust is only a hard requirement when bridge is connected
        if self.bridge is not None:
            result.trust_allowed = avg_trust >= 0.2
            if not result.trust_allowed:
                result.errors.append(
                    f"Trust too low for deployment: avg={avg_trust:.4f} (min=0.2)"
                )
        else:
            result.trust_allowed = True  # No bridge = no deployment needed

        # Step 4: Deploy (if bridge available and all checks pass)
        if (
            result.safety_passed
            and result.trust_allowed
            and self.bridge is not None
            and result.bytecode
        ):
            try:
                deploy_result = self.bridge.deploy_bytecode(
                    bytecode=result.bytecode,
                    source_reflex=f"rosetta-{hashlib.sha256(command.encode()).hexdigest()[:8]}",
                    provenance={
                        "author": "nexus-orchestrator",
                        "source": "natural_language",
                        "command": command,
                    },
                )
                result.deployed = deploy_result.success
                result.commit_hash = deploy_result.commit_hash
                result.deployment_status = deploy_result.deployment_status

                if deploy_result.success:
                    self._deployment_count += 1
                    # Record trust event for successful deployment
                    self.trust_engine.record_event(
                        TrustEvent.good(
                            event_type="command_ack",
                            quality=0.6,
                            timestamp=time.time(),
                            subsystem="navigation",
                        )
                    )
                else:
                    result.errors.append(deploy_result.error)

            except Exception as exc:
                result.errors.append(f"Deployment failed: {exc}")

        # Success = translation + safety passed (trust/deploy optional)
        result.success = (
            result.safety_passed
            and len(result.errors) == 0
        )
        return result

    # ── Skill Loading ─────────────────────────────────────────────

    def load_skill(self, skill_name: str) -> SkillLoadResult:
        """Load a skill cartridge, validate, and make ready for deployment.

        Args:
            skill_name: Name of the skill to load.

        Returns:
            SkillLoadResult with load status.
        """
        result = SkillLoadResult(success=False, skill_name=skill_name)

        # Check trust level — only required for deployment, not loading
        avg_trust = self._get_average_trust()
        result.trust_allowed = avg_trust >= 0.2

        # Try to find the skill in the registry
        existing = self.skill_registry.get_by_name(skill_name)
        if existing is not None:
            result.success = True
            result.skill_id = existing.skill_id
            result.safety_passed = True
            result.trust_allowed = True  # Already loaded
            return result

        # Create a minimal builtin skill if not found
        cartridge = self._create_builtin_skill(skill_name)
        if cartridge is None:
            result.error = f"Skill '{skill_name}' not found"
            return result

        # Validate bytecode
        if cartridge.bytecode and len(cartridge.bytecode) > 0:
            safety = self.safety_pipeline.validate(cartridge.bytecode)
            result.safety_passed = safety.overall_passed
            if not safety.overall_passed:
                result.error = "Skill bytecode failed safety validation"
                return result

        # Register in the registry
        try:
            skill_id = self.skill_registry.register(
                cartridge, source="builtin"
            )
            result.skill_id = skill_id
            result.success = True
        except ValueError as exc:
            result.error = str(exc)

        return result

    def _create_builtin_skill(self, name: str) -> SkillCartridge | None:
        """Create a builtin skill cartridge for common marine operations."""
        from agent.skill_system.cartridge import SkillParameter

        builtin_map = {
            "surface_navigation": {
                "description": "Basic surface navigation using GPS waypoints",
                "trust_required": 2,
                "inputs": [
                    SkillParameter(name="gps_heading", type="sensor", pin=0,
                                   unit="degrees"),
                    SkillParameter(name="gps_position", type="sensor", pin=1,
                                   unit="meters"),
                ],
                "outputs": [
                    SkillParameter(name="throttle", type="actuator", pin=0,
                                   range_min=-1.0, range_max=1.0),
                    SkillParameter(name="rudder", type="actuator", pin=1,
                                   range_min=-1.0, range_max=1.0),
                ],
            },
            "station_keeping": {
                "description": "Maintain position using PID control",
                "trust_required": 3,
                "inputs": [
                    SkillParameter(name="position_error_x", type="sensor", pin=2),
                    SkillParameter(name="position_error_y", type="sensor", pin=3),
                ],
                "outputs": [
                    SkillParameter(name="thrust_x", type="actuator", pin=0),
                    SkillParameter(name="thrust_y", type="actuator", pin=1),
                ],
            },
            "depth_control": {
                "description": "Submarine depth hold using pressure sensor",
                "trust_required": 3,
                "inputs": [
                    SkillParameter(name="depth", type="sensor", pin=4,
                                   unit="meters", range_min=0.0, range_max=500.0),
                ],
                "outputs": [
                    SkillParameter(name="ballast", type="actuator", pin=2,
                                   range_min=-1.0, range_max=1.0),
                ],
            },
            "sensor_survey": {
                "description": "Read all configured sensors and log results",
                "trust_required": 1,
                "inputs": [],
                "outputs": [],
            },
        }

        spec = builtin_map.get(name)
        if spec is None:
            return None

        return SkillCartridge(
            name=name,
            version="1.0.0",
            description=spec["description"],
            domain="marine",
            trust_required=spec["trust_required"],
            inputs=spec.get("inputs", []),
            outputs=spec.get("outputs", []),
            parameters={},
            provenance={"author": "nexus-orchestrator", "source": "builtin"},
        )

    # ── Emergency Handling ────────────────────────────────────────

    def handle_emergency(
        self, incident_type: str, details: dict
    ) -> EmergencyResult:
        """Process emergency through the full pipeline.

        Args:
            incident_type: Type of emergency (e.g., "SENSOR", "TRUST",
                           "SAFETY", "COMMUNICATION", "MISSION").
            details: Dict with incident details (context-specific).

        Returns:
            EmergencyResult with assessment and actions taken.
        """
        result = EmergencyResult(
            success=False,
            category=incident_type,
        )

        # Build vessel state for assessment
        vessel_state = {
            "safety_state": details.get("safety_state", {}),
            "last_comm_time": details.get("last_comm_time", time.time()),
            "mission_start": details.get("mission_start", time.time()),
            "expected_duration": details.get("expected_duration", 3600),
        }

        trust_scores = self._get_all_trust_scores()
        sensor_status = details.get("sensor_status", {})

        # Run emergency assessment
        assessment = self.emergency_protocol.assess(
            vessel_state, trust_scores, sensor_status
        )

        result.level = assessment.current_level
        result.actions_taken = assessment.actions_taken
        result.escalated = assessment.escalated

        # If incidents detected, escalate the first one
        if assessment.incidents_detected:
            incident = assessment.incidents_detected[0]
            result.incident_id = incident.id

            escalation = self.emergency_protocol.escalate(incident)
            result.actions_taken.extend(incident.auto_actions_taken)

            # Record trust event for emergency
            severity_map = {"GREEN": 0.0, "YELLOW": 0.3, "ORANGE": 0.6, "RED": 1.0}
            sev = severity_map.get(incident.level, 0.0)
            self.trust_engine.record_event(
                TrustEvent.bad(
                    event_type=f"emergency_{incident.level.lower()}",
                    severity=sev,
                    timestamp=time.time(),
                    subsystem="navigation",
                )
            )

        result.success = True
        return result

    # ── System Status ─────────────────────────────────────────────

    def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status.

        Returns:
            SystemStatus aggregated from all subsystems.
        """
        return self.status_aggregator.collect(self)

    # ── Mission Simulation ────────────────────────────────────────

    def simulate_mission(
        self,
        mission_text: str,
        sensor_data: dict[int, float] | None = None,
        max_cycles: int = 1000,
    ) -> SimulationResult:
        """Dry-run a mission without deploying. Returns predicted behavior.

        Args:
            mission_text: Natural language mission command.
            sensor_data: Optional simulated sensor readings.
            max_cycles: Maximum simulation cycles.

        Returns:
            SimulationResult with full execution trace.
        """
        # Translate to bytecode
        translation = self.rosetta.translate(mission_text)
        if not translation.success or not translation.bytecode:
            return SimulationResult(
                success=False,
                error=f"Translation failed: {'; '.join(translation.errors)}",
            )

        # Run simulation
        return self.simulator.simulate(
            translation.bytecode, sensor_data, max_cycles
        )

    def compare_bytecode(
        self,
        bytecode_a: bytes,
        bytecode_b: bytes,
        sensor_data: dict[int, float] | None = None,
    ) -> ComparisonResult:
        """Compare two bytecode programs side by side.

        Args:
            bytecode_a: First bytecode program.
            bytecode_b: Second bytecode program.
            sensor_data: Optional shared simulated sensor data.

        Returns:
            ComparisonResult with verdict.
        """
        comp = self.simulator.compare_bytecode(
            bytecode_a, bytecode_b, sensor_data
        )
        result = ComparisonResult(
            verdict=comp.verdict,
            differences=comp.differences,
            result_a=comp.result_a,
            result_b=comp.result_b,
        )
        return result

    # ── Trust Operations ──────────────────────────────────────────

    def record_trust_event(self, event: TrustEvent) -> None:
        """Record a trust event and update subsystem scores.

        Args:
            event: The trust event to record.
        """
        self.trust_engine.record_event(event)
        self._sync_trust_to_heartbeat()

    def create_attestation(self, bytecode: bytes) -> str:
        """Create a cryptographic trust attestation for bytecode.

        Args:
            bytecode: The bytecode to attest.

        Returns:
            Signed attestation string.
        """
        trust_scores = self._get_all_trust_scores()
        avg_trust = self._get_average_trust()
        bc_hash = TrustAttestation.compute_bytecode_hash(bytecode)
        level = min(
            (self.trust_engine.get_autonomy_level(s)
             for s in trust_scores),
            default=0,
        )

        return self.trust_attestation.create_attestation(
            vessel_id=self.vessel_id,
            trust_scores=trust_scores,
            bytecode_hash=bc_hash,
            trust_level=level,
        )

    def verify_attestation(
        self, attestation: str, bytecode: bytes | None = None
    ) -> bool:
        """Verify a trust attestation.

        Args:
            attestation: The attestation string to verify.
            bytecode: Optional bytecode to verify hash against.

        Returns:
            True if attestation is valid.
        """
        bc_hash = None
        if bytecode is not None:
            bc_hash = TrustAttestation.compute_bytecode_hash(bytecode)

        return self.trust_attestation.verify_attestation(
            attestation,
            vessel_id=self.vessel_id,
            expected_bytecode_hash=bc_hash,
        )

    # ── Internal helpers ──────────────────────────────────────────

    def _get_average_trust(self) -> float:
        """Compute average trust across all subsystems."""
        scores = self.trust_engine.get_all_scores()
        if not scores:
            return 0.0
        return sum(st.trust_score for st in scores.values()) / len(scores)

    def _get_all_trust_scores(self) -> dict[str, float]:
        """Get trust scores as a dict."""
        scores = self.trust_engine.get_all_scores()
        return {name: st.trust_score for name, st in scores.items()}

    def _get_min_trust(self) -> float:
        """Get minimum trust across all subsystems."""
        scores = self.trust_engine.get_all_scores()
        if not scores:
            return 0.0
        return min(st.trust_score for st in scores.values())

    def _sync_trust_to_heartbeat(self) -> None:
        """Sync trust scores to the heartbeat vessel state."""
        if self.heartbeat is not None:
            trust_scores = self._get_all_trust_scores()
            self.heartbeat.state_manager.update_from_trust(trust_scores)

            # Set autonomy level to min across subsystems
            levels = [
                self.trust_engine.get_autonomy_level(s)
                for s in trust_scores
            ]
            if levels:
                self.heartbeat.state_manager.set_autonomy_level(min(levels))
