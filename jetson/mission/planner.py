"""Mission planning and optimization for NEXUS marine robotics platform.

Provides plan creation, optimization, risk assessment, phase decomposition,
and alternative plan generation.
"""

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MissionObjective:
    """Objective for a mission plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    type: str = "general"
    target: Any = None
    priority: int = 3  # 1=CRITICAL, 2=HIGH, 3=MEDIUM, 4=LOW
    constraints: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None
    weight: float = 1.0


@dataclass
class MissionAction:
    """A single action within a mission phase."""
    name: str = ""
    action_type: str = "navigate"
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    resource_cost: Dict[str, float] = field(default_factory=dict)


@dataclass
class MissionPhase:
    """A phase within a mission plan."""
    name: str = ""
    actions: List[MissionAction] = field(default_factory=list)
    duration: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    resource_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResourceRequirements:
    """Resource requirements for a mission."""
    energy_wh: float = 0.0
    compute_percent: float = 0.0
    storage_mb: float = 0.0
    bandwidth_kbps: float = 0.0
    actuators: List[str] = field(default_factory=list)
    sensors: List[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Risk assessment for a mission plan."""
    overall_score: float = 0.0  # 0.0 (safe) to 1.0 (risky)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    factors: List[Dict[str, Any]] = field(default_factory=list)
    mitigation: List[str] = field(default_factory=list)


@dataclass
class MissionPlan:
    """Complete mission plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    objectives: List[MissionObjective] = field(default_factory=list)
    phases: List[MissionPhase] = field(default_factory=list)
    estimated_duration: float = 0.0
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    risk_assessment: RiskAssessment = field(default_factory=RiskAssessment)
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MissionPlanner:
    """Plans, optimizes, and validates marine robotics mission plans."""

    def __init__(self):
        self._plan_history: List[MissionPlan] = []

    def create_plan(self, objectives: List[MissionObjective],
                    constraints: Optional[Dict[str, Any]] = None) -> MissionPlan:
        """Create a mission plan from objectives and constraints."""
        constraints = constraints or {}
        phases = self._generate_phases(objectives, constraints)
        plan = MissionPlan(
            name=f"Mission_{objectives[0].type if objectives else 'general'}",
            objectives=objectives,
            phases=phases,
            constraints=constraints,
        )
        plan.estimated_duration = self.estimate_duration(phases)
        plan.resource_requirements = self.estimate_resources(phases)
        plan.risk_assessment = self.compute_risk_assessment(plan)
        self._plan_history.append(plan)
        return plan

    def optimize_plan(self, plan: MissionPlan,
                      criteria: Optional[Dict[str, Any]] = None) -> MissionPlan:
        """Optimize a plan by given criteria. Returns new optimized plan."""
        criteria = criteria or {"objective": "duration"}
        optimized = MissionPlan(
            name=f"{plan.name}_optimized",
            objectives=list(plan.objectives),
            phases=self._copy_phases(plan.phases),
            constraints=dict(plan.constraints),
            metadata={"source_plan": plan.id, "optimization_criteria": criteria},
        )
        if criteria.get("objective") == "duration":
            self._optimize_duration(optimized)
        elif criteria.get("objective") == "resources":
            self._optimize_resources(optimized)
        elif criteria.get("objective") == "risk":
            self._optimize_risk(optimized)
        elif criteria.get("objective") == "balanced":
            self._optimize_balanced(optimized)

        optimized.estimated_duration = self.estimate_duration(optimized.phases)
        optimized.resource_requirements = self.estimate_resources(optimized.phases)
        optimized.risk_assessment = self.compute_risk_assessment(optimized)
        self._plan_history.append(optimized)
        return optimized

    def estimate_duration(self, phases: List[MissionPhase]) -> float:
        """Estimate total mission duration from phases."""
        total = 0.0
        visited = set()
        cache = {}

        def phase_dur(name: str) -> float:
            if name in cache:
                return cache[name]
            phase_map = {p.name: p for p in phases}
            if name not in phase_map:
                return 0.0
            phase = phase_map[name]
            if name in visited:
                return 0.0  # Circular dependency
            visited.add(name)
            dur = phase.duration
            for dep in phase.dependencies:
                dur += phase_dur(dep)
            visited.discard(name)
            cache[name] = dur
            return dur

        for p in phases:
            phase_dur(p.name)

        return sum(phase_dur(p.name) for p in phases) / max(len(phases), 1)

    def estimate_resources(self, phases: List[MissionPhase]) -> ResourceRequirements:
        """Estimate resource requirements for a set of phases."""
        total_energy = 0.0
        max_compute = 0.0
        total_storage = 0.0
        max_bandwidth = 0.0
        all_actuators = set()
        all_sensors = set()

        for phase in phases:
            rr = phase.resource_requirements
            total_energy += rr.get("energy_wh", 0.0)
            compute = rr.get("compute_percent", 0.0)
            max_compute = max(max_compute, compute)
            total_storage += rr.get("storage_mb", 0.0)
            bw = rr.get("bandwidth_kbps", 0.0)
            max_bandwidth = max(max_bandwidth, bw)
            all_actuators.update(rr.get("actuators", []))
            all_sensors.update(rr.get("sensors", []))

        return ResourceRequirements(
            energy_wh=round(total_energy, 2),
            compute_percent=round(max_compute, 2),
            storage_mb=round(total_storage, 2),
            bandwidth_kbps=round(max_bandwidth, 2),
            actuators=sorted(all_actuators),
            sensors=sorted(all_sensors),
        )

    def compute_risk_assessment(self, plan: MissionPlan) -> RiskAssessment:
        """Compute risk assessment for a mission plan."""
        factors = []
        risk_score = 0.0

        # Phase risk
        high_risk_phases = [p for p in plan.phases if p.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]
        if high_risk_phases:
            ratio = len(high_risk_phases) / max(len(plan.phases), 1)
            risk_score += ratio * 0.3
            factors.append({
                "factor": "high_risk_phases",
                "count": len(high_risk_phases),
                "impact": ratio * 0.3,
            })

        # Duration risk (longer = more risk)
        if plan.estimated_duration > 3600:  # > 1 hour
            duration_risk = min((plan.estimated_duration - 3600) / 7200, 0.3)
            risk_score += duration_risk
            factors.append({
                "factor": "long_duration",
                "impact": duration_risk,
            })

        # Resource risk
        if plan.resource_requirements.energy_wh > 1000:
            risk_score += 0.1
            factors.append({"factor": "high_energy", "impact": 0.1})

        # Objective count risk
        if len(plan.objectives) > 5:
            risk_score += 0.1
            factors.append({"factor": "many_objectives", "impact": 0.1})

        # Deadline pressure
        for obj in plan.objectives:
            if obj.deadline and plan.estimated_duration > obj.deadline:
                risk_score += 0.2
                factors.append({"factor": "deadline_miss_risk", "objective": obj.id})
                break

        risk_score = min(risk_score, 1.0)

        # Determine level
        if risk_score < 0.25:
            level = RiskLevel.LOW
        elif risk_score < 0.5:
            level = RiskLevel.MEDIUM
        elif risk_score < 0.75:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL

        # Generate mitigations
        mitigations = []
        if any(f["factor"] == "long_duration" for f in factors):
            mitigations.append("Consider adding rest phases for system cooldown")
        if any(f["factor"] == "high_risk_phases" for f in factors):
            mitigations.append("Add pre-flight checks before high-risk phases")
        if any(f["factor"] == "deadline_miss_risk" for f in factors):
            mitigations.append("Reduce non-critical objectives or parallelize phases")
        if risk_score > 0.5:
            mitigations.append("Implement contingency plans for all phases")

        return RiskAssessment(
            overall_score=round(risk_score, 3),
            risk_level=level,
            factors=factors,
            mitigation=mitigations,
        )

    def generate_alternatives(self, plan: MissionPlan,
                              constraints: Optional[Dict[str, Any]] = None) -> List[MissionPlan]:
        """Generate alternative plans based on constraints."""
        constraints = constraints or {}
        alternatives = []

        # Duration-optimized alternative
        alt1 = self.optimize_plan(plan, {"objective": "duration"})
        alt1.name = f"{plan.name}_alt_duration"
        alternatives.append(alt1)

        # Resource-optimized alternative
        alt2 = self.optimize_plan(plan, {"objective": "resources"})
        alt2.name = f"{plan.name}_alt_resources"
        alternatives.append(alt2)

        # Risk-minimized alternative
        alt3 = self.optimize_plan(plan, {"objective": "risk"})
        alt3.name = f"{plan.name}_alt_safe"
        alternatives.append(alt3)

        # Apply constraints filtering
        if constraints.get("max_duration"):
            max_dur = constraints["max_duration"]
            alternatives = [a for a in alternatives if a.estimated_duration <= max_dur]

        return alternatives

    def decompose_phase(self, phase: MissionPhase) -> List[MissionPhase]:
        """Decompose a phase into sub-phases based on actions."""
        if not phase.actions:
            return [MissionPhase(name=f"{phase.name}_empty", dependencies=phase.dependencies)]

        sub_phases = []
        for i, action in enumerate(phase.actions):
            sub_name = f"{phase.name}_sub{i}_{action.name}"
            deps = list(phase.dependencies) if i == 0 else [sub_phases[-1].name]
            sub_phase = MissionPhase(
                name=sub_name,
                actions=[action],
                duration=action.duration,
                dependencies=deps,
                success_criteria=phase.success_criteria,
                risk_level=phase.risk_level,
                resource_requirements=phase.resource_requirements,
            )
            sub_phases.append(sub_phase)

        return sub_phases

    def validate_plan(self, plan: MissionPlan) -> Tuple[bool, List[str]]:
        """Validate a mission plan. Returns (is_valid, list_of_issues)."""
        issues = []

        # Check objectives
        if not plan.objectives:
            issues.append("Plan has no objectives")

        # Check phases
        if not plan.phases:
            issues.append("Plan has no phases")

        # Check phase dependencies
        phase_names = {p.name for p in plan.phases}
        for p in plan.phases:
            for dep in p.dependencies:
                if dep not in phase_names:
                    issues.append(f"Phase '{p.name}' depends on unknown phase '{dep}'")

        # Check for circular dependencies
        circular = self._detect_circular_deps(plan.phases)
        if circular:
            issues.append(f"Circular dependency detected: {' -> '.join(circular)}")

        # Check duration
        if plan.estimated_duration <= 0 and plan.phases:
            issues.append("Estimated duration is non-positive but phases exist")

        # Check constraints
        for obj in plan.objectives:
            if obj.deadline and plan.estimated_duration > obj.deadline:
                issues.append(f"Objective '{obj.id}' deadline ({obj.deadline}s) "
                              f"cannot be met (estimated: {plan.estimated_duration:.1f}s)")

        # Check success criteria
        for p in plan.phases:
            if not p.success_criteria:
                issues.append(f"Phase '{p.name}' has no success criteria")

        # Check for duplicate phase names
        names = [p.name for p in plan.phases]
        if len(names) != len(set(names)):
            issues.append("Duplicate phase names detected")

        return (len(issues) == 0, issues)

    def get_plan_history(self) -> List[MissionPlan]:
        """Return history of all generated plans."""
        return list(self._plan_history)

    def clear_history(self):
        """Clear plan history."""
        self._plan_history.clear()

    # --- Private helpers ---

    def _generate_phases(self, objectives: List[MissionObjective],
                         constraints: Dict[str, Any]) -> List[MissionPhase]:
        """Generate mission phases from objectives."""
        phases = []
        for i, obj in enumerate(objectives):
            phase = MissionPhase(
                name=f"phase_{obj.type}_{i}",
                duration=constraints.get("phase_duration", 60.0),
                dependencies=[phases[-1].name] if phases else [],
                success_criteria=[f"Complete {obj.type} objective {obj.id}"],
                risk_level=RiskLevel.LOW if obj.priority >= 3 else RiskLevel.MEDIUM,
                resource_requirements={
                    "energy_wh": 50.0 * obj.priority,
                    "compute_percent": 20.0,
                    "storage_mb": 10.0,
                    "actuators": ["thruster", "rudder"],
                    "sensors": ["gps", "imu"],
                },
            )
            phases.append(phase)
        return phases

    def _copy_phases(self, phases: List[MissionPhase]) -> List[MissionPhase]:
        """Deep copy phases."""
        return [
            MissionPhase(
                name=p.name,
                actions=[MissionAction(
                    name=a.name, action_type=a.action_type,
                    parameters=dict(a.parameters), duration=a.duration,
                    resource_cost=dict(a.resource_cost),
                ) for a in p.actions],
                duration=p.duration,
                dependencies=list(p.dependencies),
                success_criteria=list(p.success_criteria),
                risk_level=p.risk_level,
                resource_requirements=dict(p.resource_requirements),
            ) for p in phases
        ]

    def _optimize_duration(self, plan: MissionPlan):
        """Optimize plan for minimum duration."""
        for phase in plan.phases:
            phase.duration *= 0.85  # 15% reduction
            phase.resource_requirements["energy_wh"] *= 0.9

    def _optimize_resources(self, plan: MissionPlan):
        """Optimize plan for minimum resource usage."""
        for phase in plan.phases:
            phase.duration *= 1.1  # Slightly slower to save resources
            phase.resource_requirements["energy_wh"] *= 0.7
            phase.resource_requirements["compute_percent"] *= 0.8

    def _optimize_risk(self, plan: MissionPlan):
        """Optimize plan for minimum risk."""
        for phase in plan.phases:
            if phase.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                phase.risk_level = RiskLevel.MEDIUM
                phase.duration *= 1.2  # More time = less risk
                phase.success_criteria.append("Additional safety checks passed")

    def _optimize_balanced(self, plan: MissionPlan):
        """Balanced optimization."""
        for phase in plan.phases:
            phase.duration *= 0.95
            phase.resource_requirements["energy_wh"] *= 0.85

    def _detect_circular_deps(self, phases: List[MissionPhase]) -> Optional[List[str]]:
        """Detect circular dependencies in phases."""
        graph = {p.name: p.dependencies for p in phases}
        visited = set()
        rec_stack = set()

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    result = dfs(neighbor, path + [neighbor])
                    if result:
                        return result
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor) if neighbor in path else 0
                    return path[cycle_start:] + [neighbor]
            rec_stack.discard(node)
            return None

        for p in phases:
            if p.name not in visited:
                result = dfs(p.name, [p.name])
                if result:
                    return result
        return None
