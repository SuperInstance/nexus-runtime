"""Fault injection testing framework for marine robotics systems."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class FaultType(Enum):
    STUCK_AT = "stuck_at"
    BIAS = "bias"
    NOISE = "noise"
    DROP = "drop"
    DELAY = "delay"
    CORRUPTION = "corruption"
    CRASH = "crash"
    BIT_FLIP = "bit_flip"
    OUT_OF_BOUNDS = "out_of_bounds"
    FREEZE = "freeze"


class FaultSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FaultConfig:
    fault_type: FaultType
    target: str
    probability: float = 1.0
    duration: float = 0.0
    severity: FaultSeverity = FaultSeverity.MEDIUM
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FaultScenario:
    name: str
    faults: List[FaultConfig] = field(default_factory=list)
    expected_behavior: str = "system_recovers"
    recovery_method: str = "default"


@dataclass
class FaultRecord:
    fault_type: FaultType
    target: str
    params: Dict[str, Any]
    injected: bool
    timestamp: float


@dataclass
class TestResult:
    scenario_name: str
    passed: bool
    recovery_time_ms: float
    observed_behavior: str = ""
    faults_injected: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


class FaultInjector:
    """Inject faults into sensors, communication links, computation modules, and timing."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._active_faults: Dict[str, FaultRecord] = {}
        self._injection_log: List[FaultRecord] = []

    def inject_sensor_fault(
        self, sensor_id: str, fault_type: FaultType, params: Optional[Dict[str, Any]] = None
    ) -> FaultRecord:
        p = params or {}
        record = FaultRecord(
            fault_type=fault_type,
            target=sensor_id,
            params=p,
            injected=True,
            timestamp=time.time(),
        )
        self._active_faults[sensor_id] = record
        self._injection_log.append(record)
        return record

    def inject_communication_fault(
        self, link_id: str, fault_type: FaultType, params: Optional[Dict[str, Any]] = None
    ) -> FaultRecord:
        p = params or {}
        record = FaultRecord(
            fault_type=fault_type,
            target=link_id,
            params=p,
            injected=True,
            timestamp=time.time(),
        )
        key = f"link:{link_id}"
        self._active_faults[key] = record
        self._injection_log.append(record)
        return record

    def inject_computation_fault(
        self, module_id: str, fault_type: FaultType, params: Optional[Dict[str, Any]] = None
    ) -> FaultRecord:
        p = params or {}
        record = FaultRecord(
            fault_type=fault_type,
            target=module_id,
            params=p,
            injected=True,
            timestamp=time.time(),
        )
        key = f"module:{module_id}"
        self._active_faults[key] = record
        self._injection_log.append(record)
        return record

    def inject_timing_fault(self, target: str, delay_ms: float) -> FaultRecord:
        record = FaultRecord(
            fault_type=FaultType.DELAY,
            target=target,
            params={"delay_ms": delay_ms},
            injected=True,
            timestamp=time.time(),
        )
        key = f"timing:{target}"
        self._active_faults[key] = record
        self._injection_log.append(record)
        return record

    def apply_fault(self, target: str, value: float) -> float:
        """Apply active fault to a value. Returns modified value."""
        if target not in self._active_faults:
            return value
        record = self._active_faults[target]
        ft = record.fault_type
        p = record.params
        if ft == FaultType.STUCK_AT:
            return p.get("stuck_value", 0.0)
        elif ft == FaultType.BIAS:
            return value + p.get("bias", 0.0)
        elif ft == FaultType.NOISE:
            magnitude = p.get("magnitude", 1.0)
            return value + self._rng.gauss(0, magnitude)
        elif ft == FaultType.BIT_FLIP:
            bits = p.get("bits", 1)
            mask = (1 << bits) - 1
            return float(int(value) ^ self._rng.randint(0, mask))
        elif ft == FaultType.OUT_OF_BOUNDS:
            return p.get("invalid_value", 999999.0)
        return value

    def clear_fault(self, target: str) -> bool:
        if target in self._active_faults:
            del self._active_faults[target]
            return True
        return False

    def clear_all_faults(self) -> None:
        self._active_faults.clear()

    def get_active_faults(self) -> List[FaultRecord]:
        return list(self._active_faults.values())

    def get_injection_log(self) -> List[FaultRecord]:
        return list(self._injection_log)


class FaultTestRunner:
    """Run fault scenarios against systems under test."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._results: List[TestResult] = []

    def run_scenario(
        self,
        scenario: FaultScenario,
        system_under_test: Callable[[FaultInjector], Dict[str, Any]],
    ) -> TestResult:
        injector = FaultInjector(seed=self._rng.randint(0, 2**31))
        start = time.time()
        try:
            for fc in scenario.faults:
                if "sensor" in fc.target.lower():
                    injector.inject_sensor_fault(fc.target, fc.fault_type, fc.params)
                elif "link" in fc.target.lower() or "comm" in fc.target.lower():
                    injector.inject_communication_fault(fc.target, fc.fault_type, fc.params)
                elif "module" in fc.target.lower() or "comp" in fc.target.lower():
                    injector.inject_computation_fault(fc.target, fc.fault_type, fc.params)
                elif "timing" in fc.target.lower() or "delay" in fc.target.lower():
                    injector.inject_timing_fault(fc.target, fc.params.get("delay_ms", 0))
                else:
                    injector.inject_sensor_fault(fc.target, fc.fault_type, fc.params)
            outcome = system_under_test(injector)
            end = time.time()
            passed = outcome.get("recovered", False)
            result = TestResult(
                scenario_name=scenario.name,
                passed=passed,
                recovery_time_ms=(end - start) * 1000.0,
                observed_behavior=outcome.get("behavior", scenario.expected_behavior),
                faults_injected=len(scenario.faults),
                details=outcome,
            )
        except Exception as e:
            end = time.time()
            result = TestResult(
                scenario_name=scenario.name,
                passed=False,
                recovery_time_ms=(end - start) * 1000.0,
                observed_behavior=f"crashed: {e}",
                faults_injected=len(scenario.faults),
                details={"error": str(e)},
            )
        self._results.append(result)
        return result

    def generate_scenarios(self, config: Dict[str, Any]) -> List[FaultScenario]:
        scenarios: List[FaultScenario] = []
        targets = config.get("targets", ["sensor_0", "sensor_1"])
        fault_types = config.get("fault_types", [ft for ft in FaultType])
        count = config.get("count", 3)
        for i in range(count):
            faults: List[FaultConfig] = []
            num_faults = config.get("faults_per_scenario", 2)
            for _ in range(num_faults):
                ft = self._rng.choice(fault_types)
                target = self._rng.choice(targets)
                faults.append(FaultConfig(
                    fault_type=ft,
                    target=target,
                    params={"test": True},
                ))
            scenarios.append(FaultScenario(
                name=f"scenario_{i}",
                faults=faults,
                expected_behavior="system_recovers",
                recovery_method="default",
            ))
        return scenarios

    def measure_recovery_time(
        self,
        scenario: FaultScenario,
        system: Callable[[FaultInjector], Dict[str, Any]],
    ) -> float:
        start = time.time()
        result = self.run_scenario(scenario, system)
        return result.recovery_time_ms

    def evaluate_robustness(self, results: Optional[List[TestResult]] = None) -> float:
        if results is None:
            results = self._results
        if not results:
            return 0.0
        passed = sum(1 for r in results if r.passed)
        return passed / len(results)

    def get_results(self) -> List[TestResult]:
        return list(self._results)
