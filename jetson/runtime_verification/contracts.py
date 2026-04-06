"""
Design-by-contract verification for NEXUS runtime.

Provides pre/post condition checking, function wrapping, and
contract coverage computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Contract:
    """Represents a design-by-contract specification."""

    name: str
    preconditions: List[Callable[[Any], bool]] = field(default_factory=list)
    postconditions: List[Callable[[Any, Any], bool]] = field(default_factory=list)
    invariants: List[Callable[[Any, Any], bool]] = field(default_factory=list)
    description: str = ""


@dataclass
class ContractResult:
    """Result of a contract check (pre/post/invariant)."""

    passed: bool
    phase: str  # "precondition", "postcondition", "invariant"
    condition: Optional[str] = None
    details: Optional[str] = None


class ContractChecker:
    """Manages design-by-contract verification."""

    def __init__(self) -> None:
        self._contracts: Dict[str, Contract] = {}

    def register_contract(self, contract: Contract) -> None:
        """Register a contract by name."""
        self._contracts[contract.name] = contract

    def check_preconditions(self, contract_name: str, inputs: Any) -> ContractResult:
        """Check all preconditions for a named contract against inputs."""
        contract = self._contracts.get(contract_name)
        if contract is None:
            return ContractResult(
                passed=False,
                phase="precondition",
                condition="contract_not_found",
                details=f"Contract '{contract_name}' not registered",
            )
        for idx, pre_fn in enumerate(contract.preconditions):
            try:
                ok = pre_fn(inputs)
            except Exception as exc:
                return ContractResult(
                    passed=False,
                    phase="precondition",
                    condition=f"precondition_{idx}",
                    details=str(exc),
                )
            if not ok:
                return ContractResult(
                    passed=False,
                    phase="precondition",
                    condition=f"precondition_{idx}",
                    details=f"Precondition {idx} failed for inputs: {inputs!r}",
                )
        return ContractResult(
            passed=True,
            phase="precondition",
            condition=None,
            details="All preconditions passed",
        )

    def check_postconditions(
        self, contract_name: str, inputs: Any, outputs: Any
    ) -> ContractResult:
        """Check all postconditions for a named contract."""
        contract = self._contracts.get(contract_name)
        if contract is None:
            return ContractResult(
                passed=False,
                phase="postcondition",
                condition="contract_not_found",
                details=f"Contract '{contract_name}' not registered",
            )
        for idx, post_fn in enumerate(contract.postconditions):
            try:
                ok = post_fn(inputs, outputs)
            except Exception as exc:
                return ContractResult(
                    passed=False,
                    phase="postcondition",
                    condition=f"postcondition_{idx}",
                    details=str(exc),
                )
            if not ok:
                return ContractResult(
                    passed=False,
                    phase="postcondition",
                    condition=f"postcondition_{idx}",
                    details=f"Postcondition {idx} failed for outputs: {outputs!r}",
                )
        # Check invariants too
        for idx, inv_fn in enumerate(contract.invariants):
            try:
                ok = inv_fn(inputs, outputs)
            except Exception as exc:
                return ContractResult(
                    passed=False,
                    phase="invariant",
                    condition=f"invariant_{idx}",
                    details=str(exc),
                )
            if not ok:
                return ContractResult(
                    passed=False,
                    phase="invariant",
                    condition=f"invariant_{idx}",
                    details=f"Invariant {idx} violated",
                )
        return ContractResult(
            passed=True,
            phase="postcondition",
            condition=None,
            details="All postconditions and invariants passed",
        )

    def wrap_function(self, fn: Callable, contract: Contract) -> Callable:
        """Wrap a function so pre/post conditions are checked automatically."""
        def wrapped_fn(*args, **kwargs):
            inputs = {"args": args, "kwargs": kwargs}
            pre_result = self.check_preconditions(contract.name, inputs)
            if not pre_result.passed:
                raise AssertionError(
                    f"Precondition failed: {pre_result.details}"
                )
            result = fn(*args, **kwargs)
            post_result = self.check_postconditions(contract.name, inputs, result)
            if not post_result.passed:
                raise AssertionError(
                    f"Postcondition failed: {post_result.details}"
                )
            return result

        wrapped_fn.__name__ = f"contract_wrapped_{fn.__name__}"
        wrapped_fn.__qualname__ = wrapped_fn.__name__
        wrapped_fn._contract_name = contract.name  # type: ignore[attr-defined]
        return wrapped_fn

    def verify_function(
        self, fn: Callable, contract: Contract, test_inputs: List[Any]
    ) -> List[ContractResult]:
        """Run a function against test inputs, collecting contract results."""
        results: List[ContractResult] = []
        for inp in test_inputs:
            pre = self.check_preconditions(contract.name, inp)
            results.append(pre)
            if not pre.passed:
                continue
            try:
                output = fn(inp)
            except Exception as exc:
                results.append(
                    ContractResult(
                        passed=False,
                        phase="execution",
                        condition="exception",
                        details=str(exc),
                    )
                )
                continue
            post = self.check_postconditions(contract.name, inp, output)
            results.append(post)
        return results

    @staticmethod
    def compute_contract_coverage(contracts: int, calls: int) -> float:
        """Compute contract coverage as a percentage.

        Args:
            contracts: Total number of registered contracts.
            calls: Number of contracts that have been exercised.

        Returns:
            Percentage between 0.0 and 100.0.
        """
        if contracts <= 0:
            return 0.0
        coverage = (calls / contracts) * 100.0
        return min(max(coverage, 0.0), 100.0)
