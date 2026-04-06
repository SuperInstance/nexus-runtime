"""
Temporal logic checking for NEXUS runtime verification.

Supports LTL-like operators: Always (G), Eventually (F),
Until (U), Next (X), Never, and Implication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class TemporalFormula:
    """A parsed temporal logic formula."""

    formula_str: str
    formula_type: str  # "always", "eventually", "until", "next", "never", "imply", "atomic", "compound"
    atomic_props: List[str] = field(default_factory=list)
    description: str = ""


class TemporalLogicChecker:
    """Checks traces against temporal logic formulas."""

    def always(self, atomic: Callable[[Any], bool], trace: List[Any]) -> bool:
        """G operator: `atomic` must hold at every step in the trace."""
        if not trace:
            return True
        return all(atomic(state) for state in trace)

    def eventually(self, atomic: Callable[[Any], bool], trace: List[Any]) -> bool:
        """F operator: `atomic` must hold at some step in the trace."""
        if not trace:
            return False
        return any(atomic(state) for state in trace)

    def until(
        self,
        prop_a: Callable[[Any], bool],
        prop_b: Callable[[Any], bool],
        trace: List[Any],
    ) -> bool:
        """U operator: `prop_a` must hold until `prop_b` first holds.

        prop_b must eventually hold in the trace.
        """
        if not trace:
            return False
        for state in trace:
            if prop_b(state):
                return True
            if not prop_a(state):
                return False
        return False

    def next_step(
        self, prop: Callable[[Any], bool], trace: List[Any], step: int
    ) -> bool:
        """X operator: `prop` must hold at the next step after `step`."""
        if step < 0 or step + 1 >= len(trace):
            return False
        return prop(trace[step + 1])

    def never(self, atomic: Callable[[Any], bool], trace: List[Any]) -> bool:
        """`atomic` must never hold in the trace."""
        if not trace:
            return True
        return not any(atomic(state) for state in trace)

    def imply(
        self,
        antecedent: Callable[[Any], bool],
        consequent: Callable[[Any], bool],
        trace: List[Any],
    ) -> bool:
        """Implication: whenever `antecedent` holds, `consequent` must also hold."""
        if not trace:
            return True
        for state in trace:
            if antecedent(state) and not consequent(state):
                return False
        return True

    def parse_formula(self, formula_str: str) -> TemporalFormula:
        """Parse a formula string into a TemporalFormula dataclass.

        Supported syntax (case-insensitive):
          G(prop)          - always
          F(prop)          - eventually
          A U B            - A until B  (two identifiers)
          X(prop)          - next
          !prop            - never
          A => B           - imply
          prop             - atomic
        """
        formula_str = formula_str.strip()
        formula_type = "atomic"
        atomic_props: List[str] = []

        upper = formula_str.upper()

        if upper.startswith("G(") and formula_str.endswith(")"):
            formula_type = "always"
            inner = formula_str[2:-1]
            atomic_props.append(inner)
        elif upper.startswith("F(") and formula_str.endswith(")"):
            formula_type = "eventually"
            inner = formula_str[2:-1]
            atomic_props.append(inner)
        elif upper.startswith("X(") and formula_str.endswith(")"):
            formula_type = "next"
            inner = formula_str[2:-1]
            atomic_props.append(inner)
        elif upper.startswith("!"):
            formula_type = "never"
            inner = formula_str[1:]
            atomic_props.append(inner)
        elif " U " in upper or " u " in formula_str:
            formula_type = "until"
            parts = re.split(r"\s+[Uu]\s+", formula_str)
            atomic_props = [p.strip() for p in parts]
        elif "=>" in formula_str:
            formula_type = "imply"
            parts = formula_str.split("=>")
            atomic_props = [p.strip() for p in parts]
        else:
            atomic_props.append(formula_str)

        return TemporalFormula(
            formula_str=formula_str,
            formula_type=formula_type,
            atomic_props=atomic_props,
        )

    def check_trace(
        self,
        trace: List[Any],
        formula: TemporalFormula,
    ) -> Tuple[bool, Optional[str]]:
        """Check a trace against a parsed temporal formula.

        Returns:
            (passed, counterexample_description)
        """
        if not trace:
            return True, None

        # Build atomic checkers from formula atomic_props
        def make_atomic(prop_name: str) -> Callable[[Any], bool]:
            def checker(state: Any) -> bool:
                if isinstance(state, dict):
                    val = state.get(prop_name, False)
                    return bool(val)
                return False
            return checker

        if formula.formula_type == "always":
            fn = make_atomic(formula.atomic_props[0])
            if self.always(fn, trace):
                return True, None
            return False, f"G({formula.atomic_props[0]}) violated: property does not hold at all steps"

        elif formula.formula_type == "eventually":
            fn = make_atomic(formula.atomic_props[0])
            if self.eventually(fn, trace):
                return True, None
            return False, f"F({formula.atomic_props[0]}) violated: property never became true"

        elif formula.formula_type == "until":
            if len(formula.atomic_props) < 2:
                return False, "Invalid until formula: need two propositions"
            fn_a = make_atomic(formula.atomic_props[0])
            fn_b = make_atomic(formula.atomic_props[1])
            if self.until(fn_a, fn_b, trace):
                return True, None
            return False, f"{formula.atomic_props[0]} U {formula.atomic_props[1]} violated"

        elif formula.formula_type == "never":
            fn = make_atomic(formula.atomic_props[0])
            if self.never(fn, trace):
                return True, None
            return False, f"!{formula.atomic_props[0]} violated: property held when it should not"

        elif formula.formula_type == "imply":
            if len(formula.atomic_props) < 2:
                return False, "Invalid imply formula: need two propositions"
            fn_a = make_atomic(formula.atomic_props[0])
            fn_b = make_atomic(formula.atomic_props[1])
            if self.imply(fn_a, fn_b, trace):
                return True, None
            return False, f"{formula.atomic_props[0]} => {formula.atomic_props[1]} violated"

        elif formula.formula_type == "next":
            fn = make_atomic(formula.atomic_props[0])
            if len(trace) >= 2 and self.next_step(fn, trace, 0):
                return True, None
            return False, f"X({formula.atomic_props[0]}) violated at step 0"

        elif formula.formula_type == "atomic":
            fn = make_atomic(formula.atomic_props[0])
            if self.always(fn, trace):
                return True, None
            return False, f"{formula.atomic_props[0]} not always true in trace"

        return False, f"Unknown formula type: {formula.formula_type}"
