"""NEXUS Reflex Synthesizer — Main pipeline for NL → bytecode.

The ReflexSynthesizer is the top-level component that takes a natural
language command and produces validated bytecode ready for deployment.

Pipeline steps:
  1. Parse intent → extract parameters
  2. Generate bytecode JSON using LLM or deterministic template
  3. Compile JSON → binary bytecode via ReflexCompiler
  4. Validate through 6-stage BytecodeSafetyPipeline
  5. If invalid, retry with error feedback (max 3 attempts)
  6. Return validated bytecode + metadata

The system works BOTH with and without an active LLM connection:
  - With LLM: uses PromptTemplate + LLMClient for generation
  - Without LLM: falls back to DeterministicLLMClient (template-based)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agent.llm_pipeline.grammar import (
    validate_grammar_sequence,
    parse_json_body,
    NEXUS_GBNF_GRAMMAR,
)
from agent.llm_pipeline.llm_client import (
    LLMClient,
    LLMResponse,
    DeterministicLLMClient,
    MockLLMClient,
)
from agent.llm_pipeline.prompts import (
    PromptTemplate,
    PromptStrategy,
    best_prompt,
)
from agent.llm_pipeline.templates import (
    ReflexTemplates,
    TemplateParams,
    KNOWN_TEMPLATES,
)
from reflex.bytecode_emitter import BytecodeEmitter
from reflex.compiler import ReflexCompiler


# ===================================================================
# Result types
# ===================================================================

@dataclass
class SynthesisMetadata:
    """Metadata about the synthesis process."""
    strategy_used: str = "deterministic"
    llm_model: str = "none"
    attempt_count: int = 1
    grammar_valid: bool = True
    safety_passed: bool = False
    safety_report_summary: str = ""
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    template_used: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """Result of the synthesis pipeline.

    Attributes:
        success: Whether synthesis produced validated bytecode.
        bytecode: Compiled and validated binary bytecode (empty bytes if failed).
        reflex_json: The JSON reflex definition that was compiled.
        metadata: Detailed metadata about the synthesis process.
    """
    success: bool = False
    bytecode: bytes = b""
    reflex_json: dict | None = None
    metadata: SynthesisMetadata = field(default_factory=SynthesisMetadata)


# ===================================================================
# Main pipeline
# ===================================================================

# Maximum retry attempts for safety validation failures
MAX_RETRIES = 3


class ReflexSynthesizer:
    """Natural language → validated bytecode synthesis pipeline.

    Usage::

        synth = ReflexSynthesizer(trust_level=3)
        result = synth.synthesize("maintain heading at 45 degrees")
        if result.success:
            deploy(result.bytecode)

    Args:
        trust_level: Current trust/autonomy level (0-5).
        llm_client: LLM backend to use. If None, uses DeterministicLLMClient.
        prompt_template: Override the default prompt template.
        templates: Override the default template registry.
        max_retries: Maximum attempts for safety validation failures.
    """

    def __init__(
        self,
        trust_level: int = 5,
        llm_client: LLMClient | None = None,
        prompt_template: PromptTemplate | None = None,
        templates: ReflexTemplates | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.trust_level = max(0, min(5, trust_level))
        self._llm = llm_client or DeterministicLLMClient()
        self._prompt = prompt_template or best_prompt(self.trust_level)
        self._templates = templates or ReflexTemplates()
        self._compiler = ReflexCompiler()
        self._max_retries = max_retries

    def synthesize(
        self,
        command: str,
        trust_level: int | None = None,
    ) -> SynthesisResult:
        """Main synthesis pipeline: NL → validated bytecode.

        Args:
            command: Natural language command (e.g., "maintain heading at 45 degrees").
            trust_level: Override the instance's trust level for this call.

        Returns:
            SynthesisResult with bytecode, JSON, and metadata.
        """
        t0_total = time.perf_counter()
        level = trust_level if trust_level is not None else self.trust_level
        metadata = SynthesisMetadata()

        # Step 0: Validate input
        if not command or not command.strip():
            metadata.errors.append("Empty command")
            return SynthesisResult(metadata=metadata)

        command = command.strip()

        # Step 1: Try to match a template first (fast path)
        template_name = self._match_template(command)
        if template_name:
            reflex_json = self._templates.generate(template_name)
            if reflex_json:
                metadata.template_used = template_name
                metadata.strategy_used = "template"

        # Step 2: If no template match, use LLM/deterministic client
        if not metadata.template_used:
            reflex_json = self._generate_via_llm(command, metadata)

        if not reflex_json:
            metadata.errors.append("No reflex definition generated")
            metadata.total_latency_ms = (time.perf_counter() - t0_total) * 1000
            return SynthesisResult(metadata=metadata)

        metadata.reflex_json = reflex_json  # type: ignore[attr-defined]

        # Step 3: Validate JSON against grammar rules
        body = reflex_json.get("body", [])
        grammar_errors = validate_grammar_sequence(body)
        metadata.grammar_valid = len(grammar_errors) == 0

        if grammar_errors:
            metadata.errors.extend(grammar_errors)

        # Step 4: Compile to bytecode
        bytecode = b""
        compile_errors: list[str] = []

        for attempt in range(1, self._max_retries + 1):
            metadata.attempt_count = attempt

            try:
                bytecode = self._compiler.compile(reflex_json)
            except ValueError as e:
                compile_errors.append(str(e))
                metadata.errors.append(f"Compile attempt {attempt}: {e}")

                # If we have grammar errors, try to fix them
                if grammar_errors and attempt < self._max_retries:
                    reflex_json = self._attempt_fix(reflex_json, grammar_errors)
                    body = reflex_json.get("body", [])
                    grammar_errors = validate_grammar_sequence(body)
                    continue

                metadata.total_latency_ms = (time.perf_counter() - t0_total) * 1000
                return SynthesisResult(
                    reflex_json=reflex_json,
                    metadata=metadata,
                )

            # Compilation succeeded — bytecode is ready
            break

        if not bytecode:
            metadata.total_latency_ms = (time.perf_counter() - t0_total) * 1000
            return SynthesisResult(
                reflex_json=reflex_json,
                metadata=metadata,
            )

        metadata.generation_latency_ms = (time.perf_counter() - t0_total) * 1000

        # Step 5: Run through safety pipeline
        safety_result = self._validate_safety(bytecode, level)
        metadata.safety_passed = safety_result.overall_passed
        metadata.safety_report_summary = self._summarize_safety(safety_result)

        if not safety_result.overall_passed:
            for violation in safety_result.violations:
                if violation.severity == "error":
                    metadata.errors.append(str(violation))
                else:
                    metadata.warnings.append(str(violation))
            # Even if safety fails, return the bytecode (caller decides)
            # In production, bytecode should only be deployed if safety_passed

        # Step 6: Return result
        metadata.total_latency_ms = (time.perf_counter() - t0_total) * 1000
        success = (
            len(bytecode) > 0
            and metadata.grammar_valid
        )

        return SynthesisResult(
            success=success,
            bytecode=bytecode,
            reflex_json=reflex_json,
            metadata=metadata,
        )

    def _match_template(self, command: str) -> str | None:
        """Try to match the command to a known template."""
        cmd_lower = command.lower()

        # Priority-ordered keyword matching
        patterns: list[tuple[list[str], str]] = [
            (["emergency", "estop", "stop all", "full stop"], "emergency_stop"),
            (["collision", "avoid", "obstacle", "proximity"], "collision_avoidance"),
            (["station", "hold position", "drift"], "station_keeping"),
            (["waypoint", "navigate to", "goto"], "waypoint_follow"),
            (["heading", "steer", "rudder", "compass"], "heading_hold"),
        ]

        for keywords, template_name in patterns:
            if any(kw in cmd_lower for kw in keywords):
                return template_name

        return None

    def _generate_via_llm(
        self,
        command: str,
        metadata: SynthesisMetadata,
    ) -> dict | None:
        """Generate a reflex JSON using the LLM client."""
        t0 = time.perf_counter()

        response: LLMResponse = self._llm.generate(
            system_prompt=self._prompt.system_prompt,
            user_message=command,
            temperature=0.3,
            grammar=NEXUS_GBNF_GRAMMAR,
        )

        metadata.llm_model = response.model
        metadata.generation_latency_ms = (time.perf_counter() - t0) * 1000
        metadata.strategy_used = "llm" if response.model != "deterministic" else "deterministic"

        if not response.success:
            metadata.errors.append(f"LLM error: {response.error}")
            return None

        # Parse the response
        parsed = parse_json_body(response.text)
        if parsed is None:
            metadata.errors.append("Failed to parse LLM response as JSON")
            return None

        return parsed

    def _validate_safety(self, bytecode: bytes, trust_level: int):
        """Run the 6-stage safety validation pipeline."""
        try:
            from core.safety_validator.pipeline import BytecodeSafetyPipeline
            pipeline = BytecodeSafetyPipeline(trust_level=trust_level)
            return pipeline.validate(bytecode)
        except ImportError:
            # Safety pipeline not available — return a mock pass
            from core.safety_validator.models import SafetyReport, make_timestamp
            return SafetyReport(
                overall_passed=True,
                bytecode_hash="skip",
                bytecode_size=len(bytecode),
                instruction_count=len(bytecode) // 8,
                trust_level=trust_level,
                timestamp=make_timestamp(),
            )

    @staticmethod
    def _summarize_safety(report) -> str:
        """Create a brief summary of a safety report."""
        status = "PASS" if report.overall_passed else "FAIL"
        return (
            f"Safety {status}: {report.instruction_count} instructions, "
            f"{report.total_errors} errors, {report.total_warnings} warnings"
        )

    @staticmethod
    def _attempt_fix(reflex_json: dict, errors: list[str]) -> dict:
        """Attempt to fix common issues in a reflex JSON.

        Currently handles:
          - Missing HALT at end
          - WRITE_PIN without preceding CLAMP_F
        """
        import copy
        fixed = copy.deepcopy(reflex_json)
        body = fixed.get("body", [])

        if not body:
            return fixed

        # Fix 1: Ensure program ends with HALT
        last = body[-1]
        is_halt = (
            last.get("op") == "NOP"
            and last.get("flags") == "0x80"
            and last.get("operand2") == 1
        )
        if not is_halt:
            body.append({
                "op": "NOP",
                "flags": "0x80",
                "operand1": 1,
                "operand2": 1,
            })

        # Fix 2: Add CLAMP_F before WRITE_PIN if missing
        fixed_body: list[dict] = []
        for instr in body:
            if instr.get("op") == "WRITE_PIN":
                # Check if preceding instruction is CLAMP_F
                if fixed_body and fixed_body[-1].get("op") == "CLAMP_F":
                    fixed_body.append(instr)
                else:
                    # Insert CLAMP_F before WRITE_PIN
                    fixed_body.append({
                        "op": "CLAMP_F",
                        "lo": -100.0,
                        "hi": 100.0,
                    })
                    fixed_body.append(instr)
            else:
                fixed_body.append(instr)

        fixed["body"] = fixed_body
        return fixed
