"""NEXUS LLM Client — Interface for LLM inference with mock/deterministic fallback.

The NEXUS platform is designed to work BOTH with and without an active LLM
connection.  This module provides:

  - ``LLMClient``: abstract base for any LLM backend
  - ``MockLLMClient``: returns canned responses for testing
  - ``DeterministicLLMClient``: generates bytecode deterministically from
    templates (no network required, always works)

The deterministic client is the primary production path: it maps natural
language commands to pre-built reflex templates, bypassing the LLM entirely.
This ensures the vessel can always respond to commands even offline.
"""

from __future__ import annotations

import json
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ===================================================================
# Response type
# ===================================================================

@dataclass
class LLMResponse:
    """Response from an LLM call.

    Attributes:
        text: Raw text response from the LLM.
        parsed: Parsed JSON dictionary (if the response was valid JSON).
        success: Whether the LLM call succeeded.
        error: Error message if the call failed.
        model: Which model was used (identifier string).
        latency_ms: Round-trip latency in milliseconds.
        tokens_used: Approximate token count (if available).
    """
    text: str = ""
    parsed: dict | None = None
    success: bool = True
    error: str = ""
    model: str = "unknown"
    latency_ms: float = 0.0
    tokens_used: int = 0


# ===================================================================
# Abstract base client
# ===================================================================

class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        grammar: str | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            system_prompt: System prompt defining the LLM's role.
            user_message: User's natural language command.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            grammar: Optional GBNF grammar for constrained decoding.

        Returns:
            LLMResponse with the generated text and metadata.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the LLM backend is reachable."""
        ...


# ===================================================================
# Mock client — returns canned responses
# ===================================================================

class MockLLMClient(LLMClient):
    """Mock LLM client for testing. Returns predefined responses.

    Args:
        canned_responses: Map of user message substring → JSON response.
                          If no match, returns a fallback HALT-only program.
    """

    def __init__(
        self,
        canned_responses: dict[str, dict] | None = None,
        latency_ms: float = 50.0,
    ) -> None:
        self._responses = canned_responses or {}
        self._latency_ms = latency_ms

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        grammar: str | None = None,
    ) -> LLMResponse:
        """Return a canned response matching the user message."""
        # Find best matching canned response
        for key, value in self._responses.items():
            if key.lower() in user_message.lower():
                text = json.dumps(value, indent=2)
                return LLMResponse(
                    text=text,
                    parsed=value,
                    success=True,
                    model="mock",
                    latency_ms=self._latency_ms,
                )

        # Fallback: simple HALT program
        fallback = {
            "name": "fallback",
            "intent": user_message,
            "body": [
                {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
            ],
        }
        text = json.dumps(fallback, indent=2)
        return LLMResponse(
            text=text,
            parsed=fallback,
            success=True,
            model="mock",
            latency_ms=self._latency_ms,
        )

    def is_available(self) -> bool:
        return True


# ===================================================================
# Deterministic client — template-based, no LLM needed
# ===================================================================

class DeterministicLLMClient(LLMClient):
    """Deterministic bytecode generation using reflex templates.

    This client does NOT call any external LLM. Instead, it maps natural
    language commands to pre-built bytecode templates using keyword matching.
    Always works, zero latency, zero network dependency.

    The pipeline:
      1. Parse the command to extract intent and parameters
      2. Match against known template patterns
      3. Generate parameterized bytecode from the template
      4. Return as a JSON reflex definition
    """

    def __init__(self) -> None:
        # Template patterns: (keywords, template_generator)
        self._patterns: list[tuple[list[str], str]] = [
            # Emergency stop — highest priority
            (["emergency", "estop", "stop all", "full stop"], "emergency_stop"),
            # Collision avoidance
            (["collision", "avoid", "obstacle", "proximity"], "collision_avoidance"),
            # Station keeping
            (["station", "hold position", "drift"], "station_keeping"),
            # Waypoint following
            (["waypoint", "navigate to", "course to", "goto", "go to"], "waypoint_follow"),
            # Heading hold
            (["heading", "steer", "rudder", "compass", "course"], "heading_hold"),
            # Generic read
            (["read", "sensor", "measure", "get"], "read_sensor"),
            # Generic write
            (["set", "write", "output", "drive"], "set_actuator"),
        ]

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        grammar: str | None = None,
    ) -> LLMResponse:
        """Generate a deterministic reflex program from the command."""
        t0 = time.perf_counter()

        # Try to match a template
        msg_lower = user_message.lower()
        matched_template = None

        for keywords, template_name in self._patterns:
            if any(kw in msg_lower for kw in keywords):
                matched_template = template_name
                break

        if matched_template:
            reflex = self._build_from_template(matched_template, user_message)
        else:
            # Default: simple read + halt
            reflex = {
                "name": "default_reflex",
                "intent": user_message,
                "body": [
                    {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
                ],
            }

        text = json.dumps(reflex, indent=2)
        latency = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            text=text,
            parsed=reflex,
            success=True,
            model="deterministic",
            latency_ms=latency,
        )

    def is_available(self) -> bool:
        return True

    def _build_from_template(self, template_name: str, command: str) -> dict:
        """Build a reflex JSON from a template name and command text."""
        if template_name == "emergency_stop":
            return {
                "name": "emergency_stop",
                "intent": command,
                "body": [
                    {"op": "PUSH_F32", "value": 0.0},
                    {"op": "CLAMP_F", "lo": -100.0, "hi": 100.0},
                    {"op": "WRITE_PIN", "arg": 5},
                    {"op": "PUSH_F32", "value": 0.0},
                    {"op": "CLAMP_F", "lo": -90.0, "hi": 90.0},
                    {"op": "WRITE_PIN", "arg": 4},
                    {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
                ],
            }
        elif template_name == "heading_hold":
            heading = self._extract_number(command, default=45.0)
            rudder_pin = self._extract_pin(command, default=4)
            rudder_range = 30.0
            return {
                "name": f"heading_hold_{int(heading)}",
                "intent": command,
                "body": [
                    {"op": "READ_PIN", "arg": 2, "label": "loop"},
                    {"op": "PUSH_F32", "value": heading},
                    {"op": "SUB_F"},
                    {"op": "CLAMP_F", "lo": -rudder_range, "hi": rudder_range},
                    {"op": "WRITE_PIN", "arg": rudder_pin},
                    {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
                ],
            }
        elif template_name == "collision_avoidance":
            threshold = self._extract_number(command, default=5.0)
            safe_throttle = self._extract_number(command, default=10.0, index=1)
            return {
                "name": "collision_avoidance",
                "intent": command,
                "body": [
                    {"op": "READ_PIN", "arg": 8},
                    {"op": "PUSH_F32", "value": threshold},
                    {"op": "LT_F"},
                    {"op": "JUMP_IF_FALSE", "target": "safe"},
                    {"op": "PUSH_F32", "value": safe_throttle},
                    {"op": "CLAMP_F", "lo": -100.0, "hi": 100.0},
                    {"op": "WRITE_PIN", "arg": 5, "label": "safe"},
                    {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
                ],
            }
        elif template_name == "station_keeping":
            return {
                "name": "station_keeping",
                "intent": command,
                "body": [
                    {"op": "READ_PIN", "arg": 9},
                    {"op": "PUSH_F32", "value": 0.0},
                    {"op": "SUB_F"},
                    {"op": "PUSH_F32", "value": 2.0},
                    {"op": "MUL_F"},
                    {"op": "CLAMP_F", "lo": -30.0, "hi": 30.0},
                    {"op": "WRITE_PIN", "arg": 4},
                    {"op": "READ_PIN", "arg": 10},
                    {"op": "PUSH_F32", "value": 0.0},
                    {"op": "SUB_F"},
                    {"op": "PUSH_F32", "value": 2.0},
                    {"op": "MUL_F"},
                    {"op": "CLAMP_F", "lo": -50.0, "hi": 50.0},
                    {"op": "WRITE_PIN", "arg": 5},
                    {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
                ],
            }
        elif template_name == "waypoint_follow":
            target_heading = self._extract_number(command, default=0.0)
            return {
                "name": "waypoint_follow",
                "intent": command,
                "body": [
                    {"op": "READ_PIN", "arg": 9},
                    {"op": "PUSH_F32", "value": target_heading},
                    {"op": "SUB_F"},
                    {"op": "ABS_F"},
                    {"op": "PUSH_F32", "value": 5.0},
                    {"op": "LT_F"},
                    {"op": "JUMP_IF_FALSE", "target": "steer"},
                    {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1, "label": "arrived"},
                    {"op": "READ_PIN", "arg": 2, "label": "steer"},
                    {"op": "PUSH_F32", "value": target_heading},
                    {"op": "SUB_F"},
                    {"op": "CLAMP_F", "lo": -30.0, "hi": 30.0},
                    {"op": "WRITE_PIN", "arg": 4},
                    {"op": "JUMP", "target": "arrived"},
                ],
            }
        elif template_name == "read_sensor":
            pin = self._extract_pin(command, default=2)
            return {
                "name": "read_sensor",
                "intent": command,
                "body": [
                    {"op": "READ_PIN", "arg": pin},
                    {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
                ],
            }
        elif template_name == "set_actuator":
            value = self._extract_number(command, default=50.0)
            pin = self._extract_pin(command, default=5)
            return {
                "name": "set_actuator",
                "intent": command,
                "body": [
                    {"op": "PUSH_F32", "value": value},
                    {"op": "CLAMP_F", "lo": -100.0, "hi": 100.0},
                    {"op": "WRITE_PIN", "arg": pin},
                    {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
                ],
            }
        else:
            return {
                "name": "unknown_template",
                "intent": command,
                "body": [
                    {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
                ],
            }

    @staticmethod
    def _extract_number(text: str, default: float = 0.0, index: int = 0) -> float:
        """Extract the nth floating point number from text."""
        import re
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        if index < len(numbers):
            return float(numbers[index])
        return default

    @staticmethod
    def _extract_pin(text: str, default: int = 2) -> int:
        """Extract a pin number from text."""
        import re
        # Look for "pin X" pattern
        match = re.search(r"pin\s+(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return default


# ===================================================================
# SDK Bridge — call z-ai-web-dev-sdk via Node.js subprocess
# ===================================================================

class SDKBridgeClient(LLMClient):
    """LLM client that calls z-ai-web-dev-sdk via a Node.js helper.

    This is the production LLM path. It spawns a Node.js process to call
    the SDK's chat completions API.

    Falls back to DeterministicLLMClient if Node.js is not available.

    Args:
        node_helper_path: Path to the Node.js helper script.
        model: Model identifier (e.g., "qwen2.5-coder-7b").
        fallback_client: Client to use if Node.js is unavailable.
    """

    def __init__(
        self,
        node_helper_path: str = "/tmp/nexus-runtime/jetson/agent/llm_pipeline/llm_bridge.js",
        model: str = "qwen2.5-coder-7b",
        fallback_client: LLMClient | None = None,
    ) -> None:
        self._node_path = node_helper_path
        self._model = model
        self._fallback = fallback_client or DeterministicLLMClient()

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        grammar: str | None = None,
    ) -> LLMResponse:
        """Call the SDK via Node.js subprocess."""
        if not self.is_available():
            return self._fallback.generate(
                system_prompt, user_message, temperature, max_tokens, grammar,
            )

        t0 = time.perf_counter()
        try:
            payload = json.dumps({
                "system": system_prompt,
                "user": user_message,
                "temperature": temperature,
                "maxTokens": max_tokens,
                "grammar": grammar,
            })

            result = subprocess.run(
                ["node", self._node_path],
                input=payload,
                capture_output=True,
                text=True,
                timeout=30,
            )

            latency = (time.perf_counter() - t0) * 1000

            if result.returncode != 0:
                return LLMResponse(
                    text="",
                    success=False,
                    error=result.stderr.strip(),
                    model=self._model,
                    latency_ms=latency,
                )

            # Parse the output
            output = result.stdout.strip()
            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    text = parsed.get("content", output)
                else:
                    text = output
            except json.JSONDecodeError:
                text = output
                parsed = None

            return LLMResponse(
                text=text,
                parsed=parsed,
                success=True,
                model=self._model,
                latency_ms=latency,
            )

        except subprocess.TimeoutExpired:
            latency = (time.perf_counter() - t0) * 1000
            return LLMResponse(
                text="",
                success=False,
                error="LLM call timed out (30s)",
                model=self._model,
                latency_ms=latency,
            )
        except FileNotFoundError:
            # Node.js not available
            return self._fallback.generate(
                system_prompt, user_message, temperature, max_tokens, grammar,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return LLMResponse(
                text="",
                success=False,
                error=str(e),
                model=self._model,
                latency_ms=latency,
            )

    def is_available(self) -> bool:
        """Check if Node.js and the helper script are available."""
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
