"""Comprehensive tests for the NEXUS LLM Inference Pipeline.

Tests organized by module:
  1. Prompt templates (Strategy A/B/C) — 20+ tests
  2. GBNF grammar — 15+ tests
  3. LLM client (mock/deterministic) — 15+ tests
  4. Reflex templates — 15+ tests
  5. ReflexSynthesizer — 15+ tests
  Total: 80+ tests
"""

from __future__ import annotations

import json
import math
import struct

import pytest

from agent.llm_pipeline.prompts import (
    PromptStrategy,
    PromptTemplate,
    strategy_a_comprehensive,
    strategy_b_grammar,
    strategy_c_fewshot,
    best_prompt,
    all_strategies,
)
from agent.llm_pipeline.grammar import (
    NEXUS_GBNF_GRAMMAR,
    validate_grammar_sequence,
    parse_json_body,
    VALID_OPCODE_NAMES,
)
from agent.llm_pipeline.llm_client import (
    LLMClient,
    LLMResponse,
    MockLLMClient,
    DeterministicLLMClient,
    SDKBridgeClient,
)
from agent.llm_pipeline.templates import (
    ReflexTemplates,
    TemplateParams,
    KNOWN_TEMPLATES,
    heading_hold,
    collision_avoidance,
    waypoint_follow,
    station_keeping,
    emergency_stop,
)
from agent.llm_pipeline.synthesizer import (
    ReflexSynthesizer,
    SynthesisResult,
    SynthesisMetadata,
)
from reflex.bytecode_emitter import (
    BytecodeEmitter,
    INSTR_SIZE,
    pack_instruction,
    unpack_instruction,
    f32_to_f16_bits,
    float_to_u32,
    u32_to_float,
    FLAGS_SYSCALL,
    FLAGS_IS_FLOAT,
    FLAGS_HAS_IMMEDIATE,
    FLAGS_EXTENDED_CLAMP,
)
from shared.opcodes import (
    OPCODE_NAMES,
    OPCODE_VALUES,
    OPCODE_CORE_COUNT,
    OPCODE_A2A_COUNT,
    OPCODE_TOTAL_COUNT,
    opcode_name,
    is_core_opcode,
    is_a2a_opcode,
    is_valid_opcode,
    NOP,
    PUSH_I8,
    PUSH_I16,
    PUSH_F32,
    CLAMP_F,
    READ_PIN,
    WRITE_PIN,
    JUMP,
    JUMP_IF_FALSE,
    ADD_F,
    SUB_F,
    MUL_F,
    LT_F,
    ABS_F,
    HALT_OPCODE,
)


# ===================================================================
# 1. PROMPT TEMPLATE TESTS
# ===================================================================

class TestPromptStrategy:
    """Tests for the PromptStrategy enum."""

    def test_comprehensive_exists(self):
        assert PromptStrategy.COMPREHENSIVE.value == "comprehensive"

    def test_grammar_based_exists(self):
        assert PromptStrategy.GRAMMAR_BASED.value == "grammar_based"

    def test_fewshot_exists(self):
        assert PromptStrategy.FEWSHOT.value == "fewshot"

    def test_three_strategies(self):
        assert len(PromptStrategy) == 3


class TestStrategyA:
    """Tests for Strategy A (Comprehensive)."""

    def test_is_prompt_template(self):
        assert isinstance(strategy_a_comprehensive, PromptTemplate)

    def test_strategy_type(self):
        assert strategy_a_comprehensive.strategy == PromptStrategy.COMPREHENSIVE

    def test_has_system_prompt(self):
        assert len(strategy_a_comprehensive.system_prompt) > 100

    def test_system_prompt_contains_opcode_reference(self):
        assert "PUSH_F32" in strategy_a_comprehensive.system_prompt
        assert "WRITE_PIN" in strategy_a_comprehensive.system_prompt
        assert "CLAMP_F" in strategy_a_comprehensive.system_prompt

    def test_system_prompt_contains_safety_rules(self):
        assert "CLAMP_F" in strategy_a_comprehensive.system_prompt
        assert "WRITE_PIN" in strategy_a_comprehensive.system_prompt
        assert "safety" in strategy_a_comprehensive.system_prompt.lower()

    def test_system_prompt_contains_trust_levels(self):
        assert "L0" in strategy_a_comprehensive.system_prompt
        assert "L5" in strategy_a_comprehensive.system_prompt

    def test_system_prompt_contains_instruction_format(self):
        assert "8 bytes" in strategy_a_comprehensive.system_prompt
        assert "opcode" in strategy_a_comprehensive.system_prompt.lower()

    def test_system_prompt_contains_stack_effects(self):
        assert "stack" in strategy_a_comprehensive.system_prompt.lower()

    def test_system_prompt_contains_examples(self):
        assert "EXAMPLE" in strategy_a_comprehensive.system_prompt

    def test_opcode_count(self):
        assert strategy_a_comprehensive.opcode_count > 50

    def test_safety_rule_count(self):
        assert strategy_a_comprehensive.safety_rule_count >= 5

    def test_example_count(self):
        assert strategy_a_comprehensive.example_count >= 3

    def test_contains_all_core_opcodes(self):
        """System prompt should reference all 32 core opcodes."""
        prompt = strategy_a_comprehensive.system_prompt
        for name in [
            "NOP", "PUSH_I8", "PUSH_I16", "PUSH_F32", "POP", "DUP",
            "SWAP", "ROT", "ADD_F", "SUB_F", "MUL_F", "DIV_F",
            "NEG_F", "ABS_F", "MIN_F", "MAX_F", "CLAMP_F",
            "EQ_F", "LT_F", "GT_F", "LTE_F", "GTE_F",
            "AND_B", "OR_B", "XOR_B", "NOT_B",
            "READ_PIN", "WRITE_PIN", "READ_TIMER_MS",
            "JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE",
        ]:
            assert name in prompt, f"Missing opcode {name} in Strategy A"

    def test_contains_a2a_opcodes(self):
        prompt = strategy_a_comprehensive.system_prompt
        assert "DECLARE_INTENT" in prompt
        assert "TELL" in prompt
        assert "TRUST_CHECK" in prompt

    def test_trust_level_default(self):
        assert strategy_a_comprehensive.trust_level == 5


class TestStrategyB:
    """Tests for Strategy B (Grammar-based)."""

    def test_is_prompt_template(self):
        assert isinstance(strategy_b_grammar, PromptTemplate)

    def test_strategy_type(self):
        assert strategy_b_grammar.strategy == PromptStrategy.GRAMMAR_BASED

    def test_has_grammar_definition(self):
        assert "GRAMMAR" in strategy_b_grammar.system_prompt

    def test_grammar_has_production_rules(self):
        prompt = strategy_b_grammar.system_prompt
        assert "program" in prompt.lower()
        assert "instruction" in prompt.lower()

    def test_grammar_safety_constraints(self):
        prompt = strategy_b_grammar.system_prompt
        assert "io_write" in prompt.lower() or "write" in prompt.lower()
        assert "clamp" in prompt.lower()

    def test_grammar_has_marine_pins(self):
        assert "Pin 4" in strategy_b_grammar.system_prompt or "pin 4" in strategy_b_grammar.system_prompt.lower()

    def test_grammar_no_examples(self):
        assert strategy_b_grammar.example_count == 0


class TestStrategyC:
    """Tests for Strategy C (Few-shot)."""

    def test_is_prompt_template(self):
        assert isinstance(strategy_c_fewshot, PromptTemplate)

    def test_strategy_type(self):
        assert strategy_c_fewshot.strategy == PromptStrategy.FEWSHOT

    def test_has_examples(self):
        assert strategy_c_fewshot.example_count >= 5

    def test_has_halt_in_examples(self):
        assert "0x80" in strategy_c_fewshot.system_prompt

    def test_has_clamp_before_write_in_examples(self):
        prompt = strategy_c_fewshot.system_prompt
        # At least one example should show CLAMP_F before WRITE_PIN
        assert "CLAMP_F" in prompt
        assert "WRITE_PIN" in prompt

    def test_has_emergency_stop_example(self):
        assert "emergency" in strategy_c_fewshot.system_prompt.lower() or "estop" in strategy_c_fewshot.system_prompt.lower()

    def test_minimal_preamble(self):
        # Strategy C should have a shorter preamble than A
        assert len(strategy_c_fewshot.system_prompt) < len(strategy_a_comprehensive.system_prompt)


class TestPromptHelpers:
    """Tests for prompt helper functions."""

    def test_best_prompt_returns_template(self):
        result = best_prompt()
        assert isinstance(result, PromptTemplate)

    def test_best_prompt_returns_strategy_a(self):
        result = best_prompt()
        assert result.strategy == PromptStrategy.COMPREHENSIVE

    def test_all_strategies(self):
        strategies = all_strategies()
        assert len(strategies) == 3
        assert PromptStrategy.COMPREHENSIVE in strategies
        assert PromptStrategy.GRAMMAR_BASED in strategies
        assert PromptStrategy.FEWSHOT in strategies


# ===================================================================
# 2. GBNF GRAMMAR TESTS
# ===================================================================

class TestGBNFGrammar:
    """Tests for the GBNF grammar string."""

    def test_grammar_is_nonempty(self):
        assert len(NEXUS_GBNF_GRAMMAR) > 200

    def test_grammar_has_root_rule(self):
        assert "root" in NEXUS_GBNF_GRAMMAR

    def test_grammar_has_reflex_program(self):
        assert "reflex-program" in NEXUS_GBNF_GRAMMAR

    def test_grammar_has_instruction_rules(self):
        assert "push-f32" in NEXUS_GBNF_GRAMMAR
        assert "push-i8" in NEXUS_GBNF_GRAMMAR
        assert "push-i16" in NEXUS_GBNF_GRAMMAR

    def test_grammar_has_halt(self):
        assert "halt" in NEXUS_GBNF_GRAMMAR

    def test_grammar_has_clamp(self):
        assert "clamp-f" in NEXUS_GBNF_GRAMMAR

    def test_grammar_has_write_pin(self):
        assert "write-pin" in NEXUS_GBNF_GRAMMAR

    def test_grammar_has_branch(self):
        assert "branch-name" in NEXUS_GBNF_GRAMMAR

    def test_grammar_has_a2a(self):
        assert "a2a" in NEXUS_GBNF_GRAMMAR

    def test_grammar_has_string_value(self):
        assert "string-value" in NEXUS_GBNF_GRAMMAR

    def test_grammar_has_float_value(self):
        assert "float-value" in NEXUS_GBNF_GRAMMAR


class TestValidOpcodeNames:
    """Tests for the VALID_OPCODE_NAMES constant."""

    def test_has_all_core_opcodes(self):
        core_names = set(OPCODE_NAMES.values())
        for name in core_names:
            assert name in VALID_OPCODE_NAMES, f"Missing {name}"

    def test_has_a2a_opcodes(self):
        assert "DECLARE_INTENT" in VALID_OPCODE_NAMES
        assert "TELL" in VALID_OPCODE_NAMES
        assert "TRUST_CHECK" in VALID_OPCODE_NAMES

    def test_count(self):
        assert len(VALID_OPCODE_NAMES) >= 40


class TestValidateGrammarSequence:
    """Tests for validate_grammar_sequence()."""

    def test_valid_simple_read_halt(self):
        body = [
            {"op": "READ_PIN", "arg": 2},
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ]
        errors = validate_grammar_sequence(body)
        assert errors == []

    def test_valid_heading_hold(self):
        body = [
            {"op": "READ_PIN", "arg": 2},
            {"op": "PUSH_F32", "value": 45.0},
            {"op": "SUB_F"},
            {"op": "CLAMP_F", "lo": -30.0, "hi": 30.0},
            {"op": "WRITE_PIN", "arg": 4},
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ]
        errors = validate_grammar_sequence(body)
        assert errors == []

    def test_valid_emergency_stop(self):
        body = [
            {"op": "PUSH_F32", "value": 0.0},
            {"op": "CLAMP_F", "lo": -100.0, "hi": 100.0},
            {"op": "WRITE_PIN", "arg": 5},
            {"op": "PUSH_F32", "value": 0.0},
            {"op": "CLAMP_F", "lo": -90.0, "hi": 90.0},
            {"op": "WRITE_PIN", "arg": 4},
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ]
        errors = validate_grammar_sequence(body)
        assert errors == []

    def test_empty_body(self):
        errors = validate_grammar_sequence([])
        assert any("empty" in e.lower() for e in errors)

    def test_missing_op(self):
        body = [{"not_op": "READ_PIN"}]
        errors = validate_grammar_sequence(body)
        assert any("missing" in e.lower() for e in errors)

    def test_unknown_opcode(self):
        body = [
            {"op": "FAKE_OPCODE"},
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ]
        errors = validate_grammar_sequence(body)
        assert any("unknown" in e.lower() for e in errors)

    def test_write_without_clamp(self):
        body = [
            {"op": "PUSH_F32", "value": 50.0},
            {"op": "WRITE_PIN", "arg": 5},
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ]
        errors = validate_grammar_sequence(body)
        assert any("CLAMP_F" in e for e in errors)

    def test_no_halt_at_end(self):
        body = [
            {"op": "READ_PIN", "arg": 2},
            {"op": "POP"},
        ]
        errors = validate_grammar_sequence(body)
        assert any("HALT" in e for e in errors)

    def test_jump_to_zero(self):
        body = [
            {"op": "JUMP", "target": 0},
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ]
        errors = validate_grammar_sequence(body)
        assert any("address 0" in e for e in errors)

    def test_clamp_lo_ge_hi(self):
        body = [
            {"op": "CLAMP_F", "lo": 30.0, "hi": -30.0},
            {"op": "POP"},
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ]
        errors = validate_grammar_sequence(body)
        assert any("lo=" in e for e in errors)

    def test_missing_required_field(self):
        body = [
            {"op": "PUSH_I8"},  # missing "arg"
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ]
        errors = validate_grammar_sequence(body)
        assert any("arg" in e for e in errors)

    def test_multiple_errors(self):
        body = [
            {"op": "FAKE"},
            {"op": "WRITE_PIN", "arg": 5},  # no clamp
        ]
        errors = validate_grammar_sequence(body)
        assert len(errors) >= 2


class TestParseJsonBody:
    """Tests for parse_json_body()."""

    def test_pure_json(self):
        text = '{"name": "test", "body": []}'
        result = parse_json_body(text)
        assert result is not None
        assert result["name"] == "test"

    def test_json_in_code_fence(self):
        text = '```json\n{"name": "test", "body": []}\n```'
        result = parse_json_body(text)
        assert result is not None

    def test_json_with_prose(self):
        text = 'Here is the program:\n{"name": "test", "body": []}\nDone.'
        result = parse_json_body(text)
        assert result is not None

    def test_invalid_json(self):
        text = 'this is not json at all'
        result = parse_json_body(text)
        assert result is None

    def test_empty_string(self):
        result = parse_json_body("")
        assert result is None

    def test_nested_json(self):
        text = '{"name": "test", "body": [{"op": "READ_PIN", "arg": 2}]}'
        result = parse_json_body(text)
        assert result is not None
        assert len(result["body"]) == 1


# ===================================================================
# 3. LLM CLIENT TESTS
# ===================================================================

class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_default_values(self):
        resp = LLMResponse()
        assert resp.text == ""
        assert resp.parsed is None
        assert resp.success is True
        assert resp.error == ""
        assert resp.model == "unknown"
        assert resp.latency_ms == 0.0

    def test_with_values(self):
        resp = LLMResponse(
            text="hello",
            parsed={"key": "val"},
            success=True,
            model="test-model",
            latency_ms=42.0,
        )
        assert resp.text == "hello"
        assert resp.parsed == {"key": "val"}
        assert resp.latency_ms == 42.0


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    def test_is_llm_client(self):
        client = MockLLMClient()
        assert isinstance(client, LLMClient)

    def test_is_available(self):
        client = MockLLMClient()
        assert client.is_available() is True

    def test_canned_response_match(self):
        canned = {"test": "matched"}
        client = MockLLMClient(canned_responses={"heading": canned})
        resp = client.generate("system", "set heading to 45")
        assert resp.success is True
        assert resp.parsed == canned

    def test_canned_response_no_match(self):
        client = MockLLMClient(canned_responses={"xyz": {"test": True}})
        resp = client.generate("system", "something else entirely")
        assert resp.success is True
        # Should return fallback HALT program
        assert resp.parsed is not None
        assert "body" in resp.parsed

    def test_fallback_has_halt(self):
        client = MockLLMClient()
        resp = client.generate("system", "unknown command")
        body = resp.parsed["body"]
        assert body[-1]["op"] == "NOP"
        assert body[-1].get("flags") == "0x80"

    def test_latency(self):
        client = MockLLMClient(latency_ms=123.0)
        resp = client.generate("system", "test")
        assert resp.latency_ms == 123.0

    def test_model_is_mock(self):
        client = MockLLMClient()
        resp = client.generate("system", "test")
        assert resp.model == "mock"

    def test_case_insensitive_match(self):
        canned = {"found": True}
        client = MockLLMClient(canned_responses={"EMERGENCY": canned})
        resp = client.generate("system", "emergency stop now")
        assert resp.parsed == canned


class TestDeterministicLLMClient:
    """Tests for DeterministicLLMClient."""

    def test_is_llm_client(self):
        client = DeterministicLLMClient()
        assert isinstance(client, LLMClient)

    def test_is_available(self):
        client = DeterministicLLMClient()
        assert client.is_available() is True

    def test_emergency_stop(self):
        client = DeterministicLLMClient()
        resp = client.generate("system", "emergency stop")
        assert resp.success is True
        assert resp.parsed is not None
        assert "emergency" in resp.parsed.get("name", "").lower()

    def test_heading_hold(self):
        client = DeterministicLLMClient()
        resp = client.generate("system", "maintain heading at 90")
        assert resp.success is True
        assert resp.parsed is not None

    def test_collision_avoidance(self):
        client = DeterministicLLMClient()
        resp = client.generate("system", "collision avoidance")
        assert resp.success is True
        assert resp.parsed is not None

    def test_default_fallback(self):
        client = DeterministicLLMClient()
        resp = client.generate("system", "do something completely custom")
        assert resp.success is True
        assert resp.parsed is not None

    def test_model_is_deterministic(self):
        client = DeterministicLLMClient()
        resp = client.generate("system", "test")
        assert resp.model == "deterministic"

    def test_response_is_valid_json(self):
        client = DeterministicLLMClient()
        resp = client.generate("system", "heading hold")
        # Should be parseable JSON
        parsed = json.loads(resp.text)
        assert "body" in parsed

    def test_all_outputs_have_halt(self):
        client = DeterministicLLMClient()
        for cmd in [
            "emergency stop",
            "heading hold at 45",
            "collision avoid",
            "station keeping",
            "waypoint follow",
        ]:
            resp = client.generate("system", cmd)
            body = resp.parsed["body"]
            last = body[-1]
            assert last.get("op") == "NOP"
            assert last.get("flags") == "0x80"


class TestSDKBridgeClient:
    """Tests for SDKBridgeClient."""

    def test_is_llm_client(self):
        client = SDKBridgeClient()
        assert isinstance(client, LLMClient)

    def test_fallback_when_unavailable(self):
        client = SDKBridgeClient(
            node_helper_path="/nonexistent/path.js",
        )
        # Should fall back to deterministic
        resp = client.generate("system", "heading hold")
        assert resp.success is True
        assert resp.model in ("deterministic", "unknown")

    def test_custom_fallback(self):
        mock = MockLLMClient()
        client = SDKBridgeClient(fallback_client=mock)
        resp = client.generate("system", "test")
        assert resp.success is True


# ===================================================================
# 4. REFLEX TEMPLATE TESTS
# ===================================================================

class TestTemplateParams:
    """Tests for TemplateParams dataclass."""

    def test_default_values(self):
        p = TemplateParams()
        assert p.target_heading == 45.0
        assert p.rudder_pin == 4
        assert p.throttle_pin == 5
        assert p.compass_pin == 2

    def test_custom_values(self):
        p = TemplateParams(target_heading=270.0, rudder_pin=6)
        assert p.target_heading == 270.0
        assert p.rudder_pin == 6

    def test_as_dict(self):
        p = TemplateParams()
        d = p.as_dict()
        assert isinstance(d, dict)
        assert "target_heading" in d

    def test_frozen(self):
        p = TemplateParams()
        with pytest.raises(AttributeError):
            p.target_heading = 999.0


class TestHeadingHoldTemplate:
    """Tests for heading_hold template."""

    def test_returns_dict(self):
        result = heading_hold()
        assert isinstance(result, dict)

    def test_has_name(self):
        result = heading_hold()
        assert "name" in result
        assert "heading_hold" in result["name"]

    def test_has_body(self):
        result = heading_hold()
        assert "body" in result
        assert len(result["body"]) > 0

    def test_starts_with_read_pin(self):
        result = heading_hold()
        assert result["body"][0]["op"] == "READ_PIN"

    def test_has_sub_f(self):
        result = heading_hold()
        ops = [i["op"] for i in result["body"]]
        assert "SUB_F" in ops

    def test_has_clamp_before_write(self):
        result = heading_hold()
        body = result["body"]
        write_indices = [i for i, b in enumerate(body) if b["op"] == "WRITE_PIN"]
        for idx in write_indices:
            # Check that CLAMP_F appears before this WRITE_PIN
            found_clamp = any(
                b["op"] == "CLAMP_F" for b in body[:idx]
            )
            assert found_clamp, f"WRITE_PIN at {idx} not preceded by CLAMP_F"

    def test_ends_with_halt(self):
        result = heading_hold()
        last = result["body"][-1]
        assert last["op"] == "NOP"
        assert last.get("flags") == "0x80"

    def test_custom_heading(self):
        result = heading_hold(TemplateParams(target_heading=270.0))
        assert "270" in result["name"]

    def test_grammar_valid(self):
        result = heading_hold()
        errors = validate_grammar_sequence(result["body"])
        assert errors == []

    def test_compiles_successfully(self):
        from reflex.compiler import ReflexCompiler
        result = heading_hold()
        compiler = ReflexCompiler()
        bytecode = compiler.compile(result)
        assert len(bytecode) > 0
        assert len(bytecode) % INSTR_SIZE == 0


class TestCollisionAvoidanceTemplate:
    """Tests for collision_avoidance template."""

    def test_returns_dict(self):
        result = collision_avoidance()
        assert isinstance(result, dict)

    def test_has_conditional_branch(self):
        result = collision_avoidance()
        ops = [i["op"] for i in result["body"]]
        assert "LT_F" in ops
        assert "JUMP_IF_FALSE" in ops

    def test_grammar_valid(self):
        result = collision_avoidance()
        errors = validate_grammar_sequence(result["body"])
        assert errors == []

    def test_compiles(self):
        from reflex.compiler import ReflexCompiler
        result = collision_avoidance()
        compiler = ReflexCompiler()
        bytecode = compiler.compile(result)
        assert len(bytecode) > 0

    def test_custom_threshold(self):
        result = collision_avoidance(
            TemplateParams(collision_threshold=10.0)
        )
        # Find the PUSH_F32 with the threshold value
        pushes = [i for i in result["body"] if i["op"] == "PUSH_F32"]
        values = [i.get("value") for i in pushes]
        assert 10.0 in values


class TestWaypointFollowTemplate:
    """Tests for waypoint_follow template."""

    def test_returns_dict(self):
        result = waypoint_follow()
        assert isinstance(result, dict)

    def test_has_abs_f(self):
        result = waypoint_follow()
        ops = [i["op"] for i in result["body"]]
        assert "ABS_F" in ops

    def test_grammar_valid(self):
        result = waypoint_follow()
        errors = validate_grammar_sequence(result["body"])
        assert errors == []


class TestStationKeepingTemplate:
    """Tests for station_keeping template."""

    def test_returns_dict(self):
        result = station_keeping()
        assert isinstance(result, dict)

    def test_has_multiply(self):
        result = station_keeping()
        ops = [i["op"] for i in result["body"]]
        assert "MUL_F" in ops

    def test_grammar_valid(self):
        result = station_keeping()
        errors = validate_grammar_sequence(result["body"])
        assert errors == []

    def test_two_write_pins(self):
        result = station_keeping()
        writes = [i for i in result["body"] if i["op"] == "WRITE_PIN"]
        assert len(writes) == 2


class TestEmergencyStopTemplate:
    """Tests for emergency_stop template."""

    def test_returns_dict(self):
        result = emergency_stop()
        assert isinstance(result, dict)

    def test_has_zero_values(self):
        result = emergency_stop()
        pushes = [i for i in result["body"] if i["op"] == "PUSH_F32"]
        for p in pushes:
            assert p["value"] == 0.0

    def test_grammar_valid(self):
        result = emergency_stop()
        errors = validate_grammar_sequence(result["body"])
        assert errors == []

    def test_compiles_and_validates(self):
        from reflex.compiler import ReflexCompiler
        result = emergency_stop()
        compiler = ReflexCompiler()
        bytecode = compiler.compile(result)
        assert len(bytecode) > 0


class TestReflexTemplatesRegistry:
    """Tests for ReflexTemplates registry."""

    def test_default_templates(self):
        templates = ReflexTemplates()
        assert len(templates) == 5

    def test_list_templates(self):
        templates = ReflexTemplates()
        names = templates.list_templates()
        assert "heading_hold" in names
        assert "emergency_stop" in names
        assert "collision_avoidance" in names

    def test_generate_known(self):
        templates = ReflexTemplates()
        result = templates.generate("emergency_stop")
        assert result is not None
        assert "body" in result

    def test_generate_unknown(self):
        templates = ReflexTemplates()
        result = templates.generate("nonexistent_template")
        assert result is None

    def test_contains(self):
        templates = ReflexTemplates()
        assert "heading_hold" in templates
        assert "nonexistent" not in templates

    def test_get_info(self):
        templates = ReflexTemplates()
        info = templates.get("heading_hold")
        assert info is not None
        assert info.name == "heading_hold"

    def test_extra_templates(self):
        from agent.llm_pipeline.templates import TemplateInfo
        extra = {
            "custom": TemplateInfo(
                name="custom",
                description="test",
                generator=emergency_stop,
                default_params=TemplateParams(),
            )
        }
        templates = ReflexTemplates(extra_templates=extra)
        assert "custom" in templates
        assert len(templates) == 6

    def test_all_templates_compile(self):
        """Every default template should compile successfully."""
        from reflex.compiler import ReflexCompiler
        compiler = ReflexCompiler()
        templates = ReflexTemplates()
        for name in templates.list_templates():
            reflex = templates.generate(name)
            assert reflex is not None, f"Template {name} returned None"
            bytecode = compiler.compile(reflex)
            assert len(bytecode) > 0, f"Template {name} compiled to empty bytecode"


# ===================================================================
# 5. REFLEX SYNTHESIZER TESTS
# ===================================================================

class TestSynthesisMetadata:
    """Tests for SynthesisMetadata."""

    def test_defaults(self):
        m = SynthesisMetadata()
        assert m.strategy_used == "deterministic"
        assert m.attempt_count == 1
        assert m.errors == []
        assert m.warnings == []


class TestSynthesisResult:
    """Tests for SynthesisResult."""

    def test_defaults(self):
        r = SynthesisResult()
        assert r.success is False
        assert r.bytecode == b""
        assert r.reflex_json is None


class TestReflexSynthesizerBasic:
    """Basic tests for ReflexSynthesizer."""

    def test_init_default(self):
        synth = ReflexSynthesizer()
        assert synth.trust_level == 5

    def test_init_custom_trust(self):
        synth = ReflexSynthesizer(trust_level=3)
        assert synth.trust_level == 3

    def test_init_trust_clamped(self):
        synth = ReflexSynthesizer(trust_level=10)
        assert synth.trust_level == 5

    def test_init_negative_trust(self):
        synth = ReflexSynthesizer(trust_level=-1)
        assert synth.trust_level == 0

    def test_synthesize_heading_hold(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("maintain heading at 45 degrees")
        assert result.success is True
        assert len(result.bytecode) > 0

    def test_synthesize_emergency_stop(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("emergency stop")
        assert result.success is True
        assert len(result.bytecode) > 0

    def test_synthesize_collision_avoidance(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("collision avoidance")
        assert result.success is True

    def test_synthesize_station_keeping(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("station keeping")
        assert result.success is True

    def test_synthesize_waypoint(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("waypoint follow")
        assert result.success is True

    def test_synthesize_empty_command(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("")
        assert result.success is False
        assert any("empty" in e.lower() for e in result.metadata.errors)

    def test_synthesize_whitespace_command(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("   ")
        assert result.success is False

    def test_bytecode_multiple_of_8(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("emergency stop")
        assert len(result.bytecode) % INSTR_SIZE == 0

    def test_has_metadata(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("heading hold at 90")
        assert result.metadata is not None
        assert result.metadata.total_latency_ms > 0


class TestReflexSynthesizerWithMockClient:
    """Tests for ReflexSynthesizer with mock LLM client."""

    def test_mock_client_heading(self):
        canned = {
            "name": "mock_heading",
            "intent": "test",
            "body": [
                {"op": "READ_PIN", "arg": 2},
                {"op": "PUSH_F32", "value": 45.0},
                {"op": "SUB_F"},
                {"op": "CLAMP_F", "lo": -30.0, "hi": 30.0},
                {"op": "WRITE_PIN", "arg": 4},
                {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
            ],
        }
        mock = MockLLMClient(canned_responses={"heading": canned})
        synth = ReflexSynthesizer(llm_client=mock)
        # Template matching should still kick in before LLM
        result = synth.synthesize("heading at 45")
        assert result.success is True

    def test_custom_command_with_mock(self):
        """When no template matches, mock client is used."""
        canned = {
            "name": "custom",
            "intent": "custom",
            "body": [
                {"op": "READ_PIN", "arg": 3},
                {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
            ],
        }
        mock = MockLLMClient(canned_responses={"xyzabc": canned})
        synth = ReflexSynthesizer(llm_client=mock)
        result = synth.synthesize("xyzabc special command")
        assert result.success is True


class TestReflexSynthesizerTrustLevels:
    """Tests for trust level handling."""

    def test_l2_heading_hold(self):
        synth = ReflexSynthesizer(trust_level=2)
        result = synth.synthesize("maintain heading at 45 degrees")
        # L2 allows WRITE_PIN and JUMP — should compile
        assert result.success is True

    def test_override_trust_per_call(self):
        synth = ReflexSynthesizer(trust_level=0)
        result = synth.synthesize("emergency stop", trust_level=5)
        # Template match works regardless, but trust_level is overridden
        assert len(result.bytecode) > 0


class TestReflexSynthesizerErrorHandling:
    """Tests for error handling in the synthesizer."""

    def test_invalid_json_from_llm(self):
        """LLM returns non-JSON — should handle gracefully."""
        mock = MockLLMClient(canned_responses={"test": {"not": "a reflex"}})
        synth = ReflexSynthesizer(llm_client=mock)
        result = synth.synthesize("test command")
        # Template will likely match or deterministic fallback used
        assert result.metadata is not None

    def test_metadata_has_strategy(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("heading hold")
        assert result.metadata.strategy_used in (
            "template", "llm", "deterministic"
        )


class TestReflexSynthesizerIntegration:
    """Integration tests: NL → template → bytecode → safety check."""

    def test_full_pipeline_emergency(self):
        synth = ReflexSynthesizer(trust_level=5)
        result = synth.synthesize("emergency stop all actuators")
        assert result.success is True
        assert len(result.bytecode) > 0
        assert len(result.bytecode) % INSTR_SIZE == 0
        assert result.metadata.grammar_valid is True

    def test_full_pipeline_heading_hold(self):
        synth = ReflexSynthesizer(trust_level=5)
        result = synth.synthesize(
            "maintain heading at 270 degrees with PID control"
        )
        assert result.success is True
        assert result.reflex_json is not None
        assert "body" in result.reflex_json
        assert len(result.reflex_json["body"]) > 0

    def test_full_pipeline_collision(self):
        synth = ReflexSynthesizer(trust_level=5)
        result = synth.synthesize(
            "collision avoidance if obstacle within 3 meters"
        )
        assert result.success is True

    def test_result_json_roundtrip(self):
        """Reflex JSON from result should be compilable."""
        synth = ReflexSynthesizer()
        result = synth.synthesize("emergency stop")
        if result.reflex_json:
            from reflex.compiler import ReflexCompiler
            compiler = ReflexCompiler()
            # Should not raise
            bytecode = compiler.compile(result.reflex_json)
            assert len(bytecode) > 0

    def test_bytecode_contains_halt(self):
        """All synthesized bytecode should end with HALT."""
        synth = ReflexSynthesizer()
        commands = [
            "emergency stop",
            "heading hold at 45",
            "collision avoidance",
            "station keeping",
            "waypoint follow",
        ]
        for cmd in commands:
            result = synth.synthesize(cmd)
            if len(result.bytecode) >= INSTR_SIZE:
                # Last instruction should be HALT (NOP + SYSCALL)
                last_instr = result.bytecode[-INSTR_SIZE:]
                opcode, flags, op1, op2 = unpack_instruction(
                    last_instr, 0
                )
                # HALT = NOP(0x00) + SYSCALL(0x80) + operand2=1
                assert opcode == 0x00, f"{cmd}: last opcode = {opcode:#x}"
                assert flags & 0x80, f"{cmd}: last flags missing SYSCALL"

    def test_multiple_synthesis_calls(self):
        """Synthesizer should be reusable for multiple calls."""
        synth = ReflexSynthesizer()
        for cmd in ["heading hold", "emergency stop", "collision avoid"]:
            result = synth.synthesize(cmd)
            assert result.success is True


# ===================================================================
# 6. EDGE CASE TESTS
# ===================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_very_long_command(self):
        synth = ReflexSynthesizer()
        cmd = "heading " * 1000
        result = synth.synthesize(cmd)
        # Should still work (template match on "heading")
        assert result.success is True

    def test_command_with_numbers(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("maintain heading at 127.5 degrees")
        assert result.success is True

    def test_command_with_negative_numbers(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("set rudder to -45 degrees")
        assert result.success is True

    def test_command_mixed_case(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("EMERGENCY STOP")
        assert result.success is True

    def test_command_with_special_chars(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("heading hold @ 45° with ±5° tolerance!")
        # Should work via template match
        assert result.success is True

    def test_none_command_handled(self):
        synth = ReflexSynthesizer()
        # Python will pass None through; the check should handle it
        result = synth.synthesize(None)  # type: ignore[arg-type]
        assert result.success is False

    def test_unicode_command(self):
        synth = ReflexSynthesizer()
        result = synth.synthesize("紧急停止")  # "emergency stop" in Chinese
        # Won't match template, but deterministic fallback should work
        assert result.reflex_json is not None or result.success is False


# ===================================================================
# 7. OPCODE COVERAGE TESTS (ensuring all opcodes are represented)
# ===================================================================

class TestOpcodeCoverage:
    """Ensure all opcodes are documented in prompts and grammar."""

    def test_all_core_opcodes_in_strategy_a(self):
        prompt = strategy_a_comprehensive.system_prompt
        for code, name in OPCODE_NAMES.items():
            if code <= 0x1F:
                assert name in prompt, f"Core opcode {name} not in Strategy A"

    def test_all_a2a_opcodes_in_strategy_a(self):
        prompt = strategy_a_comprehensive.system_prompt
        for code, name in OPCODE_NAMES.items():
            if 0x20 <= code <= 0x56:
                assert name in prompt, f"A2A opcode {name} not in Strategy A"

    def test_valid_opcode_names_complete(self):
        """VALID_OPCODE_NAMES should cover all opcodes from shared/opcodes.py."""
        for code, name in OPCODE_NAMES.items():
            assert name in VALID_OPCODE_NAMES, f"{name} not in VALID_OPCODE_NAMES"

    def test_opcode_count_matches(self):
        assert OPCODE_CORE_COUNT == 32
        assert OPCODE_A2A_COUNT == 29
        assert OPCODE_TOTAL_COUNT == 61
