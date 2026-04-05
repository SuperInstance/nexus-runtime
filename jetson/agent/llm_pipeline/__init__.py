"""NEXUS LLM Inference Pipeline — reflex synthesis and natural language control.

Provides:
  - PromptTemplate / strategies for teaching the LLM bytecode format
  - GBNF grammar for constraining LLM output to valid bytecode sequences
  - ReflexSynthesizer: NL command → validated bytecode pipeline
  - ReflexTemplates: pre-built parameterized bytecode programs
  - LLMClient: mock/deterministic fallback (no external LLM required)
"""

from agent.llm_pipeline.prompts import (
    PromptStrategy,
    PromptTemplate,
    strategy_a_comprehensive,
    strategy_b_grammar,
    strategy_c_fewshot,
    best_prompt,
)
from agent.llm_pipeline.grammar import NEXUS_GBNF_GRAMMAR, validate_grammar_sequence
from agent.llm_pipeline.synthesizer import ReflexSynthesizer, SynthesisResult
from agent.llm_pipeline.templates import (
    ReflexTemplates,
    TemplateParams,
    KNOWN_TEMPLATES,
)
from agent.llm_pipeline.llm_client import (
    LLMClient,
    LLMResponse,
    MockLLMClient,
    DeterministicLLMClient,
)

__all__ = [
    "PromptStrategy",
    "PromptTemplate",
    "strategy_a_comprehensive",
    "strategy_b_grammar",
    "strategy_c_fewshot",
    "best_prompt",
    "NEXUS_GBNF_GRAMMAR",
    "validate_grammar_sequence",
    "ReflexSynthesizer",
    "SynthesisResult",
    "ReflexTemplates",
    "TemplateParams",
    "KNOWN_TEMPLATES",
    "LLMClient",
    "LLMResponse",
    "MockLLMClient",
    "DeterministicLLMClient",
]
