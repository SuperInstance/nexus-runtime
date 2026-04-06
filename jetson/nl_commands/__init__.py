"""
NL Commands — Natural Language Command Interface for NEXUS Marine Robotics.

This package provides:
- **parser**: Tokenization, entity extraction, and command normalization
- **intent**: Intent recognition and classification
- **executor**: Command execution engine
- **dialogue**: Multi-turn dialogue management
- **validation**: Command validation and safety checks
"""

from .parser import Token, Entity, ParseTree, NLParser
from .intent import Intent, IntentType, IntentRecognizer
from .executor import (
    Command, CommandPriority, ExecutionResult,
    ImpactAssessment, ExecutionStep, CommandExecutor,
)
from .dialogue import DialogueState, DialogueManager
from .validation import ValidationResult, RiskLevel, SafeAlternative, CommandValidator

__all__ = [
    # parser
    "Token", "Entity", "ParseTree", "NLParser",
    # intent
    "Intent", "IntentType", "IntentRecognizer",
    # executor
    "Command", "CommandPriority", "ExecutionResult",
    "ImpactAssessment", "ExecutionStep", "CommandExecutor",
    # dialogue
    "DialogueState", "DialogueManager",
    # validation
    "ValidationResult", "RiskLevel", "SafeAlternative", "CommandValidator",
]
