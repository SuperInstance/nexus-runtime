"""NEXUS Skill Loading System — Dynamic marine operation skill cartridges.

The "I-know-kung-fu" concept from the Lucineer ecosystem: pre-built skill
cartridges that can be loaded into agents to instantly gain capabilities.
For NEXUS, these are marine operation patterns compiled to AAB bytecode.

Usage:
    from agent.skill_system import SkillLoader, SkillRegistry
    from agent.skill_system.builtin_skills import BUILTIN_SKILLS
"""

from agent.skill_system.cartridge import SkillCartridge, SkillParameter
from agent.skill_system.cartridge_builder import CartridgeBuilder, ValidationResult
from agent.skill_system.skill_loader import SkillLoader
from agent.skill_system.skill_registry import SkillRegistry

__all__ = [
    "SkillCartridge",
    "SkillParameter",
    "CartridgeBuilder",
    "ValidationResult",
    "SkillLoader",
    "SkillRegistry",
]
