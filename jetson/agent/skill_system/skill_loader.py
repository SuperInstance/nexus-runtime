"""NEXUS Skill Loader — Runtime loading and management of skill cartridges.

The SkillLoader manages the lifecycle of skill cartridges:
  - Loading cartridges from a directory of JSON files
  - Tracking which skills are currently loaded in memory
  - Checking trust level permissions before loading
  - Listing available and loaded skills

Skills are loaded on-demand by name. The loader discovers available
cartridges by scanning the cartridge directory for .json files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from agent.skill_system.cartridge import SkillCartridge
from agent.skill_system.cartridge_builder import CartridgeBuilder


class SkillLoader:
    """Load and manage skill cartridges at runtime.

    The loader scans a cartridge directory for .json files and provides
    on-demand loading with trust level gating.

    Example:
        loader = SkillLoader("/path/to/cartridges")

        # List all available skills
        available = loader.list_available()

        # Load a skill (checks trust level)
        if loader.check_trust("surface_navigation", current_trust=2):
            skill = loader.load("surface_navigation")
            print(skill.summary())
    """

    def __init__(self, cartridge_dir: str) -> None:
        """Initialize the skill loader.

        Args:
            cartridge_dir: Path to directory containing cartridge JSON files.
        """
        self.cartridge_dir = Path(cartridge_dir)
        self.loaded_skills: dict[str, SkillCartridge] = {}
        self._builder = CartridgeBuilder()
        self._available_cache: list[str] | None = None

    def load(self, skill_name: str) -> SkillCartridge:
        """Load a skill cartridge by name from the cartridge directory.

        Searches for a JSON file named '{skill_name}.json' in the
        cartridge directory, parses it, and caches it in memory.

        Args:
            skill_name: Name of the skill to load.

        Returns:
            The loaded SkillCartridge.

        Raises:
            FileNotFoundError: If the cartridge file does not exist.
            ValueError: If the cartridge file is invalid.
        """
        # Check if already loaded
        if skill_name in self.loaded_skills:
            return self.loaded_skills[skill_name]

        # Find the cartridge file
        cartridge_path = self.cartridge_dir / f"{skill_name}.json"
        if not cartridge_path.exists():
            raise FileNotFoundError(
                f"Skill cartridge '{skill_name}' not found in {self.cartridge_dir}"
            )

        # Load from JSON
        cartridge = self._builder.from_json(str(cartridge_path))

        # Validate the name matches
        if cartridge.name != skill_name:
            # Allow it but warn (the file name and internal name differ)
            pass

        # Cache the loaded cartridge
        self.loaded_skills[skill_name] = cartridge
        self._available_cache = None  # Invalidate cache

        return cartridge

    def load_all(self) -> dict[str, SkillCartridge]:
        """Load all available cartridges from the cartridge directory.

        Returns:
            Dictionary mapping skill names to loaded SkillCartridge instances.
        """
        available = self.list_available()
        for name in available:
            try:
                self.load(name)
            except (FileNotFoundError, ValueError) as e:
                # Skip invalid cartridges
                continue
        return dict(self.loaded_skills)

    def load_cartridge_direct(self, cartridge: SkillCartridge) -> SkillCartridge:
        """Load a cartridge directly (not from file system).

        Useful for loading built-in cartridges or cartridges received
        over the network.

        Args:
            cartridge: The SkillCartridge to load.

        Returns:
            The loaded cartridge (same instance, cached by name).
        """
        self.loaded_skills[cartridge.name] = cartridge
        self._available_cache = None
        return cartridge

    def unload(self, skill_name: str) -> None:
        """Unload a skill cartridge from memory.

        Args:
            skill_name: Name of the skill to unload.

        Raises:
            KeyError: If the skill is not currently loaded.
        """
        if skill_name not in self.loaded_skills:
            raise KeyError(
                f"Skill '{skill_name}' is not loaded; "
                f"loaded skills: {list(self.loaded_skills.keys())}"
            )
        del self.loaded_skills[skill_name]
        self._available_cache = None

    def list_available(self) -> list[str]:
        """List all available cartridge names in the cartridge directory.

        Scans the directory for .json files and returns their stem names.

        Returns:
            Sorted list of available skill names.
        """
        if self._available_cache is not None:
            return self._available_cache

        if not self.cartridge_dir.exists():
            self._available_cache = []
            return self._available_cache

        available = []
        for path in sorted(self.cartridge_dir.glob("*.json")):
            # Try to read the name from the JSON file
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "name" in data:
                    available.append(data["name"])
                else:
                    available.append(path.stem)
            except (json.JSONDecodeError, OSError):
                available.append(path.stem)

        self._available_cache = sorted(set(available))
        return self._available_cache

    def list_loaded(self) -> list[str]:
        """List all currently loaded skill names.

        Returns:
            Sorted list of loaded skill names.
        """
        return sorted(self.loaded_skills.keys())

    def get_skill(self, skill_name: str) -> SkillCartridge | None:
        """Get a loaded skill cartridge by name.

        Unlike load(), this only returns already-loaded cartridges
        and does not trigger file I/O.

        Args:
            skill_name: Name of the skill to retrieve.

        Returns:
            The SkillCartridge if loaded, None otherwise.
        """
        return self.loaded_skills.get(skill_name)

    def check_trust(self, skill_name: str, current_trust_level: int) -> bool:
        """Check if current trust level permits loading this skill.

        The skill must already be loaded (or loadable) for trust checking.

        Args:
            skill_name: Name of the skill to check.
            current_trust_level: Current system trust level (0-5).

        Returns:
            True if the current trust level meets the skill's requirement.
        """
        cartridge = self.get_skill(skill_name)
        if cartridge is None:
            # Try to load it
            try:
                cartridge = self.load(skill_name)
            except (FileNotFoundError, ValueError):
                return False

        return current_trust_level >= cartridge.trust_required

    def is_loaded(self, skill_name: str) -> bool:
        """Check if a skill is currently loaded in memory.

        Args:
            skill_name: Name of the skill to check.

        Returns:
            True if the skill is loaded.
        """
        return skill_name in self.loaded_skills

    def get_by_domain(self, domain: str) -> list[SkillCartridge]:
        """Get all loaded skills matching a domain.

        Args:
            domain: Domain to filter by (e.g., "marine").

        Returns:
            List of matching SkillCartridge instances.
        """
        return [
            c for c in self.loaded_skills.values()
            if c.domain == domain
        ]

    def get_by_trust_level(self, max_trust: int) -> list[SkillCartridge]:
        """Get all loaded skills that require at most the given trust level.

        Args:
            max_trust: Maximum trust level to include.

        Returns:
            List of SkillCartridge instances with trust_required <= max_trust.
        """
        return [
            c for c in self.loaded_skills.values()
            if c.trust_required <= max_trust
        ]

    def reload(self, skill_name: str) -> SkillCartridge:
        """Reload a skill cartridge from disk (refresh from file).

        Args:
            skill_name: Name of the skill to reload.

        Returns:
            The freshly loaded SkillCartridge.
        """
        # Unload first
        if skill_name in self.loaded_skills:
            del self.loaded_skills[skill_name]
        self._available_cache = None

        # Re-load from disk
        return self.load(skill_name)

    def validate_loaded(self, skill_name: str) -> bool:
        """Validate a loaded skill's bytecode against the safety pipeline.

        Args:
            skill_name: Name of the loaded skill to validate.

        Returns:
            True if the skill passes validation.
        """
        cartridge = self.get_skill(skill_name)
        if cartridge is None:
            return False

        result = self._builder.validate(cartridge)
        return result.valid
