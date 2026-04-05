"""NEXUS Skill Registry — Central registry for tracking available and deployed skills.

The SkillRegistry maintains a global view of all known skill cartridges,
their registration status, and deployment state. It provides search and
filtering capabilities to find compatible skills for a given context.

Unlike the SkillLoader (which manages on-demand loading from disk),
the Registry is an in-memory catalog that can be populated from any source:
built-in skills, network-discovered skills, or user-installed cartridges.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable

from agent.skill_system.cartridge import SkillCartridge


@dataclass
class RegistrationRecord:
    """Record tracking a registered skill cartridge.

    Attributes:
        skill_id: Unique registration identifier (UUID).
        cartridge: The registered SkillCartridge.
        status: Current status — "available", "deployed", "revoked", "expired".
        registered_at: ISO-8601 timestamp of registration.
        source: Where this skill came from (e.g., "builtin", "network", "file").
    """

    skill_id: str
    cartridge: SkillCartridge
    status: str = "available"
    registered_at: str = ""
    source: str = "unknown"


class SkillRegistry:
    """Tracks all available skills and their deployment status.

    The registry acts as a central catalog for skill discovery. Skills
    can be registered from any source (built-in, file, network) and
    queried by domain, trust level, name, or custom filters.

    Example:
        registry = SkillRegistry()
        skill_id = registry.register(cartridge)
        compatible = registry.get_compatible(trust_level=2, domain="marine")
        registry.deregister(skill_id)
    """

    def __init__(self) -> None:
        """Initialize an empty skill registry."""
        self._records: dict[str, RegistrationRecord] = {}
        self._name_index: dict[str, str] = {}  # name -> skill_id

    def register(
        self,
        cartridge: SkillCartridge,
        source: str = "unknown",
    ) -> str:
        """Register a skill cartridge in the registry.

        If a skill with the same name already exists, the old one is
        replaced (updated).

        Args:
            cartridge: The SkillCartridge to register.
            source: Provenance source (e.g., "builtin", "network", "file").

        Returns:
            Registration ID (UUID string).

        Raises:
            ValueError: If the cartridge has no name.
        """
        if not cartridge.name:
            raise ValueError("Cannot register a cartridge without a name")

        # Generate registration ID
        skill_id = str(uuid.uuid4())

        # Deregister existing skill with same name (if any)
        if cartridge.name in self._name_index:
            old_id = self._name_index[cartridge.name]
            if old_id in self._records:
                del self._records[old_id]

        # Create registration record
        from datetime import datetime, timezone
        record = RegistrationRecord(
            skill_id=skill_id,
            cartridge=cartridge,
            status="available",
            registered_at=datetime.now(timezone.utc).isoformat(),
            source=source,
        )

        self._records[skill_id] = record
        self._name_index[cartridge.name] = skill_id

        return skill_id

    def deregister(self, skill_id: str) -> None:
        """Remove a skill from the registry.

        Args:
            skill_id: Registration ID of the skill to remove.

        Raises:
            KeyError: If the skill_id is not found.
        """
        if skill_id not in self._records:
            raise KeyError(f"Skill ID '{skill_id}' not found in registry")

        record = self._records[skill_id]
        # Remove from name index
        if record.cartridge.name in self._name_index:
            del self._name_index[record.cartridge.name]

        del self._records[skill_id]

    def get(self, skill_id: str) -> RegistrationRecord | None:
        """Get a registration record by ID.

        Args:
            skill_id: Registration ID to look up.

        Returns:
            RegistrationRecord if found, None otherwise.
        """
        return self._records.get(skill_id)

    def get_by_name(self, name: str) -> RegistrationRecord | None:
        """Get a registration record by skill name.

        Args:
            name: Skill name to look up.

        Returns:
            RegistrationRecord if found, None otherwise.
        """
        skill_id = self._name_index.get(name)
        if skill_id is None:
            return None
        return self._records.get(skill_id)

    def find_by_domain(self, domain: str) -> list[SkillCartridge]:
        """Find all registered skills in a given domain.

        Args:
            domain: Domain to filter by (e.g., "marine", "aerial").

        Returns:
            List of matching SkillCartridge instances.
        """
        return [
            record.cartridge
            for record in self._records.values()
            if record.cartridge.domain == domain
        ]

    def find_by_trust_level(self, max_trust: int) -> list[SkillCartridge]:
        """Find all registered skills requiring at most the given trust level.

        Args:
            max_trust: Maximum trust level to include (0-5).

        Returns:
            List of SkillCartridge instances with trust_required <= max_trust.
        """
        return [
            record.cartridge
            for record in self._records.values()
            if record.cartridge.trust_required <= max_trust
        ]

    def get_compatible(
        self,
        trust_level: int,
        domain: str | None = None,
    ) -> list[SkillCartridge]:
        """Find all skills compatible with current trust level and domain.

        A skill is compatible if its trust_required <= trust_level AND
        (domain is None OR its domain matches).

        Args:
            trust_level: Current system trust level (0-5).
            domain: Optional domain filter.

        Returns:
            List of compatible SkillCartridge instances, sorted by trust_required.
        """
        results = []
        for record in self._records.values():
            cartridge = record.cartridge
            if cartridge.trust_required > trust_level:
                continue
            if domain is not None and cartridge.domain != domain:
                continue
            results.append(cartridge)

        # Sort by trust_required (lower first = easier to deploy)
        return sorted(results, key=lambda c: c.trust_required)

    def list_all(self) -> list[SkillCartridge]:
        """List all registered skill cartridges.

        Returns:
            List of all registered SkillCartridge instances.
        """
        return [record.cartridge for record in self._records.values()]

    def list_names(self) -> list[str]:
        """List all registered skill names.

        Returns:
            Sorted list of unique skill names.
        """
        return sorted(self._name_index.keys())

    def count(self) -> int:
        """Return the total number of registered skills.

        Returns:
            Number of registered cartridges.
        """
        return len(self._records)

    def has_skill(self, name: str) -> bool:
        """Check if a skill is registered by name.

        Args:
            name: Skill name to check.

        Returns:
            True if the skill is registered.
        """
        return name in self._name_index

    def update_status(self, skill_id: str, status: str) -> None:
        """Update the deployment status of a registered skill.

        Args:
            skill_id: Registration ID of the skill.
            status: New status — "available", "deployed", "revoked", "expired".

        Raises:
            KeyError: If the skill_id is not found.
            ValueError: If the status is not recognized.
        """
        valid_statuses = {"available", "deployed", "revoked", "expired"}
        if status not in valid_statuses:
            raise ValueError(
                f"Invalid status '{status}', must be one of: {valid_statuses}"
            )

        if skill_id not in self._records:
            raise KeyError(f"Skill ID '{skill_id}' not found in registry")

        self._records[skill_id].status = status

    def find_by_source(self, source: str) -> list[SkillCartridge]:
        """Find all skills registered from a specific source.

        Args:
            source: Source identifier (e.g., "builtin", "network").

        Returns:
            List of matching SkillCartridge instances.
        """
        return [
            record.cartridge
            for record in self._records.values()
            if record.source == source
        ]

    def clear(self) -> None:
        """Remove all registered skills from the registry."""
        self._records.clear()
        self._name_index.clear()
