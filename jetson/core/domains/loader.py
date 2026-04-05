"""NEXUS Domain Loader — load, validate, apply, and switch domain profiles."""

from __future__ import annotations

import copy
import json
from typing import Any

from .profile import DomainProfile, BUILT_IN_PROFILES


class DomainValidationError(Exception):
    """Raised when a domain profile fails validation."""
    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


class DomainLoader:
    """Load, validate, and manage domain profiles."""

    def __init__(self) -> None:
        self._profiles: dict[str, DomainProfile] = {}
        self._active: DomainProfile | None = None
        # Register built-in profiles
        for name, factory in BUILT_IN_PROFILES.items():
            self._profiles[name] = factory()

    def load_builtin(self, domain_id: str) -> DomainProfile:
        """Load a built-in profile by ID."""
        if domain_id not in BUILT_IN_PROFILES:
            raise ValueError(f"Unknown built-in domain: {domain_id}. Available: {list(BUILT_IN_PROFILES.keys())}")
        profile = BUILT_IN_PROFILES[domain_id]()
        self._profiles[domain_id] = profile
        return profile

    def load_from_dict(self, data: dict[str, Any], domain_id: str | None = None) -> DomainProfile:
        """Load a profile from a dictionary."""
        profile = DomainProfile.from_dict(data)
        if domain_id:
            profile.domain_id = domain_id
        self._profiles[profile.domain_id] = profile
        return profile

    def load_from_json(self, json_str: str) -> DomainProfile:
        """Load a profile from a JSON string."""
        data = json.loads(json_str)
        return self.load_from_dict(data)

    def load_from_file(self, filepath: str) -> DomainProfile:
        """Load a profile from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return self.load_from_dict(data)

    def validate(self, profile: DomainProfile) -> list[str]:
        """Validate a domain profile. Returns list of errors (empty = valid)."""
        errors = []
        if not profile.domain_id:
            errors.append("domain_id is required")
        if profile.max_speed < 0:
            errors.append("max_speed must be >= 0")
        if profile.max_throttle_pct < 0 or profile.max_throttle_pct > 100:
            errors.append("max_throttle_pct must be 0-100")
        if profile.proximity_limit_m < 0:
            errors.append("proximity_limit_m must be >= 0")
        if profile.initial_trust < 0 or profile.initial_trust > 1:
            errors.append("initial_trust must be 0.0-1.0")
        if profile.trust_decay_rate < 0:
            errors.append("trust_decay_rate must be >= 0")
        if profile.min_temperature_c >= profile.max_temperature_c:
            errors.append("min_temperature_c must be < max_temperature_c")
        if profile.max_humidity_pct < 0 or profile.max_humidity_pct > 100:
            errors.append("max_humidity_pct must be 0-100")
        # Check actuator ranges
        for name, (lo, hi) in profile.actuator_ranges.items():
            if lo >= hi:
                errors.append(f"actuator_range '{name}': min ({lo}) must be < max ({hi})")
        # Check no-go zones have bounds
        for zone in profile.no_go_zones:
            if "bounds" not in zone:
                errors.append(f"no_go_zone missing 'bounds': {zone.get('name', '?')}")
        return errors

    def activate(self, domain_id: str) -> DomainProfile:
        """Activate a profile. Raises DomainValidationError if invalid."""
        if domain_id not in self._profiles:
            raise ValueError(f"Domain '{domain_id}' not loaded. Available: {list(self._profiles.keys())}")
        profile = self._profiles[domain_id]
        errors = self.validate(profile)
        if errors:
            raise DomainValidationError(domain_id, "; ".join(errors))
        self._active = copy.deepcopy(profile)
        return self._active

    @property
    def active(self) -> DomainProfile | None:
        return self._active

    @property
    def available_domains(self) -> list[str]:
        return list(self._profiles.keys())

    def diff(self, domain_a: str, domain_b: str) -> dict[str, tuple[Any, Any]]:
        """Compare two profiles. Returns dict of field: (value_a, value_b)."""
        if domain_a not in self._profiles or domain_b not in self._profiles:
            raise ValueError("Both domains must be loaded")
        pa, pb = self._profiles[domain_a], self._profiles[domain_b]
        differences = {}
        for field in DomainProfile.__dataclass_fields__:
            va = getattr(pa, field)
            vb = getattr(pb, field)
            if va != vb:
                differences[field] = (va, vb)
        return differences

    def to_json(self, domain_id: str) -> str:
        """Serialize a profile to JSON."""
        if domain_id not in self._profiles:
            raise ValueError(f"Domain '{domain_id}' not loaded")
        return json.dumps(self._profiles[domain_id].to_dict(), indent=2, default=str)
