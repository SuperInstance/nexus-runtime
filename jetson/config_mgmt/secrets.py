"""Secrets management with storage, rotation, strength validation, and masking."""

from __future__ import annotations

import hashlib
import os
import re
import string
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SecretEntry:
    """Represents a stored secret with metadata."""
    name: str
    value: str
    version: int = 1
    created: float = 0.0
    last_accessed: float = 0.0


class SecretManager:
    """Manage secrets — store, retrieve, rotate, mask, validate strength, and search."""

    def __init__(self) -> None:
        self._secrets: Dict[str, SecretEntry] = {}

    def store_secret(self, name: str, value: str) -> int:
        """Store a secret. Returns version number (1 for new, incremented for existing)."""
        now = time.time()
        if name in self._secrets:
            existing = self._secrets[name]
            existing.value = value
            existing.version += 1
            existing.last_accessed = now
            return existing.version
        else:
            entry = SecretEntry(
                name=name,
                value=value,
                version=1,
                created=now,
                last_accessed=now,
            )
            self._secrets[name] = entry
            return 1

    def retrieve_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret value by name. Returns None if not found."""
        entry = self._secrets.get(name)
        if entry is None:
            return None
        entry.last_accessed = time.time()
        return entry.value

    def rotate_secret(self, name: str) -> int:
        """Rotate a secret by generating a new random value. Returns new version.

        Raises KeyError if secret not found.
        """
        if name not in self._secrets:
            raise KeyError(f"Secret '{name}' not found")
        entry = self._secrets[name]
        new_value = self._generate_random_value()
        entry.value = new_value
        entry.version += 1
        entry.last_accessed = time.time()
        return entry.version

    def list_secrets(self) -> List[str]:
        """List all secret names."""
        return sorted(self._secrets.keys())

    def delete_secret(self, name: str) -> None:
        """Delete a secret by name. Raises KeyError if not found."""
        if name not in self._secrets:
            raise KeyError(f"Secret '{name}' not found")
        del self._secrets[name]

    def get_secret_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a secret without exposing the value. Returns None if not found."""
        entry = self._secrets.get(name)
        if entry is None:
            return None
        return {
            "name": entry.name,
            "version": entry.version,
            "created": entry.created,
            "last_accessed": entry.last_accessed,
            "length": len(entry.value),
        }

    def mask_secret(self, value: str, visible_chars: int = 4, mask_char: str = "*") -> str:
        """Mask a secret value, showing only the first visible_chars characters."""
        if len(value) <= visible_chars:
            return mask_char * len(value)
        return value[:visible_chars] + mask_char * (len(value) - visible_chars)

    def validate_secret_strength(self, value: str) -> int:
        """Validate secret strength. Returns a score from 0 to 100.

        Scoring:
        - length (up to 30 points): 0-7 chars=5, 8-11=10, 12-15=20, 16+=30
        - uppercase (15 points)
        - lowercase (15 points)
        - digits (15 points)
        - special chars (15 points)
        - no repeated chars bonus (10 points)
        """
        score = 0
        length = len(value)

        # Length scoring
        if length == 0:
            return 0
        elif length < 8:
            score += 5
        elif length < 12:
            score += 10
        elif length < 16:
            score += 20
        else:
            score += 30

        # Character class checks
        has_upper = bool(re.search(r'[A-Z]', value))
        has_lower = bool(re.search(r'[a-z]', value))
        has_digit = bool(re.search(r'[0-9]', value))
        has_special = bool(re.search(r'[^A-Za-z0-9]', value))

        if has_upper:
            score += 15
        if has_lower:
            score += 15
        if has_digit:
            score += 15
        if has_special:
            score += 15

        # No repeated sequences bonus
        has_repeat = bool(re.search(r'(.)\1{2,}', value))
        if not has_repeat:
            score += 10

        return min(score, 100)

    def search_secrets(self, pattern: str) -> List[Dict[str, Any]]:
        """Search secrets by name pattern. Returns matching entries (metadata only, no values)."""
        results = []
        for name, entry in self._secrets.items():
            if re.search(pattern, name, re.IGNORECASE):
                results.append(self.get_secret_metadata(name))
        return results

    def _generate_random_value(self, length: int = 32) -> str:
        """Generate a cryptographically random secret value."""
        import secrets as _secrets
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(_secrets.choice(alphabet) for _ in range(length))
