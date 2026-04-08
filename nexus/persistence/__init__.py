"""NEXUS Persistence — pluggable state persistence for crash recovery.

Provides JSON-file-backed storage for trust profiles, agent records,
and task state. Data survives process restarts.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class JSONStore:
    """File-backed JSON storage for NEXUS state.
    
    Usage::
    
        store = JSONStore("/var/lib/nexus/state.json")
        store.set("trust:AUV-001:AUV-002", {"score": 0.85})
        score = store.get("trust:AUV-001:AUV-002")
        store.save()  # persist to disk
    
    Thread safety: callers should use external locking if accessing
    from multiple threads.
    """
    
    def __init__(self, path: str | Path, autosave: bool = False) -> None:
        self._path = Path(path)
        self._autosave = autosave
        self._data: Dict[str, Any] = {}
        self._load()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key."""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value (in-memory until save())."""
        self._data[key] = value
        if self._autosave:
            self.save()
    
    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        if key in self._data:
            del self._data[key]
            if self._autosave:
                self.save()
            return True
        return False
    
    def keys(self) -> list[str]:
        """Return all stored keys."""
        return list(self._data.keys())
    
    def save(self) -> None:
        """Persist current state to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
            tmp.replace(self._path)
            logger.debug("State saved to %s (%d keys)", self._path, len(self._data))
        except Exception:
            logger.exception("Failed to save state to %s", self._path)
    
    def _load(self) -> None:
        """Load state from disk if file exists."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
                logger.info("Loaded %d keys from %s", len(self._data), self._path)
            except Exception:
                logger.exception("Failed to load state from %s", self._path)
                self._data = {}
    
    def clear(self) -> None:
        """Clear all in-memory data."""
        self._data.clear()
