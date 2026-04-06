"""Hot configuration reload with file watching, change detection, and fingerprinting."""

from __future__ import annotations

import hashlib
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .loader import ConfigLoader


@dataclass
class ReloadEvent:
    """Describes a configuration reload event."""
    timestamp: float
    config_name: str
    changes: Dict[str, Any] = field(default_factory=dict)
    source: str = "file"


class ConfigWatcher:
    """Watch configuration files for changes and manage hot reload."""

    def __init__(self) -> None:
        self._loader = ConfigLoader()
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._fingerprints: Dict[str, str] = {}
        self._change_history: Dict[str, List[ReloadEvent]] = {}
        self._subscriptions: Dict[str, List[Callable[[ReloadEvent], None]]] = {}
        self._watched_paths: Dict[str, Path] = {}
        self._next_sub_id = 0
        self._sub_lookup: Dict[int, tuple] = {}  # sub_id -> (config_name, callback)
        self._mtimes: Dict[str, float] = {}

    def watch(self, config_path: Union[str, Path], callback: Optional[Callable[[ReloadEvent], None]] = None) -> str:
        """Start watching a configuration file. Returns config name (file stem).

        If callback is provided, it is registered as a subscriber.
        """
        config_path = Path(config_path)
        config_name = config_path.stem

        if config_path.exists():
            self._configs[config_name] = self._loader.load_from_file(config_path)
            self._fingerprints[config_name] = self.compute_config_fingerprint(self._configs[config_name])
            self._mtimes[config_name] = config_path.stat().st_mtime

        self._watched_paths[config_name] = config_path
        self._change_history.setdefault(config_name, [])

        if callback is not None:
            self.subscribe(config_name, callback)

        return config_name

    def check_changes(self) -> Optional[ReloadEvent]:
        """Check all watched configs for file changes. Returns first ReloadEvent or None."""
        for config_name, path in self._watched_paths.items():
            if not path.exists():
                continue
            current_mtime = path.stat().st_mtime
            last_mtime = self._mtimes.get(config_name, 0)

            if current_mtime > last_mtime:
                try:
                    new_config = self._loader.load_from_file(path)
                    old_config = self._configs.get(config_name, {})

                    changes = self._compute_changes(old_config, new_config)

                    self._configs[config_name] = new_config
                    new_fingerprint = self.compute_config_fingerprint(new_config)
                    self._fingerprints[config_name] = new_fingerprint
                    self._mtimes[config_name] = current_mtime

                    event = ReloadEvent(
                        timestamp=time.time(),
                        config_name=config_name,
                        changes=changes,
                        source="file",
                    )
                    self._change_history[config_name].append(event)
                    self._notify_subscribers(config_name, event)
                    return event

                except Exception:
                    continue

        return None

    def get_current_config(self, config_name: str) -> Dict[str, Any]:
        """Get the current in-memory configuration for a watched config."""
        if config_name not in self._configs:
            raise KeyError(f"No configuration loaded for '{config_name}'")
        return deepcopy(self._configs[config_name])

    def reload(self, config_name: str) -> Dict[str, Any]:
        """Force reload a configuration from its watched file path.

        Raises KeyError if config_name is not being watched.
        """
        if config_name not in self._watched_paths:
            raise KeyError(f"'{config_name}' is not being watched")
        path = self._watched_paths[config_name]
        new_config = self._loader.load_from_file(path)
        old_config = self._configs.get(config_name, {})

        changes = self._compute_changes(old_config, new_config)
        self._configs[config_name] = new_config
        self._fingerprints[config_name] = self.compute_config_fingerprint(new_config)
        self._mtimes[config_name] = path.stat().st_mtime

        event = ReloadEvent(
            timestamp=time.time(),
            config_name=config_name,
            changes=changes,
            source="reload",
        )
        self._change_history[config_name].append(event)
        self._notify_subscribers(config_name, event)

        return new_config

    def subscribe(self, config_name: str, callback: Callable[[ReloadEvent], None]) -> int:
        """Subscribe to change events for a config. Returns subscription ID."""
        sub_id = self._next_sub_id
        self._next_sub_id += 1
        self._sub_lookup[sub_id] = (config_name, callback)
        self._subscriptions.setdefault(config_name, []).append(callback)
        return sub_id

    def unsubscribe(self, subscription_id: int) -> None:
        """Unsubscribe by subscription ID."""
        if subscription_id not in self._sub_lookup:
            raise KeyError(f"Subscription ID {subscription_id} not found")
        config_name, callback = self._sub_lookup.pop(subscription_id)
        cbs = self._subscriptions.get(config_name, [])
        try:
            cbs.remove(callback)
        except ValueError:
            pass

    def get_change_history(self, config_name: str, limit: int = 10) -> List[ReloadEvent]:
        """Get change history for a config, most recent first, up to limit."""
        events = self._change_history.get(config_name, [])
        return list(reversed(events[-limit:]))

    def compute_config_fingerprint(self, config: Dict[str, Any]) -> str:
        """Compute a SHA-256 fingerprint of a configuration dict."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

    def _compute_changes(self, old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Compute what changed between old and new configs."""
        added = {}
        removed = {}
        changed = {}

        all_keys = set(old.keys()) | set(new.keys())
        for key in all_keys:
            if key not in old:
                added[key] = deepcopy(new[key])
            elif key not in new:
                removed[key] = deepcopy(old[key])
            elif old[key] != new[key]:
                changed[key] = {"old": old[key], "new": new[key]}

        return {"added": added, "removed": removed, "changed": changed}

    def _notify_subscribers(self, config_name: str, event: ReloadEvent) -> None:
        """Notify all subscribers of a config change."""
        for callback in self._subscriptions.get(config_name, []):
            try:
                callback(event)
            except Exception:
                pass  # subscriber errors don't propagate
