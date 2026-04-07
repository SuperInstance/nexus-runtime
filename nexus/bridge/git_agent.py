"""
NEXUS Git Agent Bridge — repository-as-agent paradigm.

Uses git repositories as persistent, versioned agent state stores.
Each commit represents an agent state transition, enabling:
    - Full audit trail of agent decisions
    - Branch-based experimentation
    - Rollback to any previous state
    - Distributed consensus via merge

This module uses a simulated git backend for portability (no git binary required).
For production use, it can be swapped with gitpython-backed implementations.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CommitInfo:
    """Parsed information from a git commit."""

    commit_hash: str
    parent_hash: str = ""
    author: str = ""
    message: str = ""
    timestamp: float = 0.0
    agent_state: Dict[str, Any] = field(default_factory=dict)
    branch: str = "main"

    def __repr__(self) -> str:
        short_hash = self.commit_hash[:8] if self.commit_hash else "?"
        return f"CommitInfo({short_hash} '{self.message[:30]}')"


@dataclass
class GitAgentConfig:
    """Configuration for the Git Agent bridge."""

    repo_path: str = ""
    agent_id: str = ""
    author_name: str = "nexus-agent"
    author_email: str = "agent@nexus.local"
    auto_commit: bool = True
    branch: str = "main"


# ---------------------------------------------------------------------------
# Simple in-memory git backend (simulated)
# ---------------------------------------------------------------------------

class _SimulatedGitBackend:
    """In-memory simulated git repository for testing and portable usage."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        self._commits: List[CommitInfo] = []
        self._branches: Dict[str, str] = {"main": ""}  # branch -> head hash
        self._HEAD = "main"
        self._state: Dict[str, Any] = {}

    @property
    def current_branch(self) -> str:
        return self._HEAD

    def init(self) -> None:
        """Initialize the repository."""
        self._commits.clear()
        self._branches = {"main": ""}
        self._HEAD = "main"
        self._state = {}

    def commit(self, message: str, state: Dict[str, Any], author: str = "") -> CommitInfo:
        """Create a new commit with the given agent state."""
        parent = self._branches.get(self._HEAD, "")

        state_json = json.dumps(state, sort_keys=True)
        raw = f"{parent}{state_json}{message}{time.time()}"
        commit_hash = hashlib.sha256(raw.encode()).hexdigest()

        commit = CommitInfo(
            commit_hash=commit_hash,
            parent_hash=parent,
            author=author,
            message=message,
            timestamp=time.time(),
            agent_state=copy.deepcopy(state),
            branch=self._HEAD,
        )

        self._commits.append(commit)
        self._branches[self._HEAD] = commit_hash
        self._state = copy.deepcopy(state)
        return commit

    def create_branch(self, name: str) -> bool:
        """Create a new branch at the current HEAD."""
        if name in self._branches:
            return False
        self._branches[name] = self._branches.get(self._HEAD, "")
        return True

    def checkout(self, branch: str) -> bool:
        """Switch to a branch."""
        if branch not in self._branches:
            return False
        self._HEAD = branch
        # Restore state from branch head
        head_hash = self._branches[branch]
        if head_hash:
            commit = self._find_commit(head_hash)
            if commit:
                self._state = copy.deepcopy(commit.agent_state)
        return True

    def merge(self, source_branch: str) -> bool:
        """Merge source_branch into current branch (fast-forward only)."""
        if source_branch not in self._branches:
            return False
        self._branches[self._HEAD] = self._branches[source_branch]
        return True

    def get_head_commit(self) -> Optional[CommitInfo]:
        """Get the latest commit on the current branch."""
        head = self._branches.get(self._HEAD, "")
        if not head:
            return None
        return self._find_commit(head)

    def get_commit_history(self, limit: int = 50) -> List[CommitInfo]:
        """Get commit history for the current branch."""
        head = self._branches.get(self._HEAD, "")
        if not head:
            return []
        history: List[CommitInfo] = []
        current = head
        visited = set()
        while current and current not in visited and len(history) < limit:
            commit = self._find_commit(current)
            if commit is None:
                break
            history.append(commit)
            visited.add(current)
            current = commit.parent_hash
        return history

    def _find_commit(self, commit_hash: str) -> Optional[CommitInfo]:
        for c in self._commits:
            if c.commit_hash == commit_hash:
                return c
        return None

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def list_branches(self) -> List[str]:
        return list(self._branches.keys())


# Need copy
import copy


# ---------------------------------------------------------------------------
# Git Agent
# ---------------------------------------------------------------------------

class GitAgent:
    """NEXUS Git Agent — manages agent state via a git-like repository.

    Usage::

        agent = GitAgent(agent_id="AUV-001")
        agent.init()
        agent.save_state({"position": [1.0, 2.0], "battery": 85})
        agent.save_state({"position": [1.5, 2.3], "battery": 82})
        history = agent.get_history()
    """

    def __init__(
        self,
        agent_id: str = "",
        config: Optional[GitAgentConfig] = None,
    ) -> None:
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.config = config or GitAgentConfig(agent_id=self.agent_id)
        self._backend = _SimulatedGitBackend(self.config.repo_path or f"/tmp/nexus-agent-{self.agent_id}")
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def current_branch(self) -> str:
        return self._backend.current_branch

    # ----- lifecycle -----

    def init(self) -> None:
        """Initialize the agent's git repository."""
        self._backend.init()
        self._initialized = True
        # Create initial commit
        self._backend.commit(
            message=f"Initialize agent {self.agent_id}",
            state={"agent_id": self.agent_id, "created_at": time.time()},
            author=self.config.author_name,
        )

    # ----- state management -----

    def save_state(self, state: Dict[str, Any], message: str = "") -> CommitInfo:
        """Save agent state as a new commit."""
        if not self._initialized:
            self.init()

        if not message:
            message = f"State update at {datetime.now(timezone.utc).isoformat()}"

        return self._backend.commit(
            message=message,
            state=state,
            author=self.config.author_name,
        )

    def get_state(self) -> Dict[str, Any]:
        """Get the current agent state."""
        return self._backend.get_state()

    def get_history(self, limit: int = 50) -> List[CommitInfo]:
        """Get commit history."""
        return self._backend.get_commit_history(limit=limit)

    def get_latest_commit(self) -> Optional[CommitInfo]:
        """Get the most recent commit."""
        return self._backend.get_head_commit()

    # ----- branch management -----

    def create_branch(self, name: str) -> bool:
        """Create a new branch at the current HEAD."""
        return self._backend.create_branch(name)

    def switch_branch(self, name: str) -> bool:
        """Switch to a branch."""
        return self._backend.checkout(name)

    def merge_branch(self, source: str) -> bool:
        """Fast-forward merge source branch into current branch."""
        return self._backend.merge(source)

    def list_branches(self) -> List[str]:
        """List all branches."""
        return self._backend.list_branches()

    # ----- queries -----

    def parse_commit_message(self, commit: CommitInfo) -> Dict[str, Any]:
        """Parse a commit message for structured metadata.

        Supports key=value pairs in commit messages.
        """
        metadata: Dict[str, Any] = {"raw_message": commit.message}
        for part in commit.message.split():
            if "=" in part:
                key, _, value = part.partition("=")
                metadata[key] = self._parse_value(value)
        return metadata

    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse a string value to its native type."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    def rollback(self, steps: int = 1) -> Optional[CommitInfo]:
        """Roll back the current branch by *steps* commits.

        Creates a new branch with the rolled-back state.
        """
        history = self._backend.get_commit_history(limit=steps + 2)
        if len(history) <= steps:
            return None

        target = history[-(steps + 1)]
        rollback_branch = f"rollback-{int(time.time())}"
        self._backend.create_branch(rollback_branch)
        self._backend.checkout(rollback_branch)
        # We need to set the branch head to the target commit
        self._backend._branches[rollback_branch] = target.commit_hash
        self._backend._state = copy.deepcopy(target.agent_state)

        return self._backend.get_head_commit()

    def diff(self, commit_a: Optional[CommitInfo] = None, commit_b: Optional[CommitInfo] = None) -> Dict[str, Any]:
        """Compare state between two commits. Returns changed keys."""
        a = commit_a or self._backend.get_head_commit()
        b = commit_b
        if b is None:
            history = self._backend.get_commit_history(limit=2)
            b = history[1] if len(history) > 1 else a

        if a is None or b is None:
            return {}

        diff: Dict[str, Any] = {}
        all_keys = set(a.agent_state.keys()) | set(b.agent_state.keys())
        for key in all_keys:
            val_a = a.agent_state.get(key)
            val_b = b.agent_state.get(key)
            if val_a != val_b:
                diff[key] = {"before": val_a, "after": val_b}
        return diff
