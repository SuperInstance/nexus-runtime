"""NEXUS A/B Testing — Git Branch Integration.

Manages experiment branches and results:
  - Branch naming: experiment/{test_name}/variant_{A|B|C}
  - Results committed as JSON to .nexus/experiments/{test_name}/results.json
  - Winner merged, losers archived with analysis
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class GitOperationError(Exception):
    """Error during git operations."""


@dataclass
class BranchInfo:
    """Information about an experiment branch."""

    branch_name: str
    variant_name: str
    experiment_name: str
    commit_hash: str = ""
    created_at: str = ""
    status: str = "active"  # active, merged, archived

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_name": self.branch_name,
            "variant_name": self.variant_name,
            "experiment_name": self.experiment_name,
            "commit_hash": self.commit_hash,
            "created_at": self.created_at,
            "status": self.status,
        }


@dataclass
class ExperimentResult:
    """Stored experiment result for git persistence."""

    experiment_name: str
    winner: str
    recommendation: str
    variant_branches: dict[str, str] = field(default_factory=dict)
    statistical_summary: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    total_iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "winner": self.winner,
            "recommendation": self.recommendation,
            "variant_branches": self.variant_branches,
            "statistical_summary": self.statistical_summary,
            "timestamp": self.timestamp,
            "total_iterations": self.total_iterations,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class BranchIntegration:
    """Manages git-branch-based A/B testing workflow.

    Handles branch creation, result storage, winner merge, and loser archival.
    Supports both real git operations and dry-run mode for testing.
    """

    def __init__(
        self,
        repo_root: str = ".",
        nexus_dir: str = ".nexus",
        dry_run: bool = False,
        git_bin: str = "git",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.nexus_dir = Path(nexus_dir)
        self.experiments_dir = self.nexus_dir / "experiments"
        self.dry_run = dry_run
        self.git_bin = git_bin
        self._branch_cache: dict[str, BranchInfo] = {}

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a git command."""
        if self.dry_run:
            return subprocess.CompletedProcess(
                args=["git"] + list(args), returncode=0, stdout="", stderr=""
            )
        result = subprocess.run(
            [self.git_bin] + list(args),
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            raise GitOperationError(
                f"git {' '.join(args)} failed: {result.stderr.strip()}"
            )
        return result

    def branch_name(self, experiment_name: str, variant_name: str) -> str:
        """Generate branch name: experiment/{test_name}/variant_{A|B|C}."""
        safe_exp = experiment_name.replace(" ", "-").replace("/", "-").lower()
        safe_var = variant_name.replace(" ", "-").replace("/", "-").upper()
        return f"experiment/{safe_exp}/variant_{safe_var}"

    def create_variant_branch(
        self,
        experiment_name: str,
        variant_name: str,
        base_branch: str = "main",
    ) -> BranchInfo:
        """Create a new branch for an experiment variant."""
        branch = self.branch_name(experiment_name, variant_name)

        # Check if already on base branch or switch to it
        self._git("checkout", base_branch, check=False)
        # Create and checkout new branch
        self._git("checkout", "-b", branch)

        info = BranchInfo(
            branch_name=branch,
            variant_name=variant_name,
            experiment_name=experiment_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="active",
        )
        self._branch_cache[branch] = info
        return info

    def archive_variant_branch(
        self,
        experiment_name: str,
        variant_name: str,
        analysis: str = "",
    ) -> None:
        """Archive a losing variant branch with analysis notes."""
        branch = self.branch_name(experiment_name, variant_name)
        archive_dir = self.experiments_dir / experiment_name / "archived"
        archive_file = archive_dir / f"{variant_name}.json"

        archive_data = {
            "variant_name": variant_name,
            "branch": branch,
            "analysis": analysis,
            "archived_at": datetime.now(timezone.utc).isoformat(),
        }

        if not self.dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_file.write_text(json.dumps(archive_data, indent=2))

        if branch in self._branch_cache:
            self._branch_cache[branch].status = "archived"

    def merge_winner(
        self,
        experiment_name: str,
        winner_variant: str,
        target_branch: str = "main",
    ) -> str:
        """Merge the winning variant branch into target."""
        branch = self.branch_name(experiment_name, winner_variant)

        if not self.dry_run:
            self._git("checkout", target_branch)
            self._git("merge", "--no-ff", branch, "-m",
                       f"A/B test winner: {winner_variant} for {experiment_name}")

        if branch in self._branch_cache:
            self._branch_cache[branch].status = "merged"

        return branch

    def store_results(
        self,
        experiment_name: str,
        result: Any,
    ) -> Path:
        """Store experiment results as JSON in .nexus/experiments/."""
        result_dir = self.experiments_dir / experiment_name
        result_file = result_dir / "results.json"

        if hasattr(result, "to_dict"):
            data = result.to_dict()
        elif hasattr(result, "to_json"):
            data = json.loads(result.to_json())
        elif isinstance(result, dict):
            data = result
        else:
            data = {"result": str(result)}

        data["stored_at"] = datetime.now(timezone.utc).isoformat()

        if not self.dry_run:
            result_dir.mkdir(parents=True, exist_ok=True)
            result_file.write_text(json.dumps(data, indent=2))

        return result_file

    def load_results(self, experiment_name: str) -> dict[str, Any] | None:
        """Load stored experiment results."""
        result_file = self.experiments_dir / experiment_name / "results.json"
        if not result_file.exists():
            return None
        return json.loads(result_file.read_text())

    def list_experiments(self) -> list[str]:
        """List all stored experiment names."""
        if not self.experiments_dir.exists():
            return []
        return [
            d.name
            for d in self.experiments_dir.iterdir()
            if d.is_dir() and (d / "results.json").exists()
        ]

    def cleanup_experiment(self, experiment_name: str) -> None:
        """Remove experiment data from .nexus/experiments/."""
        exp_dir = self.experiments_dir / experiment_name
        if exp_dir.exists() and not self.dry_run:
            for f in exp_dir.iterdir():
                if f.is_file():
                    f.unlink()
            exp_dir.rmdir()

    def run_full_workflow(
        self,
        experiment_name: str,
        variants: dict[str, bytes],
        results: Any,
    ) -> dict[str, Any]:
        """Run the full A/B test git workflow.

        1. Create branches for each variant
        2. Store results
        3. Archive losers
        4. Merge winner

        Returns workflow summary.
        """
        # Parse winner from results
        winner = "unknown"
        recommendation = "inconclusive"
        if hasattr(results, "winner"):
            winner = results.winner
        if hasattr(results, "recommendation"):
            recommendation = results.recommendation
        elif isinstance(results, dict):
            winner = results.get("winner", "unknown")
            recommendation = results.get("recommendation", "inconclusive")

        # Create variant branches
        branch_infos = {}
        for var_name, _ in variants.items():
            info = self.create_variant_branch(experiment_name, var_name)
            branch_infos[var_name] = info

        # Store results
        result_path = self.store_results(experiment_name, results)

        # If we have a clear winner
        if recommendation in ("A wins", "B wins") and winner != "unknown":
            # Archive losers
            for var_name in variants:
                if var_name != winner:
                    self.archive_variant_branch(
                        experiment_name, var_name,
                        f"Lost A/B test to variant {winner}"
                    )
            # Merge winner
            merge_branch = self.merge_winner(experiment_name, winner)
        else:
            # Inconclusive — archive all as "inconclusive"
            for var_name in variants:
                self.archive_variant_branch(
                    experiment_name, var_name,
                    f"Test inconclusive: {recommendation}"
                )

        return {
            "experiment_name": experiment_name,
            "winner": winner,
            "recommendation": recommendation,
            "branches_created": list(branch_infos.keys()),
            "results_stored": str(result_path),
            "dry_run": self.dry_run,
        }
