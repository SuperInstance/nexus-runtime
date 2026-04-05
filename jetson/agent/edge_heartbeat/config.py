"""NEXUS Edge Heartbeat — Configuration.

HeartbeatConfig dataclass and loaders for vessel configuration.
Supports JSON file loading and sensible defaults for Jetson deployment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HeartbeatConfig:
    """Vessel heartbeat configuration.

    All parameters for the edge heartbeat loop, including vessel identity,
    git-agent coordination paths, serial I/O, and trust reporting.
    """

    vessel_id: str = "nexus-vessel-001"
    repo_path: str = "/opt/nexus-runtime"
    heartbeat_interval: int = 300  # seconds (5 minutes)
    telemetry_batch_size: int = 100
    trust_reporting: bool = True
    github_token: str | None = None
    github_repo: str | None = None
    serial_port: str = "/dev/ttyUSB0"
    serial_baud: int = 115200
    agent_dir_name: str = ".agent"
    next_file_name: str = "next"
    done_file_name: str = "done"
    identity_file_name: str = "identity"
    max_mission_retries: int = 3
    mission_timeout_seconds: int = 120
    log_level: str = "INFO"


def default_config() -> HeartbeatConfig:
    """Create a HeartbeatConfig with all defaults."""
    return HeartbeatConfig()


def load_config(config_path: str) -> HeartbeatConfig:
    """Load HeartbeatConfig from a JSON file.

    Missing keys use default values. Extra keys are ignored.

    Args:
        config_path: Path to JSON configuration file.

    Returns:
        Populated HeartbeatConfig instance.

    Raises:
        FileNotFoundError: If config_path does not exist.
        json.JSONDecodeError: If config_path is not valid JSON.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    defaults = default_config()
    return HeartbeatConfig(
        vessel_id=data.get("vessel_id", defaults.vessel_id),
        repo_path=data.get("repo_path", defaults.repo_path),
        heartbeat_interval=data.get("heartbeat_interval", defaults.heartbeat_interval),
        telemetry_batch_size=data.get("telemetry_batch_size", defaults.telemetry_batch_size),
        trust_reporting=data.get("trust_reporting", defaults.trust_reporting),
        github_token=data.get("github_token", defaults.github_token),
        github_repo=data.get("github_repo", defaults.github_repo),
        serial_port=data.get("serial_port", defaults.serial_port),
        serial_baud=data.get("serial_baud", defaults.serial_baud),
        agent_dir_name=data.get("agent_dir_name", defaults.agent_dir_name),
        next_file_name=data.get("next_file_name", defaults.next_file_name),
        done_file_name=data.get("done_file_name", defaults.done_file_name),
        identity_file_name=data.get("identity_file_name", defaults.identity_file_name),
        max_mission_retries=data.get("max_mission_retries", defaults.max_mission_retries),
        mission_timeout_seconds=data.get("mission_timeout_seconds", defaults.mission_timeout_seconds),
        log_level=data.get("log_level", defaults.log_level),
    )


def save_config(config: HeartbeatConfig, config_path: str) -> None:
    """Save HeartbeatConfig to a JSON file.

    Args:
        config: Configuration to save.
        config_path: Output file path.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "vessel_id": config.vessel_id,
        "repo_path": config.repo_path,
        "heartbeat_interval": config.heartbeat_interval,
        "telemetry_batch_size": config.telemetry_batch_size,
        "trust_reporting": config.trust_reporting,
        "github_token": config.github_token,
        "github_repo": config.github_repo,
        "serial_port": config.serial_port,
        "serial_baud": config.serial_baud,
        "agent_dir_name": config.agent_dir_name,
        "next_file_name": config.next_file_name,
        "done_file_name": config.done_file_name,
        "identity_file_name": config.identity_file_name,
        "max_mission_retries": config.max_mission_retries,
        "mission_timeout_seconds": config.mission_timeout_seconds,
        "log_level": config.log_level,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def config_to_json(config: HeartbeatConfig) -> str:
    """Serialize a HeartbeatConfig to a JSON string."""
    data = {
        "vessel_id": config.vessel_id,
        "repo_path": config.repo_path,
        "heartbeat_interval": config.heartbeat_interval,
        "telemetry_batch_size": config.telemetry_batch_size,
        "trust_reporting": config.trust_reporting,
        "github_token": config.github_token,
        "github_repo": config.github_repo,
        "serial_port": config.serial_port,
        "serial_baud": config.serial_baud,
    }
    return json.dumps(data, indent=2)
