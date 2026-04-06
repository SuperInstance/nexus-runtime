"""
Curriculum learning: stages, scheduling, and difficulty scaling.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CurriculumStage:
    """A single stage in a curriculum."""
    name: str
    level: int
    config: Dict[str, Any] = field(default_factory=dict)
    reward_multiplier: float = 1.0
    max_episodes: int = 100
    min_reward_threshold: float = 0.0
    is_final: bool = False


class CurriculumScheduler:
    """Manages curriculum stage transitions."""

    def __init__(self, stages: Optional[List[CurriculumStage]] = None,
                 transition_mode: str = "threshold"):
        self.stages = stages or []
        self.transition_mode = transition_mode
        self.current_index = 0
        self._episode_rewards: List[float] = []
        self._episode_count = 0
        self._stage_history: List[int] = []

    def add_stage(self, stage: CurriculumStage):
        self.stages.append(stage)

    def current_stage(self) -> Optional[CurriculumStage]:
        if not self.stages:
            return None
        return self.stages[self.current_index]

    def record_episode(self, reward: float):
        self._episode_rewards.append(reward)
        self._episode_count += 1
        self._stage_history.append(self.current_index)

    def should_advance(self) -> bool:
        stage = self.current_stage()
        if stage is None or stage.is_final:
            return False
        if self.transition_mode == "threshold":
            return self._check_threshold(stage)
        elif self.transition_mode == "linear":
            return self._check_linear(stage)
        elif self.transition_mode == "adaptive":
            return self._check_adaptive(stage)
        return False

    def _check_threshold(self, stage: CurriculumStage) -> bool:
        recent = self._episode_rewards[-10:] if self._episode_rewards else []
        if len(recent) < 5:
            return False
        avg = sum(recent) / len(recent)
        return avg >= stage.min_reward_threshold

    def _check_linear(self, stage: CurriculumStage) -> bool:
        return self._episode_count >= stage.max_episodes

    def _check_adaptive(self, stage: CurriculumStage) -> bool:
        recent = self._episode_rewards[-20:] if self._episode_rewards else []
        if len(recent) < 10:
            return False
        avg = sum(recent) / len(recent)
        # Also check if improvement is plateauing
        if len(recent) >= 10:
            first_half = sum(recent[:len(recent) // 2]) / (len(recent) // 2)
            second_half = sum(recent[len(recent) // 2:]) / (len(recent) - len(recent) // 2)
            plateau = abs(second_half - first_half) < 0.5
            if avg >= stage.min_reward_threshold * 0.8 and plateau:
                return True
        return avg >= stage.min_reward_threshold

    def advance(self) -> bool:
        if self.current_index < len(self.stages) - 1:
            self.current_index += 1
            self._episode_rewards = []
            self._episode_count = 0
            return True
        return False

    def get_progress(self) -> float:
        if not self.stages:
            return 0.0
        return self.current_index / max(len(self.stages) - 1, 1)

    def get_stage_history(self) -> List[int]:
        return list(self._stage_history)

    def get_episode_rewards(self) -> List[float]:
        return list(self._episode_rewards)

    def total_episodes(self) -> int:
        return sum(len(self._stage_history) for _ in [1]) + len(self._stage_history)
        # Actually just return the full history length

    def reset(self):
        self.current_index = 0
        self._episode_rewards = []
        self._episode_count = 0
        self._stage_history = []

    def num_stages(self) -> int:
        return len(self.stages)


class DifficultyScaler:
    """Scales environment difficulty parameters."""

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def _lerp(self, t: float) -> float:
        """Clamp and normalize difficulty to [0, 1]."""
        return max(0.0, min(1.0, t))

    def scale_obstacles(self, difficulty: float, base_count: int = 5,
                        max_count: int = 20) -> int:
        count = base_count + int((max_count - base_count) * self._lerp(difficulty))
        return max(base_count, min(max_count, count))

    def scale_speed(self, difficulty: float, base_speed: float = 1.0,
                    max_speed: float = 5.0) -> float:
        return base_speed + (max_speed - base_speed) * self._lerp(difficulty)

    def scale_noise(self, difficulty: float, base_noise: float = 0.0,
                    max_noise: float = 1.0) -> float:
        return base_noise + (max_noise - base_noise) * self._lerp(difficulty)

    def interpolate_config(self, base_config: Dict[str, Any],
                           target_config: Dict[str, Any],
                           t: float) -> Dict[str, Any]:
        """Interpolate between two configs by factor t in [0, 1]."""
        result = {}
        all_keys = set(base_config.keys()) | set(target_config.keys())
        for key in all_keys:
            bv = base_config.get(key, 0.0)
            tv = target_config.get(key, 0.0)
            if isinstance(bv, (int, float)) and isinstance(tv, (int, float)):
                result[key] = bv + (tv - bv) * self._lerp(t)
            elif isinstance(bv, list) and isinstance(tv, list):
                result[key] = [bv[i] + (tv[i] - bv[i]) * self._lerp(t)
                               for i in range(min(len(bv), len(tv)))]
            else:
                result[key] = tv if t >= 0.5 else bv
        return result

    def get_difficulty_level(self, progress: float) -> str:
        if progress < 0.25:
            return "beginner"
        elif progress < 0.5:
            return "intermediate"
        elif progress < 0.75:
            return "advanced"
        return "expert"
