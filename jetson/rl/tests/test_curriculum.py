"""Tests for curriculum.py — 38 tests."""

import pytest
from jetson.rl.curriculum import CurriculumStage, CurriculumScheduler, DifficultyScaler


class TestCurriculumStage:
    def test_create_default(self):
        stage = CurriculumStage(name="stage1", level=1)
        assert stage.name == "stage1"
        assert stage.level == 1
        assert stage.config == {}
        assert stage.reward_multiplier == 1.0
        assert stage.is_final is False

    def test_create_full(self):
        stage = CurriculumStage(
            name="hard", level=3, config={"obs": 20},
            reward_multiplier=2.0, max_episodes=200,
            min_reward_threshold=5.0, is_final=True
        )
        assert stage.config == {"obs": 20}
        assert stage.reward_multiplier == 2.0
        assert stage.is_final is True
        assert stage.min_reward_threshold == 5.0


class TestCurriculumScheduler:
    def setup_method(self):
        self.scheduler = CurriculumScheduler()

    def test_create_empty(self):
        assert self.scheduler.num_stages() == 0
        assert self.scheduler.current_stage() is None

    def test_add_stage(self):
        self.scheduler.add_stage(CurriculumStage(name="s1", level=1))
        assert self.scheduler.num_stages() == 1

    def test_current_stage(self):
        s1 = CurriculumStage(name="s1", level=1)
        self.scheduler.add_stage(s1)
        assert self.scheduler.current_stage() is s1

    def test_record_episode(self):
        self.scheduler.add_stage(CurriculumStage(name="s1", level=1))
        self.scheduler.record_episode(5.0)
        rewards = self.scheduler.get_episode_rewards()
        assert rewards == [5.0]

    def test_should_advance_no_stages(self):
        assert self.scheduler.should_advance() is False

    def test_should_advance_final_stage(self):
        s1 = CurriculumStage(name="final", level=1, is_final=True)
        self.scheduler.add_stage(s1)
        assert self.scheduler.should_advance() is False

    def test_should_advance_threshold_not_met(self):
        s1 = CurriculumStage(name="s1", level=1, min_reward_threshold=10.0)
        self.scheduler.add_stage(s1)
        for _ in range(5):
            self.scheduler.record_episode(1.0)
        assert self.scheduler.should_advance() is False

    def test_should_advance_threshold_met(self):
        s1 = CurriculumStage(name="s1", level=1, min_reward_threshold=5.0)
        self.scheduler.add_stage(s1)
        s2 = CurriculumStage(name="s2", level=2)
        self.scheduler.add_stage(s2)
        for _ in range(10):
            self.scheduler.record_episode(10.0)
        assert self.scheduler.should_advance() is True

    def test_advance_success(self):
        s1 = CurriculumStage(name="s1", level=1)
        s2 = CurriculumStage(name="s2", level=2)
        self.scheduler.add_stage(s1)
        self.scheduler.add_stage(s2)
        ok = self.scheduler.advance()
        assert ok is True
        assert self.scheduler.current_stage().name == "s2"

    def test_advance_last_stage(self):
        s1 = CurriculumStage(name="s1", level=1)
        self.scheduler.add_stage(s1)
        ok = self.scheduler.advance()
        assert ok is False

    def test_get_progress(self):
        s1 = CurriculumStage(name="s1", level=1)
        s2 = CurriculumStage(name="s2", level=2)
        s3 = CurriculumStage(name="s3", level=3)
        self.scheduler.add_stage(s1)
        self.scheduler.add_stage(s2)
        self.scheduler.add_stage(s3)
        assert self.scheduler.get_progress() == 0.0
        self.scheduler.advance()
        assert self.scheduler.get_progress() == 0.5
        self.scheduler.advance()
        assert self.scheduler.get_progress() == 1.0

    def test_progress_single_stage(self):
        self.scheduler.add_stage(CurriculumStage(name="s1", level=1))
        assert self.scheduler.get_progress() == 0.0

    def test_stage_history(self):
        self.scheduler.add_stage(CurriculumStage(name="s1", level=1))
        self.scheduler.record_episode(1.0)
        self.scheduler.record_episode(2.0)
        history = self.scheduler.get_stage_history()
        assert history == [0, 0]

    def test_reset(self):
        s1 = CurriculumStage(name="s1", level=1)
        s2 = CurriculumStage(name="s2", level=2)
        self.scheduler.add_stage(s1)
        self.scheduler.add_stage(s2)
        self.scheduler.advance()
        self.scheduler.record_episode(5.0)
        self.scheduler.reset()
        assert self.scheduler.current_stage().name == "s1"
        assert self.scheduler.get_episode_rewards() == []

    def test_linear_transition(self):
        scheduler = CurriculumScheduler(transition_mode="linear")
        s1 = CurriculumStage(name="s1", level=1, max_episodes=5)
        s2 = CurriculumStage(name="s2", level=2)
        scheduler.add_stage(s1)
        scheduler.add_stage(s2)
        for _ in range(4):
            scheduler.record_episode(-100.0)
        assert scheduler.should_advance() is False
        scheduler.record_episode(-100.0)
        assert scheduler.should_advance() is True

    def test_linear_not_enough_episodes(self):
        scheduler = CurriculumScheduler(transition_mode="linear")
        s1 = CurriculumStage(name="s1", level=1, max_episodes=100)
        s2 = CurriculumStage(name="s2", level=2)
        scheduler.add_stage(s1)
        scheduler.add_stage(s2)
        for _ in range(10):
            scheduler.record_episode(100.0)
        assert scheduler.should_advance() is False

    def test_adaptive_transition(self):
        scheduler = CurriculumScheduler(transition_mode="adaptive")
        s1 = CurriculumStage(name="s1", level=1, min_reward_threshold=5.0)
        s2 = CurriculumStage(name="s2", level=2)
        scheduler.add_stage(s1)
        scheduler.add_stage(s2)
        for _ in range(20):
            scheduler.record_episode(6.0)
        assert scheduler.should_advance() is True

    def test_adaptive_plateau(self):
        scheduler = CurriculumScheduler(transition_mode="adaptive")
        s1 = CurriculumStage(name="s1", level=1, min_reward_threshold=10.0)
        s2 = CurriculumStage(name="s2", level=2)
        scheduler.add_stage(s1)
        scheduler.add_stage(s2)
        # Give high enough rewards that plateau
        for _ in range(20):
            scheduler.record_episode(9.0)
        # Should advance due to plateau + high avg
        assert scheduler.should_advance() is True

    def test_advance_resets_rewards(self):
        s1 = CurriculumStage(name="s1", level=1)
        s2 = CurriculumStage(name="s2", level=2)
        self.scheduler.add_stage(s1)
        self.scheduler.add_stage(s2)
        self.scheduler.record_episode(5.0)
        self.scheduler.advance()
        assert self.scheduler.get_episode_rewards() == []


class TestDifficultyScaler:
    def setup_method(self):
        self.scaler = DifficultyScaler(min_val=0.0, max_val=1.0)

    def test_create(self):
        assert self.scaler.min_val == 0.0
        assert self.scaler.max_val == 1.0

    def test_scale_obstacles_min(self):
        count = self.scaler.scale_obstacles(0.0)
        assert count == 5

    def test_scale_obstacles_max(self):
        count = self.scaler.scale_obstacles(1.0)
        assert count == 20

    def test_scale_obstacles_mid(self):
        count = self.scaler.scale_obstacles(0.5)
        assert 5 <= count <= 20

    def test_scale_speed_min(self):
        speed = self.scaler.scale_speed(0.0)
        assert speed == 1.0

    def test_scale_speed_max(self):
        speed = self.scaler.scale_speed(1.0)
        assert speed == 5.0

    def test_scale_noise_min(self):
        noise = self.scaler.scale_noise(0.0)
        assert noise == 0.0

    def test_scale_noise_max(self):
        noise = self.scaler.scale_noise(1.0)
        assert noise == 1.0

    def test_interpolate_config_numeric(self):
        base = {"speed": 1.0, "obstacles": 5}
        target = {"speed": 5.0, "obstacles": 20}
        result = self.scaler.interpolate_config(base, target, 0.5)
        assert abs(result["speed"] - 3.0) < 1e-6
        assert abs(result["obstacles"] - 12.5) < 1e-6

    def test_interpolate_config_list(self):
        base = {"weights": [0.0, 0.0]}
        target = {"weights": [1.0, 2.0]}
        result = self.scaler.interpolate_config(base, target, 0.5)
        assert abs(result["weights"][0] - 0.5) < 1e-6
        assert abs(result["weights"][1] - 1.0) < 1e-6

    def test_interpolate_config_t_zero(self):
        base = {"x": 10.0}
        target = {"x": 20.0}
        result = self.scaler.interpolate_config(base, target, 0.0)
        assert result["x"] == 10.0

    def test_interpolate_config_t_one(self):
        base = {"x": 10.0}
        target = {"x": 20.0}
        result = self.scaler.interpolate_config(base, target, 1.0)
        assert result["x"] == 20.0

    def test_interpolate_non_numeric_fallback(self):
        base = {"mode": "easy"}
        target = {"mode": "hard"}
        result = self.scaler.interpolate_config(base, target, 0.3)
        assert result["mode"] == "easy"

    def test_interpolate_non_numeric_fallback_high(self):
        base = {"mode": "easy"}
        target = {"mode": "hard"}
        result = self.scaler.interpolate_config(base, target, 0.7)
        assert result["mode"] == "hard"

    def test_get_difficulty_level(self):
        assert self.scaler.get_difficulty_level(0.0) == "beginner"
        assert self.scaler.get_difficulty_level(0.2) == "beginner"
        assert self.scaler.get_difficulty_level(0.25) == "intermediate"
        assert self.scaler.get_difficulty_level(0.49) == "intermediate"
        assert self.scaler.get_difficulty_level(0.5) == "advanced"
        assert self.scaler.get_difficulty_level(0.74) == "advanced"
        assert self.scaler.get_difficulty_level(0.75) == "expert"
        assert self.scaler.get_difficulty_level(1.0) == "expert"

    def test_clamp_high(self):
        count = self.scaler.scale_obstacles(5.0)
        assert count == 20

    def test_clamp_low(self):
        count = self.scaler.scale_obstacles(-1.0)
        assert count == 5

    def test_custom_range(self):
        scaler = DifficultyScaler(min_val=10.0, max_val=100.0)
        speed = scaler.scale_speed(0.5, base_speed=0.0, max_speed=10.0)
        assert speed == 5.0
