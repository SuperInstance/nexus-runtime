"""
NEXUS Phase 4 Round 2: Reinforcement Learning Framework
Marine robotics intelligence platform — pure Python RL toolkit.
"""

from jetson.rl.environments import (
    ActionSpace, ObservationSpace, StepResult,
    MarineNavigationEnv, MarinePatrolEnv, CollisionAvoidanceEnv,
)
from jetson.rl.agents import (
    TabularQLearning, DeepQLearning, PolicyGradientAgent, PPOLiteAgent,
)
from jetson.rl.replay_buffer import (
    Transition, ReplayBuffer, PrioritizedReplayBuffer, EpisodeBuffer,
)
from jetson.rl.reward_shaping import (
    NavigationRewardShaper, PatrolRewardShaper, MultiObjectiveReward,
)
from jetson.rl.multi_agent import (
    MultiAgentEnv, IndependentLearner, CentralizedCritic, CommunicationProtocol,
)
from jetson.rl.curriculum import (
    CurriculumStage, CurriculumScheduler, DifficultyScaler,
)

__all__ = [
    "ActionSpace", "ObservationSpace", "StepResult",
    "MarineNavigationEnv", "MarinePatrolEnv", "CollisionAvoidanceEnv",
    "TabularQLearning", "DeepQLearning", "PolicyGradientAgent", "PPOLiteAgent",
    "Transition", "ReplayBuffer", "PrioritizedReplayBuffer", "EpisodeBuffer",
    "NavigationRewardShaper", "PatrolRewardShaper", "MultiObjectiveReward",
    "MultiAgentEnv", "IndependentLearner", "CentralizedCritic", "CommunicationProtocol",
    "CurriculumStage", "CurriculumScheduler", "DifficultyScaler",
]
