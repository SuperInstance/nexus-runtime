"""Phase 6 Round 4: Adaptive Autonomy System for NEXUS Jetson runtime.

Provides autonomy level management (L0-L5), transition logic,
situation assessment, human override controls, and experience-based learning.
"""

from jetson.adaptive_autonomy.levels import (
    AutonomyLevel,
    LevelCapabilities,
    AutonomyLevelManager,
)
from jetson.adaptive_autonomy.transition import (
    TransitionRequest,
    TransitionPolicy,
    TransitionManager,
)
from jetson.adaptive_autonomy.assessment import (
    SituationAssessment,
    SituationAssessor,
)
from jetson.adaptive_autonomy.override import (
    OverrideRequest,
    OverrideResult,
    OverrideManager,
)
from jetson.adaptive_autonomy.learning import (
    TransitionExperience,
    AutonomyLearner,
)

__all__ = [
    "AutonomyLevel",
    "LevelCapabilities",
    "AutonomyLevelManager",
    "TransitionRequest",
    "TransitionPolicy",
    "TransitionManager",
    "SituationAssessment",
    "SituationAssessor",
    "OverrideRequest",
    "OverrideResult",
    "OverrideManager",
    "TransitionExperience",
    "AutonomyLearner",
]
