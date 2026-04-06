"""Cooperative perception module for multi-vessel autonomous maritime systems.

Provides perception data sharing, multi-vessel fusion, cooperative tracking,
consensus-based perception, and quality assessment capabilities.
"""

from .sharing import PerceptionMessage, PerceivedObject, PerceptionSharer
from .fusion import FusedObject, FusionResult, PerceptionFusion
from .tracking import CooperativeTrack, TrackAssociation, CooperativeTracker
from .consensus_percept import PerceptionVote, PerceptionConsensus, PerceptionConsensusEngine
from .quality import QualityMetrics, PerceptionQuality

__all__ = [
    "PerceptionMessage", "PerceivedObject", "PerceptionSharer",
    "FusedObject", "FusionResult", "PerceptionFusion",
    "CooperativeTrack", "TrackAssociation", "CooperativeTracker",
    "PerceptionVote", "PerceptionConsensus", "PerceptionConsensusEngine",
    "QualityMetrics", "PerceptionQuality",
]
