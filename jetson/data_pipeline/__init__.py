"""NEXUS Data Pipeline Engine — Phase 5 Round 7.

Pure-Python data stream processing, transformations, time-series aggregation,
in-memory storage, and pipeline orchestration for edge AI workloads.
"""

from .stream import DataPoint, DataStream
from .transform import Transform, TransformPipeline
from .aggregation import AggregationType, TimeWindow, TimeSeriesAggregator
from .storage import StorageEntry, StorageBackend, TimeSeriesStore
from .pipeline import PipelineStage, PipelineResult, DataPipeline

__all__ = [
    "DataPoint", "DataStream",
    "Transform", "TransformPipeline",
    "AggregationType", "TimeWindow", "TimeSeriesAggregator",
    "StorageEntry", "StorageBackend", "TimeSeriesStore",
    "PipelineStage", "PipelineResult", "DataPipeline",
]
