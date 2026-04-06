"""
NEXUS Phase 4 Round 4: Predictive Maintenance Engine

Pure-Python predictive maintenance for marine robotics equipment.
Modules: health_scoring, failure_prediction, remaining_life,
         scheduling, fault_classification, degradation.
"""

from .health_scoring import SensorReading, EquipmentHealth, HealthScorer
from .failure_prediction import TimeSeriesPoint, PredictionResult, FailurePredictor
from .remaining_life import RULEstimate, RemainingLifeEstimator
from .scheduling import MaintenanceTask, ScheduleResult, MaintenanceScheduler
from .fault_classification import FaultSignature, FaultReport, FaultClassifier
from .degradation import DegradationModel, DegradationCurve, DegradationModeler

__all__ = [
    "SensorReading", "EquipmentHealth", "HealthScorer",
    "TimeSeriesPoint", "PredictionResult", "FailurePredictor",
    "RULEstimate", "RemainingLifeEstimator",
    "MaintenanceTask", "ScheduleResult", "MaintenanceScheduler",
    "FaultSignature", "FaultReport", "FaultClassifier",
    "DegradationModel", "DegradationCurve", "DegradationModeler",
]
