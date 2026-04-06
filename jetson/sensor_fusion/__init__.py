"""NEXUS Sensor Fusion — Kalman filters, particle filter, Bayesian estimation,
data association, and sensor calibration.

Pure Python — no external dependencies.
"""

from jetson.sensor_fusion.kalman import (
    KalmanState,
    LinearKalmanFilter,
    ExtendedKalmanFilter,
)

from jetson.sensor_fusion.particle import (
    Particle,
    ParticleFilter,
)

from jetson.sensor_fusion.bayesian import (
    BayesEstimate,
    BayesianEstimator,
)

from jetson.sensor_fusion.associative import (
    Track,
    Association,
    DataAssociator,
)

from jetson.sensor_fusion.calibration import (
    CalibrationParams,
    SensorCalibrator,
)

__all__ = [
    "KalmanState", "LinearKalmanFilter", "ExtendedKalmanFilter",
    "Particle", "ParticleFilter",
    "BayesEstimate", "BayesianEstimator",
    "Track", "Association", "DataAssociator",
    "CalibrationParams", "SensorCalibrator",
]
