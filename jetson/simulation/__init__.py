"""NEXUS Simulation Framework — Phase 5 Round 2.

Marine robotics simulation including world representation, multi-body dynamics,
sensor simulation, communication modeling, scenario construction, and
Monte Carlo statistical analysis.
"""

from .world import Vector3, WorldObject, TerrainCell, World
from .dynamics import RigidBody, DynamicsEngine
from .sensor_sim import SensorConfig, SensorReading, SensorSimulator
from .communication import CommLink, Message, CommSimulator
from .scenario_builder import ScenarioObject, ScenarioConfig, ScenarioBuilder
from .statistics import MonteCarloResult, SimulationRun, SimulationStatistics

__all__ = [
    "Vector3",
    "WorldObject",
    "TerrainCell",
    "World",
    "RigidBody",
    "DynamicsEngine",
    "SensorConfig",
    "SensorReading",
    "SensorSimulator",
    "CommLink",
    "Message",
    "CommSimulator",
    "ScenarioObject",
    "ScenarioConfig",
    "ScenarioBuilder",
    "MonteCarloResult",
    "SimulationRun",
    "SimulationStatistics",
]
