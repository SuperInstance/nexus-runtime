"""
AI Perception Pipeline Configuration for NEXUS Marine Robotics.

Provides configurable object detection, segmentation, depth estimation,
and tracking pipeline settings optimized for Jetson hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class ObjectDetector(str, Enum):
    """Supported object detector model variants for NEXUS perception."""
    YOLOV5N = "yolov5n"
    YOLOV8S = "yolov8s"
    YOLOV8M = "yolov8m"
    YOLOV8L = "yolov8l"


@dataclass(frozen=True)
class CameraConfig:
    """Camera hardware configuration for NEXUS marine vision."""
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 15
    codec: str = "H264"
    underwater_mode: bool = True
    iso_range: Tuple[int, int] = (100, 1600)
    exposure_ms: Tuple[int, int] = (1, 33)


@dataclass
class ModelConfig:
    """ML model inference configuration."""
    model_name: str = "yolov8s"
    input_size: int = 640
    confidence: float = 0.5
    nms_threshold: float = 0.45
    max_detections: int = 100


@dataclass
class PerceptionPipeline:
    """
    Configurable multi-stage perception pipeline.

    Chains detection, segmentation, depth estimation, and tracking
    stages for real-time marine perception.
    """
    detection: ModelConfig = field(default_factory=ModelConfig)
    segmentation: ModelConfig = field(default_factory=ModelConfig)
    depth_estimation: bool = True
    tracking: bool = True
    stages: List[str] = field(
        default_factory=lambda: ["detection", "tracking"]
    )


def get_pipeline_profile(profile: str) -> PerceptionPipeline:
    """
    Return a pre-configured PerceptionPipeline for the named profile.

    Args:
        profile: One of 'low_power', 'balanced', or 'high_performance'.

    Returns:
        PerceptionPipeline tuned for the requested performance tier.

    Raises:
        ValueError: If profile name is not recognised.
    """
    profiles: Dict[str, PerceptionPipeline] = {
        "low_power": PerceptionPipeline(
            detection=ModelConfig(
                model_name="yolov5n",
                input_size=416,
                confidence=0.4,
                nms_threshold=0.5,
                max_detections=50,
            ),
            segmentation=ModelConfig(
                model_name="yolov5n",
                input_size=416,
                confidence=0.35,
                nms_threshold=0.5,
                max_detections=30,
            ),
            depth_estimation=False,
            tracking=False,
            stages=["detection"],
        ),
        "balanced": PerceptionPipeline(
            detection=ModelConfig(
                model_name="yolov8s",
                input_size=640,
                confidence=0.5,
                nms_threshold=0.45,
                max_detections=100,
            ),
            segmentation=ModelConfig(
                model_name="yolov8s",
                input_size=640,
                confidence=0.4,
                nms_threshold=0.45,
                max_detections=80,
            ),
            depth_estimation=True,
            tracking=True,
            stages=["detection", "tracking"],
        ),
        "high_performance": PerceptionPipeline(
            detection=ModelConfig(
                model_name="yolov8l",
                input_size=1280,
                confidence=0.5,
                nms_threshold=0.4,
                max_detections=200,
            ),
            segmentation=ModelConfig(
                model_name="yolov8l",
                input_size=1280,
                confidence=0.45,
                nms_threshold=0.4,
                max_detections=150,
            ),
            depth_estimation=True,
            tracking=True,
            stages=["detection", "segmentation", "depth_estimation", "tracking"],
        ),
    }

    if profile not in profiles:
        raise ValueError(
            f"Unknown pipeline profile '{profile}'. "
            f"Choose from: {', '.join(sorted(profiles.keys()))}"
        )

    return profiles[profile]
