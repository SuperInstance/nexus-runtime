"""
NEXUS STM32 Marine Robotics Configuration Library

Provides hardware configuration objects for STM32F4 and STM32H7 microcontrollers
deployed in NEXUS distributed intelligence platform for marine robotics.

Modules:
    config_stm32f4: STM32F407 (Cortex-M4 168MHz) peripheral & clock configuration
    config_stm32h7: STM32H743 (Cortex-M7 480MHz) peripheral & clock configuration
    can_bus:         CAN bus configuration for NMEA 2000 marine networking
    motor_control:   Brushless DC motor controller & thruster configuration
"""

__version__ = "1.0.0"
__author__ = "NEXUS Marine Robotics Team"

from hardware.stm32.config_stm32f4 import STM32F407Config
from hardware.stm32.config_stm32h7 import STM32H743Config
from hardware.stm32.can_bus import CANConfig, CANNodeConfig, PGNDefinition
from hardware.stm32.motor_control import ESCConfig, PIDParams, ThrusterConfig

__all__ = [
    "STM32F407Config",
    "STM32H743Config",
    "CANConfig",
    "CANNodeConfig",
    "PGNDefinition",
    "ESCConfig",
    "PIDParams",
    "ThrusterConfig",
]
