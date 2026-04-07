"""
NEXUS RP2040 Hardware Library

Marine robotics microcontroller configuration for the RP2040 platform.
Includes base RP2040 config, Pico W wireless extension, and PIO programs.
"""

from .config_rp2040 import (
    RP2040Config,
    ClockConfig,
    PinMapping,
    PinFunction,
    GPIOPin,
    MemoryLayout,
    MemoryRegion,
    CORE_COUNT,
    CPU_FREQ_MAX,
    CPU_FREQ_DEFAULT,
    SRAM_TOTAL,
    FLASH_TOTAL,
    PIO_COUNT,
    SM_PER_PIO,
    GPIO_COUNT,
)

from .config_pico_w import (
    PicoWConfig,
    WiFiConfig,
    BLEConfig,
    MQTTConfig,
    WiFiSecurity,
    BLERole,
)

from .pio_programs import (
    SonarPingProgram,
    SonarTimingParams,
    ServoPWMProgram,
    ServoPWMParams,
    UARTBridgeProgram,
    UARTBridgeParams,
    PIOProgramRegistry,
    PIOInstruction,
)

__all__ = [
    # Core config
    "RP2040Config", "ClockConfig", "PinMapping", "PinFunction", "GPIOPin",
    "MemoryLayout", "MemoryRegion",
    # Constants
    "CORE_COUNT", "CPU_FREQ_MAX", "CPU_FREQ_DEFAULT", "SRAM_TOTAL",
    "FLASH_TOTAL", "PIO_COUNT", "SM_PER_PIO", "GPIO_COUNT",
    # Pico W
    "PicoWConfig", "WiFiConfig", "BLEConfig", "MQTTConfig",
    "WiFiSecurity", "BLERole",
    # PIO Programs
    "SonarPingProgram", "SonarTimingParams",
    "ServoPWMProgram", "ServoPWMParams",
    "UARTBridgeProgram", "UARTBridgeParams",
    "PIOProgramRegistry", "PIOInstruction",
]
