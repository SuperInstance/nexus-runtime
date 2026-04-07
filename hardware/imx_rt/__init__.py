"""
NEXUS NXP i.MX RT Hardware Configuration Library

High-performance crossover MCU configurations for the NEXUS distributed
intelligence platform. Supports i.MX RT1060, RT1064, RT1170, and RT1050
for real-time marine sensor processing, motor control, and edge AI inference.
"""

from .config_imxrt1060 import (
    IMXRT1060Config,
    CORE_COUNT as RT1060_CORES,
    CPU_FREQ_MAX as RT1060_FREQ_MAX,
    FLASH_TOTAL as RT1060_FLASH,
    SRAM_TOTAL as RT1060_SRAM,
)
from .config_imxrt1064 import (
    IMXRT1064Config,
    CORE_COUNT as RT1064_CORES,
    CPU_FREQ_MAX as RT1064_FREQ_MAX,
    FLASH_TOTAL as RT1064_FLASH,
    SRAM_TOTAL as RT1064_SRAM,
)
from .config_imxrt1170 import (
    IMXRT1170Config,
    APP_CORE_FREQ_MAX as RT1170_APP_FREQ,
    NET_CORE_FREQ_MAX as RT1170_NET_FREQ,
    APP_SRAM_TOTAL as RT1170_APP_SRAM,
    NET_SRAM_TOTAL as RT1170_NET_SRAM,
)
from .config_imxrt1050 import (
    IMXRT1050Config,
    CORE_COUNT as RT1050_CORES,
    CPU_FREQ_MAX as RT1050_FREQ_MAX,
    FLASH_TOTAL as RT1050_FLASH,
    SRAM_TOTAL as RT1050_SRAM,
)

BOARD_REGISTRY = {
    "imxrt1060": {"name": "i.MX RT1060", "class": IMXRT1060Config,
                   "cpu": "Cortex-M7 @ 600MHz", "sram": "1 MB"},
    "imxrt1064": {"name": "i.MX RT1064", "class": IMXRT1064Config,
                   "cpu": "Cortex-M7 @ 600MHz", "sram": "1 MB", "flash": "4 MB onboard"},
    "imxrt1170": {"name": "i.MX RT1170", "class": IMXRT1170Config,
                   "cpu": "Dual M7@1GHz + M4@400MHz", "sram": "3.5 MB"},
    "imxrt1050": {"name": "i.MX RT1050", "class": IMXRT1050Config,
                   "cpu": "Cortex-M7 @ 600MHz", "sram": "512 KB"},
}

__all__ = [
    "IMXRT1060Config", "IMXRT1064Config", "IMXRT1170Config", "IMXRT1050Config",
    "BOARD_REGISTRY",
]
