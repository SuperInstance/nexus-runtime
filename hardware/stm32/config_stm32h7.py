"""
NEXUS STM32H743 Configuration Module

Complete peripheral, clock, DMA, and power configuration for the STM32H743VI
(Cortex-M7 @ 480 MHz, 1 MB Flash, 1 MB SRAM, double-precision FPU).

Target applications:
    - NEXUS Nav Computer: High-rate sensor fusion, INS, DVL processing
    - NEXUS Vision Processor: Underwater image preprocessing
    - NEXUS High-Speed DAQ: Multi-channel ADC sampling at 1+ MSPS
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class H7ClockSource(Enum):
    HSI = "hsi"           # Internal 64 MHz RC
    HSE = "hse"           # External 25 MHz crystal (typical for H7)
    CSI = "csi"           # Internal 4 MHz RC
    LSE = "lse"           # External 32.768 kHz
    LSI = "lsi"           # Internal 32 kHz


class H7PLLState(Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"


class H7VoltageScale(Enum):
    VOS0 = "vos0"   # 1.26V – Overdrive, up to 480 MHz
    VOS1 = "vos1"   # 1.20V – up to 400 MHz
    VOS2 = "vos2"   # 1.10V – up to 300 MHz
    VOS3 = "vos3"   # 1.00V – up to 200 MHz


class H7PowerMode(Enum):
    RUN = "run"
    LOW_POWER_RUN = "low_power_run"
    SLEEP = "sleep"
    STOP0 = "stop0"
    STOP1 = "stop1"
    STOP2 = "stop2"
    STANDBY = "standby"
    SHUTDOWN = "shutdown"


class AXIBus(Enum):
    D1 = "d1_domain"   # Cortex-M7 bus matrix
    D2 = "d2_domain"   # DMA, peripheral bus
    D3 = "d3_domain"   # Backup domain, low-power


class DMAPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class H7PLLConfig:
    """Individual PLL configuration (PLL1, PLL2, or PLL3)."""
    pll_name: str = "PLL1"
    source: H7ClockSource = H7ClockSource.HSE
    enabled: bool = True
    divm: int = 5            # Input divider (1..63)
    divn: int = 192          # VCO multiplier (4..512)
    divp: int = 2            # P output divider (1..128)
    divq: int = 4            # Q output divider (1..128)
    divr: int = 2            # R output divider (1..128)
    fracn: int = 0           # Fractional part (0..8191)
    frac_en: bool = False

    @property
    def vco_in_hz(self) -> int:
        base = 25_000_000  # Default HSE
        if self.source == H7ClockSource.HSI:
            base = 64_000_000
        elif self.source == H7ClockSource.CSI:
            base = 4_000_000
        return base // self.divm

    @property
    def vco_out_hz(self) -> int:
        if self.frac_en:
            return int(self.vco_in_hz * (self.divn + self.fracn / 8192.0))
        return self.vco_in_hz * self.divn

    @property
    def p_freq_hz(self) -> int:
        return self.vco_out_hz // self.divp if self.divp > 0 else 0

    @property
    def q_freq_hz(self) -> int:
        return self.vco_out_hz // self.divq if self.divq > 0 else 0

    @property
    def r_freq_hz(self) -> int:
        return self.vco_out_hz // self.divr if self.divr > 0 else 0

    def validate(self) -> List[str]:
        errors = []
        if self.divm < 1 or self.divm > 63:
            errors.append(f"{self.pll_name}: DIVM {self.divm} out of [1, 63]")
        if self.divn < 4 or self.divn > 512:
            errors.append(f"{self.pll_name}: DIVN {self.divn} out of [4, 512]")
        for name, val in [("DIVP", self.divp), ("DIVQ", self.divq), ("DIVR", self.divr)]:
            if val < 1 or val > 128:
                errors.append(f"{self.pll_name}: {name} {val} out of [1, 128]")
        vco_in = self.vco_in_hz
        if vco_in < 1_000_000 or vco_in > 16_000_000:
            errors.append(f"{self.pll_name}: VCO input {vco_in} out of [1 MHz, 16 MHz]")
        vco_out = self.vco_out_hz
        if vco_out < 128_000_000 or vco_out > 960_000_000:
            errors.append(f"{self.pll_name}: VCO output {vco_out} out of [128 MHz, 960 MHz]")
        return errors


@dataclass
class H7ClockConfig:
    """Full clock tree for STM32H743."""
    source: H7ClockSource = H7ClockSource.HSE
    hse_freq_hz: int = 25_000_000
    hsi_freq_hz: int = 64_000_000
    pll1: H7PLLConfig = field(default_factory=lambda: H7PLLConfig(pll_name="PLL1", divm=5, divn=192, divp=2))
    pll2: H7PLLConfig = field(default_factory=lambda: H7PLLConfig(pll_name="PLL2", divm=5, divn=192, divq=4))
    pll3: H7PLLConfig = field(default_factory=lambda: H7PLLConfig(pll_name="PLL3", divm=5, divn=192, divr=4))
    d1cpre: int = 1          # Cortex-M7 prescaler (1, 2, 4, 8, 16, 64, 128, 256, 512)
    hpre: int = 2            # AHB prescaler (1..512)
    d1ppre: int = 4          # APB3 prescaler
    d2ppre1: int = 4         # APB1 prescaler
    d2ppre2: int = 4         # APB2 prescaler
    d3ppre: int = 4          # APB4 prescaler
    enable_css: bool = True  # Clock security system
    enable_overdrive: bool = True

    @property
    def sysclk_hz(self) -> int:
        if not self.pll1.enabled:
            return self.hse_freq_hz
        return self.pll1.p_freq_hz // self.d1cpre

    @property
    def hclk_hz(self) -> int:
        return self.sysclk_hz // self.hpre

    @property
    def apb1_hz(self) -> int:
        return self.hclk_hz // self.d2ppre1

    @property
    def apb2_hz(self) -> int:
        return self.hclk_hz // self.d2ppre2

    @property
    def apb3_hz(self) -> int:
        return self.hclk_hz // self.d1ppre

    @property
    def apb4_hz(self) -> int:
        return self.hclk_hz // self.d3ppre

    def validate(self) -> List[str]:
        errors = []
        errors.extend(self.pll1.validate())
        errors.extend(self.pll2.validate())
        errors.extend(self.pll3.validate())
        if self.sysclk_hz > 480_000_000:
            errors.append(f"SYSCLK {self.sysclk_hz} exceeds 480 MHz max")
        return errors


@dataclass
class H7MemoryLayout:
    """STM32H743VI memory map: 1 MB Flash, 1 MB SRAM (split across multiple banks)."""
    flash_start: int = 0x08000000
    flash_size_kb: int = 1024
    flash_sector_size_kb: int = 128
    itcm_start: int = 0x00000000
    itcm_size_kb: int = 64
    dtcm_start: int = 0x20000000
    dtcm_size_kb: int = 128
    axi_sram_start: int = 0x24000000
    axi_sram_size_kb: int = 512
    sram1_start: int = 0x30000000
    sram1_size_kb: int = 128
    sram2_start: int = 0x30020000
    sram2_size_kb: int = 128
    backup_sram_start: int = 0x38000000
    backup_sram_size_kb: int = 4

    @property
    def total_sram_kb(self) -> int:
        return (
            self.itcm_size_kb + self.dtcm_size_kb +
            self.axi_sram_size_kb + self.sram1_size_kb +
            self.sram2_size_kb + self.backup_sram_size_kb
        )


@dataclass
class H7PowerConfig:
    """STM32H7 power domain configuration."""
    vos: H7VoltageScale = H7VoltageScale.VOS0
    scuen: bool = True          # Supply configuration update enable
    ionen: bool = True          # Overdrive enable
    enable_backup_domain: bool = True
    enable_rtc: bool = True
    brown_out_mv: float = 2700.0
    iwdg_timeout_ms: int = 100
    wwdg_window_ms: int = 30


@dataclass
class H7DMAConfig:
    """H7 has BDMA, DMA1, DMA2 with different bus masters."""
    name: str = "DMA1"
    stream: int = 0            # 0..7
    channel: int = 0           # 0..7 (or request number)
    priority: DMAPriority = DMAPriority.MEDIUM
    circular: bool = False
    peripheral_inc: bool = False
    memory_inc: bool = True
    p_data_size: int = 1
    m_data_size: int = 1
    direction: str = "peripheral_to_memory"


@dataclass
class H7CacheConfig:
    """L1 cache configuration for Cortex-M7."""
    icache_enabled: bool = True
    dcache_enabled: bool = True
    icache_way_size_kb: int = 16
    dcache_way_size_kb: int = 16


@dataclass
class STM32H743Config:
    """
    Complete hardware configuration for STM32H743VI.

    Cortex-M7 @ 480 MHz, 1 MB Flash, 1 MB SRAM,
    double-precision FPU, L1 I/D cache, 2× CAN-FD, 3× ADC (16-bit),
    8× USART, 6× SPI, 2× USB OTG.

    Typical use in NEXUS:
        config = STM32H743Config()
        config.validate()
    """
    part_number: str = "STM32H743VI"
    core: str = "Cortex-M7F"
    max_sysclk_hz: int = 480_000_000
    flash_kb: int = 1024
    sram_kb: int = 964        # Total across all SRAM banks
    dp_fpu: bool = True       # Double-precision FPU

    clock: H7ClockConfig = field(default_factory=H7ClockConfig)
    power: H7PowerConfig = field(default_factory=H7PowerConfig)
    memory: H7MemoryLayout = field(default_factory=H7MemoryLayout)
    cache: H7CacheConfig = field(default_factory=H7CacheConfig)

    dma_configs: List[H7DMAConfig] = field(default_factory=list)

    node_name: str = ""
    node_id: int = 0
    firmware_version: str = "0.1.0"

    def validate(self) -> List[str]:
        """Validate all sub-configurations."""
        errors = []
        errors.extend(self.clock.validate())
        if self.sram_kb != self.memory.total_sram_kb:
            errors.append(
                f"SRAM size mismatch: config {self.sram_kb} vs layout {self.memory.total_sram_kb}"
            )
        if self.power.vos == H7VoltageScale.VOS0 and not self.power.ionen:
            errors.append("Overdrive must be enabled for VOS0")
        return errors

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def clone(self) -> "STM32H743Config":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"STM32H743Config(part='{self.part_number}', "
            f"core={self.core}, "
            f"sysclk={self.clock.sysclk_hz // 1_000_000}MHz, "
            f"flash={self.flash_kb}KB, "
            f"sram={self.sram_kb}KB, "
            f"fpu=dp)"
        )

    def summary(self) -> str:
        lines = [
            f"NEXUS STM32H743 Configuration",
            f"  Part:          {self.part_number}",
            f"  Core:          {self.core}",
            f"  SYSCLK:        {self.clock.sysclk_hz / 1e6:.1f} MHz",
            f"  HCLK:          {self.clock.hclk_hz / 1e6:.1f} MHz",
            f"  APB1:          {self.clock.apb1_hz / 1e6:.1f} MHz",
            f"  APB2:          {self.clock.apb2_hz / 1e6:.1f} MHz",
            f"  APB3:          {self.clock.apb3_hz / 1e6:.1f} MHz",
            f"  APB4:          {self.clock.apb4_hz / 1e6:.1f} MHz",
            f"  Flash:         {self.flash_kb} KB",
            f"  AXI SRAM:      {self.memory.axi_sram_size_kb} KB",
            f"  DTCM:          {self.memory.dtcm_size_kb} KB",
            f"  ITCM:          {self.memory.itcm_size_kb} KB",
            f"  SRAM1:         {self.memory.sram1_size_kb} KB",
            f"  SRAM2:         {self.memory.sram2_size_kb} KB",
            f"  Backup SRAM:   {self.memory.backup_sram_size_kb} KB",
            f"  I-Cache:       {'enabled' if self.cache.icache_enabled else 'disabled'}",
            f"  D-Cache:       {'enabled' if self.cache.dcache_enabled else 'disabled'}",
            f"  DP FPU:        {'yes' if self.dp_fpu else 'no'}",
            f"  Node:          {self.node_name or 'unnamed'} (id={self.node_id})",
            f"  Firmware:      v{self.firmware_version}",
        ]
        return "\n".join(lines)
