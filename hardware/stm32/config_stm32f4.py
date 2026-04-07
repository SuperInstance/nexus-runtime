"""
NEXUS STM32F407 Configuration Module

Complete peripheral, clock, DMA, and power configuration for the STM32F407VG
(Cortex-M4 @ 168 MHz, 1 MB Flash, 192 KB SRAM, single-precision FPU).

Target applications:
    - NEXUS Thruster Node v2: 4-channel BLDC motor controller
    - NEXUS Sensor Hub v3: IMU, pressure, CTD, DVL sensor interface
    - NEXUS CAN Bridge: NMEA 2000 marine networking gateway
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClockSource(Enum):
    HSI = "hsi"       # Internal 16 MHz RC oscillator
    HSE = "hse"       # External crystal (8 MHz typical)
    LSE = "lse"       # External 32.768 kHz crystal
    LSI = "lsi"       # Internal 32 kHz RC


class PLLState(Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"


class APBBus(Enum):
    APB1 = "apb1"     # Max 42 MHz
    APB2 = "apb2"     # Max 84 MHz


class DMAStream(Enum):
    """DMA1 / DMA2 stream identifiers used in motor controller and sensor paths."""
    DMA1_S0 = "dma1_stream0"
    DMA1_S1 = "dma1_stream1"
    DMA1_S2 = "dma1_stream2"
    DMA1_S3 = "dma1_stream3"
    DMA1_S4 = "dma1_stream4"
    DMA1_S5 = "dma1_stream5"
    DMA1_S6 = "dma1_stream6"
    DMA1_S7 = "dma1_stream7"
    DMA2_S0 = "dma2_stream0"
    DMA2_S1 = "dma2_stream1"
    DMA2_S2 = "dma2_stream2"
    DMA2_S3 = "dma2_stream3"
    DMA2_S4 = "dma2_stream4"
    DMA2_S5 = "dma2_stream5"
    DMA2_S6 = "dma2_stream6"
    DMA2_S7 = "dma2_stream7"


class DMAPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class GPIOMode(Enum):
    INPUT = "input"
    OUTPUT_PP = "output_push_pull"
    OUTPUT_OD = "open_drain"
    AF_PP = "alternate_function_push_pull"
    AF_OD = "alternate_function_open_drain"
    ANALOG = "analog"


class GPIOPull(Enum):
    NONE = "no_pull"
    UP = "pull_up"
    DOWN = "pull_down"


class PowerMode(Enum):
    RUN = "run"
    SLEEP = "sleep"
    STOP = "stop"
    STANDBY = "standby"


class VoltageScale(Enum):
    SCALE_1 = "scale_1"   # 1.8V – for 168 MHz
    SCALE_2 = "scale_2"   # 1.5V – for 144 MHz
    SCALE_3 = "scale_3"   # 1.2V – for <= 120 MHz


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClockConfig:
    """PLL and bus clock configuration for STM32F407."""
    source: ClockSource = ClockSource.HSE
    hse_freq_hz: int = 8_000_000
    pllm: int = 8        # VCO input divider (1..63)
    plln: int = 336      # VCO multiplier (50..432)
    pllp: int = 2        # SYSCLK divider (2, 4, 6, 8)
    pllq: int = 7        # 48 MHz USB clock divider (2..15)
    pll_state: PLLState = PLLState.ENABLED
    enable_overdrive: bool = False

    @property
    def sysclk_hz(self) -> int:
        if self.pll_state == PLLState.DISABLED:
            return self.hse_freq_hz
        if self.pllm == 0:
            return 0
        vco_in = self.hse_freq_hz // self.pllm
        vco_out = vco_in * self.plln
        return vco_out // self.pllp

    @property
    def hclk_hz(self) -> int:
        """AHB bus clock (typically = SYSCLK)."""
        return self.sysclk_hz  # AHB prescaler = 1

    @property
    def apb1_hz(self) -> int:
        """APB1 peripheral clock (max 42 MHz)."""
        return self.hclk_hz // 4  # prescaler /4

    @property
    def apb2_hz(self) -> int:
        """APB2 peripheral clock (max 84 MHz)."""
        return self.hclk_hz // 2  # prescaler /2

    @property
    def usb_hz(self) -> int:
        """USB OTG FS 48 MHz clock from PLLQ."""
        return (self.hse_freq_hz // self.pllm) * self.plln // self.pllq

    def validate(self) -> List[str]:
        """Return list of constraint violations (empty if valid)."""
        errors = []
        if self.pllm < 1 or self.pllm > 63:
            errors.append(f"PLLM {self.pllm} out of range [1, 63]")
            return errors  # Early return to avoid division by zero
        if self.plln < 50 or self.plln > 432:
            errors.append(f"PLLN {self.plln} out of range [50, 432]")
        if self.pllp not in (2, 4, 6, 8):
            errors.append(f"PLLP {self.pllp} must be in (2, 4, 6, 8)")
        if self.pllq < 2 or self.pllq > 15:
            errors.append(f"PLLQ {self.pllq} out of range [2, 15]")
        if self.sysclk_hz > 168_000_000:
            errors.append(f"SYSCLK {self.sysclk_hz} exceeds 168 MHz")
        if self.apb1_hz > 42_000_000:
            errors.append(f"APB1 {self.apb1_hz} exceeds 42 MHz")
        if self.apb2_hz > 84_000_000:
            errors.append(f"APB2 {self.apb2_hz} exceeds 84 MHz")
        vco_in = self.hse_freq_hz // self.pllm
        if vco_in < 1_000_000 or vco_in > 2_000_000:
            errors.append(f"VCO input {vco_in} Hz out of [1 MHz, 2 MHz]")
        return errors


@dataclass
class DMAStreamConfig:
    """Single DMA stream configuration."""
    stream: DMAStream
    channel: int = 0           # DMA channel 0..7
    priority: DMAPriority = DMAPriority.MEDIUM
    circular: bool = False
    peripheral_inc: bool = False
    memory_inc: bool = True
    peripheral_data_size: int = 1   # bytes: 1, 2, or 4
    memory_data_size: int = 1
    direction: str = "peripheral_to_memory"
    transfer_complete_irq: bool = True
    half_transfer_irq: bool = False
    error_irq: bool = True


@dataclass
class GPIOConfig:
    """Pin-level GPIO configuration."""
    port: str             # e.g. "A", "B", "C", "D"
    pin: int              # 0..15
    mode: GPIOMode = GPIOMode.INPUT
    pull: GPIOPull = GPIOPull.NONE
    speed: str = "50MHz"  # "2MHz", "25MHz", "50MHz", "100MHz"
    alternate: int = 0    # AF function number

    @property
    def pin_name(self) -> str:
        return f"P{self.port}{self.pin}"


@dataclass
class CANPeripheralConfig:
    """CAN bus peripheral register-level settings for STM32F4."""
    instance: str = "CAN1"   # CAN1 or CAN2
    baud_rate: int = 250_000
    prescaler: int = 14
    time_seg1: int = 13      # Time Quanta in Bit Segment 1
    time_seg2: int = 2       # Time Quanta in Bit Segment 2
    sjw: int = 1             # Synchronisation Jump Width
    auto_bus_off: bool = True
    auto_wake_up: bool = False
    auto_retransmit: bool = True
    rx_fifo_locked: bool = False
    tx_fifo_priority: bool = False

    @property
    def bit_rate(self) -> int:
        """Actual bit rate from APB1 clock / (prescaler * (1 + time_seg1 + time_seg2))"""
        apb1 = 42_000_000  # standard APB1
        return apb1 // (self.prescaler * (1 + self.time_seg1 + self.time_seg2))

    @property
    def sample_point_pct(self) -> float:
        total = 1 + self.time_seg1 + self.time_seg2
        return ((1 + self.time_seg1) / total) * 100.0


@dataclass
class UARTConfig:
    """USART peripheral configuration."""
    instance: str = "USART1"
    baud_rate: int = 115200
    word_length: int = 8
    stop_bits: int = 1
    parity: str = "none"     # "none", "even", "odd"
    hw_flow_control: bool = False
    dma_rx_stream: Optional[DMAStream] = None
    dma_tx_stream: Optional[DMAStream] = None


@dataclass
class ADCConfig:
    """ADC peripheral configuration for sensor sampling."""
    instance: str = "ADC1"
    resolution_bits: int = 12  # 6, 8, 10, or 12
    sample_time_cycles: int = 84
    scan_mode: bool = True
    continuous: bool = False
    dma_stream: Optional[DMAStream] = None
    channels: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    def validate(self) -> List[str]:
        errors = []
        if self.resolution_bits not in (6, 8, 10, 12):
            errors.append(f"ADC resolution {self.resolution_bits} must be 6, 8, 10, or 12")
        if len(self.channels) > 16:
            errors.append(f"Too many ADC channels: {len(self.channels)} > 16")
        if self.sample_time_cycles < 3 or self.sample_time_cycles > 480:
            errors.append(f"Sample time {self.sample_time_cycles} out of [3, 480]")
        return errors


@dataclass
class PowerConfig:
    """Power domain and regulator configuration."""
    voltage_scale: VoltageScale = VoltageScale.SCALE_1
    brown_out_threshold_mv: float = 2700.0
    enable_backup_domain: bool = True
    enable_rtc: bool = True
    enable_wwdg: bool = True
    enable_iwdg: bool = True
    iwdg_timeout_ms: int = 100
    wwdg_window_ms: int = 30


@dataclass
class MemoryLayout:
    """Flash and SRAM memory map for the STM32F407VG."""
    flash_start: int = 0x08000000
    flash_size_kb: int = 1024
    flash_sector_size_kb: int = 128  # sectors 0..3 are 16KB, 64KB, 128KB, 128KB; 4..7 are 128KB
    sram_start: int = 0x20000000
    sram_size_kb: int = 128         # 128 KB main SRAM
    ccmram_start: int = 0x10000000  # 64 KB CCM RAM (core-coupled)
    ccmram_size_kb: int = 64
    backup_sram_start: int = 0x40024000  # 4 KB backup SRAM
    backup_sram_size_kb: int = 4

    @property
    def total_ram_kb(self) -> int:
        return self.sram_size_kb + self.ccmram_size_kb


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class STM32F407Config:
    """
    Complete hardware configuration for STM32F407VG.

    Cortex-M4F @ 168 MHz, 1 MB Flash, 192 KB RAM (128 KB SRAM + 64 KB CCM),
    single-precision FPU, DMA2 with 16 streams, 2× CAN, 3× ADC, 6× USART,
    3× SPI, USB OTG FS.

    Typical use in NEXUS:
        config = STM32F407Config()
        config.validate()
    """
    # Device identity
    part_number: str = "STM32F407VG"
    core: str = "Cortex-M4F"
    max_sysclk_hz: int = 168_000_000
    flash_kb: int = 1024
    sram_kb: int = 192   # 128 + 64 CCM

    # Sub-configs
    clock: ClockConfig = field(default_factory=ClockConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    memory: MemoryLayout = field(default_factory=MemoryLayout)

    # Peripheral collections
    dma_streams: List[DMAStreamConfig] = field(default_factory=list)
    gpios: List[GPIOConfig] = field(default_factory=list)
    can_peripherals: List[CANPeripheralConfig] = field(default_factory=list)
    uarts: List[UARTConfig] = field(default_factory=list)
    adcs: List[ADCConfig] = field(default_factory=list)

    # Metadata
    node_name: str = ""
    node_id: int = 0
    firmware_version: str = "0.1.0"

    def __post_init__(self):
        if not self.can_peripherals:
            self.can_peripherals = [CANPeripheralConfig()]
        if not self.uarts:
            self.uarts = [UARTConfig(instance="USART1", baud_rate=115200)]
        if not self.adcs:
            self.adcs = [ADCConfig()]

    def validate(self) -> List[str]:
        """Validate all sub-configurations. Returns list of error strings."""
        errors = []
        errors.extend(self.clock.validate())
        for adc in self.adcs:
            errors.extend(adc.validate())
        if self.flash_kb != self.memory.flash_size_kb:
            errors.append(
                f"Flash size mismatch: config {self.flash_kb} vs memory layout {self.memory.flash_size_kb}"
            )
        if self.sram_kb != self.memory.total_ram_kb:
            errors.append(
                f"SRAM size mismatch: config {self.sram_kb} vs memory layout {self.memory.total_ram_kb}"
            )
        return errors

    def to_dict(self) -> dict:
        """Serialise configuration to a plain dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    def clone(self) -> "STM32F407Config":
        """Return a deep copy of this configuration."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"STM32F407Config(part='{self.part_number}', "
            f"core={self.core}, "
            f"sysclk={self.clock.sysclk_hz // 1_000_000}MHz, "
            f"flash={self.flash_kb}KB, "
            f"sram={self.sram_kb}KB, "
            f"fpu=sp)"
        )

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            f"NEXUS STM32F407 Configuration",
            f"  Part:          {self.part_number}",
            f"  Core:          {self.core}",
            f"  SYSCLK:        {self.clock.sysclk_hz / 1e6:.1f} MHz",
            f"  HCLK (AHB):    {self.clock.hclk_hz / 1e6:.1f} MHz",
            f"  APB1:          {self.clock.apb1_hz / 1e6:.1f} MHz",
            f"  APB2:          {self.clock.apb2_hz / 1e6:.1f} MHz",
            f"  USB:           {self.clock.usb_hz / 1e6:.1f} MHz",
            f"  Flash:         {self.flash_kb} KB ({self.memory.flash_start:#010x})",
            f"  SRAM:          {self.memory.sram_size_kb} KB ({self.memory.sram_start:#010x})",
            f"  CCM RAM:       {self.memory.ccmram_size_kb} KB ({self.memory.ccmram_start:#010x})",
            f"  Backup SRAM:   {self.memory.backup_sram_size_kb} KB",
            f"  CAN instances: {len(self.can_peripherals)}",
            f"  UART instances: {len(self.uarts)}",
            f"  ADC instances:  {len(self.adcs)}",
            f"  DMA streams:   {len(self.dma_streams)}",
            f"  GPIO configs:  {len(self.gpios)}",
            f"  Node:          {self.node_name or 'unnamed'} (id={self.node_id})",
            f"  Firmware:      v{self.firmware_version}",
        ]
        return "\n".join(lines)
