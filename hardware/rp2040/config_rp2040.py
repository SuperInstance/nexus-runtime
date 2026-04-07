"""
NEXUS RP2040 Hardware Configuration Module

Provides pin mapping, clock configuration, and marine sensor setup
for the RP2040 dual-core Cortex-M0+ @ 133MHz with 264KB SRAM.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORE_COUNT = 2
CPU_FREQ_MAX = 133_000_000          # 133 MHz
CPU_FREQ_DEFAULT = 125_000_000      # 125 MHz (safe default)
SRAM_TOTAL = 264 * 1024             # 264 KB in bytes
FLASH_TOTAL = 2 * 1024 * 1024       # 2 MB (external QSPI)
PIO_COUNT = 2                       # PIO blocks (PIO0, PIO1)
SM_PER_PIO = 4                      # State machines per PIO block
GPIO_COUNT = 30                     # GPIO 0-29 on RP2040


class GPIOPin(IntEnum):
    """RP2040 GPIO pin identifiers used in NEXUS marine sensor configs."""
    GP0 = 0
    GP1 = 1
    GP2 = 2
    GP3 = 3
    GP4 = 4
    GP5 = 5
    GP6 = 6
    GP7 = 7
    GP8 = 8
    GP9 = 9
    GP10 = 10
    GP11 = 11
    GP12 = 12
    GP13 = 13
    GP14 = 14
    GP15 = 15
    GP16 = 16
    GP17 = 17
    GP18 = 18
    GP19 = 19
    GP20 = 20
    GP21 = 21
    GP22 = 22
    GP23 = 23
    GP24 = 24
    GP25 = 25
    GP26 = 26
    GP27 = 27
    GP28 = 28
    GP29 = 29


class PinFunction(IntEnum):
    """Marine sensor function identifiers."""
    SONAR_TRIG = 0
    SONAR_ECHO = 1
    SERVO_1 = 2
    SERVO_2 = 3
    SERVO_3 = 4
    SERVO_4 = 5
    I2C_SDA = 10
    I2C_SCL = 11
    UART_TX = 20
    UART_RX = 21
    SPI_MOSI = 30
    SPI_MISO = 31
    SPI_CLK = 32
    SPI_CS = 33
    ADC_TEMP = 40
    ADC_DEPTH = 41
    STATUS_LED = 50


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PinMapping:
    """Maps a PinFunction to a physical GPIO pin and optional pull-up/down."""
    function: PinFunction
    pin: int
    pull_up: bool = False
    pull_down: bool = False
    invert: bool = False
    description: str = ""

    def __post_init__(self):
        if not (0 <= self.pin <= GPIO_COUNT - 1):
            raise ValueError(
                f"Invalid GPIO pin {self.pin}. RP2040 supports GPIO 0-{GPIO_COUNT - 1}."
            )


@dataclass
class ClockConfig:
    """System clock configuration for the RP2040."""
    frequency_hz: int = CPU_FREQ_DEFAULT
    source: str = "external_xosc"  # xosc, rosc (ring oscillator)
    xosc_freq_hz: int = 12_000_000
    pll_ref_div: int = 1
    pll_fb_div: int = 125
    pll_post_div1: int = 6
    pll_post_div2: int = 2

    @property
    def vco_freq_hz(self) -> int:
        return self.xosc_freq_hz * self.pll_fb_div

    @property
    def actual_output_freq_hz(self) -> int:
        vco = self.vco_freq_hz
        return vco // self.pll_post_div1 // self.pll_post_div2

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.frequency_hz > CPU_FREQ_MAX:
            errors.append(f"Requested {self.frequency_hz} Hz exceeds max {CPU_FREQ_MAX} Hz.")
        if self.frequency_hz < 1_000_000:
            errors.append("Frequency must be >= 1 MHz.")
        if self.xosc_freq_hz not in (12_000_000,):
            errors.append("Only 12 MHz external crystal is currently supported.")
        if self.pll_post_div1 < 1 or self.pll_post_div2 < 1:
            errors.append("PLL post-divisors must be >= 1.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


@dataclass
class MemoryRegion:
    """Describes an SRAM allocation region."""
    name: str
    start: int
    size_bytes: int
    purpose: str = ""

    @property
    def end(self) -> int:
        return self.start + self.size_bytes


@dataclass
class MemoryLayout:
    """Memory layout for the RP2040."""
    regions: List[MemoryRegion] = field(default_factory=list)

    def add_region(self, name: str, start: int, size: int, purpose: str = ""):
        self.regions.append(MemoryRegion(name=name, start=start, size_bytes=size, purpose=purpose))

    def total_allocated(self) -> int:
        return sum(r.size_bytes for r in self.regions)

    def fits_in_sram(self) -> bool:
        return self.total_allocated() <= SRAM_TOTAL

    def region_by_name(self, name: str) -> Optional[MemoryRegion]:
        for r in self.regions:
            if r.name == name:
                return r
        return None

    def has_overlap(self) -> bool:
        sorted_regions = sorted(self.regions, key=lambda r: r.start)
        for i in range(len(sorted_regions) - 1):
            if sorted_regions[i].end > sorted_regions[i + 1].start:
                return True
        return False


# ---------------------------------------------------------------------------
# Main Configuration Class
# ---------------------------------------------------------------------------

class RP2040Config:
    """
    Top-level hardware configuration for the NEXUS RP2040 marine controller.

    Manages pin mapping, clock configuration, PIO state machine allocation,
    and SRAM memory layout for real-time sensor processing.
    """

    # Default pin mapping for NEXUS marine sensor array
    DEFAULT_PIN_MAP: Dict[PinFunction, int] = {
        PinFunction.SONAR_TRIG: GPIOPin.GP0,
        PinFunction.SONAR_ECHO: GPIOPin.GP1,
        PinFunction.SERVO_1: GPIOPin.GP2,
        PinFunction.SERVO_2: GPIOPin.GP3,
        PinFunction.SERVO_3: GPIOPin.GP6,
        PinFunction.SERVO_4: GPIOPin.GP7,
        PinFunction.I2C_SDA: GPIOPin.GP4,
        PinFunction.I2C_SCL: GPIOPin.GP5,
        PinFunction.UART_TX: GPIOPin.GP8,
        PinFunction.UART_RX: GPIOPin.GP9,
        PinFunction.SPI_MOSI: GPIOPin.GP11,
        PinFunction.SPI_MISO: GPIOPin.GP12,
        PinFunction.SPI_CLK: GPIOPin.GP10,
        PinFunction.SPI_CS: GPIOPin.GP13,
        PinFunction.ADC_TEMP: GPIOPin.GP26,
        PinFunction.ADC_DEPTH: GPIOPin.GP27,
        PinFunction.STATUS_LED: GPIOPin.GP25,
    }

    def __init__(self, clock_freq: int = CPU_FREQ_DEFAULT):
        self._pin_map: Dict[PinFunction, PinMapping] = {}
        self.clock = ClockConfig(frequency_hz=clock_freq)
        self.memory = MemoryLayout()
        self._pio_allocated: Dict[str, Tuple[int, int]] = {}  # name -> (pio_block, sm_index)
        self._configure_default_pins()

    # -- Pin Mapping ---------------------------------------------------------

    def _configure_default_pins(self):
        for func, pin in self.DEFAULT_PIN_MAP.items():
            desc = f"{func.name} on GP{pin}"
            self._pin_map[func] = PinMapping(function=func, pin=pin, description=desc)

    def get_pin(self, function: PinFunction) -> PinMapping:
        if function not in self._pin_map:
            raise KeyError(f"Pin function {function.name} not configured.")
        return self._pin_map[function]

    def remap_pin(self, function: PinFunction, new_pin: int, description: str = ""):
        old = self._pin_map.get(function)
        if old:
            old.pin = new_pin
            old.description = description or old.description
        else:
            self._pin_map[function] = PinMapping(function=function, pin=new_pin, description=description)

    def all_pins(self) -> List[PinMapping]:
        return list(self._pin_map.values())

    def used_gpio_set(self) -> set:
        return {pm.pin for pm in self._pin_map.values()}

    # -- Convenience properties ----------------------------------------------

    @property
    def PIN_SONAR_TRIG(self) -> int:
        return self.get_pin(PinFunction.SONAR_TRIG).pin

    @property
    def PIN_SONAR_ECHO(self) -> int:
        return self.get_pin(PinFunction.SONAR_ECHO).pin

    @property
    def PIN_SERVO_1(self) -> int:
        return self.get_pin(PinFunction.SERVO_1).pin

    @property
    def PIN_SERVO_2(self) -> int:
        return self.get_pin(PinFunction.SERVO_2).pin

    @property
    def PIN_I2C_SDA(self) -> int:
        return self.get_pin(PinFunction.I2C_SDA).pin

    @property
    def PIN_I2C_SCL(self) -> int:
        return self.get_pin(PinFunction.I2C_SCL).pin

    @property
    def PIN_UART_TX(self) -> int:
        return self.get_pin(PinFunction.UART_TX).pin

    @property
    def PIN_UART_RX(self) -> int:
        return self.get_pin(PinFunction.UART_RX).pin

    @property
    def PIN_STATUS_LED(self) -> int:
        return self.get_pin(PinFunction.STATUS_LED).pin

    # -- Clock ---------------------------------------------------------------

    def set_clock_frequency(self, freq_hz: int):
        if freq_hz > CPU_FREQ_MAX:
            raise ValueError(f"Max clock is {CPU_FREQ_MAX} Hz, got {freq_hz}.")
        if freq_hz < 1_000_000:
            raise ValueError("Minimum clock frequency is 1 MHz.")
        self.clock.frequency_hz = freq_hz

    def set_pll_params(self, fb_div: int, post_div1: int, post_div2: int):
        self.clock.pll_fb_div = fb_div
        self.clock.pll_post_div1 = post_div1
        self.clock.pll_post_div2 = post_div2

    # -- PIO allocation ------------------------------------------------------

    def allocate_pio(self, name: str, pio_block: int = 0, sm_index: int = 0) -> Tuple[int, int]:
        if pio_block not in (0, 1):
            raise ValueError("PIO block must be 0 or 1.")
        if sm_index not in range(SM_PER_PIO):
            raise ValueError(f"State machine index must be 0-{SM_PER_PIO - 1}.")
        for existing_name, (pb, sm) in self._pio_allocated.items():
            if pb == pio_block and sm == sm_index:
                raise ValueError(f"PIO{pio_block} SM{sm_index} already allocated to '{existing_name}'.")
        self._pio_allocated[name] = (pio_block, sm_index)
        return (pio_block, sm_index)

    def release_pio(self, name: str):
        self._pio_allocated.pop(name, None)

    def pio_allocation_summary(self) -> Dict[str, Tuple[int, int]]:
        return dict(self._pio_allocated)

    def available_sm(self, pio_block: int = 0) -> List[int]:
        used = {sm for (_, sm) in self._pio_allocated.values()}
        return [i for i in range(SM_PER_PIO) if i not in used]

    # -- Memory layout -------------------------------------------------------

    def configure_marine_sensors(self):
        """Set up the default SRAM memory regions for marine sensor processing."""
        self.memory.add_region("stack_core0", 0x20000000, 4096, "Core 0 stack")
        self.memory.add_region("stack_core1", 0x20001000, 4096, "Core 1 stack")
        self.memory.add_region("sensor_ring_buf", 0x20002000, 8192, "Sensor data ring buffer")
        self.memory.add_region("sonar_samples", 0x20004000, 16384, "Sonar return samples")
        self.memory.add_region("uart_rx_buf", 0x20008000, 4096, "UART NMEA receive buffer")
        self.memory.add_region("i2c scratch", 0x20009000, 2048, "I2C temporary buffers")
        self.memory.add_region("servo_state", 0x20009800, 256, "Servo angle state")

    # -- Summary -------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "cpu_cores": CORE_COUNT,
            "clock_hz": self.clock.frequency_hz,
            "sram_total_bytes": SRAM_TOTAL,
            "sram_allocated_bytes": self.memory.total_allocated(),
            "pio_allocated": dict(self._pio_allocated),
            "pins_configured": len(self._pin_map),
        }
