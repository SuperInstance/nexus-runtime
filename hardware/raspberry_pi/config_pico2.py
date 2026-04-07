"""
NEXUS Raspberry Pi Pico 2 Configuration — RP2350 SoC.

Target hardware:
  - SoC:    Raspberry Pi RP2350, Dual-core Cortex-M33 @ 150 MHz
  - RAM:    520 KB SRAM
  - Flash:  4 MB (external QSPI) — on Pico 2 board
  - PIO:    3x PIO blocks (vs 2 on RP2040), 4 SM each = 12 total
  - GPIO:   30 GPIO (GP0-GP29) + analog inputs
  - ADC:    4-channel 12-bit ADC + temperature sensor
  - UART:   2x UART (8-pin assignable)
  - SPI:    2x SPI (8-pin assignable)
  - I2C:    2x I2C (4-pin assignable)
  - PWM:    16 PWM slices (vs 4 on RP2040)
  - USB:    1x USB 1.1 OTG (Micro-USB B)
  - Debug:  SWD (3-pin)
  - Form factor: 52.5 x 21.0 mm, ~3.3 g

Marine robotics use: real-time sensor processing, PIO-based custom protocols,
high-speed ADC sampling, thruster ESC signal generation, underwater
acoustic signal processing, low-power sensor nodes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORE_COUNT = 2                          # Dual Cortex-M33
CPU_FREQ_MAX = 150_000_000              # 150 MHz
CPU_FREQ_DEFAULT = 133_000_000          # 133 MHz safe default
SRAM_TOTAL = 520 * 1024                 # 520 KB
FLASH_TOTAL = 4 * 1024 * 1024           # 4 MB external QSPI
PIO_COUNT = 3                           # 3 PIO blocks
SM_PER_PIO = 4                          # 4 state machines per PIO
GPIO_COUNT = 30                         # GP0-GP29
ADC_CHANNELS = 4                        # 4 external + 1 internal temp
ADC_RESOLUTION = 12
PWM_SLICES = 16                         # 16 PWM slices
RTC_COUNT = 1


class GPIOPin(IntEnum):
    """RP2350 GPIO pin identifiers."""
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
    """Marine sensor function identifiers for Pico 2."""
    SONAR_TRIG = 0
    SONAR_ECHO = 1
    SERVO_1 = 2
    SERVO_2 = 3
    SERVO_3 = 4
    SERVO_4 = 5
    SERVO_5 = 6
    SERVO_6 = 7
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
    ADC_CONDUCTIVITY = 42
    ADC_DO = 43
    STATUS_LED = 50
    ADC_VSYS = 51
    BUZZER = 60


@dataclass
class PinMapping:
    """Maps a PinFunction to a physical GPIO pin."""
    function: PinFunction
    pin: int
    pull_up: bool = False
    pull_down: bool = False
    invert: bool = False
    description: str = ""

    def __post_init__(self):
        if not (0 <= self.pin <= GPIO_COUNT - 1):
            raise ValueError(
                f"Invalid GPIO pin {self.pin}. RP2350 supports GPIO 0-{GPIO_COUNT - 1}."
            )


@dataclass
class MemoryRegion:
    """SRAM allocation region."""
    name: str
    start: int
    size_bytes: int
    purpose: str = ""

    @property
    def end(self) -> int:
        return self.start + self.size_bytes


@dataclass
class MemoryLayout:
    """Memory layout for the RP2350."""
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


@dataclass(frozen=True)
class ThermalProfile:
    idle_temp_c: float = 25.0
    typical_load_c: float = 35.0
    throttle_start_c: float = 85.0   # RP2350 has no thermal throttle; flash limit
    critical_c: float = 90.0
    enclosure_note: str = (
        "No active cooling needed. RP2350 is extremely low power. "
        "Can be fully potted in marine epoxy for IP68 deployment."
    )


@dataclass(frozen=True)
class PowerProfile:
    idle_w: float = 0.05
    typical_load_w: float = 0.12
    max_load_w: float = 0.30
    voltage_range: Tuple[float, float] = (1.8, 3.3)
    battery_life_estimate_mah: float = 35.0
    solar_compatible: bool = True


class Pico2Config:
    """
    Complete hardware configuration for Raspberry Pi Pico 2 (RP2350).

    Dual-core Cortex-M33 @ 150 MHz with 520 KB SRAM, 3 PIO blocks,
    16 PWM slices, and 4-channel 12-bit ADC. Optimized for real-time
    marine sensor processing and custom protocol implementation.
    """

    DEFAULT_PIN_MAP: Dict[PinFunction, int] = {
        PinFunction.SONAR_TRIG: GPIOPin.GP0,
        PinFunction.SONAR_ECHO: GPIOPin.GP1,
        PinFunction.SERVO_1: GPIOPin.GP2,
        PinFunction.SERVO_2: GPIOPin.GP3,
        PinFunction.SERVO_3: GPIOPin.GP4,
        PinFunction.SERVO_4: GPIOPin.GP5,
        PinFunction.SERVO_5: GPIOPin.GP6,
        PinFunction.SERVO_6: GPIOPin.GP7,
        PinFunction.I2C_SDA: GPIOPin.GP12,
        PinFunction.I2C_SCL: GPIOPin.GP13,
        PinFunction.UART_TX: GPIOPin.GP8,
        PinFunction.UART_RX: GPIOPin.GP9,
        PinFunction.SPI_MOSI: GPIOPin.GP15,
        PinFunction.SPI_MISO: GPIOPin.GP16,
        PinFunction.SPI_CLK: GPIOPin.GP14,
        PinFunction.SPI_CS: GPIOPin.GP17,
        PinFunction.ADC_TEMP: GPIOPin.GP26,
        PinFunction.ADC_DEPTH: GPIOPin.GP27,
        PinFunction.ADC_CONDUCTIVITY: GPIOPin.GP28,
        PinFunction.ADC_DO: GPIOPin.GP29,
        PinFunction.STATUS_LED: GPIOPin.GP25,
        PinFunction.ADC_VSYS: GPIOPin.GP23,
        PinFunction.BUZZER: GPIOPin.GP18,
    }

    def __init__(self, clock_freq: int = CPU_FREQ_DEFAULT):
        self._pin_map: Dict[PinFunction, PinMapping] = {}
        self.clock_freq = clock_freq
        self.memory = MemoryLayout()
        self._pio_allocated: Dict[str, Tuple[int, int]] = {}
        self.thermal = ThermalProfile()
        self.power = PowerProfile()
        self.nexus_platform = "raspberry_pi"
        self.nexus_role = "realtime_sensor"
        self._configure_default_pins()
        self._config_hash = self._compute_hash()

    def _configure_default_pins(self):
        for func, pin in self.DEFAULT_PIN_MAP.items():
            self._pin_map[func] = PinMapping(
                function=func, pin=pin,
                description=f"{func.name} on GP{pin}",
            )

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
            self._pin_map[function] = PinMapping(
                function=function, pin=new_pin, description=description
            )

    def all_pins(self) -> List[PinMapping]:
        return list(self._pin_map.values())

    def used_gpio_set(self) -> set:
        return {pm.pin for pm in self._pin_map.values()}

    @property
    def PIN_SONAR_TRIG(self) -> int:
        return self.get_pin(PinFunction.SONAR_TRIG).pin

    @property
    def PIN_SONAR_ECHO(self) -> int:
        return self.get_pin(PinFunction.SONAR_ECHO).pin

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

    def allocate_pio(self, name: str, pio_block: int = 0, sm_index: int = 0) -> Tuple[int, int]:
        if pio_block not in range(PIO_COUNT):
            raise ValueError(f"PIO block must be 0-{PIO_COUNT - 1}.")
        if sm_index not in range(SM_PER_PIO):
            raise ValueError(f"State machine index must be 0-{SM_PER_PIO - 1}.")
        for existing_name, (pb, sm) in self._pio_allocated.items():
            if pb == pio_block and sm == sm_index:
                raise ValueError(f"PIO{pio_block} SM{sm_index} already allocated.")
        self._pio_allocated[name] = (pio_block, sm_index)
        return (pio_block, sm_index)

    def release_pio(self, name: str):
        self._pio_allocated.pop(name, None)

    def pio_allocation_summary(self) -> Dict[str, Tuple[int, int]]:
        return dict(self._pio_allocated)

    def available_sm(self, pio_block: int = 0) -> List[int]:
        used = {sm for (_, sm) in self._pio_allocated.values()}
        return [i for i in range(SM_PER_PIO) if i not in used]

    def configure_marine_sensors(self):
        """Set up SRAM memory regions for marine sensor processing."""
        self.memory.add_region("stack_core0", 0x20000000, 4096, "Core 0 stack")
        self.memory.add_region("stack_core1", 0x20001000, 4096, "Core 1 stack")
        self.memory.add_region("sensor_ring_buf", 0x20002000, 16384, "Sensor data ring buffer")
        self.memory.add_region("sonar_samples", 0x20006000, 32768, "Sonar return samples")
        self.memory.add_region("uart_rx_buf", 0x2000E000, 4096, "UART NMEA buffer")
        self.memory.add_region("i2c_scratch", 0x2000F000, 2048, "I2C temp buffers")
        self.memory.add_region("adc_dma_buf", 0x2000F800, 8192, "ADC DMA buffer")
        self.memory.add_region("servo_state", 0x20011800, 256, "Servo angle state")

    def get_config_hash(self) -> str:
        return self._config_hash

    def summary(self) -> Dict[str, Any]:
        return {
            "board": "Raspberry Pi Pico 2",
            "soc": "RP2350",
            "cpu": f"{CORE_COUNT}x Cortex-M33 @ {self.clock_freq // 1_000_000}MHz",
            "sram_total_kb": SRAM_TOTAL // 1024,
            "sram_allocated_bytes": self.memory.total_allocated(),
            "pio_blocks": PIO_COUNT,
            "pwm_slices": PWM_SLICES,
            "adc_channels": ADC_CHANNELS,
            "gpio_count": GPIO_COUNT,
            "pins_configured": len(self._pin_map),
            "pio_allocated": dict(self._pio_allocated),
            "power_idle_w": self.power.idle_w,
            "config_hash": self._config_hash[:16],
        }

    def _compute_hash(self) -> str:
        data = (
            f"RP2350:{CORE_COUNT}:{self.clock_freq}:"
            f"{GPIO_COUNT}:{SRAM_TOTAL}:{PIO_COUNT}:{PWM_SLICES}:"
            f"{len(self._pin_map)}"
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def __repr__(self) -> str:
        return (
            f"Pico2Config(soc='RP2350', cores={CORE_COUNT}, "
            f"clock={self.clock_freq // 1_000_000}MHz, sram={SRAM_TOTAL // 1024}KB, "
            f"cpu=Cortex-M33)"
        )
