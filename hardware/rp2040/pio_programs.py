"""
NEXUS PIO Program Configurations for RP2040

Provides PIO assembly-level program definitions and timing parameters
for marine robotics I/O tasks: sonar ping/echo timing, servo PWM generation,
and full-duplex UART bridging for sensor telemetry.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PIO Instruction Set (subset used by NEXUS programs)
# ---------------------------------------------------------------------------

class PIOInstruction(IntEnum):
    """Subset of RP2040 PIO instructions used in NEXUS programs."""
    JMP = 0x0000
    WAIT_IRQ = 0x0020
    SET_PINS = 0x00E0
    SET_X = 0x00E1
    SET_Y = 0x00E2
    OUT_PINS = 0x0060
    IN_PINS = 0x0040
    PULL = 0x0080
    PUSH = 0x0080
    MOV_X_Y = 0x00A1
    MOV_Y_X = 0x00A3
    MOV_PINS_X = 0x00A4
    IRQ_SET = 0x00C0
    IRQ_WAIT = 0x00C4


# ---------------------------------------------------------------------------
# Sonar Ping / Echo Program
# ---------------------------------------------------------------------------

@dataclass
class SonarTimingParams:
    """Timing parameters for ultrasonic sonar ping/echo cycle."""
    ping_pulse_us: int = 10               # Trigger pulse width in microseconds
    max_echo_wait_ms: int = 30            # Max wait for echo return
    speed_of_sound_m_s: float = 1500.0    # Speed of sound in seawater (m/s)
    max_range_m: float = 4.0              # Maximum measurable range

    @property
    def echo_timeout_cycles(self) -> int:
        """Echo timeout in PIO clock cycles (assuming 125 MHz)."""
        return int(self.max_echo_wait_ms * 1000 * 1.25)  # 1.25 cycles per us @ 125 MHz

    def distance_from_echo_us(self, echo_us: int) -> float:
        """Convert echo pulse duration (us) to distance in meters."""
        if echo_us <= 0:
            return 0.0
        # Round-trip time: distance = (echo_us / 1e6) * speed_of_sound / 2
        return (echo_us / 1_000_000.0) * self.speed_of_sound_m_s / 2.0

    def echo_us_from_distance(self, distance_m: float) -> int:
        """Convert distance in meters to expected echo pulse duration (us)."""
        if distance_m <= 0:
            return 0
        return int((2.0 * distance_m / self.speed_of_sound_m_s) * 1_000_000)


@dataclass
class SonarPingProgram:
    """
    PIO program for sonar ping/echo timing.

    Generates a precise trigger pulse and measures the echo return width
    with deterministic timing independent of Python execution.
    """
    trigger_pin: int = 0
    echo_pin: int = 1
    pio_block: int = 0
    sm_index: int = 0
    timing: SonarTimingParams = field(default_factory=SonarTimingParams)
    assembled: bool = False
    instruction_count: int = 0

    # Reference PIO assembly (human-readable)
    ASSEMBLY = [
        "set pins, 1            ; Assert trigger HIGH",
        "set pins, 0            ; De-assert trigger (generates ping pulse)",
        "wait 0 pin, ECHO_PIN   ; Wait for echo rising edge",
        "set x, 0               ; Clear counter",
        ".measure_loop:",
        "jmp pin, .measure_done ; If echo LOW, measurement complete",
        "jmp x--, .measure_loop ; Decrement counter (count echo width)",
        ".measure_done:",
        "irq set 0              ; Signal measurement complete",
    ]

    def assemble(self) -> List[str]:
        """Return assembled instruction list and set program as assembled."""
        self.instruction_count = len(self.ASSEMBLY)
        self.assembled = True
        return self.ASSEMBLY

    def get_config_dict(self) -> dict:
        return {
            "trigger_pin": self.trigger_pin,
            "echo_pin": self.echo_pin,
            "pio_block": self.pio_block,
            "sm_index": self.sm_index,
            "ping_pulse_us": self.timing.ping_pulse_us,
            "max_range_m": self.timing.max_range_m,
        }


# ---------------------------------------------------------------------------
# Servo PWM Program
# ---------------------------------------------------------------------------

@dataclass
class ServoPWMParams:
    """Parameters for servo PWM signal generation via PIO."""
    frequency_hz: int = 50                # Standard servo frequency
    min_pulse_us: int = 1000              # 1 ms minimum (0 degrees)
    max_pulse_us: int = 2000              # 2 ms maximum (180 degrees)
    clock_div: float = 1.0                # PIO clock divider
    channels: int = 4                     # Number of servo channels
    resolution_bits: int = 16             # PWM counter resolution

    @property
    def period_cycles(self) -> int:
        """PWM period in PIO clock cycles at 125 MHz base clock."""
        base_clock = 125_000_000
        pio_clock = base_clock / self.clock_div
        return int(pio_clock / self.frequency_hz)

    @property
    def min_duty_cycles(self) -> int:
        base_clock = 125_000_000
        pio_clock = base_clock / self.clock_div
        return int(pio_clock * (self.min_pulse_us / 1_000_000))

    @property
    def max_duty_cycles(self) -> int:
        base_clock = 125_000_000
        pio_clock = base_clock / self.clock_div
        return int(pio_clock * (self.max_pulse_us / 1_000_000))

    def angle_to_duty(self, angle_deg: float) -> int:
        """Convert servo angle (0-180) to duty cycle value."""
        clamped = max(0.0, min(180.0, angle_deg))
        fraction = clamped / 180.0
        return int(self.min_duty_cycles + fraction * (self.max_duty_cycles - self.min_duty_cycles))

    def duty_to_angle(self, duty: int) -> float:
        """Convert duty cycle to servo angle in degrees."""
        if self.max_duty_cycles == self.min_duty_cycles:
            return 90.0
        fraction = (duty - self.min_duty_cycles) / (self.max_duty_cycles - self.min_duty_cycles)
        return max(0.0, min(180.0, fraction * 180.0))


@dataclass
class ServoPWMProgram:
    """
    PIO program for multi-channel servo PWM generation.

    Generates a 50 Hz PWM signal with configurable pulse width (1-2 ms)
    for up to 4 servo channels simultaneously.
    """
    pins: List[int] = field(default_factory=lambda: [2, 3, 6, 7])
    pio_block: int = 0
    sm_index: int = 1
    params: ServoPWMParams = field(default_factory=ServoPWMParams)
    assembled: bool = False
    instruction_count: int = 0

    ASSEMBLY = [
        "pull block            ; Load period + high time from TX FIFO",
        "mov x, osr            ; Period into X",
        ".pulse_start:",
        "set pins, 1           ; Start pulse (all servo pins HIGH)",
        "mov y, osr            ; Pulse width into Y",
        ".pulse_high:",
        "jmp y--, .pulse_high  ; Maintain HIGH for pulse width",
        "set pins, 0           ; End pulse (all servo pins LOW)",
        ".pulse_low:",
        "jmp x--, .pulse_low   ; Wait for rest of period",
        "jmp .pulse_start      ; Repeat",
    ]

    def assemble(self) -> List[str]:
        self.instruction_count = len(self.ASSEMBLY)
        self.assembled = True
        return self.ASSEMBLY

    def get_config_dict(self) -> dict:
        return {
            "pins": self.pins,
            "pio_block": self.pio_block,
            "sm_index": self.sm_index,
            "frequency_hz": self.params.frequency_hz,
            "min_pulse_us": self.params.min_pulse_us,
            "max_pulse_us": self.params.max_pulse_us,
            "channels": self.params.channels,
        }


# ---------------------------------------------------------------------------
# UART Bridge Program
# ---------------------------------------------------------------------------

@dataclass
class UARTBridgeParams:
    """Parameters for PIO-based UART bridge."""
    baud_rate: int = 9600
    bits_per_char: int = 8
    stop_bits: int = 1
    parity: str = "none"     # none, even, odd
    clock_div: float = 1.0
    rx_fifo_depth: int = 4    # PIO RX FIFO depth
    tx_fifo_depth: int = 4    # PIO TX FIFO depth

    @property
    def bit_period_cycles(self) -> int:
        base_clock = 125_000_000
        pio_clock = base_clock / self.clock_div
        return int(pio_clock / self.baud_rate)

    @property
    def frame_bits(self) -> int:
        total = self.bits_per_char + self.stop_bits
        if self.parity != "none":
            total += 1
        return total

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.baud_rate < 300 or self.baud_rate > 1_000_000:
            errors.append(f"Baud rate {self.baud_rate} out of supported range (300-1000000).")
        if self.bits_per_char not in (5, 6, 7, 8):
            errors.append(f"Bits per char must be 5-8, got {self.bits_per_char}.")
        if self.stop_bits not in (1, 2):
            errors.append(f"Stop bits must be 1 or 2, got {self.stop_bits}.")
        if self.parity not in ("none", "even", "odd"):
            errors.append(f"Parity must be none/even/odd, got '{self.parity}'.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


@dataclass
class UARTBridgeProgram:
    """
    PIO program for UART-to-PIO bridging.

    Implements full-duplex UART TX and RX using two state machines,
    suitable for GPS NMEA and other marine sensor serial protocols.
    """
    tx_pin: int = 8
    rx_pin: int = 9
    pio_block: int = 1
    tx_sm: int = 0
    rx_sm: int = 1
    params: UARTBridgeParams = field(default_factory=UARTBridgeParams)
    assembled: bool = False
    instruction_count: int = 0

    TX_ASSEMBLY = [
        ".tx_wait:",
        "pull block            ; Wait for a byte to transmit",
        "set x, 7              ; 8 bits to send",
        ".tx_bit_loop:",
        "out pins, 1           ; Output LSB",
        "set y, 0",
        ".tx_delay:",
        "jmp y--, .tx_delay [period] ; Bit period delay",
        "jmp x--, .tx_bit_loop",
    ]

    RX_ASSEMBLY = [
        ".rx_wait_start:",
        "wait 0 pin, RX_PIN    ; Wait for start bit (line goes LOW)",
        "set x, 7              ; 8 bits to receive",
        ".rx_sample:",
        "set y, 0",
        ".rx_half_delay:",
        "jmp y--, .rx_half_delay [period/2] ; Sample at mid-bit",
        "in pins, 1            ; Sample bit",
        "set y, 0",
        ".rx_bit_delay:",
        "jmp y--, .rx_bit_delay [period] ; Full bit period",
        "jmp x--, .rx_sample",
        "push block            ; Push received byte to RX FIFO",
    ]

    def assemble(self) -> Dict[str, List[str]]:
        self.instruction_count = len(self.TX_ASSEMBLY) + len(self.RX_ASSEMBLY)
        self.assembled = True
        return {"tx": self.TX_ASSEMBLY, "rx": self.RX_ASSEMBLY}

    def get_config_dict(self) -> dict:
        return {
            "tx_pin": self.tx_pin,
            "rx_pin": self.rx_pin,
            "pio_block": self.pio_block,
            "tx_sm": self.tx_sm,
            "rx_sm": self.rx_sm,
            "baud_rate": self.params.baud_rate,
            "bits_per_char": self.params.bits_per_char,
            "stop_bits": self.params.stop_bits,
            "parity": self.params.parity,
        }


# ---------------------------------------------------------------------------
# PIO Program Registry
# ---------------------------------------------------------------------------

class PIOProgramRegistry:
    """Registry for managing all PIO programs in the NEXUS system."""

    def __init__(self):
        self._programs: Dict[str, object] = {}

    def register(self, name: str, program) -> None:
        if name in self._programs:
            raise ValueError(f"PIO program '{name}' is already registered.")
        self._programs[name] = program

    def unregister(self, name: str) -> None:
        self._programs.pop(name, None)

    def get(self, name: str):
        if name not in self._programs:
            raise KeyError(f"PIO program '{name}' not found in registry.")
        return self._programs[name]

    def list_programs(self) -> List[str]:
        return list(self._programs.keys())

    def assembled_programs(self) -> List[str]:
        return [name for name, prog in self._programs.items() if getattr(prog, "assembled", False)]

    def assemble_all(self) -> Dict[str, bool]:
        results = {}
        for name, prog in self._programs.items():
            if hasattr(prog, "assemble"):
                prog.assemble()
                results[name] = True
            else:
                results[name] = False
        return results
