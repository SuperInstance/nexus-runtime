"""
NEXUS FlexIO Peripheral Configuration for i.MX RT Series

FlexIO is a highly configurable peripheral that can emulate various
serial/parallel protocols (I2S, SPI, UART, PWM, etc.) using
programmable shifters and timers.

Marine use cases:
  - Custom underwater acoustic protocol implementation
  - Concurrent sensor bus multiplexing
  - High-speed sonar pulse timing
  - Multi-channel thruster ESC signal generation
  - DVL Doppler shift measurement interface
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


class FlexIOInstance(IntEnum):
    FLEXIO1 = 0
    FLEXIO2 = 1


class ShifterMode(IntEnum):
    """FlexIO shifter operating modes."""
    DISABLE = 0
    RECEIVE = 1
    TRANSMIT = 2
    MATCH_STORE = 3
    MATCH_CONTINUOUS = 4
    STATE = 5
    LOGIC = 6
    PARALLEL_RX = 7
    PARALLEL_TX = 8


class TimerMode(IntEnum):
    """FlexIO timer trigger/pin modes."""
    DISABLE = 0
    DUAL_EDGE_CAPTURE = 1
    SINGLE_EDGE_CAPTURE = 2
    TRANSFER_COUNT = 3
    PWM_HIGH_TRUE = 4
    PWM_LOW_TRUE = 5
    BAUD = 6
    FREQ_8X_BAUD = 7
    PIN_OUTPUT = 8
    PIN_INPUT = 9
    TRIGGER_START = 10
    TRIGGER_START_CAPTURE = 11


class TimerOutput(IntEnum):
    """Timer output/polarity configuration."""
    ONE_SHOT_DISABLE = 0
    LOGIC_ONE_ON_COMPARE = 1
    TOGGLE_ON_COMPARE = 2
    LOGIC_ZERO_ON_COMPARE = 3
    ONE_SHOT_LOGIC_ONE = 4
    ONE_SHOT_TOGGLE = 5
    ONE_SHOT_LOGIC_ZERO = 6


@dataclass
class ShifterConfig:
    """Configuration for a single FlexIO shifter."""
    index: int                           # Shifter index (0-7)
    mode: ShifterMode = ShifterMode.DISABLE
    pin_select: int = 0                 # Pin select (0-31)
    pin_polarity: bool = False
    pin_config: int = 0                 # 0=disabled, 1=open-drain, 2=bidir, 3=bidir OD
    timer_select: int = 0
    parallel_width: int = 1             # 1, 2, or 4 bits
    source: int = 0                     # Shift source (for RX modes)
    start_bit: int = 0

    def validate(self, max_shifters: int = 8) -> List[str]:
        errors: List[str] = []
        if not (0 <= self.index < max_shifters):
            errors.append(f"Shifter index must be 0-{max_shifters-1}, got {self.index}.")
        if not (0 <= self.pin_select <= 31):
            errors.append(f"Pin select must be 0-31, got {self.pin_select}.")
        return errors

    def is_valid(self, max_shifters: int = 8) -> bool:
        return len(self.validate(max_shifters)) == 0


@dataclass
class TimerConfig:
    """Configuration for a single FlexIO timer."""
    index: int
    mode: TimerMode = TimerMode.DISABLE
    pin_select: int = 0
    pin_polarity: bool = False
    pin_config: int = 0
    timer_output: TimerOutput = TimerOutput.ONE_SHOT_DISABLE
    trigger_source: int = 0
    trigger_polarity: bool = False
    timer_decrement: int = 1
    timer_compare: int = 0
    timer_counter: int = 0

    def validate(self, max_timers: int = 4) -> List[str]:
        errors: List[str] = []
        if not (0 <= self.index < max_timers):
            errors.append(f"Timer index must be 0-{max_timers-1}, got {self.index}.")
        if not (0 <= self.pin_select <= 31):
            errors.append(f"Pin select must be 0-31, got {self.pin_select}.")
        return errors

    def is_valid(self, max_timers: int = 4) -> bool:
        return len(self.validate(max_timers)) == 0


@dataclass
class FlexIOProtocol:
    """A pre-defined FlexIO protocol configuration."""
    name: str
    description: str
    shifters: List[ShifterConfig] = field(default_factory=list)
    timers: List[TimerConfig] = field(default_factory=list)
    baud_rate: int = 0
    data_bits: int = 8
    clock_pin: int = -1
    data_pins: List[int] = field(default_factory=list)

    def validate(self) -> List[str]:
        errors: List[str] = []
        shifter_indices = {s.index for s in self.shifters}
        if len(shifter_indices) != len(self.shifters):
            errors.append("Duplicate shifter indices detected.")
        timer_indices = {t.index for t in self.timers}
        if len(timer_indices) != len(self.timers):
            errors.append("Duplicate timer indices detected.")
        for s in self.shifters:
            errors.extend(s.validate())
        for t in self.timers:
            errors.extend(t.validate())
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


class FlexIOManager:
    """
    Manages FlexIO peripheral allocation and protocol configuration
    for the NEXUS i.MX RT marine controller.
    """

    MAX_SHIFTERS_PER_INSTANCE = 8
    MAX_TIMERS_PER_INSTANCE = 4
    MAX_PIN_SELECT = 32

    def __init__(self):
        self._protocols: Dict[str, FlexIOProtocol] = {}
        self._shifters_used: Dict[FlexIOInstance, List[int]] = {
            FlexIOInstance.FLEXIO1: [],
            FlexIOInstance.FLEXIO2: [],
        }
        self._timers_used: Dict[FlexIOInstance, List[int]] = {
            FlexIOInstance.FLEXIO1: [],
            FlexIOInstance.FLEXIO2: [],
        }
        self._pins_used: Set[int] = set()

    def register_protocol(self, protocol: FlexIOProtocol, instance: FlexIOInstance) -> List[str]:
        """Register a protocol and allocate its shifters/timers/pins."""
        errors: List[str] = []
        errors.extend(protocol.validate())

        if errors:
            return errors

        # Allocate shifters
        for s in protocol.shifters:
            if s.index in self._shifters_used[instance]:
                errors.append(f"Shifter {s.index} already allocated on FlexIO{instance.value+1}.")
            if s.pin_select in self._pins_used:
                errors.append(f"Pin {s.pin_select} already in use.")
            self._shifters_used[instance].append(s.index)
            self._pins_used.add(s.pin_select)

        # Allocate timers
        for t in protocol.timers:
            if t.index in self._timers_used[instance]:
                errors.append(f"Timer {t.index} already allocated on FlexIO{instance.value+1}.")

        if errors:
            # Rollback
            for s in protocol.shifters:
                if s.index in self._shifters_used[instance]:
                    self._shifters_used[instance].remove(s.index)
                self._pins_used.discard(s.pin_select)
            return errors

        for t in protocol.timers:
            self._timers_used[instance].append(t.index)
            if t.pin_select not in self._pins_used:
                self._pins_used.add(t.pin_select)

        self._protocols[protocol.name] = protocol
        return []

    def unregister_protocol(self, name: str) -> bool:
        if name not in self._protocols:
            return False
        protocol = self._protocols[name]
        for s in protocol.shifters:
            for instance_shifters in self._shifters_used.values():
                if s.index in instance_shifters:
                    instance_shifters.remove(s.index)
            self._pins_used.discard(s.pin_select)
        for t in protocol.timers:
            for instance_timers in self._timers_used.values():
                if t.index in instance_timers:
                    instance_timers.remove(t.index)
            self._pins_used.discard(t.pin_select)
        del self._protocols[name]
        return True

    def list_protocols(self) -> List[str]:
        return list(self._protocols.keys())

    def get_protocol(self, name: str) -> Optional[FlexIOProtocol]:
        return self._protocols.get(name)

    def available_shifters(self, instance: FlexIOInstance) -> List[int]:
        used = self._shifters_used[instance]
        return [i for i in range(self.MAX_SHIFTERS_PER_INSTANCE) if i not in used]

    def available_timers(self, instance: FlexIOInstance) -> List[int]:
        used = self._timers_used[instance]
        return [i for i in range(self.MAX_TIMERS_PER_INSTANCE) if i not in used]

    def available_pins(self) -> List[int]:
        return sorted(set(range(self.MAX_PIN_SELECT)) - self._pins_used)

    def summary(self) -> Dict[str, Any]:
        return {
            "protocols": self.list_protocols(),
            "flexio1_shifters_used": len(self._shifters_used[FlexIOInstance.FLEXIO1]),
            "flexio1_timers_used": len(self._timers_used[FlexIOInstance.FLEXIO1]),
            "flexio2_shifters_used": len(self._shifters_used[FlexIOInstance.FLEXIO2]),
            "flexio2_timers_used": len(self._timers_used[FlexIOInstance.FLEXIO2]),
            "pins_used": len(self._pins_used),
        }
