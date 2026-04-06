"""Li-ion battery modelling with Peukert's law and CC-CV charging."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


# Typical Peukert exponent for Li-ion cells
DEFAULT_PEUKERT_EXPONENT = 1.05


@dataclass
class BatteryState:
    """Snapshot of battery condition."""
    charge_percent: float          # 0-100
    voltage: float                 # V
    current: float                 # A (positive = charging)
    temperature: float             # °C
    capacity_wh: float             # remaining capacity in Wh
    cycles: int = 0                # charge/discharge cycles completed


@dataclass
class BatteryModel:
    """Fixed parameters describing a battery pack."""
    nominal_capacity: float        # Wh
    nominal_voltage: float         # V
    internal_resistance: float     # Ohm (at 25 °C, 100 % SoC)
    temp_coeff: float = 0.005      # resistance increase per °C above 25 °C
    peukert_exponent: float = DEFAULT_PEUKERT_EXPONENT
    max_charge_voltage: float = 4.2
    max_charge_current: float = 10.0  # A
    min_voltage: float = 3.0


@dataclass
class LoadPoint:
    """A single point in a load profile."""
    current_a: float
    duration_s: float


class BatterySimulator:
    """Simulates Li-ion battery charge / discharge behaviour."""

    # Coulombic efficiency (fraction of charge that is stored)
    COULOMBIC_EFFICIENCY = 0.995

    # Self-discharge rate per hour (fraction)
    SELF_DISCHARGE_PER_HOUR = 0.0005

    @staticmethod
    def _effective_current(current: float, peukert: float) -> float:
        """Compute Peukert-equivalent current at 1 A reference."""
        if current <= 0:
            return 0.0
        return current ** peukert

    @staticmethod
    def discharge(
        state: BatteryState,
        current_draw: float,
        dt: float,
        battery_model: BatteryModel | None = None,
    ) -> BatteryState:
        """Discharge the battery for *dt* seconds at *current_draw* amps.

        Uses Peukert's law to adjust effective capacity consumed.
        Returns a new BatteryState.
        """
        peukert = DEFAULT_PEUKERT_EXPONENT
        if battery_model is not None:
            peukert = battery_model.peukert_exponent

        # Energy drawn (Wh) adjusted by Peukert
        # At 1 A reference: capacity_Ah_1A = capacity_Ah * (I / I_ref)^(n-1)
        # Effective Ah drawn = (current ** peukert) * dt / 3600
        effective_ah_drawn = (abs(current_draw) ** peukert) * dt / 3600.0
        nominal_v = state.voltage if state.voltage > 0 else 3.7
        energy_drawn_wh = effective_ah_drawn * nominal_v

        new_capacity = max(0.0, state.capacity_wh - energy_drawn_wh)

        # Derive new SoC
        ref_capacity = battery_model.nominal_capacity if battery_model else 100.0
        new_soc = (new_capacity / ref_capacity) * 100.0 if ref_capacity > 0 else 0.0

        # Voltage drops with SoC (linear approximation)
        soc_factor = max(0.0, min(1.0, new_soc / 100.0))
        base_voltage = 3.0 + 1.2 * soc_factor  # 3.0 V empty → 4.2 V full
        # Add IR drop
        ir_drop = 0.0
        if battery_model is not None:
            ir_drop = current_draw * battery_model.internal_resistance
        new_voltage = max(0.0, base_voltage - ir_drop)

        return BatteryState(
            charge_percent=round(new_soc, 4),
            voltage=round(new_voltage, 4),
            current=-abs(current_draw),
            temperature=state.temperature,
            capacity_wh=round(new_capacity, 6),
            cycles=state.cycles,
        )

    @staticmethod
    def charge(
        state: BatteryState,
        current: float,
        dt: float,
        battery_model: BatteryModel | None = None,
    ) -> BatteryState:
        """Charge the battery using a simplified CC-CV model.

        *current* is the applied charge current in amps (positive).
        *dt* is in seconds.

        In constant-current phase the full current is applied.
        When voltage reaches *max_charge_voltage* the model switches to
        constant-voltage and current tapers linearly to zero.
        """
        if battery_model is None:
            battery_model = BatteryModel(
                nominal_capacity=100.0,
                nominal_voltage=3.7,
                internal_resistance=0.05,
            )

        eff = BatterySimulator.COULOMBIC_EFFICIENCY
        nominal_v = battery_model.nominal_voltage
        max_v = battery_model.max_charge_voltage
        max_i = battery_model.max_charge_current
        nominal_cap = battery_model.nominal_capacity

        applied_current = min(abs(current), max_i)

        if state.voltage >= max_v:
            # CV phase: current tapers
            soc = state.charge_percent / 100.0 if nominal_cap > 0 else 0
            headroom = 1.0 - soc
            taper = max(0.0, headroom * max_i)
            applied_current = min(applied_current, taper)

        energy_stored_wh = (applied_current * dt / 3600.0) * nominal_v * eff
        new_capacity = min(nominal_cap, state.capacity_wh + energy_stored_wh)
        new_soc = (new_capacity / nominal_cap) * 100.0 if nominal_cap > 0 else 0.0

        soc_factor = min(1.0, new_soc / 100.0)
        new_voltage = 3.0 + 1.2 * soc_factor
        new_voltage = min(new_voltage, max_v)

        # Temperature rise (simplified)
        power_loss = (applied_current ** 2) * battery_model.internal_resistance
        dt_hours = dt / 3600.0
        temp_rise = power_loss * dt_hours * 0.5  # rough °C rise

        return BatteryState(
            charge_percent=round(new_soc, 4),
            voltage=round(new_voltage, 4),
            current=round(applied_current, 4),
            temperature=round(state.temperature + temp_rise, 4),
            capacity_wh=round(new_capacity, 6),
            cycles=state.cycles,
        )

    @staticmethod
    def compute_capacity_remaining(state: BatteryState) -> float:
        """Return remaining capacity in Wh."""
        return max(0.0, state.capacity_wh)

    @staticmethod
    def compute_state_of_health(
        battery_model: BatteryModel,
        current_capacity: float,
    ) -> float:
        """SOH = current_capacity / nominal_capacity × 100."""
        if battery_model.nominal_capacity <= 0:
            return 0.0
        soh = (current_capacity / battery_model.nominal_capacity) * 100.0
        return max(0.0, min(100.0, soh))

    @staticmethod
    def estimate_cycles_remaining(
        soh: float,
        degradation_per_cycle: float,
    ) -> int:
        """Rough estimate of cycles until SOH hits 80 % (end of life)."""
        if degradation_per_cycle <= 0:
            return 0
        remaining_headroom = soh - 80.0
        if remaining_headroom <= 0:
            return 0
        return int(remaining_headroom / degradation_per_cycle)

    @staticmethod
    def compute_internal_resistance(
        temperature: float,
        soc: float,
        battery_model: BatteryModel | None = None,
    ) -> float:
        """Compute resistance adjusted for temperature and SoC.

        Resistance increases at lower temperatures and lower SoC.
        """
        base_r = 0.05 if battery_model is None else battery_model.internal_resistance
        tc = 0.005 if battery_model is None else battery_model.temp_coeff

        # Temperature factor: higher temp → slightly higher resistance
        delta_t = max(0.0, temperature - 25.0)
        temp_factor = 1.0 + tc * delta_t

        # Cold temperatures increase resistance significantly
        if temperature < 0:
            cold_factor = 1.0 + 0.02 * abs(temperature)
            temp_factor *= cold_factor

        # SoC factor: low SoC → higher resistance
        soc_norm = max(0.0, min(1.0, soc / 100.0))
        soc_factor = 1.0 + 0.3 * (1.0 - soc_norm)

        return base_r * temp_factor * soc_factor

    @staticmethod
    def simulate_cycle(
        initial_state: BatteryState,
        load_profile: List[LoadPoint],
        battery_model: BatteryModel | None = None,
    ) -> BatteryState:
        """Run a full discharge cycle and return the final state."""
        state = BatteryState(
            charge_percent=initial_state.charge_percent,
            voltage=initial_state.voltage,
            current=0.0,
            temperature=initial_state.temperature,
            capacity_wh=initial_state.capacity_wh,
            cycles=initial_state.cycles,
        )
        for lp in load_profile:
            state = BatterySimulator.discharge(state, lp.current_a, lp.duration_s, battery_model)
        state = BatteryState(
            charge_percent=state.charge_percent,
            voltage=state.voltage,
            current=state.current,
            temperature=state.temperature,
            capacity_wh=state.capacity_wh,
            cycles=state.cycles + 1,
        )
        return state
