"""Tests for battery module."""

import math
import pytest

from jetson.energy.battery import (
    BatteryState,
    BatteryModel,
    BatterySimulator,
    LoadPoint,
    DEFAULT_PEUKERT_EXPONENT,
)


# ---------------------------------------------------------------------------
# BatteryState tests
# ---------------------------------------------------------------------------

class TestBatteryState:
    def test_create_state(self):
        s = BatteryState(
            charge_percent=80.0, voltage=3.8, current=-2.0,
            temperature=25.0, capacity_wh=400.0, cycles=10,
        )
        assert s.charge_percent == 80.0
        assert s.voltage == 3.8
        assert s.current == -2.0
        assert s.temperature == 25.0
        assert s.capacity_wh == 400.0
        assert s.cycles == 10


# ---------------------------------------------------------------------------
# BatteryModel tests
# ---------------------------------------------------------------------------

class TestBatteryModel:
    def test_default_values(self):
        m = BatteryModel(nominal_capacity=500.0, nominal_voltage=3.7, internal_resistance=0.05)
        assert m.temp_coeff == 0.005
        assert m.peukert_exponent == DEFAULT_PEUKERT_EXPONENT
        assert m.max_charge_voltage == 4.2

    def test_custom_peukert(self):
        m = BatteryModel(nominal_capacity=500.0, nominal_voltage=3.7,
                         internal_resistance=0.05, peukert_exponent=1.1)
        assert m.peukert_exponent == 1.1


# ---------------------------------------------------------------------------
# BatterySimulator discharge tests
# ---------------------------------------------------------------------------

class TestDischarge:
    def _full_state(self, capacity=500.0, soc=100.0, voltage=4.2):
        return BatteryState(
            charge_percent=soc, voltage=voltage, current=0.0,
            temperature=25.0, capacity_wh=capacity, cycles=0,
        )

    def _default_model(self, capacity=500.0):
        return BatteryModel(
            nominal_capacity=capacity, nominal_voltage=3.7,
            internal_resistance=0.05,
        )

    def test_basic_discharge(self):
        state = self._full_state()
        model = self._default_model()
        new_state = BatterySimulator.discharge(state, 10.0, 3600.0, model)
        assert new_state.capacity_wh < state.capacity_wh
        assert new_state.current == -10.0
        assert new_state.cycles == 0  # cycles not incremented by discharge

    def test_discharge_reduces_soc(self):
        state = self._full_state()
        model = self._default_model()
        new_state = BatterySimulator.discharge(state, 10.0, 3600.0, model)
        assert new_state.charge_percent < 100.0

    def test_discharge_voltage_drops(self):
        state = self._full_state()
        model = self._default_model()
        new_state = BatterySimulator.discharge(state, 10.0, 3600.0, model)
        assert new_state.voltage < state.voltage

    def test_discharge_zero_current(self):
        state = self._full_state()
        model = self._default_model()
        new_state = BatterySimulator.discharge(state, 0.0, 3600.0, model)
        assert new_state.capacity_wh == state.capacity_wh

    def test_discharge_cannot_go_below_zero(self):
        state = self._full_state(capacity=1.0, soc=0.2)
        model = self._default_model()
        new_state = BatterySimulator.discharge(state, 100.0, 36000.0, model)
        assert new_state.capacity_wh >= 0.0

    def test_discharge_without_model(self):
        state = self._full_state()
        new_state = BatterySimulator.discharge(state, 5.0, 3600.0)
        assert new_state.capacity_wh < state.capacity_wh

    def test_discharge_high_current_more_loss(self):
        """Peukert: higher current → proportionally more capacity consumed."""
        state1 = self._full_state()
        state2 = self._full_state()
        model = self._default_model()

        s1 = BatterySimulator.discharge(state1, 5.0, 3600.0, model)
        s2 = BatterySimulator.discharge(state2, 20.0, 3600.0, model)
        # 20A draws significantly more than 4x of 5A due to Peukert
        assert s2.capacity_wh < s1.capacity_wh * 4  # more than linear

    def test_discharge_longer_dt_more_loss(self):
        state = self._full_state()
        model = self._default_model()
        s1 = BatterySimulator.discharge(state, 10.0, 1800.0, model)
        s2 = BatterySimulator.discharge(state, 10.0, 7200.0, model)
        assert s2.capacity_wh < s1.capacity_wh

    def test_discharge_ir_drop(self):
        state = self._full_state()
        model = BatteryModel(nominal_capacity=500.0, nominal_voltage=3.7, internal_resistance=0.1)
        new_state = BatterySimulator.discharge(state, 10.0, 1.0, model)
        # With 10A * 0.1Ω = 1V IR drop
        assert new_state.voltage < state.voltage - 0.5


# ---------------------------------------------------------------------------
# BatterySimulator charge tests
# ---------------------------------------------------------------------------

class TestCharge:
    def _empty_state(self):
        return BatteryState(
            charge_percent=10.0, voltage=3.12, current=0.0,
            temperature=25.0, capacity_wh=50.0, cycles=0,
        )

    def _model(self, capacity=500.0):
        return BatteryModel(
            nominal_capacity=capacity, nominal_voltage=3.7,
            internal_resistance=0.05, max_charge_current=10.0,
        )

    def test_basic_charge(self):
        state = self._empty_state()
        model = self._model()
        new_state = BatterySimulator.charge(state, 5.0, 3600.0, model)
        assert new_state.capacity_wh > state.capacity_wh
        assert new_state.current > 0

    def test_charge_increases_soc(self):
        state = self._empty_state()
        model = self._model()
        new_state = BatterySimulator.charge(state, 5.0, 3600.0, model)
        assert new_state.charge_percent > state.charge_percent

    def test_charge_increases_voltage(self):
        state = self._empty_state()
        model = self._model()
        new_state = BatterySimulator.charge(state, 5.0, 3600.0, model)
        assert new_state.voltage >= state.voltage

    def test_charge_capped_at_nominal(self):
        state = BatteryState(
            charge_percent=99.0, voltage=4.15, current=0.0,
            temperature=25.0, capacity_wh=495.0, cycles=0,
        )
        model = self._model()
        new_state = BatterySimulator.charge(state, 10.0, 7200.0, model)
        assert new_state.capacity_wh <= model.nominal_capacity

    def test_charge_without_model(self):
        state = self._empty_state()
        new_state = BatterySimulator.charge(state, 5.0, 3600.0)
        assert new_state.capacity_wh > state.capacity_wh

    def test_charge_cv_phase_tapers(self):
        state = BatteryState(
            charge_percent=95.0, voltage=4.2, current=0.0,
            temperature=25.0, capacity_wh=475.0, cycles=0,
        )
        model = self._model()
        new_state = BatterySimulator.charge(state, 10.0, 3600.0, model)
        # In CV phase current should taper below 10A
        assert new_state.current < 10.0

    def test_charge_zero_current(self):
        state = self._empty_state()
        model = self._model()
        new_state = BatterySimulator.charge(state, 0.0, 3600.0, model)
        assert new_state.capacity_wh == state.capacity_wh


# ---------------------------------------------------------------------------
# BatterySimulator utility tests
# ---------------------------------------------------------------------------

class TestBatteryUtils:
    def test_compute_capacity_remaining(self):
        s = BatteryState(charge_percent=80.0, voltage=3.8, current=0.0,
                         temperature=25.0, capacity_wh=400.0)
        assert BatterySimulator.compute_capacity_remaining(s) == 400.0

    def test_capacity_remaining_clamps_zero(self):
        s = BatteryState(charge_percent=0.0, voltage=3.0, current=0.0,
                         temperature=25.0, capacity_wh=-10.0)
        assert BatterySimulator.compute_capacity_remaining(s) == 0.0

    def test_state_of_health_new(self):
        m = BatteryModel(nominal_capacity=500.0, nominal_voltage=3.7, internal_resistance=0.05)
        soh = BatterySimulator.compute_state_of_health(m, 500.0)
        assert soh == 100.0

    def test_state_of_health_degraded(self):
        m = BatteryModel(nominal_capacity=500.0, nominal_voltage=3.7, internal_resistance=0.05)
        soh = BatterySimulator.compute_state_of_health(m, 400.0)
        assert soh == 80.0

    def test_state_of_health_clamps(self):
        m = BatteryModel(nominal_capacity=500.0, nominal_voltage=3.7, internal_resistance=0.05)
        soh = BatterySimulator.compute_state_of_health(m, 600.0)
        assert soh == 100.0

    def test_state_of_health_zero_capacity(self):
        m = BatteryModel(nominal_capacity=0.0, nominal_voltage=3.7, internal_resistance=0.05)
        soh = BatterySimulator.compute_state_of_health(m, 100.0)
        assert soh == 0.0

    def test_estimate_cycles_remaining(self):
        assert BatterySimulator.estimate_cycles_remaining(95.0, 0.5) == 30  # (95-80)/0.5

    def test_estimate_cycles_remaining_zero(self):
        assert BatterySimulator.estimate_cycles_remaining(75.0, 0.5) == 0

    def test_estimate_cycles_remaining_no_degradation(self):
        assert BatterySimulator.estimate_cycles_remaining(90.0, 0.0) == 0

    def test_internal_resistance_standard(self):
        r = BatterySimulator.compute_internal_resistance(25.0, 100.0)
        assert r > 0

    def test_internal_resistance_high_temp(self):
        r1 = BatterySimulator.compute_internal_resistance(25.0, 100.0)
        r2 = BatterySimulator.compute_internal_resistance(45.0, 100.0)
        assert r2 > r1

    def test_internal_resistance_cold(self):
        r1 = BatterySimulator.compute_internal_resistance(25.0, 100.0)
        r2 = BatterySimulator.compute_internal_resistance(-10.0, 100.0)
        assert r2 > r1

    def test_internal_resistance_low_soc(self):
        r1 = BatterySimulator.compute_internal_resistance(25.0, 100.0)
        r2 = BatterySimulator.compute_internal_resistance(25.0, 10.0)
        assert r2 > r1

    def test_internal_resistance_with_model(self):
        m = BatteryModel(nominal_capacity=500.0, nominal_voltage=3.7, internal_resistance=0.1)
        r = BatterySimulator.compute_internal_resistance(25.0, 100.0, m)
        assert abs(r - 0.1) < 0.001  # base case: 25°C, 100% SoC


# ---------------------------------------------------------------------------
# BatterySimulator cycle simulation
# ---------------------------------------------------------------------------

class TestSimulateCycle:
    def _full_state(self):
        return BatteryState(
            charge_percent=100.0, voltage=4.2, current=0.0,
            temperature=25.0, capacity_wh=500.0, cycles=0,
        )

    def _model(self):
        return BatteryModel(nominal_capacity=500.0, nominal_voltage=3.7, internal_resistance=0.05)

    def test_simulate_cycle_increments_cycles(self):
        state = self._full_state()
        model = self._model()
        profile = [LoadPoint(current_a=5.0, duration_s=3600.0)]
        result = BatterySimulator.simulate_cycle(state, profile, model)
        assert result.cycles == 1

    def test_simulate_cycle_reduces_capacity(self):
        state = self._full_state()
        model = self._model()
        profile = [LoadPoint(current_a=10.0, duration_s=7200.0)]
        result = BatterySimulator.simulate_cycle(state, profile, model)
        assert result.capacity_wh < state.capacity_wh

    def test_simulate_cycle_multiple_load_points(self):
        state = self._full_state()
        model = self._model()
        profile = [
            LoadPoint(current_a=5.0, duration_s=1800.0),
            LoadPoint(current_a=20.0, duration_s=600.0),
            LoadPoint(current_a=2.0, duration_s=3600.0),
        ]
        result = BatterySimulator.simulate_cycle(state, profile, model)
        assert result.cycles == 1
        assert result.capacity_wh < state.capacity_wh

    def test_simulate_cycle_empty_profile(self):
        state = self._full_state()
        result = BatterySimulator.simulate_cycle(state, [])
        assert result.cycles == 1
        assert result.capacity_wh == state.capacity_wh


class TestDischargeThermal:
    def _full_state(self, capacity=500.0, soc=100.0, voltage=4.2):
        return BatteryState(
            charge_percent=soc, voltage=voltage, current=0.0,
            temperature=30.0, capacity_wh=capacity, cycles=0,
        )

    def _default_model(self, capacity=500.0):
        return BatteryModel(
            nominal_capacity=capacity, nominal_voltage=3.7,
            internal_resistance=0.05,
        )

    def test_discharge_temperature_unchanged(self):
        state = self._full_state()
        model = self._default_model()
        new_state = BatterySimulator.discharge(state, 5.0, 3600.0, model)
        assert new_state.temperature == state.temperature


class TestChargeThermal:
    def _empty_state(self):
        return BatteryState(
            charge_percent=10.0, voltage=3.12, current=0.0,
            temperature=25.0, capacity_wh=50.0, cycles=0,
        )

    def _model(self, capacity=500.0):
        return BatteryModel(
            nominal_capacity=capacity, nominal_voltage=3.7,
            internal_resistance=0.05, max_charge_current=10.0,
        )

    def test_charge_temperature_increases(self):
        state = self._empty_state()
        model = self._model()
        new_state = BatterySimulator.charge(state, 10.0, 3600.0, model)
        assert new_state.temperature > state.temperature
