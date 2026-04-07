"""
Tests for NEXUS PIO Program Configurations.
"""

import pytest
from hardware.rp2040.pio_programs import (
    SonarPingProgram,
    SonarTimingParams,
    ServoPWMProgram,
    ServoPWMParams,
    UARTBridgeProgram,
    UARTBridgeParams,
    PIOProgramRegistry,
    PIOInstruction,
)


class TestSonarTimingParams:
    def test_default_ping_pulse(self):
        t = SonarTimingParams()
        assert t.ping_pulse_us == 10

    def test_distance_from_echo(self):
        t = SonarTimingParams(speed_of_sound_m_s=1500.0)
        # 1ms echo -> distance = (0.001 * 1500) / 2 = 0.75m
        dist = t.distance_from_echo_us(1000)
        assert abs(dist - 0.75) < 0.01

    def test_distance_zero_echo(self):
        t = SonarTimingParams()
        assert t.distance_from_echo_us(0) == 0.0

    def test_distance_negative_echo(self):
        t = SonarTimingParams()
        assert t.distance_from_echo_us(-5) == 0.0

    def test_echo_from_distance(self):
        t = SonarTimingParams(speed_of_sound_m_s=1500.0)
        echo = t.echo_us_from_distance(0.75)
        assert echo == 1000

    def test_echo_zero_distance(self):
        t = SonarTimingParams()
        assert t.echo_us_from_distance(0) == 0

    def test_echo_timeout_cycles(self):
        t = SonarTimingParams(max_echo_wait_ms=30)
        # 30ms * 1000 * 1.25 = 37500 cycles
        assert t.echo_timeout_cycles == 37500

    def test_max_range(self):
        t = SonarTimingParams()
        assert t.max_range_m == 4.0


class TestSonarPingProgram:
    def test_assemble(self):
        prog = SonarPingProgram(trigger_pin=0, echo_pin=1)
        asm = prog.assemble()
        assert len(asm) > 0
        assert prog.assembled is True

    def test_instruction_count(self):
        prog = SonarPingProgram()
        prog.assemble()
        assert prog.instruction_count == len(prog.ASSEMBLY)

    def test_config_dict(self):
        prog = SonarPingProgram(trigger_pin=5, echo_pin=6)
        d = prog.get_config_dict()
        assert d["trigger_pin"] == 5
        assert d["echo_pin"] == 6
        assert "ping_pulse_us" in d


class TestServoPWMParams:
    def test_default_frequency(self):
        p = ServoPWMParams()
        assert p.frequency_hz == 50

    def test_period_cycles(self):
        p = ServoPWMParams(frequency_hz=50, clock_div=1.0)
        expected = int(125_000_000 / 50)
        assert p.period_cycles == expected

    def test_angle_to_duty_min(self):
        p = ServoPWMParams(clock_div=1.0)
        duty_0 = p.angle_to_duty(0.0)
        duty_180 = p.angle_to_duty(180.0)
        assert duty_0 < duty_180

    def test_angle_clamp(self):
        p = ServoPWMParams(clock_div=1.0)
        duty_neg = p.angle_to_duty(-10.0)
        duty_over = p.angle_to_duty(200.0)
        assert duty_neg == p.angle_to_duty(0.0)
        assert duty_over == p.angle_to_duty(180.0)

    def test_duty_to_angle_roundtrip(self):
        p = ServoPWMParams(clock_div=1.0)
        for angle in [0.0, 45.0, 90.0, 135.0, 180.0]:
            duty = p.angle_to_duty(angle)
            recovered = p.duty_to_angle(duty)
            assert abs(recovered - angle) < 1.0


class TestServoPWMProgram:
    def test_assemble(self):
        prog = ServoPWMProgram()
        asm = prog.assemble()
        assert len(asm) > 0
        assert prog.assembled is True

    def test_config_dict(self):
        prog = ServoPWMProgram(pins=[10, 11])
        d = prog.get_config_dict()
        assert d["pins"] == [10, 11]
        assert d["frequency_hz"] == 50

    def test_custom_params(self):
        params = ServoPWMParams(frequency_hz=100, min_pulse_us=500, max_pulse_us=2500)
        prog = ServoPWMProgram(params=params)
        d = prog.get_config_dict()
        assert d["frequency_hz"] == 100
        assert d["min_pulse_us"] == 500


class TestUARTBridgeParams:
    def test_default_valid(self):
        params = UARTBridgeParams()
        assert params.is_valid() is True

    def test_bit_period(self):
        params = UARTBridgeParams(baud_rate=9600, clock_div=1.0)
        expected = int(125_000_000 / 9600)
        assert params.bit_period_cycles == expected

    def test_frame_bits_no_parity(self):
        params = UARTBridgeParams(bits_per_char=8, stop_bits=1, parity="none")
        assert params.frame_bits == 9

    def test_frame_bits_with_parity(self):
        params = UARTBridgeParams(bits_per_char=8, stop_bits=1, parity="even")
        assert params.frame_bits == 10

    def test_invalid_baud(self):
        params = UARTBridgeParams(baud_rate=100)
        errors = params.validate()
        assert any("Baud rate" in e for e in errors)

    def test_invalid_bits_per_char(self):
        params = UARTBridgeParams(bits_per_char=9)
        errors = params.validate()
        assert any("Bits per char" in e for e in errors)

    def test_invalid_parity(self):
        params = UARTBridgeParams(parity="mark")
        errors = params.validate()
        assert any("Parity" in e for e in errors)


class TestUARTBridgeProgram:
    def test_assemble(self):
        prog = UARTBridgeProgram()
        result = prog.assemble()
        assert "tx" in result
        assert "rx" in result
        assert prog.assembled is True

    def test_instruction_count(self):
        prog = UARTBridgeProgram()
        prog.assemble()
        total = len(prog.TX_ASSEMBLY) + len(prog.RX_ASSEMBLY)
        assert prog.instruction_count == total

    def test_config_dict(self):
        prog = UARTBridgeProgram(tx_pin=10, rx_pin=11, params=UARTBridgeParams(baud_rate=115200))
        d = prog.get_config_dict()
        assert d["tx_pin"] == 10
        assert d["rx_pin"] == 11
        assert d["baud_rate"] == 115200


class TestPIOProgramRegistry:
    def test_register_and_get(self):
        reg = PIOProgramRegistry()
        prog = SonarPingProgram()
        reg.register("sonar", prog)
        assert reg.get("sonar") is prog

    def test_list_programs(self):
        reg = PIOProgramRegistry()
        reg.register("a", SonarPingProgram())
        reg.register("b", ServoPWMProgram())
        assert len(reg.list_programs()) == 2

    def test_duplicate_register_raises(self):
        reg = PIOProgramRegistry()
        reg.register("dup", SonarPingProgram())
        with pytest.raises(ValueError, match="already registered"):
            reg.register("dup", SonarPingProgram())

    def test_unregister(self):
        reg = PIOProgramRegistry()
        reg.register("tmp", SonarPingProgram())
        reg.unregister("tmp")
        assert "tmp" not in reg.list_programs()

    def test_get_missing_raises(self):
        reg = PIOProgramRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_assembled_programs(self):
        reg = PIOProgramRegistry()
        p1 = SonarPingProgram()
        p2 = ServoPWMProgram()
        reg.register("sonar", p1)
        reg.register("servo", p2)
        assert len(reg.assembled_programs()) == 0
        p1.assemble()
        assert reg.assembled_programs() == ["sonar"]

    def test_assemble_all(self):
        reg = PIOProgramRegistry()
        reg.register("s1", SonarPingProgram())
        reg.register("s2", ServoPWMProgram())
        results = reg.assemble_all()
        assert all(results.values())
