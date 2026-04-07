"""
Tests for NEXUS STM32F407 Configuration Module.

Covers: clock tree, DMA, GPIO, CAN peripheral, UART, ADC, power, memory layout,
        top-level config validation, serialisation, and cloning.
"""

import pytest
from hardware.stm32.config_stm32f4 import (
    ClockConfig, ClockSource, PLLState,
    DMAStreamConfig, DMAStream, DMAPriority,
    GPIOConfig, GPIOMode, GPIOPull,
    CANPeripheralConfig,
    UARTConfig,
    ADCConfig,
    PowerConfig, PowerMode, VoltageScale,
    MemoryLayout,
    STM32F407Config,
)


# ===================================================================
# ClockConfig Tests
# ===================================================================

class TestClockConfig:
    def test_default_creates_168mhz(self):
        clk = ClockConfig()
        assert clk.sysclk_hz == 168_000_000

    def test_apb1_max_42mhz(self):
        clk = ClockConfig()
        assert clk.apb1_hz == 42_000_000

    def test_apb2_max_84mhz(self):
        clk = ClockConfig()
        assert clk.apb2_hz == 84_000_000

    def test_usb_48mhz(self):
        clk = ClockConfig()
        assert clk.usb_hz == 48_000_000

    def test_pll_disabled_returns_hse(self):
        clk = ClockConfig(pll_state=PLLState.DISABLED)
        assert clk.sysclk_hz == clk.hse_freq_hz

    def test_hse_source(self):
        clk = ClockConfig(source=ClockSource.HSE)
        assert clk.source == ClockSource.HSE

    def test_custom_pll_settings(self):
        clk = ClockConfig(pllm=4, plln=168, pllp=2)
        # VCO_in = 8MHz/4 = 2MHz, VCO_out = 2MHz*168 = 336MHz, SYSCLK = 336/2 = 168MHz
        assert clk.sysclk_hz == 168_000_000

    def test_validate_no_errors_for_defaults(self):
        clk = ClockConfig()
        assert clk.validate() == []

    def test_validate_bad_pllm(self):
        clk = ClockConfig(pllm=0)
        errors = clk.validate()
        assert any("PLLM" in e for e in errors)

    def test_validate_bad_pllp(self):
        clk = ClockConfig(pllp=3)
        errors = clk.validate()
        assert any("PLLP" in e for e in errors)


# ===================================================================
# DMAStreamConfig Tests
# ===================================================================

class TestDMAStreamConfig:
    def test_default_config(self):
        dma = DMAStreamConfig(stream=DMAStream.DMA1_S0)
        assert dma.stream == DMAStream.DMA1_S0
        assert dma.circular is False

    def test_circular_mode(self):
        dma = DMAStreamConfig(stream=DMAStream.DMA2_S3, circular=True)
        assert dma.circular is True

    def test_all_dma_streams(self):
        """Verify all 16 DMA streams are defined."""
        assert len(DMAStream) == 16


# ===================================================================
# GPIOConfig Tests
# ===================================================================

class TestGPIOConfig:
    def test_pin_name(self):
        gpio = GPIOConfig(port="A", pin=5)
        assert gpio.pin_name == "PA5"

    def test_pwm_af_mode(self):
        gpio = GPIOConfig(port="A", pin=8, mode=GPIOMode.AF_PP, alternate=1)
        assert gpio.mode == GPIOMode.AF_PP
        assert gpio.alternate == 1

    def test_analog_mode(self):
        gpio = GPIOConfig(port="C", pin=0, mode=GPIOMode.ANALOG, pull=GPIOPull.NONE)
        assert gpio.mode == GPIOMode.ANALOG


# ===================================================================
# CAN Peripheral Tests
# ===================================================================

class TestCANPeripheralConfig:
    def test_default_baud_rate(self):
        can = CANPeripheralConfig()
        assert can.baud_rate == 250_000

    def test_bit_rate_calculation(self):
        can = CANPeripheralConfig(prescaler=14, time_seg1=9, time_seg2=2)
        assert can.bit_rate == 250_000

    def test_sample_point(self):
        can = CANPeripheralConfig(prescaler=14, time_seg1=9, time_seg2=2)
        # (1 + 9) / (1 + 9 + 2) = 10/12 = 83.33%
        assert can.sample_point_pct == pytest.approx(83.33, rel=1e-3)


# ===================================================================
# UART & ADC Tests
# ===================================================================

class TestUARTConfig:
    def test_default_baud(self):
        uart = UARTConfig()
        assert uart.baud_rate == 115200

    def test_dma_streams_optional(self):
        uart = UARTConfig(dma_rx_stream=DMAStream.DMA1_S5, dma_tx_stream=DMAStream.DMA1_S6)
        assert uart.dma_rx_stream == DMAStream.DMA1_S5


class TestADCConfig:
    def test_default_12bit(self):
        adc = ADCConfig()
        assert adc.resolution_bits == 12
        assert adc.validate() == []

    def test_invalid_resolution(self):
        adc = ADCConfig(resolution_bits=7)
        errors = adc.validate()
        assert len(errors) > 0

    def test_too_many_channels(self):
        adc = ADCConfig(channels=list(range(20)))
        errors = adc.validate()
        assert any("Too many" in e for e in errors)


# ===================================================================
# MemoryLayout Tests
# ===================================================================

class TestMemoryLayout:
    def test_total_ram(self):
        mem = MemoryLayout()
        assert mem.total_ram_kb == 192  # 128 + 64 (excl. backup domain)

    def test_flash_address(self):
        mem = MemoryLayout()
        assert mem.flash_start == 0x08000000


# ===================================================================
# Top-level STM32F407Config Tests
# ===================================================================

class TestSTM32F407Config:
    def test_defaults(self):
        cfg = STM32F407Config()
        assert cfg.part_number == "STM32F407VG"
        assert cfg.core == "Cortex-M4F"
        assert cfg.flash_kb == 1024
        assert cfg.sram_kb == 192

    def test_validate_passes(self):
        cfg = STM32F407Config()
        assert cfg.validate() == []

    def test_to_dict(self):
        cfg = STM32F407Config(node_name="thruster_node", node_id=1)
        d = cfg.to_dict()
        assert d["node_name"] == "thruster_node"
        assert d["node_id"] == 1

    def test_clone_independence(self):
        cfg = STM32F407Config()
        clone = cfg.clone()
        clone.node_id = 99
        assert cfg.node_id != 99

    def test_repr(self):
        cfg = STM32F407Config()
        r = repr(cfg)
        assert "STM32F407Config" in r
        assert "168MHz" in r

    def test_summary(self):
        cfg = STM32F407Config()
        s = cfg.summary()
        assert "168.0 MHz" in s
        assert "1024 KB" in s

    def test_custom_clock_validation(self):
        cfg = STM32F407Config()
        cfg.clock.pllm = 0  # Invalid
        errors = cfg.validate()
        assert len(errors) > 0

    def test_flash_sram_mismatch(self):
        cfg = STM32F407Config()
        cfg.sram_kb = 999
        errors = cfg.validate()
        assert any("mismatch" in e for e in errors)
