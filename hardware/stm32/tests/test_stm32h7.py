"""
Tests for NEXUS STM32H743 Configuration Module.

Covers: PLL config, clock tree, memory layout, cache, power, top-level config.
"""

import pytest
from hardware.stm32.config_stm32h7 import (
    H7ClockSource, H7PLLState, H7VoltageScale, H7PowerMode,
    H7PLLConfig,
    H7ClockConfig,
    H7MemoryLayout,
    H7PowerConfig,
    H7DMAConfig, DMAPriority,
    H7CacheConfig,
    STM32H743Config,
)


class TestH7PLLConfig:
    def test_default_480mhz(self):
        pll = H7PLLConfig(divm=5, divn=192, divp=2)
        # VCO_in = 25MHz/5 = 5MHz, VCO_out = 5MHz*192 = 960MHz, P = 960/2 = 480MHz
        assert pll.p_freq_hz == 480_000_000

    def test_vco_input_range(self):
        pll = H7PLLConfig(divm=5, divn=192, divp=2)
        assert 1_000_000 <= pll.vco_in_hz <= 16_000_000

    def test_vco_output_range(self):
        pll = H7PLLConfig(divm=5, divn=192, divp=2)
        assert 128_000_000 <= pll.vco_out_hz <= 960_000_000

    def test_validate_ok(self):
        pll = H7PLLConfig()
        assert pll.validate() == []

    def test_validate_bad_divm(self):
        pll = H7PLLConfig(divm=100)
        errors = pll.validate()
        assert any("DIVM" in e for e in errors)

    def test_validate_bad_divn(self):
        pll = H7PLLConfig(divn=2)
        errors = pll.validate()
        assert any("DIVN" in e for e in errors)


class TestH7ClockConfig:
    def test_default_sysclk(self):
        clk = H7ClockConfig()
        assert clk.sysclk_hz == 480_000_000

    def test_hclk_from_prescaler(self):
        clk = H7ClockConfig(hpre=2)
        assert clk.hclk_hz == clk.sysclk_hz // 2

    def test_validate_no_errors(self):
        clk = H7ClockConfig()
        assert clk.validate() == []


class TestH7MemoryLayout:
    def test_total_sram(self):
        mem = H7MemoryLayout()
        assert mem.total_sram_kb == 964  # 64+128+512+128+128+4

    def test_axi_sram_largest(self):
        mem = H7MemoryLayout()
        assert mem.axi_sram_size_kb == 512

    def test_flash_address(self):
        mem = H7MemoryLayout()
        assert mem.flash_start == 0x08000000


class TestH7PowerConfig:
    def test_default_vos0(self):
        pwr = H7PowerConfig()
        assert pwr.vos == H7VoltageScale.VOS0
        assert pwr.ionen is True

    def test_overdrive_required_for_vos0(self):
        pwr = H7PowerConfig(vos=H7VoltageScale.VOS0, ionen=False)
        cfg = STM32H743Config(power=pwr)
        errors = cfg.validate()
        assert any("Overdrive" in e for e in errors)


class TestH7CacheConfig:
    def test_cache_enabled(self):
        cache = H7CacheConfig()
        assert cache.icache_enabled is True
        assert cache.dcache_enabled is True


class TestH7DMAConfig:
    def test_default(self):
        dma = H7DMAConfig()
        assert dma.name == "DMA1"
        assert dma.priority == DMAPriority.MEDIUM


class TestSTM32H743Config:
    def test_defaults(self):
        cfg = STM32H743Config()
        assert cfg.part_number == "STM32H743VI"
        assert cfg.core == "Cortex-M7F"
        assert cfg.flash_kb == 1024
        assert cfg.dp_fpu is True

    def test_validate_ok(self):
        cfg = STM32H743Config()
        assert cfg.validate() == []

    def test_to_dict(self):
        cfg = STM32H743Config(node_name="nav_computer")
        d = cfg.to_dict()
        assert d["node_name"] == "nav_computer"

    def test_clone_independence(self):
        cfg = STM32H743Config()
        clone = cfg.clone()
        clone.node_id = 50
        assert cfg.node_id != 50

    def test_repr(self):
        cfg = STM32H743Config()
        r = repr(cfg)
        assert "STM32H743Config" in r
        assert "480MHz" in r
        assert "fpu=dp" in r

    def test_summary(self):
        cfg = STM32H743Config()
        s = cfg.summary()
        assert "480.0 MHz" in s
        assert "I-Cache" in s
        assert "DP FPU" in s
