"""
Tests for NEXUS RP2040 Hardware Configuration Library.
"""

import pytest
from hardware.rp2040.config_rp2040 import (
    RP2040Config,
    ClockConfig,
    PinMapping,
    PinFunction,
    GPIOPin,
    MemoryLayout,
    MemoryRegion,
    CORE_COUNT,
    CPU_FREQ_MAX,
    CPU_FREQ_DEFAULT,
    SRAM_TOTAL,
    FLASH_TOTAL,
    PIO_COUNT,
    SM_PER_PIO,
    GPIO_COUNT,
)


class TestConstants:
    """Verify RP2040 hardware constants."""

    def test_core_count(self):
        assert CORE_COUNT == 2

    def test_cpu_freq_max(self):
        assert CPU_FREQ_MAX == 133_000_000

    def test_cpu_freq_default(self):
        assert CPU_FREQ_DEFAULT == 125_000_000

    def test_sram_total(self):
        assert SRAM_TOTAL == 264 * 1024

    def test_flash_total(self):
        assert FLASH_TOTAL == 2 * 1024 * 1024

    def test_pio_count(self):
        assert PIO_COUNT == 2

    def test_sm_per_pio(self):
        assert SM_PER_PIO == 4

    def test_gpio_count(self):
        assert GPIO_COUNT == 30


class TestClockConfig:
    def test_default_config(self):
        clock = ClockConfig()
        assert clock.frequency_hz == CPU_FREQ_DEFAULT
        assert clock.source == "external_xosc"

    def test_vco_frequency(self):
        clock = ClockConfig()
        assert clock.vco_freq_hz == 12_000_000 * 125  # 1500 MHz

    def test_actual_output_freq(self):
        clock = ClockConfig()
        expected = 12_000_000 * 125 // 6 // 2
        assert clock.actual_output_freq_hz == expected

    def test_valid_clock(self):
        clock = ClockConfig()
        assert clock.is_valid() is True

    def test_invalid_clock_too_fast(self):
        clock = ClockConfig(frequency_hz=200_000_000)
        errors = clock.validate()
        assert len(errors) > 0
        assert any("exceeds max" in e for e in errors)

    def test_invalid_clock_too_slow(self):
        clock = ClockConfig(frequency_hz=500_000)
        errors = clock.validate()
        assert any(">= 1 MHz" in e for e in errors)


class TestPinMapping:
    def test_valid_pin(self):
        pm = PinMapping(function=PinFunction.SONAR_TRIG, pin=0)
        assert pm.pin == 0

    def test_invalid_pin_raises(self):
        with pytest.raises(ValueError, match="Invalid GPIO pin"):
            PinMapping(function=PinFunction.SONAR_TRIG, pin=50)

    def test_pin_with_options(self):
        pm = PinMapping(
            function=PinFunction.UART_TX,
            pin=8,
            pull_up=True,
            invert=True,
            description="GPS TX",
        )
        assert pm.pull_up is True
        assert pm.invert is True


class TestMemoryLayout:
    def test_add_and_total(self):
        ml = MemoryLayout()
        ml.add_region("buf1", 0x20000000, 4096)
        ml.add_region("buf2", 0x20001000, 8192)
        assert ml.total_allocated() == 12288

    def test_fits_in_sram(self):
        ml = MemoryLayout()
        ml.add_region("small", 0x20000000, 1024)
        assert ml.fits_in_sram() is True

    def test_exceeds_sram(self):
        ml = MemoryLayout()
        ml.add_region("huge", 0x20000000, SRAM_TOTAL + 1)
        assert ml.fits_in_sram() is False

    def test_region_by_name(self):
        ml = MemoryLayout()
        ml.add_region("test_region", 0x20000000, 512, "test")
        r = ml.region_by_name("test_region")
        assert r is not None
        assert r.name == "test_region"

    def test_no_overlap(self):
        ml = MemoryLayout()
        ml.add_region("a", 0x20000000, 1024)
        ml.add_region("b", 0x20000400, 1024)
        assert ml.has_overlap() is False

    def test_overlap_detected(self):
        ml = MemoryLayout()
        ml.add_region("a", 0x20000000, 4096)
        ml.add_region("b", 0x20000F00, 1024)  # starts within a's range
        assert ml.has_overlap() is True

    def test_region_end(self):
        r = MemoryRegion("test", 0x20000000, 1024)
        assert r.end == 0x20000000 + 1024


class TestRP2040Config:
    def setup_method(self):
        self.config = RP2040Config()
        self.config.configure_marine_sensors()

    def test_default_pins_configured(self):
        pins = self.config.all_pins()
        assert len(pins) >= 10

    def test_sonar_pins(self):
        assert self.config.PIN_SONAR_TRIG == GPIOPin.GP0
        assert self.config.PIN_SONAR_ECHO == GPIOPin.GP1

    def test_servo_pins(self):
        assert self.config.PIN_SERVO_1 == GPIOPin.GP2
        assert self.config.PIN_SERVO_2 == GPIOPin.GP3

    def test_i2c_pins(self):
        assert self.config.PIN_I2C_SDA == GPIOPin.GP4
        assert self.config.PIN_I2C_SCL == GPIOPin.GP5

    def test_uart_pins(self):
        assert self.config.PIN_UART_TX == GPIOPin.GP8
        assert self.config.PIN_UART_RX == GPIOPin.GP9

    def test_status_led(self):
        assert self.config.PIN_STATUS_LED == GPIOPin.GP25

    def test_remap_pin(self):
        self.config.remap_pin(PinFunction.SONAR_TRIG, 15, "moved")
        assert self.config.PIN_SONAR_TRIG == 15

    def test_get_unknown_pin_raises(self):
        # Clear all pins so SONAR_TRIG is no longer configured
        self.config._pin_map.clear()
        with pytest.raises(KeyError, match="not configured"):
            self.config.get_pin(PinFunction.SONAR_TRIG)

    def test_set_clock_frequency(self):
        self.config.set_clock_frequency(133_000_000)
        assert self.config.clock.frequency_hz == 133_000_000

    def test_set_clock_too_high_raises(self):
        with pytest.raises(ValueError, match="Max clock"):
            self.config.set_clock_frequency(200_000_000)

    def test_set_clock_too_low_raises(self):
        with pytest.raises(ValueError, match="Minimum"):
            self.config.set_clock_frequency(500_000)

    def test_pio_allocate(self):
        result = self.config.allocate_pio("sonar", 0, 0)
        assert result == (0, 0)

    def test_pio_double_allocate_raises(self):
        self.config.allocate_pio("sonar", 0, 0)
        with pytest.raises(ValueError, match="already allocated"):
            self.config.allocate_pio("dup", 0, 0)

    def test_pio_release(self):
        self.config.allocate_pio("sonar", 0, 0)
        self.config.release_pio("sonar")
        assert "sonar" not in self.config.pio_allocation_summary()

    def test_pio_invalid_block(self):
        with pytest.raises(ValueError, match="PIO block must be"):
            self.config.allocate_pio("bad", 2, 0)

    def test_pio_invalid_sm(self):
        with pytest.raises(ValueError, match="State machine index"):
            self.config.allocate_pio("bad", 0, 5)

    def test_available_sm(self):
        avail = self.config.available_sm(0)
        assert avail == [0, 1, 2, 3]
        self.config.allocate_pio("test", 0, 2)
        avail = self.config.available_sm(0)
        assert 2 not in avail

    def test_used_gpio_set(self):
        used = self.config.used_gpio_set()
        assert GPIOPin.GP0 in used
        assert GPIOPin.GP1 in used

    def test_marine_memory_configured(self):
        assert self.config.memory.total_allocated() > 0
        assert self.config.memory.fits_in_sram()

    def test_summary_keys(self):
        s = self.config.summary()
        assert s["cpu_cores"] == 2
        assert "clock_hz" in s
        assert "sram_total_bytes" in s

    def test_pll_params(self):
        self.config.set_pll_params(100, 5, 3)
        assert self.config.clock.pll_fb_div == 100
        assert self.config.clock.pll_post_div1 == 5
        assert self.config.clock.pll_post_div2 == 3
