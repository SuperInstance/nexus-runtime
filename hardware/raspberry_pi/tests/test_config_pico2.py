"""Tests for NEXUS Raspberry Pi Pico 2 (RP2350) configuration."""

import pytest
from hardware.raspberry_pi.config_pico2 import (
    Pico2Config,
    GPIOPin,
    PinFunction,
    PinMapping,
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
    ADC_CHANNELS,
    ADC_RESOLUTION,
    PWM_SLICES,
)


@pytest.fixture
def config():
    return Pico2Config()


class TestPico2Constants:
    def test_core_count(self):
        assert CORE_COUNT == 2

    def test_cpu_freq_max(self):
        assert CPU_FREQ_MAX == 150_000_000

    def test_sram_total(self):
        assert SRAM_TOTAL == 520 * 1024

    def test_flash_total(self):
        assert FLASH_TOTAL == 4 * 1024 * 1024

    def test_gpio_count(self):
        assert GPIO_COUNT == 30

    def test_pio_count(self):
        assert PIO_COUNT == 3

    def test_sm_per_pio(self):
        assert SM_PER_PIO == 4

    def test_adc_channels(self):
        assert ADC_CHANNELS == 4

    def test_adc_resolution(self):
        assert ADC_RESOLUTION == 12

    def test_pwm_slices(self):
        assert PWM_SLICES == 16


class TestPico2PinMapping:
    def test_valid_pin(self, config):
        p = config.get_pin(PinFunction.STATUS_LED)
        assert p.pin == 25

    def test_i2c_pins(self, config):
        assert config.PIN_I2C_SDA == 12
        assert config.PIN_I2C_SCL == 13

    def test_uart_pins(self, config):
        assert config.PIN_UART_TX == 8
        assert config.PIN_UART_RX == 9

    def test_sonar_pins(self, config):
        assert config.PIN_SONAR_TRIG == 0
        assert config.PIN_SONAR_ECHO == 1

    def test_remap_pin(self, config):
        config.remap_pin(PinFunction.STATUS_LED, 10)
        assert config.PIN_STATUS_LED == 10

    def test_invalid_pin_raises(self):
        with pytest.raises(ValueError, match="Invalid GPIO pin"):
            PinMapping(function=PinFunction.STATUS_LED, pin=50)

    def test_all_pins(self, config):
        pins = config.all_pins()
        assert len(pins) >= 20

    def test_used_gpio_set(self, config):
        used = config.used_gpio_set()
        assert len(used) >= 20


class TestPico2PIOAllocation:
    def test_allocate_pio0(self, config):
        result = config.allocate_pio("sonar", 0, 0)
        assert result == (0, 0)

    def test_allocate_pio2(self, config):
        result = config.allocate_pio("spi_protocol", 2, 1)
        assert result == (2, 1)

    def test_double_allocate_raises(self, config):
        config.allocate_pio("test", 0, 0)
        with pytest.raises(ValueError, match="already allocated"):
            config.allocate_pio("test2", 0, 0)

    def test_invalid_pio_block(self, config):
        with pytest.raises(ValueError, match="PIO block must be"):
            config.allocate_pio("bad", 3, 0)

    def test_release_pio(self, config):
        config.allocate_pio("temp", 1, 2)
        config.release_pio("temp")
        assert "temp" not in config.pio_allocation_summary()

    def test_available_sm(self, config):
        config.allocate_pio("test", 0, 0)
        avail = config.available_sm(0)
        assert 0 not in avail
        assert 1 in avail


class TestPico2Memory:
    def test_configure_marine_sensors(self, config):
        config.configure_marine_sensors()
        assert config.memory.total_allocated() > 0
        assert config.memory.fits_in_sram()

    def test_sonar_samples_region(self, config):
        config.configure_marine_sensors()
        r = config.memory.region_by_name("sonar_samples")
        assert r is not None
        assert r.size_bytes == 32768

    def test_region_end(self):
        r = MemoryRegion("test", 0x20000000, 1024, "test")
        assert r.end == 0x20000000 + 1024

    def test_memory_overflow(self, config):
        config.memory.add_region("huge", 0x20000000, SRAM_TOTAL + 1, "test")
        assert config.memory.fits_in_sram() is False


class TestPico2ThermalAndPower:
    def test_idle_power(self, config):
        assert config.power.idle_w == 0.05

    def test_max_power(self, config):
        assert config.power.max_load_w == 0.30

    def test_solar_compatible(self, config):
        assert config.power.solar_compatible is True


class TestPico2ConfigHash:
    def test_hash_is_sha256(self, config):
        assert len(config.get_config_hash()) == 64

    def test_hash_deterministic(self, config):
        assert config.get_config_hash() == config.get_config_hash()


class TestPico2Summary:
    def test_summary_keys(self, config):
        s = config.summary()
        assert "board" in s
        assert "soc" in s
        assert "pio_blocks" in s
        assert "pwm_slices" in s
        assert "adc_channels" in s

    def test_summary_values(self, config):
        s = config.summary()
        assert s["soc"] == "RP2350"
        assert s["sram_total_kb"] == 520
        assert s["pio_blocks"] == 3
        assert s["pwm_slices"] == 16

    def test_repr(self, config):
        assert "RP2350" in repr(config)
        assert "Cortex-M33" in repr(config)
