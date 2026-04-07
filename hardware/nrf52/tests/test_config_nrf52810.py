"""Tests for NEXUS nRF52810 ultra-low-cost hardware configuration."""

import pytest
from hardware.nrf52.config_nrf52810 import (
    NRF52810Config,
    PinConfig,
    PinFunction,
    GPIOPort,
    ProtocolConfig,
    MemoryLayout,
    CORE_COUNT,
    CPU_FREQ_MAX,
    CPU_FREQ_DEFAULT,
    FLASH_TOTAL,
    RAM_TOTAL,
    GPIO_COUNT,
    ADC_CHANNELS,
    SPI_COUNT,
    I2C_COUNT,
    UART_COUNT,
    PWM_COUNT,
    TIMER_COUNT,
    CODE_PAGE_SIZE,
    CODE_PAGES,
)


@pytest.fixture
def config():
    cfg = NRF52810Config()
    cfg.configure_sensor_tag()
    return cfg


class TestChipConstants:
    def test_core_count(self):
        assert CORE_COUNT == 1

    def test_cpu_freq(self):
        assert CPU_FREQ_MAX == 64_000_000

    def test_flash_total(self):
        assert FLASH_TOTAL == 192 * 1024

    def test_ram_total(self):
        assert RAM_TOTAL == 24 * 1024

    def test_gpio_count(self):
        assert GPIO_COUNT == 32

    def test_spi_count(self):
        assert SPI_COUNT == 1

    def test_i2c_count(self):
        assert I2C_COUNT == 1

    def test_uart_count(self):
        assert UART_COUNT == 1

    def test_pwm_count(self):
        assert PWM_COUNT == 1

    def test_timer_count(self):
        assert TIMER_COUNT == 3

    def test_code_pages(self):
        assert CODE_PAGE_SIZE == 4096
        assert CODE_PAGES == 48


class TestProtocolConfig:
    def test_default_valid(self):
        p = ProtocolConfig()
        assert p.is_valid()

    def test_max_conn_4(self):
        p = ProtocolConfig(ble_max_conn=4)
        assert p.is_valid()

    def test_max_conn_exceeds(self):
        p = ProtocolConfig(ble_max_conn=5)
        errors = p.validate()
        assert any("1-4" in e for e in errors)

    def test_mtu_max_158(self):
        p = ProtocolConfig(ble_mtu=159)
        errors = p.validate()
        assert any("MTU" in e for e in errors)


class TestNRF52810Config:
    def test_default_pins(self, config):
        assert len(config.all_pins()) >= 5

    def test_status_led(self, config):
        assert config.PIN_STATUS_LED.pin == 17

    def test_remap_pin(self, config):
        config.remap_pin(PinFunction.STATUS_LED, 0, 10)
        assert config.PIN_STATUS_LED.pin == 10

    def test_configure_ble(self, config):
        config.configure_ble(max_conn=2, mtu=23, tx_power=4)
        assert config.protocol.ble_max_conn == 2
        assert config.protocol.ble_mtu == 23

    def test_sensor_tag_memory(self, config):
        assert config.memory.total_ram() > 0
        assert config.memory.fits_in_ram()
        assert config.memory.fits_in_flash()

    def test_stack_region(self, config):
        r = config.memory.region_by_name("stack")
        assert r is not None
        assert r.size_bytes == 2048

    def test_validate_clean(self, config):
        assert config.is_valid()

    def test_validate_flash_overflow(self):
        from hardware.nrf52.config_nrf52810 import FLASH_BASE
        cfg = NRF52810Config()
        cfg.configure_sensor_tag()
        cfg.memory.add_region("overflow", FLASH_BASE, FLASH_TOTAL + 1, "flash")
        assert not cfg.is_valid()

    def test_summary(self, config):
        s = config.summary()
        assert s["chip"] == "nRF52810"
        assert s["cpu"] == "Cortex-M4"
        assert s["flash_bytes"] == 192 * 1024
        assert s["ram_bytes"] == 24 * 1024
