"""Tests for NEXUS i.MX RT1050 hardware configuration."""

import pytest
from hardware.imx_rt.config_imxrt1050 import (
    IMXRT1050Config,
    GPIOPin,
    PinFunction,
    MemoryRegion,
    MemoryLayout,
    ClockConfig,
    CORE_COUNT,
    CPU_FREQ_MAX,
    CPU_FREQ_DEFAULT,
    SRAM_TOTAL,
    FLASH_TOTAL,
    ADC_CHANNELS,
    PWM_MODULES,
    UART_COUNT,
    SPI_COUNT,
    I2C_COUNT,
    CAN_FD_COUNT,
    ENET_COUNT,
    FLEXIO_COUNT,
    USB_COUNT,
)


@pytest.fixture
def config():
    cfg = IMXRT1050Config()
    cfg.configure_marine_controller()
    return cfg


class TestChipConstants:
    def test_core_count(self):
        assert CORE_COUNT == 1

    def test_cpu_freq(self):
        assert CPU_FREQ_MAX == 600_000_000

    def test_sram_total(self):
        assert SRAM_TOTAL == 512 * 1024  # 512 KB

    def test_no_internal_flash(self):
        assert FLASH_TOTAL == 0

    def test_adc_channels(self):
        assert ADC_CHANNELS == 4

    def test_uart_count(self):
        assert UART_COUNT == 8

    def test_enet_count(self):
        assert ENET_COUNT == 1

    def test_can_fd_count(self):
        assert CAN_FD_COUNT == 1

    def test_flexio_count(self):
        assert FLEXIO_COUNT == 2

    def test_usb_count(self):
        assert USB_COUNT == 2


class TestPinMapping:
    def test_default_pins(self, config):
        assert len(config.all_pins()) >= 10

    def test_get_pin(self, config):
        assert config.get_pin(PinFunction.I2C_SDA) == GPIOPin.GPIO_AD_B0_12

    def test_remap_pin(self, config):
        config.remap_pin(PinFunction.STATUS_LED, GPIOPin.GPIO_AD_B0_00)
        assert config.get_pin(PinFunction.STATUS_LED) == GPIOPin.GPIO_AD_B0_00


class TestMemoryLayout:
    def test_configure_marine(self, config):
        assert config.memory.total_allocated() > 0
        assert config.memory.fits_in_sram()

    def test_dtcm_region(self, config):
        r = config.memory.region_by_name("dtcm")
        assert r is not None
        assert r.size_bytes == 128 * 1024

    def test_sram_overflow(self):
        cfg = IMXRT1050Config()
        cfg.memory.add_region("overflow", 0x20000000, SRAM_TOTAL + 1, "ram")
        assert not cfg.memory.fits_in_sram()


class TestPeripheralAllocation:
    def test_allocate(self, config):
        key = config.allocate_peripheral("adc", "ADC", 0)
        assert key == "ADC0"

    def test_double_allocate_raises(self, config):
        config.allocate_peripheral("a", "LPUART", 0)
        with pytest.raises(ValueError, match="already allocated"):
            config.allocate_peripheral("b", "LPUART", 0)


class TestValidation:
    def test_valid(self, config):
        assert config.is_valid()


class TestSummary:
    def test_summary_keys(self, config):
        s = config.summary()
        assert "chip" in s
        assert "cpu" in s
        assert "sram_total_bytes" in s
        assert "enet_count" in s

    def test_summary_values(self, config):
        s = config.summary()
        assert s["chip"] == "i.MX RT1050"
        assert s["cpu"] == "Cortex-M7"
        assert s["sram_total_bytes"] == 512 * 1024
        assert s["enet_count"] == 1
