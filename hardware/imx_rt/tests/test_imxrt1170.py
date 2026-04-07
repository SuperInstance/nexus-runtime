"""Tests for NEXUS i.MX RT1170 dual-core hardware configuration."""

import pytest
from hardware.imx_rt.config_imxrt1170 import (
    IMXRT1170Config,
    GPIOPin,
    PinFunction,
    MemoryRegion,
    MemoryLayout,
    ClockConfig,
    CORE_COUNT,
    APP_CORE_FREQ_MAX,
    APP_CORE_FREQ_DEFAULT,
    NET_CORE_FREQ_MAX,
    NET_CORE_FREQ_DEFAULT,
    APP_ITCM_SIZE,
    APP_DTCM_SIZE,
    NET_TCM_SIZE,
    SHARED_OCRAM_SIZE,
    APP_SRAM_TOTAL,
    NET_SRAM_TOTAL,
    SRAM_TOTAL,
    FLASH_TOTAL,
    ADC_CHANNELS,
    UART_COUNT,
    CAN_FD_COUNT,
    ENET_COUNT,
    FLEXIO_COUNT,
    USB_COUNT,
)


@pytest.fixture
def config():
    cfg = IMXRT1170Config()
    cfg.configure_marine_autopilot()
    return cfg


class TestChipConstants:
    def test_total_cores(self):
        assert CORE_COUNT == 2

    def test_m7_freq(self):
        assert APP_CORE_FREQ_MAX == 1_000_000_000

    def test_m4_freq(self):
        assert NET_CORE_FREQ_MAX == 400_000_000

    def test_sram_total(self):
        assert SRAM_TOTAL == APP_SRAM_TOTAL + NET_SRAM_TOTAL + SHARED_OCRAM_SIZE

    def test_app_itcm(self):
        assert APP_ITCM_SIZE == 256 * 1024

    def test_app_dtcm(self):
        assert APP_DTCM_SIZE == 256 * 1024

    def test_net_tcm(self):
        assert NET_TCM_SIZE == 128 * 1024

    def test_shared_ocram(self):
        assert SHARED_OCRAM_SIZE == 2560 * 1024

    def test_no_internal_flash(self):
        assert FLASH_TOTAL == 0

    def test_adc_channels(self):
        assert ADC_CHANNELS == 10

    def test_uart_count(self):
        assert UART_COUNT == 10

    def test_enet_count(self):
        assert ENET_COUNT == 3

    def test_flexio_count(self):
        assert FLEXIO_COUNT == 2


class TestPinMapping:
    def test_default_pins(self, config):
        assert len(config.all_pins()) >= 10

    def test_get_pin(self, config):
        assert config.get_pin(PinFunction.I2C_SDA) == GPIOPin.GPIO_AD_B0_12

    def test_remap_pin(self, config):
        config.remap_pin(PinFunction.STATUS_LED, GPIOPin.GPIO_AD_B0_00)
        assert config.get_pin(PinFunction.STATUS_LED) == GPIOPin.GPIO_AD_B0_00

    def test_missing_pin_raises(self):
        cfg = IMXRT1170Config()
        cfg._pin_map.clear()
        with pytest.raises(KeyError):
            cfg.get_pin(PinFunction.STATUS_LED)


class TestClockConfig:
    def test_default_valid(self):
        c = ClockConfig()
        assert c.is_valid()

    def test_m7_exceeds(self):
        c = ClockConfig(m7_freq_hz=1_200_000_000)
        errors = c.validate()
        assert len(errors) > 0

    def test_m4_exceeds(self):
        c = ClockConfig(m4_freq_hz=500_000_000)
        errors = c.validate()
        assert len(errors) > 0


class TestMemoryLayout:
    def test_dual_core_regions(self, config):
        m7_itcm = config.memory.region_by_name("m7_itcm")
        m4_tcm = config.memory.region_by_name("m4_tcm")
        shared_ipc = config.memory.region_by_name("shared_ipc")
        assert m7_itcm is not None
        assert m4_tcm is not None
        assert shared_ipc is not None

    def test_fits_in_sram(self, config):
        assert config.memory.fits_in_sram()

    def test_sram_overflow(self):
        cfg = IMXRT1170Config()
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
        assert "app_cpu" in s
        assert "net_cpu" in s
        assert "sram_total_bytes" in s

    def test_summary_values(self, config):
        s = config.summary()
        assert s["chip"] == "i.MX RT1170"
        assert "Cortex-M7" in s["app_cpu"]
        assert "Cortex-M4" in s["net_cpu"]
        assert s["enet_count"] == 3
        assert s["uart_count"] == 10
