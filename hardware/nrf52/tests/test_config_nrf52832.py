"""Tests for NEXUS nRF52832 hardware configuration."""

import pytest
from hardware.nrf52.config_nrf52832 import (
    NRF52832Config,
    PinConfig,
    PinFunction,
    GPIOPort,
    PinDrive,
    PinPull,
    MemoryLayout,
    MemoryRegion,
    ProtocolConfig,
    CORE_COUNT,
    CPU_FREQ_MAX,
    CPU_FREQ_DEFAULT,
    FLASH_TOTAL,
    RAM_TOTAL,
    GPIO_COUNT,
    ADC_RESOLUTION,
    ADC_CHANNELS,
    SPI_COUNT,
    I2C_COUNT,
    UART_COUNT,
    RADIO_BLE_5,
    NFC_AVAILABLE,
    CODE_PAGE_SIZE,
    CODE_PAGES,
)


@pytest.fixture
def config():
    cfg = NRF52832Config()
    cfg.configure_marine_node()
    return cfg


class TestChipConstants:
    def test_core_count(self):
        assert CORE_COUNT == 1

    def test_cpu_freq(self):
        assert CPU_FREQ_MAX == 64_000_000

    def test_flash_total(self):
        assert FLASH_TOTAL == 512 * 1024

    def test_ram_total(self):
        assert RAM_TOTAL == 64 * 1024

    def test_gpio_count(self):
        assert GPIO_COUNT == 32  # No Port 1

    def test_spi_count(self):
        assert SPI_COUNT == 3

    def test_i2c_count(self):
        assert I2C_COUNT == 2

    def test_uart_count(self):
        assert UART_COUNT == 1

    def test_no_ble5(self):
        assert RADIO_BLE_5 is False

    def test_no_nfc(self):
        assert NFC_AVAILABLE is False

    def test_code_pages(self):
        assert CODE_PAGE_SIZE == 4096
        assert CODE_PAGES == 128


class TestPinConfig:
    def test_valid_pin(self):
        p = PinConfig(function=PinFunction.STATUS_LED, port=GPIOPort.PORT0, pin=17)
        assert p.port == GPIOPort.PORT0

    def test_invalid_pin(self):
        with pytest.raises(ValueError, match="out of range"):
            PinConfig(function=PinFunction.STATUS_LED, port=GPIOPort.PORT0, pin=32)

    def test_absolute_pin(self):
        p = PinConfig(function=PinFunction.STATUS_LED, port=GPIOPort.PORT0, pin=5)
        assert p.absolute_pin == 5

    def test_hardware_address(self):
        p = PinConfig(function=PinFunction.STATUS_LED, port=GPIOPort.PORT0, pin=0)
        assert p.hardware_address() == 0x40000500


class TestProtocolConfig:
    def test_default_valid(self):
        p = ProtocolConfig()
        assert p.is_valid()

    def test_max_conn_8(self):
        p = ProtocolConfig(ble_max_conn=8)
        assert p.is_valid()

    def test_max_conn_exceeds(self):
        p = ProtocolConfig(ble_max_conn=20)
        errors = p.validate()
        assert any("1-8" in e for e in errors)

    def test_mtu_max_247(self):
        p = ProtocolConfig(ble_mtu=248)
        errors = p.validate()
        assert any("MTU" in e for e in errors)

    def test_tx_power_max_4(self):
        p = ProtocolConfig(ble_tx_power_dbm=8)
        errors = p.validate()
        assert any("TX power" in e for e in errors)


class TestNRF52832Config:
    def test_default_pins(self, config):
        assert len(config.all_pins()) >= 7

    def test_i2c_pins(self, config):
        assert config.PIN_I2C_SDA.port == GPIOPort.PORT0
        assert config.PIN_I2C_SDA.pin == 26

    def test_uart_pins(self, config):
        assert config.PIN_UART_TX.port == GPIOPort.PORT0
        assert config.PIN_UART_RX.port == GPIOPort.PORT0

    def test_status_led(self, config):
        assert config.PIN_STATUS_LED.pin == 17

    def test_remap_pin(self, config):
        config.remap_pin(PinFunction.STATUS_LED, 0, 5)
        assert config.PIN_STATUS_LED.pin == 5

    def test_configure_ble(self, config):
        config.configure_ble(max_conn=4, tx_power=2)
        assert config.protocol.ble_max_conn == 4

    def test_marine_memory(self, config):
        assert config.memory.total_ram() > 0
        assert config.memory.fits_in_ram()
        assert config.memory.fits_in_flash()

    def test_softdevice_flash(self, config):
        sd = config.memory.region_by_name("softdevice")
        assert sd is not None
        assert sd.type == "flash"

    def test_peripheral_allocate(self, config):
        key = config.allocate_peripheral("spi", "SPIM", 0)
        assert key == "SPIM0"

    def test_peripheral_double_raises(self, config):
        config.allocate_peripheral("a", "TWIM", 0)
        with pytest.raises(ValueError, match="already allocated"):
            config.allocate_peripheral("b", "TWIM", 0)

    def test_validate_clean(self, config):
        assert config.is_valid()

    def test_validate_overflow(self, config):
        from hardware.nrf52.config_nrf52832 import RAM_BASE
        config.memory.add_region("overflow", RAM_BASE, RAM_TOTAL + 1, "ram")
        assert not config.is_valid()

    def test_summary(self, config):
        s = config.summary()
        assert s["chip"] == "nRF52832"
        assert s["cpu"] == "Cortex-M4F"
        assert s["flash_bytes"] == 512 * 1024
