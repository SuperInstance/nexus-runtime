"""Tests for NEXUS nRF5340 dual-core hardware configuration."""

import pytest
from hardware.nrf52.config_nrf5340 import (
    NRF5340Config,
    PinConfig,
    PinFunction,
    GPIOPort,
    CoreID,
    MemoryLayout,
    MemoryRegion,
    ProtocolConfig,
    CORE_COUNT,
    APP_CORE_COUNT,
    APP_CPU_FREQ_MAX,
    APP_FLASH_TOTAL,
    APP_RAM_TOTAL,
    NET_CORE_COUNT,
    NET_CPU_FREQ_MAX,
    NET_FLASH_TOTAL,
    NET_RAM_TOTAL,
    GPIO_COUNT,
    SPI_COUNT,
    I2C_COUNT,
    UART_COUNT,
    RADIO_BLE_5,
    NFC_AVAILABLE,
    USB_AVAILABLE,
    CODE_PAGE_SIZE,
    APP_CODE_PAGES,
    NET_CODE_PAGES,
)


@pytest.fixture
def config():
    cfg = NRF5340Config()
    cfg.configure_marine_node()
    return cfg


class TestChipConstants:
    def test_total_cores(self):
        assert CORE_COUNT == 2

    def test_app_core_freq(self):
        assert APP_CPU_FREQ_MAX == 128_000_000

    def test_net_core_freq(self):
        assert NET_CPU_FREQ_MAX == 64_000_000

    def test_app_flash(self):
        assert APP_FLASH_TOTAL == 1_048_576

    def test_app_ram(self):
        assert APP_RAM_TOTAL == 512 * 1024

    def test_net_flash(self):
        assert NET_FLASH_TOTAL == 256 * 1024

    def test_net_ram(self):
        assert NET_RAM_TOTAL == 64 * 1024

    def test_gpio_count(self):
        assert GPIO_COUNT == 48

    def test_spi_count(self):
        assert SPI_COUNT == 4

    def test_i2c_count(self):
        assert I2C_COUNT == 4

    def test_uart_count(self):
        assert UART_COUNT == 3

    def test_ble5(self):
        assert RADIO_BLE_5 is True

    def test_nfc(self):
        assert NFC_AVAILABLE is True

    def test_usb(self):
        assert USB_AVAILABLE is True

    def test_code_pages(self):
        assert APP_CODE_PAGES == 256
        assert NET_CODE_PAGES == 64


class TestPinConfig:
    def test_valid_port0(self):
        p = PinConfig(function=PinFunction.STATUS_LED, port=GPIOPort.PORT0, pin=17)
        assert p.absolute_pin == 17

    def test_valid_port1(self):
        p = PinConfig(function=PinFunction.GPS_PPS, port=GPIOPort.PORT1, pin=0)
        assert p.absolute_pin == 32

    def test_invalid_port1(self):
        with pytest.raises(ValueError, match="out of range"):
            PinConfig(function=PinFunction.GPS_PPS, port=GPIOPort.PORT1, pin=16)

    def test_core_assignment(self):
        p = PinConfig(function=PinFunction.STATUS_LED, port=GPIOPort.PORT0,
                       pin=17, core=CoreID.APPLICATION)
        assert p.core == CoreID.APPLICATION


class TestProtocolConfig:
    def test_default_ble53(self):
        p = ProtocolConfig()
        assert p.ble_version == "5.3"
        assert p.is_valid()

    def test_mtu_517(self):
        p = ProtocolConfig(ble_mtu=517)
        assert p.is_valid()

    def test_invalid_mtu(self):
        p = ProtocolConfig(ble_mtu=518)
        errors = p.validate()
        assert any("MTU" in e for e in errors)

    def test_le_audio_flag(self):
        p = ProtocolConfig(le_audio=True)
        assert p.le_audio is True

    def test_direction_finding_flag(self):
        p = ProtocolConfig(direction_finding=True)
        assert p.direction_finding is True


class TestNRF5340Config:
    def test_default_pins(self, config):
        assert len(config.all_pins()) >= 15

    def test_i2c_pins(self, config):
        assert config.PIN_I2C_SDA.port == GPIOPort.PORT0
        assert config.PIN_I2C_SDA.pin == 26

    def test_status_led(self, config):
        assert config.PIN_STATUS_LED.port == GPIOPort.PORT0
        assert config.PIN_STATUS_LED.pin == 17

    def test_remap_pin(self, config):
        config.remap_pin(PinFunction.STATUS_LED, 1, 5)
        assert config.PIN_STATUS_LED.port == GPIOPort.PORT1

    def test_configure_ble(self, config):
        config.configure_ble(max_conn=15, mtu=517)
        assert config.protocol.ble_max_conn == 15
        assert config.protocol.ble_mtu == 517

    def test_configure_thread(self, config):
        config.configure_thread()
        assert config.protocol.thread_enabled is True
        assert config.protocol.radio_mode == "thread"

    def test_configure_audio(self, config):
        config.configure_audio()
        assert config.protocol.le_audio is True

    def test_configure_direction_finding(self, config):
        config.configure_direction_finding()
        assert config.protocol.direction_finding is True

    def test_marine_memory(self, config):
        assert config.memory.total_ram() > 0
        assert config.memory.fits_in_ram()
        assert config.memory.fits_in_flash()

    def test_dual_core_memory(self, config):
        app_stack = config.memory.region_by_name("app_stack")
        net_ble = config.memory.region_by_name("net_ble")
        assert app_stack is not None
        assert net_ble is not None

    def test_validate_clean(self, config):
        assert config.is_valid()

    def test_peripheral_allocate(self, config):
        key = config.allocate_peripheral("spi", "SPIM", 0)
        assert key == "SPIM0"

    def test_summary(self, config):
        s = config.summary()
        assert s["chip"] == "nRF5340"
        assert "Cortex-M33" in s["app_cpu"]
        assert "Cortex-M33" in s["net_cpu"]
        assert s["ble_version"] == "5.3"
        assert s["app_flash_bytes"] == 1_048_576
        assert s["net_ram_bytes"] == 64 * 1024
