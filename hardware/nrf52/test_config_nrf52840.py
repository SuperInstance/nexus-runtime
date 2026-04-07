"""
Tests for NEXUS nRF52840 Hardware Configuration.
"""

import pytest
from hardware.nrf52.config_nrf52840 import (
    NRF52840Config,
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
    RTC_COUNT,
    TIMER_COUNT,
    PWM_COUNT,
    SPI_COUNT,
    I2C_COUNT,
    UART_COUNT,
    RADIO_BLE_5,
    RADIO_ZIGBEE,
    RADIO_THREAD,
    NFC_AVAILABLE,
    CODE_PAGE_SIZE,
    CODE_PAGES,
)


class TestChipConstants:
    def test_core_count(self):
        assert CORE_COUNT == 1

    def test_cpu_freq_max(self):
        assert CPU_FREQ_MAX == 64_000_000

    def test_cpu_freq_default(self):
        assert CPU_FREQ_DEFAULT == 64_000_000

    def test_flash_total(self):
        assert FLASH_TOTAL == 1_048_576

    def test_ram_total(self):
        assert RAM_TOTAL == 256 * 1024

    def test_gpio_count(self):
        assert GPIO_COUNT == 48

    def test_adc(self):
        assert ADC_RESOLUTION == 12
        assert ADC_CHANNELS == 8

    def test_rtc(self):
        assert RTC_COUNT == 3

    def test_timer(self):
        assert TIMER_COUNT == 5

    def test_pwm(self):
        assert PWM_COUNT == 4

    def test_spi(self):
        assert SPI_COUNT == 4

    def test_i2c(self):
        assert I2C_COUNT == 4

    def test_uart(self):
        assert UART_COUNT == 2

    def test_radio_caps(self):
        assert RADIO_BLE_5 is True
        assert RADIO_ZIGBEE is True
        assert RADIO_THREAD is True
        assert NFC_AVAILABLE is True

    def test_code_pages(self):
        assert CODE_PAGE_SIZE == 4096
        assert CODE_PAGES == 256


class TestPinConfig:
    def test_valid_port0(self):
        p = PinConfig(function=PinFunction.STATUS_LED, port=GPIOPort.PORT0, pin=17)
        assert p.port == GPIOPort.PORT0
        assert p.pin == 17

    def test_valid_port1(self):
        p = PinConfig(function=PinFunction.GPS_PPS, port=GPIOPort.PORT1, pin=0)
        assert p.port == GPIOPort.PORT1

    def test_invalid_port0_pin(self):
        with pytest.raises(ValueError, match="out of range"):
            PinConfig(function=PinFunction.STATUS_LED, port=GPIOPort.PORT0, pin=32)

    def test_invalid_port1_pin(self):
        with pytest.raises(ValueError, match="out of range"):
            PinConfig(function=PinFunction.GPS_PPS, port=GPIOPort.PORT1, pin=16)

    def test_absolute_pin_port0(self):
        p = PinConfig(function=PinFunction.STATUS_LED, port=GPIOPort.PORT0, pin=17)
        assert p.absolute_pin == 17

    def test_absolute_pin_port1(self):
        p = PinConfig(function=PinFunction.GPS_PPS, port=GPIOPort.PORT1, pin=5)
        assert p.absolute_pin == 37

    def test_hardware_address(self):
        p = PinConfig(function=PinFunction.STATUS_LED, port=GPIOPort.PORT0, pin=0)
        assert p.hardware_address() == 0x40000500

    def test_pin_options(self):
        p = PinConfig(
            function=PinFunction.I2C_SDA,
            port=GPIOPort.PORT0,
            pin=26,
            pull=PinPull.PULLUP,
            drive=PinDrive.S0H1,
        )
        assert p.pull == PinPull.PULLUP
        assert p.drive == PinDrive.S0H1


class TestMemoryLayout:
    def test_add_and_total(self):
        ml = MemoryLayout()
        ml.add_region("buf1", 0x20000000, 4096)
        ml.add_region("buf2", 0x20001000, 8192, purpose="test")
        assert ml.total_allocated() == 12288

    def test_ram_specific(self):
        ml = MemoryLayout()
        ml.add_region("ram_buf", 0x20000000, 4096, "ram")
        ml.add_region("flash_app", 0x00005000, 8192, "flash")
        assert ml.total_ram() == 4096
        assert ml.total_flash() == 8192

    def test_fits_in_ram(self):
        ml = MemoryLayout()
        ml.add_region("small", 0x20000000, 1024, "ram")
        assert ml.fits_in_ram() is True

    def test_exceeds_ram(self):
        ml = MemoryLayout()
        ml.add_region("huge", 0x20000000, RAM_TOTAL + 1, "ram")
        assert ml.fits_in_ram() is False

    def test_region_by_name(self):
        ml = MemoryLayout()
        ml.add_region("test_region", 0x20000000, 512, purpose="test")
        r = ml.region_by_name("test_region")
        assert r is not None
        assert r.purpose == "test"

    def test_no_overlap(self):
        ml = MemoryLayout()
        ml.add_region("a", 0x20000000, 1024, "ram")
        ml.add_region("b", 0x20000400, 1024, "ram")
        assert ml.has_overlap() is False

    def test_overlap_detected(self):
        ml = MemoryLayout()
        ml.add_region("a", 0x20000000, 4096, "ram")
        ml.add_region("b", 0x20000F00, 1024, "ram")  # starts within a's range
        assert ml.has_overlap() is True

    def test_region_end(self):
        r = MemoryRegion("test", 0x20000000, 1024, "ram")
        assert r.end == 0x20000000 + 1024


class TestProtocolConfig:
    def test_default_ble(self):
        p = ProtocolConfig()
        assert p.ble_enabled is True
        assert p.is_valid() is True

    def test_invalid_max_conn(self):
        p = ProtocolConfig(ble_max_conn=0)
        errors = p.validate()
        assert any("max connections" in e.lower() for e in errors)

    def test_invalid_mtu(self):
        p = ProtocolConfig(ble_mtu=10)
        errors = p.validate()
        assert any("MTU" in e for e in errors)

    def test_invalid_tx_power(self):
        p = ProtocolConfig(ble_tx_power_dbm=10)
        errors = p.validate()
        assert any("TX power" in e for e in errors)

    def test_multiple_protocols_invalid(self):
        p = ProtocolConfig(ble_enabled=True, zigbee_enabled=True)
        errors = p.validate()
        assert any("one radio protocol" in e for e in errors)


class TestNRF52840Config:
    def setup_method(self):
        self.config = NRF52840Config()
        self.config.configure_marine_node()

    def test_default_pins(self):
        pins = self.config.all_pins()
        assert len(pins) >= 10

    def test_i2c_pins(self):
        assert self.config.PIN_I2C_SDA.port == GPIOPort.PORT0
        assert self.config.PIN_I2C_SDA.pin == 26
        assert self.config.PIN_I2C_SCL.port == GPIOPort.PORT0
        assert self.config.PIN_I2C_SCL.pin == 27

    def test_uart_pins(self):
        assert self.config.PIN_UART_TX.port == GPIOPort.PORT0
        assert self.config.PIN_UART_TX.pin == 20
        assert self.config.PIN_UART_RX.port == GPIOPort.PORT0
        assert self.config.PIN_UART_RX.pin == 19

    def test_status_led(self):
        assert self.config.PIN_STATUS_LED.port == GPIOPort.PORT0
        assert self.config.PIN_STATUS_LED.pin == 17

    def test_nfc_pin(self):
        assert self.config.PIN_NFC.port == GPIOPort.PORT0
        assert self.config.PIN_NFC.pin == 9

    def test_remap_pin(self):
        self.config.remap_pin(PinFunction.STATUS_LED, 1, 5)
        assert self.config.PIN_STATUS_LED.port == GPIOPort.PORT1
        assert self.config.PIN_STATUS_LED.pin == 5

    def test_get_missing_pin_raises(self):
        self.config._pin_map.clear()
        with pytest.raises(KeyError):
            self.config.get_pin(PinFunction.STATUS_LED)

    def test_configure_ble(self):
        self.config.configure_ble(max_conn=10, tx_power=4)
        assert self.config.protocol.ble_max_conn == 10
        assert self.config.protocol.ble_tx_power_dbm == 4

    def test_configure_zigbee(self):
        self.config.configure_zigbee()
        assert self.config.protocol.zigbee_enabled is True
        assert self.config.protocol.ble_enabled is False

    def test_configure_thread(self):
        self.config.configure_thread()
        assert self.config.protocol.thread_enabled is True
        assert self.config.protocol.radio_mode == "thread"

    def test_marine_memory(self):
        assert self.config.memory.total_ram() > 0
        assert self.config.memory.total_flash() > 0
        assert self.config.memory.fits_in_ram()
        assert self.config.memory.fits_in_flash()

    def test_softdevice_flash(self):
        sd = self.config.memory.region_by_name("softdevice")
        assert sd is not None
        assert sd.type == "flash"

    def test_peripheral_allocate(self):
        key = self.config.allocate_peripheral("spi_flash", "SPIM", 0)
        assert key == "SPIM0"
        assert "SPIM0" in self.config.allocated_peripherals()

    def test_peripheral_double_allocate_raises(self):
        self.config.allocate_peripheral("spi1", "SPIM", 1)
        with pytest.raises(ValueError, match="already allocated"):
            self.config.allocate_peripheral("spi2", "SPIM", 1)

    def test_peripheral_release(self):
        self.config.allocate_peripheral("tmp", "TWIM", 0)
        self.config.release_peripheral("TWIM0")
        assert "TWIM0" not in self.config.allocated_peripherals()

    def test_used_pins_set(self):
        used = self.config.used_pins_set()
        assert len(used) >= 10

    def test_validate_clean(self):
        assert self.config.is_valid() is True
        assert self.config.validate() == []

    def test_validate_memory_overflow(self):
        from hardware.nrf52.config_nrf52840 import RAM_BASE
        self.config.memory.add_region("overflow", RAM_BASE, RAM_TOTAL + 1, "ram")
        errors = self.config.validate()
        assert any("overflow" in e.lower() for e in errors)

    def test_summary_keys(self):
        s = self.config.summary()
        assert s["chip"] == "nRF52840"
        assert s["cpu"] == "Cortex-M4F"
        assert s["flash_bytes"] == 1_048_576
        assert s["ram_bytes"] == 256 * 1024
        assert "radio_mode" in s
        assert "pins_configured" in s
