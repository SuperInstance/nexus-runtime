"""Tests for Teensy 4.0 configuration."""

import pytest
from hardware.teensy.config_teensy40 import (
    Teensy40Config, PinMapping, create_teensy40,
    CPU_FREQ_HZ, SRAM_BYTES, FLASH_MBYTES, NUM_DMA_CHANNELS, NUM_FLEXIO_SHIFTERS,
)


class TestTeensy40Constants:
    def test_cpu_freq(self):
        assert CPU_FREQ_HZ == 600_000_000

    def test_sram(self):
        assert SRAM_BYTES == 1_048_576

    def test_flash(self):
        assert FLASH_MBYTES == 4

    def test_dma(self):
        assert NUM_DMA_CHANNELS == 32

    def test_flexio(self):
        assert NUM_FLEXIO_SHIFTERS == 8


class TestPinMapping:
    def test_defaults(self):
        pins = PinMapping()
        assert pins.digital_pins == 40
        assert pins.analog_inputs == 14
        assert pins.pwm_pins == 28

    def test_frozen(self):
        with pytest.raises(AttributeError):
            PinMapping().digital_pins = 99

    def test_spi_pins(self):
        pins = PinMapping()
        assert pins.spi_mosi == 11
        assert pins.spi_miso == 12
        assert pins.spi_sck == 13

    def test_i2c_pins(self):
        pins = PinMapping()
        assert pins.i2c_sda == 18
        assert pins.i2c_scl == 19


class TestTeensy40Config:
    def test_defaults(self):
        cfg = Teensy40Config()
        assert cfg.cpu_freq_hz == 600_000_000
        assert cfg.sram_bytes == 1_048_576
        assert cfg.flash_mbytes == 4

    def test_validate_ok(self):
        assert Teensy40Config().validate() is True

    def test_validate_bad_cpu(self):
        assert Teensy40Config(cpu_freq_hz=-1).validate() is False

    def test_validate_bad_trust(self):
        assert Teensy40Config(trust_level=9).validate() is False

    def test_to_dict(self):
        d = Teensy40Config().to_dict()
        assert isinstance(d, dict)
        assert "cpu_freq_hz" in d
        assert d["flash_mbytes"] == 4

    def test_deploy_manifest(self):
        m = Teensy40Config().get_deploy_manifest()
        assert m["board"] == "teensy_40"
        assert m["mcu"] == "IMXRT1062DVL6A"
        assert m["architecture"] == "ARM Cortex-M7"
        assert m["cpu_freq_mhz"] == 600
        assert m["memory"]["sram_kb"] == 1024
        assert "interfaces" in m
        assert "nexus" in m

    def test_nexus_config(self):
        nc = Teensy40Config().get_nexus_config()
        assert nc["role"] == "edge_sensor"
        assert nc["trust_level"] == 2
        assert nc["wire_protocol"] == "COBS_CRC16"
        assert nc["vm_enabled"] is True

    def test_repr(self):
        r = repr(Teensy40Config())
        assert "600MHz" in r
        assert "1024KB" in r


class TestFactory:
    def test_default(self):
        cfg = create_teensy40()
        assert isinstance(cfg, Teensy40Config)
        assert cfg.cpu_freq_hz == 600_000_000

    def test_override_role(self):
        cfg = create_teensy40(nexus_role="relay", trust_level=4)
        assert cfg.nexus_role == "relay"
        assert cfg.trust_level == 4
        assert cfg.cpu_freq_hz == 600_000_000  # unchanged
