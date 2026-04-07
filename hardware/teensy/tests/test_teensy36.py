"""Tests for Teensy 3.6 configuration."""

import pytest
from hardware.teensy.config_teensy36 import (
    Teensy36Config, create_teensy36,
    CPU_FREQ_HZ, SRAM_BYTES, FLASH_BYTES, NUM_DMA_CHANNELS,
)


class TestConstants:
    def test_cpu_freq(self):
        assert CPU_FREQ_HZ == 180_000_000
    def test_sram(self):
        assert SRAM_BYTES == 262_144
    def test_flash(self):
        assert FLASH_BYTES == 1_048_576
    def test_dma(self):
        assert NUM_DMA_CHANNELS == 16


class TestTeensy36Config:
    def test_defaults(self):
        cfg = Teensy36Config()
        assert cfg.cpu_freq_hz == 180_000_000
        assert cfg.sram_bytes == 256 * 1024
        assert cfg.flash_bytes == 1024 * 1024
        assert cfg.usb_high_speed is True
        assert cfg.dac_channels == 2
        assert cfg.digital_pins == 64

    def test_validate_ok(self):
        assert Teensy36Config().validate() is True

    def test_validate_bad(self):
        assert Teensy36Config(cpu_freq_hz=0).validate() is False

    def test_to_dict(self):
        d = Teensy36Config().to_dict()
        assert d["flash_bytes"] == 1_048_576
        assert d["usb_high_speed"] is True

    def test_deploy_manifest(self):
        m = Teensy36Config().get_deploy_manifest()
        assert m["board"] == "teensy_36"
        assert m["mcu"] == "MK66FX1M0VMD18"
        assert m["architecture"] == "ARM Cortex-M4F"
        assert m["cpu_freq_mhz"] == 180
        assert m["interfaces"]["usb_high_speed"] is True
        assert m["interfaces"]["dac_channels"] == 2

    def test_nexus_config(self):
        nc = Teensy36Config().get_nexus_config()
        assert nc["role"] == "edge_sensor"
        assert nc["capabilities"]["usb_high_speed"] is True
        assert nc["capabilities"]["fpu"] is True

    def test_repr(self):
        r = repr(Teensy36Config())
        assert "180MHz" in r


class TestFactory:
    def test_default(self):
        cfg = create_teensy36()
        assert isinstance(cfg, Teensy36Config)

    def test_override(self):
        cfg = create_teensy36(nexus_role="legacy_bridge")
        assert cfg.nexus_role == "legacy_bridge"
