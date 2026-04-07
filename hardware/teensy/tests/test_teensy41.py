"""Tests for Teensy 4.1 configuration."""

import pytest
from hardware.teensy.config_teensy41 import (
    Teensy41Config, create_teensy41,
    CPU_FREQ_HZ, SRAM_BYTES, PSRAM_BYTES, FLASH_MBYTES,
    ETH_MDIO_PIN, SD_CS_PIN,
)


class TestConstants:
    def test_cpu_freq(self):
        assert CPU_FREQ_HZ == 600_000_000
    def test_sram(self):
        assert SRAM_BYTES == 1_048_576
    def test_psram(self):
        assert PSRAM_BYTES == 8_388_608
    def test_eth_mdio(self):
        assert ETH_MDIO_PIN == 17
    def test_sd_cs(self):
        assert SD_CS_PIN == 10


class TestTeensy41Config:
    def test_defaults(self):
        cfg = Teensy41Config()
        assert cfg.cpu_freq_hz == 600_000_000
        assert cfg.psram_bytes == 8_388_608
        assert cfg.ethernet_enabled is True
        assert cfg.sd_card_enabled is True
        assert cfg.digital_pins == 55

    def test_validate_ok(self):
        assert Teensy41Config().validate() is True

    def test_validate_bad_cpu(self):
        assert Teensy41Config(cpu_freq_hz=0).validate() is False

    def test_to_dict(self):
        d = Teensy41Config().to_dict()
        assert d["psram_bytes"] == 8_388_608
        assert d["ethernet_enabled"] is True

    def test_deploy_manifest(self):
        m = Teensy41Config().get_deploy_manifest()
        assert m["board"] == "teensy_41"
        assert m["memory"]["psram_mb"] == 8
        assert m["interfaces"]["ethernet"]["enabled"] is True
        assert m["interfaces"]["ethernet"]["phy"] == "RMII"
        assert m["interfaces"]["sd_card"]["cs_pin"] == 10

    def test_nexus_config(self):
        nc = Teensy41Config().get_nexus_config()
        assert nc["role"] == "network_relay"
        assert nc["trust_level"] == 3
        assert nc["network"]["ethernet_enabled"] is True
        assert nc["storage"]["psram_buffer_mb"] == 8

    def test_repr(self):
        r = repr(Teensy41Config())
        assert "600MHz" in r
        assert "eth=True" in r


class TestFactory:
    def test_default(self):
        cfg = create_teensy41()
        assert isinstance(cfg, Teensy41Config)

    def test_override(self):
        cfg = create_teensy41(ethernet_enabled=False, trust_level=5)
        assert cfg.ethernet_enabled is False
        assert cfg.trust_level == 5
