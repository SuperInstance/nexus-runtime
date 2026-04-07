"""
Tests for NEXUS CAN Bus Configuration Module.

Covers: CANConfig, CANNodeConfig, PGN definitions, CAN ID construction,
        filter config, validation, and serialisation.
"""

import pytest
from hardware.stm32.can_bus import (
    CANBaudRate, CANMode, CANFilterMode, CANFrameType,
    PGNPriority, TransmitRate,
    PGNDefinition,
    PGN_HEADING, PGN_RATE_OF_TURN, PGN_SPEED_WATER, PGN_SPEED_GROUND,
    PGN_WATER_DEPTH, PGN_ENVIRONMENTAL, PGN_WIND,
    PGN_BATTERY, PGN_ENGINE_RPM,
    PGN_THRUSTER_COMMAND, PGN_THRUSTER_STATUS, PGN_SENSOR_DATA, PGN_NAV_ATTITUDE,
    PGN_REGISTRY,
    CANFilterConfig,
    CANConfig,
    CANNodeConfig,
)


class TestCANBaudRate:
    def test_nmea_2000(self):
        assert CANBaudRate.NMEA_2000 == 250_000

    def test_high_speed(self):
        assert CANBaudRate.HIGH_SPEED == 500_000

    def test_very_high_speed(self):
        assert CANBaudRate.VERY_HIGH_SPEED == 1_000_000


class TestPGNDefinition:
    def test_single_frame(self):
        pgn = PGN_HEADING
        assert pgn.is_single_frame is True
        assert pgn.required_frames == 1

    def test_fast_packet(self):
        pgn = PGN_ENVIRONMENTAL  # 28 bytes
        assert pgn.is_fast_packet is True
        assert pgn.required_frames > 1

    def test_can_id_construction(self):
        can_id = PGN_HEADING.can_id(source_addr=42)
        # Priority 2, EDP=0, DP=0, PGN=127250, SA=42
        assert can_id & 0xFF == 42
        assert (can_id >> 26) & 0x07 == 2

    def test_can_id_with_custom_priority(self):
        can_id = PGN_HEADING.can_id(source_addr=10, priority=PGNPriority.EMERGENCY)
        assert (can_id >> 26) & 0x07 == 0

    def test_thruster_command_fields(self):
        pgn = PGN_THRUSTER_COMMAND
        assert pgn.source == "NEXUS"
        assert len(pgn.fields) == 5

    def test_required_frames_for_64_bytes(self):
        pgn = PGN_SENSOR_DATA  # 64 bytes
        assert pgn.required_frames == 1 + ((64 - 6) + 6) // 7


class TestPGNRegistry:
    def test_registry_populated(self):
        assert len(PGN_REGISTRY) >= 13

    def test_all_nexus_pgns_in_registry(self):
        assert PGN_THRUSTER_COMMAND.pgn in PGN_REGISTRY
        assert PGN_THRUSTER_STATUS.pgn in PGN_REGISTRY
        assert PGN_SENSOR_DATA.pgn in PGN_REGISTRY
        assert PGN_NAV_ATTITUDE.pgn in PGN_REGISTRY

    def test_standard_marine_pgns(self):
        assert 127250 in PGN_REGISTRY  # Heading
        assert 128267 in PGN_REGISTRY  # Water Depth
        assert 127506 in PGN_REGISTRY  # Battery


class TestCANConfig:
    def test_default_nmea_2000(self):
        cfg = CANConfig()
        assert cfg.baud_rate == 250_000
        assert cfg.nominal_baud == 250_000

    def test_sample_point(self):
        cfg = CANConfig(prescaler=14, time_seg1=9, time_seg2=2)
        assert cfg.sample_point_pct == pytest.approx(83.33, rel=1e-3)

    def test_bit_quanta(self):
        cfg = CANConfig(prescaler=14, time_seg1=9, time_seg2=2)
        assert cfg.bit_quanta_total == 12

    def test_validate_ok(self):
        cfg = CANConfig()
        assert cfg.validate() == []

    def test_validate_bad_node_id(self):
        cfg = CANConfig(node_id=255)
        errors = cfg.validate()
        assert any("Node ID" in e for e in errors)

    def test_validate_bad_prescaler(self):
        cfg = CANConfig(prescaler=0)
        errors = cfg.validate()
        assert any("Prescaler" in e for e in errors)

    def test_clone(self):
        cfg = CANConfig(node_id=42)
        clone = cfg.clone()
        clone.node_id = 99
        assert cfg.node_id == 42

    def test_to_dict(self):
        cfg = CANConfig(instance="CAN2")
        d = cfg.to_dict()
        assert d["instance"] == "CAN2"


class TestCANFilterConfig:
    def test_mask_mode(self):
        f = CANFilterConfig(filter_id=0, mode=CANFilterMode.MASK, mask=0x1FFFFFFF)
        assert f.mask == 0x1FFFFFFF

    def test_list_mode(self):
        f = CANFilterConfig(filter_id=1, mode=CANFilterMode.LIST,
                            identifiers=[0x18FEF200, 0x18FEF201])
        assert len(f.identifiers) == 2


class TestCANNodeConfig:
    def test_default_supported_pgns(self):
        node = CANNodeConfig(node_id=42)
        assert len(node.supported_pgns) == len(PGN_REGISTRY)

    def test_build_can_id(self):
        node = CANNodeConfig(node_id=10)
        can_id = node.build_can_id(PGN_HEADING.pgn)
        assert can_id is not None
        assert can_id & 0xFF == 10

    def test_build_unknown_pgn(self):
        node = CANNodeConfig(node_id=10)
        can_id = node.build_can_id(999999)
        assert can_id is None

    def test_get_pgn(self):
        node = CANNodeConfig()
        pgn = node.get_pgn(PGN_HEADING.pgn)
        assert pgn is not None
        assert pgn.name == "Vessel Heading"

    def test_validate_ok(self):
        node = CANNodeConfig(node_id=42, transmit_pgns=[PGN_THRUSTER_COMMAND.pgn])
        assert node.validate() == []

    def test_validate_unknown_pgn(self):
        node = CANNodeConfig(node_id=42, transmit_pgns=[999999])
        errors = node.validate()
        assert any("999999" in e for e in errors)

    def test_to_dict(self):
        node = CANNodeConfig(name="test_node")
        d = node.to_dict()
        assert d["name"] == "test_node"
