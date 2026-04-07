"""
Test suite for PRU (Programmable Real-time Unit) configuration and Cape Manager.
"""

import pytest
from dataclasses import fields, FrozenInstanceError

from hardware.beaglebone.pru_config import (
    PRUMode,
    PRUCoreConfig,
    PRUSharedMemory,
    MotorChannelConfig,
    PRUControllerConfig,
    create_pru_controller_config,
)
from hardware.beaglebone.cape_manager import (
    CapeSlot,
    CapeInfo,
    CapeManager,
    create_cape_manager,
)


# ═══════════════════════════════════════════════════════════════════════════
# PRUMode enum
# ═══════════════════════════════════════════════════════════════════════════

class TestPRUMode:
    def test_motor_control(self):
        assert PRUMode.MOTOR_CONTROL == "motor_control"

    def test_sensor_polling(self):
        assert PRUMode.SENSOR_POLLING == "sensor_polling"

    def test_pwm_generation(self):
        assert PRUMode.PWM_GENERATION == "pwm_generation"

    def test_idle(self):
        assert PRUMode.IDLE == "idle"

    def test_custom_firmware(self):
        assert PRUMode.CUSTOM_FIRMWARE == "custom_firmware"

    def test_count(self):
        assert len(PRUMode) == 5

    def test_is_str(self):
        assert isinstance(PRUMode.MOTOR_CONTROL, str)

    def test_iteration(self):
        modes = list(PRUMode)
        assert len(modes) == 5


# ═══════════════════════════════════════════════════════════════════════════
# PRUCoreConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestPRUCoreConfig:
    def test_default_core_id(self):
        assert PRUCoreConfig().core_id == 0

    def test_default_mode(self):
        assert PRUCoreConfig().mode == PRUMode.IDLE

    def test_default_clock(self):
        assert PRUCoreConfig().clock_hz == 200_000_000

    def test_default_enabled(self):
        assert PRUCoreConfig().enabled is True

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            PRUCoreConfig().core_id = 1

    def test_custom_mode(self):
        cfg = PRUCoreConfig(mode=PRUMode.MOTOR_CONTROL)
        assert cfg.mode == PRUMode.MOTOR_CONTROL

    def test_equality(self):
        assert PRUCoreConfig() == PRUCoreConfig()

    def test_field_count(self):
        assert len(fields(PRUCoreConfig)) == 5


# ═══════════════════════════════════════════════════════════════════════════
# PRUSharedMemory
# ═══════════════════════════════════════════════════════════════════════════

class TestPRUSharedMemory:
    def test_base_address(self):
        assert PRUSharedMemory().base_address == 0x4A300000

    def test_size_kb(self):
        assert PRUSharedMemory().size_kb == 12

    def test_ddr_offset(self):
        assert PRUSharedMemory().ddr_offset == 0x82000000

    def test_ddr_size_kb(self):
        assert PRUSharedMemory().ddr_size_kb == 256

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            PRUSharedMemory().size_kb = 24

    def test_equality(self):
        assert PRUSharedMemory() == PRUSharedMemory()

    def test_field_count(self):
        assert len(fields(PRUSharedMemory)) == 4


# ═══════════════════════════════════════════════════════════════════════════
# MotorChannelConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestMotorChannelConfig:
    def test_default_channel_id(self):
        assert MotorChannelConfig().channel_id == 0

    def test_default_pwm_frequency(self):
        assert MotorChannelConfig().pwm_frequency_hz == 50

    def test_default_min_pulse(self):
        assert MotorChannelConfig().min_pulse_us == 1000

    def test_default_max_pulse(self):
        assert MotorChannelConfig().max_pulse_us == 2000

    def test_default_dead_zone(self):
        assert MotorChannelConfig().dead_zone_percent == 5.0

    def test_default_feedback_enabled(self):
        assert MotorChannelConfig().feedback_enabled is False

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            MotorChannelConfig().channel_id = 5

    def test_custom_pwm_pin(self):
        cfg = MotorChannelConfig(pwm_pin="P8.45")
        assert cfg.pwm_pin == "P8.45"

    def test_field_count(self):
        assert len(fields(MotorChannelConfig)) == 9


# ═══════════════════════════════════════════════════════════════════════════
# PRUControllerConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestPRUControllerConfig:
    def test_default_pru_cores_count(self):
        assert len(PRUControllerConfig().pru_cores) == 2

    def test_default_core0_mode(self):
        assert PRUControllerConfig().pru_cores[0].mode == PRUMode.MOTOR_CONTROL

    def test_default_core1_mode(self):
        assert PRUControllerConfig().pru_cores[1].mode == PRUMode.SENSOR_POLLING

    def test_default_motor_channels_count(self):
        assert len(PRUControllerConfig().motor_channels) == 4

    def test_default_control_loop_hz(self):
        assert PRUControllerConfig().control_loop_hz == 200

    def test_default_watchdog(self):
        assert PRUControllerConfig().watchdog_timeout_ms == 100

    def test_safety_clamp_enabled(self):
        assert PRUControllerConfig().safety_clamp_enabled is True

    def test_emergency_stop_pin(self):
        assert PRUControllerConfig().emergency_stop_pin == "P8.12"

    def test_shared_memory_type(self):
        assert isinstance(PRUControllerConfig().shared_memory, PRUSharedMemory)


# ═══════════════════════════════════════════════════════════════════════════
# create_pru_controller_config
# ═══════════════════════════════════════════════════════════════════════════

class TestCreatePRUControllerConfig:
    def test_factory_returns(self):
        cfg = create_pru_controller_config()
        assert isinstance(cfg, PRUControllerConfig)

    def test_default_control_loop(self):
        assert create_pru_controller_config().control_loop_hz == 200

    def test_override_control_loop(self):
        cfg = create_pru_controller_config(control_loop_hz=1000)
        assert cfg.control_loop_hz == 1000

    def test_override_watchdog(self):
        cfg = create_pru_controller_config(watchdog_timeout_ms=50)
        assert cfg.watchdog_timeout_ms == 50

    def test_override_safety_clamp(self):
        cfg = create_pru_controller_config(safety_clamp_enabled=False)
        assert cfg.safety_clamp_enabled is False


# ═══════════════════════════════════════════════════════════════════════════
# CapeSlot enum
# ═══════════════════════════════════════════════════════════════════════════

class TestCapeSlot:
    def test_count(self):
        assert len(CapeSlot) == 4

    def test_slot_0(self):
        assert CapeSlot.SLOT_0 == "cape0"

    def test_is_str(self):
        assert isinstance(CapeSlot.SLOT_0, str)


# ═══════════════════════════════════════════════════════════════════════════
# CapeInfo
# ═══════════════════════════════════════════════════════════════════════════

class TestCapeInfo:
    def test_creation(self):
        cape = CapeInfo(
            name="Test Cape", version="1.0", manufacturer="Test Co"
        )
        assert cape.name == "Test Cape"
        assert cape.version == "1.0"
        assert cape.manufacturer == "Test Co"

    def test_frozen(self):
        cape = CapeInfo(
            name="Test Cape", version="1.0", manufacturer="Test Co"
        )
        with pytest.raises(FrozenInstanceError):
            cape.name = "Changed"

    def test_default_slot(self):
        cape = CapeInfo(
            name="Test Cape", version="1.0", manufacturer="Test Co"
        )
        assert cape.slot == CapeSlot.SLOT_0


# ═══════════════════════════════════════════════════════════════════════════
# CapeManager
# ═══════════════════════════════════════════════════════════════════════════

class TestCapeManager:
    def test_create(self):
        mgr = create_cape_manager()
        assert isinstance(mgr, CapeManager)

    def test_list_known_capes(self):
        mgr = create_cape_manager()
        capes = mgr.list_known_capes()
        assert len(capes) >= 3
        assert "nexus-motor-controller" in capes

    def test_load_known_cape(self):
        mgr = create_cape_manager()
        assert mgr.load_cape("nexus-motor-controller") is True

    def test_load_unknown_cape(self):
        mgr = create_cape_manager()
        assert mgr.load_cape("nonexistent-cape") is False

    def test_loaded_count_empty(self):
        mgr = create_cape_manager()
        assert mgr.loaded_count() == 0

    def test_loaded_count_after_load(self):
        mgr = create_cape_manager()
        mgr.load_cape("nexus-motor-controller")
        assert mgr.loaded_count() == 1

    def test_get_cape_at_slot(self):
        mgr = create_cape_manager()
        assert mgr.get_cape_at_slot(CapeSlot.SLOT_0) is None
        mgr.load_cape("nexus-motor-controller", CapeSlot.SLOT_1)
        cape = mgr.get_cape_at_slot(CapeSlot.SLOT_1)
        assert cape is not None
        assert cape.name == "NEXUS Motor Controller"

    def test_unload_cape(self):
        mgr = create_cape_manager()
        mgr.load_cape("nexus-motor-controller")
        assert mgr.unload_cape(CapeSlot.SLOT_0) is True
        assert mgr.loaded_count() == 0

    def test_unload_empty_slot(self):
        mgr = create_cape_manager()
        assert mgr.unload_cape(CapeSlot.SLOT_0) is False

    def test_detect_capes_empty(self):
        mgr = create_cape_manager()
        capes = mgr.detect_capes()
        assert len(capes) == 0

    def test_detect_capes_after_load(self):
        mgr = create_cape_manager()
        mgr.load_cape("nexus-motor-controller")
        mgr.load_cape("nexus-sensor-array", CapeSlot.SLOT_1)
        capes = mgr.detect_capes()
        assert len(capes) == 2

    def test_available_slots(self):
        mgr = create_cape_manager()
        slots = mgr.available_slots()
        assert len(slots) == 4

    def test_available_slots_after_load(self):
        mgr = create_cape_manager()
        mgr.load_cape("nexus-motor-controller")
        slots = mgr.available_slots()
        assert len(slots) == 3

    def test_get_known_cape(self):
        mgr = create_cape_manager()
        cape = mgr.get_known_cape("nexus-motor-controller")
        assert cape is not None
        assert cape.part_number == "NXS-MC-001"

    def test_get_unknown_known_cape(self):
        mgr = create_cape_manager()
        assert mgr.get_known_cape("nonexistent") is None

    def test_load_multiple_slots(self):
        mgr = create_cape_manager()
        assert mgr.load_cape("nexus-motor-controller", CapeSlot.SLOT_0)
        assert mgr.load_cape("nexus-sensor-array", CapeSlot.SLOT_1)
        assert mgr.load_cape("nexus-power-monitor", CapeSlot.SLOT_2)
        assert mgr.loaded_count() == 3

    def test_load_same_slot_overwrite(self):
        mgr = create_cape_manager()
        mgr.load_cape("nexus-motor-controller", CapeSlot.SLOT_0)
        mgr.load_cape("nexus-sensor-array", CapeSlot.SLOT_0)
        assert mgr.loaded_count() == 1
        assert mgr.get_cape_at_slot(CapeSlot.SLOT_0).name == "NEXUS Sensor Array"
