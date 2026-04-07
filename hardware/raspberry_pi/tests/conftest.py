"""Conftest for NEXUS Raspberry Pi hardware tests."""

import pytest


@pytest.fixture
def pi4_config():
    """Provide a Pi4Config instance for tests."""
    from hardware.raspberry_pi.config_pi4 import Pi4Config
    return Pi4Config()


@pytest.fixture
def pi5_config():
    """Provide a Pi5Config instance for tests."""
    from hardware.raspberry_pi.config_pi5 import Pi5Config
    return Pi5Config()


@pytest.fixture
def pizerow_config():
    """Provide a PiZero2WConfig instance for tests."""
    from hardware.raspberry_pi.config_pizerow import PiZero2WConfig
    return PiZero2WConfig()


@pytest.fixture
def sensor_hat():
    """Provide a MarineSensorHAT instance for tests."""
    from hardware.raspberry_pi.sensor_hat import MarineSensorHAT
    return MarineSensorHAT(board="pi4b", revision="v2")


@pytest.fixture
def sensor_hat_v1():
    """Provide a v1 MarineSensorHAT instance."""
    from hardware.raspberry_pi.sensor_hat import MarineSensorHAT
    return MarineSensorHAT(board="pi4b", revision="v1")


@pytest.fixture
def pi3b_config():
    """Provide a Pi3BConfig instance for tests."""
    from hardware.raspberry_pi.config_pi3b import Pi3BConfig
    return Pi3BConfig()


@pytest.fixture
def pi400_config():
    """Provide a Pi400Config instance for tests."""
    from hardware.raspberry_pi.config_pi400 import Pi400Config
    return Pi400Config()


@pytest.fixture
def cm4_config():
    """Provide a CM4Config instance for tests."""
    from hardware.raspberry_pi.config_cm4 import CM4Config
    return CM4Config()


@pytest.fixture
def pico2_config():
    """Provide a Pico2Config instance for tests."""
    from hardware.raspberry_pi.config_pico2 import Pico2Config
    return Pico2Config()
