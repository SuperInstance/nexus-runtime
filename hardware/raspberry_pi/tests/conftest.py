"""Conftest for NEXUS Raspberry Pi hardware tests."""

import pytest


@pytest.fixture
def pi4_config():
    """Provide a Pi4Config instance for tests."""
    from nexus.hardware.raspberry_pi.config_pi4 import Pi4Config
    return Pi4Config()


@pytest.fixture
def pi5_config():
    """Provide a Pi5Config instance for tests."""
    from nexus.hardware.raspberry_pi.config_pi5 import Pi5Config
    return Pi5Config()


@pytest.fixture
def pizerow_config():
    """Provide a PiZero2WConfig instance for tests."""
    from nexus.hardware.raspberry_pi.config_pizerow import PiZero2WConfig
    return PiZero2WConfig()


@pytest.fixture
def sensor_hat():
    """Provide a MarineSensorHAT instance for tests."""
    from nexus.hardware.raspberry_pi.sensor_hat import MarineSensorHAT
    return MarineSensorHAT(board="pi4b", revision="v2")


@pytest.fixture
def sensor_hat_v1():
    """Provide a v1 MarineSensorHAT instance."""
    from nexus.hardware.raspberry_pi.sensor_hat import MarineSensorHAT
    return MarineSensorHAT(board="pi4b", revision="v1")
