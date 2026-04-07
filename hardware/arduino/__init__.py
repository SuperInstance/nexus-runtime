"""
NEXUS Arduino Hardware Configuration Package.

Provides board-level configuration, pin mappings, and sensor driver configs
for Arduino boards used in marine robotics (AUV / ROV / USV).

Quick start::

    from hardware.arduino import get_board_config, list_supported_boards

    boards = list_supported_boards()
    cfg = get_board_config("mega")
    print(cfg.pin_mapping.GPS_TX)
"""

from hardware.arduino.config_uno import (
    BoardConfig as UnoBoardConfig,
    SerialConfig as UnoSerialConfig,
    WireProtocolConfig as UnoWireProtocolConfig,
    PinMapping as UnoPinMapping,
    UnoConfig,
    get_uno_config,
)
from hardware.arduino.config_mega import (
    BoardConfig as MegaBoardConfig,
    SerialConfig as MegaSerialConfig,
    WireProtocolConfig as MegaWireProtocolConfig,
    PinMapping as MegaPinMapping,
    MegaConfig,
    get_mega_config,
)
from hardware.arduino.sensor_drivers import (
    GPSSensorConfig,
    IMUSensorConfig,
    SonarConfig,
    PressureSensorConfig,
    TemperatureSensorConfig,
    ServoConfig,
    MotorControllerConfig,
    Interface,
)

__all__ = [
    # Uno
    "UnoBoardConfig",
    "UnoSerialConfig",
    "UnoWireProtocolConfig",
    "UnoPinMapping",
    "UnoConfig",
    "get_uno_config",
    # Mega
    "MegaBoardConfig",
    "MegaSerialConfig",
    "MegaWireProtocolConfig",
    "MegaPinMapping",
    "MegaConfig",
    "get_mega_config",
    # Sensors
    "GPSSensorConfig",
    "IMUSensorConfig",
    "SonarConfig",
    "PressureSensorConfig",
    "TemperatureSensorConfig",
    "ServoConfig",
    "MotorControllerConfig",
    "Interface",
    # Convenience
    "get_board_config",
    "list_supported_boards",
]

# ---------------------------------------------------------------------------
# Board registry
# ---------------------------------------------------------------------------

_BOARD_REGISTRY = {
    "uno": ("Arduino Uno R3", get_uno_config),
    "mega": ("Arduino Mega 2560", get_mega_config),
    "nano": ("Arduino Nano", None),         # placeholder — same as Uno
    "due": ("Arduino Due", None),            # placeholder
    "leonardo": ("Arduino Leonardo", None),  # placeholder
}


def list_supported_boards() -> list:
    """Return a sorted list of supported board identifier strings."""
    return sorted(_BOARD_REGISTRY.keys())


def get_board_config(board_name: str, **overrides):
    """Factory: return the configuration object for *board_name*.

    Parameters
    ----------
    board_name : str
        One of ``"uno"``, ``"mega"``, ``"nano"``, ``"due"``, ``"leonardo"``.
    **overrides
        Keyword overrides forwarded to the board's config constructor.

    Returns
    -------
    UnoConfig | MegaConfig

    Raises
    ------
    ValueError
        If *board_name* is not recognised.
    NotImplementedError
        If the board's config module has not yet been implemented.
    """
    key = board_name.lower().strip()
    if key not in _BOARD_REGISTRY:
        raise ValueError(
            f"Unknown board '{board_name}'. "
            f"Supported: {list_supported_boards()}"
        )
    _, factory = _BOARD_REGISTRY[key]
    if factory is None:
        raise NotImplementedError(
            f"Board '{board_name}' is registered but its config module "
            f"is not yet implemented."
        )
    return factory(**overrides)
