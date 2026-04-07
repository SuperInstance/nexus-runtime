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
from hardware.arduino.config_nano import (
    BoardConfig as NanoBoardConfig,
    SerialConfig as NanoSerialConfig,
    WireProtocolConfig as NanoWireProtocolConfig,
    PinMapping as NanoPinMapping,
    InterfacePins as NanoInterfacePins,
    NanoConfig,
    get_nano_config,
)
from hardware.arduino.config_due import (
    BoardConfig as DueBoardConfig,
    SerialConfig as DueSerialConfig,
    WireProtocolConfig as DueWireProtocolConfig,
    NexusEdgeConfig as DueNexusEdgeConfig,
    PinMapping as DuePinMapping,
    InterfacePins as DueInterfacePins,
    DueConfig,
    get_due_config,
)
from hardware.arduino.config_mkr_wifi import (
    BoardConfig as MKRBoardConfig,
    SerialConfig as MKRSerialConfig,
    WireProtocolConfig as MKRWireProtocolConfig,
    WiFiConfig as MKRWiFiNetworkConfig,
    NexusBridgeConfig as MKRNexusBridgeConfig,
    PinMapping as MKRPinMapping,
    InterfacePins as MKRInterfacePins,
    MKRWiFiConfig,
    get_mkr_wifi_config,
)
from hardware.arduino.config_nano33_iot import (
    BoardConfig as Nano33IoTBoardConfig,
    SerialConfig as Nano33IoTSerialConfig,
    WireProtocolConfig as Nano33IoTWireProtocolConfig,
    WiFiConfig as Nano33IoTWiFiConfig,
    NexusMeshConfig as Nano33IoTNexusMeshConfig,
    PinMapping as Nano33IoTPinMapping,
    InterfacePins as Nano33IoTInterfacePins,
    Nano33IoTConfig,
    get_nano33_iot_config,
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
    # Nano
    "NanoBoardConfig",
    "NanoSerialConfig",
    "NanoWireProtocolConfig",
    "NanoPinMapping",
    "NanoInterfacePins",
    "NanoConfig",
    "get_nano_config",
    # Due
    "DueBoardConfig",
    "DueSerialConfig",
    "DueWireProtocolConfig",
    "DueNexusEdgeConfig",
    "DuePinMapping",
    "DueInterfacePins",
    "DueConfig",
    "get_due_config",
    # MKR WiFi 1010
    "MKRBoardConfig",
    "MKRSerialConfig",
    "MKRWireProtocolConfig",
    "MKRWiFiNetworkConfig",
    "MKRNexusBridgeConfig",
    "MKRPinMapping",
    "MKRInterfacePins",
    "MKRWiFiConfig",
    "get_mkr_wifi_config",
    # Nano 33 IoT
    "Nano33IoTBoardConfig",
    "Nano33IoTSerialConfig",
    "Nano33IoTWireProtocolConfig",
    "Nano33IoTWiFiConfig",
    "Nano33IoTNexusMeshConfig",
    "Nano33IoTPinMapping",
    "Nano33IoTInterfacePins",
    "Nano33IoTConfig",
    "get_nano33_iot_config",
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
    "nano": ("Arduino Nano", get_nano_config),
    "due": ("Arduino Due", get_due_config),
    "mkr_wifi": ("Arduino MKR WiFi 1010", get_mkr_wifi_config),
    "nano33_iot": ("Arduino Nano 33 IoT", get_nano33_iot_config),
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
        One of ``"uno"``, ``"mega"``, ``"nano"``, ``"due"``,
        ``"mkr_wifi"``, ``"nano33_iot"``, or ``"leonardo"``.
    **overrides
        Keyword overrides forwarded to the board's config constructor.

    Returns
    -------
    UnoConfig | MegaConfig | NanoConfig | DueConfig | MKRWiFiConfig | Nano33IoTConfig

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
