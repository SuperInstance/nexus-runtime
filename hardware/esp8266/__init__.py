"""NEXUS Marine Robotics Platform - ESP8266 Hardware Configuration Package."""

from .config_esp8266 import (
    ESP8266PinMap,
    SerialConfig,
    WireProtocolConfig,
    CommsConfig,
    ESP8266BoardConfig,
    create_esp8266_config,
)
from .config_d1_mini import (
    D1MiniPinMap,
    D1MiniBoardConfig,
    create_d1_mini_config,
)
from .wifi_manager import (
    WiFiConnectionState,
    WiFiCredentials,
    WiFiManagerConfig,
    WiFiManager,
)

__all__ = [
    "ESP8266PinMap",
    "SerialConfig",
    "WireProtocolConfig",
    "CommsConfig",
    "ESP8266BoardConfig",
    "create_esp8266_config",
    "D1MiniPinMap",
    "D1MiniBoardConfig",
    "create_d1_mini_config",
    "WiFiConnectionState",
    "WiFiCredentials",
    "WiFiManagerConfig",
    "WiFiManager",
]
