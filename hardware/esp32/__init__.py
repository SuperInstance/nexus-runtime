"""NEXUS Marine Robotics Platform - ESP32 Hardware Configuration Package."""

from .config_esp32 import (
    ESP32PinMap,
    SerialConfig,
    WireProtocolConfig,
    CommsConfig,
    ESP32BoardConfig,
    create_esp32_config,
)
from .config_esp32_s3 import (
    ESP32S3PinMap,
    ESP32S3BoardConfig,
    create_esp32_s3_config,
)
from .config_esp32_c3 import (
    ESP32C3PinMap,
    NexusNodeConfig as ESP32C3NexusNodeConfig,
    ESP32C3BoardConfig,
    create_esp32_c3_config,
)
from .config_esp32_c6 import (
    ESP32C6PinMap,
    ESP32C6BoardConfig,
    create_esp32_c6_config,
)
from .config_esp32_h2 import (
    ESP32H2PinMap,
    IoTGatewayConfig,
    ESP32H2BoardConfig,
    create_esp32_h2_config,
)
from .wifi_mesh import (
    MeshType,
    NodeRole,
    MeshConfig,
    MeshTopology,
    DataRateConfig,
    MeshNodeInfo,
)

__all__ = [
    "ESP32PinMap",
    "SerialConfig",
    "WireProtocolConfig",
    "CommsConfig",
    "ESP32BoardConfig",
    "create_esp32_config",
    "ESP32S3PinMap",
    "ESP32S3BoardConfig",
    "create_esp32_s3_config",
    "ESP32C3PinMap",
    "ESP32C3NexusNodeConfig",
    "ESP32C3BoardConfig",
    "create_esp32_c3_config",
    "ESP32C6PinMap",
    "ESP32C6BoardConfig",
    "create_esp32_c6_config",
    "ESP32H2PinMap",
    "IoTGatewayConfig",
    "ESP32H2BoardConfig",
    "create_esp32_h2_config",
    "MeshType",
    "NodeRole",
    "MeshConfig",
    "MeshTopology",
    "DataRateConfig",
    "MeshNodeInfo",
]
