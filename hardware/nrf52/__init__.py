"""
NEXUS nRF52 Hardware Library

Marine robotics wireless sensor configuration for the nRF52840 platform.
Includes chip config, BLE GATT profiles, and NEXUS marine sensor service definitions.
"""

from .config_nrf52840 import (
    NRF52840Config,
    PinConfig,
    PinFunction,
    GPIOPort,
    PinDrive,
    PinPull,
    MemoryLayout,
    MemoryRegion,
    ProtocolConfig,
    CORE_COUNT,
    CPU_FREQ_MAX,
    CPU_FREQ_DEFAULT,
    FLASH_TOTAL,
    RAM_TOTAL,
    GPIO_COUNT,
    ADC_RESOLUTION,
    ADC_CHANNELS,
    RTC_COUNT,
    TIMER_COUNT,
    PWM_COUNT,
    SPI_COUNT,
    I2C_COUNT,
    UART_COUNT,
    RADIO_BLE_5,
    RADIO_ZIGBEE,
    RADIO_THREAD,
    NFC_AVAILABLE,
    CODE_PAGE_SIZE,
    CODE_PAGES,
)

from .ble_profiles import (
    NEXUS_BASE_UUID,
    NEXUS_SERVICE_UUID,
    NEXUS_DEVICE_INFO_UUID,
    NEXUS_COMMAND_UUID,
    NEXUS_TELEMETRY_UUID,
    NEXUSCharacteristicUUID,
    BLE_UUID_BATTERY_SERVICE,
    BLE_UUID_BATTERY_LEVEL,
    BLE_UUID_DEVICE_NAME,
    BLE_UUID_MANUFACTURER,
    CharProperty,
    CharFormat,
    GATTError,
    CharacteristicDescriptor,
    CharacteristicMetadata,
    BLECharacteristic,
    BLEService,
    NEXUSMarineService,
    GATTServerConfig,
)

__all__ = [
    # Chip config
    "NRF52840Config", "PinConfig", "PinFunction", "GPIOPort",
    "PinDrive", "PinPull", "MemoryLayout", "MemoryRegion", "ProtocolConfig",
    # Constants
    "CORE_COUNT", "CPU_FREQ_MAX", "CPU_FREQ_DEFAULT",
    "FLASH_TOTAL", "RAM_TOTAL", "GPIO_COUNT",
    "ADC_RESOLUTION", "ADC_CHANNELS", "RTC_COUNT", "TIMER_COUNT",
    "PWM_COUNT", "SPI_COUNT", "I2C_COUNT", "UART_COUNT",
    "RADIO_BLE_5", "RADIO_ZIGBEE", "RADIO_THREAD", "NFC_AVAILABLE",
    "CODE_PAGE_SIZE", "CODE_PAGES",
    # BLE Profiles
    "NEXUS_BASE_UUID", "NEXUS_SERVICE_UUID", "NEXUS_DEVICE_INFO_UUID",
    "NEXUS_COMMAND_UUID", "NEXUS_TELEMETRY_UUID",
    "NEXUSCharacteristicUUID",
    "BLE_UUID_BATTERY_SERVICE", "BLE_UUID_BATTERY_LEVEL",
    "BLE_UUID_DEVICE_NAME", "BLE_UUID_MANUFACTURER",
    "CharProperty", "CharFormat", "GATTError",
    "CharacteristicDescriptor", "CharacteristicMetadata",
    "BLECharacteristic", "BLEService",
    "NEXUSMarineService", "GATTServerConfig",
]
