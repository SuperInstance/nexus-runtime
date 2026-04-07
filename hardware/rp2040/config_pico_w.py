"""
NEXUS Pico W Configuration Module

Extends the base RP2040 configuration with WiFi (CYW43439) and BLE
support for wireless telemetry in the NEXUS marine robotics platform.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import IntEnum

from .config_rp2040 import (
    RP2040Config,
    CPU_FREQ_DEFAULT,
    SRAM_TOTAL,
    GPIOPin,
    PinFunction,
    PinMapping,
    MemoryRegion,
)


class WiFiSecurity(IntEnum):
    OPEN = 0
    WPA = 1
    WPA2 = 2
    WPA3 = 3
    WPA2_WPA3 = 4


class BLERole(IntEnum):
    PERIPHERAL = 0
    CENTRAL = 1
    BROADCASTER = 2
    OBSERVER = 3


@dataclass
class WiFiConfig:
    """WiFi connection parameters for the CYW43439 wireless chip."""
    ssid: str = ""
    password: str = ""
    security: WiFiSecurity = WiFiSecurity.WPA2
    channel: int = 0            # 0 = auto
    hostname: str = "nexus-pico"
    static_ip: Optional[str] = None
    static_gateway: Optional[str] = None
    static_dns: Optional[str] = None
    timeout_ms: int = 15000
    reconnect: bool = True
    reconnect_delay_ms: int = 5000
    power_save: bool = False

    def is_complete(self) -> bool:
        """Check if enough config is present to attempt a connection."""
        if self.security == WiFiSecurity.OPEN:
            return len(self.ssid) > 0
        return len(self.ssid) > 0 and len(self.password) > 0

    def to_dict(self) -> dict:
        d = {"ssid": self.ssid, "security": self.security.name, "channel": self.channel}
        if self.password:
            d["password"] = self.password
        if self.static_ip:
            d["static_ip"] = self.static_ip
        if self.static_gateway:
            d["static_gateway"] = self.static_gateway
        return d


@dataclass
class BLEConfig:
    """BLE radio configuration."""
    device_name: str = "NEXUS-PicoW"
    role: BLERole = BLERole.PERIPHERAL
    advertising_interval_ms: int = 100
    min_connection_interval_ms: int = 20
    max_connection_interval_ms: int = 40
    tx_power_dbm: int = 6       # range: -20 to +8
    mtu: int = 23               # ATT MTU (default 23, up to 517)

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.device_name:
            errors.append("BLE device name cannot be empty.")
        if self.mtu < 23 or self.mtu > 517:
            errors.append(f"MTU must be between 23 and 517, got {self.mtu}.")
        if self.tx_power_dbm < -20 or self.tx_power_dbm > 8:
            errors.append(f"TX power must be -20..+8 dBm, got {self.tx_power_dbm}.")
        if self.advertising_interval_ms < 20 or self.advertising_interval_ms > 10240:
            errors.append("Advertising interval must be 20..10240 ms.")
        if self.min_connection_interval_ms > self.max_connection_interval_ms:
            errors.append("Min connection interval exceeds max.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


@dataclass
class MQTTConfig:
    """MQTT telemetry config for NEXUS data pipeline."""
    broker: str = ""
    port: int = 1883
    client_id: str = "nexus-pico"
    username: str = ""
    password: str = ""
    keepalive: int = 60
    tls: bool = False
    topic_prefix: str = "nexus/marine/"
    qos: int = 1                # 0=at_most_once, 1=at_least_once, 2=exactly_once

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.broker:
            errors.append("MQTT broker address is required.")
        if self.port < 1 or self.port > 65535:
            errors.append(f"Invalid port {self.port}.")
        if self.qos not in (0, 1, 2):
            errors.append(f"Invalid QoS {self.qos}, must be 0, 1, or 2.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


class PicoWConfig(RP2040Config):
    """
    NEXUS Pico W configuration extending the base RP2040 with
    CYW43439 wireless (WiFi + BLE) support.

    The CYW43439 uses an SDIO interface to the RP2040 and shares
    GPIO 23-25 with the onboard LED and wireless activity indicator.
    """

    # CYW43439 SDIO interface pins
    PIN_WL_SDIO_CMD = GPIOPin.GP23
    PIN_WL_SDIO_CLK = GPIOPin.GP24
    PIN_WL_SDIO_D0 = GPIOPin.GP24
    PIN_WL_ONBOARD_LED = GPIOPin.GP25
    PIN_WL_IRQ = GPIOPin.GP23

    # CYW43439 specs
    WIFI_BANDS = ["2.4 GHz"]
    WIFI_PROTOCOLS = ["802.11 b", "802.11 g", "802.11 n"]
    BLE_VERSION = "4.2"

    def __init__(self, clock_freq: int = CPU_FREQ_DEFAULT):
        super().__init__(clock_freq=clock_freq)
        self.wifi = WiFiConfig()
        self.ble = BLEConfig()
        self.mqtt = MQTTConfig()
        # Reserve additional SRAM for wireless buffers
        self._reserve_wireless_memory()

    def _reserve_wireless_memory(self):
        """Reserve SRAM regions for WiFi and BLE stack buffers."""
        self.memory.add_region("lwip_heap", 0x2000A000, 32768, "lwIP network stack heap")
        self.memory.add_region("cyw43_buf", 0x20012000, 16384, "CYW43439 driver buffers")
        self.memory.add_region("ble_stack", 0x20016000, 8192, "BLE NimBLE stack")
        self.memory.add_region("mqtt_buf", 0x20018000, 4096, "MQTT publish/subscribe buffers")

    def configure_wifi(
        self,
        ssid: str,
        password: str = "",
        security: WiFiSecurity = WiFiSecurity.WPA2,
        hostname: str = "nexus-pico",
    ):
        self.wifi.ssid = ssid
        self.wifi.password = password
        self.wifi.security = security
        self.wifi.hostname = hostname

    def configure_ble(
        self,
        device_name: str = "NEXUS-PicoW",
        role: BLERole = BLERole.PERIPHERAL,
        tx_power_dbm: int = 6,
    ):
        self.ble.device_name = device_name
        self.ble.role = role
        self.ble.tx_power_dbm = tx_power_dbm

    def configure_mqtt(
        self,
        broker: str,
        port: int = 1883,
        client_id: str = "nexus-pico",
        topic_prefix: str = "nexus/marine/",
    ):
        self.mqtt.broker = broker
        self.mqtt.port = port
        self.mqtt.client_id = client_id
        self.mqtt.topic_prefix = topic_prefix

    def set_static_ip(self, ip: str, gateway: str = "", dns: str = ""):
        self.wifi.static_ip = ip
        self.wifi.static_gateway = gateway
        self.wifi.static_dns = dns

    def validate_wireless(self) -> List[str]:
        """Validate all wireless configuration and return errors."""
        errors: List[str] = []
        if self.wifi.ssid and not self.wifi.is_complete():
            errors.append("WiFi config incomplete: missing password for secured network.")
        ble_errors = self.ble.validate()
        errors.extend(ble_errors)
        if self.mqtt.broker:
            mqtt_errors = self.mqtt.validate()
            errors.extend(mqtt_errors)
        if not self.memory.fits_in_sram():
            errors.append(
                f"Memory overflow: {self.memory.total_allocated()} bytes "
                f"allocated but only {SRAM_TOTAL} bytes available."
            )
        return errors

    def summary(self) -> dict:
        base = super().summary()
        base.update({
            "variant": "Pico W",
            "wireless_chip": "CYW43439",
            "wifi_bands": self.WIFI_BANDS,
            "ble_version": self.BLE_VERSION,
            "wifi_ssid": self.wifi.ssid or "(not configured)",
            "ble_device_name": self.ble.device_name,
            "mqtt_broker": self.mqtt.broker or "(not configured)",
        })
        return base
