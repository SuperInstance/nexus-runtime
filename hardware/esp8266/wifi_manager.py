"""NEXUS Marine Robotics Platform - ESP8266 WiFi Connection Manager.

Provides configuration types and a connection manager for ESP8266-based
nodes in the NEXUS fleet.  Handles automatic reconnection, AP fallback,
and MQTT broker connection lifecycle for marine sensor nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class WiFiConnectionState(str, Enum):
    """WiFi connection state machine states."""

    DISCONNECTED = "disconnected"
    SCANNING = "scanning"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    AP_MODE = "ap_mode"
    CONNECTION_FAILED = "connection_failed"
    AP_FALLBACK = "ap_fallback"


# ---------------------------------------------------------------------------
# WiFi credentials
# ---------------------------------------------------------------------------

@dataclass
class WiFiCredentials:
    """WiFi network access credentials."""

    ssid: str = ""
    password: str = ""
    bssid: str = ""               # Optional: lock to specific AP by MAC
    channel: int = 0              # 0 = auto-select

    def is_valid(self) -> bool:
        """Check if credentials contain at least an SSID."""
        return len(self.ssid) >= 1


# ---------------------------------------------------------------------------
# WiFi Manager configuration
# ---------------------------------------------------------------------------

@dataclass
class WiFiManagerConfig:
    """Configuration parameters for the WiFi connection manager.

    Attributes:
        auto_connect: Automatically connect on startup.
        auto_reconnect: Reconnect when connection is lost.
        max_reconnect_attempts: Maximum reconnection attempts before fallback.
        reconnect_interval_ms: Delay between reconnection attempts.
        connection_timeout_ms: Time to wait for a connection attempt.
        ap_fallback: Fall back to AP mode if STA connection fails.
        ap_ssid: SSID for AP fallback mode.
        ap_password: Password for AP fallback mode (empty = open).
        ap_channel: WiFi channel for AP fallback mode.
        static_ip_enabled: Use static IP instead of DHCP.
        static_ip: Static IP address.
        gateway: Default gateway.
        subnet: Subnet mask.
        dns_primary: Primary DNS server.
        dns_secondary: Secondary DNS server.
        hostname: mDNS hostname to advertise.
        power_save: Enable WiFi power-saving mode.
    """

    auto_connect: bool = True
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 10
    reconnect_interval_ms: int = 5000
    connection_timeout_ms: int = 15000
    ap_fallback: bool = True
    ap_ssid: str = "NEXUS-ESP8266"
    ap_password: str = ""
    ap_channel: int = 6
    static_ip_enabled: bool = False
    static_ip: str = ""
    gateway: str = ""
    subnet: str = "255.255.255.0"
    dns_primary: str = "8.8.8.8"
    dns_secondary: str = "8.8.4.4"
    hostname: str = "nexus-esp8266"
    power_save: bool = False


# ---------------------------------------------------------------------------
# WiFi connection manager
# ---------------------------------------------------------------------------

@dataclass
class WiFiManager:
    """WiFi connection manager for ESP8266-based NEXUS nodes.

    Manages the WiFi connection lifecycle including station mode,
    automatic reconnection, and AP fallback.  Tracks connection
    state, signal quality, and timing information.
    """

    config: WiFiManagerConfig = field(default_factory=WiFiManagerConfig)
    credentials: WiFiCredentials = field(default_factory=WiFiCredentials)

    # Runtime state
    state: WiFiConnectionState = WiFiConnectionState.DISCONNECTED
    connected_ssid: str = ""
    connected_bssid: str = ""
    rssi_dbm: int = 0
    reconnect_count: int = 0
    last_connected_ms: int = 0
    last_disconnect_reason: str = ""
    uptime_connected_ms: int = 0
    ap_mode_active: bool = False

    def set_credentials(self, ssid: str, password: str, bssid: str = "") -> None:
        """Set WiFi credentials and reset connection state."""
        self.credentials = WiFiCredentials(
            ssid=ssid, password=password, bssid=bssid
        )
        self.state = WiFiConnectionState.DISCONNECTED
        self.connected_ssid = ""
        self.reconnect_count = 0

    def get_connection_info(self) -> dict:
        """Return a dict with current connection status information."""
        return {
            "state": self.state.value,
            "connected_ssid": self.connected_ssid,
            "connected_bssid": self.connected_bssid,
            "rssi_dbm": self.rssi_dbm,
            "reconnect_count": self.reconnect_count,
            "last_connected_ms": self.last_connected_ms,
            "uptime_connected_ms": self.uptime_connected_ms,
            "ap_mode_active": self.ap_mode_active,
            "config": {
                "auto_connect": self.config.auto_connect,
                "auto_reconnect": self.config.auto_reconnect,
                "max_reconnect_attempts": self.config.max_reconnect_attempts,
                "ap_fallback": self.config.ap_fallback,
                "hostname": self.config.hostname,
            },
        }

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to WiFi in station mode."""
        return self.state == WiFiConnectionState.CONNECTED

    @property
    def should_reconnect(self) -> bool:
        """Check if a reconnection attempt should be made."""
        if not self.config.auto_reconnect:
            return False
        if self.reconnect_count >= self.config.max_reconnect_attempts:
            return False
        if self.state not in (
            WiFiConnectionState.DISCONNECTED,
            WiFiConnectionState.CONNECTION_FAILED,
        ):
            return False
        return self.credentials.is_valid()

    @property
    def should_fallback_to_ap(self) -> bool:
        """Check if the manager should fall back to AP mode."""
        if not self.config.ap_fallback:
            return False
        if self.ap_mode_active:
            return False
        if self.reconnect_count < self.config.max_reconnect_attempts:
            return False
        return True

    def signal_quality_percent(self) -> int:
        """Convert RSSI to approximate signal quality percentage.

        Based on typical WiFi RSSI ranges:
        -30 dBm = 100%, -80 dBm = 0%
        """
        if self.rssi_dbm >= -30:
            return 100
        elif self.rssi_dbm <= -80:
            return 0
        else:
            return int(100 * (self.rssi_dbm + 80) / 50)
