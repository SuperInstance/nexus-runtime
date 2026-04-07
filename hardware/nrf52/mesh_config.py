"""
NEXUS Bluetooth Mesh Configuration for nRF52 Devices

Provides provisioning, relay, and network configuration for Bluetooth mesh
deployments in marine sensor networks. Supports provisioning as node/provisioner,
mesh relay configuration, and friend/low-power node setup.

Bluetooth mesh enables multi-hop sensor networks where data can relay from
underwater sensor nodes through surface buoys to the NEXUS command station.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


class MeshRole(IntEnum):
    """Bluetooth mesh node roles."""
    UNPROVISIONED = 0
    NODE = 1
    PROVISIONER = 2
    RELAY_NODE = 3
    FRIEND_NODE = 4
    LOW_POWER_NODE = 5
    GATT_PROXY = 6


class ProvisioningMethod(IntEnum):
    """Provisioning methods supported."""
    STATIC_OOB = 0
    OUTPUT_OOB = 1
    INPUT_OOB = 2
    PUBLIC_KEY_OOB = 3


class SecurityLevel(IntEnum):
    """Mesh security levels."""
    LOW = 0       # No OOB
    MEDIUM = 1    # OOB authentication
    HIGH = 2      # Full OOB with 16-byte key


@dataclass
class MeshAddress:
    """Bluetooth mesh address."""
    address: int

    def __post_init__(self):
        if not (0x0000 <= self.address <= 0xFFFF):
            raise ValueError(f"Mesh address must be 0x0000-0xFFFF, got 0x{self.address:04X}.")

    @property
    def is_unicast(self) -> bool:
        return 0x0001 <= self.address <= 0x7FFF

    @property
    def is_group(self) -> bool:
        return 0xC000 <= self.address <= 0xFFFF

    @property
    def is_virtual(self) -> bool:
        return 0xC000 <= self.address <= 0xFF00

    @property
    def is_fixed_group(self) -> bool:
        return 0xFF00 <= self.address <= 0xFFFF

    def __repr__(self) -> str:
        return f"MeshAddress(0x{self.address:04X})"


@dataclass
class MeshKey:
    """Bluetooth mesh application or network key."""
    key_index: int
    key: bytes
    name: str = ""

    def __post_init__(self):
        if len(self.key) != 16:
            raise ValueError(f"Mesh key must be 16 bytes, got {len(self.key)}.")
        if not (0 <= self.key_index <= 4095):
            raise ValueError(f"Key index must be 0-4095, got {self.key_index}.")

    def __repr__(self) -> str:
        return f"MeshKey(idx={self.key_index}, name='{self.name}')"


@dataclass
class ProvisioningConfig:
    """Configuration for device provisioning."""
    method: ProvisioningMethod = ProvisioningMethod.OUTPUT_OOB
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    oob_static_key: Optional[bytes] = None
    attention_timer_s: int = 5
    public_key_type: str = "secp256r1"    # or "secp384r1"

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.method == ProvisioningMethod.STATIC_OOB and not self.oob_static_key:
            errors.append("Static OOB provisioning requires oob_static_key.")
        if self.oob_static_key and len(self.oob_static_key) != 16:
            errors.append(f"OOB static key must be 16 bytes, got {len(self.oob_static_key)}.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


@dataclass
class RelayConfig:
    """Bluetooth mesh relay configuration."""
    enabled: bool = True
    retransmit_count: int = 2            # 0-7
    retransmit_interval_ms: int = 20     # 10-320 ms, steps of 10

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not (0 <= self.retransmit_count <= 7):
            errors.append(f"Retransmit count must be 0-7, got {self.retransmit_count}.")
        if not (10 <= self.retransmit_interval_ms <= 320):
            errors.append(f"Retransmit interval must be 10-320ms, got {self.retransmit_interval_ms}.")
        if self.retransmit_interval_ms % 10 != 0:
            errors.append("Retransmit interval must be a multiple of 10ms.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


@dataclass
class FriendConfig:
    """Friend node configuration for low-power node support."""
    enabled: bool = False
    friendship_establishment_timeout_s: int = 60
    receive_window_factor: float = 1.0    # 0.5-16.0
    subscription_list_size: int = 10

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not (0.5 <= self.receive_window_factor <= 16.0):
            errors.append(f"Receive window factor must be 0.5-16.0, got {self.receive_window_factor}.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


@dataclass
class GattProxyConfig:
    """GATT Proxy configuration for connecting non-mesh devices."""
    enabled: bool = False
    filter_type: str = "whitelist"       # whitelist or blacklist
    max_filter_entries: int = 16

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.filter_type not in ("whitelist", "blacklist"):
            errors.append(f"Filter type must be whitelist or blacklist, got {self.filter_type}.")
        if not (1 <= self.max_filter_entries <= 32767):
            errors.append(f"Max filter entries must be 1-32767, got {self.max_filter_entries}.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


@dataclass
class MeshNetwork:
    """Bluetooth mesh network configuration."""
    name: str = "NEXUS Marine Mesh"
    net_key: MeshKey = field(default_factory=lambda: MeshKey(
        key_index=0,
        key=b"\x00" * 16,
        name="Primary Network Key",
    ))
    app_keys: List[MeshKey] = field(default_factory=list)
    iv_index: int = 0
    iv_update_interval_hours: int = 96   # At least 96 hours

    def add_app_key(self, key: bytes, key_index: int, name: str = "") -> MeshKey:
        mk = MeshKey(key_index=key_index, key=key, name=name)
        self.app_keys.append(mk)
        return mk

    def validate(self) -> List[str]:
        errors: List[str] = []
        # net_key validated in __post_init__
        idx_set = {self.net_key.key_index}
        for ak in self.app_keys:
            if ak.key_index in idx_set:
                errors.append(f"Duplicate key index: {ak.key_index}.")
            idx_set.add(ak.key_index)
        if self.iv_index < 0:
            errors.append("IV index must be non-negative.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


class NEXUSMeshConfig:
    """
    NEXUS Bluetooth Mesh configuration for marine sensor networks.

    Manages provisioning, relay, friend, and GATT proxy settings
    for multi-hop underwater/surface sensor mesh deployments.
    """

    def __init__(self, role: MeshRole = MeshRole.UNPROVISIONED):
        self.role = role
        self.unicast_address: Optional[MeshAddress] = None
        self.provisioning = ProvisioningConfig()
        self.relay = RelayConfig()
        self.friend = FriendConfig()
        self.gatt_proxy = GattProxyConfig()
        self.network = MeshNetwork()
        self._model_ids: List[str] = []

    def provision_as_node(
        self,
        unicast_address: int,
        oob_key: Optional[bytes] = None,
        security: SecurityLevel = SecurityLevel.MEDIUM,
    ):
        """Configure as a mesh node."""
        self.role = MeshRole.NODE
        self.unicast_address = MeshAddress(unicast_address)
        self.provisioning.method = ProvisioningMethod.STATIC_OOB
        self.provisioning.oob_static_key = oob_key
        self.provisioning.security_level = security

    def provision_as_relay(
        self,
        unicast_address: int,
        retransmit_count: int = 2,
        retransmit_interval_ms: int = 20,
    ):
        """Configure as a mesh relay node for multi-hop routing."""
        self.provision_as_node(unicast_address)
        self.role = MeshRole.RELAY_NODE
        self.relay.enabled = True
        self.relay.retransmit_count = retransmit_count
        self.relay.retransmit_interval_ms = retransmit_interval_ms

    def provision_as_friend(self, unicast_address: int):
        """Configure as a friend node supporting low-power nodes."""
        self.provision_as_node(unicast_address)
        self.role = MeshRole.FRIEND_NODE
        self.friend.enabled = True

    def enable_gatt_proxy(self, max_entries: int = 16):
        """Enable GATT proxy for BLE-to-mesh bridging."""
        self.gatt_proxy.enabled = True
        self.gatt_proxy.max_filter_entries = max_entries

    def add_model(self, model_id: str):
        """Register a mesh model (e.g., vendor model or SIG model)."""
        self._model_ids.append(model_id)

    def list_models(self) -> List[str]:
        return list(self._model_ids)

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.role != MeshRole.UNPROVISIONED and not self.unicast_address:
            errors.append("Provisioned node requires a unicast address.")
        if self.role == MeshRole.RELAY_NODE:
            errors.extend(self.relay.validate())
        if self.role == MeshRole.FRIEND_NODE:
            errors.extend(self.friend.validate())
        if self.gatt_proxy.enabled:
            errors.extend(self.gatt_proxy.validate())
        errors.extend(self.provisioning.validate())
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def summary(self) -> Dict[str, Any]:
        return {
            "role": self.role.name,
            "unicast_address": f"0x{self.unicast_address.address:04X}" if self.unicast_address else None,
            "relay_enabled": self.relay.enabled,
            "friend_enabled": self.friend.enabled,
            "gatt_proxy_enabled": self.gatt_proxy.enabled,
            "models": self.list_models(),
            "network": self.network.name,
            "app_keys": len(self.network.app_keys),
        }
