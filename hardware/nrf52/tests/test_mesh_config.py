"""Tests for NEXUS Bluetooth Mesh configuration."""

import pytest
from hardware.nrf52.mesh_config import (
    NEXUSMeshConfig,
    MeshRole,
    ProvisioningMethod,
    SecurityLevel,
    MeshAddress,
    MeshKey,
    ProvisioningConfig,
    RelayConfig,
    FriendConfig,
    GattProxyConfig,
    MeshNetwork,
)


class TestMeshAddress:
    def test_unicast(self):
        a = MeshAddress(0x0001)
        assert a.is_unicast

    def test_group(self):
        a = MeshAddress(0xC000)
        assert a.is_group

    def test_virtual(self):
        a = MeshAddress(0xC001)
        assert a.is_virtual

    def test_fixed_group(self):
        a = MeshAddress(0xFF00)
        assert a.is_fixed_group

    def test_invalid_address(self):
        with pytest.raises(ValueError):
            MeshAddress(0x10000)

    def test_repr(self):
        a = MeshAddress(0x0123)
        assert "0123" in repr(a)


class TestMeshKey:
    def test_valid_key(self):
        k = MeshKey(key_index=0, key=b"\x01" * 16, name="test")
        assert k.key_index == 0

    def test_invalid_key_length(self):
        with pytest.raises(ValueError, match="16 bytes"):
            MeshKey(key_index=0, key=b"\x01" * 15)

    def test_invalid_key_index(self):
        with pytest.raises(ValueError, match="Key index"):
            MeshKey(key_index=5000, key=b"\x01" * 16)

    def test_repr(self):
        k = MeshKey(key_index=5, key=b"\x00" * 16, name="app_key")
        assert "idx=5" in repr(k)


class TestProvisioningConfig:
    def test_static_oob_without_key(self):
        p = ProvisioningConfig(method=ProvisioningMethod.STATIC_OOB)
        errors = p.validate()
        assert any("static_key" in e.lower() for e in errors)

    def test_static_oob_with_key(self):
        p = ProvisioningConfig(
            method=ProvisioningMethod.STATIC_OOB,
            oob_static_key=b"\x00" * 16,
        )
        assert p.is_valid()

    def test_invalid_key_length(self):
        p = ProvisioningConfig(
            method=ProvisioningMethod.STATIC_OOB,
            oob_static_key=b"\x00" * 8,
        )
        errors = p.validate()
        assert any("16 bytes" in e for e in errors)


class TestRelayConfig:
    def test_default_valid(self):
        r = RelayConfig()
        assert r.is_valid()

    def test_valid_retransmit(self):
        r = RelayConfig(retransmit_count=5, retransmit_interval_ms=100)
        assert r.is_valid()

    def test_invalid_count(self):
        r = RelayConfig(retransmit_count=8)
        errors = r.validate()
        assert any("0-7" in e for e in errors)

    def test_invalid_interval(self):
        r = RelayConfig(retransmit_interval_ms=5)
        errors = r.validate()
        assert any("10-320" in e for e in errors)

    def test_non_multiple_10(self):
        r = RelayConfig(retransmit_interval_ms=15)
        errors = r.validate()
        assert any("multiple of 10" in e for e in errors)


class TestFriendConfig:
    def test_default_valid(self):
        f = FriendConfig()
        assert f.is_valid()

    def test_invalid_window_factor(self):
        f = FriendConfig(receive_window_factor=20.0)
        errors = f.validate()
        assert any("0.5-16.0" in e for e in errors)


class TestGattProxyConfig:
    def test_default_valid(self):
        g = GattProxyConfig()
        assert g.is_valid()

    def test_invalid_filter_type(self):
        g = GattProxyConfig(filter_type="invalid")
        errors = g.validate()
        assert any("whitelist or blacklist" in e for e in errors)


class TestMeshNetwork:
    def test_default_valid(self):
        n = MeshNetwork()
        # No is_valid method on MeshNetwork, but key validation works
        assert len(n.app_keys) == 0

    def test_add_app_key(self):
        n = MeshNetwork()
        ak = n.add_app_key(b"\x02" * 16, key_index=1, name="sensor")
        assert ak.key_index == 1

    def test_duplicate_key_index(self):
        n = MeshNetwork()
        n.add_app_key(b"\x02" * 16, key_index=0)
        errors = n.validate()
        assert any("Duplicate" in e for e in errors)


class TestNEXUSMeshConfig:
    def test_default_role(self):
        m = NEXUSMeshConfig()
        assert m.role == MeshRole.UNPROVISIONED

    def test_provision_as_node(self):
        m = NEXUSMeshConfig()
        m.provision_as_node(0x0100, security=SecurityLevel.HIGH)
        assert m.role == MeshRole.NODE
        assert m.unicast_address.address == 0x0100

    def test_provision_as_relay(self):
        m = NEXUSMeshConfig()
        m.provision_as_relay(0x0200, retransmit_count=3)
        assert m.role == MeshRole.RELAY_NODE
        assert m.relay.retransmit_count == 3

    def test_provision_as_friend(self):
        m = NEXUSMeshConfig()
        m.provision_as_friend(0x0300)
        assert m.role == MeshRole.FRIEND_NODE
        assert m.friend.enabled is True

    def test_enable_gatt_proxy(self):
        m = NEXUSMeshConfig()
        m.provision_as_node(0x0100)
        m.enable_gatt_proxy(max_entries=32)
        assert m.gatt_proxy.enabled is True
        assert m.gatt_proxy.max_filter_entries == 32

    def test_add_model(self):
        m = NEXUSMeshConfig()
        m.add_model("0001")
        m.add_model("0002")
        assert len(m.list_models()) == 2

    def test_validate_unprovisioned_ok(self):
        m = NEXUSMeshConfig()
        assert m.is_valid()

    def test_validate_relay_invalid(self):
        m = NEXUSMeshConfig()
        m.relay.retransmit_count = 8
        m.role = MeshRole.RELAY_NODE
        errors = m.validate()
        assert len(errors) > 0

    def test_summary(self):
        m = NEXUSMeshConfig()
        m.provision_as_relay(0x0500)
        m.add_model("sensor_v1")
        s = m.summary()
        assert s["role"] == "RELAY_NODE"
        assert s["unicast_address"] == "0x0500"
        assert s["relay_enabled"] is True
        assert "sensor_v1" in s["models"]
