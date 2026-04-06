"""Tests for authentication module."""

import hashlib
import hmac
import time
import pytest
from jetson.security.authentication import (
    AuthResult,
    KeyManager,
    MessageAuthenticator,
    MessageEnvelope,
)


# ── MessageEnvelope ─────────────────────────────────────────────────

class TestMessageEnvelope:
    def test_create(self):
        env = MessageEnvelope(
            payload="hello", sender_id="node_1",
            timestamp=100.0, nonce="abc", signature="sig",
        )
        assert env.payload == "hello"
        assert env.sender_id == "node_1"
        assert env.timestamp == 100.0
        assert env.nonce == "abc"
        assert env.signature == "sig"

    def test_default_signature(self):
        env = MessageEnvelope(payload="x", sender_id="y", timestamp=0, nonce="z")
        assert env.signature == ""


# ── AuthResult ──────────────────────────────────────────────────────

class TestAuthResult:
    def test_verified(self):
        r = AuthResult(verified=True, sender_id="a", freshness=True, replay_detected=False)
        assert r.verified is True
        assert r.replay_detected is False

    def test_replay(self):
        r = AuthResult(verified=False, sender_id="a", freshness=True, replay_detected=True)
        assert r.replay_detected is True


# ── MessageAuthenticator ───────────────────────────────────────────

class TestMessageAuthenticatorConstruction:
    def test_construct(self):
        ma = MessageAuthenticator()
        assert ma.get_seen_count() == 0


class TestGenerateNonce:
    def test_nonce_length(self):
        ma = MessageAuthenticator()
        nonce = ma.generate_nonce()
        assert len(nonce) == 32  # 16 bytes -> 32 hex chars

    def test_nonce_unique(self):
        ma = MessageAuthenticator()
        nonces = {ma.generate_nonce() for _ in range(100)}
        assert len(nonces) == 100

    def test_nonce_hex(self):
        ma = MessageAuthenticator()
        nonce = ma.generate_nonce()
        # Must be valid hex
        int(nonce, 16)


class TestSignMessage:
    def test_sign_deterministic(self):
        ma = MessageAuthenticator()
        key = b"test_key_123456"
        sig1 = ma.sign_message("hello", key, 100.0, "nonce1")
        sig2 = ma.sign_message("hello", key, 100.0, "nonce1")
        assert sig1 == sig2

    def test_sign_different_keys(self):
        ma = MessageAuthenticator()
        sig1 = ma.sign_message("hello", b"key1", 100.0, "n")
        sig2 = ma.sign_message("hello", b"key2", 100.0, "n")
        assert sig1 != sig2

    def test_sign_different_payloads(self):
        ma = MessageAuthenticator()
        key = b"key"
        sig1 = ma.sign_message("hello", key, 100.0, "n")
        sig2 = ma.sign_message("world", key, 100.0, "n")
        assert sig1 != sig2

    def test_sign_different_timestamps(self):
        ma = MessageAuthenticator()
        key = b"key"
        sig1 = ma.sign_message("hello", key, 100.0, "n")
        sig2 = ma.sign_message("hello", key, 200.0, "n")
        assert sig1 != sig2

    def test_sign_length(self):
        ma = MessageAuthenticator()
        sig = ma.sign_message("hello", b"key", 0, "n")
        # SHA-256 hex = 64 chars
        assert len(sig) == 64


class TestVerifyMessage:
    def test_verify_valid(self):
        ma = MessageAuthenticator()
        key = b"my_secret_key!"
        envelope = ma.create_envelope("hello", "node_1", key)
        result = ma.verify_message(envelope, key)
        assert result.verified is True
        assert result.sender_id == "node_1"
        assert result.freshness is True
        assert result.replay_detected is False

    def test_verify_wrong_key(self):
        ma = MessageAuthenticator()
        key = b"correct_key"
        envelope = ma.create_envelope("hello", "node_1", key)
        result = ma.verify_message(envelope, b"wrong_key")
        assert result.verified is False

    def test_verify_tampered_payload(self):
        ma = MessageAuthenticator()
        key = b"key"
        envelope = ma.create_envelope("hello", "node_1", key)
        envelope.payload = "tampered"
        result = ma.verify_message(envelope, key)
        assert result.verified is False

    def test_verify_stale_message(self):
        ma = MessageAuthenticator()
        key = b"key"
        envelope = MessageEnvelope(
            payload="hello", sender_id="node_1",
            timestamp=time.time() - 1000,  # very old
            nonce=ma.generate_nonce(),
            signature="",
        )
        envelope.signature = ma.sign_message(envelope.payload, key, envelope.timestamp, envelope.nonce)
        result = ma.verify_message(envelope, key, max_age=10.0)
        assert result.verified is True  # signature valid
        assert result.freshness is False  # but stale

    def test_verify_fresh_message(self):
        ma = MessageAuthenticator()
        key = b"key"
        envelope = ma.create_envelope("hello", "node_1", key)
        result = ma.verify_message(envelope, key, max_age=300.0)
        assert result.freshness is True


class TestReplayDetection:
    def test_no_replay_first_time(self):
        ma = MessageAuthenticator()
        key = b"key"
        envelope = ma.create_envelope("hello", "node_1", key)
        assert ma.detect_replay(envelope) is False

    def test_replay_detected(self):
        ma = MessageAuthenticator()
        key = b"key"
        envelope = ma.create_envelope("hello", "node_1", key)
        ma.verify_message(envelope, key)  # consumes nonce
        result = ma.verify_message(envelope, key)  # replay
        assert result.replay_detected is True
        assert result.verified is False

    def test_detect_replay_custom_set(self):
        ma = MessageAuthenticator()
        env = MessageEnvelope(payload="x", sender_id="y", timestamp=0, nonce="unique_nonce")
        custom_set = {"unique_nonce"}
        assert ma.detect_replay(env, custom_set) is True

    def test_different_envelopes_not_replay(self):
        ma = MessageAuthenticator()
        key = b"key"
        env1 = ma.create_envelope("msg1", "node_1", key)
        env2 = ma.create_envelope("msg2", "node_1", key)
        ma.verify_message(env1, key)
        r2 = ma.verify_message(env2, key)
        assert r2.replay_detected is False
        assert r2.verified is True


class TestCreateEnvelope:
    def test_envelope_fields(self):
        ma = MessageAuthenticator()
        env = ma.create_envelope("payload", "sender", b"key")
        assert env.payload == "payload"
        assert env.sender_id == "sender"
        assert env.nonce != ""
        assert env.signature != ""
        assert env.timestamp > 0

    def test_envelope_signature_valid(self):
        ma = MessageAuthenticator()
        key = b"test_key"
        env = ma.create_envelope("test", "s", key)
        result = ma.verify_message(env, key)
        assert result.verified is True


class TestClearSeen:
    def test_clear_seen(self):
        ma = MessageAuthenticator()
        key = b"key"
        ma.create_envelope("msg", "s", key)
        ma.clear_seen()
        assert ma.get_seen_count() == 0


# ── KeyManager ─────────────────────────────────────────────────────

class TestKeyManagerConstruction:
    def test_construct(self):
        km = KeyManager()
        assert km.list_keys() == []


class TestGenerateKey:
    def test_key_length(self):
        km = KeyManager()
        key = km.generate_key()
        assert len(key) == 32

    def test_key_randomness(self):
        km = KeyManager()
        keys = {km.generate_key() for _ in range(100)}
        assert len(keys) == 100

    def test_key_bytes(self):
        km = KeyManager()
        key = km.generate_key()
        assert isinstance(key, bytes)


class TestGenerateKeyWithId:
    def test_store_and_retrieve(self):
        km = KeyManager()
        key = km.generate_key_with_id("node_1")
        assert km.get_key("node_1") == key

    def test_list_keys(self):
        km = KeyManager()
        km.generate_key_with_id("a")
        km.generate_key_with_id("b")
        assert set(km.list_keys()) == {"a", "b"}


class TestDeriveSharedKey:
    def test_deterministic(self):
        km = KeyManager()
        priv = b"private_key_1234_ab"
        pub = b"public_key_1234_abc"
        k1 = km.derive_shared_key(priv, pub)
        k2 = km.derive_shared_key(priv, pub)
        assert k1 == k2

    def test_symmetric(self):
        km = KeyManager()
        priv = b"key_a"
        pub = b"key_b"
        k1 = km.derive_shared_key(priv, pub)
        k2 = km.derive_shared_key(pub, priv)
        assert k1 == k2  # XOR is symmetric

    def test_key_length(self):
        km = KeyManager()
        key = km.derive_shared_key(b"a" * 32, b"b" * 32)
        assert len(key) == 32  # SHA-256 output


class TestRotateKey:
    def test_different_from_old(self):
        km = KeyManager()
        old = b"old_key_12345678901"
        new = km.rotate_key(old)
        assert new != old

    def test_deterministic_given_same_salt(self):
        # Rotation uses random salt, so results differ
        km = KeyManager()
        old = b"key" * 8
        keys = {km.rotate_key(old) for _ in range(20)}
        # With random salt, should get different keys each time
        assert len(keys) > 1

    def test_key_length(self):
        km = KeyManager()
        new = km.rotate_key(b"old" * 8)
        assert len(new) == 32


class TestRevokeKey:
    def test_revoke_existing(self):
        km = KeyManager()
        km.generate_key_with_id("node_1")
        km.revoke_key("node_1")
        assert km.get_key("node_1") is None

    def test_revoke_nonexistent(self):
        km = KeyManager()
        km.revoke_key("nope")  # should not raise

    def test_is_revoked(self):
        km = KeyManager()
        km.generate_key_with_id("node_1")
        assert km.is_revoked("node_1") is False
        km.revoke_key("node_1")
        assert km.is_revoked("node_1") is True

    def test_revoked_not_in_list(self):
        km = KeyManager()
        km.generate_key_with_id("node_1")
        km.revoke_key("node_1")
        assert "node_1" not in km.list_keys()


class TestGetKey:
    def test_nonexistent_key(self):
        km = KeyManager()
        assert km.get_key("missing") is None

    def test_revoked_returns_none(self):
        km = KeyManager()
        km.generate_key_with_id("node")
        km.revoke_key("node")
        assert km.get_key("node") is None
