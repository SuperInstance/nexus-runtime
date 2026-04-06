"""Tests for auth.py — API key management, token auth, permissions."""

import time
import hashlib
import hmac as hmac_mod

import pytest

from jetson.api_gateway.auth import APIKey, AuthToken, AuthManager


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def auth():
    return AuthManager(secret="test-secret-key")


# ── APIKey dataclass ─────────────────────────────────────────────────

class TestAPIKey:
    def test_creation(self):
        key = APIKey(
            key_id="nxk_abc",
            key_hash="sha256hash",
            name="test-key",
            permissions=["read", "write"],
        )
        assert key.key_id == "nxk_abc"
        assert key.key_hash == "sha256hash"
        assert key.name == "test-key"
        assert key.permissions == ["read", "write"]
        assert key.created == 0.0
        assert key.last_used == 0.0
        assert key.rate_limit == 1000

    def test_defaults(self):
        key = APIKey(key_id="id", key_hash="hash", name="n")
        assert key.permissions == []
        assert key.created == 0.0
        assert key.rate_limit == 1000


# ── AuthToken dataclass ──────────────────────────────────────────────

class TestAuthToken:
    def test_creation(self):
        token = AuthToken(
            token="nxt_abc",
            user_id="user1",
            scopes=["read"],
            expires_at=9999999999.0,
            issued_at=1000000000.0,
        )
        assert token.token == "nxt_abc"
        assert token.user_id == "user1"
        assert token.scopes == ["read"]

    def test_defaults(self):
        token = AuthToken(token="tok", user_id="u")
        assert token.scopes == []
        assert token.expires_at == 0.0
        assert token.issued_at == 0.0


# ── AuthManager: hash_key ────────────────────────────────────────────

class TestHashKey:
    def test_deterministic(self):
        h1 = AuthManager.hash_key("mykey", "secret1")
        h2 = AuthManager.hash_key("mykey", "secret1")
        assert h1 == h2

    def test_different_keys_different_hashes(self):
        h1 = AuthManager.hash_key("key1", "secret")
        h2 = AuthManager.hash_key("key2", "secret")
        assert h1 != h2

    def test_different_secrets_different_hashes(self):
        h1 = AuthManager.hash_key("key", "secret1")
        h2 = AuthManager.hash_key("key", "secret2")
        assert h1 != h2

    def test_is_hex_string(self):
        h = AuthManager.hash_key("test", "sec")
        assert all(c in "0123456789abcdef" for c in h)

    def test_sha256_length(self):
        h = AuthManager.hash_key("test", "sec")
        assert len(h) == 64  # SHA-256 hex digest

    def test_default_secret(self):
        h = AuthManager.hash_key("test")
        assert len(h) == 64


# ── AuthManager: generate_api_key ────────────────────────────────────

class TestGenerateAPIKey:
    def test_returns_api_key(self, auth):
        key = auth.generate_api_key("test-key")
        assert isinstance(key, APIKey)
        assert key.name == "test-key"

    def test_key_id_has_prefix(self, auth):
        key = auth.generate_api_key("test")
        assert key.key_id.startswith("nxk_")

    def test_key_hash_is_hex(self, auth):
        key = auth.generate_api_key("test")
        assert all(c in "0123456789abcdef" for c in key.key_hash)

    def test_permissions(self, auth):
        key = auth.generate_api_key("test", permissions=["read", "write"])
        assert key.permissions == ["read", "write"]

    def test_custom_rate_limit(self, auth):
        key = auth.generate_api_key("test", rate_limit=500)
        assert key.rate_limit == 500

    def test_created_timestamp(self, auth):
        before = time.time()
        key = auth.generate_api_key("test")
        after = time.time()
        assert before <= key.created <= after

    def test_unique_keys(self, auth):
        k1 = auth.generate_api_key("k1")
        k2 = auth.generate_api_key("k2")
        assert k1.key_id != k2.key_id


# ── AuthManager: validate_api_key ────────────────────────────────────

class TestValidateAPIKey:
    def test_validate_valid_key(self, auth):
        key = auth.generate_api_key("test")
        # The raw key is key_id without "nxk_" prefix
        raw = key.key_id[4:]
        validated = auth.validate_api_key(raw)
        assert validated is not None
        assert validated.name == "test"

    def test_validate_invalid_key(self, auth):
        auth.generate_api_key("test")
        result = auth.validate_api_key("invalid_key_123")
        assert result is None

    def test_validate_updates_last_used(self, auth):
        key = auth.generate_api_key("test")
        raw = key.key_id[4:]
        time.sleep(0.01)
        validated = auth.validate_api_key(raw)
        assert validated.last_used > key.created

    def test_validate_after_remove(self, auth):
        key = auth.generate_api_key("test")
        raw = key.key_id[4:]
        auth.remove_api_key(key.key_id)
        result = auth.validate_api_key(raw)
        assert result is None


# ── AuthManager: generate_token / validate_token ────────────────────

class TestTokenAuth:
    def test_generate_token(self, auth):
        token = auth.generate_token("user1", scopes=["read"])
        assert isinstance(token, AuthToken)
        assert token.user_id == "user1"
        assert token.scopes == ["read"]
        assert token.token.startswith("nxt_")

    def test_token_has_expiration(self, auth):
        token = auth.generate_token("user1", ttl=3600)
        now = time.time()
        assert token.expires_at > now
        assert token.expires_at <= now + 3601

    def test_validate_valid_token(self, auth):
        token = auth.generate_token("user1", scopes=["read"], ttl=3600)
        validated = auth.validate_token(token.token)
        assert validated is not None
        assert validated.user_id == "user1"

    def test_validate_invalid_token(self, auth):
        result = auth.validate_token("nxt_nonexistent")
        assert result is None

    def test_validate_revoked_token(self, auth):
        token = auth.generate_token("user1", ttl=3600)
        auth.revoke_token(token.token)
        result = auth.validate_token(token.token)
        assert result is None

    def test_validate_expired_token(self, auth):
        token = auth.generate_token("user1", ttl=0)
        time.sleep(0.01)
        result = auth.validate_token(token.token)
        assert result is None

    def test_token_issued_at(self, auth):
        before = time.time()
        token = auth.generate_token("user1")
        after = time.time()
        assert before <= token.issued_at <= after


# ── AuthManager: check_permission ────────────────────────────────────

class TestCheckPermission:
    def test_has_scope(self, auth):
        token = auth.generate_token("user1", scopes=["read", "write"], ttl=3600)
        assert auth.check_permission(token.token, "read") is True
        assert auth.check_permission(token.token, "write") is True

    def test_missing_scope(self, auth):
        token = auth.generate_token("user1", scopes=["read"], ttl=3600)
        assert auth.check_permission(token.token, "admin") is False

    def test_admin_scope_grants_all(self, auth):
        token = auth.generate_token("user1", scopes=["admin"], ttl=3600)
        assert auth.check_permission(token.token, "anything") is True

    def test_invalid_token(self, auth):
        assert auth.check_permission("invalid", "read") is False

    def test_revoked_token_no_permission(self, auth):
        token = auth.generate_token("user1", scopes=["read"], ttl=3600)
        auth.revoke_token(token.token)
        assert auth.check_permission(token.token, "read") is False


# ── AuthManager: revoke_token ────────────────────────────────────────

class TestRevokeToken:
    def test_revoke_existing(self, auth):
        token = auth.generate_token("user1", ttl=3600)
        assert auth.revoke_token(token.token) is True

    def test_revoke_nonexistent(self, auth):
        assert auth.revoke_token("nxt_no_such_token") is False

    def test_double_revoke(self, auth):
        token = auth.generate_token("user1", ttl=3600)
        assert auth.revoke_token(token.token) is True
        assert auth.revoke_token(token.token) is True  # still returns True (existed in _tokens)

    def test_revoke_expired_still_works(self, auth):
        token = auth.generate_token("user1", ttl=0)
        time.sleep(0.01)
        assert auth.revoke_token(token.token) is True


# ── AuthManager: get / remove / list ─────────────────────────────────

class TestKeyManagement:
    def test_get_api_key(self, auth):
        key = auth.generate_api_key("test")
        found = auth.get_api_key(key.key_id)
        assert found is not None
        assert found.name == "test"

    def test_get_nonexistent_key(self, auth):
        assert auth.get_api_key("nope") is None

    def test_remove_api_key(self, auth):
        key = auth.generate_api_key("test")
        assert auth.remove_api_key(key.key_id) is True
        assert auth.get_api_key(key.key_id) is None

    def test_remove_nonexistent_key(self, auth):
        assert auth.remove_api_key("nope") is False

    def test_list_api_keys(self, auth):
        auth.generate_api_key("k1")
        auth.generate_api_key("k2")
        keys = auth.list_api_keys()
        assert len(keys) == 2
        names = {k.name for k in keys}
        assert "k1" in names
        assert "k2" in names

    def test_list_tokens(self, auth):
        t1 = auth.generate_token("u1", ttl=3600)
        t2 = auth.generate_token("u2", ttl=3600)
        tokens = auth.list_tokens()
        assert len(tokens) == 2

    def test_list_tokens_excludes_revoked(self, auth):
        t1 = auth.generate_token("u1", ttl=3600)
        t2 = auth.generate_token("u2", ttl=3600)
        auth.revoke_token(t1.token)
        tokens = auth.list_tokens()
        assert len(tokens) == 1
        assert tokens[0].user_id == "u2"

    def test_cleanup_expired_tokens(self, auth):
        auth.generate_token("u1", ttl=0)
        auth.generate_token("u2", ttl=3600)
        time.sleep(0.01)
        removed = auth.cleanup_expired_tokens()
        assert removed == 1

    def test_cleanup_no_expired(self, auth):
        auth.generate_token("u1", ttl=3600)
        removed = auth.cleanup_expired_tokens()
        assert removed == 0
