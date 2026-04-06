"""Tests for secrets.py — SecretEntry, SecretManager."""

import pytest
from jetson.config_mgmt.secrets import SecretEntry, SecretManager


class TestSecretEntry:
    def test_creation_defaults(self):
        entry = SecretEntry(name="key", value="secret")
        assert entry.name == "key"
        assert entry.value == "secret"
        assert entry.version == 1
        assert entry.created == 0.0
        assert entry.last_accessed == 0.0

    def test_creation_with_all_fields(self):
        entry = SecretEntry(name="token", value="abc123", version=3, created=100.0, last_accessed=200.0)
        assert entry.name == "token"
        assert entry.value == "abc123"
        assert entry.version == 3
        assert entry.created == 100.0
        assert entry.last_accessed == 200.0


class TestSecretManager:
    def _make_manager(self):
        return SecretManager()

    # ── store_secret ────────────────────────────────────────────────────

    def test_store_new_secret(self):
        mgr = self._make_manager()
        version = mgr.store_secret("db_password", "s3cret")
        assert version == 1

    def test_store_overwrite_increments_version(self):
        mgr = self._make_manager()
        mgr.store_secret("key", "v1")
        version = mgr.store_secret("key", "v2")
        assert version == 2

    def test_store_overwrite_updates_value(self):
        mgr = self._make_manager()
        mgr.store_secret("key", "old")
        mgr.store_secret("key", "new")
        assert mgr.retrieve_secret("key") == "new"

    def test_store_empty_value(self):
        mgr = self._make_manager()
        version = mgr.store_secret("empty", "")
        assert version == 1
        assert mgr.retrieve_secret("empty") == ""

    # ── retrieve_secret ─────────────────────────────────────────────────

    def test_retrieve_existing(self):
        mgr = self._make_manager()
        mgr.store_secret("token", "abc123")
        assert mgr.retrieve_secret("token") == "abc123"

    def test_retrieve_nonexistent(self):
        mgr = self._make_manager()
        assert mgr.retrieve_secret("missing") is None

    def test_retrieve_updates_last_accessed(self):
        mgr = self._make_manager()
        mgr.store_secret("key", "val")
        import time
        time.sleep(0.01)
        mgr.retrieve_secret("key")
        meta = mgr.get_secret_metadata("key")
        assert meta["last_accessed"] > meta["created"]

    # ── rotate_secret ───────────────────────────────────────────────────

    def test_rotate_existing(self):
        mgr = self._make_manager()
        mgr.store_secret("key", "old_value")
        new_version = mgr.rotate_secret("key")
        assert new_version == 2
        new_value = mgr.retrieve_secret("key")
        assert new_value != "old_value"
        assert len(new_value) == 32

    def test_rotate_nonexistent_raises(self):
        mgr = self._make_manager()
        with pytest.raises(KeyError):
            mgr.rotate_secret("nonexistent")

    def test_rotate_generates_different_value(self):
        mgr = self._make_manager()
        mgr.store_secret("key", "original")
        v1 = mgr.rotate_secret("key")
        val1 = mgr.retrieve_secret("key")
        v2 = mgr.rotate_secret("key")
        val2 = mgr.retrieve_secret("key")
        assert v1 == 2
        assert v2 == 3
        assert val1 != val2

    # ── list_secrets ────────────────────────────────────────────────────

    def test_list_empty(self):
        mgr = self._make_manager()
        assert mgr.list_secrets() == []

    def test_list_secrets_sorted(self):
        mgr = self._make_manager()
        mgr.store_secret("charlie", "c")
        mgr.store_secret("alpha", "a")
        mgr.store_secret("bravo", "b")
        assert mgr.list_secrets() == ["alpha", "bravo", "charlie"]

    # ── delete_secret ───────────────────────────────────────────────────

    def test_delete_existing(self):
        mgr = self._make_manager()
        mgr.store_secret("key", "val")
        mgr.delete_secret("key")
        assert "key" not in mgr.list_secrets()
        assert mgr.retrieve_secret("key") is None

    def test_delete_nonexistent_raises(self):
        mgr = self._make_manager()
        with pytest.raises(KeyError):
            mgr.delete_secret("nope")

    # ── get_secret_metadata ─────────────────────────────────────────────

    def test_metadata_existing(self):
        mgr = self._make_manager()
        mgr.store_secret("token", "my-secret-token")
        meta = mgr.get_secret_metadata("token")
        assert meta is not None
        assert meta["name"] == "token"
        assert meta["version"] == 1
        assert meta["length"] == 15
        assert "created" in meta
        assert "last_accessed" in meta

    def test_metadata_no_value_exposed(self):
        mgr = self._make_manager()
        mgr.store_secret("key", "super-secret")
        meta = mgr.get_secret_metadata("key")
        assert "value" not in meta
        assert meta["length"] == 12

    def test_metadata_nonexistent(self):
        mgr = self._make_manager()
        assert mgr.get_secret_metadata("missing") is None

    def test_metadata_after_rotate(self):
        mgr = self._make_manager()
        mgr.store_secret("key", "v1")
        mgr.rotate_secret("key")
        meta = mgr.get_secret_metadata("key")
        assert meta["version"] == 2

    # ── mask_secret ─────────────────────────────────────────────────────

    def test_mask_long_value(self):
        mgr = self._make_manager()
        masked = mgr.mask_secret("my-super-secret-password")
        assert masked.startswith("my-s")
        assert "*" in masked
        assert masked.endswith("****")

    def test_mask_short_value(self):
        mgr = self._make_manager()
        masked = mgr.mask_secret("abc")
        assert masked == "***"

    def test_mask_empty_value(self):
        mgr = self._make_manager()
        masked = mgr.mask_secret("")
        assert masked == ""

    def test_mask_custom_visible_chars(self):
        mgr = self._make_manager()
        masked = mgr.mask_secret("abcdefgh", visible_chars=2)
        assert masked.startswith("ab")
        assert masked[2:] == "******"

    def test_mask_custom_mask_char(self):
        mgr = self._make_manager()
        masked = mgr.mask_secret("abcdefgh", visible_chars=3, mask_char="#")
        assert masked == "abc#####"

    # ── validate_secret_strength ────────────────────────────────────────

    def test_strength_empty(self):
        mgr = self._make_manager()
        assert mgr.validate_secret_strength("") == 0

    def test_strength_very_weak(self):
        mgr = self._make_manager()
        score = mgr.validate_secret_strength("abc")
        assert 0 < score < 50

    def test_strength_weak(self):
        mgr = self._make_manager()
        score = mgr.validate_secret_strength("password")
        assert 0 < score < 70

    def test_strength_medium(self):
        mgr = self._make_manager()
        score = mgr.validate_secret_strength("Password1")
        assert 40 <= score <= 80

    def test_strength_strong(self):
        mgr = self._make_manager()
        score = mgr.validate_secret_strength("MyStr0ng!P@ss")
        assert score >= 70

    def test_strength_very_strong(self):
        mgr = self._make_manager()
        score = mgr.validate_secret_strength("Th1s!s@V3ry$tr0ng&S3cret")
        assert score >= 85

    def test_strength_only_digits(self):
        mgr = self._make_manager()
        score = mgr.validate_secret_strength("12345678")
        assert score > 0
        assert score < 60  # missing many categories

    def test_strength_no_repeat_bonus(self):
        mgr = self._make_manager()
        s1 = mgr.validate_secret_strength("abcABC123!@#")
        s2 = mgr.validate_secret_strength("aaaBBB111!!!")
        assert s1 > s2  # no repeats gets bonus

    def test_strength_capped_at_100(self):
        mgr = self._make_manager()
        score = mgr.validate_secret_strength("A" * 100 + "1" + "a" + "!" + "B")
        assert score <= 100

    # ── search_secrets ──────────────────────────────────────────────────

    def test_search_by_exact_name(self):
        mgr = self._make_manager()
        mgr.store_secret("database_url", "postgresql://...")
        mgr.store_secret("api_key", "key123")
        results = mgr.search_secrets("database_url")
        assert len(results) == 1
        assert results[0]["name"] == "database_url"

    def test_search_by_pattern(self):
        mgr = self._make_manager()
        mgr.store_secret("db_host", "localhost")
        mgr.store_secret("db_port", "5432")
        mgr.store_secret("api_key", "key123")
        results = mgr.search_secrets("db_")
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "db_host" in names
        assert "db_port" in names

    def test_search_case_insensitive(self):
        mgr = self._make_manager()
        mgr.store_secret("API_KEY", "key")
        results = mgr.search_secrets("api_key")
        assert len(results) == 1

    def test_search_no_results(self):
        mgr = self._make_manager()
        mgr.store_secret("key", "val")
        results = mgr.search_secrets("nonexistent")
        assert results == []

    def test_search_empty_pattern(self):
        mgr = self._make_manager()
        mgr.store_secret("a", "1")
        mgr.store_secret("b", "2")
        results = mgr.search_secrets("")
        assert len(results) == 2

    def test_search_results_no_values(self):
        mgr = self._make_manager()
        mgr.store_secret("secret_key", "top-secret-value")
        results = mgr.search_secrets("secret")
        for r in results:
            assert "value" not in r
            assert r["name"] == "secret_key"
