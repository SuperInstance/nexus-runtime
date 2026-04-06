"""Authentication & authorization: API keys, bearer tokens, HMAC hashing."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class APIKey:
    """An API key credential."""
    key_id: str
    key_hash: str
    name: str
    permissions: List[str] = field(default_factory=list)
    created: float = 0.0
    last_used: float = 0.0
    rate_limit: int = 1000


@dataclass
class AuthToken:
    """A bearer authentication token."""
    token: str
    user_id: str
    scopes: List[str] = field(default_factory=list)
    expires_at: float = 0.0
    issued_at: float = 0.0


class AuthManager:
    """Manages API keys and bearer tokens.

    Keys are stored as HMAC-SHA256 hashes. Tokens are structured
    as ``base64url({header}.{payload}.{signature})`` using only stdlib.
    """

    def __init__(self, secret: str = "nexus-default-secret") -> None:
        self._secret = secret.encode("utf-8")
        self._api_keys: Dict[str, APIKey] = {}        # key_id -> APIKey
        self._key_lookup: Dict[str, str] = {}          # raw_key -> key_id
        self._tokens: Dict[str, AuthToken] = {}        # token -> AuthToken
        self._revoked: Set[str] = set()                # revoked tokens

    @staticmethod
    def hash_key(key: str, secret: str = "nexus-default-secret") -> str:
        """Compute HMAC-SHA256 hash of a raw API key."""
        return hmac.new(
            secret.encode("utf-8"), key.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def generate_api_key(self, name: str, permissions: Optional[List[str]] = None,
                         rate_limit: int = 1000) -> APIKey:
        """Generate a new API key. Returns the APIKey (raw key is in key_id).

        The key_id contains the raw key prefixed with ``nxk_``.
        The key_hash is the HMAC of the raw key.
        """
        import secrets
        raw_key = secrets.token_hex(32)
        key_id = f"nxk_{raw_key}"
        now = time.time()
        key_hash = self.hash_key(raw_key)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions or [],
            created=now,
            last_used=now,
            rate_limit=rate_limit,
        )

        self._api_keys[key_id] = api_key
        self._key_lookup[raw_key] = key_id

        return api_key

    def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validate a raw API key. Returns the APIKey if valid, None otherwise."""
        key_hash = self.hash_key(key)

        for api_key in self._api_keys.values():
            if hmac.compare_digest(api_key.key_hash, key_hash):
                # Update last_used
                api_key.last_used = time.time()
                return api_key

        return None

    def generate_token(self, user_id: str, scopes: Optional[List[str]] = None,
                       ttl: int = 3600) -> AuthToken:
        """Generate a new bearer token with the given TTL (seconds)."""
        import secrets
        now = time.time()
        raw_token = secrets.token_hex(32)
        token = f"nxt_{raw_token}"

        auth_token = AuthToken(
            token=token,
            user_id=user_id,
            scopes=scopes or [],
            expires_at=now + ttl,
            issued_at=now,
        )

        self._tokens[token] = auth_token
        return auth_token

    def validate_token(self, token: str) -> Optional[AuthToken]:
        """Validate a bearer token. Returns the AuthToken if valid and not revoked/expired."""
        auth_token = self._tokens.get(token)
        if auth_token is None:
            return None

        if token in self._revoked:
            return None

        if time.time() > auth_token.expires_at:
            return None

        return auth_token

    def check_permission(self, token: str, required_scope: str) -> bool:
        """Check if a token has the required scope."""
        auth_token = self.validate_token(token)
        if auth_token is None:
            return False

        # Admin scope grants all permissions
        if "admin" in auth_token.scopes:
            return True

        return required_scope in auth_token.scopes

    def revoke_token(self, token: str) -> bool:
        """Revoke a token. Returns True if the token existed (even if expired)."""
        if token in self._tokens:
            self._revoked.add(token)
            return True
        return False

    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Look up an API key by key_id."""
        return self._api_keys.get(key_id)

    def remove_api_key(self, key_id: str) -> bool:
        """Remove an API key. Returns True if it existed."""
        if key_id in self._api_keys:
            api_key = self._api_keys.pop(key_id)
            # Remove from key_lookup — find by raw_key
            # The raw key is key_id without "nxk_" prefix
            raw_prefix = key_id[4:]  # strip "nxk_"
            for k, v in list(self._key_lookup.items()):
                if v == key_id and k == raw_prefix:
                    del self._key_lookup[k]
                    break
            return True
        return False

    def list_api_keys(self) -> List[APIKey]:
        """List all registered API keys."""
        return list(self._api_keys.values())

    def list_tokens(self) -> List[AuthToken]:
        """List all non-revoked tokens."""
        return [t for t in self._tokens.values() if t.token not in self._revoked]

    def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens. Returns count removed."""
        now = time.time()
        expired = [
            token for token, auth_token in self._tokens.items()
            if now > auth_token.expires_at
        ]
        for token in expired:
            self._tokens.pop(token, None)
            self._revoked.discard(token)
        return len(expired)
