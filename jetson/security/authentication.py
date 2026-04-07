"""HMAC message authentication and replay protection."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MessageEnvelope:
    payload: str
    sender_id: str
    timestamp: float
    nonce: str
    signature: str = ""


@dataclass
class AuthResult:
    verified: bool
    sender_id: str
    freshness: bool
    replay_detected: bool


class MessageAuthenticator:
    """HMAC-SHA256 based message authentication with replay detection."""

    HASH_NAME = "sha256"
    NONCE_LENGTH = 16
    MAX_AGE_SECONDS = 300.0  # 5 minutes

    MAX_NONCE_ENTRIES = 10000

    def __init__(self) -> None:
        self._seen_nonces: Dict[str, float] = {}  # nonce -> timestamp
        self._seen_messages: List[MessageEnvelope] = []

    @staticmethod
    def _hmac_sign(key: bytes, message: bytes) -> str:
        return hmac.new(key, message, hashlib.sha256).hexdigest()

    @staticmethod
    def _hmac_verify(key: bytes, message: bytes, signature: str) -> bool:
        expected = hmac.new(key, message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    def generate_nonce(self) -> str:
        """Generate a cryptographically random nonce."""
        raw = os.urandom(self.NONCE_LENGTH)
        return raw.hex()

    def sign_message(
        self,
        payload: str,
        sender_key: bytes,
        timestamp: float,
        nonce: str,
    ) -> str:
        """Sign a message with HMAC-SHA256."""
        message_bytes = f"{payload}:{timestamp}:{nonce}".encode("utf-8")
        return self._hmac_sign(sender_key, message_bytes)

    def create_envelope(
        self,
        payload: str,
        sender_id: str,
        sender_key: bytes,
    ) -> MessageEnvelope:
        """Create a fully signed message envelope."""
        timestamp = time.time()
        nonce = self.generate_nonce()
        signature = self.sign_message(payload, sender_key, timestamp, nonce)
        return MessageEnvelope(
            payload=payload,
            sender_id=sender_id,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
        )

    def verify_message(
        self,
        envelope: MessageEnvelope,
        sender_key: bytes,
        max_age: Optional[float] = None,
    ) -> AuthResult:
        """Verify a message envelope. Returns AuthResult."""
        age_limit = max_age if max_age is not None else self.MAX_AGE_SECONDS
        message_bytes = f"{envelope.payload}:{envelope.timestamp}:{envelope.nonce}".encode("utf-8")
        sig_valid = self._hmac_verify(sender_key, message_bytes, envelope.signature)
        fresh = (time.time() - envelope.timestamp) <= age_limit
        replay = envelope.nonce in self._seen_nonces
        if sig_valid and not replay:
            self._seen_nonces[envelope.nonce] = time.time()
            self._seen_messages.append(envelope)
            self._cleanup_stale_nonces()
        return AuthResult(
            verified=sig_valid and not replay,
            sender_id=envelope.sender_id,
            freshness=fresh,
            replay_detected=replay,
        )

    def detect_replay(
        self,
        envelope: MessageEnvelope,
        seen_messages: Optional[Set[str]] = None,
    ) -> bool:
        """Check if a message is a replay."""
        store = seen_messages if seen_messages is not None else self._seen_nonces
        return envelope.nonce in store

    def _cleanup_stale_nonces(self) -> None:
        """Remove nonces older than MAX_AGE_SECONDS and enforce max size."""
        now = time.time()
        cutoff = now - self.MAX_AGE_SECONDS

        # Remove expired nonces
        stale = [nonce for nonce, ts in self._seen_nonces.items() if ts < cutoff]
        for nonce in stale:
            del self._seen_nonces[nonce]

        # If still over max, evict oldest entries
        if len(self._seen_nonces) > self.MAX_NONCE_ENTRIES:
            sorted_nonces = sorted(self._seen_nonces.items(), key=lambda x: x[1])
            excess = len(self._seen_nonces) - self.MAX_NONCE_ENTRIES
            for nonce, _ in sorted_nonces[:excess]:
                del self._seen_nonces[nonce]

    def get_seen_count(self) -> int:
        return len(self._seen_nonces)

    def clear_seen(self) -> None:
        count = len(self._seen_nonces)
        self._seen_nonces.clear()
        self._seen_messages.clear()
        logger.info("Cleared %d seen nonces and messages", count)


class KeyManager:
    """Manage cryptographic keys — generate, derive, rotate, revoke."""

    KEY_LENGTH = 32  # 256-bit keys

    def __init__(self) -> None:
        self._keys: Dict[str, bytes] = {}
        self._revoked: Set[str] = set()

    def generate_key(self) -> bytes:
        """Generate a random 256-bit key."""
        return os.urandom(self.KEY_LENGTH)

    def generate_key_with_id(self, key_id: str) -> bytes:
        """Generate a key and store it under the given ID."""
        key = self.generate_key()
        self._keys[key_id] = key
        return key

    def derive_shared_key(self, private_key: bytes, public_key: bytes) -> bytes:
        """Derive a shared key using XOR of keys (simplified for pure-Python)."""
        # In production use HKDF or ECDH; here we use a deterministic XOR+hash approach
        combined = bytes(a ^ b for a, b in zip(private_key, public_key))
        # Extend to KEY_LENGTH using SHA-256
        derived = hashlib.sha256(combined).digest()
        return derived

    def rotate_key(self, old_key: bytes) -> bytes:
        """Generate a new key derived from the old one."""
        old_hash = hashlib.sha256(old_key).digest()
        # Mix in random bytes for forward secrecy
        salt = os.urandom(self.KEY_LENGTH)
        mixed = bytes(a ^ b for a, b in zip(old_hash, salt))
        new_key = hashlib.sha256(mixed).digest()
        return new_key

    def revoke_key(self, key_id: str) -> None:
        """Revoke a key by ID."""
        self._revoked.add(key_id)
        if key_id in self._keys:
            del self._keys[key_id]

    def get_key(self, key_id: str) -> Optional[bytes]:
        """Get a key by ID if not revoked."""
        if key_id in self._revoked:
            return None
        return self._keys.get(key_id)

    def is_revoked(self, key_id: str) -> bool:
        return key_id in self._revoked

    def list_keys(self) -> List[str]:
        return list(self._keys.keys())
