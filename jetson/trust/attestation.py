"""NEXUS Trust Engine - Cryptographic trust attestation for deployment signing.

Provides HMAC-signed trust attestations that bind vessel identity,
trust scores, and bytecode hashes together. Enables secure trust
verification across distributed nodes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import base64
import os
import time
from dataclasses import dataclass, field


# Default signing key - in production this would come from a KMS
# Load from environment variable NEXUS_ATTESTATION_KEY with fallback
_DEFAULT_SIGNING_KEY = os.environ.get(
    "NEXUS_ATTESTATION_KEY",
    b"nexus-trust-attestation-key-v1",
)
if isinstance(_DEFAULT_SIGNING_KEY, str):
    _DEFAULT_SIGNING_KEY = _DEFAULT_SIGNING_KEY.encode("utf-8")


@dataclass
class AttestationPayload:
    """Structured content of a trust attestation."""

    vessel_id: str
    trust_scores: dict[str, float]
    bytecode_hash: str
    timestamp: float
    trust_level: int = 0
    subsystems: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "vessel_id": self.vessel_id,
            "trust_scores": self.trust_scores,
            "bytecode_hash": self.bytecode_hash,
            "timestamp": self.timestamp,
            "trust_level": self.trust_level,
            "subsystems": self.subsystems,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AttestationPayload:
        """Deserialize from dict."""
        return cls(
            vessel_id=data["vessel_id"],
            trust_scores=data["trust_scores"],
            bytecode_hash=data["bytecode_hash"],
            timestamp=data["timestamp"],
            trust_level=data.get("trust_level", 0),
            subsystems=data.get("subsystems", []),
            metadata=data.get("metadata", {}),
        )


class TrustAttestation:
    """Cryptographic trust attestation for deployment signing.

    Creates, verifies, and decodes HMAC-signed trust attestations.

    Attestation format:
      <base64-encoded JSON payload>.<base64-encoded HMAC signature>

    The HMAC is computed over the canonical JSON payload using SHA-256.
    """

    def __init__(self, signing_key: bytes | None = None) -> None:
        self.signing_key = signing_key or _DEFAULT_SIGNING_KEY
        self.hash_algorithm = "sha256"

    def create_attestation(
        self,
        vessel_id: str,
        trust_scores: dict[str, float],
        bytecode_hash: str,
        trust_level: int = 0,
        subsystems: list[str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Create HMAC-signed trust attestation.

        Args:
            vessel_id: unique vessel identifier
            trust_scores: dict of subsystem -> trust_score
            bytecode_hash: SHA-256 hash of the deployed bytecode
            trust_level: current autonomy level (0-5)
            subsystems: list of subsystem names (optional)
            metadata: additional metadata (optional)

        Returns:
            Base64-encoded attestation string in format: payload.signature
        """
        payload = AttestationPayload(
            vessel_id=vessel_id,
            trust_scores=trust_scores,
            bytecode_hash=bytecode_hash,
            timestamp=time.time(),
            trust_level=trust_level,
            subsystems=subsystems or list(trust_scores.keys()),
            metadata=metadata or {},
        )

        return self._sign_payload(payload)

    def create_attestation_from_payload(
        self, payload: AttestationPayload
    ) -> str:
        """Sign a pre-built AttestationPayload."""
        return self._sign_payload(payload)

    def verify_attestation(
        self,
        attestation: str,
        vessel_id: str | None = None,
        expected_trust_range: tuple[float, float] | None = None,
        expected_bytecode_hash: str | None = None,
        max_age_seconds: float | None = None,
    ) -> bool:
        """Verify trust attestation signature and optional constraints.

        Args:
            attestation: attestation string to verify
            vessel_id: if provided, must match payload vessel_id
            expected_trust_range: if provided, average trust must be in [min, max]
            expected_bytecode_hash: if provided, must match payload hash
            max_age_seconds: if provided, attestation must not be older than this

        Returns:
            True if attestation is valid and all constraints pass
        """
        try:
            parts = attestation.split(".")
            if len(parts) != 2:
                return False

            payload_b64, signature_b64 = parts

            # Verify HMAC signature
            expected_sig = hmac.new(
                self.signing_key,
                payload_b64.encode("utf-8"),
                hashlib.sha256,
            ).digest()

            try:
                actual_sig = base64.urlsafe_b64decode(signature_b64)
            except Exception:
                return False

            if not hmac.compare_digest(expected_sig, actual_sig):
                return False

            # Decode payload for constraint checks
            try:
                payload_data = json.loads(base64.urlsafe_b64decode(payload_b64))
            except Exception:
                return False

            # Check vessel_id
            if vessel_id is not None and payload_data.get("vessel_id") != vessel_id:
                return False

            # Check bytecode_hash
            if (
                expected_bytecode_hash is not None
                and payload_data.get("bytecode_hash") != expected_bytecode_hash
            ):
                return False

            # Check trust range
            if expected_trust_range is not None:
                scores = payload_data.get("trust_scores", {})
                if scores:
                    avg_trust = sum(scores.values()) / len(scores)
                    min_t, max_t = expected_trust_range
                    if not (min_t <= avg_trust <= max_t):
                        return False

            # Check age
            if max_age_seconds is not None:
                attestation_time = payload_data.get("timestamp", 0)
                if time.time() - attestation_time > max_age_seconds:
                    return False

            return True

        except Exception:
            return False

    def attestation_to_dict(self, attestation: str) -> dict | None:
        """Decode attestation into structured dict.

        Args:
            attestation: attestation string

        Returns:
            Decoded payload as dict, or None if attestation is malformed.
        """
        try:
            parts = attestation.split(".")
            if len(parts) != 2:
                return None

            payload_b64 = parts[0]
            payload_data = json.loads(base64.urlsafe_b64decode(payload_b64))
            return payload_data

        except Exception:
            return None

    def attestation_to_payload(self, attestation: str) -> AttestationPayload | None:
        """Decode attestation into AttestationPayload object.

        Args:
            attestation: attestation string

        Returns:
            AttestationPayload object, or None if attestation is malformed.
        """
        data = self.attestation_to_dict(attestation)
        if data is None:
            return None
        return AttestationPayload.from_dict(data)

    def _sign_payload(self, payload: AttestationPayload) -> str:
        """Sign a payload and return the attestation string."""
        payload_json = json.dumps(
            payload.to_dict(), sort_keys=True, separators=(",", ":")
        )
        payload_b64 = base64.urlsafe_b64encode(
            payload_json.encode("utf-8")
        ).decode("utf-8")

        signature = hmac.new(
            self.signing_key,
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode("utf-8")

        return f"{payload_b64}.{signature_b64}"

    @staticmethod
    def compute_bytecode_hash(bytecode: bytes) -> str:
        """Compute SHA-256 hash of bytecode.

        Args:
            bytecode: raw bytecode bytes

        Returns:
            Hex-encoded SHA-256 hash string
        """
        return hashlib.sha256(bytecode).hexdigest()

    @staticmethod
    def compute_trust_hash(trust_scores: dict[str, float]) -> str:
        """Compute a deterministic hash of trust scores for comparison.

        Args:
            trust_scores: dict of subsystem -> trust_score

        Returns:
            Hex-encoded SHA-256 hash string
        """
        canonical = json.dumps(trust_scores, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
