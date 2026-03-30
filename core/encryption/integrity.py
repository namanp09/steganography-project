"""
Data integrity verification using SHA-256.
Used to verify that extracted data matches the original.
"""

import hashlib


def compute_hash(data: bytes) -> str:
    """Compute SHA-256 hex digest of data."""
    return hashlib.sha256(data).hexdigest()


def verify_hash(data: bytes, expected_hash: str) -> bool:
    """Verify data integrity against expected SHA-256 hash."""
    return hashlib.sha256(data).hexdigest() == expected_hash
