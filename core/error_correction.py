"""
Error correction codes for GAN-based steganography.

GAN models recover bits at ~85-90% accuracy → ~10% bit error rate.
At 10% BIT errors, byte error rate ≈ 1 - 0.9^8 ≈ 57% (too high for byte-level RS).

Two complementary tools:
  1. BitRepetitionECC — encode each bit 3 or 5 times, majority-vote on decode.
     Best for very high bit error rates (the GAN case).
  2. ReedSolomonECC — byte-level RS coding (works at <5% bit error rate).
     Best after the GAN is reliable enough that bit errors are rare.
"""

from typing import Optional
import numpy as np
from reedsolo import RSCodec, ReedSolomonError


class BitRepetitionECC:
    """
    Encode each bit `factor` times. At decode, majority-vote each group.

    With factor=3 and 10% per-bit error:
        P(majority wrong) = C(3,2)*0.1^2*0.9 + 0.1^3 = 0.028 → 2.8%
        Effective bit accuracy: 97.2%

    With factor=5 and 10% per-bit error:
        Effective bit accuracy: ~99.1%
    """

    def __init__(self, factor: int = 3):
        if factor < 1 or factor % 2 == 0:
            raise ValueError("factor must be odd and >= 1")
        self.factor = factor

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Repeat each bit `factor` times. (N,) → (N*factor,)"""
        return np.repeat(bits, self.factor)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Majority-vote each group of `factor` bits. (N*factor,) → (N,)"""
        n = len(bits) // self.factor
        groups = bits[:n * self.factor].reshape(n, self.factor)
        # Majority vote: sum > factor/2 → 1, else 0
        return (groups.sum(axis=1) > self.factor // 2).astype(np.uint8)


class ReedSolomonECC:
    """
    Reed-Solomon byte-level error correction.
    Can correct up to redundancy_bytes/2 byte errors per chunk.
    """

    def __init__(self, redundancy_bytes: int = 40, chunk_size: int = 255):
        self.redundancy = redundancy_bytes
        self.chunk_size = chunk_size
        self.codec = RSCodec(redundancy_bytes, nsize=chunk_size)

    def encode(self, data: bytes) -> bytes:
        return bytes(self.codec.encode(data))

    def decode(self, data: bytes) -> bytes:
        decoded, _, _ = self.codec.decode(data)
        return bytes(decoded)

    def decode_safe(self, data: bytes) -> Optional[bytes]:
        try:
            return self.decode(data)
        except ReedSolomonError:
            return None


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to a uint8 array of bits (MSB-first)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert a uint8 array of bits back to bytes (MSB-first, drops trailing partial byte)."""
    n = (len(bits) // 8) * 8
    if n == 0:
        return b""
    return np.packbits(bits[:n]).tobytes()
