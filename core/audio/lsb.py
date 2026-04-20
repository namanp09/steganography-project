"""
Audio LSB Steganography — Baseline Method.

Embeds data in the least significant bits of 16-bit PCM audio samples.
Supports multi-bit embedding and randomized sample selection.
"""

import numpy as np
from typing import Optional, Tuple


class AudioLSB:
    """LSB steganography on 16-bit PCM audio."""

    SCALE = 32768  # 2^15 — use power of 2 for exact float<->int16 round-trip

    def __init__(self, num_bits: int = 1, seed: Optional[int] = None):
        assert 1 <= num_bits <= 4
        self.num_bits = num_bits
        self.seed = seed

    def _get_sample_order(self, n: int) -> np.ndarray:
        indices = np.arange(n)
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        return indices

    def capacity(self, audio: np.ndarray) -> int:
        total_bits = len(audio.flatten()) * self.num_bits
        return (total_bits // 8) - 4

    def _to_uint16(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype.kind == "f":
            samples = np.clip(audio * self.SCALE, -self.SCALE, self.SCALE - 1).astype(np.int16)
        else:
            samples = audio.astype(np.int16)
        return samples.view(np.uint16).flatten()

    def _from_uint16(self, flat: np.ndarray, shape, dtype) -> np.ndarray:
        samples = flat.view(np.int16).reshape(shape)
        if np.issubdtype(dtype, np.floating):
            return samples.astype(np.float64) / self.SCALE
        return samples.astype(dtype)

    def encode(
        self, audio: np.ndarray, sr: int, secret_data: bytes
    ) -> Tuple[np.ndarray, int]:
        flat = self._to_uint16(audio)

        length_header = len(secret_data).to_bytes(4, "big")
        payload = length_header + secret_data
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

        max_bits = len(flat) * self.num_bits
        if len(bits) > max_bits:
            raise ValueError(f"Data too large: {len(bits)} bits, {max_bits} available")

        order = self._get_sample_order(len(flat))
        clear_mask = np.uint16(~((1 << self.num_bits) - 1) & 0xFFFF)

        for i in range(0, len(bits), self.num_bits):
            chunk = bits[i : i + self.num_bits]
            if len(chunk) < self.num_bits:
                chunk = np.pad(chunk, (0, self.num_bits - len(chunk)))
            value = 0
            for b in chunk:
                value = (value << 1) | int(b)
            sample_idx = order[i // self.num_bits]
            flat[sample_idx] = (flat[sample_idx] & clear_mask) | np.uint16(value)

        return self._from_uint16(flat, audio.shape, audio.dtype), sr

    def decode(self, audio: np.ndarray) -> bytes:
        flat = self._to_uint16(audio)
        order = self._get_sample_order(len(flat))
        lsb_mask = np.uint16((1 << self.num_bits) - 1)

        all_bits = []
        for i in range(len(flat)):
            val = int(flat[order[i]] & lsb_mask)
            for shift in range(self.num_bits - 1, -1, -1):
                all_bits.append((val >> shift) & 1)

        all_bits = np.array(all_bits, dtype=np.uint8)
        header = np.packbits(all_bits[:32]).tobytes()
        length = int.from_bytes(header[:4], "big")

        msg_bits = all_bits[32 : 32 + length * 8]
        pad_len = (8 - len(msg_bits) % 8) % 8
        if pad_len:
            msg_bits = np.concatenate([msg_bits, np.zeros(pad_len, dtype=np.uint8)])

        return np.packbits(msg_bits).tobytes()[:length]
