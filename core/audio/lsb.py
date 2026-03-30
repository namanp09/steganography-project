"""
Audio LSB Steganography — Baseline Method.

Embeds data in the least significant bits of 16-bit PCM audio samples.
Supports multi-bit embedding and randomized sample selection.
"""

import numpy as np
import soundfile as sf
from typing import Optional, Tuple


class AudioLSB:
    """LSB steganography on 16-bit PCM audio."""

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
        """Return capacity in bytes."""
        total_bits = len(audio.flatten()) * self.num_bits
        return (total_bits // 8) - 4

    def encode(
        self, audio: np.ndarray, sr: int, secret_data: bytes
    ) -> Tuple[np.ndarray, int]:
        """
        Embed data into audio samples.

        Args:
            audio: Audio samples as float64 array.
            sr: Sample rate.
            secret_data: Bytes to embed.

        Returns:
            (stego_audio, sample_rate)
        """
        # Convert to 16-bit integer representation
        if audio.dtype == np.float64 or audio.dtype == np.float32:
            samples = (audio * 32767).astype(np.int16)
        else:
            samples = audio.astype(np.int16)

        flat = samples.flatten()

        # Prepare payload
        length_header = len(secret_data).to_bytes(4, "big")
        payload = length_header + secret_data
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

        max_bits = len(flat) * self.num_bits
        if len(bits) > max_bits:
            raise ValueError(f"Data too large: {len(bits)} bits needed, {max_bits} available")

        order = self._get_sample_order(len(flat))
        mask = np.int16(0xFFFF << self.num_bits)

        for i in range(0, len(bits), self.num_bits):
            chunk = bits[i : i + self.num_bits]
            if len(chunk) < self.num_bits:
                chunk = np.pad(chunk, (0, self.num_bits - len(chunk)))
            value = 0
            for b in chunk:
                value = (value << 1) | int(b)

            sample_idx = order[i // self.num_bits]
            flat[sample_idx] = (flat[sample_idx] & mask) | np.int16(value)

        stego = flat.reshape(samples.shape).astype(np.float64) / 32767.0
        return stego, sr

    def decode(self, audio: np.ndarray) -> bytes:
        """Extract data from stego audio."""
        if audio.dtype == np.float64 or audio.dtype == np.float32:
            samples = (audio * 32767).astype(np.int16)
        else:
            samples = audio.astype(np.int16)

        flat = samples.flatten()
        order = self._get_sample_order(len(flat))
        lsb_mask = np.int16((1 << self.num_bits) - 1)

        all_bits = []
        for i in range(len(flat)):
            idx = order[i]
            val = int(flat[idx] & lsb_mask)
            for shift in range(self.num_bits - 1, -1, -1):
                all_bits.append((val >> shift) & 1)

        all_bits = np.array(all_bits, dtype=np.uint8)

        # Read length
        header = np.packbits(all_bits[:32]).tobytes()
        length = int.from_bytes(header[:4], "big")

        msg_bits = all_bits[32 : 32 + length * 8]
        pad_len = (8 - len(msg_bits) % 8) % 8
        if pad_len:
            msg_bits = np.concatenate([msg_bits, np.zeros(pad_len, dtype=np.uint8)])

        return np.packbits(msg_bits).tobytes()[:length]
