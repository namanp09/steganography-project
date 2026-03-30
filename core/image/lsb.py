"""
Image LSB Steganography — Baseline Method.

Embeds secret data in the least significant bits of pixel values.
Supports multi-bit embedding (1-4 bits per channel) and
randomized embedding positions using a seed key for added security.
"""

import numpy as np
import cv2
from typing import Optional


class ImageLSB:
    """LSB steganography with optional randomized embedding positions."""

    def __init__(self, num_bits: int = 1, seed: Optional[int] = None):
        """
        Args:
            num_bits: Number of LSBs to use per channel (1-4).
            seed: Random seed for shuffling pixel positions.
        """
        assert 1 <= num_bits <= 4, "num_bits must be between 1 and 4"
        self.num_bits = num_bits
        self.seed = seed

    def _get_pixel_order(self, total_pixels: int) -> np.ndarray:
        """Get pixel indices — shuffled if seed is provided."""
        indices = np.arange(total_pixels)
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        return indices

    def _data_to_bits(self, data: bytes) -> np.ndarray:
        """Convert bytes to a flat array of bits."""
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return bits

    def _bits_to_data(self, bits: np.ndarray) -> bytes:
        """Convert flat bit array back to bytes."""
        # Pad to multiple of 8
        pad_len = (8 - len(bits) % 8) % 8
        if pad_len:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
        return np.packbits(bits).tobytes()

    def capacity(self, image: np.ndarray) -> int:
        """Return embedding capacity in bytes."""
        h, w, c = image.shape
        total_bits = h * w * c * self.num_bits
        return (total_bits // 8) - 4  # 4 bytes reserved for length header

    def encode(self, cover_image: np.ndarray, secret_data: bytes) -> np.ndarray:
        """
        Embed secret data into cover image.

        Args:
            cover_image: BGR image (H, W, 3), uint8.
            secret_data: Bytes to embed.

        Returns:
            Stego image (H, W, 3), uint8.
        """
        image = cover_image.copy()
        h, w, c = image.shape

        # Prepend 32-bit length header
        length_header = len(secret_data).to_bytes(4, "big")
        payload = length_header + secret_data
        bits = self._data_to_bits(payload)

        max_bits = h * w * c * self.num_bits
        if len(bits) > max_bits:
            raise ValueError(
                f"Data too large: {len(bits)} bits needed, {max_bits} available"
            )

        flat = image.reshape(-1)
        order = self._get_pixel_order(len(flat))

        mask = (0xFF << self.num_bits) & 0xFF  # Clear lower bits, keep in uint8 range

        for i, bit_idx in enumerate(range(0, len(bits), self.num_bits)):
            chunk = bits[bit_idx : bit_idx + self.num_bits]
            if len(chunk) < self.num_bits:
                chunk = np.pad(chunk, (0, self.num_bits - len(chunk)))
            value = 0
            for b in chunk:
                value = (value << 1) | int(b)
            pixel_idx = order[i]
            flat[pixel_idx] = np.uint8((int(flat[pixel_idx]) & mask) | value)

        return flat.reshape(h, w, c)

    def decode(self, stego_image: np.ndarray) -> bytes:
        """
        Extract secret data from stego image.

        Args:
            stego_image: BGR stego image (H, W, 3), uint8.

        Returns:
            Extracted bytes.
        """
        h, w, c = stego_image.shape
        flat = stego_image.reshape(-1)
        order = self._get_pixel_order(len(flat))

        lsb_mask = np.uint8((1 << self.num_bits) - 1)

        # Extract all available bits
        all_bits = []
        for i in range(len(flat)):
            pixel_idx = order[i]
            val = int(flat[pixel_idx] & lsb_mask)
            for shift in range(self.num_bits - 1, -1, -1):
                all_bits.append((val >> shift) & 1)

        all_bits = np.array(all_bits, dtype=np.uint8)

        # Read 32-bit length header (first 32 bits)
        header_bits = all_bits[:32]
        length = int.from_bytes(self._bits_to_data(header_bits)[:4], "big")

        # Extract message bits
        msg_bits = all_bits[32 : 32 + length * 8]
        return self._bits_to_data(msg_bits)[:length]
