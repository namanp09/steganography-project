"""
Image DCT Steganography — Transform Domain Method.

Embeds data in quantized DCT coefficients of 8x8 blocks.
Uses mid-frequency band coefficient modification (robust against compression).
Uses fixed-strength QIM for reliable extraction.
"""

import numpy as np
import cv2
from typing import Optional


class ImageDCT:
    """DCT-based image steganography with QIM embedding."""

    def __init__(
        self,
        block_size: int = 8,
        alpha: float = 10.0,
        seed: Optional[int] = None,
    ):
        self.block_size = block_size
        self.alpha = alpha
        self.seed = seed
        # Mid-frequency zigzag positions (robust embedding locations)
        self.embed_positions = [
            (0, 3), (1, 2), (2, 1), (3, 0),
            (0, 4), (1, 3), (2, 2), (3, 1), (4, 0),
        ]

    def _get_block_coords(self, h: int, w: int):
        """Get list of (y, x) block coordinates."""
        coords = []
        for i in range(h // self.block_size):
            for j in range(w // self.block_size):
                coords.append((i * self.block_size, j * self.block_size))
        return coords

    def capacity(self, image: np.ndarray) -> int:
        h, w = image.shape[:2]
        n_blocks = (h // self.block_size) * (w // self.block_size)
        bits_per_block = len(self.embed_positions)
        total_bits = n_blocks * bits_per_block
        return (total_bits // 8) - 4

    def _is_safe_block(self, block_raw: np.ndarray) -> bool:
        """A block is safe for embedding if it has enough headroom to avoid uint8 clipping."""
        margin = max(self.alpha, 20.0)
        m = float(block_raw.mean())
        return margin < m < (255.0 - margin)

    def encode(self, cover_image: np.ndarray, secret_data: bytes) -> np.ndarray:
        """Embed data in DCT mid-frequency coefficients using QIM."""
        image = cover_image.copy()

        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channel = ycrcb[:, :, 0].astype(np.float64)
        else:
            channel = image.astype(np.float64)
            ycrcb = None

        h, w = channel.shape
        length_header = len(secret_data).to_bytes(4, "big")
        payload = length_header + secret_data
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

        coords = self._get_block_coords(h, w)
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(coords)

        bit_idx = 0
        delta = self.alpha

        for y, x in coords:
            if bit_idx >= len(bits):
                break
            block_raw = channel[y:y+8, x:x+8]
            if not self._is_safe_block(block_raw):
                continue
            block = block_raw - 128.0
            dct_block = cv2.dct(block)

            for pos in self.embed_positions:
                if bit_idx >= len(bits):
                    break
                coeff = dct_block[pos[0], pos[1]]
                bit = int(bits[bit_idx])

                q = int(np.floor(coeff / delta))
                if (q % 2) != bit:
                    if ((q + 1) % 2) == bit:
                        dct_block[pos[0], pos[1]] = (q + 1) * delta + delta / 2
                    else:
                        dct_block[pos[0], pos[1]] = q * delta + delta / 2
                else:
                    dct_block[pos[0], pos[1]] = q * delta + delta / 2

                bit_idx += 1

            channel[y:y+8, x:x+8] = cv2.idct(dct_block) + 128.0

        channel = np.clip(channel, 0, 255)

        if ycrcb is not None:
            ycrcb[:, :, 0] = channel.astype(np.uint8)
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return channel.astype(np.uint8)

    def decode(self, stego_image: np.ndarray) -> bytes:
        """Extract data from DCT coefficients using QIM."""
        if len(stego_image.shape) == 3:
            ycrcb = cv2.cvtColor(stego_image, cv2.COLOR_BGR2YCrCb)
            channel = ycrcb[:, :, 0].astype(np.float64)
        else:
            channel = stego_image.astype(np.float64)

        h, w = channel.shape
        coords = self._get_block_coords(h, w)
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(coords)

        delta = self.alpha
        all_bits = []

        for y, x in coords:
            block_raw = channel[y:y+8, x:x+8]
            if not self._is_safe_block(block_raw):
                continue
            block = block_raw - 128.0
            dct_block = cv2.dct(block)

            for pos in self.embed_positions:
                coeff = dct_block[pos[0], pos[1]]
                q = int(np.floor(coeff / delta))
                all_bits.append(q % 2)

        all_bits = np.array(all_bits, dtype=np.uint8)

        header_bytes = np.packbits(all_bits[:32]).tobytes()
        length = int.from_bytes(header_bytes[:4], "big")

        msg_bits = all_bits[32 : 32 + length * 8]
        pad_len = (8 - len(msg_bits) % 8) % 8
        if pad_len:
            msg_bits = np.concatenate([msg_bits, np.zeros(pad_len, dtype=np.uint8)])

        return np.packbits(msg_bits).tobytes()[:length]
