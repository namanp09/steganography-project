"""
Image DWT Steganography — Transform Domain Method.

Embeds data in Discrete Wavelet Transform detail coefficients.
Uses multi-level decomposition with fixed-strength QIM for reliable extraction.
"""

import numpy as np
import cv2
import pywt
from typing import Optional


class ImageDWT:
    """DWT-based image steganography with multi-level decomposition."""

    def __init__(
        self,
        wavelet: str = "haar",
        level: int = 2,
        alpha: float = 0.5,
        subband: str = "LH",
        seed: Optional[int] = None,
    ):
        self.wavelet = wavelet
        self.level = level
        self.alpha = alpha
        self.subband = subband
        self.seed = seed
        self._subband_idx = {"LH": 0, "HL": 1, "HH": 2}[subband]

    def capacity(self, image: np.ndarray) -> int:
        h, w = image.shape[:2]
        coeff_h = h // (2 ** self.level)
        coeff_w = w // (2 ** self.level)
        total_bits = coeff_h * coeff_w
        return (total_bits // 8) - 4

    def encode(self, cover_image: np.ndarray, secret_data: bytes) -> np.ndarray:
        """Embed data in DWT detail coefficients using QIM."""
        image = cover_image.copy()

        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channel = ycrcb[:, :, 0].astype(np.float64)
        else:
            channel = image.astype(np.float64)
            ycrcb = None

        # Multi-level DWT
        coeffs = pywt.wavedec2(channel, self.wavelet, level=self.level)

        # Target subband at deepest detail level
        detail_coeffs = list(coeffs[1])
        target = detail_coeffs[self._subband_idx].copy()
        flat = target.flatten()

        # Prepare payload
        length_header = len(secret_data).to_bytes(4, "big")
        payload = length_header + secret_data
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

        if len(bits) > len(flat):
            raise ValueError(f"Data too large: {len(bits)} bits, {len(flat)} coefficients")

        indices = np.arange(len(flat))
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        delta = self.alpha

        # QIM embedding: quantize to odd/even multiples of delta
        for i, bit in enumerate(bits):
            idx = indices[i]
            coeff = flat[idx]
            q = int(np.floor(coeff / delta))
            bit = int(bit)

            if (q % 2) != bit:
                # Snap to nearest correct-parity level
                if ((q + 1) % 2) == bit:
                    flat[idx] = (q + 1) * delta + delta / 2
                else:
                    flat[idx] = q * delta + delta / 2
            else:
                flat[idx] = q * delta + delta / 2

        detail_coeffs[self._subband_idx] = flat.reshape(target.shape)
        coeffs[1] = tuple(detail_coeffs)

        # Inverse DWT
        reconstructed = pywt.waverec2(coeffs, self.wavelet)
        reconstructed = reconstructed[:channel.shape[0], :channel.shape[1]]

        if ycrcb is not None:
            ycrcb[:, :, 0] = np.clip(reconstructed, 0, 255).astype(np.uint8)
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return np.clip(reconstructed, 0, 255).astype(np.uint8)

    def decode(self, stego_image: np.ndarray) -> bytes:
        """Extract data from DWT detail coefficients using QIM."""
        if len(stego_image.shape) == 3:
            ycrcb = cv2.cvtColor(stego_image, cv2.COLOR_BGR2YCrCb)
            channel = ycrcb[:, :, 0].astype(np.float64)
        else:
            channel = stego_image.astype(np.float64)

        coeffs = pywt.wavedec2(channel, self.wavelet, level=self.level)
        detail_coeffs = coeffs[1]
        target = detail_coeffs[self._subband_idx]
        flat = target.flatten()

        indices = np.arange(len(flat))
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        delta = self.alpha
        all_bits = []

        for i in range(len(flat)):
            idx = indices[i]
            coeff = flat[idx]
            q = int(np.floor(coeff / delta))
            all_bits.append(q % 2)

        all_bits = np.array(all_bits, dtype=np.uint8)

        # Read length header
        header_bytes = np.packbits(all_bits[:32]).tobytes()
        length = int.from_bytes(header_bytes[:4], "big")

        msg_bits = all_bits[32 : 32 + length * 8]
        pad_len = (8 - len(msg_bits) % 8) % 8
        if pad_len:
            msg_bits = np.concatenate([msg_bits, np.zeros(pad_len, dtype=np.uint8)])

        return np.packbits(msg_bits).tobytes()[:length]
